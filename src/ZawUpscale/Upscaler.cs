namespace ZawUpscale;

/// <summary>A super-resolution network: RGB in [0, 1], planar NCHW with N = 1, output <see cref="Scale"/> times larger.</summary>
public interface ISrModel
{
    int Scale { get; }
    /// <summary>Input size the network requires, or null if it takes any size.</summary>
    (int Width, int Height)? FixedSize { get; }
    /// <summary>Runs the network on one tile: <paramref name="chw"/> holds 3 × h × w floats; the result 3 × h·Scale × w·Scale.</summary>
    float[] Run(float[] chw, int h, int w);
}

/// <summary>Thrown when a tile doesn't fit in (GPU) memory, so the caller can retry with smaller tiles.</summary>
public sealed class TileTooLargeException(string message, Exception inner) : Exception(message, inner);

/// <summary>
/// Upscales images of any size by running the network over overlapping tiles: each tile is fed with
/// <see cref="Pad"/> pixels of context around it, and only the centre of each result is kept, so the
/// seams between tiles don't show. The alpha channel is resized separately.
/// </summary>
public sealed class Upscaler(ISrModel model)
{
    public const int Pad = 16;
    /// <summary>Tile dimensions are rounded up to a multiple of this; some architectures need even or 4-divisible sizes.</summary>
    public const int Multiple = 4;
    public const long MaxOutputPixels = 400_000_000;

    public ISrModel Model => model;

    /// <summary>
    /// The input size of each network pass. A pass multiplies by the model's scale; when the last pass would
    /// overshoot (6× with a 4× model), the image is first reduced to target / scale, rather than upscaled to
    /// 16× and thrown mostly away.
    /// </summary>
    public List<(int Width, int Height)> Plan(int width, int height, double scale)
    {
        var (tw, th) = TargetSize(width, height, scale);
        var s = model.Scale;
        var plan = new List<(int, int)>();
        int w = width, h = height;
        do
        {
            if (plan.Count > 0 && (double)tw / w < s)
                (w, h) = ((int)Math.Ceiling((double)tw / s), (int)Math.Ceiling((double)th / s));
            plan.Add((w, h));
            w *= s;
            h *= s;
        } while (s > 1 && (w < tw || h < th)); // a 1× (restoration) model runs once
        return plan;
    }

    public static (int Width, int Height) TargetSize(int width, int height, double scale) =>
        (Math.Max(1, (int)Math.Round(width * scale)), Math.Max(1, (int)Math.Round(height * scale)));

    /// <summary>Tiles a pass over a w × h image takes.</summary>
    public static int TileCount(int w, int h, int tile) => ((w + tile - 1) / tile) * ((h + tile - 1) / tile);

    /// <summary>Upscales by <paramref name="scale"/>. Progress reports the fraction done (0–1).</summary>
    public RgbaImage Upscale(RgbaImage src, double scale, int tile, IProgress<double>? progress = null, CancellationToken ct = default)
    {
        if (scale <= 0 || scale > 32) throw new ApiException("The scale must be between 1 and 32.");
        var (tw, th) = TargetSize(src.Width, src.Height, scale);
        if ((long)tw * th > MaxOutputPixels)
            throw new ApiException($"The result would be {tw}×{th} pixels ({(double)tw * th / 1e6:0} megapixels); the limit is {MaxOutputPixels / 1_000_000} megapixels. Choose a smaller scale.");

        tile = EffectiveTile(tile);
        var plan = Plan(src.Width, src.Height, scale);
        // Count tiles of the whole job, so progress moves evenly across passes.
        long total = plan.Sum(p => (long)TileCount(p.Width, p.Height, tile));
        long done = 0;
        var cur = src;
        foreach (var (w, h) in plan)
        {
            if (cur.Width != w || cur.Height != h) cur = ImageIO.Resize(cur, w, h);
            cur = RunPass(cur, tile, () => progress?.Report(Math.Min(1.0, (double)++done / total)), ct);
        }
        if (cur.Width != tw || cur.Height != th) cur = ImageIO.Resize(cur, tw, th);
        if (src.HasAlpha()) ImageIO.TransferAlpha(src, cur);
        return cur;
    }

    /// <summary>The tile size actually used: what the user asked for, limited by a fixed-size network.</summary>
    public int EffectiveTile(int tile)
    {
        tile = Math.Max(32, tile);
        if (model.FixedSize is { } f) tile = Math.Max(8, Math.Min(f.Width, f.Height) - 2 * Pad);
        return tile;
    }

    /// <summary>One network pass over the whole image, tile by tile.</summary>
    internal RgbaImage RunPass(RgbaImage src, int tile, Action? tileDone, CancellationToken ct)
    {
        var s = model.Scale;
        int W = src.Width, H = src.Height;
        var dst = new RgbaImage(W * s, H * s);
        var sp = src.Pixels;
        var dp = dst.Pixels;
        var dstStride = dst.Width * 4;

        for (var y0 = 0; y0 < H; y0 += tile)
            for (var x0 = 0; x0 < W; x0 += tile)
            {
                ct.ThrowIfCancellationRequested();
                int x1 = Math.Min(x0 + tile, W), y1 = Math.Min(y0 + tile, H);
                // Context around the tile, clamped to the image.
                int px0 = Math.Max(x0 - Pad, 0), py0 = Math.Max(y0 - Pad, 0);
                int px1 = Math.Min(x1 + Pad, W), py1 = Math.Min(y1 + Pad, H);
                int rw = px1 - px0, rh = py1 - py0;
                // Network input size: the region, grown to the fixed size or the next multiple; the extra is filled by repeating the edge.
                int iw, ih;
                if (model.FixedSize is { } f) (iw, ih) = (f.Width, f.Height);
                else (iw, ih) = (RoundUp(rw, Multiple), RoundUp(rh, Multiple));

                var input = new float[3 * iw * ih];
                var plane = iw * ih;
                const float k = 1f / 255f;
                for (var y = 0; y < ih; y++)
                {
                    var sy = py0 + Math.Min(y, rh - 1);
                    var row = sy * W;
                    for (var x = 0; x < iw; x++)
                    {
                        var si = (row + px0 + Math.Min(x, rw - 1)) * 4;
                        var di = y * iw + x;
                        input[di] = sp[si] * k;
                        input[plane + di] = sp[si + 1] * k;
                        input[2 * plane + di] = sp[si + 2] * k;
                    }
                }

                var output = model.Run(input, ih, iw);
                int ow = iw * s, oh = ih * s, oplane = ow * oh;
                if (output.Length != 3 * oplane)
                    throw new ApiException($"The model returned {output.Length} values for a {iw}×{ih} tile; expected {3 * oplane} (scale {s}).");

                // Keep only the tile itself, without the context.
                int cx0 = (x0 - px0) * s, cy0 = (y0 - py0) * s;
                int cw = (x1 - x0) * s, ch = (y1 - y0) * s;
                for (var y = 0; y < ch; y++)
                {
                    var oi = (cy0 + y) * ow + cx0;
                    var di = (y0 * s + y) * dstStride + x0 * s * 4;
                    for (var x = 0; x < cw; x++, oi++, di += 4)
                    {
                        dp[di] = ToByte(output[oi]);
                        dp[di + 1] = ToByte(output[oplane + oi]);
                        dp[di + 2] = ToByte(output[2 * oplane + oi]);
                        dp[di + 3] = 255;
                    }
                }
                tileDone?.Invoke();
            }
        return dst;
    }

    static int RoundUp(int v, int m) => (v + m - 1) / m * m;

    static byte ToByte(float v)
    {
        var x = v * 255f + 0.5f;
        return x <= 0 ? (byte)0 : x >= 255 ? (byte)255 : (byte)x;
    }
}
