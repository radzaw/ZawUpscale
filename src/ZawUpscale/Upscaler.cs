namespace ZawUpscale;

/// <summary>A super-resolution network: RGB in [0, 1], planar NCHW with N = 1, output <see cref="Scale"/> times larger.</summary>
public interface ISrModel
{
    int Scale { get; }
    /// <summary>Input size the network requires, or null if it takes any size.</summary>
    (int Width, int Height)? FixedSize { get; }
    /// <summary>Runs the network on one tile: <paramref name="chw"/> holds 3 × h × w floats; the result 3 × h·Scale × w·Scale.</summary>
    float[] Run(float[] chw, int h, int w);
    /// <summary>True if every tile of a pass should have the same input size (edge tiles padded), for backends that compile per shape.</summary>
    bool UniformTiles => false;
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

    /// <summary>
    /// One network pass over the whole image, tile by tile. The image is cut into as many tiles as <paramref name="tile"/>
    /// takes, all (nearly) the same size, so no edge strip runs mostly padding. While the network runs a tile, the result
    /// of the one before is copied into the image on another thread.
    /// </summary>
    internal RgbaImage RunPass(RgbaImage src, int tile, Action? tileDone, CancellationToken ct)
    {
        var s = model.Scale;
        int W = src.Width, H = src.Height;
        var dst = new RgbaImage(W * s, H * s);
        var sp = src.Pixels;
        int nx = (W + tile - 1) / tile, ny = (H + tile - 1) / tile;
        int tw = (W + nx - 1) / nx, th = (H + ny - 1) / ny; // the largest tile
        // The input size of every tile, when they share one: the fixed size, or with uniform tiles that of an inner tile
        // with context on both sides.
        var same = model.FixedSize is { } f ? f
            : model.UniformTiles ? (RoundUp(Math.Min(W, tw + 2 * Pad), Multiple), RoundUp(Math.Min(H, th + 2 * Pad), Multiple))
            : ((int W, int H)?)null;

        Task? copying = null;
        try
        {
            for (var j = 0; j < ny; j++)
                for (var i = 0; i < nx; i++)
                {
                    ct.ThrowIfCancellationRequested();
                    int x0 = (int)((long)i * W / nx), x1 = (int)((long)(i + 1) * W / nx);
                    int y0 = (int)((long)j * H / ny), y1 = (int)((long)(j + 1) * H / ny);
                    // The region read from the image (rx0, ry0, rw × rh) and the network input size (iw × ih); any input
                    // beyond the region is filled by repeating its edge.
                    int rx0, ry0, rw, rh, iw, ih;
                    if (same is { } u)
                    {
                        (iw, ih) = u;
                        // Context around the tile, but at the image's edges the region moves inward, to be filled with
                        // more of the image rather than repeated edge pixels.
                        rx0 = Math.Clamp(x0 - Pad, 0, Math.Max(0, W - iw));
                        ry0 = Math.Clamp(y0 - Pad, 0, Math.Max(0, H - ih));
                        (rw, rh) = (Math.Min(iw, W - rx0), Math.Min(ih, H - ry0));
                    }
                    else
                    {
                        // Context around the tile, clamped to the image; the input is grown to the next multiple.
                        (rx0, ry0) = (Math.Max(x0 - Pad, 0), Math.Max(y0 - Pad, 0));
                        (rw, rh) = (Math.Min(x1 + Pad, W) - rx0, Math.Min(y1 + Pad, H) - ry0);
                        (iw, ih) = (RoundUp(rw, Multiple), RoundUp(rh, Multiple));
                    }

                    var input = new float[3 * iw * ih];
                    var plane = iw * ih;
                    const float k = 1f / 255f;
                    for (var y = 0; y < ih; y++)
                    {
                        var row = (ry0 + Math.Min(y, rh - 1)) * W;
                        for (var x = 0; x < iw; x++)
                        {
                            var si = (row + rx0 + Math.Min(x, rw - 1)) * 4;
                            var di = y * iw + x;
                            input[di] = sp[si] * k;
                            input[plane + di] = sp[si + 1] * k;
                            input[2 * plane + di] = sp[si + 2] * k;
                        }
                    }

                    var output = model.Run(input, ih, iw);
                    if (output.Length != 3 * iw * s * ih * s)
                        throw new ApiException($"The model returned {output.Length} values for a {iw}×{ih} tile; expected {3 * iw * s * ih * s} (scale {s}).");

                    // Keep only the tile itself, without the context.
                    copying?.GetAwaiter().GetResult();
                    int cx0 = (x0 - rx0) * s, cy0 = (y0 - ry0) * s, ow = iw * s, oh = ih * s;
                    int tx = x0 * s, ty = y0 * s, cw = (x1 - x0) * s, ch = (y1 - y0) * s;
                    copying = Task.Run(() => CopyOut(output, ow, oh, cx0, cy0, dst, tx, ty, cw, ch));
                    tileDone?.Invoke();
                }
            copying?.GetAwaiter().GetResult();
            copying = null;
        }
        finally
        {
            // On an error or cancellation, let the last copy finish before the image is dropped.
            try { copying?.Wait(); }
            catch (AggregateException) { /* the first error is the one thrown */ }
        }
        return dst;
    }

    /// <summary>Copies cw × ch pixels at (cx0, cy0) of a planar RGB network output (ow × oh) to (tx, ty) of the image.</summary>
    static void CopyOut(float[] output, int ow, int oh, int cx0, int cy0, RgbaImage dst, int tx, int ty, int cw, int ch)
    {
        var dp = dst.Pixels;
        int oplane = ow * oh, dstStride = dst.Width * 4;
        for (var y = 0; y < ch; y++)
        {
            var oi = (cy0 + y) * ow + cx0;
            var di = (ty + y) * dstStride + tx * 4;
            for (var x = 0; x < cw; x++, oi++, di += 4)
            {
                dp[di] = ToByte(output[oi]);
                dp[di + 1] = ToByte(output[oplane + oi]);
                dp[di + 2] = ToByte(output[2 * oplane + oi]);
                dp[di + 3] = 255;
            }
        }
    }

    static int RoundUp(int v, int m) => (v + m - 1) / m * m;

    static byte ToByte(float v)
    {
        var x = v * 255f + 0.5f;
        return x <= 0 ? (byte)0 : x >= 255 ? (byte)255 : (byte)x;
    }
}
