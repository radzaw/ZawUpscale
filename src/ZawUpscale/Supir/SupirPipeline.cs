namespace ZawUpscale;

/// <summary>Text conditioning for SDXL: token embeddings of both encoders [77, 2048] and the pooled bigG embedding [1280].</summary>
public sealed record TextEmbedding(float[] Context, float[] Pooled);

/// <summary>
/// SUPIR's networks. Tensors are planar float32 with a batch of 1 unless stated; images are RGB in [-1, 1]
/// and latents are the VAE's, already multiplied by SDXL's scale factor.
/// </summary>
public interface ISupirNetworks
{
    /// <summary>SUPIR's degradation-robust encoder: [3, h, w] → [4, h/8, w/8].</summary>
    float[] DenoiseEncode(float[] image, int h, int w);
    /// <summary>The SDXL VAE encoder: [3, h, w] → [4, h/8, w/8].</summary>
    float[] Encode(float[] image, int h, int w);
    /// <summary>The VAE decoder: [4, h, w] latent → [3, 8h, 8w].</summary>
    float[] Decode(float[] latent, int h, int w);
    /// <summary>
    /// The SDXL UNet with SUPIR's control network, predicting noise (eps) for a batch: sample, control
    /// [b, 4, h, w], timestep [b], context [b, 77, 2048], y [b, 2816].
    /// </summary>
    float[] Unet(float[] sample, float[] timestep, float[] context, float[] y, float[] control, float controlScale, int batch, int h, int w);
    TextEmbedding EncodeText(string text);
}

/// <summary>
/// Runs SUPIR (Yu et al., "Scaling Up to Excellence", 2024) as its reference implementation does: the input,
/// enlarged to the target size, is encoded by the denoising encoder; that latent conditions the control
/// network while the EDM sampler denoises pure noise in overlapping latent tiles, blended with Gaussian
/// weights at every step; the result is decoded and given the colors of the input (wavelet color fix).
/// </summary>
public sealed class SupirUpscaler(ISupirNetworks net, int minSize = SupirPackage.DefaultMinSize)
{
    /// <summary>Diffusion tile in pixels (128 latents, SDXL's native 1024 px), overlapping by half.</summary>
    public const int DefaultTile = 1024;
    public const int MinTile = 512;
    /// <summary>VAE tiles: encoder in pixels, decoder in latents, each overlapping its neighbours.</summary>
    const int EncodeTile = 512, EncodeOverlap = 64, DecodeTile = 64, DecodeOverlap = 8;
    /// <summary>Image sizes are padded to a multiple of this (the VAE divides by 8, the UNet by 4 more, plus margin).</summary>
    const int Multiple = 64;
    public const long MaxOutputPixels = 100_000_000;

    /// <summary>The seed of the last run (random unless one was chosen), for reproducing a result.</summary>
    public int LastSeed { get; private set; }

    /// <summary>Restores the image enlarged by <paramref name="scale"/>. Progress reports the fraction done (0–1).</summary>
    public RgbaImage Upscale(RgbaImage src, double scale, SupirOptions o, int tile, IProgress<double>? progress = null, CancellationToken ct = default)
    {
        if (scale <= 0 || scale > 32) throw new ApiException("The scale must be between 1 and 32.");
        var (tw, th) = Upscaler.TargetSize(src.Width, src.Height, scale);
        if ((long)tw * th > MaxOutputPixels)
            throw new ApiException($"The result would be {tw}×{th} pixels ({(double)tw * th / 1e6:0} megapixels); SUPIR is limited to {MaxOutputPixels / 1_000_000} megapixels. Choose a smaller scale.");

        // SDXL needs room to work: small results are made at a larger size and reduced at the end.
        var (ww, wh) = (tw, th);
        if (Math.Min(tw, th) < minSize)
        {
            var f = (double)minSize / Math.Min(tw, th);
            (ww, wh) = ((int)Math.Ceiling(tw * f - 1e-6), (int)Math.Ceiling(th * f - 1e-6));
        }
        var lq = ImageIO.Resize(src, ww, wh);
        int pw = RoundUp(ww, Multiple), ph = RoundUp(wh, Multiple);
        var image = ToTensor(lq, pw, ph);

        var restored = Restore(image, ph, pw, o, tile, progress, ct);
        var result = FromTensor(restored, pw, ph, ww, wh);
        if (result.Width != tw || result.Height != th) result = ImageIO.Resize(result, tw, th);
        if (src.HasAlpha()) ImageIO.TransferAlpha(src, result);
        return result;
    }

    /// <summary>SUPIR on an image tensor [3, h, w] in [-1, 1] whose sides are multiples of 8 (64 for the best tiling).</summary>
    public float[] Restore(float[] image, int h, int w, SupirOptions o, int tile, IProgress<double>? progress, CancellationToken ct)
    {
        int lh = h / 8, lw = w / 8;
        var lt = Math.Max(16, RoundUp(Math.Max(tile, 128), 8) / 8);
        int th = Math.Min(lt, lh), tw = Math.Min(lt, lw);
        var tiles = Windows(lh, th, Math.Max(1, lt / 2)).Count * Windows(lw, tw, Math.Max(1, lt / 2)).Count;
        int encTiles = VaeTiles(h, w, EncodeTile, EncodeOverlap), decTiles = VaeTiles(lh, lw, DecodeTile, DecodeOverlap);
        // A diffusion tile step (two UNet passes) costs roughly ten VAE tiles.
        double total = 2 * encTiles + 2 * decTiles + 10.0 * tiles * o.Steps, done = 0;
        void Done(double units)
        {
            done += units;
            progress?.Report(Math.Min(1.0, done / total));
        }

        var cond = Condition(o.PositivePrompt);
        var uncond = Condition(o.NegativePrompt);
        ct.ThrowIfCancellationRequested();

        // The control latent, and the "stage 1" image it decodes to (the input with degradations removed), re-encoded as the restoration target.
        var zLq = Tiled(image, 3, h, w, 4, 1, 8, EncodeTile, EncodeOverlap, net.DenoiseEncode, () => Done(1), ct);
        var stage1 = Tiled(zLq, 4, lh, lw, 3, 8, 1, DecodeTile, DecodeOverlap, net.Decode, () => Done(1), ct);
        var zCenter = Tiled(stage1, 3, h, w, 4, 1, 8, EncodeTile, EncodeOverlap, net.Encode, () => Done(1), ct);

        var seed = LastSeed = o.Seed ?? Random.Shared.Next();
        var z = Sample(zLq, zCenter, lh, lw, cond, uncond, o, lt, new Random(seed), () => Done(10), ct);
        var samples = Tiled(z, 4, lh, lw, 3, 8, 1, DecodeTile, DecodeOverlap, net.Decode, () => Done(1), ct);
        return o.ColorFix switch
        {
            "wavelet" => WaveletColorFix(samples, stage1, h, w),
            "adain" => AdaInColorFix(samples, stage1, h * w),
            _ => samples,
        };
    }

    /// <summary>A prompt's cross-attention context and SDXL's "vector": pooled text plus size and crop embeddings.</summary>
    (float[] Context, float[] Y) Condition(string prompt)
    {
        var t = net.EncodeText(prompt);
        if (t.Context.Length != ContextSize || t.Pooled.Length != PooledSize)
            throw new ApiException($"The text encoders returned {t.Context.Length} and {t.Pooled.Length} values; expected {ContextSize} and {PooledSize}.");
        var y = new float[VectorSize];
        t.Pooled.CopyTo(y, 0);
        // SUPIR conditions on original size 1024×1024, crop (0, 0) and target size 1024×1024, whatever the image.
        float[] sizes = [1024, 1024, 0, 0, 1024, 1024];
        for (var i = 0; i < sizes.Length; i++) TimestepEmbedding(sizes[i], 256).CopyTo(y, PooledSize + 256 * i);
        return (t.Context, y);
    }

    public const int ContextSize = 77 * 2048, PooledSize = 1280, VectorSize = 2816;

    /// <summary>The sinusoidal embedding of SDXL's size conditioning: cosines, then sines.</summary>
    internal static float[] TimestepEmbedding(double t, int dim)
    {
        var half = dim / 2;
        var r = new float[dim];
        for (var i = 0; i < half; i++)
        {
            var a = t * Math.Exp(-Math.Log(10000) * i / half);
            r[i] = (float)Math.Cos(a);
            r[half + i] = (float)Math.Sin(a);
        }
        return r;
    }

    // ------------------------------------------------------------------ sampler

    /// <summary>The noise level SUPIR treats as the start of sampling.</summary>
    public const double SigmaMax = 14.6146;
    const double RestoreSigmaMin = 0.05;

    /// <summary>SD's training noise levels: σ(t) = √((1 − ᾱ) / ᾱ) for t = 0…999, with the scaled-linear β schedule.</summary>
    internal static readonly double[] TrainingSigmas = MakeTrainingSigmas();

    static double[] MakeTrainingSigmas()
    {
        const int n = 1000;
        double a = Math.Sqrt(0.00085), b = Math.Sqrt(0.012), cum = 1;
        var s = new double[n];
        for (var i = 0; i < n; i++)
        {
            var sq = a + (b - a) * i / (n - 1);
            cum *= 1 - sq * sq;
            s[i] = Math.Sqrt((1 - cum) / cum);
        }
        return s;
    }

    /// <summary>The sampling schedule (sgm's LegacyDDPMDiscretization): <paramref name="steps"/> roughly evenly spaced training levels, descending, then 0.</summary>
    internal static double[] Sigmas(int steps)
    {
        var n = TrainingSigmas.Length;
        var r = new double[steps + 1];
        for (var k = 0; k < steps; k++)
        {
            // np.linspace(999, 0, steps, endpoint=False).astype(int), largest first.
            var t = steps >= n ? n - 1 - k : (int)(n - 1 - k * (double)(n - 1) / steps);
            r[k] = TrainingSigmas[t];
        }
        return r;
    }

    /// <summary>The training timestep nearest to σ (the denoiser quantizes noise levels to these).</summary>
    internal static int Timestep(double sigma)
    {
        var best = 0;
        for (var i = 1; i < TrainingSigmas.Length; i++)
            if (Math.Abs(TrainingSigmas[i] - sigma) < Math.Abs(TrainingSigmas[best] - sigma)) best = i;
        return best;
    }

    /// <summary>
    /// SUPIR's TiledRestoreEDMSampler: Euler steps with churn noise, linear CFG and control schedules, and
    /// restoration guidance towards <paramref name="center"/>, run tile by tile and blended after each step.
    /// </summary>
    float[] Sample(float[] control, float[] center, int h, int w, (float[] Context, float[] Y) c, (float[] Context, float[] Y) uc,
        SupirOptions o, int tile, Random rng, Action tileDone, CancellationToken ct)
    {
        var sigmas = Sigmas(o.Steps);
        int n = o.Steps, plane = h * w, size = 4 * plane;
        var x = Gaussian(rng, size);
        var x0 = (float)Math.Sqrt(1 + sigmas[0] * sigmas[0]);
        for (var i = 0; i < size; i++) x[i] *= x0;

        int th = Math.Min(tile, h), tw = Math.Min(tile, w), stride = Math.Max(1, tile / 2);
        var ys = Windows(h, th, stride);
        var xs = Windows(w, tw, stride);
        var weights = GaussianWeights(th, tw);
        // Unconditional and conditional passes go through the UNet as one batch of two.
        var context = Concat(uc.Context, c.Context);
        var y = Concat(uc.Y, c.Y);
        var gamma = Math.Min(o.Churn / n, Math.Sqrt(2) - 1);
        int tplane = th * tw, tsize = 4 * tplane;
        var sample = new float[2 * tsize];
        var ctrl = new float[2 * tsize];
        var xt = new float[tsize];
        var cen = new float[tsize];

        for (var step = 0; step < n; step++)
        {
            double sigma = sigmas[step], next = sigmas[step + 1], hat = sigma * (gamma + 1);
            var noise = gamma > 0 ? Gaussian(rng, size) : null;
            var noiseAmp = (float)(o.Noise * Math.Sqrt(Math.Max(0, hat * hat - sigma * sigma)));
            var t = Timestep(hat);
            var q = TrainingSigmas[t];
            var cIn = (float)(1 / Math.Sqrt(q * q + 1));
            var controlScale = (float)(sigma / SigmaMax * (o.ControlStart - o.Control) + o.Control);
            var cfg = (float)((o.GuidanceStart - o.Guidance) * hat / SigmaMax + o.Guidance);
            var restore = next > RestoreSigmaMin && o.Restoration > 0 ? (float)Math.Pow(sigma / SigmaMax, o.Restoration) : 0f;
            var k = (float)((next - hat) / hat);
            var acc = new float[size];
            var count = new float[plane];

            foreach (var ty in ys)
                foreach (var tx in xs)
                {
                    ct.ThrowIfCancellationRequested();
                    Crop(x, 4, h, w, ty, tx, th, tw, xt, 0);
                    if (noise != null)
                    {
                        Crop(noise, 4, h, w, ty, tx, th, tw, cen, 0);
                        for (var i = 0; i < tsize; i++) xt[i] += cen[i] * noiseAmp;
                    }
                    for (var i = 0; i < tsize; i++) sample[i] = sample[tsize + i] = xt[i] * cIn;
                    Crop(control, 4, h, w, ty, tx, th, tw, ctrl, 0);
                    Array.Copy(ctrl, 0, ctrl, tsize, tsize);
                    Crop(center, 4, h, w, ty, tx, th, tw, cen, 0);

                    var eps = net.Unet(sample, [t, t], context, y, ctrl, controlScale, 2, th, tw);
                    if (eps.Length != 2 * tsize) throw new ApiException($"The UNet returned {eps.Length} values; expected {2 * tsize}.");
                    for (var ch = 0; ch < 4; ch++)
                        for (var yy = 0; yy < th; yy++)
                        {
                            var src = ch * tplane + yy * tw;
                            var dst = ch * plane + (ty + yy) * w + tx;
                            var wrow = yy * tw;
                            for (var xx = 0; xx < tw; xx++)
                            {
                                var i = src + xx;
                                // Denoised estimates (EDM eps scaling: x − σ·eps), mixed by classifier-free guidance.
                                var du = xt[i] - (float)q * eps[i];
                                var dc = xt[i] - (float)q * eps[tsize + i];
                                var d = du + cfg * (dc - du);
                                if (restore > 0) d -= (d - cen[i]) * restore;
                                var wt = weights[wrow + xx];
                                acc[dst + xx] += (xt[i] + (xt[i] - d) * k) * wt;
                                if (ch == 0) count[(ty + yy) * w + tx + xx] += wt;
                            }
                        }
                    tileDone();
                }
            for (var ch = 0; ch < 4; ch++)
                for (var i = 0; i < plane; i++) acc[ch * plane + i] /= count[i];
            x = acc;
        }
        return x;
    }

    /// <summary>Standard normal samples (Box–Muller).</summary>
    static float[] Gaussian(Random rng, int n)
    {
        var r = new float[n];
        for (var i = 0; i < n; i += 2)
        {
            var u1 = 1.0 - rng.NextDouble();
            var u2 = rng.NextDouble();
            var m = Math.Sqrt(-2 * Math.Log(u1));
            r[i] = (float)(m * Math.Cos(2 * Math.PI * u2));
            if (i + 1 < n) r[i + 1] = (float)(m * Math.Sin(2 * Math.PI * u2));
        }
        return r;
    }

    /// <summary>SUPIR's tile weights: a narrow Gaussian (variance 0.01 of the tile size squared), so tile centres dominate.</summary>
    internal static float[] GaussianWeights(int h, int w)
    {
        static double[] Axis(int n)
        {
            var mid = (n - 1) / 2.0;
            return Enumerable.Range(0, n).Select(i => Math.Exp(-(i - mid) * (i - mid) / ((double)n * n) / 0.02) / Math.Sqrt(2 * Math.PI * 0.01)).ToArray();
        }
        double[] ay = Axis(h), ax = Axis(w);
        var r = new float[h * w];
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++) r[y * w + x] = (float)(ay[y] * ax[x]);
        return r;
    }

    /// <summary>Tile starts along one axis: every <paramref name="stride"/>, the last flush with the end, so all tiles are full size.</summary>
    internal static List<int> Windows(int n, int tile, int stride)
    {
        if (tile >= n) return [0];
        var r = new List<int>();
        for (var s = 0; s <= n - tile; s += stride) r.Add(s);
        if (r[^1] != n - tile) r.Add(n - tile);
        return r;
    }

    // ------------------------------------------------------------------ VAE tiling

    static int VaeTiles(int h, int w, int tile, int overlap)
    {
        var stride = Math.Max(1, tile - overlap);
        return Windows(h, Math.Min(tile, h), stride).Count * Windows(w, Math.Min(tile, w), stride).Count;
    }

    /// <summary>
    /// Runs a network over overlapping tiles of a planar tensor whose output is <paramref name="num"/>/<paramref name="den"/>
    /// times its size, cross-fading the overlaps. All tiles have one size, so the network sees a single shape.
    /// </summary>
    internal static float[] Tiled(float[] src, int cin, int h, int w, int cout, int num, int den, int tile, int overlap,
        Func<float[], int, int, float[]> run, Action? tileDone, CancellationToken ct)
    {
        int th = Math.Min(tile, h), tw = Math.Min(tile, w), stride = Math.Max(1, tile - overlap);
        int oh = h * num / den, ow = w * num / den, oth = th * num / den, otw = tw * num / den;
        var ramp = Math.Max(1, overlap * num / den);
        static float[] Feather(int n, int ramp) => Enumerable.Range(0, n).Select(i => (float)Math.Min(ramp, Math.Min(i + 1, n - i)) / ramp).ToArray();
        float[] fy = Feather(oth, ramp), fx = Feather(otw, ramp);
        var outPlane = oh * ow;
        var acc = new float[cout * outPlane];
        var weight = new float[outPlane];
        var input = new float[cin * th * tw];
        foreach (var ty in Windows(h, th, stride))
            foreach (var tx in Windows(w, tw, stride))
            {
                ct.ThrowIfCancellationRequested();
                Crop(src, cin, h, w, ty, tx, th, tw, input, 0);
                var o = run(input, th, tw);
                if (o.Length != cout * oth * otw) throw new ApiException($"A SUPIR network returned {o.Length} values for a {tw}×{th} tile; expected {cout * oth * otw}.");
                int oy = ty * num / den, ox = tx * num / den, tp = oth * otw;
                for (var y = 0; y < oth; y++)
                    for (var x = 0; x < otw; x++)
                    {
                        var wt = fy[y] * fx[x];
                        var di = (oy + y) * ow + ox + x;
                        weight[di] += wt;
                        for (var c = 0; c < cout; c++) acc[c * outPlane + di] += o[c * tp + y * otw + x] * wt;
                    }
                tileDone?.Invoke();
            }
        for (var c = 0; c < cout; c++)
            for (var i = 0; i < outPlane; i++) acc[c * outPlane + i] /= weight[i];
        return acc;
    }

    static void Crop(float[] src, int channels, int h, int w, int y0, int x0, int th, int tw, float[] dst, int offset)
    {
        for (var c = 0; c < channels; c++)
            for (var y = 0; y < th; y++)
                Array.Copy(src, c * h * w + (y0 + y) * w + x0, dst, offset + (c * th + y) * tw, tw);
    }

    static float[] Concat(float[] a, float[] b)
    {
        var r = new float[a.Length + b.Length];
        a.CopyTo(r, 0);
        b.CopyTo(r, a.Length);
        return r;
    }

    // ------------------------------------------------------------------ color fix

    /// <summary>The result's detail (all but the coarsest of five wavelet levels) on the input's colors and tones.</summary>
    internal static float[] WaveletColorFix(float[] content, float[] style, int h, int w)
    {
        var lowContent = WaveletLow(content, h, w);
        var lowStyle = WaveletLow(style, h, w);
        var r = new float[content.Length];
        for (var i = 0; i < r.Length; i++) r[i] = content[i] - lowContent[i] + lowStyle[i];
        return r;
    }

    /// <summary>Five à trous blurs (a 3×3 binomial kernel, dilated 1, 2, 4, 8, 16, edges repeated): the low-frequency residual.</summary>
    static float[] WaveletLow(float[] img, int h, int w)
    {
        var cur = (float[])img.Clone();
        var tmp = new float[cur.Length];
        var channels = img.Length / (h * w);
        for (var level = 0; level < 5; level++)
        {
            var r = 1 << level;
            for (var c = 0; c < channels; c++)
            {
                var o = c * h * w;
                for (var y = 0; y < h; y++)
                    for (var x = 0; x < w; x++)
                    {
                        var row = o + y * w;
                        tmp[row + x] = 0.25f * cur[row + Math.Max(x - r, 0)] + 0.5f * cur[row + x] + 0.25f * cur[row + Math.Min(x + r, w - 1)];
                    }
                for (var y = 0; y < h; y++)
                {
                    int up = o + Math.Max(y - r, 0) * w, mid = o + y * w, down = o + Math.Min(y + r, h - 1) * w;
                    for (var x = 0; x < w; x++) cur[mid + x] = 0.25f * tmp[up + x] + 0.5f * tmp[mid + x] + 0.25f * tmp[down + x];
                }
            }
        }
        return cur;
    }

    /// <summary>Adaptive instance normalization: each channel of the result given the input's mean and spread.</summary>
    internal static float[] AdaInColorFix(float[] content, float[] style, int plane)
    {
        static (double Mean, double Std) Stats(float[] a, int o, int n)
        {
            double sum = 0, sq = 0;
            for (var i = 0; i < n; i++) sum += a[o + i];
            var mean = sum / n;
            for (var i = 0; i < n; i++) sq += (a[o + i] - mean) * (a[o + i] - mean);
            return (mean, Math.Sqrt(sq / Math.Max(1, n - 1) + 1e-5));
        }
        var r = new float[content.Length];
        for (var c = 0; c < content.Length / plane; c++)
        {
            var o = c * plane;
            var (cm, cs) = Stats(content, o, plane);
            var (sm, ss) = Stats(style, o, plane);
            for (var i = 0; i < plane; i++) r[o + i] = (float)((content[o + i] - cm) / cs * ss + sm);
        }
        return r;
    }

    // ------------------------------------------------------------------ conversions

    /// <summary>RGB in [-1, 1], padded to <paramref name="pw"/> × <paramref name="ph"/> by repeating the edges.</summary>
    internal static float[] ToTensor(RgbaImage img, int pw, int ph)
    {
        int W = img.Width, H = img.Height, plane = pw * ph;
        var t = new float[3 * plane];
        var p = img.Pixels;
        for (var y = 0; y < ph; y++)
        {
            var row = Math.Min(y, H - 1) * W;
            for (var x = 0; x < pw; x++)
            {
                var si = (row + Math.Min(x, W - 1)) * 4;
                var di = y * pw + x;
                t[di] = p[si] / 127.5f - 1;
                t[plane + di] = p[si + 1] / 127.5f - 1;
                t[2 * plane + di] = p[si + 2] / 127.5f - 1;
            }
        }
        return t;
    }

    /// <summary>The top-left <paramref name="w"/> × <paramref name="h"/> of a [3, ph, pw] tensor as an opaque image.</summary>
    internal static RgbaImage FromTensor(float[] t, int pw, int ph, int w, int h)
    {
        var img = new RgbaImage(w, h);
        var p = img.Pixels;
        var plane = pw * ph;
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
            {
                var si = y * pw + x;
                var di = (y * w + x) * 4;
                p[di] = ToByte(t[si]);
                p[di + 1] = ToByte(t[plane + si]);
                p[di + 2] = ToByte(t[2 * plane + si]);
                p[di + 3] = 255;
            }
        return img;
    }

    static byte ToByte(float v)
    {
        var x = (v + 1) * 127.5f + 0.5f;
        return x <= 0 ? (byte)0 : x >= 255 ? (byte)255 : (byte)x;
    }

    static int RoundUp(int v, int m) => (v + m - 1) / m * m;
}
