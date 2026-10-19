namespace ZawUpscale.Tests.Infrastructure;

/// <summary>
/// SUPIR's networks in plain C#, chosen so the whole pipeline has a known answer: the "VAE" averages
/// 8 × 8 blocks into latents and repeats them back, and the "UNet" predicts exactly the noise that leads
/// to the control latent. Sampling then ends on the control latent, and a block-constant image comes back
/// unchanged, through the schedule, the tiling, the blending and the color fix.
/// </summary>
public sealed class FakeSupirNetworks : ISupirNetworks
{
    public List<(int Batch, int H, int W)> UnetShapes { get; } = [];
    public List<(int H, int W)> EncodeShapes { get; } = [];
    public List<string> Prompts { get; } = [];
    public Action? OnUnet { get; set; }

    public float[] DenoiseEncode(float[] image, int h, int w) => Encode(image, h, w);

    public float[] Encode(float[] image, int h, int w)
    {
        EncodeShapes.Add((h, w));
        int lh = h / 8, lw = w / 8;
        var z = new float[4 * lh * lw];
        for (var c = 0; c < 3; c++)
            for (var y = 0; y < lh; y++)
                for (var x = 0; x < lw; x++)
                {
                    float sum = 0;
                    for (var dy = 0; dy < 8; dy++)
                        for (var dx = 0; dx < 8; dx++) sum += image[c * h * w + (y * 8 + dy) * w + x * 8 + dx];
                    z[c * lh * lw + y * lw + x] = sum / 64;
                }
        return z;
    }

    public float[] Decode(float[] latent, int h, int w)
    {
        int H = h * 8, W = w * 8;
        var img = new float[3 * H * W];
        for (var c = 0; c < 3; c++)
            for (var y = 0; y < H; y++)
                for (var x = 0; x < W; x++) img[c * H * W + y * W + x] = latent[c * h * w + y / 8 * w + x / 8];
        return img;
    }

    public float[] Unet(float[] sample, float[] timestep, float[] context, float[] y, float[] control, float controlScale, int batch, int h, int w)
    {
        UnetShapes.Add((batch, h, w));
        OnUnet?.Invoke();
        Assert.Equal(batch * 77 * 2048, context.Length);
        Assert.Equal(batch * SupirUpscaler.VectorSize, y.Length);
        var size = 4 * h * w;
        var eps = new float[batch * size];
        for (var b = 0; b < batch; b++)
        {
            var q = SupirUpscaler.TrainingSigmas[(int)timestep[b]];
            var undo = (float)Math.Sqrt(q * q + 1); // the sample came in scaled by c_in = 1 / √(σ² + 1)
            for (var i = 0; i < size; i++)
                eps[b * size + i] = (sample[b * size + i] * undo - control[b * size + i]) / (float)q;
        }
        return eps;
    }

    public TextEmbedding EncodeText(string text)
    {
        Prompts.Add(text);
        return new TextEmbedding(new float[77 * 2048], new float[1280]);
    }

    /// <summary>An image made of 8 × 8 blocks of one color each, which the fake VAE reproduces exactly.</summary>
    public static RgbaImage BlockImage(int w, int h, bool alpha = false)
    {
        var img = new RgbaImage(w, h);
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
            {
                int bx = x / 8, by = y / 8, i = (y * w + x) * 4;
                img.Pixels[i] = (byte)(bx * 37 + by * 11);
                img.Pixels[i + 1] = (byte)(bx * 5 ^ by * 23);
                img.Pixels[i + 2] = (byte)(by * 41 + 60);
                img.Pixels[i + 3] = alpha ? (byte)(x * 255 / Math.Max(1, w - 1)) : (byte)255;
            }
        return img;
    }
}
