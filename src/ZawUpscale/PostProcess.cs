namespace ZawUpscale;

/// <summary>
/// Adjustments applied to an upscaled image: how much of the AI's detail to keep (against a plain enlargement of the
/// original), the original's colors, and sharpening or softening. Alpha is left alone.
/// </summary>
public static class PostProcess
{
    /// <summary>True when the settings change anything.</summary>
    public static bool Needed(double detail, double sharpen, bool keepColors) => detail < 1 || sharpen != 0 || keepColors;

    /// <summary>Adjusts <paramref name="result"/> (changed in place), the upscaled version of <paramref name="original"/>.</summary>
    public static void Apply(RgbaImage original, RgbaImage result, double detail, double sharpen, bool keepColors)
    {
        if (detail < 1) BlendWithPlain(original, result, Math.Clamp(detail, 0, 1));
        if (keepColors) KeepColors(original, result);
        if (sharpen != 0) Sharpen(result, Math.Clamp(sharpen, -1, 1));
    }

    /// <summary>result = plain + detail · (result − plain), with the plain one a high-quality (cubic) enlargement.</summary>
    static void BlendWithPlain(RgbaImage original, RgbaImage result, double detail)
    {
        var plain = ImageIO.Resize(original, result.Width, result.Height).Pixels;
        var r = result.Pixels;
        var k = (float)detail;
        for (var i = 0; i < r.Length; i += 4)
            for (var c = 0; c < 3; c++)
                r[i + c] = Byte(plain[i + c] + (r[i + c] - plain[i + c]) * k);
    }

    /// <summary>
    /// Gives the result the original's colors and tones, keeping its detail: the difference between the original and the
    /// result shrunk to the original's size is smoothed (so only colors move, not edges) and added back enlarged.
    /// </summary>
    internal static void KeepColors(RgbaImage original, RgbaImage result)
    {
        int w = original.Width, h = original.Height;
        var small = ImageIO.Resize(result, w, h).Pixels;
        var o = original.Pixels;
        // Signed differences, stored as 128 + d/2 so they fit in bytes through the resizes.
        var diff = new RgbaImage(w, h);
        var d = diff.Pixels;
        for (var i = 0; i < d.Length; i += 4)
        {
            for (var c = 0; c < 3; c++) d[i + c] = Byte(128 + (o[i + c] - small[i + c]) * 0.5f);
            d[i + 3] = 255;
        }
        BoxBlur(diff, Math.Max(1, Math.Min(w, h) / 64));
        var big = ImageIO.Resize(diff, result.Width, result.Height).Pixels;
        var r = result.Pixels;
        for (var i = 0; i < r.Length; i += 4)
            for (var c = 0; c < 3; c++)
                r[i + c] = Byte(r[i + c] + (big[i + c] - 128) * 2f);
    }

    /// <summary>Unsharp mask for positive amounts (up to 1.5× the detail added), a blend towards a blur for negative ones.</summary>
    static void Sharpen(RgbaImage img, double amount)
    {
        var blurred = new RgbaImage(img.Width, img.Height, (byte[])img.Pixels.Clone());
        BoxBlur(blurred, 1);
        BoxBlur(blurred, 1); // twice: close to a Gaussian of radius ~1.4 px
        var b = blurred.Pixels;
        var p = img.Pixels;
        var k = (float)(amount > 0 ? amount * 1.5 : amount);
        for (var i = 0; i < p.Length; i += 4)
            for (var c = 0; c < 3; c++)
                p[i + c] = Byte(p[i + c] + (p[i + c] - b[i + c]) * k);
    }

    /// <summary>In-place box blur of the RGB channels with the given radius (horizontal, then vertical pass; edges repeated).</summary>
    internal static void BoxBlur(RgbaImage img, int radius)
    {
        int w = img.Width, h = img.Height;
        var p = img.Pixels;
        var line = new int[Math.Max(w, h) * 3];
        var n = 2 * radius + 1;
        // Rows.
        for (var y = 0; y < h; y++)
        {
            var row = y * w * 4;
            for (var x = 0; x < w; x++)
                for (var c = 0; c < 3; c++) line[x * 3 + c] = p[row + x * 4 + c];
            for (var c = 0; c < 3; c++)
            {
                var sum = 0;
                for (var k = -radius; k <= radius; k++) sum += line[Math.Clamp(k, 0, w - 1) * 3 + c];
                for (var x = 0; x < w; x++)
                {
                    p[row + x * 4 + c] = (byte)((sum + n / 2) / n);
                    sum += line[Math.Min(x + radius + 1, w - 1) * 3 + c] - line[Math.Max(x - radius, 0) * 3 + c];
                }
            }
        }
        // Columns.
        for (var x = 0; x < w; x++)
        {
            for (var y = 0; y < h; y++)
                for (var c = 0; c < 3; c++) line[y * 3 + c] = p[(y * w + x) * 4 + c];
            for (var c = 0; c < 3; c++)
            {
                var sum = 0;
                for (var k = -radius; k <= radius; k++) sum += line[Math.Clamp(k, 0, h - 1) * 3 + c];
                for (var y = 0; y < h; y++)
                {
                    p[(y * w + x) * 4 + c] = (byte)((sum + n / 2) / n);
                    sum += line[Math.Min(y + radius + 1, h - 1) * 3 + c] - line[Math.Max(y - radius, 0) * 3 + c];
                }
            }
        }
    }

    static byte Byte(float v) => v <= 0 ? (byte)0 : v >= 255 ? (byte)255 : (byte)(v + 0.5f);
}
