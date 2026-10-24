using System.Numerics;

namespace ZawUpscale;

/// <summary>A face found in an image: its box, detection score and five landmarks (eyes, nose tip, mouth corners, as seen in the image from left to right).</summary>
public sealed record DetectedFace(float X, float Y, float Width, float Height, float Score, Vector2[] Landmarks);

/// <summary>
/// A similarity transform (rotation, uniform scale, translation): x' = A·x − B·y + Tx, y' = B·x + A·y + Ty.
/// Points are continuous coordinates, where pixel i covers [i, i + 1).
/// </summary>
public readonly record struct Similarity(double A, double B, double Tx, double Ty)
{
    public double Scale => Math.Sqrt(A * A + B * B);

    public Vector2 Apply(double x, double y) => new((float)(A * x - B * y + Tx), (float)(B * x + A * y + Ty));

    public Similarity Inverse()
    {
        var d = A * A + B * B;
        double ia = A / d, ib = -B / d;
        return new Similarity(ia, ib, -(ia * Tx - ib * Ty), -(ib * Tx + ia * Ty));
    }

    /// <summary>The least-squares similarity that maps <paramref name="from"/> onto <paramref name="to"/> (Umeyama, without reflection).</summary>
    public static Similarity Fit(IReadOnlyList<Vector2> from, IReadOnlyList<Vector2> to)
    {
        if (from.Count != to.Count || from.Count < 2) throw new ArgumentException("At least two matching points are needed.");
        double fx = 0, fy = 0, tx = 0, ty = 0;
        for (var i = 0; i < from.Count; i++) { fx += from[i].X; fy += from[i].Y; tx += to[i].X; ty += to[i].Y; }
        fx /= from.Count; fy /= from.Count; tx /= from.Count; ty /= from.Count;
        double a = 0, b = 0, n = 0;
        for (var i = 0; i < from.Count; i++)
        {
            double px = from[i].X - fx, py = from[i].Y - fy, qx = to[i].X - tx, qy = to[i].Y - ty;
            a += px * qx + py * qy;
            b += px * qy - py * qx;
            n += px * px + py * py;
        }
        if (n < 1e-9) throw new ArgumentException("The points coincide.");
        a /= n;
        b /= n;
        return new Similarity(a, b, tx - (a * fx - b * fy), ty - (b * fx + a * fy));
    }
}

/// <summary>
/// Cuts faces out of an image in the pose face restoration networks are trained on (FFHQ: eyes level, at fixed
/// positions in a square), and blends restored faces back into a larger version of the image.
/// </summary>
public static class FaceAlign
{
    /// <summary>The five landmarks of a 512 × 512 FFHQ-aligned face, as used by GFPGAN and CodeFormer.</summary>
    public static readonly Vector2[] Template512 =
    [
        new(192.98138f, 239.94708f), new(318.90277f, 240.19360f), new(256.63416f, 314.01935f),
        new(201.26117f, 371.41043f), new(313.08905f, 371.15118f),
    ];

    /// <summary>Feathering of the pasted face: fully transparent up to this distance from the crop's edge (in 512 px units)…</summary>
    const double FeatherStart = 12;
    /// <summary>…and fully opaque from this one on.</summary>
    const double FeatherEnd = 88;

    /// <summary>The transform from image coordinates to a <paramref name="size"/> × <paramref name="size"/> aligned crop.</summary>
    public static Similarity ToCrop(DetectedFace face, int size)
    {
        var k = size / 512f;
        return Similarity.Fit(face.Landmarks, Template512.Select(p => p * k).ToArray());
    }

    /// <summary>
    /// The aligned crop as planar RGB in [0, 1]. Where the face is larger than the crop, several samples per pixel
    /// are averaged so it doesn't alias; outside the image the edge is repeated.
    /// </summary>
    public static float[] Crop(RgbaImage img, Similarity toCrop, int size)
    {
        var inv = toCrop.Inverse();
        var k = Math.Clamp((int)Math.Ceiling(1 / toCrop.Scale), 1, 4);
        var plane = size * size;
        var chw = new float[3 * plane];
        var acc = new float[3];
        for (var v = 0; v < size; v++)
            for (var u = 0; u < size; u++)
            {
                acc[0] = acc[1] = acc[2] = 0;
                for (var j = 0; j < k; j++)
                    for (var i = 0; i < k; i++)
                    {
                        var p = inv.Apply(u + (i + 0.5) / k, v + (j + 0.5) / k);
                        Bilinear(img, p.X - 0.5f, p.Y - 0.5f, acc);
                    }
                var di = v * size + u;
                var norm = 1f / (255f * k * k);
                chw[di] = acc[0] * norm;
                chw[plane + di] = acc[1] * norm;
                chw[2 * plane + di] = acc[2] * norm;
            }
        return chw;
    }

    /// <summary>
    /// Blends a restored crop (planar RGB in [0, 1]) into <paramref name="target"/>, which shows the image the face was
    /// found in at another size. The crop fades out towards its edges; <paramref name="strength"/> (0–1) scales it all.
    /// Alpha is left alone.
    /// </summary>
    public static void Paste(RgbaImage target, int sourceWidth, int sourceHeight, float[] crop, int size, Similarity toCrop, double strength)
    {
        double sx = (double)target.Width / sourceWidth, sy = (double)target.Height / sourceHeight;
        // The crop's corners in the target, to visit only the pixels it covers.
        var inv = toCrop.Inverse();
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        foreach (var (cx, cy) in new (double, double)[] { (0, 0), (size, 0), (0, size), (size, size) })
        {
            var p = inv.Apply(cx, cy);
            minX = Math.Min(minX, p.X * sx); maxX = Math.Max(maxX, p.X * sx);
            minY = Math.Min(minY, p.Y * sy); maxY = Math.Max(maxY, p.Y * sy);
        }
        int x0 = Math.Max(0, (int)Math.Floor(minX)), x1 = Math.Min(target.Width, (int)Math.Ceiling(maxX));
        int y0 = Math.Max(0, (int)Math.Floor(minY)), y1 = Math.Min(target.Height, (int)Math.Ceiling(maxY));
        var px = target.Pixels;
        var plane = size * size;
        var unit = size / 512.0;
        var rgb = new float[3];
        for (var y = y0; y < y1; y++)
            for (var x = x0; x < x1; x++)
            {
                var q = toCrop.Apply((x + 0.5) / sx, (y + 0.5) / sy);
                double edge = Math.Min(Math.Min(q.X, q.Y), Math.Min(size - q.X, size - q.Y)) / unit;
                if (edge <= FeatherStart) continue;
                var t = Math.Min(1, (edge - FeatherStart) / (FeatherEnd - FeatherStart));
                var w = (float)(strength * t * t * (3 - 2 * t));
                if (w <= 0) continue;
                Bicubic(crop, size, plane, q.X - 0.5f, q.Y - 0.5f, rgb);
                var i = (y * target.Width + x) * 4;
                for (var c = 0; c < 3; c++)
                {
                    var v = px[i + c] + (rgb[c] * 255f - px[i + c]) * w;
                    px[i + c] = (byte)Math.Clamp((int)(v + 0.5f), 0, 255);
                }
            }
    }

    /// <summary>Adds the bilinearly interpolated RGB at (x, y) (pixel centres at integers, edges repeated) to <paramref name="acc"/>.</summary>
    static void Bilinear(RgbaImage img, float x, float y, float[] acc)
    {
        int w = img.Width, h = img.Height;
        x = Math.Clamp(x, 0, w - 1);
        y = Math.Clamp(y, 0, h - 1);
        int xa = (int)x, ya = (int)y;
        int xb = Math.Min(xa + 1, w - 1), yb = Math.Min(ya + 1, h - 1);
        float fx = x - xa, fy = y - ya;
        var p = img.Pixels;
        int i00 = (ya * w + xa) * 4, i10 = (ya * w + xb) * 4, i01 = (yb * w + xa) * 4, i11 = (yb * w + xb) * 4;
        for (var c = 0; c < 3; c++)
        {
            var top = p[i00 + c] + (p[i10 + c] - p[i00 + c]) * fx;
            var bottom = p[i01 + c] + (p[i11 + c] - p[i01 + c]) * fx;
            acc[c] += top + (bottom - top) * fy;
        }
    }

    /// <summary>Catmull-Rom interpolation of a planar RGB crop at (x, y), so a face stretched over more pixels stays crisp.</summary>
    static void Bicubic(float[] chw, int size, int plane, float x, float y, float[] rgb)
    {
        int xi = (int)Math.Floor(x), yi = (int)Math.Floor(y);
        float fx = x - xi, fy = y - yi;
        Span<float> wx = stackalloc float[4], wy = stackalloc float[4];
        Weights(fx, wx);
        Weights(fy, wy);
        rgb[0] = rgb[1] = rgb[2] = 0;
        for (var j = 0; j < 4; j++)
        {
            var row = Math.Clamp(yi - 1 + j, 0, size - 1) * size;
            for (var i = 0; i < 4; i++)
            {
                var idx = row + Math.Clamp(xi - 1 + i, 0, size - 1);
                var w = wx[i] * wy[j];
                rgb[0] += chw[idx] * w;
                rgb[1] += chw[plane + idx] * w;
                rgb[2] += chw[2 * plane + idx] * w;
            }
        }
        for (var c = 0; c < 3; c++) rgb[c] = Math.Clamp(rgb[c], 0, 1);
    }

    static void Weights(float t, Span<float> w)
    {
        float t2 = t * t, t3 = t2 * t;
        w[0] = 0.5f * (-t3 + 2 * t2 - t);
        w[1] = 0.5f * (3 * t3 - 5 * t2 + 2);
        w[2] = 0.5f * (-3 * t3 + 4 * t2 + t);
        w[3] = 0.5f * (t3 - t2);
    }
}
