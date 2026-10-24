using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using SessionOptions = Microsoft.ML.OnnxRuntime.SessionOptions;

namespace ZawUpscale;

public interface IFaceDetector
{
    /// <summary>The faces in the image, best first.</summary>
    List<DetectedFace> Detect(RgbaImage img, CancellationToken ct);
}

public interface IFaceRestorer
{
    /// <summary>Width and height of the aligned crops it takes and returns.</summary>
    int Size { get; }
    /// <summary>Restores one aligned face: planar RGB in [0, 1], 3 × Size × Size, in and out.</summary>
    float[] Restore(float[] chw);
}

/// <summary>
/// Face restoration after upscaling: faces are found in the original image, cut out aligned, rebuilt by the restoration
/// network and blended over the upscaled result. Taking them from the original gives the network the kind of
/// degraded face it was trained on, rather than one the upscaler has already smoothed.
/// </summary>
public static class FaceEnhancer
{
    /// <summary>Restores the faces of <paramref name="original"/> in <paramref name="upscaled"/> (changed in place). Returns how many there were.</summary>
    public static int Enhance(IFaceDetector detector, IFaceRestorer restorer, RgbaImage original, RgbaImage upscaled, double strength,
        IProgress<double>? progress = null, CancellationToken ct = default)
    {
        var faces = detector.Detect(original, ct);
        progress?.Report(faces.Count == 0 ? 1 : 0.1);
        var size = restorer.Size;
        for (var i = 0; i < faces.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            var toCrop = FaceAlign.ToCrop(faces[i], size);
            var restored = restorer.Restore(FaceAlign.Crop(original, toCrop, size));
            if (restored.Length != 3 * size * size)
                throw new ApiException($"The face model returned {restored.Length} values; expected {3 * size * size}.");
            FaceAlign.Paste(upscaled, original.Width, original.Height, restored, size, toCrop, strength);
            progress?.Report(0.1 + 0.9 * (i + 1) / faces.Count);
        }
        return faces.Count;
    }
}

/// <summary>
/// The YuNet face detector (OpenCV Zoo, 2023mar): a 640 × 640 BGR input in 0–255, and for strides 8, 16 and 32 a
/// class score, an objectness score, a box and five landmarks per grid cell.
/// </summary>
public sealed class YuNetDetector(InferenceSession session, Func<RunOptions> runOptions) : IFaceDetector
{
    public const int InputSize = 640;
    public const float ScoreThreshold = 0.6f;
    const float NmsThreshold = 0.3f;
    /// <summary>Faces smaller than this (in pixels of the original) are left alone: there's too little to go on.</summary>
    const float MinFace = 10;
    static readonly int[] Strides = [8, 16, 32];

    public List<DetectedFace> Detect(RgbaImage img, CancellationToken ct)
    {
        int w = img.Width, h = img.Height;
        var found = new List<DetectedFace>();
        // The whole image, fitted to the network's input (small images are enlarged, which helps with small faces).
        RunRegion(img, 0, 0, w, h, found);
        // In large images, faces of a group photo would be a few pixels at 640: look again in tiles, at up to 2048 px.
        var longSide = Math.Max(w, h);
        if (longSide > InputSize * 3 / 2)
        {
            var tile = (int)Math.Ceiling(InputSize / Math.Min(1.0, 2048.0 / longSide));
            var step = tile * 4 / 5;
            for (var y = 0; y < h; y += step)
            {
                for (var x = 0; x < w; x += step)
                {
                    ct.ThrowIfCancellationRequested();
                    RunRegion(img, x, y, Math.Min(tile, w - x), Math.Min(tile, h - y), found);
                    if (x + tile >= w) break;
                }
                if (y + tile >= h) break;
            }
        }
        return Nms(found.Where(f => Math.Min(f.Width, f.Height) >= MinFace).ToList(), NmsThreshold);
    }

    void RunRegion(RgbaImage img, int x0, int y0, int rw, int rh, List<DetectedFace> found)
    {
        var region = x0 == 0 && y0 == 0 && rw == img.Width && rh == img.Height ? img : Cut(img, x0, y0, rw, rh);
        var k = (double)InputSize / Math.Max(rw, rh);
        int iw = Math.Clamp((int)Math.Round(rw * k), 1, InputSize), ih = Math.Clamp((int)Math.Round(rh * k), 1, InputSize);
        var fitted = ImageIO.Resize(region, iw, ih);
        // Planar BGR, unnormalized; the rest stays black.
        const int plane = InputSize * InputSize;
        var input = new float[3 * plane];
        var p = fitted.Pixels;
        for (var y = 0; y < ih; y++)
            for (var x = 0; x < iw; x++)
            {
                var si = (y * iw + x) * 4;
                var di = y * InputSize + x;
                input[di] = p[si + 2];
                input[plane + di] = p[si + 1];
                input[2 * plane + di] = p[si];
            }

        using var tensor = OrtValue.CreateTensorValueFromMemory(input, [1, 3, InputSize, InputSize]);
        var names = Strides.SelectMany(s => new[] { $"cls_{s}", $"obj_{s}", $"bbox_{s}", $"kps_{s}" }).ToArray();
        using var outputs = session.Run(runOptions(), ["input"], [tensor], names);
        var levels = Strides.Select((s, i) => new YuNetLevel(s,
            outputs[4 * i].GetTensorDataAsSpan<float>().ToArray(), outputs[4 * i + 1].GetTensorDataAsSpan<float>().ToArray(),
            outputs[4 * i + 2].GetTensorDataAsSpan<float>().ToArray(), outputs[4 * i + 3].GetTensorDataAsSpan<float>().ToArray())).ToList();

        float kx = (float)iw / rw, ky = (float)ih / rh;
        foreach (var f in Decode(levels, InputSize, ScoreThreshold))
            found.Add(f with
            {
                X = x0 + f.X / kx, Y = y0 + f.Y / ky, Width = f.Width / kx, Height = f.Height / ky,
                Landmarks = f.Landmarks.Select(l => new Vector2(x0 + l.X / kx, y0 + l.Y / ky)).ToArray(),
            });
    }

    static RgbaImage Cut(RgbaImage img, int x0, int y0, int w, int h)
    {
        var dst = new RgbaImage(w, h);
        for (var y = 0; y < h; y++)
            Buffer.BlockCopy(img.Pixels, ((y0 + y) * img.Width + x0) * 4, dst.Pixels, y * w * 4, w * 4);
        return dst;
    }

    /// <summary>One stride's outputs: per grid cell (row-major) a class and an objectness score, a box (dx, dy, log w, log h) and 5 landmarks.</summary>
    internal sealed record YuNetLevel(int Stride, float[] Cls, float[] Obj, float[] Box, float[] Kps);

    /// <summary>Decodes YuNet's outputs as OpenCV's FaceDetectorYN does, in input pixels.</summary>
    internal static List<DetectedFace> Decode(IEnumerable<YuNetLevel> levels, int size, float threshold)
    {
        var faces = new List<DetectedFace>();
        foreach (var l in levels)
        {
            var cols = size / l.Stride;
            for (var idx = 0; idx < l.Cls.Length; idx++)
            {
                var score = MathF.Sqrt(Math.Clamp(l.Cls[idx], 0, 1) * Math.Clamp(l.Obj[idx], 0, 1));
                if (score < threshold) continue;
                int r = idx / cols, c = idx % cols, s = l.Stride;
                float cx = (c + l.Box[idx * 4]) * s, cy = (r + l.Box[idx * 4 + 1]) * s;
                float bw = MathF.Exp(l.Box[idx * 4 + 2]) * s, bh = MathF.Exp(l.Box[idx * 4 + 3]) * s;
                var marks = new Vector2[5];
                for (var n = 0; n < 5; n++) marks[n] = new Vector2((l.Kps[idx * 10 + 2 * n] + c) * s, (l.Kps[idx * 10 + 2 * n + 1] + r) * s);
                faces.Add(new DetectedFace(cx - bw / 2, cy - bh / 2, bw, bh, score, marks));
            }
        }
        return faces;
    }

    /// <summary>Non-maximum suppression: keeps the best of overlapping boxes.</summary>
    internal static List<DetectedFace> Nms(List<DetectedFace> faces, float iouThreshold)
    {
        var kept = new List<DetectedFace>();
        foreach (var f in faces.OrderByDescending(f => f.Score))
            if (kept.All(k => Iou(k, f) <= iouThreshold)) kept.Add(f);
        return kept;
    }

    static float Iou(DetectedFace a, DetectedFace b)
    {
        var ix = Math.Max(0, Math.Min(a.X + a.Width, b.X + b.Width) - Math.Max(a.X, b.X));
        var iy = Math.Max(0, Math.Min(a.Y + a.Height, b.Y + b.Height) - Math.Max(a.Y, b.Y));
        var inter = ix * iy;
        return inter / (a.Width * a.Height + b.Width * b.Height - inter);
    }
}

/// <summary>
/// An ONNX face restoration network on 512 × 512 aligned faces, RGB in [−1, 1]: GFPGAN, or CodeFormer, which also takes
/// a fidelity weight (0 rebuilds the most, 1 stays closest to the input).
/// </summary>
public sealed class OnnxFaceRestorer : IFaceRestorer
{
    /// <summary>CodeFormer's fidelity: its authors' default, a balance between quality and likeness.</summary>
    public const double Fidelity = 0.5;
    readonly InferenceSession session;
    readonly Func<RunOptions> runOptions;
    readonly string inputName, outputName;
    readonly string? weightName;

    public int Size { get; }

    public OnnxFaceRestorer(InferenceSession session, Func<RunOptions> runOptions)
    {
        this.session = session;
        this.runOptions = runOptions;
        var image = session.InputMetadata.FirstOrDefault(i => i.Value.Dimensions.Length == 4);
        if (image.Key == null || image.Value.ElementDataType != Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float)
            throw new ApiException("This doesn't look like a face restoration model: it should take a float image tensor [1, 3, 512, 512].");
        inputName = image.Key;
        var d = image.Value.Dimensions;
        Size = d[2] > 0 ? d[2] : 512;
        weightName = session.InputMetadata.Keys.FirstOrDefault(k => k == "weight");
        outputName = session.OutputMetadata.Keys.Contains("output") ? "output" : session.OutputMetadata.Keys.First();
    }

    public float[] Restore(float[] chw)
    {
        var x = new float[chw.Length];
        for (var i = 0; i < x.Length; i++) x[i] = chw[i] * 2 - 1;
        using var image = OrtValue.CreateTensorValueFromMemory(x, [1, 3, Size, Size]);
        using var weight = weightName != null ? OrtValue.CreateTensorValueFromMemory(new[] { Fidelity }, []) : null;
        string[] names = weightName != null ? [inputName, weightName] : [inputName];
        OrtValue[] values = weight != null ? [image, weight] : [image];
        using var outputs = session.Run(runOptions(), names, values, [outputName]);
        var y = outputs[0].GetTensorDataAsSpan<float>();
        var r = new float[y.Length];
        for (var i = 0; i < r.Length; i++) r[i] = Math.Clamp((y[i] + 1) * 0.5f, 0, 1);
        return r;
    }
}

/// <summary>A face detector and a restoration network loaded together on one device, as the engine keeps them.</summary>
public sealed class FaceModels : IEngineModel
{
    readonly InferenceSession detectorSession, restorerSession;
    RunOptions runOptions = new();

    public string Device { get; }
    public IFaceDetector Detector { get; }
    public IFaceRestorer Restorer { get; }

    public FaceModels(string restorerPath, string detectorPath, SessionOptions options, string device)
    {
        Device = device;
        // The detector is tiny and runs a handful of times per image; the CPU spares the GPU a session.
        detectorSession = new InferenceSession(detectorPath, new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL });
        try
        {
            restorerSession = new InferenceSession(restorerPath, options);
            Detector = new YuNetDetector(detectorSession, () => runOptions);
            Restorer = new OnnxFaceRestorer(restorerSession, () => runOptions);
        }
        catch
        {
            detectorSession.Dispose();
            restorerSession?.Dispose();
            throw;
        }
    }

    public int Enhance(RgbaImage original, RgbaImage upscaled, double strength, IProgress<double>? progress, CancellationToken ct)
    {
        try
        {
            return FaceEnhancer.Enhance(Detector, Restorer, original, upscaled, strength, progress, ct);
        }
        catch (OnnxRuntimeException ex) when (runOptions.Terminate)
        {
            throw new OperationCanceledException("Cancelled.", ex);
        }
        catch (OnnxRuntimeException ex) when (OnnxSrModel.IsOutOfMemory(ex))
        {
            throw new ApiException($"Not enough memory for face restoration: {ex.Message}");
        }
    }

    public IDisposable BindCancellation(CancellationToken ct)
    {
        runOptions.Dispose();
        runOptions = new RunOptions();
        var ro = runOptions;
        return ct.Register(() =>
        {
            try { ro.Terminate = true; }
            catch (ObjectDisposedException) { /* unloaded meanwhile */ }
        });
    }

    public void Dispose()
    {
        detectorSession.Dispose();
        restorerSession.Dispose();
        runOptions.Dispose();
    }
}
