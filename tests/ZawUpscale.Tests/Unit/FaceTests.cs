using System.Numerics;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.ML.OnnxRuntime;
using ZawUpscale.Tests.Infrastructure;

namespace ZawUpscale.Tests.Unit;

public sealed class FaceTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "zawupscale-tests", Guid.NewGuid().ToString("n"));

    public void Dispose()
    {
        try { Directory.Delete(root, recursive: true); } catch (IOException) { /* best effort */ }
    }

    sealed class FakeDetector(params DetectedFace[] faces) : IFaceDetector
    {
        public List<DetectedFace> Detect(RgbaImage img, CancellationToken ct) => [.. faces];
    }

    sealed class FakeRestorer(Func<float, float> f, int size = 512) : IFaceRestorer
    {
        public int Size => size;
        public int Calls { get; private set; }
        public float[] Restore(float[] chw)
        {
            Calls++;
            return chw.Select(f).ToArray();
        }
    }

    /// <summary>A face whose landmarks are the template's, scaled to <paramref name="size"/> px, rotated by <paramref name="degrees"/> about its centre and moved to (<paramref name="cx"/>, <paramref name="cy"/>).</summary>
    static DetectedFace Face(float cx, float cy, float size, float degrees = 0)
    {
        var rot = Matrix3x2.CreateRotation(degrees * MathF.PI / 180, new Vector2(256, 256));
        var marks = FaceAlign.Template512.Select(p => (Vector2.Transform(p, rot) - new Vector2(256, 256)) * (size / 512) + new Vector2(cx, cy)).ToArray();
        return new DetectedFace(cx - size / 4, cy - size / 4, size / 2, size / 2, 0.9f, marks);
    }

    /// <summary>A smooth image (linear in x and y), which interpolation reproduces exactly; with <paramref name="scale"/>, the same picture enlarged.</summary>
    static RgbaImage Gradient(int w, int h, double scale = 1, byte alpha = 255)
    {
        var img = new RgbaImage((int)(w * scale), (int)(h * scale));
        for (var y = 0; y < img.Height; y++)
            for (var x = 0; x < img.Width; x++)
            {
                double sx = (x + 0.5) / scale - 0.5, sy = (y + 0.5) / scale - 0.5;
                var i = (y * img.Width + x) * 4;
                img.Pixels[i] = (byte)Math.Round(20 + 1.5 * sx);
                img.Pixels[i + 1] = (byte)Math.Round(20 + 1.5 * sy);
                img.Pixels[i + 2] = (byte)Math.Round(10 + 0.7 * (sx + sy));
                img.Pixels[i + 3] = alpha;
            }
        return img;
    }

    static int MaxDiff(RgbaImage a, RgbaImage b, int x0, int y0, int x1, int y1)
    {
        var max = 0;
        for (var y = y0; y < y1; y++)
            for (var x = x0; x < x1; x++)
                for (var c = 0; c < 3; c++)
                {
                    var i = (y * a.Width + x) * 4 + c;
                    max = Math.Max(max, Math.Abs(a.Pixels[i] - b.Pixels[i]));
                }
        return max;
    }

    [Fact]
    public void Similarity_fit_recovers_a_transform_and_inverts()
    {
        var t = new Similarity(1.2 * Math.Cos(0.3), 1.2 * Math.Sin(0.3), 15, -7);
        var from = new[] { new Vector2(3, 4), new Vector2(50, 10), new Vector2(20, 60), new Vector2(-5, 33) };
        var fit = Similarity.Fit(from, from.Select(p => t.Apply(p.X, p.Y)).ToArray());
        Assert.Equal(t.A, fit.A, 5);
        Assert.Equal(t.B, fit.B, 5);
        Assert.Equal(t.Tx, fit.Tx, 4);
        Assert.Equal(t.Ty, fit.Ty, 4);
        var back = fit.Inverse().Apply(fit.Apply(12, 34).X, fit.Apply(12, 34).Y);
        Assert.Equal(12, back.X, 3);
        Assert.Equal(34, back.Y, 3);
    }

    [Fact]
    public void Aligned_crop_puts_the_landmarks_on_the_template()
    {
        var face = Face(60, 50, 80, degrees: 25);
        var t = FaceAlign.ToCrop(face, 512);
        for (var i = 0; i < 5; i++)
        {
            var p = t.Apply(face.Landmarks[i].X, face.Landmarks[i].Y);
            Assert.Equal(FaceAlign.Template512[i].X, p.X, 2);
            Assert.Equal(FaceAlign.Template512[i].Y, p.Y, 2);
        }
    }

    [Theory]
    [InlineData(0, 2.0)]
    [InlineData(20, 2.0)]
    [InlineData(-10, 3.0)]
    public void Unchanged_faces_paste_back_seamlessly(float degrees, double scale)
    {
        var src = Gradient(128, 128);
        var target = Gradient(128, 128, scale);
        var expected = Gradient(128, 128, scale);
        var n = FaceEnhancer.Enhance(new FakeDetector(Face(64, 64, 64, degrees)), new FakeRestorer(v => v), src, target, 1);
        Assert.Equal(1, n);
        // The crop is cut from the original and blended over the enlarged image: where nothing was restored, nothing changes.
        Assert.True(MaxDiff(target, expected, 0, 0, target.Width, target.Height) <= 2);
    }

    [Fact]
    public void Restored_faces_are_blended_by_strength_inside_their_crop_only()
    {
        var src = Gradient(128, 128, alpha: 200);
        var invert = new FakeRestorer(v => 1 - v);
        var full = Gradient(128, 128, 2, alpha: 200);
        FaceEnhancer.Enhance(new FakeDetector(Face(64, 64, 64)), invert, src, full, 1);
        var half = Gradient(128, 128, 2, alpha: 200);
        FaceEnhancer.Enhance(new FakeDetector(Face(64, 64, 64)), invert, src, half, 0.5);
        var none = Gradient(128, 128, 2, alpha: 200);
        FaceEnhancer.Enhance(new FakeDetector(Face(64, 64, 64)), invert, src, none, 0);
        var orig = Gradient(128, 128, 2, alpha: 200);

        int Px(RgbaImage img, int x, int y, int c) => img.Pixels[(y * img.Width + x) * 4 + c];
        // The centre of the face (64, 64 in the original) is inverted, or half way at half strength.
        var v = Px(orig, 128, 128, 0);
        Assert.InRange(Px(full, 128, 128, 0), 255 - v - 2, 255 - v + 2);
        Assert.InRange(Px(half, 128, 128, 0), 125, 130);
        // Far outside the crop, and with no strength, nothing changes; alpha is never touched.
        Assert.Equal(Px(orig, 4, 4, 0), Px(full, 4, 4, 0));
        Assert.Equal(Px(orig, 250, 250, 1), Px(full, 250, 250, 1));
        Assert.Equal(orig.Pixels, none.Pixels);
        Assert.All(Enumerable.Range(0, full.Width * full.Height), i => Assert.Equal(200, full.Pixels[i * 4 + 3]));
    }

    [Fact]
    public void Every_face_is_restored_and_progress_ends_at_one()
    {
        var restorer = new FakeRestorer(v => v, size: 128);
        var reports = new List<double>();
        var n = FaceEnhancer.Enhance(new FakeDetector(Face(30, 30, 40), Face(90, 80, 50)), restorer, Gradient(128, 128), Gradient(128, 128, 2), 0.8,
            new SyncProgress(reports.Add));
        Assert.Equal(2, n);
        Assert.Equal(2, restorer.Calls);
        Assert.Equal(1, reports[^1], 6);

        var target = Gradient(128, 128, 2);
        Assert.Equal(0, FaceEnhancer.Enhance(new FakeDetector(), restorer, Gradient(128, 128), target, 1));
        Assert.Equal(Gradient(128, 128, 2).Pixels, target.Pixels);
    }

    sealed class SyncProgress(Action<double> report) : IProgress<double>
    {
        public void Report(double value) => report(value);
    }

    [Fact]
    public void YuNet_outputs_decode_like_OpenCV()
    {
        // One 640 × 640 level of stride 32: 20 × 20 cells. Cell (row 3, column 5) holds a face; another is below the threshold.
        var cls = new float[400];
        var obj = new float[400];
        var box = new float[400 * 4];
        var kps = new float[400 * 10];
        var idx = 3 * 20 + 5;
        cls[idx] = 0.81f;
        obj[idx] = 1.5f; // clamped to 1
        box[idx * 4] = 0.5f;
        box[idx * 4 + 1] = 0.25f;
        box[idx * 4 + 2] = MathF.Log(2);
        kps[idx * 10] = 0.1f;
        kps[idx * 10 + 1] = 0.2f;
        cls[100] = obj[100] = 0.5f;

        var faces = YuNetDetector.Decode([new YuNetDetector.YuNetLevel(32, cls, obj, box, kps)], 640, YuNetDetector.ScoreThreshold);
        var f = Assert.Single(faces);
        Assert.Equal(0.9f, f.Score, 4);
        Assert.Equal(176 - 32, f.X, 3);
        Assert.Equal(104 - 16, f.Y, 3);
        Assert.Equal(64, f.Width, 3);
        Assert.Equal(32, f.Height, 3);
        Assert.Equal(new Vector2(5.1f * 32, 3.2f * 32), f.Landmarks[0]);
        Assert.Equal(new Vector2(5 * 32, 3 * 32), f.Landmarks[4]);
    }

    [Fact]
    public void Overlapping_detections_keep_the_best()
    {
        static DetectedFace Box(float x, float y, float score) => new(x, y, 50, 50, score, new Vector2[5]);
        var kept = YuNetDetector.Nms([Box(0, 0, 0.7f), Box(5, 5, 0.9f), Box(200, 0, 0.8f)], 0.3f);
        Assert.Equal([0.9f, 0.8f], kept.Select(k => k.Score));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Onnx_restorer_maps_colors_through_minus_one_to_one(bool weight)
    {
        var path = TestModels.Write(root, weight ? "codeformer-like" : "gfpgan-like", TestModels.FaceRestorer(weight, size: 64));
        using var session = new InferenceSession(path);
        using var ro = new RunOptions();
        var restorer = new OnnxFaceRestorer(session, () => ro);
        Assert.Equal(64, restorer.Size);
        var input = Enumerable.Range(0, 3 * 64 * 64).Select(i => (i % 101) / 100f).ToArray();
        var output = restorer.Restore(input);
        // The stand-in negates in [−1, 1], which is 1 − x in [0, 1].
        for (var i = 0; i < input.Length; i++) Assert.Equal(1 - input[i], output[i], 5);
    }

    [Fact]
    public void Face_models_need_the_detector_and_are_not_upscalers()
    {
        var opts = AppOptions.Parse(["--config", Path.Combine(root, "cfg")]);
        Directory.CreateDirectory(opts.ConfigDir);
        var store = new ModelStore(opts, NullLogger<ModelStore>.Instance);
        var list = store.List();
        Assert.Equal("face", list.Single(m => m.Id == "gfpgan-v1.4").Kind);
        Assert.Equal("face", list.Single(m => m.Id == "codeformer").Kind);
        Assert.DoesNotContain(list, m => m.Id == ModelStore.FaceDetectorId);

        File.WriteAllBytes(store.PathOf("gfpgan-v1.4"), TestModels.FaceRestorer());
        Assert.False(store.IsInstalled("gfpgan-v1.4"));
        File.WriteAllBytes(store.PathOf(ModelStore.FaceDetectorId), [0]);
        Assert.True(store.IsInstalled("gfpgan-v1.4"));
        Assert.DoesNotContain(store.List(), m => m.Id == ModelStore.FaceDetectorId); // not shown as a custom model either

        using var engine = new Engine(NullLogger<Engine>.Instance);
        using var jobs = new Jobs(engine, store, opts, NullLogger<Jobs>.Instance);
        JobRequest Request(UpscaleSettings s) => new([new JobItemRequest(Path.Combine(root, "a.png"))], s);
        Assert.Contains("restores faces", Assert.Throws<ApiException>(() => jobs.Start(Request(new UpscaleSettings { Model = "gfpgan-v1.4" }))).Message);
        File.WriteAllBytes(store.PathOf("realesr-general-x4v3"), TestModels.NearestResize(4));
        Assert.Contains("face restoration model isn't installed", Assert.Throws<ApiException>(() => jobs.Start(Request(new UpscaleSettings { Faces = "codeformer" }))).Message);

        // Results with faces restored go in their own folder, so they don't count as done when faces are turned on.
        var input = Path.Combine(root, "pics", "cat.jpg");
        Assert.Equal(Path.Combine(root, "pics", "realesr-general-x4v3_x4_faces", "cat.jpg"),
            jobs.OutputPath(new JobItemRequest(input), Jobs.Validate(new UpscaleSettings { Faces = "gfpgan-v1.4" })));
    }

    [Fact]
    public void Face_settings_are_validated()
    {
        Assert.Null(Jobs.Validate(new UpscaleSettings { Faces = "" }).Faces);
        Assert.Null(Jobs.Validate(new UpscaleSettings { Faces = "none" }).Faces);
        Assert.Equal("gfpgan-v1.4", Jobs.Validate(new UpscaleSettings { Faces = " gfpgan-v1.4 " }).Faces);
        Assert.Throws<ApiException>(() => Jobs.Validate(new UpscaleSettings { Faces = "realesrgan-x4plus" }));
        Assert.Throws<ApiException>(() => Jobs.Validate(new UpscaleSettings { Faces = "gfpgan-v1.4", FaceStrength = 1.5 }));
        Assert.Equal("m_x4_faces", Jobs.ResultsFolder("m", 4, faces: true));
    }
}
