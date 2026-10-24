using Microsoft.Extensions.Logging.Abstractions;
using ZawUpscale.Tests.Infrastructure;

namespace ZawUpscale.Tests.Unit;

/// <summary>Target sizes, the adjustments after upscaling, and float16 model versions.</summary>
public sealed class SizeAndLookTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "zawupscale-tests", Guid.NewGuid().ToString("n"));
    readonly Engine engine = new(NullLogger<Engine>.Instance);
    readonly ModelStore store;
    readonly Jobs jobs;

    public SizeAndLookTests()
    {
        var opts = AppOptions.Parse(["--config", Path.Combine(root, "cfg")]);
        Directory.CreateDirectory(opts.ConfigDir);
        store = new ModelStore(opts, NullLogger<ModelStore>.Instance);
        jobs = new Jobs(engine, store, opts, NullLogger<Jobs>.Instance);
    }

    public void Dispose()
    {
        jobs.Dispose();
        engine.Dispose();
        try { Directory.Delete(root, recursive: true); } catch (IOException) { /* best effort */ }
    }

    [Theory]
    [InlineData("width", 1000, 500, 3840, 0, 0, 3.84)]
    [InlineData("height", 1000, 500, 0, 2160, 0, 4.32)]
    [InlineData("long", 1000, 500, 0, 0, 4000, 4)]
    [InlineData("long", 300, 600, 0, 0, 4000, 6.6667)]
    [InlineData("fit", 1000, 500, 3840, 2160, 0, 3.84)]   // landscape in a landscape box: width decides
    [InlineData("fit", 500, 1000, 3840, 2160, 0, 3.84)]   // portrait: the box turns to 2160 × 3840
    [InlineData("fit", 1000, 1000, 3840, 2160, 0, 2.16)]
    public void Target_sizes_give_the_right_scale(string mode, int w, int h, int tw, int th, int edge, double expected)
    {
        var s = new UpscaleSettings { Size = mode, Width = tw, Height = th, LongEdge = edge };
        Assert.Equal(expected, Jobs.ScaleFor(s, w, h), 3);
        // The output lands exactly on the target.
        var (ow, oh) = Upscaler.TargetSize(w, h, Jobs.ScaleFor(s, w, h));
        if (mode == "width") Assert.Equal(tw, ow);
        if (mode == "height") Assert.Equal(th, oh);
    }

    [Fact]
    public void Target_sizes_name_their_folders_and_files()
    {
        var input = Path.Combine(root, "pics", "cat.jpg");
        string Out(UpscaleSettings s) => jobs.OutputPath(new JobItemRequest(input), Jobs.Validate(s));
        Assert.Equal(Path.Combine(root, "pics", "realesr-general-x4v3_w3840", "cat.jpg"), Out(new UpscaleSettings { Size = "width", Width = 3840 }));
        Assert.Equal(Path.Combine(root, "pics", "realesr-general-x4v3_3840x2160", "cat.jpg"), Out(new UpscaleSettings { Size = "fit", Width = 3840, Height = 2160 }));
        Assert.Equal(Path.Combine(root, "pics", "realesr-general-x4v3_long4000", "cat_long4000.jpg"), Out(new UpscaleSettings { Size = "long", LongEdge = 4000, NameTemplate = "{name}_{scale}" }));
        Assert.Equal(Path.Combine(root, "pics", "realesr-general-x4v3_x4", "cat_x4.jpg"), Out(new UpscaleSettings { NameTemplate = "{name}_x{scale}" }));
    }

    [Fact]
    public void Scans_recognize_results_of_target_sizes()
    {
        var dir = Path.Combine(root, "tree");
        foreach (var f in new[] { "a.png", Path.Combine("m_w3840", "a.png"), Path.Combine("m_3840x2160_faces", "a.png"), Path.Combine("m_long4000", "a.png"), "a_h2160.png" })
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(dir, f))!);
            File.WriteAllBytes(Path.Combine(dir, f), [1]);
        }
        var scan = Files.Scan([dir], recursive: true, "{name}_{scale}", ["m"]);
        Assert.Equal(["a.png"], scan.Items.Select(i => i.Rel));
        Assert.Equal(4, scan.PreviousResults);
    }

    async Task<Job> RunJob(string input, UpscaleSettings s)
    {
        var job = jobs.Start(new JobRequest([new JobItemRequest(input)], s with { Device = "cpu", OutputDir = Path.Combine(root, "out"), Existing = "overwrite" }));
        for (var i = 0; i < 200 && jobs.Get(job.Id)!.Status is "queued" or "running"; i++) await Task.Delay(50);
        return jobs.Get(job.Id)!;
    }

    [Fact]
    public async Task Jobs_reach_the_target_size_up_and_down()
    {
        TestModels.Write(store.Dir, "realesr-general-x4v3", TestModels.NearestResize(2));
        var input = Path.Combine(root, "in.png");
        ImageIO.Save(TestImages.Pattern(37, 23), input, "png", 100);

        var up = (await RunJob(input, new UpscaleSettings { Size = "width", Width = 100 })).Items[0];
        Assert.Equal("done", up.Status);
        Assert.Equal((100, 62), (up.OutWidth, up.OutHeight));

        // Already larger than the target: reduced to it, then run at 1×.
        var down = (await RunJob(input, new UpscaleSettings { Size = "long", LongEdge = 20 })).Items[0];
        Assert.Equal("done", down.Status);
        Assert.Equal((20, 12), (down.OutWidth, down.OutHeight));

        var tooBig = (await RunJob(input, new UpscaleSettings { Size = "width", Width = 37 * 40 })).Items[0];
        Assert.Equal("failed", tooBig.Status);
        Assert.Contains("more than 32×", tooBig.Error);
    }

    [Fact]
    public async Task Paused_jobs_start_nothing_new_and_resume()
    {
        TestModels.Write(store.Dir, "realesr-general-x4v3", TestModels.NearestResize(2));
        var a = Path.Combine(root, "a.png");
        var b = Path.Combine(root, "b.png");
        ImageIO.Save(TestImages.Pattern(512, 512), a, "png", 100);
        ImageIO.Save(TestImages.Pattern(40, 40), b, "png", 100);
        var job = jobs.Start(new JobRequest([new JobItemRequest(a), new JobItemRequest(b)],
            new UpscaleSettings { Device = "cpu", Tile = 32, OutputDir = Path.Combine(root, "out"), Existing = "overwrite" }));
        Assert.True(jobs.Pause(job.Id));
        Assert.False(jobs.Pause(job.Id)); // already paused

        await Task.Delay(600);
        var paused = jobs.Get(job.Id)!;
        Assert.True(paused.Paused);
        Assert.Equal("queued", paused.Items[1].Status); // the second image never starts while paused
        Assert.NotEqual("done", paused.Items[0].Status);

        Assert.True(jobs.Resume(job.Id));
        for (var i = 0; i < 200 && jobs.Get(job.Id)!.Status != "done"; i++) await Task.Delay(50);
        var done = jobs.Get(job.Id)!;
        Assert.All(done.Items, it => Assert.Equal("done", it.Status));
        Assert.False(done.Paused);
        // The pause doesn't count as working time.
        Assert.True(done.ActiveMs < done.Finished!.Value - done.Started!.Value - 400, $"active {done.ActiveMs} ms of {done.Finished - done.Started} ms");
        Assert.True(done.Items[0].Ms < done.Finished!.Value - done.Started!.Value - 400, $"image {done.Items[0].Ms} ms");
    }

    [Fact]
    public async Task Cancelling_a_paused_job_stops_it()
    {
        TestModels.Write(store.Dir, "realesr-general-x4v3", TestModels.NearestResize(2));
        var a = Path.Combine(root, "a.png");
        ImageIO.Save(TestImages.Pattern(512, 512), a, "png", 100);
        var job = jobs.Start(new JobRequest([new JobItemRequest(a), new JobItemRequest(a)],
            new UpscaleSettings { Device = "cpu", Tile = 32, OutputDir = Path.Combine(root, "out"), Existing = "overwrite" }));
        jobs.Pause(job.Id);
        await Task.Delay(200);
        Assert.True(jobs.Cancel(job.Id));
        for (var i = 0; i < 100 && jobs.Get(job.Id)!.Status != "cancelled"; i++) await Task.Delay(50);
        Assert.Equal("cancelled", jobs.Get(job.Id)!.Status);
    }

    static RgbaImage Solid(int w, int h, byte r, byte g, byte b, byte a = 255)
    {
        var img = new RgbaImage(w, h);
        for (var i = 0; i < img.Pixels.Length; i += 4) { img.Pixels[i] = r; img.Pixels[i + 1] = g; img.Pixels[i + 2] = b; img.Pixels[i + 3] = a; }
        return img;
    }

    [Fact]
    public void No_detail_gives_the_plain_enlargement()
    {
        var original = TestImages.Pattern(30, 20);
        var result = Solid(90, 60, 0, 0, 0);
        PostProcess.Apply(original, result, detail: 0, sharpen: 0, keepColors: false);
        Assert.Equal(ImageIO.Resize(original, 90, 60).Pixels.Where((_, i) => i % 4 != 3), result.Pixels.Where((_, i) => i % 4 != 3));
    }

    [Fact]
    public void Keep_colors_removes_a_color_shift_but_keeps_detail()
    {
        var original = Solid(40, 40, 100, 120, 140);
        // A result with the wrong colors (too red, too dark) and some fine detail of its own: a checkerboard.
        var result = new RgbaImage(160, 160);
        for (var y = 0; y < 160; y++)
            for (var x = 0; x < 160; x++)
            {
                var i = (y * 160 + x) * 4;
                var d = ((x + y) & 1) == 0 ? 10 : -10;
                result.Pixels[i] = (byte)(160 + d); result.Pixels[i + 1] = (byte)(90 + d); result.Pixels[i + 2] = (byte)(110 + d); result.Pixels[i + 3] = 255;
            }
        PostProcess.KeepColors(original, result);
        double Mean(int c) => Enumerable.Range(0, 160 * 160).Average(i => result.Pixels[i * 4 + c]);
        Assert.InRange(Mean(0), 97, 103);
        Assert.InRange(Mean(1), 117, 123);
        Assert.InRange(Mean(2), 137, 143);
        // The checkerboard is still there.
        Assert.InRange(Math.Abs(result.Pixels[0] - result.Pixels[4]), 18, 22);
    }

    [Fact]
    public void Sharpening_raises_edge_contrast_and_softening_lowers_it_alpha_untouched()
    {
        RgbaImage Edge()
        {
            var img = Solid(20, 10, 60, 60, 60, 200);
            for (var y = 0; y < 10; y++) for (var x = 10; x < 20; x++) img.Pixels[(y * 20 + x) * 4] = img.Pixels[(y * 20 + x) * 4 + 1] = img.Pixels[(y * 20 + x) * 4 + 2] = 180;
            return img;
        }
        var original = Solid(10, 5, 0, 0, 0);
        int Step(RgbaImage img) => img.Pixels[(5 * 20 + 10) * 4] - img.Pixels[(5 * 20 + 9) * 4];
        var sharp = Edge();
        PostProcess.Apply(original, sharp, 1, 1, false);
        var soft = Edge();
        PostProcess.Apply(original, soft, 1, -1, false);
        Assert.True(Step(sharp) > 120, $"sharpened step {Step(sharp)}");
        Assert.True(Step(soft) < 120, $"softened step {Step(soft)}");
        Assert.All(Enumerable.Range(0, 200), i => Assert.Equal(200, sharp.Pixels[i * 4 + 3]));
        Assert.False(PostProcess.Needed(1, 0, false));
    }

    [Fact]
    public void Float16_versions_are_chosen_when_asked_and_present()
    {
        var full = store.PathOf("realesrgan-x4plus");
        var half = store.HalfPathOf("realesrgan-x4plus");
        Assert.False(store.IsInstalled("realesrgan-x4plus"));

        File.WriteAllBytes(half, [1]);
        Assert.True(store.IsInstalled("realesrgan-x4plus"));
        Assert.Equal(half, store.ModelPath("realesrgan-x4plus", half: false)); // the only one there
        var info = store.List().Single(m => m.Id == "realesrgan-x4plus");
        Assert.True(info.HalfInstalled);
        Assert.False(info.FullInstalled);
        Assert.Equal(33_754_617, info.HalfSize);
        Assert.DoesNotContain(store.List(), m => m.Id == "realesrgan-x4plus.fp16"); // not a custom model

        File.WriteAllBytes(full, [1]);
        Assert.Equal(half, store.ModelPath("realesrgan-x4plus", half: true));
        Assert.Equal(full, store.ModelPath("realesrgan-x4plus", half: false));

        store.Delete("realesrgan-x4plus");
        Assert.False(File.Exists(full) || File.Exists(half));
    }

    [Fact]
    public void Only_the_main_models_are_featured()
    {
        var featured = store.List().Where(m => m.Featured).Select(m => m.Id);
        Assert.Equal(["realesr-general-x4v3", "realesrgan-x4plus", "realesrgan-x4plus-anime"], featured);
    }
}
