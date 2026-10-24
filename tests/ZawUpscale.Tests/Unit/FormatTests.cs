using Microsoft.Extensions.Logging.Abstractions;
using ZawUpscale.Tests.Infrastructure;

namespace ZawUpscale.Tests.Unit;

/// <summary>The formats read and written with ImageMagick, and animated GIF and WebP.</summary>
public sealed class FormatTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "zawupscale-tests", Guid.NewGuid().ToString("n"));

    public FormatTests() => Directory.CreateDirectory(root);

    public void Dispose()
    {
        try { Directory.Delete(root, recursive: true); } catch (IOException) { /* best effort */ }
    }

    [Theory]
    [InlineData("tif", true)]
    [InlineData("avif", false)]
    [InlineData("jxl", false)]
    [InlineData("jxl", true)] // quality 100: lossless
    public void New_formats_round_trip(string format, bool exact)
    {
        var src = TestImages.Pattern(64, 40, alpha: true);
        var path = Path.Combine(root, $"img{(exact ? "-exact" : "")}{ImageIO.ExtensionOf(format)}");
        ImageIO.Save(src, path, format, exact ? 100 : 90);
        Assert.True(ImageIO.IsImage(path));
        Assert.Equal((64, 40), ImageIO.Probe(path));
        var back = ImageIO.Load(path);
        Assert.Equal((64, 40), (back.Width, back.Height));
        if (exact) Assert.Equal(src.Pixels, back.Pixels);
        else Assert.True(back.HasAlpha());
        // The viewer gets formats web views can't show as PNG.
        if (!ImageIO.IsWebImage(path)) Assert.Equal((64, 40), ImageIO.Probe(Save(ImageIO.ToPng(path), "view.png")));
        Assert.NotEmpty(ImageIO.Thumbnail(path, 32));
    }

    string Save(byte[] bytes, string name)
    {
        var p = Path.Combine(root, name);
        File.WriteAllBytes(p, bytes);
        return p;
    }

    [Fact]
    public void Output_formats_follow_the_input()
    {
        Assert.Equal("jpg", Jobs.FormatFor("a.heic", "auto"));
        Assert.Equal("tif", Jobs.FormatFor("a.TIFF", "auto"));
        Assert.Equal("avif", Jobs.FormatFor("a.avif", "auto"));
        Assert.Equal("jxl", Jobs.FormatFor("a.jxl", "auto"));
        Assert.Equal("png", Jobs.FormatFor("a.gif", "auto"));
        Assert.Equal("gif", Jobs.FormatFor("a.gif", "auto", animated: true));
        Assert.Equal("webp", Jobs.FormatFor("a.gif", "webp", animated: true));
        Assert.Equal("tif", Jobs.Validate(new UpscaleSettings { Format = "tiff" }).Format);
    }

    static ImageIO.Animation ThreeFrames(int w, int h)
    {
        var frames = Enumerable.Range(0, 3).Select(i =>
        {
            var f = TestImages.Pattern(w, h);
            for (var p = 0; p < f.Pixels.Length; p += 4) f.Pixels[p] = (byte)(i * 100); // each frame different
            return f;
        }).ToList();
        return new ImageIO.Animation(frames, [100, 200, 300], 0);
    }

    [Theory]
    [InlineData("gif")]
    [InlineData("webp")]
    public void Animations_round_trip(string format)
    {
        var path = Path.Combine(root, "anim." + format);
        ImageIO.SaveAnimation(ThreeFrames(24, 16), path, format, 100);
        Assert.Equal(3, ImageIO.FrameCount(path));
        var back = ImageIO.LoadAnimation(path);
        Assert.Equal(3, back.Frames.Count);
        Assert.Equal([100, 200, 300], back.DelaysMs);
        Assert.All(back.Frames, f => Assert.Equal((24, 16), (f.Width, f.Height)));
        Assert.True(Math.Abs(back.Frames[2].Pixels[0] - 200) < 12, $"third frame red {back.Frames[2].Pixels[0]}");
    }

    [Fact]
    public async Task Animated_gifs_are_upscaled_frame_by_frame()
    {
        var opts = AppOptions.Parse(["--config", Path.Combine(root, "cfg")]);
        Directory.CreateDirectory(opts.ConfigDir);
        var store = new ModelStore(opts, NullLogger<ModelStore>.Instance);
        TestModels.Write(store.Dir, "realesr-general-x4v3", TestModels.NearestResize(2));
        using var engine = new Engine(NullLogger<Engine>.Instance);
        using var jobs = new Jobs(engine, store, opts, NullLogger<Jobs>.Instance);
        var input = Path.Combine(root, "in.gif");
        ImageIO.SaveAnimation(ThreeFrames(24, 16), input, "gif", 100);

        var job = jobs.Start(new JobRequest([new JobItemRequest(input)], new UpscaleSettings { Device = "cpu", Scale = 2, OutputDir = Path.Combine(root, "out") }));
        for (var i = 0; i < 200 && jobs.Get(job.Id)!.Status is "queued" or "running"; i++) await Task.Delay(50);
        var item = jobs.Get(job.Id)!.Items[0];
        Assert.Equal("done", item.Status);
        Assert.EndsWith(".gif", item.Output);
        Assert.Equal(3, item.Frames);
        Assert.Equal((48, 32), (item.OutWidth, item.OutHeight));
        var result = ImageIO.LoadAnimation(item.Output!);
        Assert.Equal(3, result.Frames.Count);
        Assert.Equal([100, 200, 300], result.DelaysMs);
    }
}
