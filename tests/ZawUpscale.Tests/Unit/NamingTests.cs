using Microsoft.Extensions.Logging.Abstractions;
using ZawUpscale.Tests.Infrastructure;

namespace ZawUpscale.Tests.Unit;

public sealed class NamingTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "zawupscale-tests", Guid.NewGuid().ToString("n"));
    readonly Jobs jobs;
    readonly Engine engine = new(NullLogger<Engine>.Instance);

    public NamingTests()
    {
        Directory.CreateDirectory(root);
        var opts = AppOptions.Parse(["--config", Path.Combine(root, "cfg")]);
        Directory.CreateDirectory(opts.ConfigDir);
        jobs = new Jobs(engine, new ModelStore(opts, NullLogger<ModelStore>.Instance), opts, NullLogger<Jobs>.Instance);
    }

    public void Dispose()
    {
        jobs.Dispose();
        engine.Dispose();
        try { Directory.Delete(root, recursive: true); } catch (IOException) { /* best effort */ }
    }

    string Out(string input, UpscaleSettings s, string? rel = null) => jobs.OutputPath(new JobItemRequest(input, rel), Jobs.Validate(s));

    [Fact]
    public void Results_go_in_a_model_and_scale_subfolder_by_default()
    {
        var input = Path.Combine(root, "pics", "cat.jpeg");
        var sub = Path.Combine(root, "pics", "realesr-general-x4v3_x4");
        Assert.Equal(Path.Combine(sub, "cat.jpg"), Out(input, new UpscaleSettings()));
        Assert.Equal(Path.Combine(root, "pics", "anime_x2.5", "cat.png"), Out(input, new UpscaleSettings { Model = "anime", Scale = 2.5, Format = "png" }));
        Assert.Equal(Path.Combine(sub, "cat.png"), Out(Path.Combine(root, "pics", "cat.bmp"), new UpscaleSettings()));
        Assert.Equal(Path.Combine(sub, "cat.webp"), Out(Path.Combine(root, "pics", "cat.webp"), new UpscaleSettings(), "cat.webp"));
    }

    [Fact]
    public void A_recursive_batch_gets_one_subfolder_in_the_opened_folder()
    {
        var input = Path.Combine(root, "pics", "2024", "may", "dog.png");
        Assert.Equal(Path.Combine(root, "pics", "realesr-general-x4v3_x4", "2024", "may", "dog.png"),
            Out(input, new UpscaleSettings(), Path.Combine("2024", "may", "dog.png")));
    }

    [Fact]
    public void Pattern_tokens_and_output_folder_with_structure()
    {
        var input = Path.Combine(root, "pics", "2024", "dog.png");
        var s = new UpscaleSettings { OutputDir = Path.Combine(root, "out"), NameTemplate = "{model}-{NAME}@{scale}x", Model = "anime", Format = "jpeg" };
        Assert.Equal(Path.Combine(root, "out", "2024", "anime-dog@4x.jpg"), Out(input, s, Path.Combine("2024", "dog.png")));
    }

    [Fact]
    public void Invalid_settings_are_refused()
    {
        Assert.Throws<ApiException>(() => Jobs.Validate(new UpscaleSettings { Scale = 0.5 }));
        Assert.Throws<ApiException>(() => Jobs.Validate(new UpscaleSettings { Format = "tiff" }));
        Assert.Throws<ApiException>(() => Jobs.Validate(new UpscaleSettings { NameTemplate = "a/b" }));
        Assert.Throws<ApiException>(() => Jobs.Validate(new UpscaleSettings { Existing = "ask" }));
    }

    [Theory]
    [InlineData("{name}_x{scale}", "photo_x4", true)]
    [InlineData("{name}_x{scale}", "photo_x2.5", true)]
    [InlineData("{name}_x{scale}", "photo_xl", false)]
    [InlineData("{name}_x{scale}", "photo", false)]
    [InlineData("up-{name}-{model}", "up-photo-realesrgan-x4plus", true)]
    public void Earlier_results_are_recognised(string template, string stem, bool match)
    {
        var re = Files.PreviousResults(template)!;
        var m = re.Match(stem);
        Assert.Equal(match, m.Success);
        if (match) Assert.Equal("photo", m.Groups["name"].Value);
        Assert.Null(Files.PreviousResults("{name}"));
        Assert.Null(Files.PreviousResults("fixed"));
    }

    [Fact]
    public void Folder_scans_leave_out_earlier_results_but_not_picked_files()
    {
        var dir = Path.Combine(root, "batch");
        Directory.CreateDirectory(Path.Combine(dir, "sub"));
        foreach (var f in new[] { "a.png", "a_x4.png", "b.jpg", "b_x2.webp", "lonely_x4.png", "notes.txt", Path.Combine("sub", "c.png") })
            File.WriteAllBytes(Path.Combine(dir, f), [1, 2, 3]);

        var flat = Files.Scan([dir], recursive: false, "{name}_x{scale}");
        Assert.Equal(["a.png", "b.jpg", "lonely_x4.png"], flat.Items.Select(i => i.Rel));
        Assert.Equal(2, flat.PreviousResults);

        var deep = Files.Scan([dir], recursive: true, "{name}_x{scale}");
        Assert.Contains(Path.Combine("sub", "c.png"), deep.Items.Select(i => i.Rel));

        // Files chosen one by one are taken as they are.
        var picked = Files.Scan([Path.Combine(dir, "a.png"), Path.Combine(dir, "a_x4.png"), Path.Combine(dir, "notes.txt")], false, "{name}_x{scale}");
        Assert.Equal(2, picked.Items.Count);
        Assert.Equal(1, picked.Skipped);
    }

    [Fact]
    public void Recursive_scans_leave_out_results_folders()
    {
        var dir = Path.Combine(root, "tree");
        foreach (var f in new[] { "a.png", Path.Combine("model-1_x4", "a_x4.png"), Path.Combine("sub", "model-1_x2.5", "b_x2.5.png"), Path.Combine("sub", "b.png"), Path.Combine("other_x4", "c.png"), Path.Combine("model-1_x4_faces", "a.png") })
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(dir, f))!);
            File.WriteAllBytes(Path.Combine(dir, f), [1, 2, 3]);
        }
        var scan = Files.Scan([dir], recursive: true, "{name}_x{scale}", ["model-1"]);
        Assert.Equal(["a.png", Path.Combine("other_x4", "c.png"), Path.Combine("sub", "b.png")], scan.Items.Select(i => i.Rel));
        Assert.Equal(3, scan.PreviousResults);
    }
}
