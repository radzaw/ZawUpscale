using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using ZawUpscale.Tests.Infrastructure;

namespace ZawUpscale.Tests.Unit;

public sealed class SupirTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "zawupscale-tests", Guid.NewGuid().ToString("n"));

    public SupirTests() => Directory.CreateDirectory(root);

    public void Dispose()
    {
        try { Directory.Delete(root, recursive: true); } catch (IOException) { /* best effort */ }
    }

    static SupirOptions Options(int steps = 4, double restoration = 0, string colorFix = "wavelet") =>
        SupirPackage.BuiltInDefaults with { Steps = steps, Restoration = restoration, ColorFix = colorFix, Seed = 7 };

    static void AssertClose(RgbaImage expected, RgbaImage actual)
    {
        Assert.Equal((expected.Width, expected.Height), (actual.Width, actual.Height));
        for (var i = 0; i < expected.Pixels.Length; i++)
            if (Math.Abs(expected.Pixels[i] - actual.Pixels[i]) > 1)
                Assert.Fail($"Byte {i} (pixel {i / 4 % expected.Width}, {i / 4 / expected.Width}) is {actual.Pixels[i]}, expected {expected.Pixels[i]}.");
    }

    // ---------------------------------------------------------------- pipeline

    [Theory]
    [InlineData(96, 80, 1024, 0, "wavelet")]   // one tile
    [InlineData(328, 456, 128, 0, "wavelet")]  // many overlapping tiles, sides not multiples of 64
    [InlineData(256, 192, 256, 4, "adain")]    // restoration guidance on
    [InlineData(200, 136, 192, 0, "none")]
    public void Sampling_converges_on_the_control_latent_through_tiles_and_blending(int w, int h, int tile, double restoration, string colorFix)
    {
        var net = new FakeSupirNetworks();
        var src = FakeSupirNetworks.BlockImage(w, h);
        var result = new SupirUpscaler(net, minSize: 64).Upscale(src, 1, Options(restoration: restoration, colorFix: colorFix), tile);
        AssertClose(src, result);
    }

    [Fact]
    public void All_unet_tiles_have_one_shape_and_carry_both_prompts()
    {
        var net = new FakeSupirNetworks();
        new SupirUpscaler(net, minSize: 64).Upscale(FakeSupirNetworks.BlockImage(328, 456), 1, Options(steps: 3), 128);
        // 456 → 512 px → 64 latents; 328 → 384 → 48; 16-latent tiles every 8.
        var perStep = SupirUpscaler.Windows(64, 16, 8).Count * SupirUpscaler.Windows(48, 16, 8).Count;
        Assert.Equal(3 * perStep, net.UnetShapes.Count);
        Assert.All(net.UnetShapes, s => Assert.Equal((2, 16, 16), s));
        Assert.All(net.EncodeShapes, s => Assert.Equal((512, 384), s));
        Assert.Equal(2, net.Prompts.Count);
        Assert.StartsWith("Cinematic", net.Prompts[0]);
        Assert.StartsWith("painting", net.Prompts[1]);
    }

    [Fact]
    public void Small_results_are_made_at_the_minimum_size_and_reduced()
    {
        var net = new FakeSupirNetworks();
        var src = FakeSupirNetworks.BlockImage(50, 30, alpha: true);
        var result = new SupirUpscaler(net, minSize: 256).Upscale(src, 2, Options(steps: 2), 1024);
        Assert.Equal((100, 60), (result.Width, result.Height));
        Assert.True(result.HasAlpha());
        // 100 × 60 is worked on as 427 × 256, padded to 448 × 256.
        Assert.Contains((256, 448), net.EncodeShapes);
    }

    [Fact]
    public void Progress_rises_to_one_and_cancellation_stops_sampling()
    {
        var reports = new List<double>();
        new SupirUpscaler(new FakeSupirNetworks(), 64).Upscale(FakeSupirNetworks.BlockImage(128, 128), 1, Options(), 512, new SyncProgress(reports.Add));
        Assert.Equal(reports.OrderBy(x => x), reports);
        Assert.Equal(1.0, reports[^1], 6);

        using var cts = new CancellationTokenSource();
        var net = new FakeSupirNetworks { OnUnet = cts.Cancel };
        Assert.ThrowsAny<OperationCanceledException>(() => new SupirUpscaler(net, 64).Upscale(FakeSupirNetworks.BlockImage(128, 128), 1, Options(), 512, null, cts.Token));
        Assert.Single(net.UnetShapes);
    }

    [Fact]
    public void The_seed_used_is_reported()
    {
        var a = new SupirUpscaler(new FakeSupirNetworks(), 64);
        a.Upscale(FakeSupirNetworks.BlockImage(64, 64), 1, Options() with { Seed = 123 }, 512);
        Assert.Equal(123, a.LastSeed);
        var b = new SupirUpscaler(new FakeSupirNetworks(), 64);
        b.Upscale(FakeSupirNetworks.BlockImage(64, 64), 1, Options() with { Seed = null }, 512);
        Assert.True(b.LastSeed >= 0);
    }

    [Fact]
    public void Too_large_results_are_refused()
    {
        var ex = Assert.Throws<ApiException>(() => new SupirUpscaler(new FakeSupirNetworks()).Upscale(FakeSupirNetworks.BlockImage(4000, 3000), 4, Options(), 1024));
        Assert.Contains("megapixels", ex.Message);
    }

    // ---------------------------------------------------------------- schedule and tiling

    [Fact]
    public void Schedule_matches_sgm_legacy_ddpm_discretization()
    {
        var s = SupirUpscaler.Sigmas(50);
        Assert.Equal(51, s.Length);
        Assert.Equal(SupirUpscaler.SigmaMax, s[0], 3);
        Assert.Equal(0, s[^1]);
        for (var i = 1; i < s.Length; i++) Assert.True(s[i] < s[i - 1]);
        // np.linspace(999, 0, 50, endpoint=False).astype(int): the smallest level is t = 19.
        Assert.Equal(SupirUpscaler.TrainingSigmas[19], s[49], 9);
        Assert.Equal(0.0292, SupirUpscaler.TrainingSigmas[0], 4);
        foreach (var t in new[] { 0, 1, 250, 998, 999 }) Assert.Equal(t, SupirUpscaler.Timestep(SupirUpscaler.TrainingSigmas[t]));
    }

    [Theory]
    [InlineData(10, 4, 2, new[] { 0, 2, 4, 6 })]
    [InlineData(11, 4, 2, new[] { 0, 2, 4, 6, 7 })]
    [InlineData(3, 4, 2, new[] { 0 })]
    [InlineData(4, 4, 2, new[] { 0 })]
    public void Windows_are_full_size_and_reach_the_end(int n, int tile, int stride, int[] expected) =>
        Assert.Equal(expected, SupirUpscaler.Windows(n, tile, stride));

    [Fact]
    public void Tiled_vae_passes_blend_back_to_the_exact_input()
    {
        var src = Enumerable.Range(0, 2 * 37 * 53).Select(i => (float)Math.Sin(i * 0.37)).ToArray();
        var r = SupirUpscaler.Tiled(src, 2, 37, 53, 2, 1, 1, 16, 5, (t, h, w) => t, null, default);
        for (var i = 0; i < src.Length; i++) Assert.True(Math.Abs(src[i] - r[i]) < 1e-5, $"value {i}: {r[i]} instead of {src[i]}");
    }

    [Fact]
    public void Gaussian_weights_peak_in_the_middle_and_stay_positive()
    {
        var w = SupirUpscaler.GaussianWeights(16, 16);
        Assert.True(w.All(v => v > 0));
        Assert.Equal(w.Max(), w[7 * 16 + 7]);
        Assert.True(w[0] < w[7 * 16 + 7] * 1e-4);
    }

    [Fact]
    public void Size_conditioning_uses_cosines_then_sines()
    {
        var e = SupirUpscaler.TimestepEmbedding(0, 256);
        Assert.All(e[..128], v => Assert.Equal(1f, v));
        Assert.All(e[128..], v => Assert.Equal(0f, v));
        Assert.Equal((float)Math.Cos(1024), SupirUpscaler.TimestepEmbedding(1024, 256)[0], 5);
    }

    // ---------------------------------------------------------------- color fix

    [Fact]
    public void Wavelet_color_fix_keeps_detail_and_takes_the_original_colors()
    {
        int w = 64, h = 48, plane = w * h;
        var style = new float[3 * plane];
        var content = new float[3 * plane];
        for (var i = 0; i < style.Length; i++)
        {
            style[i] = (float)Math.Sin(i * 0.001) * 0.5f;
            content[i] = style[i] + 0.3f + ((i % w + i / w) % 2 == 0 ? 0.1f : -0.1f); // a color shift plus fine detail
        }
        var r = SupirUpscaler.WaveletColorFix(content, style, h, w);
        for (var i = 0; i < r.Length; i++) Assert.True(Math.Abs(content[i] - 0.3f - r[i]) < 0.005, $"value {i}: {r[i]} instead of {content[i] - 0.3f}");
        var same = SupirUpscaler.WaveletColorFix(content, content, h, w);
        for (var i = 0; i < r.Length; i++) Assert.True(Math.Abs(content[i] - same[i]) < 1e-4);
    }

    [Fact]
    public void AdaIn_color_fix_matches_mean_and_spread()
    {
        var content = Enumerable.Range(0, 300).Select(i => (float)(i % 7) * 2 + 5).ToArray();
        var style = Enumerable.Range(0, 300).Select(i => (float)(i % 5) * 0.1f).ToArray();
        var r = SupirUpscaler.AdaInColorFix(content, style, 100);
        for (var c = 0; c < 3; c++)
        {
            var seg = r.Skip(c * 100).Take(100).ToArray();
            var ref_ = style.Skip(c * 100).Take(100).ToArray();
            Assert.Equal(ref_.Average(), seg.Average(), 3);
        }
    }

    // ---------------------------------------------------------------- tokenizer

    static ClipTokenizer Tokenizer() => new(new Dictionary<string, int>
    {
        ["<|startoftext|>"] = 49406, ["<|endoftext|>"] = 49407,
        ["a</w>"] = 320, ["c"] = 10, ["a"] = 11, ["t</w>"] = 12, ["ca"] = 13, ["cat</w>"] = 2368, ["!</w>"] = 256,
    }, ["#version: 0.2", "c a", "ca t</w>"]);

    [Fact]
    public void Tokenizer_applies_bpe_merges_and_frames_the_text()
    {
        var ids = Tokenizer().Encode("A   CAT!", ClipTokenizer.EndOfText);
        Assert.Equal(77, ids.Length);
        Assert.Equal(new long[] { 49406, 320, 2368, 256, 49407, 49407 }, ids[..6]);
        Assert.Equal(new long[] { 49406, 320, 2368, 256, 49407, 0 }, Tokenizer().Encode("a cat!", 0)[..6]);
    }

    [Fact]
    public void Tokenizer_cuts_long_text_and_keeps_the_end_token()
    {
        var ids = Tokenizer().Encode(string.Join(" ", Enumerable.Repeat("cat", 200)), 0);
        Assert.Equal(49406, ids[0]);
        Assert.All(ids[1..76], v => Assert.Equal(2368, v));
        Assert.Equal(49407, ids[76]);
    }

    // ---------------------------------------------------------------- model folder

    string Package(string id, bool complete, object? defaults = null)
    {
        var dir = Path.Combine(root, "cfg", "models", id);
        Directory.CreateDirectory(Path.Combine(dir, "tokenizer"));
        File.WriteAllText(Path.Combine(dir, SupirPackage.ManifestName), JsonSerializer.Serialize(new
        {
            format = SupirPackage.FormatName, version = 1, name = "SUPIR test", description = "A test package.", minSize = 512, defaults,
        }));
        if (complete)
        {
            foreach (var f in SupirPackage.Files) File.WriteAllText(Path.Combine(dir, f), "");
            File.WriteAllText(Path.Combine(dir, SupirPackage.Vocab), "{}");
        }
        return dir;
    }

    [Fact]
    public void Settings_fall_back_to_the_package_then_to_supir()
    {
        var p = SupirPackage.Open(Package("supir-test", false, new { guidance = 4.0, guidanceStart = 1.0, steps = 30 }), checkFiles: false);
        Assert.Equal(512, p.MinSize);
        var o = p.Resolve(new SupirSettings { Prompt = "  a fox. ", Steps = 12 });
        Assert.Equal(12, o.Steps);
        Assert.Equal(4.0, o.Guidance);
        Assert.Equal(1.0, o.GuidanceStart);
        Assert.Equal("wavelet", o.ColorFix);
        Assert.StartsWith("a fox, Cinematic", o.PositivePrompt);
        Assert.Equal(30, p.Resolve(null).Steps);
        // A guidance below the default start doesn't ramp up to it.
        var low = SupirPackage.Open(Package("supir-q", false), checkFiles: false).Resolve(new SupirSettings { Guidance = 2 });
        Assert.Equal((2.0, 2.0), (low.Guidance, low.GuidanceStart));
    }

    [Theory]
    [InlineData(0, null)]
    [InlineData(201, null)]
    [InlineData(50, "sepia")]
    public void Bad_settings_are_refused(int steps, string? colorFix) =>
        Assert.Throws<ApiException>(() => SupirPackage.Validate(new SupirSettings { Steps = steps, ColorFix = colorFix }));

    [Fact]
    public void Incomplete_packages_say_what_is_missing()
    {
        var dir = Package("supir-test", false);
        var ex = Assert.Throws<ApiException>(() => SupirPackage.Open(dir));
        Assert.Contains("unet.onnx", ex.Message);
        File.WriteAllText(Path.Combine(dir, SupirPackage.ManifestName), """{ "format": "something-else", "version": 1 }""");
        Assert.Throws<ApiException>(() => SupirPackage.Open(dir, checkFiles: false));
    }

    [Fact]
    public void The_model_store_lists_runs_and_deletes_supir_folders()
    {
        var opts = AppOptions.Parse(["--config", Path.Combine(root, "cfg")]);
        var store = new ModelStore(opts, NullLogger<ModelStore>.Instance);
        var dir = Package("supir-test", false);
        var m = store.List().Single(x => x.Id == "supir-test");
        Assert.Equal(("supir", false, "SUPIR test"), (m.Kind, m.Installed, m.Name));
        Assert.Contains("unet.onnx", m.Error);

        Package("supir-test", true);
        m = store.List().Single(x => x.Id == "supir-test");
        Assert.True(m.Installed);
        Assert.Null(m.Error);
        Assert.Equal(50, m.Supir!.Steps);
        Assert.True(store.IsInstalled("supir-test"));
        Assert.Equal(dir, store.PathOf("supir-test"));

        // The CPU is refused before any network is loaded.
        using var engine = new Engine(NullLogger<Engine>.Instance);
        var ex = Assert.Throws<ApiException>(() => engine.Upscale(FakeSupirNetworks.BlockImage(16, 16), dir, "cpu", 2, 0, null, null, default));
        Assert.Contains("GPU", ex.Message);

        store.Delete("supir-test");
        Assert.False(Directory.Exists(dir));
    }

    sealed class SyncProgress(Action<double> a) : IProgress<double>
    {
        public void Report(double v) => a(v);
    }
}
