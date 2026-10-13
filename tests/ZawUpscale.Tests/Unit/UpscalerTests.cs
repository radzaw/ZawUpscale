using ZawUpscale.Tests.Infrastructure;

namespace ZawUpscale.Tests.Unit;

public class UpscalerTests
{
    [Theory]
    [InlineData(37, 23, 8, 2)]
    [InlineData(37, 23, 16, 4)]
    [InlineData(64, 64, 32, 2)]
    [InlineData(5, 3, 32, 4)]
    [InlineData(100, 41, 33, 3)]
    public void Tiles_are_stitched_without_seams(int w, int h, int tile, int scale)
    {
        // Any misplaced or misaligned tile would break the exact equality with the reference.
        var src = TestImages.Pattern(w, h);
        var model = new FakeModel(scale);
        var result = new Upscaler(model).RunPass(src, tile, null, default);
        Assert.Equal(TestImages.Nearest(src, scale).Pixels, result.Pixels);
    }

    [Fact]
    public void Tile_inputs_carry_context_and_are_padded_to_a_multiple()
    {
        var model = new FakeModel(2);
        new Upscaler(model).RunPass(TestImages.Pattern(70, 30), 32, null, default);
        Assert.Equal(Upscaler.TileCount(70, 30, 32), model.Calls);
        Assert.All(model.Sizes, s => { Assert.Equal(0, s.H % Upscaler.Multiple); Assert.Equal(0, s.W % Upscaler.Multiple); });
        // The first tile sees 16 px of context to the right: 32 + 16 = 48 wide.
        Assert.Equal((32, 48), model.Sizes[0]);
    }

    [Fact]
    public void Fixed_size_models_get_their_exact_input_size()
    {
        var model = new FakeModel(4, (64, 64));
        var up = new Upscaler(model);
        Assert.Equal(64 - 2 * Upscaler.Pad, up.EffectiveTile(512));
        var src = TestImages.Pattern(50, 21);
        var result = up.RunPass(src, up.EffectiveTile(512), null, default);
        Assert.Equal(TestImages.Nearest(src, 4).Pixels, result.Pixels);
    }

    [Fact]
    public void Plan_reduces_before_the_last_pass_instead_of_overshooting()
    {
        var up = new Upscaler(new FakeModel(4));
        Assert.Equal([(100, 50)], up.Plan(100, 50, 4));
        Assert.Equal([(100, 50)], up.Plan(100, 50, 2));
        Assert.Equal([(100, 50), (400, 200)], up.Plan(100, 50, 16));
        // 6×: one 4× pass, shrink to 600/4 × 300/4, then the second pass lands on 600 × 300.
        Assert.Equal([(100, 50), (150, 75)], up.Plan(100, 50, 6));
        Assert.Equal([(10, 10)], new Upscaler(new FakeModel(1)).Plan(10, 10, 4));
    }

    [Theory]
    [InlineData(2, 60, 40)]
    [InlineData(4, 120, 80)]
    [InlineData(6, 180, 120)]
    [InlineData(2.5, 75, 50)]
    public void Upscale_produces_the_requested_size(double scale, int w, int h)
    {
        var result = new Upscaler(new FakeModel(4)).Upscale(TestImages.Pattern(30, 20), scale, 64);
        Assert.Equal((w, h), (result.Width, result.Height));
    }

    [Fact]
    public void Alpha_is_resized_along()
    {
        var src = TestImages.Pattern(40, 10, alpha: true);
        var result = new Upscaler(new FakeModel(4)).Upscale(src, 4, 64);
        Assert.True(result.HasAlpha());
        int Alpha(int x, int y) => result.Pixels[(y * result.Width + x) * 4 + 3];
        Assert.True(Alpha(0, 20) < 10);
        Assert.True(Alpha(result.Width - 1, 20) > 245);
        Assert.InRange(Alpha(result.Width / 2, 20), 100, 155);
    }

    [Fact]
    public void Progress_reaches_one_and_cancellation_stops()
    {
        var reports = new List<double>();
        new Upscaler(new FakeModel(2)).Upscale(TestImages.Pattern(100, 100), 4, 32, new SyncProgress(reports.Add));
        Assert.Equal(1.0, reports[^1], 6);
        Assert.True(reports.SequenceEqual(reports.Order()));

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => new Upscaler(new FakeModel(2)).Upscale(TestImages.Pattern(10, 10), 2, 32, null, cts.Token));
    }

    [Fact]
    public void Huge_results_are_refused_up_front()
    {
        var ex = Assert.Throws<ApiException>(() => new Upscaler(new FakeModel(4)).Upscale(TestImages.Pattern(1000, 1000), 32, 64));
        Assert.Contains("megapixels", ex.Message);
    }

    sealed class SyncProgress(Action<double> a) : IProgress<double>
    {
        public void Report(double v) => a(v);
    }
}
