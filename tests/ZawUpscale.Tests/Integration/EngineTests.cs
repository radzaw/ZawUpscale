using Microsoft.Extensions.Logging.Abstractions;
using ZawUpscale.Tests.Infrastructure;

namespace ZawUpscale.Tests.Integration;

/// <summary>The real ONNX Runtime path, with small generated models whose output is known exactly.</summary>
public sealed class EngineTests : IDisposable
{
    readonly string dir = Path.Combine(Path.GetTempPath(), "zawupscale-tests", Guid.NewGuid().ToString("n"));
    readonly Engine engine = new(NullLogger<Engine>.Instance);

    public void Dispose()
    {
        engine.Dispose();
        try { Directory.Delete(dir, recursive: true); } catch (IOException) { /* best effort */ }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Dynamic_model_upscales_exactly(bool float16)
    {
        var path = TestModels.Write(dir, float16 ? "nn2-fp16" : "nn2", TestModels.NearestResize(2, float16: float16));
        var model = engine.Load(path, "cpu");
        Assert.Equal(2, model.Scale);
        Assert.Null(model.FixedSize);
        Assert.Equal("CPU", model.Device);

        var src = TestImages.Pattern(45, 29);
        var result = engine.Upscale(src, path, "cpu", 2, 32, null, null, default);
        Assert.Equal(TestImages.Nearest(src, 2).Pixels, result.Pixels);
    }

    [Fact]
    public void Fixed_size_model_is_detected_and_tiled()
    {
        var path = TestModels.Write(dir, "nn4-fixed", TestModels.NearestResize(4, fixedSize: 48));
        var model = engine.Load(path, "cpu");
        Assert.Equal(4, model.Scale);
        Assert.Equal((48, 48), model.FixedSize);

        var src = TestImages.Pattern(50, 33);
        var result = engine.Upscale(src, path, "cpu", 4, 512, null, null, default);
        Assert.Equal(TestImages.Nearest(src, 4).Pixels, result.Pixels);
    }

    [Fact]
    public void Models_are_cached_until_the_file_or_device_changes()
    {
        var path = TestModels.Write(dir, "nn2", TestModels.NearestResize(2));
        var a = engine.Load(path, "cpu");
        Assert.Same(a, engine.Load(path, "cpu"));
        File.SetLastWriteTimeUtc(path, DateTime.UtcNow.AddMinutes(1));
        Assert.NotSame(a, engine.Load(path, "cpu"));
    }

    [Fact]
    public void Not_a_model_gives_an_error()
    {
        var path = Path.Combine(dir, "junk.onnx");
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(path, [1, 2, 3, 4]);
        Assert.ThrowsAny<Exception>(() => engine.Load(path, "cpu"));
        Assert.Throws<ApiException>(() => engine.Load(Path.Combine(dir, "missing.onnx"), "cpu"));
    }

    [Fact]
    public void Device_list_always_offers_auto_and_cpu()
    {
        var d = Engine.Devices();
        Assert.Equal("auto", d[0].Id);
        Assert.Equal("cpu", d[^1].Id);
    }
}
