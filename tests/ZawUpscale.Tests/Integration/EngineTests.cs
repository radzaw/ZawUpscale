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
    public void Gpu_backend_upscales_exactly_when_there_is_a_gpu()
    {
        // DirectML on Windows, Core ML on macOS, CUDA in the Linux CUDA build; skipped where the machine has no GPU.
        if (!Engine.HasGpu || Gpu.Adapters().Count == 0) return;
        var device = $"gpu:{Gpu.Adapters()[0].Index}";
        var path = TestModels.Write(dir, "nn2", TestModels.NearestResize(2));
        var model = engine.Load(path, device);
        Assert.EndsWith($"({Engine.GpuBackend})", model.Device);

        var src = TestImages.Pattern(45, 29);
        var result = engine.Upscale(src, path, device, 2, 32, null, null, default);
        Assert.Equal(TestImages.Nearest(src, 2).Pixels, result.Pixels);
    }

    [Fact]
    public void Missing_cuda_libraries_are_explained()
    {
        // As ONNX Runtime's CUDA build reports it when cuBLAS isn't installed.
        var raw = new Exception("[ErrorCode:Fail] provider_bridge_ort.cc:1952 [ONNXRuntimeError] : 1 : FAIL : Failed to load library /root/.net/z/x/libonnxruntime_providers_cuda.so with error: libcublasLt.so.12: cannot open shared object file: No such file or directory");
        Assert.StartsWith("CUDA 12 and cuDNN 9 are needed, but libcublasLt.so.12 wasn't found.", Engine.GpuError(raw));
        Assert.Equal("Something else", Engine.GpuError(new Exception("Something else")));
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
    public void Supir_networks_run_in_onnx_runtime_with_fixed_tile_shapes()
    {
        // On the CPU here (the app insists on a GPU), but through the same sessions, names and dimension overrides.
        var path = Path.Combine(dir, "supir-test");
        TestModels.SupirPackage(path);
        var package = SupirPackage.Open(path);
        using var model = new SupirModel(package, () => new Microsoft.ML.OnnxRuntime.SessionOptions(), "CPU");

        // Each token's ViT-L features, then its bigG features; after the end token ViT-L pads with the end token, bigG with 0.
        var text = model.EncodeText("a");
        Assert.Equal([49406f, 320, 49407, 49407], Enumerable.Range(0, 4).Select(t => text.Context[t * 2048]));
        Assert.Equal([49406f, 320, 49407, 0], Enumerable.Range(0, 4).Select(t => text.Context[t * 2048 + 768]));
        Assert.All(text.Pooled, v => Assert.Equal(49407f, v));

        var src = FakeSupirNetworks.BlockImage(64, 48);
        var image = SupirUpscaler.ToTensor(src, 64, 48);
        var back = model.Decode(model.Encode(image, 48, 64), 6, 8);
        for (var i = 0; i < image.Length; i++) Assert.True(Math.Abs(image[i] - back[i]) < 1e-5);

        // The whole pipeline, with latent tiles smaller than the image.
        var result = new SupirUpscaler(model, package.MinSize).Upscale(FakeSupirNetworks.BlockImage(200, 136), 1, package.Resolve(new SupirSettings { Seed = 1 }), 128);
        Assert.Equal((200, 136), (result.Width, result.Height));
    }

    [Fact]
    public void Device_list_always_offers_auto_and_cpu()
    {
        var d = Engine.Devices();
        Assert.Equal("auto", d[0].Id);
        Assert.Equal("cpu", d[^1].Id);
    }
}
