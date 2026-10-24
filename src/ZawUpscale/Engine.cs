using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SessionOptions = Microsoft.ML.OnnxRuntime.SessionOptions;

namespace ZawUpscale;

/// <summary>A model the engine keeps loaded: an upscaling network or a SUPIR package.</summary>
public interface IEngineModel : IDisposable
{
    /// <summary>Where it runs, e.g. "NVIDIA GeForce RTX 4070 (DirectML)" or "CPU".</summary>
    string Device { get; }
    /// <summary>Makes a running inference stop when <paramref name="ct"/> is cancelled.</summary>
    IDisposable BindCancellation(CancellationToken ct);
}

/// <summary>An ONNX super-resolution network (Real-ESRGAN and similar): NCHW RGB in [0, 1], float32 or float16.</summary>
public sealed class OnnxSrModel : ISrModel, IEngineModel
{
    readonly InferenceSession session;
    readonly string inputName, outputName;
    readonly bool half;
    RunOptions runOptions = new();

    public int Scale { get; }
    public (int Width, int Height)? FixedSize { get; }
    public string Device { get; }

    public OnnxSrModel(string path, SessionOptions options, string device)
    {
        Device = device;
        session = new InferenceSession(path, options);
        try
        {
            if (session.InputMetadata.Count < 1 || session.OutputMetadata.Count < 1) throw new ApiException("The model has no inputs or outputs.");
            var (inName, inMeta) = session.InputMetadata.First();
            inputName = inName;
            outputName = session.OutputMetadata.Keys.First();
            if (!inMeta.IsTensor || inMeta.Dimensions.Length != 4)
                throw new ApiException("This model doesn't look like an image upscaler: it should take one image tensor [1, 3, height, width].");
            if (inMeta.Dimensions[1] is not (3 or -1))
                throw new ApiException($"The model expects {inMeta.Dimensions[1]} color channels; only RGB (3) models are supported.");
            half = inMeta.ElementDataType == TensorElementType.Float16;
            if (!half && inMeta.ElementDataType != TensorElementType.Float)
                throw new ApiException($"The model's input is {inMeta.ElementDataType}; only float32 and float16 models are supported.");
            int dh = inMeta.Dimensions[2], dw = inMeta.Dimensions[3];
            if (dh > 0 && dw > 0) FixedSize = (dw, dh);

            // Find the scale by running a small blank tile; this also warms the device up.
            var (pw, ph) = FixedSize ?? (32, 32);
            RunRaw(new float[3 * pw * ph], ph, pw, out var shape);
            if (shape.Length != 4 || shape[2] % ph != 0 || shape[3] % pw != 0 || shape[2] / ph != shape[3] / pw || shape[1] != 3)
                throw new ApiException($"Unexpected model output shape [{string.Join(", ", shape)}] for a {pw}×{ph} input.");
            Scale = (int)(shape[2] / ph);
            if (Scale is < 1 or > 16) throw new ApiException($"Unsupported model scale {Scale}×.");
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    public float[] Run(float[] chw, int h, int w)
    {
        try
        {
            return RunRaw(chw, h, w, out _);
        }
        catch (OnnxRuntimeException ex) when (runOptions.Terminate)
        {
            throw new OperationCanceledException("Cancelled.", ex);
        }
        catch (OnnxRuntimeException ex) when (IsOutOfMemory(ex))
        {
            throw new TileTooLargeException(ex.Message, ex);
        }
    }

    internal static bool IsOutOfMemory(Exception ex)
    {
        var m = ex.Message;
        return m.Contains("memory", StringComparison.OrdinalIgnoreCase) || m.Contains("alloc", StringComparison.OrdinalIgnoreCase)
            || m.Contains("887A0005", StringComparison.OrdinalIgnoreCase) || m.Contains("887A0006", StringComparison.OrdinalIgnoreCase)
            || m.Contains("8007000E", StringComparison.OrdinalIgnoreCase);
    }

    float[] RunRaw(float[] chw, int h, int w, out long[] shape)
    {
        long[] inShape = [1, 3, h, w];
        using var input = half
            ? OrtValue.CreateTensorValueFromMemory(ToHalf(chw), inShape)
            : OrtValue.CreateTensorValueFromMemory(chw, inShape);
        using var outputs = session.Run(runOptions, [inputName], [input], [outputName]);
        var o = outputs[0];
        shape = o.GetTensorTypeAndShape().Shape;
        if (half)
        {
            var src = o.GetTensorDataAsSpan<Float16>();
            var r = new float[src.Length];
            for (var i = 0; i < r.Length; i++) r[i] = (float)BitConverter.UInt16BitsToHalf(src[i].value);
            return r;
        }
        return o.GetTensorDataAsSpan<float>().ToArray();
    }

    static Float16[] ToHalf(float[] a)
    {
        var r = new Float16[a.Length];
        for (var i = 0; i < a.Length; i++) r[i] = new Float16(BitConverter.HalfToUInt16Bits((Half)a[i]));
        return r;
    }

    /// <summary>Makes a running inference stop when <paramref name="ct"/> is cancelled.</summary>
    public IDisposable BindCancellation(CancellationToken ct)
    {
        runOptions.Dispose();
        runOptions = new RunOptions();
        var ro = runOptions;
        return ct.Register(() =>
        {
            try { ro.Terminate = true; }
            catch (ObjectDisposedException) { /* the model was unloaded meanwhile; nothing is running */ }
        });
    }

    public void Dispose()
    {
        session.Dispose();
        runOptions.Dispose();
    }
}

public sealed record DeviceInfo(string Id, string Name, string? Detail);

/// <summary>
/// Owns the loaded network (one at a time: they can be large and live in GPU memory) and runs upscales.
/// Face restoration models are kept in a second slot beside it, so a batch doesn't reload either per image.
/// Device ids: "auto" (first GPU, falling back to the CPU), "gpu:N" (DirectML adapter N) and "cpu".
/// </summary>
public sealed class Engine(ILogger<Engine> log) : IDisposable
{
    readonly Lock gate = new();
    IEngineModel? loaded;
    string? loadedKey;
    FaceModels? faces;
    string? facesKey;
    string? lastGpuError;

#if DIRECTML
    public const bool HasGpu = true;
#else
    public const bool HasGpu = false;
#endif

    public static List<DeviceInfo> Devices()
    {
        var list = new List<DeviceInfo>();
        var gpus = HasGpu ? Gpu.Adapters() : [];
        list.Add(new DeviceInfo("auto", "Automatic", gpus.Count > 0 ? $"{gpus[0].Name}, or the CPU if the GPU fails" : "CPU"));
        foreach (var g in gpus)
            list.Add(new DeviceInfo($"gpu:{g.Index}", g.Name, g.MemoryBytes > 0 ? $"{g.MemoryBytes / (1024.0 * 1024 * 1024):0.#} GB" : null));
        list.Add(new DeviceInfo("cpu", "CPU", $"{Environment.ProcessorCount} threads"));
        return list;
    }

    /// <summary>Tile size used when the user leaves it on automatic.</summary>
    public static int AutoTile(string device) => device == "cpu" || !HasGpu || Gpu.Adapters().Count == 0 ? 256 : 512;

    /// <summary>The device of the loaded model, e.g. "NVIDIA GeForce RTX 4070 (DirectML)" or "CPU".</summary>
    public string? ActiveDevice { get { lock (gate) return loaded?.Device; } }
    public string? LastGpuError => lastGpuError;

    public OnnxSrModel Load(string modelPath, string device) =>
        (OnnxSrModel)Load(modelPath + "|" + device + "|" + File.GetLastWriteTimeUtc(modelPath).Ticks, () => Create(modelPath, device));

    /// <summary>Loads a SUPIR model folder; the networks themselves load as they are first used.</summary>
    public SupirModel LoadSupir(string dir, string device)
    {
        var manifest = Path.Combine(dir, SupirPackage.ManifestName);
        return (SupirModel)Load(dir + "|" + device + "|" + File.GetLastWriteTimeUtc(manifest).Ticks, () => CreateSupir(dir, device));
    }

    IEngineModel Load(string key, Func<IEngineModel> create)
    {
        lock (gate)
        {
            if (loaded != null && loadedKey == key) return loaded;
            Unload();
            loaded = create();
            loadedKey = key;
            return loaded;
        }
    }

    /// <summary>Loads a face restoration network with the face detector.</summary>
    public FaceModels LoadFaces(string restorerPath, string detectorPath, string device)
    {
        foreach (var p in new[] { restorerPath, detectorPath })
            if (!File.Exists(p)) throw new ApiException($"Model file not found: {p}");
        var key = restorerPath + "|" + device + "|" + File.GetLastWriteTimeUtc(restorerPath).Ticks + "|" + File.GetLastWriteTimeUtc(detectorPath).Ticks;
        lock (gate)
        {
            if (faces != null && facesKey == key) return faces;
            faces?.Dispose();
            faces = null;
            faces = OnDevice(device, (so, name) => new FaceModels(restorerPath, detectorPath, so, name));
            facesKey = key;
            return faces;
        }
    }

    /// <summary>Lets go of all models: the upscaler and the face models.</summary>
    public void Unload()
    {
        lock (gate)
        {
            loaded?.Dispose();
            loaded = null;
            loadedKey = null;
            faces?.Dispose();
            faces = null;
            facesKey = null;
        }
    }

    OnnxSrModel Create(string path, string device)
    {
        if (!File.Exists(path)) throw new ApiException($"Model file not found: {path}");
        return OnDevice(device, (so, name) => new OnnxSrModel(path, so, name));
    }

    /// <summary>Creates a model on the chosen GPU, or the CPU; "auto" falls back to the CPU when the GPU fails.</summary>
    T OnDevice<T>(string device, Func<SessionOptions, string, T> create)
    {
        if (device != "cpu" && HasGpu)
        {
            var gpus = Gpu.Adapters();
            var index = device.StartsWith("gpu:") && int.TryParse(device[4..], out var n) ? n : gpus.FirstOrDefault()?.Index ?? 0;
            var name = gpus.FirstOrDefault(g => g.Index == index)?.Name ?? $"GPU {index}";
            try
            {
                var so = GpuOptions(index);
                var m = create(so, $"{name} (DirectML)");
                lastGpuError = null;
                return m;
            }
            catch (Exception ex) when (ex is OnnxRuntimeException or EntryPointNotFoundException or DllNotFoundException)
            {
                lastGpuError = ex.Message;
                log.LogWarning("GPU {Device} unavailable, using the CPU: {Message}", name, ex.Message);
                if (device != "auto") throw new ApiException($"Could not use {name}: {ex.Message}");
            }
        }
        var cpu = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        return create(cpu, "CPU");
    }

    SupirModel CreateSupir(string dir, string device)
    {
        var package = SupirPackage.Open(dir);
        var gpus = HasGpu ? Gpu.Adapters() : [];
        // On the CPU a 50-step SDXL run takes hours, and float16 networks lack CPU kernels.
        if (device == "cpu" || gpus.Count == 0)
            throw new ApiException(HasGpu ? "SUPIR needs a GPU. Choose one as the device (in Preferences, or --device on the command line)." : "SUPIR needs a GPU, which this build can only use on Windows (DirectML).");
        var index = device.StartsWith("gpu:") && int.TryParse(device[4..], out var n) ? n : gpus[0].Index;
        var name = gpus.FirstOrDefault(g => g.Index == index)?.Name ?? $"GPU {index}";
        return new SupirModel(package, () => GpuOptions(index), $"{name} (DirectML)");
    }

    static SessionOptions GpuOptions(int deviceId)
    {
        // DirectML doesn't support memory patterns or parallel execution.
        var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            EnableMemoryPattern = false,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
        };
#if DIRECTML
        so.AppendExecutionProvider_DML(deviceId);
#endif
        return so;
    }

    /// <summary>
    /// Upscales with the given model. If a tile doesn't fit in GPU memory, the tile size is halved and the
    /// image started again (the device may need a fresh session after running out of memory).
    /// </summary>
    public RgbaImage Upscale(RgbaImage src, string modelPath, string device, double scale, int tile, IProgress<double>? progress, Action<string>? notice, CancellationToken ct,
        SupirSettings? supir = null)
    {
        if (SupirPackage.IsPackage(modelPath)) return Supir(src, modelPath, device, scale, supir, progress, notice, ct);
        if (tile <= 0) tile = AutoTile(device);
        while (true)
        {
            var model = Load(modelPath, device);
            using var reg = model.BindCancellation(ct);
            try
            {
                return new Upscaler(model).Upscale(src, scale, tile, progress, ct);
            }
            catch (TileTooLargeException ex) when (tile > 64)
            {
                tile /= 2;
                log.LogWarning("Out of memory ({Message}); retrying with {Tile}px tiles", ex.Message, tile);
                notice?.Invoke($"Out of memory; retrying with {tile}px tiles.");
                Unload();
            }
            catch (TileTooLargeException ex)
            {
                throw new ApiException($"Not enough memory even with small tiles: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Restores with SUPIR. Its tiles start at SDXL's native 1024 px whatever the tile preference (smaller ones
    /// lose quality); if they don't fit in GPU memory they are made smaller and the image started again.
    /// </summary>
    RgbaImage Supir(RgbaImage src, string dir, string device, double scale, SupirSettings? settings, IProgress<double>? progress, Action<string>? notice, CancellationToken ct)
    {
        var options = SupirPackage.Open(dir, checkFiles: false).Resolve(settings);
        var tile = SupirUpscaler.DefaultTile;
        while (true)
        {
            var model = LoadSupir(dir, device);
            using var reg = model.BindCancellation(ct);
            var upscaler = new SupirUpscaler(model, model.Package.MinSize);
            try
            {
                var result = upscaler.Upscale(src, scale, options, tile, progress, ct);
                log.LogInformation("SUPIR finished with seed {Seed}", upscaler.LastSeed);
                return result;
            }
            catch (TileTooLargeException ex) when (tile > SupirUpscaler.MinTile)
            {
                tile = Math.Max(SupirUpscaler.MinTile, tile * 3 / 4 / 64 * 64);
                log.LogWarning("Out of GPU memory ({Message}); retrying SUPIR with {Tile}px tiles", ex.Message, tile);
                notice?.Invoke($"Out of GPU memory; retrying with {tile}px tiles.");
                Unload();
            }
            catch (TileTooLargeException ex)
            {
                throw new ApiException($"Not enough GPU memory for SUPIR (it needs about 12 GB): {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Restores the faces of <paramref name="original"/> in its upscaled version <paramref name="upscaled"/> (changed in
    /// place), with <paramref name="strength"/> from 0 (unchanged) to 1. Returns the number of faces.
    /// </summary>
    public int RestoreFaces(RgbaImage original, RgbaImage upscaled, string restorerPath, string detectorPath, string device, double strength,
        IProgress<double>? progress, CancellationToken ct)
    {
        var models = LoadFaces(restorerPath, detectorPath, device);
        using var reg = models.BindCancellation(ct);
        return models.Enhance(original, upscaled, strength, progress, ct);
    }

    public void Dispose() => Unload();
}
