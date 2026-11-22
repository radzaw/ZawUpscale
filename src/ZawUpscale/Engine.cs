using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
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
    /// <summary>Static-shape sessions kept at once; each holds its own copy of the weights and buffers on the GPU.</summary>
    const int MaxShapes = 2;

    readonly Func<SessionOptions> options;
    readonly string path, inputName, outputName;
    readonly bool halfIn, halfOut;
    readonly string?[]? dimNames;
    /// <summary>The session for every input size, or null when each input size gets its own (see <see cref="StaticShapes"/>).</summary>
    readonly InferenceSession? session;
    readonly List<((int H, int W) Shape, InferenceSession Session)> shaped = [];
    RunOptions runOptions = new();

    public int Scale { get; }
    public (int Width, int Height)? FixedSize { get; }
    public string Device { get; }
    public bool UniformTiles { get; init; }
    /// <summary>
    /// True if each input size runs in a session made for it (its free dimensions fixed). DirectML plans a static graph
    /// far better: 35–40% faster for float16 Real-ESRGAN. A dynamic DirectML session also slows down for good after
    /// one run at a different size (the 32 px probe below makes later 512 px tiles 20–60% slower), so it is not kept.
    /// </summary>
    public bool StaticShapes { get; }

    public OnnxSrModel(string path, Func<SessionOptions> options, string device, bool staticShapes = false)
    {
        this.path = path;
        this.options = options;
        Device = device;
        var probe = new InferenceSession(path, options());
        try
        {
            if (probe.InputMetadata.Count < 1 || probe.OutputMetadata.Count < 1) throw new ApiException("The model has no inputs or outputs.");
            var (inName, inMeta) = probe.InputMetadata.First();
            inputName = inName;
            var (outName, outMeta) = probe.OutputMetadata.First();
            outputName = outName;
            if (!inMeta.IsTensor || inMeta.Dimensions.Length != 4)
                throw new ApiException("This model doesn't look like an image upscaler: it should take one image tensor [1, 3, height, width].");
            if (inMeta.Dimensions[1] is not (3 or -1))
                throw new ApiException($"The model expects {inMeta.Dimensions[1]} color channels; only RGB (3) models are supported.");
            halfIn = inMeta.ElementDataType == TensorElementType.Float16;
            if (!halfIn && inMeta.ElementDataType != TensorElementType.Float)
                throw new ApiException($"The model's input is {inMeta.ElementDataType}; only float32 and float16 models are supported.");
            halfOut = outMeta.IsTensor && outMeta.ElementDataType == TensorElementType.Float16;
            int dh = inMeta.Dimensions[2], dw = inMeta.Dimensions[3];
            if (dh > 0 && dw > 0) FixedSize = (dw, dh);

            // Find the scale by running a small blank tile; this also warms the device up.
            var (pw, ph) = FixedSize ?? (32, 32);
            var probeOut = Infer(probe, new float[3 * pw * ph], ph, pw, null, out var shape);
            if (shape.Length != 4 || shape[2] % ph != 0 || shape[3] % pw != 0 || shape[2] / ph != shape[3] / pw || shape[1] != 3)
                throw new ApiException($"Unexpected model output shape [{string.Join(", ", shape)}] for a {pw}×{ph} input.");
            Scale = (int)(shape[2] / ph);
            if (Scale is < 1 or > 16) throw new ApiException($"Unsupported model scale {Scale}×.");

            // Sizes can only be fixed through named free dimensions; a fixed-size network is static already.
            var names = inMeta.SymbolicDimensions;
            StaticShapes = staticShapes && FixedSize == null && names.Length == 4
                && (inMeta.Dimensions[2] > 0 || names[2] is { Length: > 0 }) && (inMeta.Dimensions[3] > 0 || names[3] is { Length: > 0 });
            if (StaticShapes)
            {
                dimNames = names;
                probe.Dispose();
            }
            else session = probe;
        }
        catch
        {
            probe.Dispose();
            throw;
        }
    }

    public float[] Run(float[] chw, int h, int w)
    {
        try
        {
            var output = GC.AllocateUninitializedArray<float>(3 * h * Scale * w * Scale);
            Infer(SessionFor(h, w), chw, h, w, output, out _);
            return output;
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

    InferenceSession SessionFor(int h, int w)
    {
        if (session != null) return session;
        var i = shaped.FindIndex(s => s.Shape == (h, w));
        if (i >= 0) return shaped[i].Session;
        // Make room first: the old session's GPU memory may be needed for the new one.
        while (shaped.Count >= MaxShapes)
        {
            shaped[0].Session.Dispose();
            shaped.RemoveAt(0);
        }
        var so = options();
        long[] values = [1, 3, h, w];
        for (var d = 0; d < 4; d++)
            if (dimNames![d] is { Length: > 0 } name) so.AddFreeDimensionOverrideByName(name, values[d]);
        var created = new InferenceSession(path, so);
        shaped.Add(((h, w), created));
        return created;
    }

    internal static bool IsOutOfMemory(Exception ex)
    {
        var m = ex.Message;
        return m.Contains("memory", StringComparison.OrdinalIgnoreCase) || m.Contains("alloc", StringComparison.OrdinalIgnoreCase)
            || m.Contains("887A0005", StringComparison.OrdinalIgnoreCase) || m.Contains("887A0006", StringComparison.OrdinalIgnoreCase)
            || m.Contains("8007000E", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Runs one input. With <paramref name="output"/> (sized for the model's scale), the result is written straight into it
    /// (and returned); without, the output's shape is unknown, so it is copied out of the result tensor.
    /// </summary>
    float[] Infer(InferenceSession s, float[] chw, int h, int w, float[]? output, out long[] shape)
    {
        long[] inShape = [1, 3, h, w];
        using var input = halfIn
            ? OrtValue.CreateTensorValueFromMemory(ToHalf(chw), inShape)
            : OrtValue.CreateTensorValueFromMemory(chw, inShape);
        if (output != null)
        {
            long[] outShape = [1, 3, h * Scale, w * Scale];
            shape = outShape;
            if (halfOut)
            {
                var buffer = new Float16[output.Length];
                using (var o = OrtValue.CreateTensorValueFromMemory(buffer, outShape))
                    s.Run(runOptions, [inputName], [input], [outputName], [o]);
                FromHalf(buffer, output);
            }
            else
            {
                using var o = OrtValue.CreateTensorValueFromMemory(output, outShape);
                s.Run(runOptions, [inputName], [input], [outputName], [o]);
            }
            return output;
        }
        using var outputs = s.Run(runOptions, [inputName], [input], [outputName]);
        var result = outputs[0];
        shape = result.GetTensorTypeAndShape().Shape;
        if (halfOut)
        {
            var src = result.GetTensorDataAsSpan<Float16>();
            var r = new float[src.Length];
            FromHalf(src, r);
            return r;
        }
        return result.GetTensorDataAsSpan<float>().ToArray();
    }

    // ORT's Float16 is the IEEE half's bits, so the vectorised conversions of System.Numerics.Tensors apply.
    static Float16[] ToHalf(float[] a)
    {
        var r = new Float16[a.Length];
        TensorPrimitives.ConvertToHalf(a, MemoryMarshal.Cast<Float16, Half>(r.AsSpan()));
        return r;
    }

    static void FromHalf(ReadOnlySpan<Float16> src, Span<float> dst) =>
        TensorPrimitives.ConvertToSingle(MemoryMarshal.Cast<Float16, Half>(src), dst);

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
        session?.Dispose();
        foreach (var (_, s) in shaped) s.Dispose();
        shaped.Clear();
        runOptions.Dispose();
    }
}

public sealed record DeviceInfo(string Id, string Name, string? Detail);

/// <summary>
/// Owns the loaded network (one at a time: they can be large and live in GPU memory) and runs upscales.
/// Face restoration models are kept in a second slot beside it, so a batch doesn't reload either per image.
/// Device ids: "auto" (first GPU, falling back to the CPU), "gpu:N" (GPU N of the build's backend, see
/// <see cref="Gpu"/>) and "cpu".
/// </summary>
public sealed class Engine(ILogger<Engine> log) : IDisposable
{
    readonly Lock gate = new();
    IEngineModel? loaded;
    string? loadedKey;
    FaceModels? faces;
    string? facesKey;
    string? lastGpuError;

    /// <summary>How this build reaches the GPU: "DirectML" (Windows), "CUDA" (the Linux CUDA build), "CoreML" (macOS), or null.</summary>
#if DIRECTML
    public const string? GpuBackend = "DirectML";
#elif CUDA
    public const string? GpuBackend = "CUDA";
#elif COREML
    public const string? GpuBackend = "CoreML";
#else
    public const string? GpuBackend = null;
#endif
    public const bool HasGpu = GpuBackend != null;

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
            faces = OnDevice(device, (options, name) => new FaceModels(restorerPath, detectorPath, options(), name));
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
        // Core ML compiles the network for each input shape it sees, and DirectML runs much faster with a static one (each
        // input size gets its own session), so on both every tile of a pass is given the same size.
        return OnDevice(device, (options, name) =>
        {
            var gpu = name != "CPU";
            return new OnnxSrModel(path, options, name, staticShapes: gpu && GpuBackend == "DirectML") { UniformTiles = gpu && (GpuBackend == "CoreML" || GpuBackend == "DirectML") };
        });
    }

    /// <summary>
    /// Creates a model on the chosen GPU, or the CPU; "auto" falls back to the CPU when the GPU fails. The model is given
    /// a factory of session options for the device, as it may need more than one session.
    /// </summary>
    T OnDevice<T>(string device, Func<Func<SessionOptions>, string, T> create)
    {
        var gpus = HasGpu ? Gpu.Adapters() : [];
        // "auto" with no GPU found goes straight to the CPU; an explicit GPU is tried anyway, to report why it fails.
        if (device != "cpu" && HasGpu && (gpus.Count > 0 || device != "auto"))
        {
            var index = device.StartsWith("gpu:") && int.TryParse(device[4..], out var n) ? n : gpus.FirstOrDefault()?.Index ?? 0;
            var name = gpus.FirstOrDefault(g => g.Index == index)?.Name ?? $"GPU {index}";
            try
            {
                var m = create(() => GpuOptions(index), $"{name} ({GpuBackend})");
                lastGpuError = null;
                return m;
            }
            catch (Exception ex) when (ex is OnnxRuntimeException or EntryPointNotFoundException or DllNotFoundException)
            {
                lastGpuError = GpuError(ex);
                log.LogWarning("GPU {Device} unavailable, using the CPU: {Message}", name, ex.Message);
                if (device != "auto") throw new ApiException($"Could not use {name}: {lastGpuError}");
            }
        }
        return create(() => new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL }, "CPU");
    }

    /// <summary>Why the GPU couldn't be used, in words a user can act on where possible.</summary>
    internal static string GpuError(Exception ex)
    {
        // The CUDA build loads the CUDA and cuDNN libraries of the system; ONNX Runtime reports the first one missing.
        var missing = System.Text.RegularExpressions.Regex.Match(ex.Message, @"(lib[\w.+-]+\.so[.\d]*): cannot open shared object file");
        return missing.Success
            ? $"CUDA 12 and cuDNN 9 are needed, but {missing.Groups[1].Value} wasn't found. Install them (NVIDIA's CUDA 12 runtime and cuDNN 9 packages), or use the CPU."
            : ex.Message;
    }

    SupirModel CreateSupir(string dir, string device)
    {
        // On the CPU a 50-step SDXL run takes hours, and float16 networks lack CPU kernels.
#if COREML
        throw new ApiException("SUPIR needs a DirectML (Windows) or CUDA (Linux) GPU; it doesn't run on Core ML.");
#elif !(DIRECTML || CUDA)
        throw new ApiException("SUPIR needs a GPU, which this build can't use. Use the Windows build (DirectML), or the linux-x64-cuda build with an NVIDIA GPU.");
#else
        var package = SupirPackage.Open(dir);
        var gpus = Gpu.Adapters();
        if (device == "cpu" || gpus.Count == 0)
            throw new ApiException("SUPIR needs a GPU. Choose one as the device (in Preferences, or --device on the command line).");
        var index = device.StartsWith("gpu:") && int.TryParse(device[4..], out var n) ? n : gpus[0].Index;
        var name = gpus.FirstOrDefault(g => g.Index == index)?.Name ?? $"GPU {index}";
        return new SupirModel(package, () => GpuOptions(index), $"{name} ({GpuBackend})");
#endif
    }

    static SessionOptions GpuOptions(int deviceId)
    {
#if DIRECTML
        // DirectML doesn't support memory patterns or parallel execution.
        var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            EnableMemoryPattern = false,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
        };
        so.AppendExecutionProvider_DML(deviceId);
#else
        var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
#endif
#if CUDA
        try
        {
            // cuDNN's default exhaustive search benchmarks every new input shape, which makes the first tiles crawl.
            using var cuda = new OrtCUDAProviderOptions();
            cuda.UpdateOptions(new Dictionary<string, string> { ["device_id"] = deviceId.ToString(), ["cudnn_conv_algo_search"] = "HEURISTIC" });
            so.AppendExecutionProvider_CUDA(cuda);
        }
        catch
        {
            so.Dispose();
            throw;
        }
#elif COREML
        // ML Program: the newer Core ML format, with more operators and float16; it runs on the GPU and the Neural Engine.
        so.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_CREATE_MLPROGRAM);
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
