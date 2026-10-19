using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SessionOptions = Microsoft.ML.OnnxRuntime.SessionOptions;

namespace ZawUpscale;

/// <summary>
/// SUPIR's ONNX networks. The UNet and the VAE run on the GPU; each session is built for one tile shape
/// (DirectML optimizes much better with fixed sizes, and all tiles of a pass have one size). The text
/// encoders run on the CPU once per prompt and are let go straight after, so they take no GPU memory.
/// </summary>
public sealed class SupirModel : ISupirNetworks, IEngineModel
{
    readonly SupirPackage package;
    readonly Func<SessionOptions> gpuOptions;
    readonly Dictionary<string, (string Shape, InferenceSession Session)> sessions = [];
    readonly Dictionary<string, TextEmbedding> prompts = [];
    readonly ClipTokenizer tokenizer;
    RunOptions runOptions = new();

    public string Device { get; }
    public SupirPackage Package => package;

    public SupirModel(SupirPackage package, Func<SessionOptions> gpuOptions, string device)
    {
        this.package = package;
        this.gpuOptions = gpuOptions;
        Device = device;
        tokenizer = ClipTokenizer.Load(package.PathOf(SupirPackage.Vocab), package.PathOf(SupirPackage.Merges));
    }

    /// <summary>The session for a network and tile shape, replacing one built for another shape (one shape at a time each, for memory).</summary>
    InferenceSession Session(string file, params (string Name, long Value)[] dims)
    {
        var shape = string.Join(",", dims.Select(d => $"{d.Name}={d.Value}"));
        if (sessions.TryGetValue(file, out var s) && s.Shape == shape) return s.Session;
        if (s.Session != null)
        {
            s.Session.Dispose();
            sessions.Remove(file);
        }
        var so = gpuOptions();
        foreach (var (name, value) in dims) so.AddFreeDimensionOverrideByName(name, value);
        try
        {
            var session = new InferenceSession(package.PathOf(file), so);
            sessions[file] = (shape, session);
            return session;
        }
        catch (OnnxRuntimeException ex) when (OnnxSrModel.IsOutOfMemory(ex))
        {
            throw new TileTooLargeException(ex.Message, ex);
        }
        finally
        {
            so.Dispose();
        }
    }

    float[] Run(InferenceSession session, string output, params (string Name, OrtValue Value)[] inputs)
    {
        try
        {
            using var outputs = session.Run(runOptions, inputs.Select(i => i.Name).ToArray(), inputs.Select(i => i.Value).ToArray(), [output]);
            return outputs[0].GetTensorDataAsSpan<float>().ToArray();
        }
        catch (OnnxRuntimeException ex) when (runOptions.Terminate)
        {
            throw new OperationCanceledException("Cancelled.", ex);
        }
        catch (OnnxRuntimeException ex) when (OnnxSrModel.IsOutOfMemory(ex))
        {
            throw new TileTooLargeException(ex.Message, ex);
        }
        finally
        {
            foreach (var i in inputs) i.Value.Dispose();
        }
    }

    static OrtValue Tensor(float[] data, params long[] shape) => OrtValue.CreateTensorValueFromMemory(data, shape);

    public float[] DenoiseEncode(float[] image, int h, int w) => EncodeWith(SupirPackage.DenoiseEncoder, image, h, w);
    public float[] Encode(float[] image, int h, int w) => EncodeWith(SupirPackage.VaeEncoder, image, h, w);

    float[] EncodeWith(string file, float[] image, int h, int w) =>
        Run(Session(file, ("batch", 1), ("img_h", h), ("img_w", w)), "latent", ("image", Tensor(image, 1, 3, h, w)));

    public float[] Decode(float[] latent, int h, int w) =>
        Run(Session(SupirPackage.VaeDecoder, ("batch", 1), ("lat_h", h), ("lat_w", w)), "image", ("latent", Tensor(latent, 1, 4, h, w)));

    public float[] Unet(float[] sample, float[] timestep, float[] context, float[] y, float[] control, float controlScale, int batch, int h, int w) =>
        Run(Session(SupirPackage.Unet, ("batch", batch), ("lat_h", h), ("lat_w", w)), "eps",
            ("sample", Tensor(sample, batch, 4, h, w)),
            ("timestep", Tensor(timestep, batch)),
            ("context", Tensor(context, batch, 77, 2048)),
            ("y", Tensor(y, batch, SupirUpscaler.VectorSize)),
            ("control", Tensor(control, batch, 4, h, w)),
            ("control_scale", Tensor([controlScale], 1)));

    /// <summary>Both CLIP encoders' penultimate hidden states side by side, and bigG's pooled projection.</summary>
    public TextEmbedding EncodeText(string text)
    {
        if (prompts.TryGetValue(text, out var hit)) return hit;
        using var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        float[] l, g, pooled;
        using (var enc = new InferenceSession(package.PathOf(SupirPackage.TextEncoderL), so))
        using (var ids = OrtValue.CreateTensorValueFromMemory(tokenizer.Encode(text, ClipTokenizer.EndOfText), [1, ClipTokenizer.Length]))
        using (var outputs = enc.Run(runOptions, ["input_ids"], [ids], ["hidden"]))
            l = outputs[0].GetTensorDataAsSpan<float>().ToArray();
        using (var enc = new InferenceSession(package.PathOf(SupirPackage.TextEncoderG), so))
        using (var ids = OrtValue.CreateTensorValueFromMemory(tokenizer.Encode(text, 0), [1, ClipTokenizer.Length]))
        using (var outputs = enc.Run(runOptions, ["input_ids"], [ids], ["hidden", "pooled"]))
        {
            g = outputs[0].GetTensorDataAsSpan<float>().ToArray();
            pooled = outputs[1].GetTensorDataAsSpan<float>().ToArray();
        }
        const int n = ClipTokenizer.Length, dl = 768, dg = 1280;
        if (l.Length != n * dl || g.Length != n * dg)
            throw new ApiException($"Unexpected text encoder outputs ({l.Length} and {g.Length} values; expected {n * dl} and {n * dg}).");
        var context = new float[n * (dl + dg)];
        for (var t = 0; t < n; t++)
        {
            Array.Copy(l, t * dl, context, t * (dl + dg), dl);
            Array.Copy(g, t * dg, context, t * (dl + dg) + dl, dg);
        }
        var e = new TextEmbedding(context, pooled);
        prompts[text] = e;
        return e;
    }

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
        foreach (var (_, s) in sessions.Values) s.Dispose();
        sessions.Clear();
        runOptions.Dispose();
    }
}
