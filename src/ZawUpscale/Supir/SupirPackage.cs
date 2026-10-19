using System.Text.Json;

namespace ZawUpscale;

/// <summary>SUPIR options from the UI or the command line. Nulls take the model package's defaults.</summary>
public sealed record SupirSettings
{
    /// <summary>What the image shows ("a red fox kit in the grass"); put in front of the quality prompt.</summary>
    public string? Prompt { get; init; }
    /// <summary>Words describing the look of a good photo, added after <see cref="Prompt"/>.</summary>
    public string? QualityPrompt { get; init; }
    public string? NegativePrompt { get; init; }
    public int? Steps { get; init; }
    /// <summary>How strongly the prompts steer the result (classifier-free guidance) at the end of sampling.</summary>
    public double? Guidance { get; init; }
    /// <summary>The guidance at the start of sampling; it moves linearly to <see cref="Guidance"/>.</summary>
    public double? GuidanceStart { get; init; }
    /// <summary>How closely the result follows the input image (SUPIR's control scale) at the end of sampling.</summary>
    public double? Control { get; init; }
    public double? ControlStart { get; init; }
    /// <summary>SUPIR's restoration guidance: pulls early steps towards the input, for fewer invented details. 0 is off.</summary>
    public double? Restoration { get; init; }
    /// <summary>Extra noise added at each step (EDM s_churn) and its scale (s_noise).</summary>
    public double? Churn { get; init; }
    public double? Noise { get; init; }
    /// <summary>Null picks a random seed for each image.</summary>
    public int? Seed { get; init; }
    /// <summary>"wavelet", "adain" or "none": take the colors of the input, removing color shifts.</summary>
    public string? ColorFix { get; init; }
}

/// <summary>SUPIR settings with every value filled in.</summary>
public sealed record SupirOptions(
    string Prompt, string QualityPrompt, string NegativePrompt, int Steps, double Guidance, double GuidanceStart,
    double Control, double ControlStart, double Restoration, double Churn, double Noise, int? Seed, string ColorFix)
{
    /// <summary>The positive prompt: the description, then the quality words (as SUPIR joins a caption and its a_prompt).</summary>
    public string PositivePrompt => string.IsNullOrWhiteSpace(Prompt) ? QualityPrompt
        : string.IsNullOrWhiteSpace(QualityPrompt) ? Prompt.Trim()
        : Prompt.Trim().TrimEnd(',', '.') + ", " + QualityPrompt;
}

/// <summary>The supir.json file in a SUPIR model folder.</summary>
public sealed record SupirManifest
{
    public string? Format { get; init; }
    public int Version { get; init; }
    public string? Name { get; init; }
    public string? Description { get; init; }
    /// <summary>Images whose shorter side is below this are processed larger and reduced afterwards; SDXL works best at 1024.</summary>
    public int? MinSize { get; init; }
    public SupirSettings? Defaults { get; init; }
}

/// <summary>
/// A SUPIR model: a folder in the models directory holding supir.json and the ONNX networks made by
/// tools/supir/export_supir.py (the SDXL UNet with SUPIR's control network, the VAE, the two CLIP text
/// encoders and their tokenizer).
/// </summary>
public sealed class SupirPackage
{
    public const string ManifestName = "supir.json";
    public const string FormatName = "zawupscale-supir";
    public const string Unet = "unet.onnx", DenoiseEncoder = "denoise_encoder.onnx", VaeEncoder = "vae_encoder.onnx", VaeDecoder = "vae_decoder.onnx";
    public const string TextEncoderL = "text_encoder_l.onnx", TextEncoderG = "text_encoder_g.onnx";
    public const string Vocab = "tokenizer/vocab.json", Merges = "tokenizer/merges.txt";
    public static readonly string[] Files = [Unet, DenoiseEncoder, VaeEncoder, VaeDecoder, TextEncoderL, TextEncoderG, Vocab, Merges];

    /// <summary>SUPIR's "Quality" preset (its Gradio demo defaults) and default prompts.</summary>
    public static readonly SupirOptions BuiltInDefaults = new(
        Prompt: "",
        QualityPrompt: "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.",
        NegativePrompt: "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth",
        Steps: 50, Guidance: 7.5, GuidanceStart: 4.0, Control: 1.0, ControlStart: 1.0, Restoration: 0, Churn: 5, Noise: 1.003,
        Seed: null, ColorFix: "wavelet");

    public const int DefaultMinSize = 1024;

    public string Dir { get; }
    public SupirManifest Manifest { get; }
    public SupirOptions Defaults { get; }
    public int MinSize => Math.Clamp(Manifest.MinSize ?? DefaultMinSize, 64, 4096);

    SupirPackage(string dir, SupirManifest manifest)
    {
        Dir = dir;
        Manifest = manifest;
        Defaults = Merge(BuiltInDefaults, manifest.Defaults);
    }

    public static bool IsPackage(string dir) => File.Exists(Path.Combine(dir, ManifestName));

    /// <summary>Reads the manifest. <paramref name="checkFiles"/> also requires every network to be present.</summary>
    public static SupirPackage Open(string dir, bool checkFiles = true)
    {
        var path = Path.Combine(dir, ManifestName);
        if (!File.Exists(path)) throw new ApiException($"Not a SUPIR model folder (no {ManifestName}): {dir}");
        SupirManifest m;
        try
        {
            m = JsonSerializer.Deserialize<SupirManifest>(File.ReadAllText(path), Workspace.Json) ?? new SupirManifest();
        }
        catch (JsonException ex)
        {
            throw new ApiException($"{ManifestName} in {dir} is damaged: {ex.Message}");
        }
        if (m.Format != FormatName) throw new ApiException($"{ManifestName} in {dir} isn't a ZawUpscale SUPIR manifest (format \"{m.Format}\").");
        if (m.Version != 1) throw new ApiException($"This SUPIR model was made for a newer ZawUpscale (format version {m.Version}).");
        var p = new SupirPackage(dir, m);
        if (checkFiles && p.MissingFiles() is { Count: > 0 } missing)
            throw new ApiException($"The SUPIR model in {dir} is incomplete; missing {string.Join(", ", missing)}. Run the export script again.");
        return p;
    }

    public List<string> MissingFiles() => Files.Where(f => !File.Exists(Path.Combine(Dir, f))).ToList();

    public string PathOf(string file) => Path.Combine(Dir, file);

    /// <summary>Total size of the folder, including the networks' external weight files.</summary>
    public long Size()
    {
        try
        {
            return new DirectoryInfo(Dir).EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return 0;
        }
    }

    /// <summary>The settings to run with: the user's, then the package's defaults, then SUPIR's.</summary>
    public SupirOptions Resolve(SupirSettings? s) => Validate(Merge(Defaults, s));

    static SupirOptions Merge(SupirOptions d, SupirSettings? s) => s == null ? d : new(
        Prompt: Text(s.Prompt) ?? d.Prompt,
        QualityPrompt: s.QualityPrompt?.Trim() ?? d.QualityPrompt,
        NegativePrompt: s.NegativePrompt?.Trim() ?? d.NegativePrompt,
        Steps: s.Steps ?? d.Steps,
        Guidance: s.Guidance ?? d.Guidance,
        // Without its own start value, guidance stays where the user put it rather than ramping from the default start.
        GuidanceStart: s.GuidanceStart ?? (s.Guidance != null && d.GuidanceStart > s.Guidance ? s.Guidance.Value : d.GuidanceStart),
        Control: s.Control ?? d.Control,
        ControlStart: s.ControlStart ?? (s.Control != null && d.ControlStart == d.Control ? s.Control.Value : d.ControlStart),
        Restoration: s.Restoration ?? d.Restoration,
        Churn: s.Churn ?? d.Churn,
        Noise: s.Noise ?? d.Noise,
        Seed: s.Seed ?? d.Seed,
        ColorFix: Text(s.ColorFix)?.ToLowerInvariant() ?? d.ColorFix);

    static string? Text(string? s) => string.IsNullOrWhiteSpace(s) ? null : s.Trim();

    /// <summary>Checks the ranges, so a typo fails before a model is loaded rather than minutes into a job.</summary>
    public static SupirOptions Validate(SupirOptions o)
    {
        static void Range(double v, double min, double max, string what)
        {
            if (double.IsNaN(v) || v < min || v > max) throw new ApiException($"SUPIR {what} must be between {min} and {max}.");
        }
        Range(o.Steps, 1, 200, "steps");
        Range(o.Guidance, 0, 30, "guidance");
        Range(o.GuidanceStart, 0, 30, "starting guidance");
        Range(o.Control, 0, 2, "control strength");
        Range(o.ControlStart, 0, 2, "starting control strength");
        Range(o.Restoration, 0, 20, "restoration guidance");
        Range(o.Churn, 0, 100, "churn");
        Range(o.Noise, 0.5, 1.5, "noise scale");
        if (o.Seed is < 0) throw new ApiException("The SUPIR seed must be 0 or more (or empty for a random one).");
        if (o.ColorFix is not ("wavelet" or "adain" or "none")) throw new ApiException($"Unknown color fix \"{o.ColorFix}\"; use wavelet, adain or none.");
        return o;
    }

    /// <summary>Validates the user's settings on their own (before the model's defaults are known).</summary>
    public static void Validate(SupirSettings? s) => Validate(Merge(BuiltInDefaults, s));
}
