using System.Collections.Concurrent;
using System.Security.Cryptography;

namespace ZawUpscale;

/// <summary>A downloadable file: where it is, how big, and its SHA-256.</summary>
public sealed record RemoteFile(string Url, long Size, string Sha256);

/// <summary>
/// A model ZawUpscale knows how to download. Kind is "upscaler", "face" (face restoration) or "detector" (a face
/// detector, not listed: it's downloaded with the face models, which <see cref="Requires"/> it). <see cref="Half"/> is
/// its float16 version, for GPUs; <see cref="Featured"/> models are shown in the panel before they're downloaded.
/// </summary>
public sealed record CatalogModel(string Id, string Name, string Description, string Url, long Size, string Sha256, string License, string Source,
    string Kind = "upscaler", string[]? Requires = null, RemoteFile? Half = null, bool Featured = false);

/// <summary>
/// A model in the list. Kind is "upscaler" (one ONNX file), "face" (face restoration) or "supir" (a SUPIR folder, with
/// its default settings). For models with a float16 version: its size, and which of the two versions are downloaded.
/// </summary>
public sealed record ModelInfo(
    string Id, string Name, string Description, bool BuiltIn, bool Installed, long Size, string? License, string? Source,
    DownloadState? Download, string Kind = "upscaler", SupirOptions? Supir = null, string? Error = null,
    bool Featured = false, long? HalfSize = null, bool FullInstalled = false, bool HalfInstalled = false);

public sealed record DownloadState(long Received, long Total, string? Error, bool Done);

/// <summary>
/// The models directory (config/models): the built-in catalog, downloaded on demand and checked against
/// its SHA-256, plus any other .onnx file the user puts there, and SUPIR folders (made by tools/supir).
/// A model's float16 version is kept beside it as {id}.fp16.onnx.
/// </summary>
public sealed class ModelStore
{
    const string Hf = "https://huggingface.co";
    const string RealEsrgan = "https://github.com/xinntao/Real-ESRGAN";

    public static readonly CatalogModel[] Catalog =
    [
        new("realesr-general-x4v3", "General (fast)",
            "Real-ESRGAN general v3: small and quick. A good default for photos, screenshots and everyday images.",
            $"{Hf}/jonathanst29/tinier-upscale-models/resolve/main/realesr-general-x4v3.onnx",
            4_866_396, "924ebad6532777303582d4ce7811849b88869231a3ae7093e0f21200249df8d5", "BSD-3-Clause", RealEsrgan,
            Half: new($"{Hf}/skillsafe-ai/realesr-general-x4v3/resolve/main/model_fp16.onnx", 2_445_496, "8dedfc4b8916f2426111561351bbd3c35d674214a3f5bc28190084cd15133a67"),
            Featured: true),
        new("realesrgan-x4plus", "Real-ESRGAN x4plus (high quality)",
            "The full Real-ESRGAN network: the most detail for real-world photos, about ten times slower than General.",
            $"{Hf}/jonathanst29/tinier-upscale-models/resolve/main/realesrgan-x4plus.onnx",
            67_051_618, "4ed6a45a8185bde6c8d7c7790a6e80e3c5b9f608946f634021068dd0fd0473c8", "BSD-3-Clause", RealEsrgan,
            Half: new($"{Hf}/skillsafe-ai/realesrgan-x4plus/resolve/main/model_fp16.onnx", 33_754_617, "38fe853c9cd8e1d6fd0ff387f2d43c51c1352feb9d7b3e2533d4c07b1b932f68"),
            Featured: true),
        new("realesrgan-x4plus-anime", "Anime & illustration",
            "Real-ESRGAN x4plus anime 6B: clean lines and flat colors for anime, cartoons, drawings and digital art.",
            $"{Hf}/skillsafe-ai/realesrgan-x4plus-anime-6b/resolve/main/model.onnx",
            17_939_975, "9a029dae374b2cd4c3eb2dbbed857e22e6d785687c767aa12828e7356b9fb378", "BSD-3-Clause", RealEsrgan,
            Half: new($"{Hf}/skillsafe-ai/realesrgan-x4plus-anime-6b/resolve/main/model_fp16.onnx", 9_031_567, "1777c7dffecdf356c234e3387d456a196ed9b72f9f14ed96bc097b9798b7ddbf"),
            Featured: true),
        new("realesrgan-x2plus", "Real-ESRGAN x2plus (2×)",
            "Real-ESRGAN trained for 2× itself: for 2× and 3×, it keeps finer detail than a 4× model reduced afterwards.",
            $"{Hf}/skillsafe-ai/realesrgan-x2plus/resolve/main/model.onnx",
            67_076_136, "7eb5e9eb507df603c0c04b49b23c86a362e0a01575bfbbac891d970f9c331888", "BSD-3-Clause", RealEsrgan,
            Half: new($"{Hf}/skillsafe-ai/realesrgan-x2plus/resolve/main/model_fp16.onnx", 33_771_615, "382dc047b478d71200ff76e9fdb920953106a8ac63380ca7370aba357c44cc5f")),
        new("realplksr-x4", "RealPLKSR (faithful)",
            "Sharp and true to the original, without invented texture: for photos where accuracy matters more than polish. Best on a GPU.",
            $"{Hf}/darktable-org/upscale-realplksr-onnx/resolve/main/onnx/model_x4.onnx",
            29_711_010, "8ce692652494e2e5c31e1f7c7ab61eb084e9a450cde727a374f5d6746cfe054b", "MIT", "https://github.com/dslisleedh/PLKSR"),
        new("realplksr-x2", "RealPLKSR 2× (faithful)",
            "RealPLKSR trained for 2×: true-to-life detail at 2× and 3×, without invented texture. Best on a GPU.",
            $"{Hf}/darktable-org/upscale-realplksr-onnx/resolve/main/onnx/model_x2.onnx",
            29_627_920, "d7abb65092f3808d3aa255ffdab42b1d672883915d90adbdd321204168b9f293", "MIT", "https://github.com/dslisleedh/PLKSR"),
        new("gfpgan-v1.4", "GFPGAN 1.4",
            "Rebuilds blurry, small or damaged faces with natural detail. Works on photos; it can make faces look smoother than they were.",
            $"{Hf}/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
            340_299_087, "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385", "Apache-2.0",
            "https://github.com/TencentARC/GFPGAN", "face", [FaceDetectorId]),
        new("codeformer", "CodeFormer",
            "Restores even badly degraded faces and keeps them closer to the person's likeness. For non-commercial use only.",
            $"{Hf}/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
            376_951_650, "21710e7ab61c82683576c428e9c1b6fe1ed419586b7b39e394c3449c294b550f", "S-Lab License 1.0 (non-commercial)",
            "https://github.com/sczhou/CodeFormer", "face", [FaceDetectorId]),
        new(FaceDetectorId, "YuNet face detector",
            "Finds faces and their landmarks for face restoration.",
            $"{Hf}/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
            232_589, "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", "MIT",
            "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet", "detector"),
    ];

    public const string FaceDetectorId = "yunet-face-2023mar";

    /// <summary>True for the face restoration models of the catalog.</summary>
    public static bool IsFaceModel(string? id) => Catalog.Any(c => c.Id == id && c.Kind == "face");

    public string Dir { get; }
    readonly ConcurrentDictionary<string, DownloadState> downloads = new();
    readonly ConcurrentDictionary<string, CancellationTokenSource> running = new();
    readonly HttpClient http;
    readonly ILogger<ModelStore> log;

    public ModelStore(AppOptions o, ILogger<ModelStore> log)
    {
        this.log = log;
        Dir = Path.Combine(o.ConfigDir, "models");
        Directory.CreateDirectory(Dir);
        http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        http.DefaultRequestHeaders.UserAgent.ParseAdd($"ZawUpscale/{Api.Version}");
    }

    public string PathOf(string id)
    {
        if (string.IsNullOrWhiteSpace(id) || id.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || id.Contains(".."))
            throw new ApiException($"Invalid model name \"{id}\".");
        var folder = Path.Combine(Dir, id);
        return SupirPackage.IsPackage(folder) ? folder : Path.Combine(Dir, id + ".onnx");
    }

    /// <summary>Where a model's float16 version is kept.</summary>
    public string HalfPathOf(string id) => Path.Combine(Dir, id + ".fp16.onnx");

    /// <summary>
    /// The file to run: the float16 version when <paramref name="half"/> is asked for and it's downloaded, otherwise
    /// the float32 one; whichever exists when only one does.
    /// </summary>
    public string ModelPath(string id, bool half)
    {
        var full = PathOf(id);
        if (SupirPackage.IsPackage(full)) return full;
        var halfPath = HalfPathOf(id);
        if (half && File.Exists(halfPath)) return halfPath;
        if (File.Exists(full)) return full;
        return File.Exists(halfPath) ? halfPath : full;
    }

    /// <summary>True if the model (either version) and the models it needs (a face model's detector) are all there.</summary>
    public bool IsInstalled(string id)
    {
        var path = PathOf(id);
        if (SupirPackage.IsPackage(path)) return true;
        var needs = Catalog.FirstOrDefault(c => c.Id == id)?.Requires ?? [];
        return (File.Exists(path) || File.Exists(HalfPathOf(id))) && needs.All(r => File.Exists(PathOf(r)));
    }

    public List<ModelInfo> List()
    {
        var list = new List<ModelInfo>();
        foreach (var c in Catalog.Where(c => c.Kind != "detector"))
        {
            downloads.TryGetValue(c.Id, out var d);
            list.Add(new ModelInfo(c.Id, c.Name, c.Description, true, IsInstalled(c.Id), c.Size, c.License, c.Source, d, c.Kind,
                Featured: c.Featured, HalfSize: c.Half?.Size, FullInstalled: File.Exists(PathOf(c.Id)), HalfInstalled: c.Half != null && File.Exists(HalfPathOf(c.Id))));
        }
        foreach (var f in Directory.EnumerateFiles(Dir, "*.onnx").OrderBy(f => f, StringComparer.OrdinalIgnoreCase))
        {
            var id = Path.GetFileNameWithoutExtension(f);
            // Catalog models, and their float16 versions, are listed above.
            if (Catalog.Any(c => c.Id == id || c.Id + ".fp16" == id)) continue;
            list.Add(new ModelInfo(id, id, "Your own model (an ONNX file in the models folder).", false, true, new FileInfo(f).Length, null, null, null, FullInstalled: true));
        }
        foreach (var d in Directory.EnumerateDirectories(Dir).Where(SupirPackage.IsPackage).OrderBy(d => d, StringComparer.OrdinalIgnoreCase))
            list.Add(SupirInfo(d));
        return list;
    }

    static ModelInfo SupirInfo(string dir)
    {
        var id = Path.GetFileName(dir);
        const string about = "SUPIR: SDXL-based restoration that rebuilds real detail in blurry, noisy or damaged photos. Slow (minutes per image) and needs a GPU with about 12 GB.";
        try
        {
            var p = SupirPackage.Open(dir, checkFiles: false);
            var missing = p.MissingFiles();
            return new ModelInfo(id, p.Manifest.Name ?? id, p.Manifest.Description ?? about, false, missing.Count == 0, p.Size(), null, "https://github.com/Fanghua-Yu/SUPIR",
                null, "supir", p.Defaults, missing.Count > 0 ? $"Incomplete: missing {string.Join(", ", missing)}." : null);
        }
        catch (ApiException ex)
        {
            return new ModelInfo(id, id, about, false, false, 0, null, null, null, "supir", null, ex.Message);
        }
    }

    /// <summary>
    /// Starts downloading a catalog model in the background (its float16 version with <paramref name="half"/>, where
    /// there is one); progress shows up in <see cref="List"/>.
    /// </summary>
    public void StartDownload(string id, bool half = false)
    {
        var c = Catalog.FirstOrDefault(m => m.Id == id) ?? throw new ApiException($"Unknown model \"{id}\".");
        if (running.ContainsKey(id)) return;
        var cts = new CancellationTokenSource();
        if (!running.TryAdd(id, cts)) return;
        // The models it needs first (the face detector), then the model itself, as one download.
        var parts = (c.Requires ?? []).Select(r => Catalog.First(m => m.Id == r))
            .Where(m => !File.Exists(PathOf(m.Id)))
            .Select(m => (File: new RemoteFile(m.Url, m.Size, m.Sha256), Target: PathOf(m.Id)))
            .Append(half && c.Half != null ? (File: c.Half, Target: HalfPathOf(c.Id)) : (File: new RemoteFile(c.Url, c.Size, c.Sha256), Target: PathOf(c.Id)))
            .ToList();
        var total = parts.Sum(p => p.File.Size);
        downloads[id] = new DownloadState(0, total, null, false);
        _ = Task.Run(async () =>
        {
            try
            {
                long offset = 0;
                foreach (var (file, target) in parts)
                {
                    await Download(file, target, id, offset, total, cts.Token);
                    offset += file.Size;
                }
                downloads[id] = new DownloadState(total, total, null, true);
            }
            catch (Exception ex)
            {
                log.LogWarning("Download of {Model} failed: {Message}", id, ex.Message);
                downloads[id] = new DownloadState(0, total, ex is OperationCanceledException ? "Cancelled." : ex.Message, true);
            }
            finally
            {
                running.TryRemove(id, out _);
                cts.Dispose();
            }
        });
    }

    public void CancelDownload(string id)
    {
        if (running.TryGetValue(id, out var cts)) cts.Cancel();
    }

    /// <summary>Forgets finished downloads (so errors don't linger in the list).</summary>
    public void ClearDownload(string id)
    {
        if (!running.ContainsKey(id)) downloads.TryRemove(id, out _);
    }

    /// <summary>Downloads one file, reporting progress as part of the download <paramref name="progressId"/> (bytes before it: <paramref name="offset"/>).</summary>
    async Task Download(RemoteFile file, string target, string progressId, long offset, long total, CancellationToken ct)
    {
        var part = target + ".part";
        try
        {
            using var res = await http.GetAsync(file.Url, HttpCompletionOption.ResponseHeadersRead, ct);
            if (!res.IsSuccessStatusCode) throw new ApiException($"The server answered {(int)res.StatusCode} {res.ReasonPhrase}.");
            await using (var src = await res.Content.ReadAsStreamAsync(ct))
            await using (var dst = File.Create(part))
            {
                using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                var buf = new byte[1 << 16];
                long got = 0;
                int n;
                while ((n = await src.ReadAsync(buf, ct)) > 0)
                {
                    await dst.WriteAsync(buf.AsMemory(0, n), ct);
                    sha.AppendData(buf, 0, n);
                    got += n;
                    downloads[progressId] = new DownloadState(offset + got, total, null, false);
                }
                var hash = Convert.ToHexString(sha.GetHashAndReset()).ToLowerInvariant();
                if (hash != file.Sha256) throw new ApiException("The downloaded file is damaged or has changed (SHA-256 mismatch). Try again later.");
            }
            File.Move(part, target, overwrite: true);
        }
        finally
        {
            try { if (File.Exists(part)) File.Delete(part); } catch (IOException) { /* best effort */ }
        }
    }

    /// <summary>Deletes a model: both versions of a catalog model, or a SUPIR folder.</summary>
    public void Delete(string id)
    {
        var path = PathOf(id);
        if (SupirPackage.IsPackage(path)) Directory.Delete(path, recursive: true);
        else
        {
            if (File.Exists(path)) File.Delete(path);
            if (Catalog.Any(c => c.Id == id && c.Half != null) && File.Exists(HalfPathOf(id))) File.Delete(HalfPathOf(id));
        }
        downloads.TryRemove(id, out _);
    }

    /// <summary>Copies a user's .onnx file into the models folder. Returns the new model id.</summary>
    public string AddFile(string source)
    {
        source = source.Trim().Trim('"');
        if (!File.Exists(source)) throw new ApiException($"File not found: {source}");
        if (!source.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase)) throw new ApiException("Only ONNX models (.onnx files) can be added.");
        var id = Path.GetFileNameWithoutExtension(source);
        var target = PathOf(id);
        if (File.Exists(target)) throw new ApiException($"There is already a model named \"{id}\".");
        File.Copy(source, target);
        return id;
    }
}
