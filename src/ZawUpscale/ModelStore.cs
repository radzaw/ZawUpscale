using System.Collections.Concurrent;
using System.Security.Cryptography;

namespace ZawUpscale;

/// <summary>
/// A model ZawUpscale knows how to download. Kind is "upscaler", "face" (face restoration) or "detector" (a face
/// detector, not listed: it's downloaded with the face models, which <see cref="Requires"/> it).
/// </summary>
public sealed record CatalogModel(string Id, string Name, string Description, string Url, long Size, string Sha256, string License, string Source,
    string Kind = "upscaler", string[]? Requires = null);

/// <summary>A model in the list. Kind is "upscaler" (one ONNX file), "face" (face restoration) or "supir" (a SUPIR folder, with its default settings).</summary>
public sealed record ModelInfo(
    string Id, string Name, string Description, bool BuiltIn, bool Installed, long Size, string? License, string? Source,
    DownloadState? Download, string Kind = "upscaler", SupirOptions? Supir = null, string? Error = null);

public sealed record DownloadState(long Received, long Total, string? Error, bool Done);

/// <summary>
/// The models directory (config/models): the built-in catalog, downloaded on demand and checked against
/// its SHA-256, plus any other .onnx file the user puts there, and SUPIR folders (made by tools/supir).
/// </summary>
public sealed class ModelStore
{
    public static readonly CatalogModel[] Catalog =
    [
        new("realesr-general-x4v3", "General (fast)",
            "Real-ESRGAN general v3: small and quick. A good default for photos, screenshots and everyday images.",
            "https://huggingface.co/jonathanst29/tinier-upscale-models/resolve/main/realesr-general-x4v3.onnx",
            4_866_396, "924ebad6532777303582d4ce7811849b88869231a3ae7093e0f21200249df8d5", "BSD-3-Clause",
            "https://github.com/xinntao/Real-ESRGAN"),
        new("realesrgan-x4plus", "Real-ESRGAN x4plus (high quality)",
            "The full Real-ESRGAN network: the most detail for real-world photos, about ten times slower than General.",
            "https://huggingface.co/jonathanst29/tinier-upscale-models/resolve/main/realesrgan-x4plus.onnx",
            67_051_618, "4ed6a45a8185bde6c8d7c7790a6e80e3c5b9f608946f634021068dd0fd0473c8", "BSD-3-Clause",
            "https://github.com/xinntao/Real-ESRGAN"),
        new("realesrgan-x4plus-anime", "Anime & illustration",
            "Real-ESRGAN x4plus anime 6B: clean lines and flat colors for anime, cartoons, drawings and digital art.",
            "https://huggingface.co/skillsafe-ai/realesrgan-x4plus-anime-6b/resolve/main/model.onnx",
            17_939_975, "9a029dae374b2cd4c3eb2dbbed857e22e6d785687c767aa12828e7356b9fb378", "BSD-3-Clause",
            "https://github.com/xinntao/Real-ESRGAN"),
        new("gfpgan-v1.4", "GFPGAN 1.4",
            "Rebuilds blurry, small or damaged faces with natural detail. Works on photos; it can make faces look smoother than they were.",
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
            340_299_087, "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385", "Apache-2.0",
            "https://github.com/TencentARC/GFPGAN", "face", [FaceDetectorId]),
        new("codeformer", "CodeFormer",
            "Restores even badly degraded faces and keeps them closer to the person's likeness. For non-commercial use only.",
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
            376_951_650, "21710e7ab61c82683576c428e9c1b6fe1ed419586b7b39e394c3449c294b550f", "S-Lab License 1.0 (non-commercial)",
            "https://github.com/sczhou/CodeFormer", "face", [FaceDetectorId]),
        new(FaceDetectorId, "YuNet face detector",
            "Finds faces and their landmarks for face restoration.",
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
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

    /// <summary>True if the model and the models it needs (a face model's detector) are all there.</summary>
    public bool IsInstalled(string id)
    {
        var path = PathOf(id);
        if (SupirPackage.IsPackage(path)) return true;
        var needs = Catalog.FirstOrDefault(c => c.Id == id)?.Requires ?? [];
        return File.Exists(path) && needs.All(r => File.Exists(PathOf(r)));
    }

    public List<ModelInfo> List()
    {
        var list = new List<ModelInfo>();
        foreach (var c in Catalog.Where(c => c.Kind != "detector"))
        {
            downloads.TryGetValue(c.Id, out var d);
            list.Add(new ModelInfo(c.Id, c.Name, c.Description, true, IsInstalled(c.Id), c.Size, c.License, c.Source, d, c.Kind));
        }
        foreach (var f in Directory.EnumerateFiles(Dir, "*.onnx").OrderBy(f => f, StringComparer.OrdinalIgnoreCase))
        {
            var id = Path.GetFileNameWithoutExtension(f);
            if (Catalog.Any(c => c.Id == id)) continue;
            list.Add(new ModelInfo(id, id, "Your own model (an ONNX file in the models folder).", false, true, new FileInfo(f).Length, null, null, null));
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

    /// <summary>Starts downloading a catalog model in the background; progress shows up in <see cref="List"/>.</summary>
    public void StartDownload(string id)
    {
        var c = Catalog.FirstOrDefault(m => m.Id == id) ?? throw new ApiException($"Unknown model \"{id}\".");
        if (running.ContainsKey(id)) return;
        var cts = new CancellationTokenSource();
        if (!running.TryAdd(id, cts)) return;
        // The models it needs first (the face detector), then the model itself, as one download.
        var parts = (c.Requires ?? []).Select(r => Catalog.First(m => m.Id == r)).Append(c)
            .Where(m => m == c || !File.Exists(PathOf(m.Id))).ToList();
        var total = parts.Sum(m => m.Size);
        downloads[id] = new DownloadState(0, total, null, false);
        _ = Task.Run(async () =>
        {
            try
            {
                long offset = 0;
                foreach (var part in parts)
                {
                    await Download(part, id, offset, total, cts.Token);
                    offset += part.Size;
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
    async Task Download(CatalogModel c, string progressId, long offset, long total, CancellationToken ct)
    {
        var target = Path.Combine(Dir, c.Id + ".onnx");
        var part = target + ".part";
        try
        {
            using var res = await http.GetAsync(c.Url, HttpCompletionOption.ResponseHeadersRead, ct);
            if (!res.IsSuccessStatusCode) throw new ApiException($"The server answered {(int)res.StatusCode} {res.ReasonPhrase}.");
            if (c.Id == progressId) total = offset + (res.Content.Headers.ContentLength ?? c.Size);
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
                if (hash != c.Sha256) throw new ApiException("The downloaded file is damaged or has changed (SHA-256 mismatch). Try again later.");
            }
            File.Move(part, target, overwrite: true);
        }
        finally
        {
            try { if (File.Exists(part)) File.Delete(part); } catch (IOException) { /* best effort */ }
        }
    }

    public void Delete(string id)
    {
        var path = PathOf(id);
        if (SupirPackage.IsPackage(path)) Directory.Delete(path, recursive: true);
        else if (File.Exists(path)) File.Delete(path);
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
