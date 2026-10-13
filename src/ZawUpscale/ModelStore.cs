using System.Collections.Concurrent;
using System.Security.Cryptography;

namespace ZawUpscale;

/// <summary>A model ZawUpscale knows how to download.</summary>
public sealed record CatalogModel(string Id, string Name, string Description, string Url, long Size, string Sha256, string License, string Source);

public sealed record ModelInfo(
    string Id, string Name, string Description, bool BuiltIn, bool Installed, long Size, string? License, string? Source,
    DownloadState? Download);

public sealed record DownloadState(long Received, long Total, string? Error, bool Done);

/// <summary>
/// The models directory (config/models): the built-in catalog, downloaded on demand and checked against
/// its SHA-256, plus any other .onnx file the user puts there.
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
    ];

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
        return Path.Combine(Dir, id + ".onnx");
    }

    public bool IsInstalled(string id) => File.Exists(PathOf(id));

    public List<ModelInfo> List()
    {
        var list = new List<ModelInfo>();
        foreach (var c in Catalog)
        {
            var path = Path.Combine(Dir, c.Id + ".onnx");
            downloads.TryGetValue(c.Id, out var d);
            list.Add(new ModelInfo(c.Id, c.Name, c.Description, true, File.Exists(path), c.Size, c.License, c.Source, d));
        }
        foreach (var f in Directory.EnumerateFiles(Dir, "*.onnx").OrderBy(f => f, StringComparer.OrdinalIgnoreCase))
        {
            var id = Path.GetFileNameWithoutExtension(f);
            if (Catalog.Any(c => c.Id == id)) continue;
            list.Add(new ModelInfo(id, id, "Your own model (an ONNX file in the models folder).", false, true, new FileInfo(f).Length, null, null, null));
        }
        return list;
    }

    /// <summary>Starts downloading a catalog model in the background; progress shows up in <see cref="List"/>.</summary>
    public void StartDownload(string id)
    {
        var c = Catalog.FirstOrDefault(m => m.Id == id) ?? throw new ApiException($"Unknown model \"{id}\".");
        if (running.ContainsKey(id)) return;
        var cts = new CancellationTokenSource();
        if (!running.TryAdd(id, cts)) return;
        downloads[id] = new DownloadState(0, c.Size, null, false);
        _ = Task.Run(async () =>
        {
            try
            {
                await Download(c, cts.Token);
                downloads[id] = new DownloadState(c.Size, c.Size, null, true);
            }
            catch (Exception ex)
            {
                log.LogWarning("Download of {Model} failed: {Message}", id, ex.Message);
                downloads[id] = new DownloadState(0, c.Size, ex is OperationCanceledException ? "Cancelled." : ex.Message, true);
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

    async Task Download(CatalogModel c, CancellationToken ct)
    {
        var target = Path.Combine(Dir, c.Id + ".onnx");
        var part = target + ".part";
        try
        {
            using var res = await http.GetAsync(c.Url, HttpCompletionOption.ResponseHeadersRead, ct);
            if (!res.IsSuccessStatusCode) throw new ApiException($"The server answered {(int)res.StatusCode} {res.ReasonPhrase}.");
            var total = res.Content.Headers.ContentLength ?? c.Size;
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
                    downloads[c.Id] = new DownloadState(got, total, null, false);
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
        if (File.Exists(path)) File.Delete(path);
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
