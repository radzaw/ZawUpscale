using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace ZawUpscale;

/// <summary>
/// What the last update check found: the newest release (<see cref="Latest"/>, its page and the download for this
/// platform), whether it's <see cref="Newer"/> than the running version, and when it was checked.
/// </summary>
public sealed record UpdateInfo(string Current, string? Latest, bool Newer, string? Url, string? Download, string? Notes,
    DateTimeOffset? Published, DateTimeOffset Checked, string? Error);

/// <summary>
/// Checks GitHub for a newer release. Only the version is compared: nothing is downloaded or installed, and the request
/// carries nothing but the app's name and version. The last result is kept in the configuration directory, so the
/// automatic check runs at most once a day however often the app starts.
/// </summary>
public sealed class Updates
{
    public const string Repo = "radzaw/ZawUpscale";
    public const string ReleasesPage = $"https://github.com/{Repo}/releases";
    /// <summary>How long a check is good for, unless one is asked for.</summary>
    public static readonly TimeSpan MaxAge = TimeSpan.FromHours(24);

    readonly string file;
    readonly string apiUrl;
    readonly HttpClient http;
    readonly SemaphoreSlim gate = new(1, 1);

    public Updates(AppOptions o) : this(o.ConfigDir, null) { }

    /// <param name="handler">For tests: answers instead of GitHub.</param>
    public Updates(string configDir, HttpMessageHandler? handler, string? apiUrl = null)
    {
        file = Path.Combine(configDir, "update.json");
        // ZAWUPSCALE_UPDATE_URL points a build at another releases API (a fork, or a local test server).
        var custom = Environment.GetEnvironmentVariable("ZAWUPSCALE_UPDATE_URL");
        this.apiUrl = apiUrl ?? (string.IsNullOrEmpty(custom) ? $"https://api.github.com/repos/{Repo}/releases/latest" : custom);
        http = handler == null ? new HttpClient() : new HttpClient(handler);
        http.Timeout = TimeSpan.FromSeconds(15);
        http.DefaultRequestHeaders.UserAgent.ParseAdd($"ZawUpscale/{Api.Version}");
        http.DefaultRequestHeaders.Accept.ParseAdd("application/vnd.github+json");
    }

    /// <summary>The release asset for this build, as CI names them: zawupscale-{rid}[-cuda][.exe].</summary>
    public static string AssetName =>
        "zawupscale-" + RuntimeInformation.RuntimeIdentifier + (Engine.GpuBackend == "CUDA" ? "-cuda" : "") + (OperatingSystem.IsWindows() ? ".exe" : "");

    /// <summary>The last check, if it's still fresh (and was made by this version); otherwise a new one.</summary>
    public async Task<UpdateInfo> Check(bool force = false, CancellationToken ct = default)
    {
        await gate.WaitAsync(ct);
        try
        {
            var last = Last();
            if (!force && last != null && last.Error == null && last.Current == Api.Version && DateTimeOffset.UtcNow - last.Checked < MaxAge)
                return last;
            var info = await Fetch(ct);
            // A failed check isn't worth keeping over a good one: the next start tries again.
            if (info.Error == null) Save(info);
            return info;
        }
        finally { gate.Release(); }
    }

    /// <summary>The last stored result, or null.</summary>
    public UpdateInfo? Last()
    {
        try { return File.Exists(file) ? JsonSerializer.Deserialize<UpdateInfo>(File.ReadAllText(file), Workspace.Json) : null; }
        catch (Exception ex) when (ex is JsonException or IOException) { return null; }
    }

    async Task<UpdateInfo> Fetch(CancellationToken ct)
    {
        var now = DateTimeOffset.UtcNow;
        try
        {
            using var res = await http.GetAsync(apiUrl, ct);
            // No releases yet.
            if (res.StatusCode == System.Net.HttpStatusCode.NotFound) return new(Api.Version, null, false, ReleasesPage, null, null, null, now, null);
            if (!res.IsSuccessStatusCode) return Failed($"GitHub answered {(int)res.StatusCode} {res.ReasonPhrase}.");
            using var doc = JsonDocument.Parse(await res.Content.ReadAsStringAsync(ct));
            var r = doc.RootElement;
            var tag = Str(r, "tag_name");
            if (tag == null) return Failed("GitHub's answer had no version in it.");
            var latest = tag.TrimStart('v', 'V');
            string? download = null;
            if (r.TryGetProperty("assets", out var assets) && assets.ValueKind == JsonValueKind.Array)
                foreach (var a in assets.EnumerateArray())
                    if (string.Equals(Str(a, "name"), AssetName, StringComparison.OrdinalIgnoreCase)) download = Str(a, "browser_download_url");
            DateTimeOffset? published = DateTimeOffset.TryParse(Str(r, "published_at"), out var p) ? p : null;
            return new(Api.Version, latest, Compare(latest, Api.Version) > 0, Safe(Str(r, "html_url")) ?? ReleasesPage, Safe(download),
                Str(r, "body"), published, now, null);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or JsonException && !ct.IsCancellationRequested)
        {
            return Failed(ex is TaskCanceledException ? "GitHub didn't answer in time." : "Could not reach GitHub: " + ex.Message);
        }

        UpdateInfo Failed(string error) => new(Api.Version, null, false, ReleasesPage, null, null, null, now, error);
    }

    void Save(UpdateInfo info)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(file)!);
            var tmp = file + ".tmp";
            File.WriteAllText(tmp, JsonSerializer.Serialize(info, Workspace.Json));
            File.Move(tmp, file, overwrite: true);
        }
        catch (IOException) { /* only a cache */ }
    }

    static string? Str(JsonElement e, string name) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    /// <summary>Only links to this project's GitHub pages are passed on (they're opened in the browser).</summary>
    static string? Safe(string? url) => IsProjectUrl(url) ? url : null;

    public static bool IsProjectUrl(string? url) =>
        url != null && url.StartsWith($"https://github.com/{Repo}/", StringComparison.OrdinalIgnoreCase) && Uri.TryCreate(url, UriKind.Absolute, out _);

    /// <summary>Opens one of this project's GitHub pages in the default browser.</summary>
    public static void Open(string url)
    {
        if (!IsProjectUrl(url)) throw new ApiException("Only ZawUpscale's own GitHub pages can be opened.");
        if (OperatingSystem.IsWindows()) Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
        else Process.Start(OperatingSystem.IsMacOS() ? "open" : "xdg-open", url);
    }

    /// <summary>
    /// Compares versions like 1.2.10 and 1.3.0-beta.1: numbers part by part, and a release is newer than its own
    /// pre-releases, as in semantic versioning. Unreadable numbers count as 0.
    /// </summary>
    public static int Compare(string a, string b)
    {
        static (int[] nums, string pre) Parse(string v)
        {
            v = v.Trim().TrimStart('v', 'V').Split('+')[0];
            var dash = v.IndexOf('-');
            var pre = dash < 0 ? "" : v[(dash + 1)..];
            var nums = (dash < 0 ? v : v[..dash]).Split('.').Select(x => int.TryParse(x, out var n) ? n : 0).ToArray();
            return (nums, pre);
        }
        var (na, pa) = Parse(a);
        var (nb, pb) = Parse(b);
        for (var i = 0; i < Math.Max(na.Length, nb.Length); i++)
        {
            var c = (i < na.Length ? na[i] : 0).CompareTo(i < nb.Length ? nb[i] : 0);
            if (c != 0) return c;
        }
        if (pa == pb) return 0;
        if (pa == "") return 1;
        if (pb == "") return -1;
        // Pre-release parts: numbers by value (beta.10 after beta.9), words alphabetically, numbers before words.
        var xa = pa.Split('.');
        var xb = pb.Split('.');
        for (var i = 0; i < Math.Min(xa.Length, xb.Length); i++)
        {
            var an = int.TryParse(xa[i], out var ia);
            var bn = int.TryParse(xb[i], out var ib);
            var c = an && bn ? ia.CompareTo(ib) : an ? -1 : bn ? 1 : string.CompareOrdinal(xa[i], xb[i]);
            if (c != 0) return c;
        }
        return xa.Length.CompareTo(xb.Length);
    }
}
