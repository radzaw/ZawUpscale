using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;

namespace ZawUpscale;

public sealed record FsEntry(string Name, string Path, bool Dir, long? Size, long? Modified);
public sealed record FsListing(string Path, string? Parent, List<FsEntry> Entries);
public sealed record Place(string Name, string Path, string Kind);
public sealed record ScanResult(List<JobItemRequest> Items, int Skipped, long Bytes, int PreviousResults = 0);

/// <summary>
/// The local file system as the UI sees it: a browser for the open dialogs (the page itself can't learn
/// local paths), folder scans, revealing results in the file manager, and an inbox for dropped images.
/// </summary>
public sealed class Files(AppOptions opts)
{
    public const int MaxScan = 20000;
    readonly string inbox = Path.Combine(opts.ConfigDir, "inbox");

    public static List<Place> Places()
    {
        var list = new List<Place>();
        void Add(string name, Environment.SpecialFolder f, string kind)
        {
            var p = Environment.GetFolderPath(f);
            if (!string.IsNullOrEmpty(p) && Directory.Exists(p) && list.All(x => x.Path != p)) list.Add(new Place(name, p, kind));
        }
        Add("Home", Environment.SpecialFolder.UserProfile, "home");
        Add("Desktop", Environment.SpecialFolder.Desktop, "desktop");
        Add("Pictures", Environment.SpecialFolder.MyPictures, "pictures");
        var downloads = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads");
        if (Directory.Exists(downloads)) list.Add(new Place("Downloads", downloads, "downloads"));
        Add("Documents", Environment.SpecialFolder.MyDocuments, "documents");
        foreach (var d in DriveInfo.GetDrives())
        {
            try
            {
                if (!d.IsReady || d.DriveType is DriveType.Ram or DriveType.NoRootDirectory) continue;
                if (!OperatingSystem.IsWindows() && d.Name != "/" && !d.Name.StartsWith("/media", StringComparison.Ordinal)
                    && !d.Name.StartsWith("/mnt", StringComparison.Ordinal) && !d.Name.StartsWith("/Volumes", StringComparison.Ordinal)) continue;
                var label = OperatingSystem.IsWindows()
                    ? (string.IsNullOrEmpty(d.VolumeLabel) ? d.Name.TrimEnd('\\') : $"{d.VolumeLabel} ({d.Name.TrimEnd('\\')})")
                    : d.Name;
                list.Add(new Place(label, d.RootDirectory.FullName, "drive"));
            }
            catch (IOException) { /* drive went away */ }
            catch (UnauthorizedAccessException) { }
        }
        return list;
    }

    /// <summary>Folders and image files (plus .onnx files when <paramref name="models"/>) in a directory.</summary>
    public static FsListing List(string path, bool models = false)
    {
        path = Path.GetFullPath(Environment.ExpandEnvironmentVariables(path.Trim().Trim('"')));
        var di = new DirectoryInfo(path);
        if (!di.Exists) throw new ApiException($"Folder not found: {path}");
        var entries = new List<FsEntry>();
        var opt = new EnumerationOptions { IgnoreInaccessible = true, AttributesToSkip = FileAttributes.System | FileAttributes.Hidden };
        foreach (var d in di.EnumerateDirectories("*", opt).OrderBy(d => d.Name, StringComparer.OrdinalIgnoreCase))
            entries.Add(new FsEntry(d.Name, d.FullName, true, null, null));
        foreach (var f in di.EnumerateFiles("*", opt).OrderBy(f => f.Name, StringComparer.OrdinalIgnoreCase))
        {
            var wanted = models ? f.Extension.Equals(".onnx", StringComparison.OrdinalIgnoreCase) : ImageIO.IsImage(f.Name);
            if (wanted) entries.Add(new FsEntry(f.Name, f.FullName, false, f.Length, new DateTimeOffset(f.LastWriteTimeUtc).ToUnixTimeMilliseconds()));
        }
        return new FsListing(di.FullName, di.Parent?.FullName, entries);
    }

    /// <summary>
    /// Expands the inputs (files and folders) into the images to upscale. Rel is the path below the folder
    /// it was found in, so a batch can keep its folder structure in the output folder.
    /// </summary>
    public static ScanResult Scan(IEnumerable<string> inputs, bool recursive, string? nameTemplate = null, IEnumerable<string>? modelIds = null)
    {
        // Results folders ("realesr-general-x4v3_x4", "…_x4_faces") found below an opened folder hold earlier results.
        var resultsFolder = modelIds?.Any() == true
            ? new Regex($"^(?:{string.Join('|', modelIds.Select(Regex.Escape))})_x[0-9]+(?:\\.[0-9]+)?(?:_faces)?$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)
            : null;
        var inResults = 0;
        var items = new List<JobItemRequest>();
        var seen = new HashSet<string>(OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
        var skipped = 0;
        var fromFolder = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        long bytes = 0;
        foreach (var raw in inputs)
        {
            var p = Path.GetFullPath(raw.Trim().Trim('"'));
            if (File.Exists(p))
            {
                if (!ImageIO.IsImage(p)) { skipped++; continue; }
                if (seen.Add(p)) { items.Add(new JobItemRequest(p, Path.GetFileName(p))); bytes += new FileInfo(p).Length; }
            }
            else if (Directory.Exists(p))
            {
                var opt = new EnumerationOptions { IgnoreInaccessible = true, RecurseSubdirectories = recursive, AttributesToSkip = FileAttributes.System | FileAttributes.Hidden };
                foreach (var f in Directory.EnumerateFiles(p, "*", opt).Order(StringComparer.OrdinalIgnoreCase))
                {
                    if (!ImageIO.IsImage(f)) continue;
                    var rel = Path.GetRelativePath(p, f);
                    if (resultsFolder != null && Path.GetDirectoryName(rel)!.Split(Path.DirectorySeparatorChar).Any(resultsFolder.IsMatch)) { inResults++; continue; }
                    if (items.Count >= MaxScan) throw new ApiException($"More than {MaxScan} images; choose a smaller folder.");
                    if (seen.Add(f)) { items.Add(new JobItemRequest(f, rel)); bytes += new FileInfo(f).Length; fromFolder.Add(f); }
                }
            }
            else throw new ApiException($"Not found: {p}");
        }
        var results = 0;
        if (PreviousResults(nameTemplate) is { } pattern)
        {
            // Leave out earlier results found in folders ("photo_x4.png" next to "photo.jpg"), so running a folder again doesn't upscale them.
            var stems = new HashSet<string>(items.Select(i => Path.Combine(Path.GetDirectoryName(i.Path)!, Path.GetFileNameWithoutExtension(i.Path))), StringComparer.OrdinalIgnoreCase);
            var kept = items.Where(i =>
            {
                var m = pattern.Match(Path.GetFileNameWithoutExtension(i.Path));
                return !(fromFolder.Contains(i.Path) && m.Success && stems.Contains(Path.Combine(Path.GetDirectoryName(i.Path)!, m.Groups["name"].Value)));
            }).ToList();
            results = items.Count - kept.Count;
            if (results > 0)
            {
                items = kept;
                bytes = items.Sum(i => new FileInfo(i.Path).Length);
            }
        }
        return new ScanResult(items, skipped, bytes, results + inResults);
    }

    /// <summary>A regex matching file names made by the output name pattern, capturing the original's name; null if the pattern has no {name}.</summary>
    internal static Regex? PreviousResults(string? nameTemplate)
    {
        var t = string.IsNullOrWhiteSpace(nameTemplate) ? "{name}" : nameTemplate.Trim();
        var parts = Regex.Split(t, @"(\{name\}|\{scale\}|\{model\})", RegexOptions.IgnoreCase);
        if (!parts.Any(x => x.Equals("{name}", StringComparison.OrdinalIgnoreCase)) || t.Equals("{name}", StringComparison.OrdinalIgnoreCase)) return null;
        var named = false;
        var sb = new StringBuilder("^");
        foreach (var part in parts)
        {
            switch (part.ToLowerInvariant())
            {
                case "{name}": sb.Append(named ? @"\k<name>" : "(?<name>.+?)"); named = true; break; // lazy: "{name}-{model}" with a dashed model id
                case "{scale}": sb.Append(@"[0-9]+(?:\.[0-9]+)?"); break;
                case "{model}": sb.Append(@"[^\\/]+"); break;
                default: sb.Append(Regex.Escape(part)); break;
            }
        }
        return new Regex(sb.Append('$').ToString(), RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
    }

    /// <summary>Saves an uploaded (dropped or pasted) image in the inbox and returns its path.</summary>
    public async Task<string> SaveUpload(IFormFile file, CancellationToken ct)
    {
        var name = Path.GetFileName(file.FileName);
        if (string.IsNullOrWhiteSpace(name) || name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0) name = "pasted.png";
        if (!ImageIO.IsImage(name)) throw new ApiException($"{name} is not a supported image ({string.Join(", ", ImageIO.InputExtensions)}).");
        var dir = Path.Combine(inbox, DateTime.Now.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("n")[..6]);
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, name);
        await using (var fs = File.Create(path)) await file.CopyToAsync(fs, ct);
        return path;
    }

    /// <summary>Removes inbox folders older than a week.</summary>
    public void CleanInbox()
    {
        if (!Directory.Exists(inbox)) return;
        foreach (var d in Directory.EnumerateDirectories(inbox))
        {
            try
            {
                if (Directory.GetLastWriteTimeUtc(d) < DateTime.UtcNow.AddDays(-7)) Directory.Delete(d, recursive: true);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* in use; next time */ }
        }
    }

    /// <summary>Shows a file (selected) or folder in the system file manager.</summary>
    public static void Reveal(string path)
    {
        path = Path.GetFullPath(path.Trim().Trim('"'));
        var isFile = File.Exists(path);
        if (!isFile && !Directory.Exists(path)) throw new ApiException($"Not found: {path}");
        if (OperatingSystem.IsWindows())
        {
            // explorer.exe parses its command line itself: "/select," must come right before the quoted path.
            Process.Start(new ProcessStartInfo("explorer.exe", isFile ? $"/select,\"{path}\"" : $"\"{path}\"") { UseShellExecute = false });
        }
        else if (OperatingSystem.IsMacOS())
        {
            Process.Start("open", isFile ? ["-R", path] : [path]);
        }
        else
        {
            Process.Start("xdg-open", isFile ? Path.GetDirectoryName(path)! : path);
        }
    }

    /// <summary>Opens a file with its default application.</summary>
    public static void Open(string path)
    {
        path = Path.GetFullPath(path.Trim().Trim('"'));
        if (!File.Exists(path)) throw new ApiException($"Not found: {path}");
        if (OperatingSystem.IsWindows()) Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
        else if (OperatingSystem.IsMacOS()) Process.Start("open", path);
        else Process.Start("xdg-open", path);
    }
}
