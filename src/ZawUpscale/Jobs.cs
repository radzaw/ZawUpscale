using System.Diagnostics;
using System.Globalization;
using System.Threading.Channels;

namespace ZawUpscale;

/// <summary>How to upscale and where to put the results.</summary>
public sealed record UpscaleSettings
{
    public string Model { get; init; } = "realesr-general-x4v3";
    public double Scale { get; init; } = 4;
    /// <summary>"auto" (keep the input's format), "png", "jpg" or "webp".</summary>
    public string Format { get; init; } = "auto";
    public int Quality { get; init; } = 92;
    /// <summary>Empty: next to each input image (dropped images go to Pictures/ZawUpscale).</summary>
    public string? OutputDir { get; init; }
    /// <summary>File name without extension; {name}, {scale} and {model} are replaced.</summary>
    public string NameTemplate { get; init; } = "{name}_x{scale}";
    /// <summary>When the output exists: "skip", "overwrite" or "rename".</summary>
    public string Existing { get; init; } = "skip";
    /// <summary>Tile size in pixels; 0 picks one for the device.</summary>
    public int Tile { get; init; }
    public string Device { get; init; } = "auto";
}

public sealed record JobItemRequest(string Path, string? Rel = null);
public sealed record JobRequest(List<JobItemRequest> Items, UpscaleSettings Settings);

public sealed class JobItem
{
    public required int Index { get; init; }
    public required string Input { get; init; }
    public string? Output { get; set; }
    /// <summary>queued, running, done, skipped, failed or cancelled.</summary>
    public string Status { get; set; } = "queued";
    public string? Error { get; set; }
    public double Progress { get; set; }
    public long? Ms { get; set; }
    public int? InWidth { get; set; }
    public int? InHeight { get; set; }
    public int? OutWidth { get; set; }
    public int? OutHeight { get; set; }
    public JobItem Copy() => (JobItem)MemberwiseClone();
}

public sealed class Job
{
    public required string Id { get; init; }
    public required UpscaleSettings Settings { get; init; }
    public required List<JobItem> Items { get; init; }
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
    /// <summary>queued, running, done or cancelled.</summary>
    public string Status { get; set; } = "queued";
    public string? Device { get; set; }
    public string? Notice { get; set; }
    public long? Started { get; set; }
    public long? Finished { get; set; }
    internal readonly CancellationTokenSource Cts = new();
}

/// <summary>
/// Runs upscale jobs one after another on a background thread (one GPU, one job). Jobs and their items
/// are mutated only under the lock; the API returns copies.
/// </summary>
public sealed class Jobs : IDisposable
{
    const int KeepFinished = 30;
    readonly Lock gate = new();
    readonly List<Job> jobs = [];
    readonly Channel<Job> queue = Channel.CreateUnbounded<Job>();
    readonly Engine engine;
    readonly ModelStore models;
    readonly ILogger<Jobs> log;
    readonly string inbox;
    readonly CancellationTokenSource stopping = new();

    public Jobs(Engine engine, ModelStore models, AppOptions opts, ILogger<Jobs> log)
    {
        this.engine = engine;
        this.models = models;
        this.log = log;
        inbox = Path.Combine(opts.ConfigDir, "inbox");
        // A dedicated thread: inference is long, synchronous and CPU or driver bound.
        new Thread(Loop) { IsBackground = true, Name = "Upscaler" }.Start();
    }

    /// <summary>Where results of dropped or pasted images go by default.</summary>
    public static string DefaultOutputDir()
    {
        var pics = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
        if (string.IsNullOrEmpty(pics)) pics = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(pics, "ZawUpscale");
    }

    public Job Start(JobRequest r)
    {
        var s = Validate(r.Settings);
        if (r.Items is not { Count: > 0 }) throw new ApiException("There are no images to upscale.");
        if (!models.IsInstalled(s.Model)) throw new ApiException("The model isn't installed. Download it in Models first.");
        var job = new Job
        {
            Id = Guid.NewGuid().ToString("n")[..12],
            Settings = s,
            Items = r.Items.Select((it, i) => new JobItem { Index = i, Input = Path.GetFullPath(it.Path), Output = OutputPath(it, s) }).ToList(),
        };
        lock (gate)
        {
            jobs.Add(job);
            // Forget old finished jobs.
            var finished = jobs.Where(j => j.Status is "done" or "cancelled").ToList();
            foreach (var j in finished.Take(Math.Max(0, finished.Count - KeepFinished))) jobs.Remove(j);
        }
        queue.Writer.TryWrite(job);
        return Snapshot(job);
    }

    internal static UpscaleSettings Validate(UpscaleSettings s)
    {
        if (s.Scale is < 1 or > 32 || double.IsNaN(s.Scale)) throw new ApiException("The scale must be between 1 and 32.");
        var format = (s.Format ?? "auto").ToLowerInvariant();
        if (format == "jpeg") format = "jpg";
        if (format != "auto" && !ImageIO.OutputFormats.Contains(format)) throw new ApiException($"Unknown output format \"{s.Format}\".");
        var template = string.IsNullOrWhiteSpace(s.NameTemplate) ? "{name}_x{scale}" : s.NameTemplate.Trim();
        if (template.IndexOfAny(['/', '\\', ':', '*', '?', '"', '<', '>', '|']) >= 0)
            throw new ApiException("The file name pattern can't contain / \\ : * ? \" < > |.");
        if (s.Existing is not ("skip" or "overwrite" or "rename")) throw new ApiException($"Unknown option \"{s.Existing}\" for existing files.");
        if (s.Tile is < 0 or > 4096) throw new ApiException("The tile size must be between 32 and 4096 pixels (or automatic).");
        var dir = string.IsNullOrWhiteSpace(s.OutputDir) ? null : Path.GetFullPath(s.OutputDir.Trim().Trim('"'));
        return s with { Format = format, NameTemplate = template, OutputDir = dir, Quality = Math.Clamp(s.Quality, 1, 100), Device = string.IsNullOrEmpty(s.Device) ? "auto" : s.Device };
    }

    public static string FormatScale(double scale) => scale.ToString("0.##", CultureInfo.InvariantCulture);

    /// <summary>Output format for an input file: the chosen one, or the input's own (PNG for formats we can't write).</summary>
    public static string FormatFor(string input, string format)
    {
        if (format != "auto") return format;
        return Path.GetExtension(input).ToLowerInvariant() switch
        {
            ".jpg" or ".jpeg" or ".jfif" => "jpg",
            ".webp" => "webp",
            _ => "png",
        };
    }

    /// <summary>The planned output path (before the existing-file rule is applied).</summary>
    internal string OutputPath(JobItemRequest it, UpscaleSettings s)
    {
        var input = Path.GetFullPath(it.Path);
        string dir;
        if (s.OutputDir != null)
        {
            // Keep the folder structure of a recursive batch.
            var relDir = string.IsNullOrEmpty(it.Rel) ? "" : Path.GetDirectoryName(it.Rel) ?? "";
            if (relDir.Contains("..")) relDir = "";
            dir = Path.Combine(s.OutputDir, relDir);
        }
        else if (IsUnder(input, inbox)) dir = DefaultOutputDir();
        else dir = Path.GetDirectoryName(input)!;
        var name = s.NameTemplate
            .Replace("{name}", Path.GetFileNameWithoutExtension(input), StringComparison.OrdinalIgnoreCase)
            .Replace("{scale}", FormatScale(s.Scale), StringComparison.OrdinalIgnoreCase)
            .Replace("{model}", s.Model, StringComparison.OrdinalIgnoreCase);
        var ext = FormatFor(input, s.Format) switch { "jpg" => ".jpg", "webp" => ".webp", _ => ".png" };
        return Path.Combine(dir, name + ext);
    }

    static bool IsUnder(string path, string dir) =>
        path.StartsWith(Path.TrimEndingDirectorySeparator(dir) + Path.DirectorySeparatorChar, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);

    static string Unique(string path)
    {
        if (!File.Exists(path)) return path;
        var dir = Path.GetDirectoryName(path)!;
        var stem = Path.GetFileNameWithoutExtension(path);
        var ext = Path.GetExtension(path);
        for (var i = 2; ; i++)
        {
            var p = Path.Combine(dir, $"{stem} ({i}){ext}");
            if (!File.Exists(p)) return p;
        }
    }

    public List<Job> List()
    {
        lock (gate) return jobs.Select(Snapshot).ToList();
    }

    public Job? Get(string id)
    {
        lock (gate)
        {
            var j = jobs.FirstOrDefault(x => x.Id == id);
            return j == null ? null : Snapshot(j);
        }
    }

    public bool Cancel(string id)
    {
        lock (gate)
        {
            var j = jobs.FirstOrDefault(x => x.Id == id);
            if (j == null || j.Status is "done" or "cancelled") return false;
            j.Cts.Cancel();
            if (j.Status == "queued") Finish(j, "cancelled");
            return true;
        }
    }

    /// <summary>A copy that's safe to serialize while the job runs.</summary>
    static Job Snapshot(Job j) => new()
    {
        Id = j.Id, Settings = j.Settings, Items = j.Items.Select(i => i.Copy()).ToList(), Created = j.Created,
        Status = j.Status, Device = j.Device, Notice = j.Notice, Started = j.Started, Finished = j.Finished,
    };

    static long Now() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

    void Finish(Job j, string status)
    {
        j.Status = status;
        j.Finished = Now();
        foreach (var it in j.Items.Where(i => i.Status is "queued" or "running"))
            it.Status = "cancelled";
    }

    void Loop()
    {
        try
        {
            while (true)
            {
                var job = queue.Reader.ReadAsync(stopping.Token).AsTask().GetAwaiter().GetResult();
                lock (gate)
                {
                    if (job.Status != "queued") continue;
                    job.Status = "running";
                    job.Started = Now();
                }
                try
                {
                    Run(job);
                }
                catch (Exception ex)
                {
                    log.LogError(ex, "Job {Id} failed", job.Id);
                }
                lock (gate) Finish(job, job.Cts.IsCancellationRequested ? "cancelled" : "done");
            }
        }
        catch (Exception ex) when (ex is OperationCanceledException or ChannelClosedException) { /* shutting down */ }
    }

    void Run(Job job)
    {
        var s = job.Settings;
        var ct = job.Cts.Token;
        var modelPath = models.PathOf(s.Model);
        foreach (var item in job.Items)
        {
            if (ct.IsCancellationRequested) return;
            var sw = Stopwatch.StartNew();
            lock (gate) item.Status = "running";
            try
            {
                var output = item.Output!;
                if (string.Equals(Path.GetFullPath(output), item.Input, StringComparison.OrdinalIgnoreCase))
                    throw new ApiException("The result would replace the original image. Change the file name pattern or the output folder.");
                if (File.Exists(output))
                {
                    if (s.Existing == "skip")
                    {
                        lock (gate) { item.Status = "skipped"; item.Error = "The output file already exists."; }
                        continue;
                    }
                    if (s.Existing == "rename") output = Unique(output);
                }

                var src = ImageIO.Load(item.Input);
                lock (gate) { item.InWidth = src.Width; item.InHeight = src.Height; }
                var progress = new Progress(p => { lock (gate) item.Progress = p; });
                var result = engine.Upscale(src, modelPath, s.Device, s.Scale, s.Tile, progress,
                    notice => { lock (gate) job.Notice = notice; }, ct);
                lock (gate) job.Device = engine.ActiveDevice;
                ImageIO.Save(result, output, FormatFor(item.Input, s.Format), s.Quality);
                lock (gate)
                {
                    item.Output = output;
                    item.OutWidth = result.Width;
                    item.OutHeight = result.Height;
                    item.Progress = 1;
                    item.Status = "done";
                    item.Ms = sw.ElapsedMilliseconds;
                }
            }
            catch (OperationCanceledException)
            {
                lock (gate) item.Status = "cancelled";
                return;
            }
            catch (Exception ex) when (ex is ApiException or IOException or UnauthorizedAccessException or ArgumentException or Microsoft.ML.OnnxRuntime.OnnxRuntimeException or OutOfMemoryException)
            {
                log.LogWarning("Upscaling {Input} failed: {Message}", item.Input, ex.Message);
                lock (gate)
                {
                    item.Status = "failed";
                    item.Error = ex is OutOfMemoryException ? "Not enough memory for an image this large." : ex.Message;
                    item.Ms = sw.ElapsedMilliseconds;
                }
            }
        }
    }

    /// <summary>Synchronous progress (System.Progress posts to the thread pool, which would reorder reports).</summary>
    sealed class Progress(Action<double> report) : IProgress<double>
    {
        public void Report(double value) => report(value);
    }

    public void Dispose()
    {
        stopping.Cancel();
        lock (gate) foreach (var j in jobs) j.Cts.Cancel();
    }
}
