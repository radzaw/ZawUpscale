using System.Diagnostics;
using System.Globalization;
using System.Threading.Channels;

namespace ZawUpscale;

/// <summary>How to upscale and where to put the results.</summary>
public sealed record UpscaleSettings
{
    public string Model { get; init; } = "realesr-general-x4v3";
    public double Scale { get; init; } = 4;
    /// <summary>
    /// "scale" (by <see cref="Scale"/>), or a target size: "width", "height", "fit" (inside <see cref="Width"/> ×
    /// <see cref="Height"/>, the box turned to match each image's orientation) or "long" (the long edge <see cref="LongEdge"/>).
    /// </summary>
    public string Size { get; init; } = "scale";
    public int Width { get; init; }
    public int Height { get; init; }
    public int LongEdge { get; init; }
    /// <summary>"auto" (keep the input's format), "png", "jpg", "webp", "avif", "tif" or "jxl".</summary>
    public string Format { get; init; } = "auto";
    public int Quality { get; init; } = 92;
    /// <summary>Empty: a "{model}_x{scale}" subfolder of the opened folder (dropped images go to Pictures/ZawUpscale).</summary>
    public string? OutputDir { get; init; }
    /// <summary>File name without extension; {name}, {scale} and {model} are replaced.</summary>
    public string NameTemplate { get; init; } = "{name}";
    /// <summary>When the output exists: "skip", "overwrite" or "rename".</summary>
    public string Existing { get; init; } = "skip";
    /// <summary>Tile size in pixels; 0 picks one for the device.</summary>
    public int Tile { get; init; }
    public string Device { get; init; } = "auto";
    /// <summary>Use a model's float16 version on the GPU, when it's downloaded.</summary>
    public bool Half { get; init; }
    /// <summary>Options for SUPIR models (ignored by the others).</summary>
    public SupirSettings? Supir { get; init; }
    /// <summary>The face restoration model to run after upscaling ("gfpgan-v1.4", "codeformer"), or empty for none. Not used with SUPIR.</summary>
    public string? Faces { get; init; }
    /// <summary>How much of the restored faces to blend in, 0–1.</summary>
    public double FaceStrength { get; init; } = 0.8;
    /// <summary>How much of the AI's detail to keep, 0–1, against a plain enlargement of the original.</summary>
    public double Detail { get; init; } = 1;
    /// <summary>Sharpening (up to 1) or softening (down to −1) of the result.</summary>
    public double Sharpen { get; init; }
    /// <summary>Give the result the original's colors and tones.</summary>
    public bool KeepColors { get; init; }
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
    /// <summary>Faces restored, when face restoration is on (for an animation, the most in one frame).</summary>
    public int? Faces { get; set; }
    /// <summary>Frames of an animated GIF or WebP.</summary>
    public int? Frames { get; set; }
    /// <summary>While running: null while upscaling, "faces" while restoring faces.</summary>
    public string? Step { get; set; }
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
    /// <summary>True while paused: the running image stops at its next tile, and nothing new starts.</summary>
    public bool Paused { get; set; }
    /// <summary>Time spent working so far, without pauses.</summary>
    public long ActiveMs { get; set; }
    public string? Device { get; set; }
    public string? Notice { get; set; }
    public long? Started { get; set; }
    public long? Finished { get; set; }
    internal readonly CancellationTokenSource Cts = new();
    internal readonly ManualResetEventSlim Running = new(true);
    internal long PausedMs;
    internal long PausedSince;
}

/// <summary>
/// Runs upscale jobs one after another on a background thread (one GPU, one job). Jobs and their items
/// are mutated only under the lock; the API returns copies.
/// </summary>
public sealed class Jobs : IDisposable
{
    const int KeepFinished = 30;
    /// <summary>All frames of an animation are kept in memory: at most this many output pixels in total (1.2 GB).</summary>
    const long MaxAnimationPixels = 300_000_000;
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
        if (ModelStore.IsFaceModel(s.Model)) throw new ApiException($"{s.Model} restores faces and can't upscale on its own: use it for face restoration (Restore faces, or --faces) with an upscaling model.");
        if (!models.IsInstalled(s.Model)) throw new ApiException("The model isn't installed. Download it in Models first.");
        if (s.Faces != null && !models.IsInstalled(s.Faces)) throw new ApiException("The face restoration model isn't installed. Download it in Models first.");
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
        var size = string.IsNullOrEmpty(s.Size) ? "scale" : s.Size.ToLowerInvariant();
        static bool Px(int v) => v is >= 1 and <= 65535;
        switch (size)
        {
            case "scale":
                if (s.Scale is < 1 or > 32 || double.IsNaN(s.Scale)) throw new ApiException("The scale must be between 1 and 32.");
                break;
            case "width" when !Px(s.Width):
            case "height" when !Px(s.Height):
            case "fit" when !Px(s.Width) || !Px(s.Height):
            case "long" when !Px(s.LongEdge):
                throw new ApiException("The target size must be between 1 and 65535 pixels.");
            case "width" or "height" or "fit" or "long":
                break;
            default:
                throw new ApiException($"Unknown size mode \"{s.Size}\"; use scale, width, height, fit or long.");
        }
        var format = (s.Format ?? "auto").ToLowerInvariant();
        format = format switch { "jpeg" => "jpg", "tiff" => "tif", _ => format };
        if (format != "auto" && !ImageIO.OutputFormats.Contains(format)) throw new ApiException($"Unknown output format \"{s.Format}\".");
        var template = string.IsNullOrWhiteSpace(s.NameTemplate) ? "{name}" : s.NameTemplate.Trim();
        if (template.IndexOfAny(['/', '\\', ':', '*', '?', '"', '<', '>', '|']) >= 0)
            throw new ApiException("The file name pattern can't contain / \\ : * ? \" < > |.");
        if (s.Existing is not ("skip" or "overwrite" or "rename")) throw new ApiException($"Unknown option \"{s.Existing}\" for existing files.");
        if (s.Tile is < 0 or > 4096) throw new ApiException("The tile size must be between 32 and 4096 pixels (or automatic).");
        SupirPackage.Validate(s.Supir);
        var faces = string.IsNullOrWhiteSpace(s.Faces) || s.Faces == "none" ? null : s.Faces.Trim();
        if (faces != null && !ModelStore.IsFaceModel(faces))
            throw new ApiException($"Unknown face restoration model \"{faces}\"; use one of {string.Join(", ", ModelStore.Catalog.Where(c => c.Kind == "face").Select(c => c.Id))}.");
        if (s.FaceStrength is < 0 or > 1 || double.IsNaN(s.FaceStrength)) throw new ApiException("The face restoration strength must be between 0 and 1.");
        if (s.Detail is < 0 or > 1 || double.IsNaN(s.Detail)) throw new ApiException("The AI detail must be between 0 and 1.");
        if (s.Sharpen is < -1 or > 1 || double.IsNaN(s.Sharpen)) throw new ApiException("Sharpening must be between -1 (soften) and 1.");
        var dir = string.IsNullOrWhiteSpace(s.OutputDir) ? null : Path.GetFullPath(s.OutputDir.Trim().Trim('"'));
        return s with
        {
            Size = size, Format = format, NameTemplate = template, OutputDir = dir, Quality = Math.Clamp(s.Quality, 1, 100),
            Device = string.IsNullOrEmpty(s.Device) ? "auto" : s.Device, Faces = faces,
        };
    }

    public static string FormatScale(double scale) => scale.ToString("0.##", CultureInfo.InvariantCulture);

    /// <summary>The size setting in short: "x4", or for a target size "w3840", "h2160", "3840x2160" or "long4000".</summary>
    public static string SizeLabel(UpscaleSettings s) => s.Size switch
    {
        "width" => $"w{s.Width}",
        "height" => $"h{s.Height}",
        "fit" => $"{s.Width}x{s.Height}",
        "long" => $"long{s.LongEdge}",
        _ => $"x{FormatScale(s.Scale)}",
    };

    /// <summary>What {scale} becomes in file names: the factor ("4"), or the target size's label.</summary>
    static string ScaleToken(UpscaleSettings s) => s.Size == "scale" ? FormatScale(s.Scale) : SizeLabel(s);

    /// <summary>
    /// The subfolder results go in when no output folder is chosen, e.g. "realesr-general-x4v3_x4", or
    /// "realesr-general-x4v3_x4_faces" with face restoration, so turning it on doesn't skip the earlier results.
    /// </summary>
    public static string ResultsFolder(string model, double scale, bool faces = false) => ResultsFolder(model, $"x{FormatScale(scale)}", faces);

    public static string ResultsFolder(string model, string sizeLabel, bool faces) => $"{model}_{sizeLabel}{(faces ? "_faces" : "")}";

    /// <summary>The scale that gives an image of <paramref name="w"/> × <paramref name="h"/> the chosen size.</summary>
    public static double ScaleFor(UpscaleSettings s, int w, int h)
    {
        switch (s.Size)
        {
            case "width": return (double)s.Width / w;
            case "height": return (double)s.Height / h;
            case "long": return (double)s.LongEdge / Math.Max(w, h);
            case "fit":
                // The box turns with the image: 3840 × 2160 fits a portrait photo into 2160 × 3840.
                var (bw, bh) = (w >= h) == (s.Width >= s.Height) ? (s.Width, s.Height) : (s.Height, s.Width);
                return Math.Min((double)bw / w, (double)bh / h);
            default: return s.Scale;
        }
    }

    /// <summary>
    /// Output format for an input file: the chosen one, or the input's own. HEIC can't be written, so it becomes JPEG;
    /// an animated GIF stays a GIF, other formats without an encoder become PNG.
    /// </summary>
    public static string FormatFor(string input, string format, bool animated = false)
    {
        if (format != "auto") return format;
        return Path.GetExtension(input).ToLowerInvariant() switch
        {
            ".jpg" or ".jpeg" or ".jfif" or ".heic" or ".heif" => "jpg",
            ".webp" => "webp",
            ".avif" => "avif",
            ".tif" or ".tiff" => "tif",
            ".jxl" => "jxl",
            ".gif" when animated => "gif",
            _ => "png",
        };
    }

    /// <summary>The planned output path (before the existing-file rule is applied).</summary>
    internal string OutputPath(JobItemRequest it, UpscaleSettings s)
    {
        var input = Path.GetFullPath(it.Path);
        var inputDir = Path.GetDirectoryName(input)!;
        // Keep the folder structure of a recursive batch.
        var relDir = string.IsNullOrEmpty(it.Rel) ? "" : Path.GetDirectoryName(it.Rel) ?? "";
        if (relDir.Contains("..")) relDir = "";
        string dir;
        if (s.OutputDir != null) dir = Path.Combine(s.OutputDir, relDir);
        else if (IsUnder(input, inbox)) dir = DefaultOutputDir();
        else
        {
            // A subfolder of the folder that was opened, named after the model and size.
            var suffix = relDir == "" ? "" : Path.DirectorySeparatorChar + relDir;
            var root = suffix != "" && inputDir.EndsWith(suffix, StringComparison.OrdinalIgnoreCase) ? inputDir[..^suffix.Length] : inputDir;
            dir = Path.Combine(root, ResultsFolder(s.Model, SizeLabel(s), FacesOn(s)), root == inputDir ? "" : relDir);
        }
        var name = s.NameTemplate
            .Replace("{name}", Path.GetFileNameWithoutExtension(input), StringComparison.OrdinalIgnoreCase)
            .Replace("{scale}", ScaleToken(s), StringComparison.OrdinalIgnoreCase)
            .Replace("{model}", s.Model, StringComparison.OrdinalIgnoreCase);
        var animated = s.Format == "auto" && ImageIO.FrameCount(input) > 1;
        return Path.Combine(dir, name + ImageIO.ExtensionOf(FormatFor(input, s.Format, animated)));
    }

    /// <summary>Whether faces are restored: when chosen, except with SUPIR, which rebuilds faces itself.</summary>
    internal bool FacesOn(UpscaleSettings s) => s.Faces != null && !SupirPackage.IsPackage(models.PathOf(s.Model));

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

    /// <summary>Pauses a queued or running job: the running image stops at its next tile until it's resumed.</summary>
    public bool Pause(string id)
    {
        lock (gate)
        {
            var j = jobs.FirstOrDefault(x => x.Id == id);
            if (j == null || j.Status is "done" or "cancelled" || j.Paused) return false;
            j.Paused = true;
            j.PausedSince = Now();
            j.Running.Reset();
            return true;
        }
    }

    public bool Resume(string id)
    {
        lock (gate)
        {
            var j = jobs.FirstOrDefault(x => x.Id == id);
            if (j == null || !j.Paused) return false;
            Unpause(j);
            return true;
        }
    }

    static void Unpause(Job j)
    {
        j.PausedMs += Now() - j.PausedSince;
        j.Paused = false;
        j.Running.Set();
    }

    /// <summary>Blocks the upscaler thread while the job is paused (cancelling still works).</summary>
    static void WaitWhilePaused(Job j, CancellationToken ct) => j.Running.Wait(ct);

    /// <summary>A copy that's safe to serialize while the job runs.</summary>
    static Job Snapshot(Job j)
    {
        var end = j.Finished ?? Now();
        var paused = j.PausedMs + (j.Paused ? end - j.PausedSince : 0);
        return new()
        {
            Id = j.Id, Settings = j.Settings, Items = j.Items.Select(i => i.Copy()).ToList(), Created = j.Created,
            Status = j.Status, Paused = j.Paused, ActiveMs = j.Started is { } st ? Math.Max(0, end - st - paused) : 0,
            Device = j.Device, Notice = j.Notice, Started = j.Started, Finished = j.Finished,
        };
    }

    static long Now() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

    void Finish(Job j, string status)
    {
        if (j.Paused) Unpause(j);
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
                    if (job.Paused) job.PausedSince = job.Started.Value; // paused while waiting: count from now
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

    /// <summary>The models one job uses.</summary>
    sealed record Plan(string ModelPath, string? Restorer, string? Detector);

    void Run(Job job)
    {
        var s = job.Settings;
        var ct = job.Cts.Token;
        // The float16 version only on the GPU: on the CPU it's slower than float32.
        var plan = new Plan(models.ModelPath(s.Model, half: s.Half && s.Device != "cpu"),
            FacesOn(s) ? models.PathOf(s.Faces!) : null, FacesOn(s) ? models.PathOf(ModelStore.FaceDetectorId) : null);
        foreach (var item in job.Items)
        {
            try { WaitWhilePaused(job, ct); }
            catch (OperationCanceledException) { return; }
            if (ct.IsCancellationRequested) return;
            var sw = Stopwatch.StartNew();
            long pausedBefore;
            lock (gate) { item.Status = "running"; pausedBefore = job.PausedMs; }
            // The image's time without pauses (it can only be paused, and resumed, while it runs).
            long Elapsed() { lock (gate) return Math.Max(0, sw.ElapsedMilliseconds - (job.PausedMs - pausedBefore)); }
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

                var animated = ImageIO.FrameCount(item.Input) > 1;
                var format = FormatFor(item.Input, s.Format, animated);
                RgbaImage result;
                if (animated && format is "gif" or "webp") result = RunAnimation(job, item, plan, output, format, ct);
                else
                {
                    var src = ImageIO.Load(item.Input);
                    lock (gate) { item.InWidth = src.Width; item.InHeight = src.Height; }
                    result = Process(job, item, plan, src, p => { lock (gate) item.Progress = p; }, ct);
                    ImageIO.Save(result, output, format, s.Quality);
                }
                lock (gate)
                {
                    item.Output = output;
                    item.OutWidth = result.Width;
                    item.OutHeight = result.Height;
                    item.Progress = 1;
                    item.Status = "done";
                    item.Ms = Elapsed();
                }
            }
            catch (OperationCanceledException)
            {
                lock (gate) item.Status = "cancelled";
                return;
            }
            catch (Exception ex) when (ex is ApiException or IOException or UnauthorizedAccessException or ArgumentException or Microsoft.ML.OnnxRuntime.OnnxRuntimeException or OutOfMemoryException or ImageMagick.MagickException)
            {
                log.LogWarning("Upscaling {Input} failed: {Message}", item.Input, ex.Message);
                lock (gate)
                {
                    item.Status = "failed";
                    item.Error = ex is OutOfMemoryException ? "Not enough memory for an image this large." : ex.Message;
                    item.Ms = Elapsed();
                }
            }
        }
    }

    /// <summary>Upscales an animated GIF or WebP frame by frame and saves it. Returns the first frame.</summary>
    RgbaImage RunAnimation(Job job, JobItem item, Plan plan, string output, string format, CancellationToken ct)
    {
        var anim = ImageIO.LoadAnimation(item.Input);
        var first = anim.Frames[0];
        var (ow, oh) = Upscaler.TargetSize(first.Width, first.Height, ScaleFor(job.Settings, first.Width, first.Height));
        if ((long)ow * oh * anim.Frames.Count > MaxAnimationPixels)
            throw new ApiException($"{anim.Frames.Count} frames of {ow}×{oh} pixels are too much to hold in memory. Choose a smaller size.");
        lock (gate) { item.InWidth = first.Width; item.InHeight = first.Height; item.Frames = anim.Frames.Count; }
        var n = anim.Frames.Count;
        for (var i = 0; i < n; i++)
        {
            var f = i;
            anim.Frames[i] = Process(job, item, plan, anim.Frames[i], p => { lock (gate) item.Progress = (f + p) / n * 0.97; }, ct);
        }
        ImageIO.SaveAnimation(anim, output, format, job.Settings.Quality);
        return anim.Frames[0];
    }

    /// <summary>One image (or frame): sized, upscaled, adjusted, and its faces restored.</summary>
    RgbaImage Process(Job job, JobItem item, Plan plan, RgbaImage src, Action<double> report, CancellationToken ct)
    {
        var s = job.Settings;
        var scale = ScaleFor(s, src.Width, src.Height);
        if (scale > 32) throw new ApiException($"That size needs more than 32× for this {src.Width}×{src.Height} image.");
        if (scale < 1)
        {
            // Larger than the target already: reduce it to the target, and let the model work on that at 1×.
            var (tw, th) = Upscaler.TargetSize(src.Width, src.Height, scale);
            src = ImageIO.Resize(src, tw, th);
            scale = 1;
        }
        // Progress: the upscale, then the faces (a few network runs, quick next to the upscale).
        var share = plan.Restorer != null ? 0.85 : 1;
        var result = engine.Upscale(src, plan.ModelPath, s.Device, scale, s.Tile, new Progress(p => { WaitWhilePaused(job, ct); report(p * share); }),
            notice => { lock (gate) job.Notice = notice; }, ct, s.Supir);
        lock (gate) job.Device = engine.ActiveDevice;
        if (PostProcess.Needed(s.Detail, s.Sharpen, s.KeepColors)) PostProcess.Apply(src, result, s.Detail, s.Sharpen, s.KeepColors);
        if (plan.Restorer != null)
        {
            lock (gate) item.Step = "faces";
            var n = engine.RestoreFaces(src, result, plan.Restorer, plan.Detector!, s.Device, s.FaceStrength,
                new Progress(p => { WaitWhilePaused(job, ct); report(share + (1 - share) * p); }), ct);
            lock (gate) { item.Faces = Math.Max(item.Faces ?? 0, n); item.Step = null; }
        }
        return result;
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
