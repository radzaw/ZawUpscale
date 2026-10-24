using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging.Abstractions;

namespace ZawUpscale;

/// <summary>Command-line mode: upscale files and folders, list and download models, without opening a window.</summary>
public static class Cli
{
    public static readonly string[] Commands = ["upscale", "models", "download"];

    public const string HelpText = """
        Command line:
          ZawUpscale upscale <file|folder>... [options]
            -m, --model <id>      Model (default: realesr-general-x4v3; see "ZawUpscale models")
            -s, --scale <n>       Scale factor, 1-32 (default 4)
            --width <px>          Or a size: this width (the height follows)
            --height <px>         ... this height
            --fit <W>x<H>         ... inside this box, turned to match each image (e.g. 3840x2160)
            --long-edge <px>      ... this long edge
            -o, --output <dir>    Output folder (default: a {model}_x{scale} subfolder)
            -f, --format <fmt>    auto, png, jpg, webp, avif, tif or jxl (default auto: same as the input)
            -q, --quality <n>     JPEG / WebP / AVIF / JPEG XL quality, 1-100 (default 92; 100 is lossless for WebP and JPEG XL)
            -n, --name <pattern>  Output name: {name}, {scale}, {model} (default {name})
            -r, --recursive       Include subfolders
            --overwrite           Replace existing outputs (default: skip them)
            --tile <n>            Tile size in pixels (default: automatic)
            --device <id>         auto, cpu or gpu:N (default auto)
            --faces <id>          Restore faces after upscaling: gfpgan-v1.4 or codeformer
            --face-strength <n>   How much of the restored faces to blend in, 0-1 (default 0.8)
            --detail <n>          How much of the AI's detail to keep, 0-1 (default 1; lower blends in a plain enlargement)
            --sharpen <n>         Sharpen (up to 1) or soften (down to -1) the result (default 0)
            --keep-colors         Give the result the original's colors and tones
            --half                Use the model's float16 version on the GPU (download it with download --half)
          SUPIR models (see tools/supir/README.md):
            --prompt <text>       What the image shows, e.g. "a red fox kit in the grass"
            --negative <text>     Negative prompt (default: the model's)
            --steps <n>           Sampling steps, 1-200 (default 50)
            --guidance <n>        Prompt guidance (default 7.5)
            --restoration <n>     Restoration guidance: fewer invented details; 0 is off (default)
            --control <n>         How closely to follow the input, 0-2 (default 1)
            --color-fix <mode>    wavelet, adain or none (default wavelet)
            --seed <n>            Seed (default: random; it is logged)
          ZawUpscale models                 List the models
          ZawUpscale download <id> [--half] Download a model (--half: its float16 version, for GPUs)
        """;

    public static int Run(string[] args)
    {
        AttachToParentConsole();
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        try
        {
            return args[0] switch
            {
                "models" => ListModels(args),
                "download" => Download(args),
                _ => Upscale(args),
            };
        }
        catch (Exception ex) when (ex is ApiException or ArgumentException or IOException or FormatException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine("Error: " + ex.Message);
            return 1;
        }
    }

    static AppOptions Options(string[] args, List<string> rest)
    {
        // --config is shared with the app; everything else is the command's own.
        var appArgs = new List<string>();
        for (var i = 1; i < args.Length; i++)
        {
            if (args[i] == "--config" && i + 1 < args.Length) { appArgs.Add(args[i]); appArgs.Add(args[++i]); }
            else rest.Add(args[i]);
        }
        var o = AppOptions.Parse([.. appArgs]);
        Directory.CreateDirectory(o.ConfigDir);
        return o;
    }

    static int ListModels(string[] args)
    {
        var store = new ModelStore(Options(args, []), NullLogger<ModelStore>.Instance);
        foreach (var m in store.List())
            Console.WriteLine($"{(m.Installed ? "*" : " ")} {m.Id,-28} {m.Name}{(m.Installed ? (m.HalfInstalled ? (m.FullInstalled ? "  (float32 + float16)" : "  (float16)") : "") : $"  (not downloaded, {m.Size / 1048576.0:0.#} MB{(m.HalfSize is { } hs ? $"; float16 {hs / 1048576.0:0.#} MB" : "")})")}");
        Console.WriteLine();
        Console.WriteLine($"* = installed. Models folder: {store.Dir}");
        return 0;
    }

    static int Download(string[] args)
    {
        var rest = new List<string>();
        var store = new ModelStore(Options(args, rest), NullLogger<ModelStore>.Instance);
        var half = rest.Remove("--half");
        if (rest.Count == 0) throw new ApiException("Which model? See \"ZawUpscale models\".");
        foreach (var id in rest)
        {
            store.StartDownload(id, half);
            while (true)
            {
                Thread.Sleep(250);
                var d = store.List().First(m => m.Id == id).Download;
                if (d == null) continue;
                Console.Write($"\r{id}: {d.Received / 1048576.0:0.0} / {d.Total / 1048576.0:0.0} MB   ");
                if (!d.Done) continue;
                Console.WriteLine();
                if (d.Error != null) throw new ApiException(d.Error);
                break;
            }
        }
        return 0;
    }

    static int Upscale(string[] args)
    {
        var rest = new List<string>();
        var opts = Options(args, rest);
        var s = new UpscaleSettings();
        var supir = new SupirSettings();
        var inputs = new List<string>();
        var recursive = false;
        for (var i = 0; i < rest.Count; i++)
        {
            string Next() => i + 1 < rest.Count ? rest[++i] : throw new ArgumentException($"Missing value for {rest[i]}");
            switch (rest[i])
            {
                case "-m" or "--model": s = s with { Model = Next() }; break;
                case "-s" or "--scale": s = s with { Scale = double.Parse(Next().TrimEnd('x', 'X'), System.Globalization.CultureInfo.InvariantCulture) }; break;
                case "-o" or "--output": s = s with { OutputDir = Next() }; break;
                case "-f" or "--format": s = s with { Format = Next() }; break;
                case "-q" or "--quality": s = s with { Quality = int.Parse(Next()) }; break;
                case "-n" or "--name": s = s with { NameTemplate = Next() }; break;
                case "-r" or "--recursive": recursive = true; break;
                case "--overwrite": s = s with { Existing = "overwrite" }; break;
                case "--tile": s = s with { Tile = int.Parse(Next()) }; break;
                case "--device": s = s with { Device = Next() }; break;
                case "--faces": s = s with { Faces = Next() }; break;
                case "--face-strength": s = s with { FaceStrength = Number(Next()) }; break;
                case "--width": s = s with { Size = "width", Width = int.Parse(Next()) }; break;
                case "--height": s = s with { Size = "height", Height = int.Parse(Next()) }; break;
                case "--long-edge": s = s with { Size = "long", LongEdge = int.Parse(Next()) }; break;
                case "--fit":
                    var box = Next().ToLowerInvariant().Split('x', '×');
                    if (box.Length != 2) throw new ArgumentException("--fit takes a size like 3840x2160.");
                    s = s with { Size = "fit", Width = int.Parse(box[0]), Height = int.Parse(box[1]) };
                    break;
                case "--detail": s = s with { Detail = Number(Next()) }; break;
                case "--sharpen": s = s with { Sharpen = Number(Next()) }; break;
                case "--keep-colors": s = s with { KeepColors = true }; break;
                case "--half": s = s with { Half = true }; break;
                case "--prompt": supir = supir with { Prompt = Next() }; break;
                case "--negative": supir = supir with { NegativePrompt = Next() }; break;
                case "--steps": supir = supir with { Steps = int.Parse(Next()) }; break;
                case "--guidance": supir = supir with { Guidance = Number(Next()) }; break;
                case "--restoration": supir = supir with { Restoration = Number(Next()) }; break;
                case "--control": supir = supir with { Control = Number(Next()) }; break;
                case "--color-fix": supir = supir with { ColorFix = Next() }; break;
                case "--seed": supir = supir with { Seed = int.Parse(Next()) }; break;
                default:
                    if (rest[i].StartsWith('-')) throw new ArgumentException($"Unknown option {rest[i]}. See ZawUpscale --help.");
                    inputs.Add(rest[i]);
                    break;
            }
        }
        s = s with { Supir = supir };
        if (inputs.Count == 0) throw new ApiException("Nothing to upscale. Usage: ZawUpscale upscale <file|folder>... [options]");

        // Information from the engine: SUPIR logs the seed of each image.
        using var loggers = LoggerFactory.Create(b => b.AddSimpleConsole(o => o.SingleLine = true).SetMinimumLevel(LogLevel.Warning).AddFilter<Microsoft.Extensions.Logging.Console.ConsoleLoggerProvider>(typeof(Engine).FullName, LogLevel.Information));
        var store = new ModelStore(opts, loggers.CreateLogger<ModelStore>());
        using var engine = new Engine(loggers.CreateLogger<Engine>());
        using var jobs = new Jobs(engine, store, opts, loggers.CreateLogger<Jobs>());
        var scan = Files.Scan(inputs, recursive, s.NameTemplate, store.List().Select(m => m.Id));
        if (scan.Items.Count == 0) throw new ApiException("No images found.");
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; foreach (var j in jobs.List()) jobs.Cancel(j.Id); };

        var job = jobs.Start(new JobRequest(scan.Items, s));
        var total = Stopwatch.StartNew();
        var reported = new HashSet<int>();
        while (true)
        {
            Thread.Sleep(200);
            job = jobs.Get(job.Id)!;
            foreach (var it in job.Items.Where(i => i.Status is not ("queued" or "running") && reported.Add(i.Index)))
            {
                Console.Write("\r" + new string(' ', 70) + "\r");
                Console.WriteLine(it.Status switch
                {
                    "done" => $"[{it.Index + 1}/{job.Items.Count}] {it.Input} -> {it.Output} ({it.InWidth}×{it.InHeight} → {it.OutWidth}×{it.OutHeight}{(it.Frames is { } fr ? $", {fr} frames" : "")}{(it.Faces is { } f ? $", {f} face{(f == 1 ? "" : "s")}" : "")}, {it.Ms / 1000.0:0.0} s)",
                    "skipped" => $"[{it.Index + 1}/{job.Items.Count}] {it.Input}: skipped, {it.Output} exists",
                    _ => $"[{it.Index + 1}/{job.Items.Count}] {it.Input}: {it.Status}{(it.Error != null ? " - " + it.Error : "")}",
                });
            }
            if (job.Items.FirstOrDefault(i => i.Status == "running") is { } cur)
                Console.Write($"\r[{cur.Index + 1}/{job.Items.Count}] {Path.GetFileName(cur.Input)} {cur.Progress * 100:0}%   ");
            if (job.Status is "done" or "cancelled") break;
        }
        var failed = job.Items.Count(i => i.Status == "failed");
        Console.WriteLine($"{job.Items.Count(i => i.Status == "done")} upscaled, {job.Items.Count(i => i.Status == "skipped")} skipped, {failed} failed in {total.Elapsed.TotalSeconds:0.0} s on {job.Device ?? "-"}.");
        return failed > 0 || job.Status == "cancelled" ? 1 : 0;
    }

    static double Number(string s) => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture);

    [DllImport("kernel32.dll")]
    static extern bool AttachConsole(int processId);

    /// <summary>A published Windows build has no console of its own; borrow the one it was started from.</summary>
    static void AttachToParentConsole()
    {
        if (!OperatingSystem.IsWindows() || !AttachConsole(-1)) return;
        Console.SetOut(new StreamWriter(Console.OpenStandardOutput()) { AutoFlush = true });
        Console.SetError(new StreamWriter(Console.OpenStandardError()) { AutoFlush = true });
        Console.WriteLine();
    }
}
