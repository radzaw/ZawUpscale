using System.Reflection;
using System.Text.Json;
using Microsoft.AspNetCore.StaticFiles;
using Microsoft.ML.OnnxRuntime;

namespace ZawUpscale;

/// <summary>An error with a message meant for the user.</summary>
public sealed class ApiException(string message) : Exception(message);

public sealed record PathRequest(string Path);
public sealed record ScanRequest(List<string> Inputs, bool Recursive, string? NameTemplate = null);

public static class Api
{
    public static readonly string Version =
        typeof(Api).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion.Split('+')[0] ?? "0.0.0";

    sealed record Response(bool Ok, object? Data, string? Error);

    static async Task<IResult> Run(Func<Task<object?>> body)
    {
        try
        {
            return Results.Json(new Response(true, await body(), null), Workspace.Json);
        }
        catch (OperationCanceledException)
        {
            return Results.Json(new Response(false, null, "The operation was cancelled."), Workspace.Json);
        }
        catch (Exception ex) when (ex is ApiException or IOException or UnauthorizedAccessException or JsonException or ArgumentException
            or FormatException or NotSupportedException or OnnxRuntimeException)
        {
            return Results.Json(new Response(false, null, ex.Message), Workspace.Json);
        }
    }

    static Task<IResult> RunSync(Func<object?> body) => Run(() => Task.FromResult(body()));

    static readonly FileExtensionContentTypeProvider ContentTypes = new();

    public static void Map(WebApplication app)
    {
        var api = app.MapGroup("/api");

        // ---- application ----
        api.MapPost("/ping", (string? page, Heartbeat hb) => { hb.Ping(page); return Results.Ok(); });
        api.MapPost("/bye", (string? page, Heartbeat hb) => { hb.Bye(page); return Results.Ok(); });
        api.MapPost("/exit", (IHostApplicationLifetime life) =>
        {
            _ = Task.Delay(300).ContinueWith(_ => life.StopApplication());
            return RunSync(() => null);
        });
        api.MapGet("/version", () => RunSync(() => new
        {
            version = Version,
            os = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
            runtime = Environment.Version.ToString(),
            onnxRuntime = OrtEnv.Instance().GetVersionString(),
        }));
        api.MapGet("/system", (Engine engine, ModelStore models) => RunSync(() => new
        {
            devices = Engine.Devices(),
            gpuSupport = Engine.HasGpu,
            activeDevice = engine.ActiveDevice,
            gpuError = engine.LastGpuError,
            defaultOutputDir = Jobs.DefaultOutputDir(),
            modelsDir = models.Dir,
            inputExtensions = ImageIO.InputExtensions,
        }));

        // ---- UI state ----
        api.MapGet("/doc/{name}", (string name, Workspace ws) => RunSync(() =>
        {
            if (!Workspace.IsDocument(name)) throw new ApiException($"Unknown document \"{name}\".");
            var d = ws.Load(name);
            return d.ValueKind == JsonValueKind.Undefined ? null : d;
        }));
        api.MapPut("/doc/{name}", async (string name, HttpRequest req, Workspace ws) =>
        {
            using var doc = await JsonDocument.ParseAsync(req.Body);
            var root = doc.RootElement.Clone();
            return await RunSync(() =>
            {
                if (!Workspace.IsDocument(name)) throw new ApiException($"Unknown document \"{name}\".");
                ws.Save(name, root);
                return null;
            });
        });

        // ---- models ----
        api.MapGet("/models", (ModelStore m) => RunSync(() => m.List()));
        api.MapPost("/models/{id}/download", (string id, ModelStore m) => RunSync(() => { m.StartDownload(id); return null; }));
        api.MapPost("/models/{id}/cancel", (string id, ModelStore m) => RunSync(() => { m.CancelDownload(id); return null; }));
        api.MapPost("/models/{id}/clear", (string id, ModelStore m) => RunSync(() => { m.ClearDownload(id); return null; }));
        api.MapDelete("/models/{id}", (string id, ModelStore m, Engine e) => RunSync(() =>
        {
            e.Unload(); // the file may be open
            m.Delete(id);
            return null;
        }));
        api.MapPost("/models/add", (PathRequest r, ModelStore m) => RunSync(() => new { id = m.AddFile(r.Path) }));

        // ---- files ----
        api.MapGet("/fs/places", () => RunSync(Files.Places));
        api.MapGet("/fs/list", (string path, bool? models) => RunSync(() => Files.List(path, models == true)));
        api.MapPost("/fs/scan", (ScanRequest r, ModelStore m) => RunSync(() => Files.Scan(r.Inputs ?? [], r.Recursive, r.NameTemplate, m.List().Select(x => x.Id))));
        api.MapPost("/fs/reveal", (PathRequest r) => RunSync(() => { Files.Reveal(r.Path); return null; }));
        api.MapPost("/fs/open", (PathRequest r) => RunSync(() => { Files.Open(r.Path); return null; }));
        api.MapGet("/fs/info", (string path) => RunSync(() =>
        {
            var fi = new FileInfo(path);
            if (!fi.Exists) throw new ApiException($"File not found: {path}");
            var dims = ImageIO.Probe(fi.FullName);
            return new { path = fi.FullName, name = fi.Name, size = fi.Length, width = dims?.Width, height = dims?.Height };
        }));
        // The image itself, for the viewer. Only image files are served.
        api.MapGet("/fs/image", (string path) =>
        {
            var full = Path.GetFullPath(path);
            if (!ImageIO.IsImage(full) || !File.Exists(full)) return Results.NotFound();
            ContentTypes.TryGetContentType(full, out var type);
            return Results.File(full, type ?? "application/octet-stream", enableRangeProcessing: false);
        });
        api.MapGet("/fs/thumb", (string path, int? size, HttpContext ctx) =>
        {
            var full = Path.GetFullPath(path);
            if (!ImageIO.IsImage(full) || !File.Exists(full)) return Results.NotFound();
            try
            {
                var bytes = ImageIO.Thumbnail(full, Math.Clamp(size ?? 160, 16, 1024));
                ctx.Response.Headers.CacheControl = "private, max-age=600";
                return Results.File(bytes, "image/jpeg");
            }
            catch (Exception ex) when (ex is ApiException or IOException or ArgumentException)
            {
                return Results.NotFound();
            }
        });
        api.MapPost("/upload", async (HttpRequest req, Files files, CancellationToken ct) => await Run(async () =>
        {
            if (!req.HasFormContentType) throw new ApiException("Expected multipart/form-data.");
            var form = await req.ReadFormAsync(ct);
            var paths = new List<string>();
            foreach (var f in form.Files) paths.Add(await files.SaveUpload(f, ct));
            return paths;
        })).DisableAntiforgery();

        // ---- jobs ----
        api.MapPost("/jobs", (JobRequest r, Jobs jobs) => RunSync(() => jobs.Start(r)));
        api.MapGet("/jobs", (Jobs jobs) => RunSync(() => jobs.List()));
        api.MapGet("/jobs/{id}", (string id, Jobs jobs) => RunSync(() => jobs.Get(id) ?? throw new ApiException("No such job.")));
        api.MapPost("/jobs/{id}/cancel", (string id, Jobs jobs) => RunSync(() => new { cancelled = jobs.Cancel(id) }));
    }
}
