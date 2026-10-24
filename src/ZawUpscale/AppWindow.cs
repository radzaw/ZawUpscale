using System.Runtime.InteropServices;
using Photino.NET;

namespace ZawUpscale;

/// <summary>
/// The native app window: the UI in the operating system's web view (WebView2 on Windows, WKWebView on macOS,
/// WebKitGTK on Linux) through Photino. It also offers the native open dialogs. The window has to run on the
/// process's main thread, which on Windows must be an STA thread.
/// </summary>
public sealed class AppWindow(AppOptions opts, ILogger<AppWindow> log)
{
    PhotinoWindow? window;
    volatile bool open;

    /// <summary>True while the native window is up, so the open dialogs can be used.</summary>
    public bool IsOpen => open;

    /// <summary>
    /// Shows the window and blocks until it's closed. Returns false, having shown nothing, when the platform's
    /// web view isn't available (for example WebKitGTK missing on Linux); the caller then uses a browser window.
    /// </summary>
    public bool Run(string url, IHostApplicationLifetime life)
    {
        // Without a display GTK ends the process instead of failing, so don't even try (e.g. over SSH).
        if (OperatingSystem.IsLinux() && string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DISPLAY"))
            && string.IsNullOrEmpty(Environment.GetEnvironmentVariable("WAYLAND_DISPLAY")))
        {
            log.LogWarning("No display found; using a browser window.");
            return false;
        }
        try
        {
            var w = new PhotinoWindow()
                .SetLogVerbosity(0)
                .SetTitle("ZawUpscale")
                .SetUseOsDefaultSize(false)
                .SetSize(1360, 860)
                .SetMinSize(760, 520)
                .Center()
                .SetDevToolsEnabled(false)
                .SetGrantBrowserPermissions(true) // notifications when a batch finishes in the background
                .SetTemporaryFilesPath(Path.Combine(opts.ConfigDir, "webview"));
            if (IconFile() is { } icon) w.SetIconFile(icon);
            w.WindowCreated += (_, _) =>
            {
                open = true;
                // Smaller screens: start maximized rather than hanging off the edge.
                try
                {
                    var area = w.MainMonitor.WorkArea;
                    if (area.Width < 1360 || area.Height < 860) w.SetMaximized(true);
                }
                catch (Exception ex) when (ex is ApplicationException or InvalidOperationException) { /* keep the default size */ }
            };
            w.RegisterWebMessageReceivedHandler((_, message) => OnMessage(w, message));
            w.Load(new Uri(url));
            window = w;

            // File > Exit and the CLI-less shutdown paths stop the host; take the window down with it.
            using var reg = life.ApplicationStopping.Register(Close);
            w.WaitForClose();
            return true;
        }
        catch (Exception ex) when (!open)
        {
            log.LogWarning("The native window is not available ({Message}); using a browser window.", Reason(ex));
            return false;
        }
        finally
        {
            open = false;
        }
    }

    /// <summary>A short reason: the loader's message lists every path it tried.</summary>
    static string Reason(Exception ex)
    {
        var m = (ex.InnerException ?? ex).Message;
        if (OperatingSystem.IsLinux() && ex.GetBaseException() is DllNotFoundException)
        {
            var missing = System.Text.RegularExpressions.Regex.Match(m, @"(lib[\w.+-]+\.so[.\d]*): cannot open");
            return missing.Success
                ? $"{missing.Groups[1].Value} is missing; install WebKitGTK 4.1 and libnotify (libwebkit2gtk-4.1-0 and libnotify4 on Debian and Ubuntu)"
                : "Photino.Native could not be loaded";
        }
        return m.Length > 300 ? m[..300] + "…" : m;
    }

    void Close()
    {
        if (!open || window == null) return;
        open = false;
        try { window.Close(); }
        catch (ApplicationException) { /* not created yet or already gone */ }
    }

    /// <summary>
    /// Messages from the page: "theme:dark" / "theme:light" so the title bar matches the app; "progress:0.42",
    /// "progress:paused", "progress:error" and "progress:none" for the taskbar button; "attention" to flash it; and
    /// "notify:title\nbody" for a system notification.
    /// </summary>
    void OnMessage(PhotinoWindow w, string? message)
    {
        if (message == null) return;
        if (message is "theme:dark" or "theme:light" && OperatingSystem.IsWindows())
            DarkTitleBar(w.WindowHandle, message == "theme:dark");
        else if (message.StartsWith("progress:") && OperatingSystem.IsWindows())
            Taskbar.SetProgress(w.WindowHandle, message["progress:".Length..]);
        else if (message == "attention" && OperatingSystem.IsWindows())
            Taskbar.Flash(w.WindowHandle);
        else if (message.StartsWith("notify:"))
        {
            var text = message["notify:".Length..].Split('\n', 2);
            try { w.SendNotification(text[0], text.Length > 1 ? text[1] : ""); }
            catch (Exception ex) when (ex is ApplicationException or InvalidOperationException) { log.LogDebug("No notification: {Message}", ex.Message); }
        }
    }

    public string[]? OpenFiles(string title, string? start, bool multiple, (string Name, string[] Extensions)[] filters) =>
        Dialog(w => w.ShowOpenFile(title, StartDir(start), multiple, filters));

    public string[]? OpenFolder(string title, string? start) =>
        Dialog(w => w.ShowOpenFolder(title, StartDir(start), false));

    /// <summary>Runs a dialog on the window's thread; null when cancelled.</summary>
    string[]? Dialog(Func<PhotinoWindow, string[]> show)
    {
        var w = window;
        if (!open || w == null) throw new ApiException("The native window isn't open.");
        var r = show(w);
        return r is { Length: > 0 } ? r.Where(p => !string.IsNullOrEmpty(p)).ToArray() : null;
    }

    static string? StartDir(string? start)
    {
        var pictures = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
        var fallback = Directory.Exists(pictures) ? pictures : null;
        if (string.IsNullOrWhiteSpace(start)) return fallback;
        try
        {
            var p = Path.GetFullPath(start);
            return Directory.Exists(p) ? p : File.Exists(p) ? Path.GetDirectoryName(p) : fallback;
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return fallback;
        }
    }

    /// <summary>The window icon as a file (Photino loads it from disk): .ico on Windows, .png elsewhere.</summary>
    string? IconFile()
    {
        var resource = OperatingSystem.IsWindows() ? "zawupscale.ico" : "wwwroot/icon-256.png";
        var asm = typeof(AppWindow).Assembly;
        var name = asm.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith(Path.GetFileName(resource), StringComparison.OrdinalIgnoreCase));
        if (name == null) return null;
        try
        {
            var dir = Path.Combine(opts.ConfigDir, "webview");
            Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, Path.GetFileName(resource));
            using var src = asm.GetManifestResourceStream(name)!;
            using var ms = new MemoryStream();
            src.CopyTo(ms);
            if (!File.Exists(path) || new FileInfo(path).Length != ms.Length) File.WriteAllBytes(path, ms.ToArray());
            return path;
        }
        catch (IOException) { return null; }
        catch (UnauthorizedAccessException) { return null; }
    }

    [DllImport("dwmapi.dll")]
    static extern int DwmSetWindowAttribute(IntPtr hwnd, int attribute, ref int value, int size);

    static void DarkTitleBar(IntPtr hwnd, bool dark)
    {
        if (hwnd == IntPtr.Zero) return;
        var on = dark ? 1 : 0;
        try { DwmSetWindowAttribute(hwnd, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, ref on, sizeof(int)); }
        catch (EntryPointNotFoundException) { /* Windows 10 before 20H1 */ }
    }
}
