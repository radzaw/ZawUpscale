namespace ZawUpscale;

static class Program
{
    // The native window must live on the main thread, and on Windows that thread must be STA (WebView2, file dialogs).
    [STAThread]
    static int Main(string[] args)
    {
        if (args.Length > 0 && Cli.Commands.Contains(args[0]))
            return Cli.Run(args);

        var opts = AppOptions.Parse(args);
        if (opts.ShowHelp)
        {
            Console.WriteLine(AppOptions.HelpText);
            Console.WriteLine();
            Console.WriteLine(Cli.HelpText);
            return 0;
        }

        var app = AppHost.Build(opts);
        app.StartAsync().GetAwaiter().GetResult();

        var url = AppHost.LaunchUrl(app, opts);
        Console.WriteLine($"ZawUpscale is running at {url}");
        Console.WriteLine($"Configuration directory: {opts.ConfigDir}");

        if (!opts.NoBrowser && !opts.UseBrowser && app.Services.GetRequiredService<AppWindow>().Run(url, app.Lifetime))
        {
            // The window was closed (or File > Exit closed it).
            app.StopAsync().GetAwaiter().GetResult();
            return 0;
        }

        if (!opts.NoBrowser)
        {
            BrowserLauncher.Launch(url, opts, app.Logger);
            if (!opts.KeepAlive)
                _ = app.Services.GetRequiredService<Heartbeat>().MonitorAsync(app.Lifetime);
        }

        app.WaitForShutdownAsync().GetAwaiter().GetResult();
        return 0;
    }
}
