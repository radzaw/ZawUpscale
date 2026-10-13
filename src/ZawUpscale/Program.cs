using ZawUpscale;

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
await app.StartAsync();

var url = AppHost.LaunchUrl(app, opts);
Console.WriteLine($"ZawUpscale is running at {url}");
Console.WriteLine($"Configuration directory: {opts.ConfigDir}");

if (!opts.NoBrowser)
{
    BrowserLauncher.Launch(url, opts, app.Logger);
    if (!opts.KeepAlive)
        _ = app.Services.GetRequiredService<Heartbeat>().MonitorAsync(app.Lifetime);
}

await app.WaitForShutdownAsync();
return 0;
