using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;
using ZawUpscale.Tests.Infrastructure;

namespace ZawUpscale.Tests.Integration;

public sealed class ApiTests : IAsyncLifetime
{
    TestApp app = null!;
    string pics = null!;

    public async Task InitializeAsync()
    {
        app = await TestApp.StartAsync();
        TestModels.Write(Path.Combine(app.ConfigDir, "models"), "test-x2", TestModels.NearestResize(2));
        pics = Path.Combine(app.ConfigDir, "..", Path.GetFileName(app.ConfigDir) + "-pics");
        Directory.CreateDirectory(pics);
        ImageIO.Save(TestImages.Pattern(40, 30), Path.Combine(pics, "one.png"), "png", 100);
        ImageIO.Save(TestImages.Pattern(20, 10, alpha: true), Path.Combine(pics, "two.png"), "png", 100);
    }

    public async Task DisposeAsync()
    {
        await app.DisposeAsync();
        app.DeleteConfig();
        try { Directory.Delete(pics, recursive: true); } catch (IOException) { /* best effort */ }
    }

    static object Settings(string existing = "skip") => new { model = "test-x2", scale = 2, format = "png", device = "cpu", existing };

    async Task<JsonElement> RunJob(object body)
    {
        var job = (await app.PostAsync("/jobs", body)).Expect();
        var id = job.GetProperty("id").GetString();
        for (var i = 0; i < 300; i++)
        {
            var j = (await app.GetAsync($"/jobs/{id}")).Expect();
            if (j.GetProperty("status").GetString() is "done" or "cancelled") return j;
            await Task.Delay(50);
        }
        throw new TimeoutException("The job didn't finish.");
    }

    [Fact]
    public async Task Api_requires_the_token()
    {
        using var anon = new HttpClient { BaseAddress = new Uri(app.BaseAddress) };
        Assert.Equal(HttpStatusCode.Unauthorized, (await anon.GetAsync("/api/system")).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await anon.GetAsync("/")).StatusCode); // the UI itself is public
    }

    [Fact]
    public async Task System_and_models()
    {
        var sys = (await app.GetAsync("/system")).Expect();
        Assert.Contains(sys.GetProperty("devices").EnumerateArray(), d => d.GetProperty("id").GetString() == "cpu");
        var models = (await app.GetAsync("/models")).Expect().EnumerateArray().ToList();
        Assert.Equal(ModelStore.Catalog.Length + 1, models.Count);
        var custom = models.Single(m => m.GetProperty("id").GetString() == "test-x2");
        Assert.True(custom.GetProperty("installed").GetBoolean());
        Assert.False(custom.GetProperty("builtIn").GetBoolean());
        Assert.False((await app.PostAsync("/models/no-such-model/download")).Ok);
    }

    [Fact]
    public async Task Folder_job_upscales_then_skips_existing_results()
    {
        var scan = (await app.PostAsync("/fs/scan", new { inputs = new[] { pics }, recursive = false })).Expect();
        var items = scan.GetProperty("items");
        Assert.Equal(2, items.GetArrayLength());

        var job = await RunJob(new { items, settings = Settings() });
        Assert.Equal("done", job.GetProperty("status").GetString());
        foreach (var it in job.GetProperty("items").EnumerateArray())
        {
            Assert.Equal("done", it.GetProperty("status").GetString());
            var output = it.GetProperty("output").GetString()!;
            var img = ImageIO.Load(output);
            Assert.Equal((it.GetProperty("inWidth").GetInt32() * 2, it.GetProperty("inHeight").GetInt32() * 2), (img.Width, img.Height));
        }
        var one = ImageIO.Load(Path.Combine(pics, "one_x2.png"));
        Assert.Equal(TestImages.Nearest(TestImages.Pattern(40, 30), 2).Pixels, one.Pixels);
        Assert.True(ImageIO.Load(Path.Combine(pics, "two_x2.png")).HasAlpha());

        // Again: the results are recognised in the folder and the originals skipped.
        scan = (await app.PostAsync("/fs/scan", new { inputs = new[] { pics }, recursive = false, nameTemplate = "{name}_x{scale}" })).Expect();
        Assert.Equal(2, scan.GetProperty("previousResults").GetInt32());
        job = await RunJob(new { items = scan.GetProperty("items"), settings = Settings() });
        Assert.All(job.GetProperty("items").EnumerateArray(), it => Assert.Equal("skipped", it.GetProperty("status").GetString()));

        job = await RunJob(new { items = new[] { new { path = Path.Combine(pics, "one.png") } }, settings = Settings("rename") });
        Assert.EndsWith("one_x2 (2).png", job.GetProperty("items")[0].GetProperty("output").GetString());
    }

    [Fact]
    public async Task Job_errors_are_reported_per_item()
    {
        File.WriteAllText(Path.Combine(pics, "broken.png"), "nope");
        var job = await RunJob(new { items = new[] { new { path = Path.Combine(pics, "broken.png") }, new { path = Path.Combine(pics, "one.png") } }, settings = Settings("overwrite") });
        var items = job.GetProperty("items");
        Assert.Equal("failed", items[0].GetProperty("status").GetString());
        Assert.Contains("broken.png", items[0].GetProperty("error").GetString());
        Assert.Equal("done", items[1].GetProperty("status").GetString());

        var bad = await app.PostAsync("/jobs", new { items = new[] { new { path = Path.Combine(pics, "one.png") } }, settings = new { model = "realesrgan-x4plus" } });
        Assert.False(bad.Ok);
        Assert.Contains("isn't installed", bad.Error);
    }

    [Fact]
    public async Task Upload_list_thumb_and_image()
    {
        using var form = new MultipartFormDataContent();
        var bytes = await File.ReadAllBytesAsync(Path.Combine(pics, "one.png"));
        var file = new ByteArrayContent(bytes);
        file.Headers.ContentType = new MediaTypeHeaderValue("image/png");
        form.Add(file, "files", "dropped.png");
        using var res = await app.Http.PostAsync("/api/upload", form);
        var json = JsonDocument.Parse(await res.Content.ReadAsStringAsync()).RootElement;
        Assert.True(json.GetProperty("ok").GetBoolean(), json.ToString());
        var path = json.GetProperty("data")[0].GetString()!;
        Assert.StartsWith(Path.Combine(app.ConfigDir, "inbox"), path);
        Assert.Equal(bytes, await File.ReadAllBytesAsync(path));

        var list = (await app.GetAsync("/fs/list?path=" + Uri.EscapeDataString(pics))).Expect();
        Assert.Contains(list.GetProperty("entries").EnumerateArray(), e => e.GetProperty("name").GetString() == "one.png");

        using var thumb = await app.Http.GetAsync("/api/fs/thumb?size=16&path=" + Uri.EscapeDataString(path));
        Assert.Equal("image/jpeg", thumb.Content.Headers.ContentType?.MediaType);
        using var image = await app.Http.GetAsync("/api/fs/image?path=" + Uri.EscapeDataString(path));
        Assert.Equal(bytes, await image.Content.ReadAsByteArrayAsync());
        // Only images are served.
        using var other = await app.Http.GetAsync("/api/fs/image?path=" + Uri.EscapeDataString(Path.Combine(app.ConfigDir, "models", "test-x2.onnx")));
        Assert.Equal(HttpStatusCode.NotFound, other.StatusCode);
    }

    [Fact]
    public async Task State_document_round_trips()
    {
        Assert.True((await app.CallAsync(HttpMethod.Put, "/doc/state", new { prefs = new { theme = "dark" } })).Ok);
        var s = (await app.GetAsync("/doc/state")).Expect();
        Assert.Equal("dark", s.GetProperty("prefs").GetProperty("theme").GetString());
        Assert.False((await app.GetAsync("/doc/secrets")).Ok);
    }
}
