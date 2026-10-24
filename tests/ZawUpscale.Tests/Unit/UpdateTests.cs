using System.Net;
using System.Text;

namespace ZawUpscale.Tests.Unit;

public sealed class UpdateTests : IDisposable
{
    readonly string dir = Path.Combine(Path.GetTempPath(), "zawupscale-tests", Guid.NewGuid().ToString("n"));

    public void Dispose()
    {
        try { Directory.Delete(dir, true); } catch (IOException) { }
    }

    /// <summary>Answers every request with one response, and counts the requests.</summary>
    sealed class FakeGitHub(HttpStatusCode status, string body) : HttpMessageHandler
    {
        public int Calls;
        public HttpRequestMessage? LastRequest;

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
        {
            Calls++;
            LastRequest = request;
            return Task.FromResult(new HttpResponseMessage(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") });
        }
    }

    static string Release(string tag, string? asset = null, string page = $"https://github.com/{Updates.Repo}/releases/tag/v9.0.0") => $$"""
        {
          "tag_name": "{{tag}}",
          "html_url": "{{page}}",
          "body": "- Faster\n- Better",
          "published_at": "2026-10-01T12:00:00Z",
          "assets": [
            { "name": "zawupscale-other-rid", "browser_download_url": "https://github.com/{{Updates.Repo}}/releases/download/{{tag}}/zawupscale-other-rid" },
            { "name": "{{asset ?? Updates.AssetName}}", "browser_download_url": "https://github.com/{{Updates.Repo}}/releases/download/{{tag}}/{{asset ?? Updates.AssetName}}" }
          ]
        }
        """;

    [Theory]
    [InlineData("1.0.1", "1.0.0", 1)]
    [InlineData("1.10.0", "1.9.0", 1)]
    [InlineData("v2.0", "1.99.99", 1)]
    [InlineData("1.0", "1.0.0", 0)]
    [InlineData("1.0.0", "1.0.0-beta.1", 1)]
    [InlineData("1.0.0-beta.10", "1.0.0-beta.9", 1)]
    [InlineData("1.0.0-rc.1", "1.0.0-beta.2", 1)]
    [InlineData("1.0.0-beta", "1.0.0-beta.1", -1)]
    [InlineData("1.0.0+abc123", "1.0.0", 0)]
    public void Compares_versions(string a, string b, int expected)
    {
        Assert.Equal(expected, Math.Sign(Updates.Compare(a, b)));
        Assert.Equal(-expected, Math.Sign(Updates.Compare(b, a)));
    }

    [Fact]
    public async Task Finds_a_newer_release_and_the_file_for_this_platform()
    {
        var gh = new FakeGitHub(HttpStatusCode.OK, Release("v9.0.0"));
        var u = await new Updates(dir, gh, "https://api.test/latest").Check();

        Assert.Null(u.Error);
        Assert.True(u.Newer);
        Assert.Equal("9.0.0", u.Latest);
        Assert.Equal(Api.Version, u.Current);
        Assert.EndsWith("/releases/tag/v9.0.0", u.Url);
        Assert.EndsWith("/" + Updates.AssetName, u.Download);
        Assert.Contains("Faster", u.Notes);
        Assert.Equal(new DateTimeOffset(2026, 10, 1, 12, 0, 0, TimeSpan.Zero), u.Published);
        // Only the app's name and version go out.
        Assert.Equal($"ZawUpscale/{Api.Version}", gh.LastRequest!.Headers.UserAgent.ToString());
    }

    [Fact]
    public async Task The_same_or_an_older_release_is_not_an_update()
    {
        var same = await new Updates(dir, new FakeGitHub(HttpStatusCode.OK, Release("v" + Api.Version)), "https://api.test/latest").Check();
        Assert.False(same.Newer);
        Assert.Equal(Api.Version, same.Latest);

        var older = await new Updates(dir, new FakeGitHub(HttpStatusCode.OK, Release("v0.0.1")), "https://api.test/latest").Check(force: true);
        Assert.False(older.Newer);
    }

    [Fact]
    public async Task No_releases_yet_is_not_an_error()
    {
        var u = await new Updates(dir, new FakeGitHub(HttpStatusCode.NotFound, """{"message":"Not Found"}"""), "https://api.test/latest").Check();
        Assert.Null(u.Error);
        Assert.False(u.Newer);
        Assert.Null(u.Latest);
    }

    [Fact]
    public async Task A_check_is_kept_for_a_day_and_errors_are_not()
    {
        var gh = new FakeGitHub(HttpStatusCode.OK, Release("v9.0.0"));
        await new Updates(dir, gh, "https://api.test/latest").Check();
        // A later start (a new instance) uses the stored answer…
        var again = await new Updates(dir, gh, "https://api.test/latest").Check();
        Assert.Equal(1, gh.Calls);
        Assert.True(again.Newer);
        // …unless asked.
        await new Updates(dir, gh, "https://api.test/latest").Check(force: true);
        Assert.Equal(2, gh.Calls);

        var failing = new FakeGitHub(HttpStatusCode.Forbidden, """{"message":"API rate limit exceeded"}""");
        var u = await new Updates(dir, failing, "https://api.test/latest").Check(force: true);
        Assert.Contains("403", u.Error);
        Assert.False(u.Newer);
        // The good answer is still the one kept.
        Assert.Equal("9.0.0", new Updates(dir, failing, "https://api.test/latest").Last()!.Latest);
    }

    [Fact]
    public async Task Links_elsewhere_are_dropped()
    {
        var gh = new FakeGitHub(HttpStatusCode.OK, Release("v9.0.0", page: "https://evil.example/zawupscale"));
        var u = await new Updates(dir, gh, "https://api.test/latest").Check();
        Assert.Equal(Updates.ReleasesPage, u.Url);

        Assert.Throws<ApiException>(() => Updates.Open("https://evil.example/"));
        Assert.Throws<ApiException>(() => Updates.Open($"https://github.com/{Updates.Repo}.evil/releases"));
        Assert.Throws<ApiException>(() => Updates.Open("file:///C:/Windows/System32/calc.exe"));
    }

    [Fact]
    public async Task A_release_without_this_platform_has_no_download()
    {
        var u = await new Updates(dir, new FakeGitHub(HttpStatusCode.OK, Release("v9.0.0", asset: "zawupscale-plan9-mips")), "https://api.test/latest").Check();
        Assert.True(u.Newer);
        Assert.Null(u.Download);
    }

    [Fact]
    public async Task An_unreadable_answer_is_an_error()
    {
        var u = await new Updates(dir, new FakeGitHub(HttpStatusCode.OK, "<html>"), "https://api.test/latest").Check();
        Assert.NotNull(u.Error);
        Assert.False(u.Newer);
    }
}
