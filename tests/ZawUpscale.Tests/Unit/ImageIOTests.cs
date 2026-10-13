using SkiaSharp;
using ZawUpscale.Tests.Infrastructure;

namespace ZawUpscale.Tests.Unit;

public sealed class ImageIOTests : IDisposable
{
    readonly string dir = Path.Combine(Path.GetTempPath(), "zawupscale-tests", Guid.NewGuid().ToString("n"));

    public ImageIOTests() => Directory.CreateDirectory(dir);

    public void Dispose()
    {
        try { Directory.Delete(dir, recursive: true); } catch (IOException) { /* best effort */ }
    }

    static uint Px(RgbaImage img, int x, int y) => BitConverter.ToUInt32(img.Pixels, (y * img.Width + x) * 4);

    static byte[] Visible(byte[] rgba)
    {
        var r = (byte[])rgba.Clone();
        for (var i = 0; i < r.Length; i += 4)
            if (r[i + 3] == 0) r[i] = r[i + 1] = r[i + 2] = 0;
        return r;
    }

    [Theory]
    [InlineData(SKEncodedOrigin.TopRight)]
    [InlineData(SKEncodedOrigin.BottomRight)]
    [InlineData(SKEncodedOrigin.BottomLeft)]
    [InlineData(SKEncodedOrigin.LeftTop)]
    [InlineData(SKEncodedOrigin.RightTop)]
    [InlineData(SKEncodedOrigin.RightBottom)]
    [InlineData(SKEncodedOrigin.LeftBottom)]
    public void Exif_orientation_is_applied(SKEncodedOrigin origin)
    {
        var src = TestImages.Pattern(3, 2);
        var o = ImageIO.Orient(src, origin);
        var swap = origin is SKEncodedOrigin.LeftTop or SKEncodedOrigin.RightTop or SKEncodedOrigin.RightBottom or SKEncodedOrigin.LeftBottom;
        Assert.Equal(swap ? (2, 3) : (3, 2), (o.Width, o.Height));
        // Where the source's top-left pixel ends up.
        var (x, y) = origin switch
        {
            SKEncodedOrigin.TopRight => (2, 0),
            SKEncodedOrigin.BottomRight => (2, 1),
            SKEncodedOrigin.BottomLeft => (0, 1),
            SKEncodedOrigin.LeftTop => (0, 0),
            SKEncodedOrigin.RightTop => (1, 0), // rotated 90° clockwise
            SKEncodedOrigin.RightBottom => (1, 2),
            _ => (0, 2), // LeftBottom: rotated 90° counter-clockwise
        };
        Assert.Equal(Px(src, 0, 0), Px(o, x, y));
    }

    [Fact]
    public void Png_and_lossless_webp_round_trip_exactly()
    {
        var img = TestImages.Pattern(31, 17, alpha: true);
        foreach (var (fmt, ext) in new[] { ("png", ".png"), ("webp", ".webp") })
        {
            var path = Path.Combine(dir, "rt" + ext);
            ImageIO.Save(img, path, fmt, 100);
            var back = ImageIO.Load(path);
            Assert.Equal((31, 17), (back.Width, back.Height));
            // Encoders may drop the color of fully transparent pixels; everything visible must survive.
            Assert.Equal(Visible(img.Pixels), Visible(back.Pixels));
            Assert.Equal((31, 17), ImageIO.Probe(path));
        }
        Assert.Empty(Directory.GetFiles(dir, "*.part"));
    }

    [Fact]
    public void Jpeg_puts_transparency_on_white()
    {
        var img = new RgbaImage(8, 8); // fully transparent black
        var path = Path.Combine(dir, "t.jpg");
        ImageIO.Save(img, path, "jpg", 95);
        var back = ImageIO.Load(path);
        Assert.False(back.HasAlpha());
        Assert.True(back.Pixels[0] > 245 && back.Pixels[1] > 245 && back.Pixels[2] > 245);
    }

    [Fact]
    public void Oversized_webp_is_refused_with_a_hint()
    {
        var ex = Assert.Throws<ApiException>(() => ImageIO.Save(new RgbaImage(16384, 1), Path.Combine(dir, "big.webp"), "webp", 90));
        Assert.Contains("PNG", ex.Message);
    }

    [Fact]
    public void Thumbnails_fit_the_box()
    {
        var path = Path.Combine(dir, "wide.png");
        ImageIO.Save(TestImages.Pattern(400, 100), path, "png", 100);
        using var bmp = SKBitmap.Decode(ImageIO.Thumbnail(path, 160));
        Assert.Equal((160, 40), (bmp.Width, bmp.Height));
    }

    [Fact]
    public void Damaged_files_give_a_clear_error()
    {
        var path = Path.Combine(dir, "bad.png");
        File.WriteAllText(path, "not an image");
        var ex = Assert.Throws<ApiException>(() => ImageIO.Load(path));
        Assert.Contains("bad.png", ex.Message);
    }
}
