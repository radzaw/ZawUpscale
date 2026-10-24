using System.Runtime.InteropServices;
using ImageMagick;
using SkiaSharp;

namespace ZawUpscale;

/// <summary>An 8-bit RGBA image (straight alpha), row-major, 4 bytes per pixel.</summary>
public sealed class RgbaImage
{
    public int Width { get; }
    public int Height { get; }
    public byte[] Pixels { get; }

    public RgbaImage(int width, int height, byte[]? pixels = null)
    {
        if (width <= 0 || height <= 0) throw new ArgumentException("Image dimensions must be positive.");
        if ((long)width * height * 4 > Array.MaxLength) throw new ApiException($"An image of {width}×{height} pixels is too large to hold in memory.");
        Width = width;
        Height = height;
        Pixels = pixels ?? new byte[width * height * 4];
        if (Pixels.Length != width * height * 4) throw new ArgumentException("Pixel buffer size doesn't match the dimensions.");
    }

    /// <summary>True if any pixel is not fully opaque.</summary>
    public bool HasAlpha()
    {
        var p = Pixels;
        for (var i = 3; i < p.Length; i += 4)
            if (p[i] != 255) return true;
        return false;
    }
}

/// <summary>Reading, writing and resizing images with SkiaSharp.</summary>
public static class ImageIO
{
    /// <summary>Formats Skia has no codec for on every platform, read (and except HEIC, written) with ImageMagick. (Declared first: static fields initialize in order.)</summary>
    static readonly string[] MagickExtensions = [".avif", ".heic", ".heif", ".tif", ".tiff", ".jxl"];

    /// <summary>Extensions that can be read: Skia's formats, and ImageMagick's <see cref="MagickExtensions"/>.</summary>
    public static readonly string[] InputExtensions = [".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".gif", ".ico", .. MagickExtensions];

    public static readonly string[] OutputFormats = ["png", "jpg", "webp", "avif", "tif", "jxl"];

    public static bool IsImage(string path) => InputExtensions.Contains(Path.GetExtension(path).ToLowerInvariant());

    static bool IsMagick(string path) => MagickExtensions.Contains(Path.GetExtension(path).ToLowerInvariant());

    /// <summary>True for formats every web view shows; the viewer is sent the others converted to PNG.</summary>
    public static bool IsWebImage(string path) => Path.GetExtension(path).ToLowerInvariant() is ".png" or ".jpg" or ".jpeg" or ".jfif" or ".webp" or ".bmp" or ".gif" or ".ico" or ".avif";

    /// <summary>The file extension for an output format.</summary>
    public static string ExtensionOf(string format) => format switch { "jpg" => ".jpg", "webp" => ".webp", "avif" => ".avif", "tif" => ".tif", "jxl" => ".jxl", "gif" => ".gif", _ => ".png" };

    /// <summary>Width and height of an image file (as displayed, after EXIF orientation), without decoding its pixels.</summary>
    public static (int Width, int Height)? Probe(string path)
    {
        try
        {
            if (!IsMagick(path))
            {
                using var codec = SKCodec.Create(path);
                if (codec != null)
                {
                    var (w, h) = (codec.Info.Width, codec.Info.Height);
                    return SwapsAxes(codec.EncodedOrigin) ? (h, w) : (w, h);
                }
            }
            using var m = new MagickImage();
            m.Ping(path);
            var swap = m.Orientation is OrientationType.LeftTop or OrientationType.RightTop or OrientationType.RightBottom or OrientationType.LeftBottom;
            return swap ? ((int)m.Height, (int)m.Width) : ((int)m.Width, (int)m.Height);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or MagickException)
        {
            return null;
        }
    }

    public static RgbaImage Load(string path)
    {
        if (!File.Exists(path)) throw new ApiException($"File not found: {path}");
        if (IsMagick(path)) return LoadMagick(path);
        using var stream = File.OpenRead(path);
        return Load(stream, path);
    }

    /// <summary>Reads the first image of a file with ImageMagick: oriented, in sRGB, 8 bits per channel.</summary>
    static RgbaImage LoadMagick(string path)
    {
        try
        {
            using var m = new MagickImage(path);
            m.AutoOrient();
            // Wide-gamut photos (Display P3 from phones, Adobe RGB) would look washed out if their numbers were taken as
            // sRGB. Profiles that are sRGB already (JPEG XL always carries one, without a description) are left alone:
            // converting them anyway shifts values by one here and there.
            var profile = m.GetColorProfile();
            if (profile is { Description: { Length: > 0 } d } && !d.Contains("sRGB", StringComparison.OrdinalIgnoreCase))
                m.TransformColorSpace(ColorProfiles.SRGB);
            var img = new RgbaImage((int)m.Width, (int)m.Height);
            using var pixels = m.GetPixelsUnsafe();
            var bytes = pixels.ToByteArray(PixelMapping.RGBA) ?? throw new ApiException($"Could not decode {Path.GetFileName(path)}.");
            Buffer.BlockCopy(bytes, 0, img.Pixels, 0, img.Pixels.Length);
            return img;
        }
        catch (MagickException ex)
        {
            throw new ApiException($"Unsupported or damaged image: {Path.GetFileName(path)} ({ex.Message.Split('`', '@')[0].Trim()})");
        }
    }

    /// <summary>Number of frames: more than one for an animated GIF or WebP.</summary>
    public static int FrameCount(string path)
    {
        if (Path.GetExtension(path).ToLowerInvariant() is not (".gif" or ".webp")) return 1;
        try
        {
            using var codec = SKCodec.Create(path);
            return codec == null ? 1 : Math.Max(1, codec.FrameCount);
        }
        catch (IOException) { return 1; }
    }

    /// <summary>An animation: full frames (each already composed over the previous ones), their durations and the loop count (0 = forever).</summary>
    public sealed record Animation(List<RgbaImage> Frames, List<int> DelaysMs, int Loops);

    public static Animation LoadAnimation(string path)
    {
        try
        {
            using var frames = new MagickImageCollection(path);
            frames.Coalesce();
            var list = new List<RgbaImage>();
            var delays = new List<int>();
            foreach (var f in frames)
            {
                var img = new RgbaImage((int)f.Width, (int)f.Height);
                using var pixels = f.GetPixelsUnsafe();
                Buffer.BlockCopy(pixels.ToByteArray(PixelMapping.RGBA)!, 0, img.Pixels, 0, img.Pixels.Length);
                list.Add(img);
                delays.Add((int)Math.Round(f.AnimationDelay * 1000.0 / Math.Max(1, f.AnimationTicksPerSecond)));
            }
            return new Animation(list, delays, (int)frames[0].AnimationIterations);
        }
        catch (MagickException ex)
        {
            throw new ApiException($"Could not read the frames of {Path.GetFileName(path)}: {ex.Message.Split('`', '@')[0].Trim()}");
        }
    }

    /// <summary>Writes an animated GIF or WebP (written next to the target and moved into place, like <see cref="Save"/>).</summary>
    public static void SaveAnimation(Animation a, string path, string format, int quality)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        var tmp = path + ".part";
        try
        {
            using (var frames = new MagickImageCollection())
            {
                for (var i = 0; i < a.Frames.Count; i++)
                {
                    var f = ToMagick(a.Frames[i]);
                    f.AnimationTicksPerSecond = 100;
                    f.AnimationDelay = (uint)Math.Max(1, Math.Round(a.DelaysMs[i] / 10.0));
                    f.AnimationIterations = (uint)a.Loops;
                    f.GifDisposeMethod = GifDisposeMethod.Background; // every frame is complete
                    if (format == "webp")
                    {
                        f.Quality = (uint)Math.Clamp(quality, 1, 100);
                        if (quality >= 100) f.Settings.SetDefine(MagickFormat.WebP, "lossless", true);
                    }
                    frames.Add(f);
                }
                if (format == "gif") frames.Quantize(new QuantizeSettings { Colors = 256, DitherMethod = DitherMethod.FloydSteinberg });
                frames.Write(tmp, format == "webp" ? MagickFormat.WebP : MagickFormat.Gif);
            }
            File.Move(tmp, path, overwrite: true);
        }
        catch
        {
            try { File.Delete(tmp); } catch (IOException) { /* best effort */ }
            throw;
        }
    }

    static MagickImage ToMagick(RgbaImage img)
    {
        var m = new MagickImage(img.Pixels, new PixelReadSettings((uint)img.Width, (uint)img.Height, StorageType.Char, PixelMapping.RGBA));
        if (!img.HasAlpha()) m.Alpha(AlphaOption.Off);
        return m;
    }

    /// <summary>A PNG of any readable image, for the viewer when the web view can't show the format itself.</summary>
    public static byte[] ToPng(string path)
    {
        var img = Load(path);
        using var ms = new MemoryStream();
        // Fast compression: it's sent once over the loopback to be shown.
        WithBitmap(img, bmp => { using var pixmap = bmp.PeekPixels(); return pixmap.Encode(ms, new SKPngEncoderOptions(SKPngEncoderFilterFlags.NoFilters, 1)); });
        return ms.ToArray();
    }

    public static RgbaImage Load(Stream stream, string nameForErrors = "image")
    {
        using var codec = SKCodec.Create(stream) ?? throw new ApiException($"Unsupported or damaged image: {Path.GetFileName(nameForErrors)}");
        var img = new RgbaImage(codec.Info.Width, codec.Info.Height);
        if (!DecodeInto(codec, img)) throw new ApiException($"Could not decode {Path.GetFileName(nameForErrors)}.");
        return Orient(img, codec.EncodedOrigin);
    }

    /// <summary>Decodes into <paramref name="img"/>, scaling to its size if the codec supports that.</summary>
    static bool DecodeInto(SKCodec codec, RgbaImage img)
    {
        var info = new SKImageInfo(img.Width, img.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        var handle = GCHandle.Alloc(img.Pixels, GCHandleType.Pinned);
        try
        {
            // An incomplete file still yields the rows that were read; anything else is a failure.
            return codec.GetPixels(info, handle.AddrOfPinnedObject()) is SKCodecResult.Success or SKCodecResult.IncompleteInput;
        }
        finally
        {
            handle.Free();
        }
    }

    static bool SwapsAxes(SKEncodedOrigin o) => o is SKEncodedOrigin.LeftTop or SKEncodedOrigin.RightTop or SKEncodedOrigin.RightBottom or SKEncodedOrigin.LeftBottom;

    /// <summary>Applies the EXIF orientation so the pixels are stored the way the photo is meant to be seen.</summary>
    internal static RgbaImage Orient(RgbaImage src, SKEncodedOrigin origin)
    {
        if (origin is SKEncodedOrigin.TopLeft or 0) return src;
        int w = src.Width, h = src.Height;
        var swap = SwapsAxes(origin);
        var dst = new RgbaImage(swap ? h : w, swap ? w : h);
        var s = MemoryMarshal.Cast<byte, uint>(src.Pixels);
        var d = MemoryMarshal.Cast<byte, uint>(dst.Pixels.AsSpan());
        var dw = dst.Width;
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
            {
                var (dx, dy) = origin switch
                {
                    SKEncodedOrigin.TopRight => (w - 1 - x, y),
                    SKEncodedOrigin.BottomRight => (w - 1 - x, h - 1 - y),
                    SKEncodedOrigin.BottomLeft => (x, h - 1 - y),
                    SKEncodedOrigin.LeftTop => (y, x),
                    SKEncodedOrigin.RightTop => (h - 1 - y, x),
                    SKEncodedOrigin.RightBottom => (h - 1 - y, w - 1 - x),
                    SKEncodedOrigin.LeftBottom => (y, w - 1 - x),
                    _ => (x, y),
                };
                d[dy * dw + dx] = s[y * w + x];
            }
        return dst;
    }

    /// <summary>Runs <paramref name="body"/> with the image wrapped (not copied) in an SKBitmap.</summary>
    static T WithBitmap<T>(RgbaImage img, Func<SKBitmap, T> body)
    {
        var handle = GCHandle.Alloc(img.Pixels, GCHandleType.Pinned);
        try
        {
            using var bmp = new SKBitmap();
            var info = new SKImageInfo(img.Width, img.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
            if (!bmp.InstallPixels(info, handle.AddrOfPinnedObject(), info.RowBytes)) throw new ApiException("Could not prepare the image for Skia.");
            return body(bmp);
        }
        finally
        {
            handle.Free();
        }
    }

    /// <summary>High-quality resize. Cubic (Mitchell) for enlarging and mild reduction, mipmaps for strong reduction.</summary>
    public static RgbaImage Resize(RgbaImage src, int width, int height)
    {
        if (src.Width == width && src.Height == height) return src;
        var dst = new RgbaImage(width, height);
        var strongReduction = width < src.Width / 2 || height < src.Height / 2;
        var sampling = strongReduction ? new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear) : new SKSamplingOptions(SKCubicResampler.Mitchell);
        WithBitmap(src, s => WithBitmap(dst, d => s.ScalePixels(d.PeekPixels(), sampling)));
        return dst;
    }

    /// <summary>Copies the alpha channel of <paramref name="alphaSource"/> (resized to fit) into <paramref name="target"/>.</summary>
    public static void TransferAlpha(RgbaImage alphaSource, RgbaImage target)
    {
        // Resize the alpha as a grey image, so it's interpolated like a color channel rather than premultiplied.
        var grey = new RgbaImage(alphaSource.Width, alphaSource.Height);
        var s = alphaSource.Pixels;
        var g = grey.Pixels;
        for (var i = 0; i < s.Length; i += 4)
        {
            g[i] = g[i + 1] = g[i + 2] = s[i + 3];
            g[i + 3] = 255;
        }
        var big = Resize(grey, target.Width, target.Height).Pixels;
        var t = target.Pixels;
        for (var i = 0; i < t.Length; i += 4) t[i + 3] = big[i];
    }

    /// <summary>Encodes to png, jpg, webp, avif, tif or jxl. JPEG has no alpha, so transparent areas are put on white.</summary>
    public static void Save(RgbaImage img, string path, string format, int quality)
    {
        if (format is "avif" or "tif" or "jxl")
        {
            SaveMagick(img, path, format, quality);
            return;
        }
        quality = Math.Clamp(quality, 1, 100);
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        if (format == "webp" && (img.Width > 16383 || img.Height > 16383))
            throw new ApiException($"WebP images can be at most 16383 pixels wide and high; this one is {img.Width}×{img.Height}. Choose PNG or JPEG.");
        if (format == "jpg" && (img.Width > 65535 || img.Height > 65535))
            throw new ApiException($"JPEG images can be at most 65535 pixels wide and high; this one is {img.Width}×{img.Height}. Choose PNG.");
        if (format == "jpg" && img.HasAlpha()) img = FlattenOnWhite(img);

        // Write next to the target and move into place, so a failed or cancelled save never leaves half a file.
        var tmp = path + ".part";
        try
        {
            using (var fs = File.Create(tmp))
            {
                var ok = WithBitmap(img, bmp =>
                {
                    using var pixmap = bmp.PeekPixels();
                    return format switch
                    {
                        "jpg" => pixmap.Encode(fs, new SKJpegEncoderOptions(quality, SKJpegEncoderDownsample.Downsample420, SKJpegEncoderAlphaOption.Ignore)),
                        "webp" => pixmap.Encode(fs, quality >= 100
                            ? new SKWebpEncoderOptions(SKWebpEncoderCompression.Lossless, 75)
                            : new SKWebpEncoderOptions(SKWebpEncoderCompression.Lossy, quality)),
                        _ => pixmap.Encode(fs, SKPngEncoderOptions.Default),
                    };
                });
                if (!ok) throw new ApiException($"Could not encode the image as {format.ToUpperInvariant()}.");
            }
            File.Move(tmp, path, overwrite: true);
        }
        catch
        {
            try { File.Delete(tmp); } catch (IOException) { /* best effort */ }
            throw;
        }
    }

    /// <summary>AVIF and JPEG XL (lossless at quality 100) and TIFF (lossless, zip-compressed) with ImageMagick.</summary>
    static void SaveMagick(RgbaImage img, string path, string format, int quality)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        var tmp = path + ".part";
        try
        {
            using (var m = ToMagick(img))
            {
                m.Quality = (uint)quality;
                if (format == "tif") m.Settings.Compression = CompressionMethod.Zip;
                // AV1's default effort makes large images take minutes; 6 of 0–9 is a good balance.
                if (format == "avif") m.Settings.SetDefine(MagickFormat.Heic, "speed", "6");
                m.Write(tmp, format switch { "avif" => MagickFormat.Avif, "jxl" => MagickFormat.Jxl, _ => MagickFormat.Tiff });
            }
            File.Move(tmp, path, overwrite: true);
        }
        catch (MagickException ex)
        {
            try { File.Delete(tmp); } catch (IOException) { /* best effort */ }
            throw new ApiException($"Could not save as {format.ToUpperInvariant()}: {ex.Message.Split('`', '@')[0].Trim()}");
        }
        catch
        {
            try { File.Delete(tmp); } catch (IOException) { /* best effort */ }
            throw;
        }
    }

    static RgbaImage FlattenOnWhite(RgbaImage src)
    {
        var dst = new RgbaImage(src.Width, src.Height, (byte[])src.Pixels.Clone());
        var p = dst.Pixels;
        for (var i = 0; i < p.Length; i += 4)
        {
            int a = p[i + 3];
            if (a == 255) continue;
            for (var c = 0; c < 3; c++) p[i + c] = (byte)((p[i + c] * a + 255 * (255 - a) + 127) / 255);
            p[i + 3] = 255;
        }
        return dst;
    }

    /// <summary>A JPEG thumbnail that fits in size×size (transparency shown on a light checkerboard is the UI's job; here it's flattened on white).</summary>
    public static byte[] Thumbnail(string path, int size)
    {
        RgbaImage img;
        if (IsMagick(path)) img = Load(path);
        else
        {
            using var codec = SKCodec.Create(path) ?? throw new ApiException("Unsupported image.");
            // Let the decoder skip work for big JPEGs by decoding at a reduced scale.
            var scale = Math.Min(1f, (float)size * 2 / Math.Max(codec.Info.Width, codec.Info.Height));
            var dims = codec.GetScaledDimensions(scale);
            var scaled = new RgbaImage(dims.Width, dims.Height);
            // Some codecs can't scale while decoding; those are decoded at full size instead.
            img = DecodeInto(codec, scaled) ? Orient(scaled, codec.EncodedOrigin) : Load(path);
        }
        var k = Math.Min(1.0, (double)size / Math.Max(img.Width, img.Height));
        var thumb = Resize(img, Math.Max(1, (int)Math.Round(img.Width * k)), Math.Max(1, (int)Math.Round(img.Height * k)));
        thumb = FlattenOnWhite(thumb);
        using var ms = new MemoryStream();
        WithBitmap(thumb, bmp =>
        {
            using var pixmap = bmp.PeekPixels();
            return pixmap.Encode(ms, new SKJpegEncoderOptions(82, SKJpegEncoderDownsample.Downsample420, SKJpegEncoderAlphaOption.Ignore));
        });
        return ms.ToArray();
    }
}
