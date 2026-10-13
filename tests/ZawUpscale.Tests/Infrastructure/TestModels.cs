using System.Text;

namespace ZawUpscale.Tests.Infrastructure;

/// <summary>
/// Builds tiny ONNX models in code, so the real ONNX Runtime path can be tested without downloads: a Resize
/// node doing nearest-neighbour upscaling, whose exact output is easy to predict. Written as raw protobuf,
/// using only the handful of ModelProto fields a Resize graph needs.
/// </summary>
public static class TestModels
{
    /// <summary>A model that upscales [1, 3, H, W] by <paramref name="scale"/> with nearest-neighbour; H and W fixed or dynamic.</summary>
    public static byte[] NearestResize(int scale, int? fixedSize = null, bool float16 = false)
    {
        const int Float = 1, Float16 = 10;
        var elem = float16 ? Float16 : Float;
        var nodes = new List<byte[]>();
        var inits = new List<byte[]>();
        var resizeInput = "x";
        if (float16)
        {
            // Resize on float16 needs opset 13+ kernels that CPU may lack; compute in float32 and cast at the edges.
            nodes.Add(Node("Cast", ["x"], ["x32"], Attr("to", Float)));
            resizeInput = "x32";
        }
        inits.Add(Tensor("scales", [4], [1f, 1f, scale, scale]));
        nodes.Add(Node("Resize", [resizeInput, "", "scales"], [float16 ? "y32" : "y"], AttrStr("mode", "nearest")));
        if (float16) nodes.Add(Node("Cast", ["y32"], ["y"], Attr("to", Float16)));

        object H = fixedSize.HasValue ? fixedSize.Value : "h", W = fixedSize.HasValue ? fixedSize.Value : "w";
        object OH = fixedSize.HasValue ? fixedSize.Value * scale : "oh", OW = fixedSize.HasValue ? fixedSize.Value * scale : "ow";
        var graph = new Proto();
        foreach (var n in nodes) graph.Bytes(1, n);
        graph.String(2, "test");
        foreach (var t in inits) graph.Bytes(5, t);
        graph.Bytes(11, ValueInfo("x", elem, [1, 3, H, W]));
        graph.Bytes(12, ValueInfo("y", elem, [1, 3, OH, OW]));

        var opset = new Proto().String(1, "").Int(2, 13).ToArray();
        return new Proto().Int(1, 8).Bytes(8, opset).Bytes(7, graph.ToArray()).ToArray();
    }

    public static string Write(string dir, string id, byte[] model)
    {
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, id + ".onnx");
        File.WriteAllBytes(path, model);
        return path;
    }

    static byte[] Node(string op, string[] inputs, string[] outputs, byte[]? attr = null)
    {
        var p = new Proto();
        foreach (var i in inputs) p.String(1, i);
        foreach (var o in outputs) p.String(2, o);
        p.String(4, op);
        if (attr != null) p.Bytes(5, attr);
        return p.ToArray();
    }

    static byte[] Attr(string name, long value) => new Proto().String(1, name).Int(3, value).Int(20, 2).ToArray(); // INT
    static byte[] AttrStr(string name, string value) => new Proto().String(1, name).Bytes(4, Encoding.UTF8.GetBytes(value)).Int(20, 3).ToArray(); // STRING

    static byte[] Tensor(string name, long[] dims, float[] data)
    {
        var p = new Proto();
        foreach (var d in dims) p.Int(1, d);
        p.Int(2, 1);
        var raw = new byte[data.Length * 4];
        Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
        p.String(8, name);
        p.Bytes(9, raw); // raw_data, little endian
        return p.ToArray();
    }

    static byte[] ValueInfo(string name, int elem, object[] dims)
    {
        var shape = new Proto();
        foreach (var d in dims)
            shape.Bytes(1, d is int v ? new Proto().Int(1, v).ToArray() : new Proto().String(2, (string)d).ToArray());
        var tensor = new Proto().Int(1, elem).Bytes(2, shape.ToArray()).ToArray();
        var type = new Proto().Bytes(1, tensor).ToArray();
        return new Proto().String(1, name).Bytes(2, type).ToArray();
    }

    /// <summary>Minimal protobuf writer: varints and length-delimited fields.</summary>
    sealed class Proto
    {
        readonly MemoryStream ms = new();

        void Varint(ulong v)
        {
            while (v >= 0x80) { ms.WriteByte((byte)(v | 0x80)); v >>= 7; }
            ms.WriteByte((byte)v);
        }

        public Proto Int(int field, long v) { Varint((ulong)(field << 3)); Varint((ulong)v); return this; }
        public Proto Bytes(int field, byte[] b) { Varint((ulong)(field << 3 | 2)); Varint((ulong)b.Length); ms.Write(b); return this; }
        public Proto String(int field, string s) => Bytes(field, Encoding.UTF8.GetBytes(s));
        public byte[] ToArray() => ms.ToArray();
    }
}

/// <summary>Test images with every pixel distinct, so misplaced tiles show up.</summary>
public static class TestImages
{
    public static RgbaImage Pattern(int w, int h, bool alpha = false)
    {
        var img = new RgbaImage(w, h);
        var p = img.Pixels;
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
            {
                var i = (y * w + x) * 4;
                p[i] = (byte)(x * 7 + y * 3);
                p[i + 1] = (byte)(x * 13 ^ y * 5);
                p[i + 2] = (byte)(y * 11 + 17);
                p[i + 3] = alpha ? (byte)(x * 255 / Math.Max(1, w - 1)) : (byte)255;
            }
        return img;
    }

    /// <summary>The exact nearest-neighbour enlargement (RGB; alpha forced opaque).</summary>
    public static RgbaImage Nearest(RgbaImage src, int s)
    {
        var dst = new RgbaImage(src.Width * s, src.Height * s);
        for (var y = 0; y < dst.Height; y++)
            for (var x = 0; x < dst.Width; x++)
            {
                var si = ((y / s) * src.Width + x / s) * 4;
                var di = (y * dst.Width + x) * 4;
                dst.Pixels[di] = src.Pixels[si];
                dst.Pixels[di + 1] = src.Pixels[si + 1];
                dst.Pixels[di + 2] = src.Pixels[si + 2];
                dst.Pixels[di + 3] = 255;
            }
        return dst;
    }
}

/// <summary>An ISrModel in plain C#: nearest-neighbour by an integer factor, optionally with a fixed input size.</summary>
public sealed class FakeModel(int scale, (int, int)? fixedSize = null) : ISrModel
{
    public int Scale => scale;
    public (int Width, int Height)? FixedSize => fixedSize;
    public int Calls { get; private set; }
    public List<(int H, int W)> Sizes { get; } = [];

    public float[] Run(float[] chw, int h, int w)
    {
        if (fixedSize is { } f) Assert.Equal((f.Item1, f.Item2), (w, h));
        Calls++;
        Sizes.Add((h, w));
        int oh = h * scale, ow = w * scale;
        var o = new float[3 * oh * ow];
        for (var c = 0; c < 3; c++)
            for (var y = 0; y < oh; y++)
                for (var x = 0; x < ow; x++)
                    o[c * oh * ow + y * ow + x] = chw[c * h * w + (y / scale) * w + x / scale];
        return o;
    }
}
