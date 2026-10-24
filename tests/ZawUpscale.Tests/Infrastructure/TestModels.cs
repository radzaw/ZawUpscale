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

    /// <summary>
    /// A stand-in face restorer with GFPGAN's interface ([1, 3, 512, 512] in [−1, 1]) that negates its input, which
    /// inverts the colors; with <paramref name="weight"/>, it also takes CodeFormer's scalar double fidelity input.
    /// </summary>
    public static byte[] FaceRestorer(bool weight = false, int size = 512)
    {
        const int Float = 1, Double = 11;
        object[] dims = [1, 3, size, size];
        if (!weight)
            return Graph([Node("Neg", ["input"], ["output"])], [], [ValueInfo("input", Float, dims)], [ValueInfo("output", Float, dims)]);
        // output = −input + 0 · weight, so the weight is a real input that has to be fed.
        return Graph(
            [Node("Neg", ["input"], ["neg"]), Node("Cast", ["weight"], ["w32"], Attr("to", Float)), Node("Mul", ["w32", "zero"], ["w0"]), Node("Add", ["neg", "w0"], ["output"])],
            [Tensor("zero", [], [0f])],
            [ValueInfo("input", Float, dims), ValueInfo("weight", Double, [])], [ValueInfo("output", Float, dims)]);
    }

    public static string Write(string dir, string id, byte[] model)
    {
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, id + ".onnx");
        File.WriteAllBytes(path, model);
        return path;
    }

    /// <summary>
    /// A SUPIR model folder with stand-in networks that have the real ones' names, inputs and dynamic
    /// dimensions: the encoders average 8 × 8 blocks (a strided Conv), the decoder repeats them back
    /// (ConvTranspose), the "UNet" returns sample − control, and the text encoders repeat each token id
    /// across the hidden size, with bigG's pooled output the largest id.
    /// </summary>
    public static void SupirPackage(string dir)
    {
        const int Float = 1, Int64 = 7;
        Directory.CreateDirectory(Path.Combine(dir, "tokenizer"));
        object[] img = ["batch", 3, "img_h", "img_w"], lat = ["batch", 4, "lat_h", "lat_w"];

        var down = new float[4 * 3 * 64];
        var up = new float[4 * 3 * 64];
        for (var c = 0; c < 3; c++)
            for (var k = 0; k < 64; k++)
            {
                down[(c * 3 + c) * 64 + k] = 1f / 64; // Conv weights [out 4, in 3, 8, 8]
                up[(c * 3 + c) * 64 + k] = 1f;        // ConvTranspose weights [in 4, out 3, 8, 8]
            }
        byte[][] strided = [AttrInts("kernel_shape", [8, 8]), AttrInts("strides", [8, 8])];
        var encoder = Graph([Node("Conv", ["image", "w"], ["latent"], strided)], [Tensor("w", [4, 3, 8, 8], down)],
            [ValueInfo("image", Float, img)], [ValueInfo("latent", Float, lat)]);
        File.WriteAllBytes(Path.Combine(dir, ZawUpscale.SupirPackage.DenoiseEncoder), encoder);
        File.WriteAllBytes(Path.Combine(dir, ZawUpscale.SupirPackage.VaeEncoder), encoder);
        File.WriteAllBytes(Path.Combine(dir, ZawUpscale.SupirPackage.VaeDecoder),
            Graph([Node("ConvTranspose", ["latent", "w"], ["image"], strided)], [Tensor("w", [4, 3, 8, 8], up)],
                [ValueInfo("latent", Float, lat)], [ValueInfo("image", Float, img)]));
        File.WriteAllBytes(Path.Combine(dir, ZawUpscale.SupirPackage.Unet),
            Graph([Node("Sub", ["sample", "control"], ["eps"])], [],
                [ValueInfo("sample", Float, lat), ValueInfo("timestep", Float, ["batch"]), ValueInfo("context", Float, ["batch", 77, 2048]),
                 ValueInfo("y", Float, ["batch", 2816]), ValueInfo("control", Float, lat), ValueInfo("control_scale", Float, [1])],
                [ValueInfo("eps", Float, lat)]));

        byte[] Text(int dim, bool pooled)
        {
            var nodes = new List<byte[]>
            {
                Node("Cast", ["input_ids"], ["f"], Attr("to", Float)),
                Node("Unsqueeze", ["f", "axis2"], ["u"]),
                Node("Tile", ["u", "reps"], ["hidden"]),
            };
            var outputs = new List<byte[]> { ValueInfo("hidden", Float, [1, 77, dim]) };
            var inits = new List<byte[]> { TensorI64("axis2", [1], [2]), TensorI64("reps", [3], [1, 1, dim]) };
            if (pooled)
            {
                nodes.Add(Node("ReduceMax", ["f"], ["m"], AttrInts("axes", [1])));
                nodes.Add(Node("Tile", ["m", "preps"], ["pooled"]));
                inits.Add(TensorI64("preps", [2], [1, dim]));
                outputs.Add(ValueInfo("pooled", Float, [1, dim]));
            }
            return Graph(nodes, inits, [ValueInfo("input_ids", Int64, [1, 77])], outputs);
        }
        File.WriteAllBytes(Path.Combine(dir, ZawUpscale.SupirPackage.TextEncoderL), Text(768, false));
        File.WriteAllBytes(Path.Combine(dir, ZawUpscale.SupirPackage.TextEncoderG), Text(1280, true));
        File.WriteAllText(Path.Combine(dir, ZawUpscale.SupirPackage.Vocab), """{ "<|startoftext|>": 49406, "<|endoftext|>": 49407, "a</w>": 320 }""");
        File.WriteAllText(Path.Combine(dir, ZawUpscale.SupirPackage.Merges), "#version: 0.2\n");
        File.WriteAllText(Path.Combine(dir, ZawUpscale.SupirPackage.ManifestName),
            """{ "format": "zawupscale-supir", "version": 1, "name": "Test SUPIR", "minSize": 64, "defaults": { "steps": 2 } }""");
    }

    static byte[] Graph(IEnumerable<byte[]> nodes, IEnumerable<byte[]> inits, IEnumerable<byte[]> inputs, IEnumerable<byte[]> outputs)
    {
        var graph = new Proto();
        foreach (var n in nodes) graph.Bytes(1, n);
        graph.String(2, "test");
        foreach (var t in inits) graph.Bytes(5, t);
        foreach (var i in inputs) graph.Bytes(11, i);
        foreach (var o in outputs) graph.Bytes(12, o);
        var opset = new Proto().String(1, "").Int(2, 13).ToArray();
        return new Proto().Int(1, 8).Bytes(8, opset).Bytes(7, graph.ToArray()).ToArray();
    }

    static byte[] Node(string op, string[] inputs, string[] outputs, params byte[][] attrs)
    {
        var p = new Proto();
        foreach (var i in inputs) p.String(1, i);
        foreach (var o in outputs) p.String(2, o);
        p.String(4, op);
        foreach (var a in attrs) p.Bytes(5, a);
        return p.ToArray();
    }

    static byte[] AttrInts(string name, long[] values)
    {
        var p = new Proto().String(1, name);
        foreach (var v in values) p.Int(8, v);
        return p.Int(20, 7).ToArray(); // INTS
    }

    static byte[] TensorI64(string name, long[] dims, long[] data)
    {
        var p = new Proto();
        foreach (var d in dims) p.Int(1, d);
        p.Int(2, 7);
        var raw = new byte[data.Length * 8];
        Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
        p.String(8, name);
        p.Bytes(9, raw);
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
