using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace ZawUpscale;

/// <summary>
/// The CLIP byte-level BPE tokenizer (as in OpenAI's CLIP and Hugging Face's CLIPTokenizer), reading the
/// usual vocab.json and merges.txt. SDXL's two text encoders share it; they differ only in padding.
/// </summary>
public sealed class ClipTokenizer
{
    public const int Length = 77;
    public const int StartOfText = 49406, EndOfText = 49407;

    static readonly Regex Pattern = new(@"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);
    static readonly Regex Spaces = new(@"\s+", RegexOptions.Compiled);
    static readonly string[] ByteChars = BytesToUnicode();

    readonly Dictionary<string, int> vocab;
    readonly Dictionary<(string, string), int> ranks = [];
    readonly Dictionary<string, string[]> cache = [];
    readonly int start, end;

    public ClipTokenizer(Dictionary<string, int> vocab, IEnumerable<string> merges)
    {
        this.vocab = vocab;
        var rank = 0;
        foreach (var line in merges)
        {
            if (line.StartsWith("#version") || string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(' ');
            if (parts.Length == 2) ranks.TryAdd((parts[0], parts[1]), rank++);
        }
        start = vocab.GetValueOrDefault("<|startoftext|>", StartOfText);
        end = vocab.GetValueOrDefault("<|endoftext|>", EndOfText);
    }

    public static ClipTokenizer Load(string vocabPath, string mergesPath)
    {
        var vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(vocabPath))
            ?? throw new ApiException($"Empty tokenizer vocabulary: {vocabPath}");
        return new ClipTokenizer(vocab, File.ReadLines(mergesPath));
    }

    /// <summary>
    /// The 77 token ids the text encoders take: start, the text (cut to fit), end, then padding.
    /// The CLIP ViT-L encoder pads with the end token, OpenCLIP's bigG with 0.
    /// </summary>
    public long[] Encode(string text, int pad)
    {
        var ids = new long[Length];
        Array.Fill(ids, pad);
        ids[0] = start;
        var n = 1;
        foreach (var t in Tokens(text))
        {
            if (n == Length - 1) break;
            ids[n++] = t;
        }
        ids[n] = end;
        return ids;
    }

    /// <summary>The text's token ids, without start and end.</summary>
    public List<int> Tokens(string text)
    {
        // CLIP's basic_clean and whitespace_clean (without ftfy's mojibake repair).
        text = WebUtility.HtmlDecode(WebUtility.HtmlDecode(text));
        text = Spaces.Replace(text, " ").Trim().ToLowerInvariant();
        var result = new List<int>();
        foreach (Match m in Pattern.Matches(text))
        {
            var sb = new StringBuilder();
            foreach (var b in Encoding.UTF8.GetBytes(m.Value)) sb.Append(ByteChars[b]);
            foreach (var piece in Bpe(sb.ToString()))
                if (vocab.TryGetValue(piece, out var id)) result.Add(id);
        }
        return result;
    }

    string[] Bpe(string token)
    {
        if (cache.TryGetValue(token, out var hit)) return hit;
        // Characters (surrogate pairs kept whole), the last one marked as the end of a word.
        var word = new List<string>();
        for (var i = 0; i < token.Length; i += char.IsSurrogatePair(token, i) ? 2 : 1)
            word.Add(token.Substring(i, char.IsSurrogatePair(token, i) ? 2 : 1));
        word[^1] += "</w>";
        while (word.Count > 1)
        {
            // Merge the best-ranked adjacent pair, everywhere it occurs.
            var best = int.MaxValue;
            (string, string) pair = default;
            for (var i = 0; i < word.Count - 1; i++)
                if (ranks.TryGetValue((word[i], word[i + 1]), out var r) && r < best) { best = r; pair = (word[i], word[i + 1]); }
            if (best == int.MaxValue) break;
            var merged = new List<string>(word.Count);
            for (var i = 0; i < word.Count; i++)
            {
                if (i < word.Count - 1 && word[i] == pair.Item1 && word[i + 1] == pair.Item2) { merged.Add(pair.Item1 + pair.Item2); i++; }
                else merged.Add(word[i]);
            }
            word = merged;
        }
        var result = word.ToArray();
        if (cache.Count < 10_000) cache[token] = result;
        return result;
    }

    /// <summary>GPT-2's reversible mapping of bytes to printable characters.</summary>
    static string[] BytesToUnicode()
    {
        var map = new string[256];
        var n = 0;
        for (var b = 0; b < 256; b++)
        {
            var printable = b is >= '!' and <= '~' or >= 0xA1 and <= 0xAC or >= 0xAE and <= 0xFF;
            map[b] = ((char)(printable ? b : 256 + n++)).ToString();
        }
        return map;
    }
}
