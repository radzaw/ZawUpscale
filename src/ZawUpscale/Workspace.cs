using System.Text.Json;

namespace ZawUpscale;

/// <summary>Persists the UI's state document (preferences, upscale settings, recent folders) in the configuration directory.</summary>
public sealed class Workspace(AppOptions o)
{
    readonly string dir = o.ConfigDir;
    readonly Lock gate = new();
    public static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);
    static readonly string[] Documents = ["state"];

    public static bool IsDocument(string name) => Documents.Contains(name);

    /// <summary>A stored document, or default (JSON null) when there is none yet.</summary>
    public JsonElement Load(string name)
    {
        var path = Path.Combine(dir, name + ".json");
        lock (gate)
        {
            if (!File.Exists(path)) return default;
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(path));
                return doc.RootElement.Clone();
            }
            catch (JsonException)
            {
                // Keep the damaged file for the user and start over rather than refusing to open.
                File.Copy(path, path + ".broken", overwrite: true);
                return default;
            }
        }
    }

    public void Save(string name, JsonElement doc)
    {
        lock (gate) WriteAtomic(Path.Combine(dir, name + ".json"), doc.GetRawText());
    }

    static void WriteAtomic(string path, string text)
    {
        var tmp = path + ".tmp";
        File.WriteAllText(tmp, text);
        File.Move(tmp, path, overwrite: true);
    }
}
