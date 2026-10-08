namespace LlmStupid.Engine.Model;

public sealed class GgufMetadata(IReadOnlyDictionary<string, object> values)
{
    public IReadOnlyDictionary<string, object> Values { get; } = values;

    public string GetString(string key) => Values.TryGetValue(key, out var value) && value is string text
        ? text
        : throw new InvalidDataException($"Missing or invalid GGUF string metadata '{key}'.");

    public string? GetOptionalString(string key) => Values.TryGetValue(key, out var value) && value is string text ? text : null;

    public long GetInt64(string key) => Values.TryGetValue(key, out var value)
        ? Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture)
        : throw new InvalidDataException($"Missing GGUF numeric metadata '{key}'.");

    public float GetSingle(string key) => Values.TryGetValue(key, out var value)
        ? Convert.ToSingle(value, System.Globalization.CultureInfo.InvariantCulture)
        : throw new InvalidDataException($"Missing GGUF numeric metadata '{key}'.");

    public string[] GetStringArray(string key) => Values.TryGetValue(key, out var value) && value is string[] array
        ? array
        : throw new InvalidDataException($"Missing or invalid GGUF string-array metadata '{key}'.");

    public int[] GetInt32Array(string key) => Values.TryGetValue(key, out var value) && value is int[] array
        ? array
        : throw new InvalidDataException($"Missing or invalid GGUF int-array metadata '{key}'.");
}
