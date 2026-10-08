using LlmStupid.Engine.Model;

namespace LlmStupid.Engine.Runtime;

/// <summary>Pluggable CPU/GPU math boundary used by the model execution graph.</summary>
public interface IQuantKernelBackend
{
    string Id { get; }

    bool SupportsQuantizedInput { get; }

    float DotFloat(ReadOnlySpan<float> left, ReadOnlySpan<float> right);

    float Dot(GgmlType type, ReadOnlySpan<byte> weights, ReadOnlySpan<float> input);

    float DotQuantized(GgmlType type, ReadOnlySpan<byte> weights, QuantizedInputVector input);

    /// <summary>
    /// Optionally computes a contiguous group of quantized output rows. The weight span contains
    /// <paramref name="rowCount"/> complete rows in matrix order. Return false when this backend
    /// does not have a suitable row-tiled kernel; callers then use <see cref="DotQuantized"/>.
    /// </summary>
    bool TryDotQuantizedRows(
        GgmlType type,
        ReadOnlySpan<byte> weights,
        int rowCount,
        QuantizedInputVector input,
        Span<float> destination) => false;

    void DecodeRow(GgmlType type, ReadOnlySpan<byte> weights, Span<float> destination);
}
