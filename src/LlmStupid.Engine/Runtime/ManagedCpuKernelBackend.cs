using LlmStupid.Engine.Model;

namespace LlmStupid.Engine.Runtime;

/// <summary>Reference managed backend with scalar fallback and an AVX2 fast path.</summary>
public sealed class ManagedCpuKernelBackend(KernelMode mode = KernelMode.Auto) : IQuantKernelBackend
{
    public string Id => mode switch
    {
        KernelMode.Scalar => "managed-cpu-scalar",
        KernelMode.Avx2 => "managed-cpu-avx2",
        _ => QuantKernels.IsAvx2Available ? "managed-cpu-avx2" : "managed-cpu-scalar",
    };

    public bool SupportsQuantizedInput => mode != KernelMode.Scalar && QuantKernels.IsAvx2Available;

    public float DotFloat(ReadOnlySpan<float> left, ReadOnlySpan<float> right) => QuantKernels.DotFloat(left, right, mode);

    public float Dot(GgmlType type, ReadOnlySpan<byte> weights, ReadOnlySpan<float> input) => QuantKernels.Dot(type, weights, input, mode);

    public float DotQuantized(GgmlType type, ReadOnlySpan<byte> weights, QuantizedInputVector input)
    {
        if (!SupportsQuantizedInput) throw new PlatformNotSupportedException("This managed CPU backend does not support AVX2 quantized-input kernels.");
        return QuantKernels.DotQuantized(type, weights, input);
    }

    public bool TryDotQuantizedRows(
        GgmlType type,
        ReadOnlySpan<byte> weights,
        int rowCount,
        QuantizedInputVector input,
        Span<float> destination)
    {
        if (!SupportsQuantizedInput) return false;
        return QuantKernels.TryDotQuantizedRows(type, weights, rowCount, input, destination);
    }

    public void DecodeRow(GgmlType type, ReadOnlySpan<byte> weights, Span<float> destination) => QuantKernels.DecodeRow(type, weights, destination);
}
