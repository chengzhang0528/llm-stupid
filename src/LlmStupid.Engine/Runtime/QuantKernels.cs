using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using LlmStupid.Engine.Model;

namespace LlmStupid.Engine.Runtime;

public static class QuantKernels
{
    public static bool IsAvx2Available => Avx2.IsSupported;

    public static float DotFloat(ReadOnlySpan<float> left, ReadOnlySpan<float> right, KernelMode mode = KernelMode.Auto)
    {
        var useAvx = mode switch
        {
            KernelMode.Scalar => false,
            KernelMode.Avx2 when !Avx2.IsSupported => throw new PlatformNotSupportedException("AVX2 is not available on this processor."),
            KernelMode.Avx2 => true,
            _ => Avx2.IsSupported,
        };
        return DotFloatSpans(left, right, useAvx);
    }

    public static float Dot(GgmlType type, ReadOnlySpan<byte> weights, ReadOnlySpan<float> input, KernelMode mode = KernelMode.Auto)
    {
        if (weights.Length == 0 || input.Length == 0) return 0;
        var useAvx = mode switch
        {
            KernelMode.Scalar => false,
            KernelMode.Avx2 when !Avx2.IsSupported => throw new PlatformNotSupportedException("AVX2 is not available on this processor."),
            KernelMode.Avx2 => true,
            _ => Avx2.IsSupported,
        };

        return type switch
        {
            GgmlType.F32 => DotF32(weights, input, useAvx),
            GgmlType.F16 => DotF16(weights, input, useAvx),
            GgmlType.Q4K => DotQuantized(weights, input, useAvx, q4: true),
            GgmlType.Q6K => DotQuantized(weights, input, useAvx, q4: false),
            _ => throw new NotSupportedException($"No dot-product kernel is available for GGML type {(uint)type}.")
        };
    }

    public static float DotQuantized(GgmlType type, ReadOnlySpan<byte> weights, QuantizedInputVector input)
    {
        if (!Avx2.IsSupported) throw new PlatformNotSupportedException("Q8 quantized activation kernels require AVX2.");
        return type switch
        {
            GgmlType.Q4K => DotQ4KQ8(weights, input),
            GgmlType.Q6K => DotQ6KQ8(weights, input),
            _ => throw new NotSupportedException($"Q8 activation kernels are not available for GGML type {(uint)type}.")
        };
    }

    internal static bool TryDotQuantizedRows(
        GgmlType type,
        ReadOnlySpan<byte> weights,
        int rowCount,
        QuantizedInputVector input,
        Span<float> destination)
    {
        if (!Avx2.IsSupported || type != GgmlType.Q4K || rowCount is not (4 or 8 or 16)) return false;
        DotQ4KQ8Rows(weights, input, rowCount, destination);
        return true;
    }

    public static void DecodeRow(GgmlType type, ReadOnlySpan<byte> weights, Span<float> destination)
    {
        switch (type)
        {
            case GgmlType.F32:
            {
                var values = MemoryMarshal.Cast<byte, float>(weights);
                if (values.Length != destination.Length) throw new ArgumentException("F32 row length does not match destination.");
                values.CopyTo(destination);
                return;
            }
            case GgmlType.F16:
            {
                var values = MemoryMarshal.Cast<byte, Half>(weights);
                if (values.Length != destination.Length) throw new ArgumentException("F16 row length does not match destination.");
                for (var i = 0; i < values.Length; i++) destination[i] = (float)values[i];
                return;
            }
            case GgmlType.Q4K:
                DecodeQuantizedRow(weights, destination, q4: true);
                return;
            case GgmlType.Q6K:
                DecodeQuantizedRow(weights, destination, q4: false);
                return;
            default:
                throw new NotSupportedException($"No row decoder is available for GGML type {(uint)type}.");
        }
    }

    private static float DotF32(ReadOnlySpan<byte> weights, ReadOnlySpan<float> input, bool useAvx)
    {
        var values = MemoryMarshal.Cast<byte, float>(weights);
        if (values.Length != input.Length) throw new ArgumentException("F32 weight and input lengths differ.");
        return DotFloatSpans(values, input, useAvx);
    }

    private static float DotF16(ReadOnlySpan<byte> weights, ReadOnlySpan<float> input, bool useAvx)
    {
        var values = MemoryMarshal.Cast<byte, Half>(weights);
        if (values.Length != input.Length) throw new ArgumentException("F16 weight and input lengths differ.");
        Span<float> scratch = input.Length <= 1024 ? stackalloc float[input.Length] : new float[input.Length];
        for (var i = 0; i < values.Length; i++) scratch[i] = (float)values[i];
        return DotFloatSpans(scratch, input, useAvx);
    }

    private static float DotQuantized(ReadOnlySpan<byte> weights, ReadOnlySpan<float> input, bool useAvx, bool q4)
    {
        const int blockElements = 256;
        var blockBytes = q4 ? 144 : 210;
        if (input.Length % blockElements != 0 || weights.Length != input.Length / blockElements * blockBytes)
        {
            throw new ArgumentException("Quantized row length is not aligned to its block format.");
        }

        Span<float> decoded = stackalloc float[blockElements];
        var sum = 0f;
        var vectorSum = Vector256<float>.Zero;
        for (var block = 0; block < input.Length / blockElements; block++)
        {
            var blockWeights = weights.Slice(block * blockBytes, blockBytes);
            var blockInput = input.Slice(block * blockElements, blockElements);
            if (q4) DecodeQ4KBlock(blockWeights, decoded);
            else DecodeQ6KBlock(blockWeights, decoded);

            if (useAvx)
            {
                for (var i = 0; i < blockElements; i += 8)
                {
                    var left = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(decoded.Slice(i, 8)));
                    var right = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(blockInput.Slice(i, 8)));
                    vectorSum = Fma.IsSupported
                        ? Fma.MultiplyAdd(left, right, vectorSum)
                        : Avx.Add(vectorSum, Avx.Multiply(left, right));
                }
            }
            else
            {
                for (var i = 0; i < blockElements; i++) sum += decoded[i] * blockInput[i];
            }
        }

        return useAvx ? sum + Vector256.Sum(vectorSum) : sum;
    }

    private static float DotFloatSpans(ReadOnlySpan<float> weights, ReadOnlySpan<float> input, bool useAvx)
    {
        if (weights.Length != input.Length) throw new ArgumentException("Weight and input lengths differ.");
        var index = 0;
        var vectorSum = Vector256<float>.Zero;
        if (useAvx)
        {
            var vectorEnd = weights.Length & ~7;
            for (; index < vectorEnd; index += 8)
            {
                var left = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(weights.Slice(index, 8)));
                var right = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(input.Slice(index, 8)));
                vectorSum = Fma.IsSupported
                    ? Fma.MultiplyAdd(left, right, vectorSum)
                    : Avx.Add(vectorSum, Avx.Multiply(left, right));
            }
        }

        var sum = useAvx ? Vector256.Sum(vectorSum) : 0;
        for (; index < weights.Length; index++) sum += weights[index] * input[index];
        return sum;
    }

    private static void DecodeQuantizedRow(ReadOnlySpan<byte> weights, Span<float> destination, bool q4)
    {
        const int blockElements = 256;
        var blockBytes = q4 ? 144 : 210;
        if (destination.Length % blockElements != 0 || weights.Length != destination.Length / blockElements * blockBytes)
        {
            throw new ArgumentException("Quantized row length is not aligned to its block format.");
        }

        for (var block = 0; block < destination.Length / blockElements; block++)
        {
            var source = weights.Slice(block * blockBytes, blockBytes);
            var target = destination.Slice(block * blockElements, blockElements);
            if (q4) DecodeQ4KBlock(source, target);
            else DecodeQ6KBlock(source, target);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ReadHalf(ReadOnlySpan<byte> source, int offset) =>
        (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(source.Slice(offset, 2)));

    private static void DecodeQ4KBlock(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var d = ReadHalf(source, 0);
        var dMin = ReadHalf(source, 2);
        var scales = source.Slice(4, 12);
        var quants = source.Slice(16, 128);
        var scaleIndex = 0;
        for (var start = 0; start < 256; start += 64)
        {
            GetScaleMin(scales, scaleIndex, out var scale0, out var min0);
            GetScaleMin(scales, scaleIndex + 1, out var scale1, out var min1);
            var scaleValue0 = d * scale0;
            var minValue0 = dMin * min0;
            var scaleValue1 = d * scale1;
            var minValue1 = dMin * min1;
            var quantBase = (start / 64) * 32;
            for (var i = 0; i < 32; i++)
            {
                var packed = quants[quantBase + i];
                destination[start + i] = scaleValue0 * (packed & 0x0f) - minValue0;
                destination[start + 32 + i] = scaleValue1 * (packed >> 4) - minValue1;
            }

            scaleIndex += 2;
        }
    }

    private static void GetScaleMin(ReadOnlySpan<byte> scales, int index, out int scale, out int min)
    {
        if (index < 4)
        {
            scale = scales[index] & 0x3f;
            min = scales[index + 4] & 0x3f;
        }
        else
        {
            scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
            min = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
        }
    }

    private static void DecodeQ6KBlock(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var low = source.Slice(0, 128);
        var high = source.Slice(128, 64);
        var scales = source.Slice(192, 16);
        var d = ReadHalf(source, 208);
        for (var start = 0; start < 256; start += 128)
        {
            var lowBase = (start / 128) * 64;
            var highBase = (start / 128) * 32;
            var scaleBase = (start / 128) * 8;
            for (var i = 0; i < 32; i++)
            {
                var highBits = high[highBase + i];
                var q1 = (low[lowBase + i] & 0x0f) | ((highBits & 0x03) << 4);
                var q2 = (low[lowBase + i + 32] & 0x0f) | (((highBits >> 2) & 0x03) << 4);
                var q3 = (low[lowBase + i] >> 4) | (((highBits >> 4) & 0x03) << 4);
                var q4 = (low[lowBase + i + 32] >> 4) | (((highBits >> 6) & 0x03) << 4);
                destination[start + i] = d * unchecked((sbyte)scales[scaleBase + i / 16]) * (q1 - 32);
                destination[start + 32 + i] = d * unchecked((sbyte)scales[scaleBase + 2 + i / 16]) * (q2 - 32);
                destination[start + 64 + i] = d * unchecked((sbyte)scales[scaleBase + 4 + i / 16]) * (q3 - 32);
                destination[start + 96 + i] = d * unchecked((sbyte)scales[scaleBase + 6 + i / 16]) * (q4 - 32);
            }
        }
    }

    private static float DotQ4KQ8(ReadOnlySpan<byte> weights, QuantizedInputVector input)
    {
        const int blockElements = 256;
        const int blockBytes = 144;
        if (input.Length % blockElements != 0 || weights.Length != input.Length / blockElements * blockBytes)
        {
            throw new ArgumentException("Q4_K and Q8 activation row lengths differ or are not block-aligned.");
        }

        var qValues = input.Values;
        var qScales = input.Scales;
        var qSums16 = input.Sums16;
        var total = 0f;
        for (var block = 0; block < input.Length / blockElements; block++)
        {
            var source = weights.Slice(block * blockBytes, blockBytes);
            var d = ReadHalf(source, 0);
            var dMin = ReadHalf(source, 2);
            var scales = source.Slice(4, 12);
            var packed = source.Slice(16, 128);
            var activationBase = block * blockElements;
            var scaleIndex = 0;
            for (var start = 0; start < blockElements; start += 64)
            {
                GetScaleMin(scales, scaleIndex, out var scale0, out var min0);
                GetScaleMin(scales, scaleIndex + 1, out var scale1, out var min1);
                var quantBase = (start / 64) * 32;
                var inputBase = activationBase + start;
                var lowDot = DotNibble32(packed.Slice(quantBase, 32), qValues.AsSpan(inputBase, 32), highNibble: false);
                var highDot = DotNibble32(packed.Slice(quantBase, 32), qValues.AsSpan(inputBase + 32, 32), highNibble: true);
                var lowSumIndex = inputBase / 16;
                var highSumIndex = lowSumIndex + 2;
                var lowInputSum = qSums16[lowSumIndex] + qSums16[lowSumIndex + 1];
                var highInputSum = qSums16[highSumIndex] + qSums16[highSumIndex + 1];
                var lowInputScale = qScales[inputBase / 32];
                var highInputScale = qScales[(inputBase + 32) / 32];
                total += d * lowInputScale * scale0 * lowDot - dMin * lowInputScale * min0 * lowInputSum;
                total += d * highInputScale * scale1 * highDot - dMin * highInputScale * min1 * highInputSum;
                scaleIndex += 2;
            }
        }

        return total;
    }

    private static void DotQ4KQ8Rows(ReadOnlySpan<byte> weights, QuantizedInputVector input, int rows, Span<float> destination)
    {
        const int blockElements = 256;
        const int blockBytes = 144;
        if (input.Length % blockElements != 0 || destination.Length < rows)
        {
            throw new ArgumentException("Q4_K row tile dimensions are invalid.");
        }

        var blocksPerRow = input.Length / blockElements;
        var rowBytes = checked(blocksPerRow * blockBytes);
        if (weights.Length != checked(rows * rowBytes))
        {
            throw new ArgumentException("Q4_K row tile weight length does not match its input and row count.");
        }

        Span<float> totals = stackalloc float[rows];
        totals.Clear();
        var qValues = input.Values;
        var qScales = input.Scales;
        var qSums16 = input.Sums16;
        for (var block = 0; block < blocksPerRow; block++)
        {
            var activationBase = block * blockElements;
            var scaleIndex = 0;
            for (var start = 0; start < blockElements; start += 64)
            {
                var inputBase = activationBase + start;
                var lowInput = qValues.AsSpan(inputBase, 32);
                var highInput = qValues.AsSpan(inputBase + 32, 32);
                var lowVector = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(lowInput));
                var highVector = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(highInput));
                var lowSumIndex = inputBase / 16;
                var highSumIndex = lowSumIndex + 2;
                var lowInputSum = qSums16[lowSumIndex] + qSums16[lowSumIndex + 1];
                var highInputSum = qSums16[highSumIndex] + qSums16[highSumIndex + 1];
                var lowInputScale = qScales[inputBase / 32];
                var highInputScale = qScales[(inputBase + 32) / 32];
                var quantBase = (start / 64) * 32;

                for (var row = 0; row < rows; row++)
                {
                    var source = weights.Slice(row * rowBytes + block * blockBytes, blockBytes);
                    var d = ReadHalf(source, 0);
                    var dMin = ReadHalf(source, 2);
                    var scales = source.Slice(4, 12);
                    var packed = source.Slice(16, 128);
                    GetScaleMin(scales, scaleIndex, out var scale0, out var min0);
                    GetScaleMin(scales, scaleIndex + 1, out var scale1, out var min1);
                    var lowDot = DotNibble32(packed.Slice(quantBase, 32), lowVector, highNibble: false);
                    var highDot = DotNibble32(packed.Slice(quantBase, 32), highVector, highNibble: true);
                    totals[row] += d * lowInputScale * scale0 * lowDot - dMin * lowInputScale * min0 * lowInputSum;
                    totals[row] += d * highInputScale * scale1 * highDot - dMin * highInputScale * min1 * highInputSum;
                }

                scaleIndex += 2;
            }
        }

        totals.CopyTo(destination);
    }

    private static float DotQ6KQ8(ReadOnlySpan<byte> weights, QuantizedInputVector input)
    {
        const int blockElements = 256;
        const int blockBytes = 210;
        if (input.Length % blockElements != 0 || weights.Length != input.Length / blockElements * blockBytes)
        {
            throw new ArgumentException("Q6_K and Q8 activation row lengths differ or are not block-aligned.");
        }

        var qValues = input.Values;
        var qScales = input.Scales;
        var qSums16 = input.Sums16;
        var total = 0f;
        var nibbleMask = Vector256.Create((byte)0x0f);
        var twoBitMask = Vector256.Create((byte)0x03);
        var pairOnes = Vector256.Create((short)1);
        for (var block = 0; block < input.Length / blockElements; block++)
        {
            var source = weights.Slice(block * blockBytes, blockBytes);
            var low = source.Slice(0, 128);
            var high = source.Slice(128, 64);
            var scales = source.Slice(192, 16);
            var d = ReadHalf(source, 208);
            var activationBase = block * blockElements;
            for (var half = 0; half < 2; half++)
            {
                var start = half * 128;
                var lowBase = half * 64;
                var highBase = half * 32;
                var scaleBase = half * 8;
                for (var group = 0; group < 4; group++)
                {
                    var activationOffset = activationBase + start + group * 32;
                    var packedLow = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(
                        low.Slice(lowBase + (group is 1 or 3 ? 32 : 0), 32)));
                    var highBits = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(high.Slice(highBase, 32)));
                    var lowBits = group >= 2
                        ? Avx2.And(Avx2.ShiftRightLogical(packedLow.AsUInt16(), 4).AsByte(), nibbleMask)
                        : Avx2.And(packedLow, nibbleMask);
                    var shiftedHighBits = group switch
                    {
                        0 => highBits.AsByte(),
                        1 => Avx2.ShiftRightLogical(highBits.AsUInt16(), 2).AsByte(),
                        2 => Avx2.ShiftRightLogical(highBits.AsUInt16(), 4).AsByte(),
                        _ => Avx2.ShiftRightLogical(highBits.AsUInt16(), 6).AsByte(),
                    };
                    var upperBits = Avx2.And(shiftedHighBits, twoBitMask);
                    var unpacked = Avx2.Or(lowBits, Avx2.ShiftLeftLogical(upperBits.AsUInt16(), 4).AsByte());
                    var activations = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(qValues.AsSpan(activationOffset, 32)));
                    var pairSums = Avx2.MultiplyAddAdjacent(unpacked, activations);
                    var quarterSums = Avx2.MultiplyAddAdjacent(pairSums, pairOnes);
                    var firstRawDot = Vector128.Sum(quarterSums.GetLower());
                    var secondRawDot = Vector128.Sum(quarterSums.GetUpper());
                    var firstActivationSum = qSums16[activationOffset / 16];
                    var secondActivationSum = qSums16[activationOffset / 16 + 1];
                    var activationScale = qScales[activationOffset / 32];
                    var firstScale = unchecked((sbyte)scales[scaleBase + group * 2]);
                    var secondScale = unchecked((sbyte)scales[scaleBase + group * 2 + 1]);
                    total += d * firstScale * activationScale * (firstRawDot - 32 * firstActivationSum);
                    total += d * secondScale * activationScale * (secondRawDot - 32 * secondActivationSum);
                }
            }
        }

        return total;
    }

    private static int DotNibble32(ReadOnlySpan<byte> packed, ReadOnlySpan<sbyte> activations, bool highNibble)
    {
        var packedVector = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(packed));
        var mask = Vector256.Create((byte)0x0f);
        var values = Avx2.And(packedVector, mask);
        if (highNibble) values = Avx2.And(Avx2.ShiftRightLogical(packedVector.AsUInt16(), 4).AsByte(), mask);
        var activationVector = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(activations));
        return DotNibble32(values, activationVector);
    }

    private static int DotNibble32(ReadOnlySpan<byte> packed, Vector256<sbyte> activations, bool highNibble)
    {
        var packedVector = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(packed));
        var mask = Vector256.Create((byte)0x0f);
        var values = Avx2.And(packedVector, mask);
        if (highNibble) values = Avx2.And(Avx2.ShiftRightLogical(packedVector.AsUInt16(), 4).AsByte(), mask);
        return DotNibble32(values, activations);
    }

    private static int DotNibble32(Vector256<byte> values, Vector256<sbyte> activationVector)
    {
        var pairs = Avx2.MultiplyAddAdjacent(values, activationVector);
        var sums = Avx2.MultiplyAddAdjacent(pairs, Vector256.Create((short)1));
        return Vector256.Sum(sums);
    }

}
