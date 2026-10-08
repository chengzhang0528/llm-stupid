namespace LlmStupid.Engine.Runtime;

public sealed class QuantizedInputVector(int length)
{
    private readonly sbyte[] _values = new sbyte[length];
    private readonly float[] _scales = new float[length / 32];
    // Q4_K and Q6_K dot products need these 16-value sums; compute them while quantizing.
    private readonly short[] _sums16 = new short[length / 16];

    public int Length { get; } = length;

    internal sbyte[] Values => _values;

    internal float[] Scales => _scales;

    internal short[] Sums16 => _sums16;

    public void Quantize(ReadOnlySpan<float> source)
    {
        if (source.Length != Length || (Length & 31) != 0)
        {
            throw new ArgumentException("Q8 activation inputs must match the buffer length and be aligned to 32 values.", nameof(source));
        }

        for (var block = 0; block < _scales.Length; block++)
        {
            var values = source.Slice(block * 32, 32);
            var maximum = 0f;
            for (var i = 0; i < values.Length; i++) maximum = MathF.Max(maximum, MathF.Abs(values[i]));
            var scale = maximum / 127f;
            _scales[block] = scale;
            if (scale == 0)
            {
                Array.Clear(_values, block * 32, 32);
                _sums16[block * 2] = 0;
                _sums16[block * 2 + 1] = 0;
                continue;
            }

            var inverseScale = 1f / scale;
            var firstSum = 0;
            var secondSum = 0;
            for (var i = 0; i < values.Length; i++)
            {
                var quantized = (int)MathF.Round(values[i] * inverseScale, MidpointRounding.AwayFromZero);
                var quantizedValue = (sbyte)Math.Clamp(quantized, -127, 127);
                _values[block * 32 + i] = quantizedValue;
                if (i < 16) firstSum += quantizedValue;
                else secondSum += quantizedValue;
            }

            _sums16[block * 2] = checked((short)firstSum);
            _sums16[block * 2 + 1] = checked((short)secondSum);
        }
    }
}
