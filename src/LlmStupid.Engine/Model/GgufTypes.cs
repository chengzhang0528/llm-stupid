namespace LlmStupid.Engine.Model;

public enum GgmlType : uint
{
    F32 = 0,
    F16 = 1,
    Q4K = 12,
    Q6K = 14,
}

public sealed record GgufTensor(string Name, long[] Dimensions, GgmlType Type, ulong Offset)
{
    public long ElementCount => Dimensions.Aggregate(1L, static (product, dimension) => checked(product * dimension));

    public int RowLength => checked((int)Dimensions[0]);

    public int RowCount => checked((int)(ElementCount / RowLength));

    public int BlockSize => Type switch
    {
        GgmlType.F32 => 1,
        GgmlType.F16 => 1,
        GgmlType.Q4K or GgmlType.Q6K => 256,
        _ => throw new NotSupportedException($"Unsupported GGML tensor type {(uint)Type} for {Name}.")
    };

    public int BytesPerBlock => Type switch
    {
        GgmlType.F32 => 4,
        GgmlType.F16 => 2,
        GgmlType.Q4K => 144,
        GgmlType.Q6K => 210,
        _ => throw new NotSupportedException($"Unsupported GGML tensor type {(uint)Type} for {Name}.")
    };

    public int ByteLength
    {
        get
        {
            var values = ElementCount;
            if (values % BlockSize != 0)
            {
                throw new InvalidDataException($"Tensor {Name} has {values} values, not aligned to block size {BlockSize}.");
            }

            return checked((int)(values / BlockSize * BytesPerBlock));
        }
    }
}
