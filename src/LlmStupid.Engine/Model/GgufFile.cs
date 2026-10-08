using System.IO.MemoryMappedFiles;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace LlmStupid.Engine.Model;

public sealed unsafe class GgufFile : IDisposable
{
    private const uint Magic = 0x46554747;
    private const int MaximumMetadataEntries = 1_000_000;
    private const int MaximumTensorCount = 1_000_000;
    private const ulong MaximumStringBytes = 128 * 1024 * 1024;
    private readonly MemoryMappedFile _mappedFile;
    private readonly MemoryMappedViewAccessor _view;
    private readonly SafeMemoryMappedViewHandle _viewHandle;
    private byte* _baseAddress;
    private bool _pointerAcquired;
    private bool _disposed;

    private GgufFile(
        string path,
        uint version,
        GgufMetadata metadata,
        IReadOnlyDictionary<string, GgufTensor> tensors,
        long dataStart,
        long fileLength)
    {
        Path = path;
        Version = version;
        Metadata = metadata;
        Tensors = tensors;
        DataStart = dataStart;
        FileLength = fileLength;
        _mappedFile = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        _view = _mappedFile.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        _viewHandle = _view.SafeMemoryMappedViewHandle;
        byte* pointer = null;
        _viewHandle.AcquirePointer(ref pointer);
        _pointerAcquired = true;
        _baseAddress = pointer + _view.PointerOffset;
    }

    public string Path { get; }

    public uint Version { get; }

    public GgufMetadata Metadata { get; }

    public IReadOnlyDictionary<string, GgufTensor> Tensors { get; }

    public long DataStart { get; }

    public long FileLength { get; }

    public static GgufFile Open(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var fullPath = System.IO.Path.GetFullPath(path);
        using var stream = new FileStream(fullPath, FileMode.Open, FileAccess.Read, FileShare.Read, 64 * 1024, FileOptions.SequentialScan);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        if (reader.ReadUInt32() != Magic)
        {
            throw new InvalidDataException("The file does not start with the GGUF magic value.");
        }

        var version = reader.ReadUInt32();
        if (version is < 2 or > 3)
        {
            throw new NotSupportedException($"GGUF version {version} is not supported; expected v2 or v3.");
        }

        var tensorCount = reader.ReadUInt64();
        var metadataCount = reader.ReadUInt64();
        if (tensorCount is 0 or > MaximumTensorCount || metadataCount > MaximumMetadataEntries)
        {
            throw new InvalidDataException("GGUF header declares an unreasonable number of entries.");
        }

        var metadata = new Dictionary<string, object>(checked((int)metadataCount), StringComparer.Ordinal);
        for (ulong i = 0; i < metadataCount; i++)
        {
            var key = ReadString(reader);
            var valueType = reader.ReadUInt32();
            if (!metadata.TryAdd(key, ReadValue(reader, valueType)))
            {
                throw new InvalidDataException($"Duplicate GGUF metadata key '{key}'.");
            }
        }

        var tensors = new Dictionary<string, GgufTensor>(checked((int)tensorCount), StringComparer.Ordinal);
        for (ulong i = 0; i < tensorCount; i++)
        {
            var name = ReadString(reader);
            var rank = reader.ReadUInt32();
            if (rank is 0 or > 4)
            {
                throw new InvalidDataException($"Tensor '{name}' has unsupported rank {rank}.");
            }

            var dimensions = new long[checked((int)rank)];
            for (var dimension = 0; dimension < dimensions.Length; dimension++)
            {
                var size = reader.ReadUInt64();
                if (size is 0 or > long.MaxValue)
                {
                    throw new InvalidDataException($"Tensor '{name}' has invalid dimension {size}.");
                }

                dimensions[dimension] = (long)size;
            }

            var typeId = reader.ReadUInt32();
            var type = (GgmlType)typeId;
            var offset = reader.ReadUInt64();
            if (!tensors.TryAdd(name, new GgufTensor(name, dimensions, type, offset)))
            {
                throw new InvalidDataException($"Duplicate GGUF tensor '{name}'.");
            }
        }

        var alignment = metadata.TryGetValue("general.alignment", out var rawAlignment)
            ? Convert.ToInt64(rawAlignment, System.Globalization.CultureInfo.InvariantCulture)
            : 32;
        if (alignment is < 1 or > 4096 || (alignment & (alignment - 1)) != 0)
        {
            throw new InvalidDataException($"Invalid GGUF tensor alignment {alignment}.");
        }

        var dataStart = checked((stream.Position + alignment - 1) / alignment * alignment);
        var fileLength = stream.Length;
        var dataBytes = fileLength - dataStart;
        foreach (var tensor in tensors.Values)
        {
            if (tensor.Offset % (ulong)alignment != 0 || tensor.Offset >= (ulong)dataBytes)
            {
                throw new InvalidDataException($"Tensor '{tensor.Name}' has an invalid or unaligned data offset.");
            }

            if (tensor.Type is GgmlType.F32 or GgmlType.F16 or GgmlType.Q4K or GgmlType.Q6K)
            {
                var length = tensor.ByteLength;
                if (tensor.Offset + (ulong)length > (ulong)dataBytes)
                {
                    throw new InvalidDataException($"Tensor '{tensor.Name}' extends beyond the GGUF data section.");
                }
            }
        }

        return new GgufFile(fullPath, version, new GgufMetadata(metadata), tensors, dataStart, fileLength);
    }

    public GgufTensor RequireTensor(string name) => Tensors.TryGetValue(name, out var tensor)
        ? tensor
        : throw new InvalidDataException($"Required GGUF tensor '{name}' is missing.");

    public ReadOnlySpan<byte> GetTensorData(GgufTensor tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var absoluteOffset = checked(DataStart + (long)tensor.Offset);
        return new ReadOnlySpan<byte>(_baseAddress + absoluteOffset, tensor.ByteLength);
    }

    internal GgufTensorDataView CreateTensorDataView(GgufTensor tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new GgufTensorDataView(this, checked(DataStart + (long)tensor.Offset), tensor.ByteLength);
    }

    internal ReadOnlySpan<byte> GetTensorDataSlice(long absoluteOffset, int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return new ReadOnlySpan<byte>(_baseAddress + absoluteOffset, length);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_pointerAcquired)
        {
            _viewHandle.ReleasePointer();
            _pointerAcquired = false;
        }

        _view.Dispose();
        _mappedFile.Dispose();
    }

    private static object ReadValue(BinaryReader reader, uint type) => type switch
    {
        0 => reader.ReadByte(),
        1 => reader.ReadSByte(),
        2 => reader.ReadUInt16(),
        3 => reader.ReadInt16(),
        4 => reader.ReadUInt32(),
        5 => reader.ReadInt32(),
        6 => reader.ReadSingle(),
        7 => reader.ReadByte() != 0,
        8 => ReadString(reader),
        9 => ReadArray(reader),
        10 => reader.ReadUInt64(),
        11 => reader.ReadInt64(),
        12 => reader.ReadDouble(),
        _ => throw new InvalidDataException($"Unknown GGUF metadata value type {type}.")
    };

    private static object ReadArray(BinaryReader reader)
    {
        var elementType = reader.ReadUInt32();
        var count = reader.ReadUInt64();
        if (count > 10_000_000)
        {
            throw new InvalidDataException($"GGUF array declares an unreasonable element count {count}.");
        }

        if (elementType == 8)
        {
            var values = new string[checked((int)count)];
            for (var i = 0; i < values.Length; i++) values[i] = ReadString(reader);
            return values;
        }

        if (elementType == 5)
        {
            var values = new int[checked((int)count)];
            for (var i = 0; i < values.Length; i++) values[i] = reader.ReadInt32();
            return values;
        }

        if (elementType == 4)
        {
            var values = new uint[checked((int)count)];
            for (var i = 0; i < values.Length; i++) values[i] = reader.ReadUInt32();
            return values;
        }

        if (elementType == 6)
        {
            var values = new float[checked((int)count)];
            for (var i = 0; i < values.Length; i++) values[i] = reader.ReadSingle();
            return values;
        }

        throw new NotSupportedException($"GGUF metadata arrays of value type {elementType} are not used by the supported model and are not supported.");
    }

    private static string ReadString(BinaryReader reader)
    {
        var length = reader.ReadUInt64();
        if (length > MaximumStringBytes || length > int.MaxValue)
        {
            throw new InvalidDataException($"GGUF string has unreasonable length {length}.");
        }

        var bytes = reader.ReadBytes((int)length);
        if ((ulong)bytes.Length != length)
        {
            throw new EndOfStreamException("GGUF string ended before its declared length.");
        }

        return Encoding.UTF8.GetString(bytes);
    }
}

internal sealed class GgufTensorDataView(GgufFile file, long absoluteOffset, int length)
{
    public ReadOnlySpan<byte> Slice(int offset, int sliceLength)
    {
        if (offset < 0 || sliceLength < 0 || offset > length - sliceLength)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), "Tensor data slice is outside the mapped tensor.");
        }

        return file.GetTensorDataSlice(checked(absoluteOffset + offset), sliceLength);
    }
}
