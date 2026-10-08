using System.Buffers;
using System.Text;
using System.Text.RegularExpressions;
using LlmStupid.Engine.Model;

namespace LlmStupid.Engine.Tokenization;

public sealed class Gpt2Tokenizer
{
    private static readonly Regex PreTokenizer = new(
        @"\p{N}|(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private static readonly char[] ByteToUnicode = BuildByteToUnicode();
    private static readonly short[] UnicodeToByte = BuildUnicodeToByte();

    private readonly string[] _vocabulary;
    private readonly int[] _tokenTypes;
    private readonly Dictionary<string, int> _tokenIds;
    private readonly Dictionary<(string Left, string Right), int> _mergeRanks;
    private readonly Dictionary<char, string[]> _specialTokensByStart;

    public Gpt2Tokenizer(GgufMetadata metadata)
    {
        _vocabulary = metadata.GetStringArray("tokenizer.ggml.tokens");
        _tokenTypes = metadata.GetInt32Array("tokenizer.ggml.token_type");
        if (_vocabulary.Length != _tokenTypes.Length)
        {
            throw new InvalidDataException("Tokenizer vocabulary and token type arrays have different lengths.");
        }

        var merges = metadata.GetStringArray("tokenizer.ggml.merges");
        _tokenIds = new Dictionary<string, int>(_vocabulary.Length, StringComparer.Ordinal);
        for (var tokenId = 0; tokenId < _vocabulary.Length; tokenId++)
        {
            _tokenIds.TryAdd(_vocabulary[tokenId], tokenId);
        }

        _mergeRanks = new Dictionary<(string Left, string Right), int>(merges.Length);
        for (var rank = 0; rank < merges.Length; rank++)
        {
            var separator = merges[rank].IndexOf(' ');
            if (separator <= 0 || separator == merges[rank].Length - 1)
            {
                throw new InvalidDataException($"Invalid GPT-2 merge entry at rank {rank}.");
            }

            _mergeRanks.TryAdd((merges[rank][..separator], merges[rank][(separator + 1)..]), rank);
        }

        _specialTokensByStart = _vocabulary
            .Select((token, id) => (token, id))
            .Where(entry => IsSpecialType(_tokenTypes[entry.id]))
            .GroupBy(entry => entry.token[0])
            .ToDictionary(
                group => group.Key,
                group => group.Select(entry => entry.token).OrderByDescending(token => token.Length).ToArray());

        if (_vocabulary.Length != metadata.GetInt64("granitemoe.vocab_size"))
        {
            throw new InvalidDataException("Tokenizer vocabulary size does not match the Granite model metadata.");
        }

        EndOfSequenceTokenId = checked((int)metadata.GetInt64("tokenizer.ggml.eos_token_id"));
    }

    public int VocabularySize => _vocabulary.Length;

    public int EndOfSequenceTokenId { get; }

    public Gpt2StreamDecoder CreateStreamDecoder() => new(_vocabulary, _tokenTypes);

    internal static int GetByteValue(char symbol) => symbol < UnicodeToByte.Length ? UnicodeToByte[symbol] : -1;

    public string GetToken(int tokenId) => (uint)tokenId < (uint)_vocabulary.Length
        ? _vocabulary[tokenId]
        : throw new ArgumentOutOfRangeException(nameof(tokenId));

    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var ids = new List<int>(Math.Max(8, text.Length / 3));
        var position = 0;
        var ordinaryStart = 0;
        while (position < text.Length)
        {
            if (TryReadSpecial(text, position, out var special, out var specialId))
            {
                EncodeOrdinary(text.AsSpan(ordinaryStart, position - ordinaryStart), ids);
                ids.Add(specialId);
                position += special.Length;
                ordinaryStart = position;
                continue;
            }

            position++;
        }

        EncodeOrdinary(text.AsSpan(ordinaryStart), ids);
        return ids.ToArray();
    }

    public string Decode(IEnumerable<int> tokenIds, bool skipSpecialTokens = false)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        var bytes = new ArrayBufferWriter<byte>();
        var result = new StringBuilder();
        foreach (var tokenId in tokenIds)
        {
            if ((uint)tokenId >= (uint)_vocabulary.Length) throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Invalid token id {tokenId}.");
            var token = _vocabulary[tokenId];
            if (IsSpecialType(_tokenTypes[tokenId]))
            {
                AppendPendingBytes(result, bytes);
                if (!skipSpecialTokens) result.Append(token);
                continue;
            }

            foreach (var character in token)
            {
                if (character >= UnicodeToByte.Length || UnicodeToByte[character] < 0)
                {
                    throw new InvalidDataException($"Tokenizer piece contains unmappable byte symbol U+{(int)character:X4}.");
                }

                bytes.GetSpan(1)[0] = (byte)UnicodeToByte[character];
                bytes.Advance(1);
            }
        }

        AppendPendingBytes(result, bytes);
        return result.ToString();
    }

    public bool IsEndOfSequence(int tokenId) => tokenId == EndOfSequenceTokenId;

    private void EncodeOrdinary(ReadOnlySpan<char> text, List<int> ids)
    {
        if (text.IsEmpty) return;
        var ordinary = text.ToString();
        foreach (Match match in PreTokenizer.Matches(ordinary))
        {
            var bytes = Encoding.UTF8.GetBytes(match.Value);
            var symbols = new string[bytes.Length];
            for (var i = 0; i < bytes.Length; i++) symbols[i] = ByteToUnicode[bytes[i]].ToString();
            foreach (var piece in ApplyBpe(symbols))
            {
                if (!_tokenIds.TryGetValue(piece, out var tokenId))
                {
                    throw new InvalidDataException($"Tokenizer produced a symbol that is absent from its vocabulary: {piece}");
                }

                ids.Add(tokenId);
            }
        }
    }

    private IReadOnlyList<string> ApplyBpe(string[] symbols)
    {
        if (symbols.Length < 2) return symbols;
        var nodes = new SymbolNode[symbols.Length];
        for (var i = 0; i < symbols.Length; i++) nodes[i] = new SymbolNode(symbols[i], i - 1, i + 1 < symbols.Length ? i + 1 : -1);
        var queue = new PriorityQueue<MergeCandidate, (int Rank, int LeftIndex)>();
        for (var i = 0; i < nodes.Length - 1; i++) EnqueuePair(i);

        while (queue.TryDequeue(out var candidate, out _))
        {
            ref var left = ref nodes[candidate.Left];
            ref var right = ref nodes[candidate.Right];
            if (!left.Active || !right.Active || left.Next != candidate.Right || left.Version != candidate.LeftVersion || right.Version != candidate.RightVersion)
            {
                continue;
            }

            left.Value = string.Concat(left.Value, right.Value);
            left.Version++;
            left.Next = right.Next;
            right.Active = false;
            right.Version++;
            if (right.Next >= 0) nodes[right.Next].Previous = candidate.Left;
            if (left.Previous >= 0) EnqueuePair(left.Previous);
            EnqueuePair(candidate.Left);
        }

        var result = new List<string>(symbols.Length);
        var current = 0;
        while (current >= 0)
        {
            if (nodes[current].Active) result.Add(nodes[current].Value);
            current = nodes[current].Next;
        }

        return result;

        void EnqueuePair(int leftIndex)
        {
            var left = nodes[leftIndex];
            var rightIndex = left.Next;
            if (rightIndex < 0 || !left.Active || !nodes[rightIndex].Active) return;
            if (_mergeRanks.TryGetValue((left.Value, nodes[rightIndex].Value), out var rank))
            {
                queue.Enqueue(
                    new MergeCandidate(leftIndex, rightIndex, left.Version, nodes[rightIndex].Version),
                    (rank, leftIndex));
            }
        }
    }

    private bool TryReadSpecial(string text, int position, out string special, out int id)
    {
        special = string.Empty;
        id = -1;
        if (!_specialTokensByStart.TryGetValue(text[position], out var candidates)) return false;
        foreach (var token in candidates)
        {
            if (position + token.Length > text.Length || string.CompareOrdinal(text, position, token, 0, token.Length) != 0) continue;
            special = token;
            id = _tokenIds[token];
            return true;
        }

        return false;
    }

    private static bool IsSpecialType(int tokenType) => tokenType is 3 or 4;

    private static void AppendPendingBytes(StringBuilder destination, ArrayBufferWriter<byte> bytes)
    {
        if (bytes.WrittenCount == 0) return;
        destination.Append(Encoding.UTF8.GetString(bytes.WrittenSpan));
        bytes.Clear();
    }

    private static char[] BuildByteToUnicode()
    {
        var visibleBytes = new List<int>(256);
        for (var value = 33; value <= 126; value++) visibleBytes.Add(value);
        for (var value = 161; value <= 172; value++) visibleBytes.Add(value);
        for (var value = 174; value <= 255; value++) visibleBytes.Add(value);
        var result = new char[256];
        var extra = 0;
        for (var value = 0; value < 256; value++)
        {
            if (visibleBytes.Contains(value)) result[value] = (char)value;
            else result[value] = (char)(256 + extra++);
        }

        return result;
    }

    private static short[] BuildUnicodeToByte()
    {
        var result = new short[512];
        Array.Fill(result, (short)-1);
        for (var value = 0; value < ByteToUnicode.Length; value++) result[ByteToUnicode[value]] = (short)value;
        return result;
    }

    private struct SymbolNode(string value, int previous, int next)
    {
        public string Value = value;
        public int Previous = previous;
        public int Next = next;
        public int Version;
        public bool Active = true;
    }

    private readonly record struct MergeCandidate(int Left, int Right, int LeftVersion, int RightVersion);
}

public sealed class Gpt2StreamDecoder
{
    private readonly string[] _vocabulary;
    private readonly int[] _tokenTypes;
    private readonly Decoder _decoder = Encoding.UTF8.GetDecoder();

    internal Gpt2StreamDecoder(string[] vocabulary, int[] tokenTypes)
    {
        _vocabulary = vocabulary;
        _tokenTypes = tokenTypes;
    }

    public string Decode(int tokenId)
    {
        if ((uint)tokenId >= (uint)_vocabulary.Length) throw new ArgumentOutOfRangeException(nameof(tokenId));
        if (_tokenTypes[tokenId] is 3 or 4) return string.Empty;
        var token = _vocabulary[tokenId];
        Span<byte> bytes = token.Length <= 512 ? stackalloc byte[token.Length] : new byte[token.Length];
        for (var i = 0; i < token.Length; i++)
        {
            var symbol = token[i];
            var byteValue = Gpt2Tokenizer.GetByteValue(symbol);
            if (byteValue < 0)
            {
                throw new InvalidDataException($"Tokenizer piece contains unmappable byte symbol U+{(int)symbol:X4}.");
            }

            bytes[i] = (byte)byteValue;
        }

        Span<char> chars = token.Length <= 512 ? stackalloc char[token.Length] : new char[token.Length];
        var written = _decoder.GetChars(bytes, chars, flush: false);
        return written == 0 ? string.Empty : new string(chars[..written]);
    }
}
