using System.Diagnostics;
using System.Runtime.InteropServices;
using LlmStupid.Engine.Inference;
using LlmStupid.Engine.Runtime;
using LlmStupid.Engine.Tokenization;

namespace LlmStupid.Engine.Model;

public sealed record ModelExecutionOptions(
    int ContextLength = 0,
    int ThreadCount = 0,
    KernelMode KernelMode = KernelMode.Auto,
    IQuantKernelBackend? KernelBackend = null);

public sealed class GraniteMoeModel : IDisposable
{
    private readonly GgufFile _file;
    private bool _disposed;

    private GraniteMoeModel(
        GgufFile file,
        Gpt2Tokenizer tokenizer,
        GraniteLayer[] layers,
        float[] outputNorm,
        ModelExecutionOptions options,
        IQuantKernelBackend kernelBackend,
        int threadCount)
    {
        _file = file;
        Tokenizer = tokenizer;
        Layers = layers;
        OutputNorm = outputNorm;
        Options = options;
        KernelBackend = kernelBackend;
        ThreadCount = threadCount;
        Embedding = new MappedMatrix(file, file.RequireTensor("token_embd.weight"), kernelBackend);
        OutputProjection = Embedding;
        ModelId = file.Metadata.GetString("general.name");
        ContextLength = checked((int)file.Metadata.GetInt64("granitemoe.context_length"));
        RmsNormEpsilon = file.Metadata.GetSingle("granitemoe.attention.layer_norm_rms_epsilon");
        EmbeddingScale = file.Metadata.GetSingle("granitemoe.embedding_scale");
        ResidualScale = file.Metadata.GetSingle("granitemoe.residual_scale");
        LogitScale = file.Metadata.GetSingle("granitemoe.logit_scale");
        AttentionScale = file.Metadata.GetSingle("granitemoe.attention.scale");
        RopeFrequencyBase = file.Metadata.GetSingle("granitemoe.rope.freq_base");
        HiddenSize = checked((int)file.Metadata.GetInt64("granitemoe.embedding_length"));
        KeyValueSize = checked((int)file.Metadata.GetInt64("granitemoe.attention.head_count_kv")) * HeadDimension;
        HeadCount = checked((int)file.Metadata.GetInt64("granitemoe.attention.head_count"));
        ExpertCount = checked((int)file.Metadata.GetInt64("granitemoe.expert_count"));
        ExpertsPerToken = checked((int)file.Metadata.GetInt64("granitemoe.expert_used_count"));
        FeedForwardSize = checked((int)file.Metadata.GetInt64("granitemoe.feed_forward_length"));
        RotaryDimension = checked((int)file.Metadata.GetInt64("granitemoe.rope.dimension_count"));
        RopeFrequencies = Enumerable.Range(0, RotaryDimension / 2)
            .Select(index => MathF.Pow(RopeFrequencyBase, -(float)(2 * index) / RotaryDimension))
            .ToArray();
        if (ContextLength > 131_072 || ContextLength <= 0) throw new NotSupportedException($"Unsupported Granite context size {ContextLength}.");
    }

    internal Gpt2Tokenizer Tokenizer { get; }

    internal GraniteLayer[] Layers { get; }

    internal float[] OutputNorm { get; }

    internal MappedMatrix Embedding { get; }

    internal MappedMatrix OutputProjection { get; }

    internal ModelExecutionOptions Options { get; }

    internal IQuantKernelBackend KernelBackend { get; }

    internal int ThreadCount { get; }

    internal float[] RopeFrequencies { get; }

    internal float RmsNormEpsilon { get; }

    internal float EmbeddingScale { get; }

    internal float ResidualScale { get; }

    internal float LogitScale { get; }

    internal float AttentionScale { get; }

    internal float RopeFrequencyBase { get; }

    internal int HeadDimension { get; } = 64;

    internal int HiddenSize { get; }

    internal int KeyValueSize { get; }

    internal int HeadCount { get; }

    internal int ExpertCount { get; }

    internal int ExpertsPerToken { get; }

    internal int FeedForwardSize { get; }

    internal int RotaryDimension { get; }

    public string ModelId { get; }

    public int ContextLength { get; }

    public int VocabularySize => Tokenizer.VocabularySize;

    public int ThreadCountUsed => ThreadCount;

    public string KernelBackendId => KernelBackend.Id;

    public KernelMode KernelMode => Options.KernelMode;

    public int[] Encode(string text) => Tokenizer.Encode(text);

    public string Decode(IEnumerable<int> tokenIds, bool skipSpecialTokens = false) => Tokenizer.Decode(tokenIds, skipSpecialTokens);

    public string RenderChat(IReadOnlyList<ChatMessage> messages, string? systemPrompt = null) => GraniteChatTemplate.Render(messages, systemPrompt);

    public IReadOnlyDictionary<string, GgufTensor> Tensors => _file.Tensors;

    public static GraniteMoeModel Load(string path, ModelExecutionOptions? options = null)
    {
        options ??= new ModelExecutionOptions();
        var file = GgufFile.Open(path);
        try
        {
            var metadata = file.Metadata;
            if (!metadata.GetString("general.architecture").Equals("granitemoe", StringComparison.Ordinal))
            {
                throw new NotSupportedException($"Model architecture '{metadata.GetString("general.architecture")}' is not Granite MoE.");
            }

            var layerCount = checked((int)metadata.GetInt64("granitemoe.block_count"));
            var hiddenSize = checked((int)metadata.GetInt64("granitemoe.embedding_length"));
            var expertCount = checked((int)metadata.GetInt64("granitemoe.expert_count"));
            var expertsPerToken = checked((int)metadata.GetInt64("granitemoe.expert_used_count"));
            var feedForwardSize = checked((int)metadata.GetInt64("granitemoe.feed_forward_length"));
            var headCount = checked((int)metadata.GetInt64("granitemoe.attention.head_count"));
            var headCountKv = checked((int)metadata.GetInt64("granitemoe.attention.head_count_kv"));
            var vocabularySize = checked((int)metadata.GetInt64("granitemoe.vocab_size"));
            if (layerCount != 24 || hiddenSize != 1024 || expertCount != 32 || expertsPerToken != 8 || feedForwardSize != 512 || headCount != 16 || headCountKv != 8 || vocabularySize != 49_155)
            {
                throw new NotSupportedException("This C# engine currently supports the Granite-3.1-1B-A400M model shape only.");
            }

            var tokenizer = new Gpt2Tokenizer(metadata);
            var kernelBackend = options.KernelBackend ?? new ManagedCpuKernelBackend(options.KernelMode);
            var embedding = file.RequireTensor("token_embd.weight");
            ValidateTensor(embedding, [1024, 49_155], GgmlType.Q6K);
            ValidateTensor(file.RequireTensor("output_norm.weight"), [1024], GgmlType.F32);
            var outputNorm = ReadFloatVector(file, "output_norm.weight", 1024);
            var layers = new GraniteLayer[layerCount];
            for (var index = 0; index < layerCount; index++)
            {
                var prefix = $"blk.{index}.";
                var attnNorm = ReadFloatVector(file, prefix + "attn_norm.weight", 1024);
                var ffnNorm = ReadFloatVector(file, prefix + "ffn_norm.weight", 1024);
                layers[index] = new GraniteLayer(
                    attnNorm,
                    ffnNorm,
                    Matrix(file, kernelBackend, prefix + "attn_q.weight", [1024, 1024], GgmlType.Q4K),
                    Matrix(file, kernelBackend, prefix + "attn_k.weight", [1024, 512], GgmlType.Q4K),
                    Matrix(file, kernelBackend, prefix + "attn_v.weight", [1024, 512], null),
                    Matrix(file, kernelBackend, prefix + "attn_output.weight", [1024, 1024], GgmlType.Q4K),
                    Matrix(file, kernelBackend, prefix + "ffn_gate_inp.weight", [1024, 32], GgmlType.F32),
                    Matrix(file, kernelBackend, prefix + "ffn_gate_exps.weight", [1024, 512, 32], GgmlType.Q4K),
                    Matrix(file, kernelBackend, prefix + "ffn_up_exps.weight", [1024, 512, 32], GgmlType.Q4K),
                    Matrix(file, kernelBackend, prefix + "ffn_down_exps.weight", [512, 1024, 32], null));
            }

            var threadCount = options.ThreadCount == 0 ? Math.Min(10, Environment.ProcessorCount) : options.ThreadCount;
            if (threadCount is < 1 or > 64) throw new ArgumentOutOfRangeException(nameof(options), "Thread count must be between 1 and 64.");
            if (options.ContextLength < 0 || options.ContextLength > metadata.GetInt64("granitemoe.context_length"))
            {
                throw new ArgumentOutOfRangeException(nameof(options), "Context length must be between 0 and the model's training context length.");
            }

            return new GraniteMoeModel(file, tokenizer, layers, outputNorm, options, kernelBackend, threadCount);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    public InferenceSession CreateSession() => new(this, Options.ContextLength == 0 ? ContextLength : Options.ContextLength);

    public InferenceResult GenerateTokens(
        int[] promptTokenIds,
        TokenGenerationOptions? options = null,
        CancellationToken cancellationToken = default,
        Action<int>? onToken = null)
    {
        using var session = CreateSession();
        return session.Generate(promptTokenIds, options ?? new TokenGenerationOptions(), cancellationToken, onToken);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _file.Dispose();
    }

    private static MappedMatrix Matrix(GgufFile file, IQuantKernelBackend backend, string name, long[] dimensions, GgmlType? type)
    {
        var tensor = file.RequireTensor(name);
        if (!tensor.Dimensions.SequenceEqual(dimensions))
        {
            throw new InvalidDataException($"Tensor '{name}' has shape [{string.Join(',', tensor.Dimensions)}], expected [{string.Join(',', dimensions)}].");
        }

        if (type is not null && tensor.Type != type)
        {
            throw new InvalidDataException($"Tensor '{name}' has type {tensor.Type}, expected {type}.");
        }

        if (tensor.Type is not (GgmlType.F32 or GgmlType.Q4K or GgmlType.Q6K))
        {
            throw new NotSupportedException($"Tensor '{name}' uses unsupported type {(uint)tensor.Type}.");
        }

        return new MappedMatrix(file, tensor, backend);
    }

    private static float[] ReadFloatVector(GgufFile file, string name, int length)
    {
        var tensor = file.RequireTensor(name);
        ValidateTensor(tensor, [length], GgmlType.F32);
        return MemoryMarshal.Cast<byte, float>(file.GetTensorData(tensor)).ToArray();
    }

    private static void ValidateTensor(GgufTensor tensor, long[] dimensions, GgmlType type)
    {
        if (tensor.Type != type || !tensor.Dimensions.SequenceEqual(dimensions))
        {
            throw new InvalidDataException($"Tensor '{tensor.Name}' has shape/type [{string.Join(',', tensor.Dimensions)}]/{tensor.Type}; expected [{string.Join(',', dimensions)}]/{type}.");
        }
    }
}

internal sealed record GraniteLayer(
    float[] AttentionNorm,
    float[] FeedForwardNorm,
    MappedMatrix Query,
    MappedMatrix Key,
    MappedMatrix Value,
    MappedMatrix AttentionOutput,
    MappedMatrix Router,
    MappedMatrix ExpertGate,
    MappedMatrix ExpertUp,
    MappedMatrix ExpertDown);
