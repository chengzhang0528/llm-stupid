using LlmStupid.Engine.Tokenization;

namespace LlmStupid.Engine.Inference;

public sealed record CompletionRequest(
    IReadOnlyList<ChatMessage> Messages,
    int MaxNewTokens = 10,
    float Temperature = 0.7f,
    int? Seed = null,
    string? SystemPrompt = null);

public sealed record TokenGenerationOptions(int MaxNewTokens = 10, float Temperature = 0, int? Seed = 1, bool StopAtEndOfSequence = true);

public sealed record GenerationMetrics(
    int PromptTokenCount,
    int GeneratedTokenCount,
    int ForwardPasses,
    TimeSpan PromptProcessingTime,
    TimeSpan TimeToFirstToken,
    TimeSpan GenerationTime,
    double PromptTokensPerSecond,
    double GeneratedTokensPerSecond,
    double ForwardPassesPerSecond,
    long WorkingSetBytes);

public sealed record InferenceResult(string Text, IReadOnlyList<int> TokenIds, string FinishReason, GenerationMetrics Metrics);

public interface IInferenceEngine : IDisposable
{
    string ModelId { get; }

    InferenceResult Generate(CompletionRequest request, CancellationToken cancellationToken = default, Action<string>? onTextDelta = null);
}
