using System.Text.Json.Serialization;

namespace LlmStupid.Service;

public sealed record ChatCompletionsRequest(
    string? Model,
    IReadOnlyList<ChatMessageRequest>? Messages,
    [property: JsonPropertyName("max_tokens")] int? MaxTokens = null,
    float? Temperature = null,
    bool Stream = false,
    int? Seed = null);

public sealed record ChatMessageRequest(string? Role, string? Content);

public sealed record ChatCompletionResponse(
    string Id,
    string Object,
    long Created,
    string Model,
    IReadOnlyList<ChatCompletionChoice> Choices,
    TokenUsage Usage,
    LocalMetrics Metrics);

public sealed record ChatCompletionChoice(int Index, AssistantMessage Message, [property: JsonPropertyName("finish_reason")] string FinishReason);

public sealed record AssistantMessage(string Role, string Content);

public sealed record TokenUsage(
    [property: JsonPropertyName("prompt_tokens")] int PromptTokens,
    [property: JsonPropertyName("completion_tokens")] int CompletionTokens,
    [property: JsonPropertyName("total_tokens")] int TotalTokens);

public sealed record LocalMetrics(
    [property: JsonPropertyName("prompt_processing_ms")] double PromptProcessingMilliseconds,
    [property: JsonPropertyName("time_to_first_token_ms")] double TimeToFirstTokenMilliseconds,
    [property: JsonPropertyName("generation_ms")] double GenerationMilliseconds,
    [property: JsonPropertyName("generated_tokens_per_second")] double GeneratedTokensPerSecond,
    [property: JsonPropertyName("working_set_mb")] double WorkingSetMegabytes);

public sealed record ChatCompletionChunk(
    string Id,
    string Object,
    long Created,
    string Model,
    IReadOnlyList<ChatCompletionChunkChoice> Choices,
    TokenUsage? Usage = null);

public sealed record ChatCompletionChunkChoice(
    int Index,
    DeltaMessage Delta,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);

public sealed record DeltaMessage(string? Role, string? Content);
