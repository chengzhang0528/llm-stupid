using System.Diagnostics;
using LlmStupid.Engine.Model;
using LlmStupid.Engine.Tokenization;

namespace LlmStupid.Engine.Inference;

public sealed class GraniteMoeEngine(GraniteMoeModel model) : IInferenceEngine
{
    public string ModelId => model.ModelId;

    public InferenceResult Generate(CompletionRequest request, CancellationToken cancellationToken = default, Action<string>? onTextDelta = null)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.MaxNewTokens is < 1 or > 4096) throw new ArgumentOutOfRangeException(nameof(request), "MaxNewTokens must be between 1 and 4096.");
        if (request.Temperature is < 0 or > 5) throw new ArgumentOutOfRangeException(nameof(request), "Temperature must be between 0 and 5.");

        var tokenizationTimer = Stopwatch.StartNew();
        var prompt = GraniteChatTemplate.Render(request.Messages, request.SystemPrompt);
        var promptTokenIds = model.Tokenizer.Encode(prompt);
        tokenizationTimer.Stop();
        var decoder = onTextDelta is null ? null : model.Tokenizer.CreateStreamDecoder();
        var result = model.GenerateTokens(
            promptTokenIds,
            new TokenGenerationOptions(request.MaxNewTokens, request.Temperature, request.Seed),
            cancellationToken,
            decoder is null ? null : tokenId =>
            {
                var delta = decoder.Decode(tokenId);
                if (delta.Length > 0) onTextDelta!(delta);
            });

        var tokenizationTime = tokenizationTimer.Elapsed;
        var promptProcessingTime = result.Metrics.PromptProcessingTime + tokenizationTime;
        var totalFirstTokenTime = result.Metrics.TimeToFirstToken + tokenizationTime;
        var promptRate = promptProcessingTime.TotalSeconds > 0
            ? promptTokenIds.Length / promptProcessingTime.TotalSeconds
            : 0;
        return result with
        {
            Metrics = result.Metrics with
            {
                PromptProcessingTime = promptProcessingTime,
                TimeToFirstToken = totalFirstTokenTime,
                PromptTokensPerSecond = promptRate,
            }
        };
    }

    public void Dispose() => model.Dispose();
}
