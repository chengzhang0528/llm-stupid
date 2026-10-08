using System.Diagnostics;
using System.Text.Json;
using System.Threading.Channels;
using LlmStupid.Engine.Inference;
using LlmStupid.Engine.Model;
using LlmStupid.Engine.Runtime;
using LlmStupid.Engine.Tokenization;
using LlmStupid.Service;

var builder = WebApplication.CreateBuilder(args);
var modelPath = builder.Configuration["model-path"]
    ?? builder.Configuration["Model:Path"]
    ?? Environment.GetEnvironmentVariable("LLM_MODEL_PATH");
if (string.IsNullOrWhiteSpace(modelPath))
{
    throw new InvalidOperationException("Set LLM_MODEL_PATH or pass --model-path <GGUF path> before starting the local inference service.");
}

var threadCount = int.TryParse(Environment.GetEnvironmentVariable("LLM_THREADS"), out var parsedThreads) ? parsedThreads : 0;
var contextLength = int.TryParse(Environment.GetEnvironmentVariable("LLM_CONTEXT"), out var parsedContext) ? parsedContext : 0;
var modelLoadTimer = Stopwatch.StartNew();
var model = GraniteMoeModel.Load(modelPath, new ModelExecutionOptions(contextLength, threadCount));
var engine = new GraniteMoeEngine(model);
modelLoadTimer.Stop();
Console.WriteLine($"Loaded {model.ModelId} in {modelLoadTimer.Elapsed.TotalSeconds:F2}s; context limit {model.ContextLength}; CPU threads {model.ThreadCountUsed}; compute backend {model.KernelBackendId}.");

builder.Services.AddSingleton<IInferenceEngine>(engine);
builder.Services.AddSingleton(new SemaphoreSlim(1, 1));
var app = builder.Build();

app.MapGet("/health", (IInferenceEngine inference) => Results.Ok(new { status = "ok", model = inference.ModelId }));
app.MapGet("/v1/models", (IInferenceEngine inference) => Results.Ok(new
{
    @object = "list",
    data = new[] { new { id = inference.ModelId, @object = "model", created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), owned_by = "local" } }
}));

app.MapPost("/v1/chat/completions", async (
    ChatCompletionsRequest request,
    HttpContext http,
    IInferenceEngine inference,
    SemaphoreSlim generationGate) =>
{
    if (request.Messages is null || request.Messages.Count == 0)
    {
        return Results.BadRequest(new { error = "messages must contain at least one chat message" });
    }

    if (request.Messages.Any(message => string.IsNullOrWhiteSpace(message.Role) || message.Content is null))
    {
        return Results.BadRequest(new { error = "each message needs a role and content" });
    }

    if (!string.IsNullOrWhiteSpace(request.Model) && !request.Model.Equals(inference.ModelId, StringComparison.OrdinalIgnoreCase))
    {
        return Results.NotFound(new { error = $"model '{request.Model}' is not loaded" });
    }

    var maxTokens = request.MaxTokens ?? 10;
    var temperature = request.Temperature ?? 0.7f;
    if (maxTokens is < 1 or > 4096 || temperature is < 0 or > 5)
    {
        return Results.BadRequest(new { error = "max_tokens must be 1..4096 and temperature must be 0..5" });
    }

    if (request.Stream)
    {
        return await StreamCompletionAsync(request, http, inference, generationGate, maxTokens, temperature);
    }

    await generationGate.WaitAsync(http.RequestAborted);
    try
    {
        var completionRequest = ToEngineRequest(request, maxTokens, temperature);
        var result = await Task.Run(() => inference.Generate(completionRequest, http.RequestAborted), http.RequestAborted);
        return Results.Ok(ToResponse(inference.ModelId, result));
    }
    finally
    {
        generationGate.Release();
    }
});

app.Run();

static CompletionRequest ToEngineRequest(ChatCompletionsRequest request, int maxTokens, float temperature) => new(
    request.Messages!.Select(message => new ChatMessage(message.Role!, message.Content!)).ToArray(),
    maxTokens,
    temperature,
    request.Seed);

static ChatCompletionResponse ToResponse(string modelId, InferenceResult result) => new(
    $"chatcmpl-{Guid.NewGuid():N}",
    "chat.completion",
    DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
    modelId,
    [new ChatCompletionChoice(0, new AssistantMessage("assistant", result.Text), result.FinishReason)],
    new TokenUsage(result.Metrics.PromptTokenCount, result.Metrics.GeneratedTokenCount, result.Metrics.PromptTokenCount + result.Metrics.GeneratedTokenCount),
    new LocalMetrics(
        result.Metrics.PromptProcessingTime.TotalMilliseconds,
        result.Metrics.TimeToFirstToken.TotalMilliseconds,
        result.Metrics.GenerationTime.TotalMilliseconds,
        result.Metrics.GeneratedTokensPerSecond,
        result.Metrics.WorkingSetBytes / (1024d * 1024d)));

static async Task<IResult> StreamCompletionAsync(
    ChatCompletionsRequest request,
    HttpContext http,
    IInferenceEngine inference,
    SemaphoreSlim generationGate,
    int maxTokens,
    float temperature)
{
    await generationGate.WaitAsync(http.RequestAborted);
    try
    {
        http.Response.StatusCode = StatusCodes.Status200OK;
        http.Response.ContentType = "text/event-stream; charset=utf-8";
        http.Response.Headers.CacheControl = "no-cache";
        http.Response.Headers["Connection"] = "keep-alive";
        await http.Response.StartAsync(http.RequestAborted);

        var id = $"chatcmpl-{Guid.NewGuid():N}";
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        await WriteChunkAsync(http.Response, new ChatCompletionChunk(id, "chat.completion.chunk", created, inference.ModelId,
            [new ChatCompletionChunkChoice(0, new DeltaMessage("assistant", null), null)]), http.RequestAborted);

        var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        var completionRequest = ToEngineRequest(request, maxTokens, temperature);
        var generation = Task.Run(
            () => inference.Generate(completionRequest, http.RequestAborted, delta => channel.Writer.TryWrite(delta)),
            http.RequestAborted);
        _ = generation.ContinueWith(
            task => channel.Writer.TryComplete(task.IsFaulted ? task.Exception?.GetBaseException() : null),
            CancellationToken.None,
            TaskContinuationOptions.ExecuteSynchronously,
            TaskScheduler.Default);

        await foreach (var delta in channel.Reader.ReadAllAsync(http.RequestAborted))
        {
            await WriteChunkAsync(http.Response, new ChatCompletionChunk(id, "chat.completion.chunk", created, inference.ModelId,
                [new ChatCompletionChunkChoice(0, new DeltaMessage(null, delta), null)]), http.RequestAborted);
        }

        var result = await generation;
        var usage = new TokenUsage(result.Metrics.PromptTokenCount, result.Metrics.GeneratedTokenCount,
            result.Metrics.PromptTokenCount + result.Metrics.GeneratedTokenCount);
        await WriteChunkAsync(http.Response, new ChatCompletionChunk(id, "chat.completion.chunk", created, inference.ModelId,
            [new ChatCompletionChunkChoice(0, new DeltaMessage(null, null), result.FinishReason)], usage), http.RequestAborted);
        await http.Response.WriteAsync("data: [DONE]\n\n", http.RequestAborted);
        await http.Response.Body.FlushAsync(http.RequestAborted);
        return Results.Empty;
    }
    finally
    {
        generationGate.Release();
    }
}

static async Task WriteChunkAsync(HttpResponse response, ChatCompletionChunk chunk, CancellationToken cancellationToken)
{
    await response.WriteAsync("data: ", cancellationToken);
    await response.WriteAsync(JsonSerializer.Serialize(chunk, JsonSerializerOptions.Web), cancellationToken);
    await response.WriteAsync("\n\n", cancellationToken);
    await response.Body.FlushAsync(cancellationToken);
}
