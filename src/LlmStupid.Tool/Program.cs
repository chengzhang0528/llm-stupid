using System.Diagnostics;
using System.Security.Cryptography;
using LlmStupid.Engine.Inference;
using LlmStupid.Engine.Model;
using LlmStupid.Engine.Runtime;
using LlmStupid.Engine.Tokenization;

const string ExpectedModelSha256 = "3A2EC1C2A78CB29D901E29BBF5162DCD03381E13803D2CBDCFF838D4D08142EB";
const double LlamaCppReferenceTokensPerSecond = 16.63;

if (args.Length < 2 || args[0] is "--help" or "-h")
{
    PrintUsage();
    return 2;
}

var command = args[0].ToLowerInvariant();
var modelPath = args[1];
try
{
    switch (command)
    {
        case "inspect":
            Inspect(modelPath);
            break;
        case "tokenize":
            Tokenize(modelPath, GetOption(args, "--text") ?? throw new ArgumentException("tokenize requires --text."));
            break;
        case "generate":
            Generate(modelPath, args);
            break;
        case "bench":
            Bench(modelPath, args);
            break;
        default:
            PrintUsage();
            return 2;
    }

    return 0;
}
catch (Exception error)
{
    Console.Error.WriteLine($"{error.GetType().Name}: {error.Message}");
    return 1;
}

static void Inspect(string modelPath)
{
    using var file = GgufFile.Open(modelPath);
    Console.WriteLine($"GGUF v{file.Version}: {file.Metadata.GetString("general.name")}");
    Console.WriteLine($"Architecture: {file.Metadata.GetString("general.architecture")}; tensors: {file.Tensors.Count}; data section: {file.FileLength - file.DataStart:N0} bytes");
    Console.WriteLine($"Layers: {file.Metadata.GetInt64("granitemoe.block_count")}; hidden: {file.Metadata.GetInt64("granitemoe.embedding_length")}; experts: {file.Metadata.GetInt64("granitemoe.expert_count")} / top-{file.Metadata.GetInt64("granitemoe.expert_used_count")}; context: {file.Metadata.GetInt64("granitemoe.context_length")}");
    foreach (var group in file.Tensors.Values.GroupBy(tensor => (uint)tensor.Type).OrderBy(group => group.Key))
    {
        Console.WriteLine($"GGML type {group.Key}: {group.Count()} tensors");
    }
}

static void Tokenize(string modelPath, string text)
{
    using var model = GraniteMoeModel.Load(modelPath, new ModelExecutionOptions(ContextLength: 128, ThreadCount: 1));
    var tokenIds = model.Encode(text);
    Console.WriteLine($"tokens={tokenIds.Length}");
    Console.WriteLine(string.Join(' ', tokenIds));
    Console.WriteLine(model.Decode(tokenIds));
}

static void Generate(string modelPath, string[] args)
{
    var options = LoadOptions(args);
    var prompt = GetOption(args, "--prompt") ?? "Hello";
    var maxTokens = ParseInt(args, "--tokens", 10);
    var temperature = ParseFloat(args, "--temperature", 0);
    var loadTimer = Stopwatch.StartNew();
    using var model = GraniteMoeModel.Load(modelPath, options);
    loadTimer.Stop();
    using var engine = new GraniteMoeEngine(model);
    var result = engine.Generate(new CompletionRequest([new ChatMessage("user", prompt)], maxTokens, temperature, 1));
    Console.WriteLine(result.Text);
    PrintMetrics(result, loadTimer.Elapsed);
    GC.KeepAlive(engine);
}

static void Bench(string modelPath, string[] args)
{
    var repetitions = ParseInt(args, "--repetitions", 5);
    var warmupRepetitions = ParseInt(args, "--warmup-repetitions", 5);
    var tokenCount = ParseInt(args, "--tokens", 10);
    var threads = ParseInt(args, "--threads", 6);
    var context = ParseInt(args, "--context", 128);
    if (repetitions < 5) throw new ArgumentOutOfRangeException(nameof(args), "Use at least five repetitions for the stored reference comparison.");
    if (warmupRepetitions is < 1 or > 20) throw new ArgumentOutOfRangeException(nameof(args), "Use between one and twenty warm-up repetitions.");
    if (tokenCount != 10) throw new ArgumentOutOfRangeException(nameof(args), "The pinned llama.cpp workload generates exactly 10 decode tokens.");
    if (threads is < 1 or > 64) throw new ArgumentOutOfRangeException(nameof(args), "Thread count must be between 1 and 64.");

    using var stream = File.OpenRead(modelPath);
    var hash = Convert.ToHexString(SHA256.HashData(stream));
    if (!hash.Equals(ExpectedModelSha256, StringComparison.OrdinalIgnoreCase))
    {
        throw new InvalidDataException($"Model SHA-256 {hash} does not match the pinned reference file {ExpectedModelSha256}.");
    }

    var loadTimer = Stopwatch.StartNew();
    using var model = GraniteMoeModel.Load(modelPath, new ModelExecutionOptions(context, threads));
    loadTimer.Stop();
    Console.WriteLine($"Model SHA-256: {hash}");
    Console.WriteLine($"C# Release model load: {loadTimer.Elapsed.TotalSeconds:F2}s (not included in decode throughput).");
    if (threads == 6)
    {
        Console.WriteLine($"Reference: llama.cpp 42916d83f4a225e56709f873aa8050ac11f5b6a4, -p 0 -n 10 -r 5 -t 6 -ngl 0, {LlamaCppReferenceTokensPerSecond:F2} ± 0.32 tokens/s.");
    }
    else
    {
        Console.WriteLine("Reference ratio omitted: this thread count differs from the pinned six-thread llama.cpp workload.");
    }
    Console.WriteLine($"Candidate: .NET 10 Release, CPU only, {model.KernelBackendId}, {model.ThreadCountUsed} threads, implicit EOS seed, no prompt tokens, 10 generated tokens.");

    var warmupTimer = Stopwatch.StartNew();
    var warmupTokenCount = 0;
    for (var iteration = 0; iteration < warmupRepetitions; iteration++)
    {
        var warmup = model.GenerateTokens([], new TokenGenerationOptions(tokenCount, 0, 1, StopAtEndOfSequence: false));
        warmupTokenCount += warmup.Metrics.GeneratedTokenCount;
    }

    warmupTimer.Stop();
    Console.WriteLine($"Warm-up: {warmupRepetitions} runs / {warmupTokenCount} decode tokens in {warmupTimer.Elapsed.TotalMilliseconds:F1} ms (excluded from {repetitions} timed runs).");
    var rates = new double[repetitions];
    var elapsed = new double[repetitions];
    var maxWorkingSet = 0L;
    for (var iteration = 0; iteration < repetitions; iteration++)
    {
        var result = model.GenerateTokens([], new TokenGenerationOptions(tokenCount, 0, 1, StopAtEndOfSequence: false));
        rates[iteration] = result.Metrics.GeneratedTokensPerSecond;
        elapsed[iteration] = result.Metrics.GenerationTime.TotalMilliseconds;
        maxWorkingSet = Math.Max(maxWorkingSet, result.Metrics.WorkingSetBytes);
        Console.WriteLine($"run {iteration + 1}: {result.Metrics.GeneratedTokensPerSecond:F2} tokens/s, {result.Metrics.GenerationTime.TotalMilliseconds:F1} ms, {result.Metrics.ForwardPasses} forward passes, {result.Metrics.GeneratedTokenCount} tokens");
    }

    var meanRate = rates.Average();
    var deviation = StandardDeviation(rates, meanRate);
    var meanMs = elapsed.Average();
    Console.WriteLine($"C# result: {meanRate:F2} ± {deviation:F2} tokens/s; mean decode {meanMs:F1} ms; working set {maxWorkingSet / (1024d * 1024d):F1} MiB.");
    if (threads == 6)
    {
        Console.WriteLine($"Relative to the pinned reference: {meanRate / LlamaCppReferenceTokensPerSecond:P1} ({meanRate / LlamaCppReferenceTokensPerSecond:F2}x). Prompt processing, real-chat TTFT, model load and context capacity are not included in the decode comparison.");
    }
    else
    {
        Console.WriteLine("Thread-scaling result only; it is not a matched llama.cpp comparison. Prompt processing, real-chat TTFT, model load and context capacity are not included.");
    }
}

static ModelExecutionOptions LoadOptions(string[] args) => new(
    ParseInt(args, "--context", 2048),
    ParseInt(args, "--threads", 10),
    KernelMode.Auto);

static int ParseInt(string[] args, string name, int defaultValue) =>
    int.TryParse(GetOption(args, name), out var value) ? value : defaultValue;

static float ParseFloat(string[] args, string name, float defaultValue) =>
    float.TryParse(GetOption(args, name), System.Globalization.CultureInfo.InvariantCulture, out var value) ? value : defaultValue;

static string? GetOption(string[] args, string name)
{
    for (var i = 2; i < args.Length - 1; i++)
    {
        if (args[i].Equals(name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
    }

    return null;
}

static double StandardDeviation(double[] values, double mean)
{
    if (values.Length < 2) return 0;
    var sum = values.Sum(value => (value - mean) * (value - mean));
    return Math.Sqrt(sum / (values.Length - 1));
}

static void PrintMetrics(InferenceResult result, TimeSpan loadTime)
{
    Console.WriteLine($"Model load: {loadTime.TotalMilliseconds:F0} ms");
    Console.WriteLine($"Prompt: {result.Metrics.PromptTokenCount} tokens, {result.Metrics.PromptProcessingTime.TotalMilliseconds:F1} ms, {result.Metrics.PromptTokensPerSecond:F2} tokens/s");
    Console.WriteLine($"Time to first token: {result.Metrics.TimeToFirstToken.TotalMilliseconds:F1} ms");
    Console.WriteLine($"Generation: {result.Metrics.GeneratedTokenCount} tokens, {result.Metrics.GenerationTime.TotalMilliseconds:F1} ms, {result.Metrics.GeneratedTokensPerSecond:F2} tokens/s ({result.Metrics.ForwardPassesPerSecond:F2} forward passes/s)");
    Console.WriteLine($"Working set: {result.Metrics.WorkingSetBytes / (1024d * 1024d):F1} MiB");
}

static void PrintUsage()
{
    Console.WriteLine("LlmStupid.Tool <inspect|tokenize|generate|bench> <model.gguf> [options]");
    Console.WriteLine("  inspect <model>                                    print GGUF metadata and tensor types");
    Console.WriteLine("  tokenize <model> --text <text>                     encode text with the model's GPT-2 BPE tokenizer");
    Console.WriteLine("  generate <model> --prompt <text> --tokens 10 [--threads 10] run a local Granite chat completion");
    Console.WriteLine("  bench <model> --repetitions 5 --warmup-repetitions 5 --tokens 10 [--threads 6] [--context 128] (6 matches the pinned reference)");
}
