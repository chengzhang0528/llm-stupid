using System.Buffers.Binary;
using LlmStupid.Engine.Inference;
using LlmStupid.Engine.Model;
using LlmStupid.Engine.Runtime;
using LlmStupid.Engine.Tokenization;

var checks = 0;
try
{
    VerifyQuantKernels();
    Console.WriteLine("Quantized kernels: Q4_K and Q6_K scalar/AVX2 checks passed.");
    if (args.Length > 0)
    {
        VerifyModel(args[0]);
        Console.WriteLine("Granite GGUF, tokenizer and one-token C# inference checks passed.");
    }

    Console.WriteLine($"Verification passed ({checks} assertions). AVX2 available: {QuantKernels.IsAvx2Available}.");
    return 0;
}
catch (Exception error)
{
    Console.Error.WriteLine($"Verification failed after {checks} assertions: {error}");
    return 1;
}

void VerifyQuantKernels()
{
    var q4 = new byte[144];
    WriteHalf(q4, 0, 1);
    var scales = q4.AsSpan(4, 12);
    scales[0] = 1;
    scales[1] = 2;
    scales[2] = 3;
    scales[3] = 4;
    scales[8] = 5;
    scales[9] = 6;
    scales[10] = 7;
    scales[11] = 8;
    q4.AsSpan(16, 128).Fill(0x21);
    var ones = Enumerable.Repeat(1f, 256).ToArray();
    EqualFloat(1792f, QuantKernels.Dot(GgmlType.Q4K, q4, ones, KernelMode.Scalar), 0.001f, "Q4_K scalar dot");
    if (QuantKernels.IsAvx2Available)
    {
        EqualFloat(1792f, QuantKernels.Dot(GgmlType.Q4K, q4, ones, KernelMode.Avx2), 0.01f, "Q4_K AVX2 dot");
    }

    var q6 = new byte[210];
    q6.AsSpan(0, 128).Fill(0x88);
    q6.AsSpan(192, 16).Fill(1);
    WriteHalf(q6, 208, 1);
    EqualFloat(-6144f, QuantKernels.Dot(GgmlType.Q6K, q6, ones, KernelMode.Scalar), 0.001f, "Q6_K scalar dot");
    if (QuantKernels.IsAvx2Available)
    {
        EqualFloat(-6144f, QuantKernels.Dot(GgmlType.Q6K, q6, ones, KernelMode.Avx2), 0.01f, "Q6_K AVX2 dot");
    }

    var decoded = new float[256];
    QuantKernels.DecodeRow(GgmlType.Q6K, q6, decoded);
    EqualFloat(-24f, decoded[0], 0.001f, "Q6_K decoded value");
    if (QuantKernels.IsAvx2Available)
    {
        VerifyQ8Approximation(GgmlType.Q4K, 144);
        VerifyQ8Approximation(GgmlType.Q6K, 210);
        VerifyQ4KRowTile(4);
        VerifyQ4KRowTile(8);
        VerifyQ4KRowTile(16);
    }

    VerifyQ8GroupSums();
}

void VerifyQ8GroupSums()
{
    const int inputLength = 64;
    var input = Enumerable.Range(0, inputLength)
        .Select(i => 1.3f * MathF.Sin(i * 0.23f) - 0.7f * MathF.Cos(i * 0.071f))
        .ToArray();
    var quantized = new QuantizedInputVector(inputLength);
    quantized.Quantize(input);
    for (var group = 0; group < quantized.Sums16.Length; group++)
    {
        var expected = 0;
        for (var i = group * 16; i < group * 16 + 16; i++) expected += quantized.Values[i];
        Equal(expected, quantized.Sums16[group], $"Q8 precomputed 16-value sum {group}");
    }

    quantized.Quantize(new float[inputLength]);
    Check(quantized.Sums16.All(sum => sum == 0), "Q8 zero quantization clears precomputed sums");
}

void VerifyQ4KRowTile(int rowCount)
{
    const int inputLength = 1024;
    const int blocksPerRow = inputLength / 256;
    const int rowBytes = blocksPerRow * 144;
    var random = new Random(73);
    var weights = new byte[rowCount * rowBytes];
    for (var row = 0; row < rowCount; row++)
    {
        for (var block = 0; block < blocksPerRow; block++)
        {
            var offset = row * rowBytes + block * 144;
            random.NextBytes(weights.AsSpan(offset, 144));
            WriteHalf(weights, offset, 0.08f + row * 0.01f + block * 0.005f);
            WriteHalf(weights, offset + 2, 0.015f + block * 0.002f);
        }
    }

    var values = Enumerable.Range(0, inputLength)
        .Select(i => 0.8f * MathF.Sin(i * 0.13f) + 0.4f * MathF.Cos(i * 0.031f))
        .ToArray();
    var quantized = new QuantizedInputVector(inputLength);
    quantized.Quantize(values);
    var backend = new ManagedCpuKernelBackend(KernelMode.Avx2);
    var actual = new float[rowCount];
    Check(backend.TryDotQuantizedRows(GgmlType.Q4K, weights, rowCount, quantized, actual),
        $"Q4_K AVX2 {rowCount}-row tile runs");
    for (var row = 0; row < rowCount; row++)
    {
        var reference = backend.DotQuantized(GgmlType.Q4K, weights.AsSpan(row * rowBytes, rowBytes), quantized);
        EqualFloat(reference, actual[row], 0f, $"Q4_K {rowCount}-row tile matches row {row} bit-for-bit");
    }
}

void VerifyQ8Approximation(GgmlType type, int blockBytes)
{
    const int blockCount = 4;
    const int blockLength = 256;
    var random = new Random(type == GgmlType.Q4K ? 41 : 61);
    var weights = new byte[blockCount * blockBytes];
    for (var block = 0; block < blockCount; block++)
    {
        var blockWeights = weights.AsSpan(block * blockBytes, blockBytes);
        random.NextBytes(blockWeights);
        if (type == GgmlType.Q4K)
        {
            WriteHalf(weights, block * blockBytes, 0.2f);
            WriteHalf(weights, block * blockBytes + 2, 0.05f);
        }
        else
        {
            WriteHalf(weights, block * blockBytes + 208, 0.02f);
            for (var i = 192; i < 208; i++) weights[block * blockBytes + i] = unchecked((byte)(sbyte)random.Next(-16, 17));
        }
    }

    var input = Enumerable.Range(0, blockCount * blockLength)
        .Select(i => 1.4f * MathF.Sin(i * 0.17f) + 0.6f * MathF.Cos(i * 0.037f))
        .ToArray();
    var decoded = new float[input.Length];
    QuantKernels.DecodeRow(type, weights, decoded);
    var reference = QuantKernels.Dot(type, weights, input, KernelMode.Scalar);
    var absoluteProductSum = 0f;
    for (var i = 0; i < input.Length; i++) absoluteProductSum += MathF.Abs(decoded[i] * input[i]);
    var q8 = new QuantizedInputVector(input.Length);
    q8.Quantize(input);
    var candidate = QuantKernels.DotQuantized(type, weights, q8);
    var error = MathF.Abs(reference - candidate);
    Check(error <= absoluteProductSum * 0.008f + 0.02f, $"{type} Q8 activation relative dot error {error:F4} exceeded tolerance");
}

void VerifyModel(string modelPath)
{
    using var file = GgufFile.Open(modelPath);
    Equal("granitemoe", file.Metadata.GetString("general.architecture"), "GGUF architecture");
    Equal(242, file.Tensors.Count, "GGUF tensor count");
    Equal(73, file.Tensors.Values.Count(tensor => tensor.Type == GgmlType.F32), "F32 tensor count");
    Equal(144, file.Tensors.Values.Count(tensor => tensor.Type == GgmlType.Q4K), "Q4_K tensor count");
    Equal(25, file.Tensors.Values.Count(tensor => tensor.Type == GgmlType.Q6K), "Q6_K tensor count");

    using var model = GraniteMoeModel.Load(modelPath, new ModelExecutionOptions(ContextLength: 128, ThreadCount: 6));
    Equal(QuantKernels.IsAvx2Available ? "managed-cpu-avx2" : "managed-cpu-scalar", model.KernelBackendId, "default injectable kernel backend");
    var original = "Hello, Granite!\n你好，世界。🙂";
    var ids = model.Encode(original);
    Equal(original, model.Decode(ids), "GPT-2 BPE round trip");
    var specialIds = model.Encode("<|start_of_role|>assistant<|end_of_role|>");
    Equal(49_152, specialIds[0], "Granite start-role token ID");
    Equal(49_153, specialIds[^1], "Granite end-role token ID");
    var rendered = model.RenderChat([new ChatMessage("user", "Hi")]);
    Check(rendered.Contains("<|start_of_role|>system<|end_of_role|>", StringComparison.Ordinal), "chat template includes default system message");
    Check(rendered.EndsWith("<|start_of_role|>assistant<|end_of_role|>", StringComparison.Ordinal), "chat template opens the assistant turn");

    var result = model.GenerateTokens([], new TokenGenerationOptions(1, 0, 1, StopAtEndOfSequence: false));
    Equal(1, result.Metrics.GeneratedTokenCount, "one-token inference result");
    Equal(1, result.Metrics.ForwardPasses, "one-token synthetic decode forward pass count");
    Check(result.TokenIds[0] >= 0 && result.TokenIds[0] < model.VocabularySize, "one-token result is a vocabulary token");
    Check(double.IsFinite(result.Metrics.GeneratedTokensPerSecond) && result.Metrics.GeneratedTokensPerSecond > 0, "one-token generation throughput is finite");

    using var injectedBackendModel = GraniteMoeModel.Load(modelPath, new ModelExecutionOptions(
        ContextLength: 128,
        ThreadCount: 1,
        KernelBackend: new ManagedCpuKernelBackend(KernelMode.Scalar)));
    Equal("managed-cpu-scalar", injectedBackendModel.KernelBackendId, "injected scalar kernel backend");
    var scalarResult = injectedBackendModel.GenerateTokens([], new TokenGenerationOptions(1, 0, 1, StopAtEndOfSequence: false));
    Equal(1, scalarResult.Metrics.GeneratedTokenCount, "scalar backend one-token inference");
}

void WriteHalf(byte[] target, int offset, float value) => BinaryPrimitives.WriteUInt16LittleEndian(target.AsSpan(offset, 2), BitConverter.HalfToUInt16Bits((Half)value));

void Equal<T>(T expected, T actual, string label) where T : IEquatable<T>
{
    checks++;
    if (!expected.Equals(actual)) throw new InvalidOperationException($"{label}: expected {expected}, got {actual}.");
}

void EqualFloat(float expected, float actual, float tolerance, string label)
{
    checks++;
    if (!float.IsFinite(actual) || MathF.Abs(expected - actual) > tolerance)
    {
        throw new InvalidOperationException($"{label}: expected {expected} ± {tolerance}, got {actual}.");
    }
}

void Check(bool condition, string label)
{
    checks++;
    if (!condition) throw new InvalidOperationException(label);
}
