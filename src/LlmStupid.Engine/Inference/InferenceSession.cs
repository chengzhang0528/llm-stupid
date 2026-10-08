using System.Diagnostics;
using LlmStupid.Engine.Model;
using LlmStupid.Engine.Runtime;

namespace LlmStupid.Engine.Inference;

public sealed class InferenceSession : IDisposable
{
    private readonly GraniteMoeModel _model;
    private readonly int _contextLength;
    private readonly ParallelOptions _parallelOptions;
    private readonly LayerCache[] _cache;
    private readonly ExpertScratch[] _expertScratch;
    private readonly float[][] _expertFeedForwardInputs;
    private readonly float[][] _expertActivationInputs;
    private readonly float[][] _expertGateOutputs;
    private readonly float[][] _expertUpOutputs;
    private readonly float[][] _expertDownOutputs;
    private readonly QuantizedInputVector[] _expertFeedForwardQuantizedInputs;
    private readonly QuantizedInputVector[] _expertActivationQuantizedInputs;
    private readonly float[] _hidden;
    private readonly float[] _attentionNormalized;
    private readonly float[] _query;
    private readonly float[] _key;
    private readonly float[] _value;
    private readonly float[] _attentionContext;
    private readonly float[] _attentionProjected;
    private readonly float[] _feedForwardInput;
    private readonly float[] _feedForwardNormalized;
    private readonly float[] _routerLogits;
    private readonly float[] _routerProbabilities;
    private readonly float[] _logits;
    private readonly float[] _outputNormalized;
    private readonly int[] _selectedExperts;
    private readonly float[] _selectedWeights;
    private readonly float[] _ropeCos;
    private readonly float[] _ropeSin;
    private readonly QuantizedInputVector _attentionQ8;
    private readonly QuantizedInputVector _feedForwardQ8;
    private float[] _attentionScores;
    private int _position;
    private bool _disposed;

    internal InferenceSession(GraniteMoeModel model, int contextLength)
    {
        _model = model;
        _contextLength = contextLength;
        _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = model.ThreadCount };
        _cache = Enumerable.Range(0, model.Layers.Length)
            .Select(_ => new LayerCache(model.KeyValueSize, Math.Min(contextLength, 16)))
            .ToArray();
        _expertScratch = Enumerable.Range(0, model.ExpertsPerToken).Select(_ => new ExpertScratch(model.HiddenSize, model.FeedForwardSize)).ToArray();
        _attentionQ8 = new QuantizedInputVector(model.HiddenSize);
        _feedForwardQ8 = new QuantizedInputVector(model.HiddenSize);
        _hidden = new float[model.HiddenSize];
        _attentionNormalized = new float[model.HiddenSize];
        _query = new float[model.HiddenSize];
        _key = new float[model.KeyValueSize];
        _value = new float[model.KeyValueSize];
        _attentionContext = new float[model.HiddenSize];
        _attentionProjected = new float[model.HiddenSize];
        _feedForwardInput = new float[model.HiddenSize];
        _feedForwardNormalized = new float[model.HiddenSize];
        _expertFeedForwardInputs = Enumerable.Repeat(_feedForwardNormalized, model.ExpertsPerToken).ToArray();
        _expertActivationInputs = _expertScratch.Select(scratch => scratch.Activation).ToArray();
        _expertGateOutputs = _expertScratch.Select(scratch => scratch.Gate).ToArray();
        _expertUpOutputs = _expertScratch.Select(scratch => scratch.Up).ToArray();
        _expertDownOutputs = _expertScratch.Select(scratch => scratch.Output).ToArray();
        _expertFeedForwardQuantizedInputs = Enumerable.Repeat(_feedForwardQ8, model.ExpertsPerToken).ToArray();
        _expertActivationQuantizedInputs = _expertScratch.Select(scratch => scratch.ActivationQ8).ToArray();
        _routerLogits = new float[model.ExpertCount];
        _routerProbabilities = new float[model.ExpertCount];
        _logits = new float[model.Tokenizer.VocabularySize];
        _outputNormalized = new float[model.HiddenSize];
        _selectedExperts = new int[model.ExpertsPerToken];
        _selectedWeights = new float[model.ExpertsPerToken];
        _ropeCos = new float[model.RotaryDimension / 2];
        _ropeSin = new float[model.RotaryDimension / 2];
        _attentionScores = new float[Math.Min(contextLength, 16)];
    }

    public InferenceResult Generate(
        int[] promptTokenIds,
        TokenGenerationOptions options,
        CancellationToken cancellationToken = default,
        Action<int>? onToken = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(promptTokenIds);
        ArgumentNullException.ThrowIfNull(options);
        if (options.MaxNewTokens is < 1 or > 4096) throw new ArgumentOutOfRangeException(nameof(options), "MaxNewTokens must be between 1 and 4096.");
        if (options.Temperature is < 0 or > 5) throw new ArgumentOutOfRangeException(nameof(options), "Temperature must be between 0 and 5.");
        if (promptTokenIds.Length >= _contextLength || promptTokenIds.Length + options.MaxNewTokens > _contextLength)
        {
            throw new ArgumentException($"Prompt plus maximum generation exceeds context length {_contextLength}.", nameof(promptTokenIds));
        }

        _parallelOptions.CancellationToken = cancellationToken;
        var totalTimer = Stopwatch.StartNew();
        var promptTimer = Stopwatch.StartNew();
        if (promptTokenIds.Length > 0)
        {
            foreach (var tokenId in promptTokenIds)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Evaluate(tokenId, cancellationToken);
            }
        }

        promptTimer.Stop();
        var promptProcessingTime = promptTimer.Elapsed;
        var generationTimer = Stopwatch.StartNew();
        var forwardPasses = 0;
        if (promptTokenIds.Length == 0)
        {
            Evaluate(_model.Tokenizer.EndOfSequenceTokenId, cancellationToken);
            forwardPasses++;
        }

        var random = options.Seed is int seed ? new Random(seed) : Random.Shared;
        var generated = new List<int>(options.MaxNewTokens);
        var firstTokenTime = TimeSpan.Zero;
        for (var index = 0; index < options.MaxNewTokens; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var token = Sample(_logits, options.Temperature, random);
            generated.Add(token);
            onToken?.Invoke(token);
            if (index == 0)
            {
                generationTimer.Stop();
                firstTokenTime = totalTimer.Elapsed;
                generationTimer.Start();
            }

            if ((options.StopAtEndOfSequence && _model.Tokenizer.IsEndOfSequence(token)) || index == options.MaxNewTokens - 1) break;
            Evaluate(token, cancellationToken);
            forwardPasses++;
        }

        generationTimer.Stop();
        totalTimer.Stop();
        var text = _model.Tokenizer.Decode(generated, skipSpecialTokens: true);
        var promptSeconds = promptProcessingTime.TotalSeconds;
        var generationSeconds = generationTimer.Elapsed.TotalSeconds;
        using var process = Process.GetCurrentProcess();
        var workingSet = process.WorkingSet64;
        var finishReason = generated.Count > 0 && options.StopAtEndOfSequence && _model.Tokenizer.IsEndOfSequence(generated[^1])
            ? "stop"
            : "length";
        return new InferenceResult(
            text,
            generated,
            finishReason,
            new GenerationMetrics(
                promptTokenIds.Length,
                generated.Count,
                forwardPasses,
                promptProcessingTime,
                firstTokenTime,
                generationTimer.Elapsed,
                promptSeconds > 0 ? promptTokenIds.Length / promptSeconds : 0,
                generationSeconds > 0 ? generated.Count / generationSeconds : 0,
                generationSeconds > 0 ? forwardPasses / generationSeconds : 0,
                workingSet));
    }

    public void Dispose()
    {
        _disposed = true;
    }

    private void Evaluate(int tokenId, CancellationToken cancellationToken)
    {
        if ((uint)tokenId >= (uint)_model.Tokenizer.VocabularySize) throw new ArgumentOutOfRangeException(nameof(tokenId));
        cancellationToken.ThrowIfCancellationRequested();
        if (_position >= _contextLength) throw new InvalidOperationException("The inference session exhausted its context.");
        _model.Embedding.DecodeRow(tokenId, _hidden);
        ScaleInPlace(_hidden, _model.EmbeddingScale);

        var position = _position;
        for (var layerIndex = 0; layerIndex < _model.Layers.Length; layerIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var layer = _model.Layers[layerIndex];
            RmsNorm(_hidden, layer.AttentionNorm, _attentionNormalized, _model.RmsNormEpsilon);
            _attentionQ8.Quantize(_attentionNormalized);
            MappedMatrix.MultiplyTogether(
                layer.Query,
                _query,
                layer.Key,
                _key,
                layer.Value,
                _value,
                _attentionNormalized,
                _parallelOptions,
                _attentionQ8);
            ApplyRope(position);
            _cache[layerIndex].Write(position, _key, _value);
            ComputeAttention(position + 1, _query, _cache[layerIndex]);
            _attentionQ8.Quantize(_attentionContext);
            layer.AttentionOutput.Multiply(_attentionContext, _attentionProjected, parallelOptions: _parallelOptions, quantizedInput: _attentionQ8);

            for (var i = 0; i < _feedForwardInput.Length; i++)
            {
                _feedForwardInput[i] = _hidden[i] + _model.ResidualScale * _attentionProjected[i];
            }

            RmsNorm(_feedForwardInput, layer.FeedForwardNorm, _feedForwardNormalized, _model.RmsNormEpsilon);
            _feedForwardQ8.Quantize(_feedForwardNormalized);
            layer.Router.Multiply(_feedForwardNormalized, _routerLogits);
            Softmax(_routerLogits, _routerProbabilities);
            SelectExperts(_routerProbabilities);
            ComputeExperts(layer, cancellationToken);
            for (var i = 0; i < _hidden.Length; i++)
            {
                var mixture = 0f;
                for (var expert = 0; expert < _model.ExpertsPerToken; expert++) mixture += _expertScratch[expert].Output[i];
                _hidden[i] = _feedForwardInput[i] + _model.ResidualScale * mixture;
            }
        }

        RmsNorm(_hidden, _model.OutputNorm, _outputNormalized, _model.RmsNormEpsilon);
        _attentionQ8.Quantize(_outputNormalized);
        _model.OutputProjection.Multiply(_outputNormalized, _logits, parallelOptions: _parallelOptions, quantizedInput: _attentionQ8);
        var outputScale = 1f / _model.LogitScale;
        ScaleInPlace(_logits, outputScale);
        _position++;
    }

    private void ApplyRope(int position)
    {
        for (var pair = 0; pair < _ropeCos.Length; pair++)
        {
            var angle = position * _model.RopeFrequencies[pair];
            _ropeCos[pair] = MathF.Cos(angle);
            _ropeSin[pair] = MathF.Sin(angle);
        }

        RotateHeads(_query, _model.HeadCount);
        RotateHeads(_key, _model.KeyValueSize / _model.HeadDimension);

        void RotateHeads(float[] values, int headCount)
        {
            for (var head = 0; head < headCount; head++)
            {
                var offset = head * _model.HeadDimension;
                for (var pair = 0; pair < _ropeCos.Length; pair++)
                {
                    var index = offset + pair * 2;
                    var x0 = values[index];
                    var x1 = values[index + 1];
                    var cos = _ropeCos[pair];
                    var sin = _ropeSin[pair];
                    values[index] = x0 * cos - x1 * sin;
                    values[index + 1] = x0 * sin + x1 * cos;
                }
            }
        }
    }

    private void ComputeAttention(int sequenceLength, float[] query, LayerCache cache)
    {
        if (_attentionScores.Length < sequenceLength) Array.Resize(ref _attentionScores, Math.Min(_contextLength, Math.Max(sequenceLength, _attentionScores.Length * 2)));
        Array.Clear(_attentionContext);
        var queryHeadsPerKv = _model.HeadCount / (_model.KeyValueSize / _model.HeadDimension);
        for (var head = 0; head < _model.HeadCount; head++)
        {
            var queryHead = query.AsSpan(head * _model.HeadDimension, _model.HeadDimension);
            var keyValueHead = head / queryHeadsPerKv;
            var maximum = float.NegativeInfinity;
            for (var position = 0; position < sequenceLength; position++)
            {
                var score = _model.KernelBackend.DotFloat(
                    queryHead,
                    cache.Key(position).Slice(keyValueHead * _model.HeadDimension, _model.HeadDimension)) * _model.AttentionScale;
                _attentionScores[position] = score;
                if (score > maximum) maximum = score;
            }

            var sum = 0f;
            for (var position = 0; position < sequenceLength; position++)
            {
                var probability = MathF.Exp(_attentionScores[position] - maximum);
                _attentionScores[position] = probability;
                sum += probability;
            }

            var inverseSum = 1f / sum;
            var outputOffset = head * _model.HeadDimension;
            for (var position = 0; position < sequenceLength; position++)
            {
                var probability = _attentionScores[position] * inverseSum;
                var values = cache.Value(position).Slice(keyValueHead * _model.HeadDimension, _model.HeadDimension);
                for (var dimension = 0; dimension < _model.HeadDimension; dimension++)
                {
                    _attentionContext[outputOffset + dimension] += probability * values[dimension];
                }
            }
        }
    }

    private void SelectExperts(float[] probabilities)
    {
        Array.Fill(_selectedExperts, -1);
        Array.Clear(_selectedWeights);
        for (var expert = 0; expert < probabilities.Length; expert++)
        {
            var weight = probabilities[expert];
            for (var slot = 0; slot < _selectedWeights.Length; slot++)
            {
                if (weight <= _selectedWeights[slot]) continue;
                for (var move = _selectedWeights.Length - 1; move > slot; move--)
                {
                    _selectedWeights[move] = _selectedWeights[move - 1];
                    _selectedExperts[move] = _selectedExperts[move - 1];
                }

                _selectedWeights[slot] = weight;
                _selectedExperts[slot] = expert;
                break;
            }
        }

        var selectedWeightSum = 0f;
        for (var slot = 0; slot < _selectedWeights.Length; slot++) selectedWeightSum += _selectedWeights[slot];
        var inverseSum = 1f / selectedWeightSum;
        for (var slot = 0; slot < _selectedWeights.Length; slot++) _selectedWeights[slot] *= inverseSum;
    }

    private void ComputeExperts(GraniteLayer layer, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        layer.ExpertGate.MultiplyExperts(
            _expertFeedForwardInputs,
            _expertGateOutputs,
            _selectedExperts,
            _expertFeedForwardQuantizedInputs,
            _parallelOptions);
        layer.ExpertUp.MultiplyExperts(
            _expertFeedForwardInputs,
            _expertUpOutputs,
            _selectedExperts,
            _expertFeedForwardQuantizedInputs,
            _parallelOptions);

        for (var slot = 0; slot < _model.ExpertsPerToken; slot++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var scratch = _expertScratch[slot];
            for (var i = 0; i < scratch.Activation.Length; i++)
            {
                var gate = scratch.Gate[i];
                scratch.Activation[i] = (gate / (1f + MathF.Exp(-gate))) * scratch.Up[i];
            }

            scratch.ActivationQ8.Quantize(scratch.Activation);
        }

        cancellationToken.ThrowIfCancellationRequested();
        layer.ExpertDown.MultiplyExperts(
            _expertActivationInputs,
            _expertDownOutputs,
            _selectedExperts,
            _expertActivationQuantizedInputs,
            _parallelOptions);
        for (var slot = 0; slot < _model.ExpertsPerToken; slot++)
        {
            var scratch = _expertScratch[slot];
            var weight = _selectedWeights[slot];
            for (var i = 0; i < scratch.Output.Length; i++) scratch.Output[i] *= weight;
        }
    }

    private int Sample(float[] logits, float temperature, Random random)
    {
        var bestId = 0;
        var bestValue = float.NegativeInfinity;
        for (var i = 0; i < logits.Length; i++)
        {
            var value = logits[i];
            if (value > bestValue)
            {
                bestId = i;
                bestValue = value;
            }
        }

        if (temperature <= 0) return bestId;
        var sum = 0d;
        for (var i = 0; i < logits.Length; i++) sum += Math.Exp((logits[i] - bestValue) / temperature);
        var target = random.NextDouble() * sum;
        for (var i = 0; i < logits.Length; i++)
        {
            target -= Math.Exp((logits[i] - bestValue) / temperature);
            if (target <= 0) return i;
        }

        return bestId;
    }

    private void RmsNorm(float[] input, float[] weights, float[] output, float epsilon)
    {
        var sumSquares = _model.KernelBackend.DotFloat(input, input);
        var scale = 1f / MathF.Sqrt(sumSquares / input.Length + epsilon);
        for (var i = 0; i < input.Length; i++) output[i] = input[i] * scale * weights[i];
    }

    private static void Softmax(float[] input, float[] output)
    {
        var maximum = float.NegativeInfinity;
        for (var i = 0; i < input.Length; i++) if (input[i] > maximum) maximum = input[i];
        var sum = 0f;
        for (var i = 0; i < input.Length; i++)
        {
            output[i] = MathF.Exp(input[i] - maximum);
            sum += output[i];
        }

        var inverse = 1f / sum;
        for (var i = 0; i < output.Length; i++) output[i] *= inverse;
    }

    private static void ScaleInPlace(float[] values, float scale)
    {
        for (var i = 0; i < values.Length; i++) values[i] *= scale;
    }

    private sealed class LayerCache(int keyValueSize, int initialCapacity)
    {
        private int _capacity = initialCapacity;
        private float[] _keys = new float[checked(keyValueSize * initialCapacity)];
        private float[] _values = new float[checked(keyValueSize * initialCapacity)];

        public void Write(int position, float[] keys, float[] values)
        {
            EnsureCapacity(position + 1);
            keys.CopyTo(_keys.AsSpan(position * keyValueSize, keyValueSize));
            values.CopyTo(_values.AsSpan(position * keyValueSize, keyValueSize));
        }

        public ReadOnlySpan<float> Key(int position) => _keys.AsSpan(position * keyValueSize, keyValueSize);

        public ReadOnlySpan<float> Value(int position) => _values.AsSpan(position * keyValueSize, keyValueSize);

        private void EnsureCapacity(int required)
        {
            if (required <= _capacity) return;
            var newCapacity = Math.Max(required, checked(_capacity * 2));
            Array.Resize(ref _keys, checked(newCapacity * keyValueSize));
            Array.Resize(ref _values, checked(newCapacity * keyValueSize));
            _capacity = newCapacity;
        }
    }

    private sealed class ExpertScratch(int hiddenSize, int feedForwardSize)
    {
        public float[] Gate { get; } = new float[feedForwardSize];
        public float[] Up { get; } = new float[feedForwardSize];
        public float[] Activation { get; } = new float[feedForwardSize];
        public float[] Output { get; } = new float[hiddenSize];
        public QuantizedInputVector ActivationQ8 { get; } = new(feedForwardSize);
    }
}
