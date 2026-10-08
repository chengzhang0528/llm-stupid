using LlmStupid.Engine.Model;

namespace LlmStupid.Engine.Runtime;

public sealed class MappedMatrix(GgufFile file, GgufTensor tensor, IQuantKernelBackend backend)
{
    private readonly IQuantKernelBackend _backend = backend;
    private readonly GgufTensorDataView _tensorData = file.CreateTensorDataView(tensor);

    public GgufTensor Tensor { get; } = tensor;

    public int InputLength => Tensor.RowLength;

    public int OutputLength => checked((int)Tensor.Dimensions[1]);

    public int ExpertCount => Tensor.Dimensions.Length >= 3 ? checked((int)Tensor.Dimensions[2]) : 1;

    public void Multiply(
        float[] input,
        float[] output,
        int expert = 0,
        ParallelOptions? parallelOptions = null,
        QuantizedInputVector? quantizedInput = null)
    {
        if (input.Length != InputLength) throw new ArgumentException($"Input length {input.Length} does not match {InputLength} for {Tensor.Name}.");
        if (output.Length < OutputLength) throw new ArgumentException($"Output buffer is too short for {Tensor.Name}.");
        if ((uint)expert >= ExpertCount) throw new ArgumentOutOfRangeException(nameof(expert));

        var rowBytes = checked((InputLength / Tensor.BlockSize) * Tensor.BytesPerBlock);
        if (InputLength % Tensor.BlockSize != 0) throw new InvalidDataException($"Tensor {Tensor.Name} has a row that crosses quantization blocks.");
        var firstRow = checked(expert * OutputLength);
        var useQuantizedInput = quantizedInput is not null
            && _backend.SupportsQuantizedInput
            && Tensor.Type is GgmlType.Q4K or GgmlType.Q6K;
        if (useQuantizedInput && quantizedInput!.Length != input.Length)
        {
            throw new ArgumentException("Quantized input length does not match the matrix input.", nameof(quantizedInput));
        }

        if (useQuantizedInput && Tensor.Type == GgmlType.Q4K && OutputLength >= 4)
        {
            const int rowsPerTile = 16;
            var tileCount = OutputLength / rowsPerTile;
            void ComputeTile(int tile)
            {
                var firstTileRow = tile * rowsPerTile;
                var weights = _tensorData.Slice(
                    checked((firstRow + firstTileRow) * rowBytes),
                    checked(rowsPerTile * rowBytes));
                if (!_backend.TryDotQuantizedRows(
                    Tensor.Type,
                    weights,
                    rowsPerTile,
                    quantizedInput!,
                    output.AsSpan(firstTileRow, rowsPerTile)))
                {
                    for (var row = firstTileRow; row < firstTileRow + rowsPerTile; row++) ComputeRow(row);
                }
            }

            if (parallelOptions is not null && OutputLength >= 128)
                Parallel.For(0, tileCount, parallelOptions, ComputeTile);
            else
                for (var tile = 0; tile < tileCount; tile++) ComputeTile(tile);

            for (var row = tileCount * rowsPerTile; row < OutputLength; row++) ComputeRow(row);
        }
        else if (parallelOptions is not null && OutputLength >= 128)
        {
            Parallel.For(0, OutputLength, parallelOptions, ComputeRow);
        }
        else
        {
            for (var row = 0; row < OutputLength; row++) ComputeRow(row);
        }

        void ComputeRow(int row)
        {
            var weights = _tensorData.Slice(checked((firstRow + row) * rowBytes), rowBytes);
            output[row] = useQuantizedInput
                ? _backend.DotQuantized(Tensor.Type, weights, quantizedInput!)
                : _backend.Dot(Tensor.Type, weights, input);
        }
    }

    internal static void MultiplyTogether(
        MappedMatrix first,
        float[] firstOutput,
        MappedMatrix second,
        float[] secondOutput,
        MappedMatrix third,
        float[] thirdOutput,
        float[] input,
        ParallelOptions parallelOptions,
        QuantizedInputVector? quantizedInput)
    {
        if (first.InputLength != input.Length || second.InputLength != input.Length || third.InputLength != input.Length)
        {
            throw new ArgumentException("Fused matrices must have matching input lengths.", nameof(input));
        }

        if (firstOutput.Length < first.OutputLength || secondOutput.Length < second.OutputLength || thirdOutput.Length < third.OutputLength)
        {
            throw new ArgumentException("One or more fused matrix outputs are too short.");
        }

        var firstUsesQuantized = first._backend.SupportsQuantizedInput && first.Tensor.Type is GgmlType.Q4K or GgmlType.Q6K;
        var secondUsesQuantized = second._backend.SupportsQuantizedInput && second.Tensor.Type is GgmlType.Q4K or GgmlType.Q6K;
        var thirdUsesQuantized = third._backend.SupportsQuantizedInput && third.Tensor.Type is GgmlType.Q4K or GgmlType.Q6K;
        if ((firstUsesQuantized || secondUsesQuantized || thirdUsesQuantized)
            && (quantizedInput is null || quantizedInput.Length != input.Length))
        {
            throw new ArgumentException("Quantized input length does not match the fused matrices.", nameof(quantizedInput));
        }

        var firstRowBytes = GetRowBytes(first);
        var secondRowBytes = GetRowBytes(second);
        var thirdRowBytes = GetRowBytes(third);
        var firstRowsPerWork = firstUsesQuantized && first.Tensor.Type == GgmlType.Q4K ? 4 : 1;
        var secondRowsPerWork = secondUsesQuantized && second.Tensor.Type == GgmlType.Q4K ? 4 : 1;
        var thirdRowsPerWork = thirdUsesQuantized && third.Tensor.Type == GgmlType.Q4K ? 4 : 1;
        var firstTileCount = first.OutputLength / firstRowsPerWork;
        var secondTileCount = second.OutputLength / secondRowsPerWork;
        var thirdTileCount = third.OutputLength / thirdRowsPerWork;
        var firstWorkCount = firstTileCount + first.OutputLength % firstRowsPerWork;
        var secondWorkCount = secondTileCount + second.OutputLength % secondRowsPerWork;
        var thirdWorkCount = thirdTileCount + third.OutputLength % thirdRowsPerWork;
        var secondWorkStart = firstWorkCount;
        var thirdWorkStart = checked(firstWorkCount + secondWorkCount);
        var totalWork = checked(thirdWorkStart + thirdWorkCount);

        if (parallelOptions.MaxDegreeOfParallelism != 1 && totalWork >= 128)
            Parallel.For(0, totalWork, parallelOptions, ComputeWork);
        else
            for (var work = 0; work < totalWork; work++) ComputeWork(work);

        void ComputeWork(int work)
        {
            if (work < secondWorkStart)
            {
                first.ComputeMatrixWork(work, firstRowBytes, firstRowsPerWork, firstTileCount, input, firstOutput, quantizedInput, firstUsesQuantized);
            }
            else if (work < thirdWorkStart)
            {
                second.ComputeMatrixWork(work - secondWorkStart, secondRowBytes, secondRowsPerWork, secondTileCount, input, secondOutput, quantizedInput, secondUsesQuantized);
            }
            else
            {
                third.ComputeMatrixWork(work - thirdWorkStart, thirdRowBytes, thirdRowsPerWork, thirdTileCount, input, thirdOutput, quantizedInput, thirdUsesQuantized);
            }
        }
    }

    private static int GetRowBytes(MappedMatrix matrix)
    {
        if (matrix.InputLength % matrix.Tensor.BlockSize != 0)
        {
            throw new InvalidDataException($"Tensor {matrix.Tensor.Name} has a row that crosses quantization blocks.");
        }

        return checked((matrix.InputLength / matrix.Tensor.BlockSize) * matrix.Tensor.BytesPerBlock);
    }

    private void ComputeMatrixWork(
        int work,
        int rowBytes,
        int rowsPerWork,
        int tileCount,
        float[] input,
        float[] output,
        QuantizedInputVector? quantizedInput,
        bool useQuantizedInput)
    {
        var firstOutputRow = rowsPerWork > 1 && work >= tileCount
            ? tileCount * rowsPerWork + work - tileCount
            : work * rowsPerWork;
        var rowCount = Math.Min(rowsPerWork, OutputLength - firstOutputRow);
        if (useQuantizedInput && Tensor.Type == GgmlType.Q4K && rowCount == 4)
        {
            var firstRow = checked(firstOutputRow * rowBytes);
            var weights = _tensorData.Slice(firstRow, checked(rowCount * rowBytes));
            if (_backend.TryDotQuantizedRows(Tensor.Type, weights, rowCount, quantizedInput!, output.AsSpan(firstOutputRow, rowCount))) return;
        }

        for (var row = 0; row < rowCount; row++)
        {
            var weights = _tensorData.Slice(checked((firstOutputRow + row) * rowBytes), rowBytes);
            output[firstOutputRow + row] = useQuantizedInput
                ? _backend.DotQuantized(Tensor.Type, weights, quantizedInput!)
                : _backend.Dot(Tensor.Type, weights, input);
        }
    }

    internal void MultiplyExperts(
        float[][] inputs,
        float[][] outputs,
        int[] experts,
        QuantizedInputVector[] quantizedInputs,
        ParallelOptions parallelOptions)
    {
        var expertCount = experts.Length;
        if (inputs.Length < expertCount || outputs.Length < expertCount || quantizedInputs.Length < expertCount)
        {
            throw new ArgumentException($"Expert buffers are incomplete for {Tensor.Name}.");
        }

        var rowBytes = checked((InputLength / Tensor.BlockSize) * Tensor.BytesPerBlock);
        if (InputLength % Tensor.BlockSize != 0) throw new InvalidDataException($"Tensor {Tensor.Name} has a row that crosses quantization blocks.");
        var useQuantizedInput = _backend.SupportsQuantizedInput && Tensor.Type is GgmlType.Q4K or GgmlType.Q6K;
        for (var slot = 0; slot < expertCount; slot++)
        {
            if (inputs[slot].Length != InputLength) throw new ArgumentException($"Input length {inputs[slot].Length} does not match {InputLength} for {Tensor.Name}.");
            if (outputs[slot].Length < OutputLength) throw new ArgumentException($"Output buffer is too short for {Tensor.Name}.");
            if ((uint)experts[slot] >= ExpertCount) throw new ArgumentOutOfRangeException(nameof(experts));
            if (useQuantizedInput && quantizedInputs[slot].Length != InputLength)
            {
                throw new ArgumentException("Quantized input length does not match the matrix input.", nameof(quantizedInputs));
            }
        }

        var rowsPerTile = useQuantizedInput && Tensor.Type == GgmlType.Q4K && OutputLength >= 4 ? 4 : 1;
        var tileCount = OutputLength / rowsPerTile;
        var workPerExpert = tileCount + OutputLength % rowsPerTile;
        var totalWork = checked(expertCount * workPerExpert);
        if (parallelOptions.MaxDegreeOfParallelism != 1 && totalWork >= 128)
            Parallel.For(0, totalWork, parallelOptions, ComputeWork);
        else
            for (var work = 0; work < totalWork; work++) ComputeWork(work);

        void ComputeWork(int work)
        {
            var slot = work / workPerExpert;
            var item = work % workPerExpert;
            if (rowsPerTile > 1 && item < tileCount)
            {
                var firstTileRow = item * rowsPerTile;
                var firstRow = checked(experts[slot] * OutputLength + firstTileRow);
                var weights = _tensorData.Slice(
                    checked(firstRow * rowBytes),
                    checked(rowsPerTile * rowBytes));
                if (!_backend.TryDotQuantizedRows(
                    Tensor.Type,
                    weights,
                    rowsPerTile,
                    quantizedInputs[slot],
                    outputs[slot].AsSpan(firstTileRow, rowsPerTile)))
                {
                    for (var row = firstTileRow; row < firstTileRow + rowsPerTile; row++) ComputeRow(slot, row);
                }

                return;
            }

            var rowIndex = rowsPerTile > 1 ? tileCount * rowsPerTile + item - tileCount : item;
            ComputeRow(slot, rowIndex);
        }

        void ComputeRow(int slot, int row)
        {
            var firstRow = checked(experts[slot] * OutputLength + row);
            var weights = _tensorData.Slice(checked(firstRow * rowBytes), rowBytes);
            outputs[slot][row] = useQuantizedInput
                ? _backend.DotQuantized(Tensor.Type, weights, quantizedInputs[slot])
                : _backend.Dot(Tensor.Type, weights, inputs[slot]);
        }
    }

    public void DecodeRow(int row, Span<float> output, int expert = 0)
    {
        if (output.Length != InputLength) throw new ArgumentException($"Output length {output.Length} does not match {InputLength} for {Tensor.Name}.");
        if ((uint)row >= OutputLength) throw new ArgumentOutOfRangeException(nameof(row));
        if ((uint)expert >= ExpertCount) throw new ArgumentOutOfRangeException(nameof(expert));
        var rowBytes = checked((InputLength / Tensor.BlockSize) * Tensor.BytesPerBlock);
        var firstRow = checked(expert * OutputLength + row);
        _backend.DecodeRow(Tensor.Type, _tensorData.Slice(checked(firstRow * rowBytes), rowBytes), output);
    }
}
