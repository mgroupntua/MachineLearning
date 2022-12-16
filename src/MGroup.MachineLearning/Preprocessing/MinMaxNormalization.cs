using System;

namespace MGroup.MachineLearning.Preprocessing
{
    /// <summary>
    /// Normalize the data using the
    /// MinMax normalization so that
    /// their values lie in
    /// the [0,1] domain.
    /// </summary>
    [Serializable]
	public class MinMaxNormalization : INormalization
	{
		public double[] MinValuePerDirection { get; private set; }
		public double[] MaxValuePerDirection { get; private set; }
		public NormalizationDirection NormalizationDirection { get; private set; }
        public double[] ScalingRatio { get; private set; }

        public void Initialize(double[,] data, NormalizationDirection direction)
		{
            NormalizationDirection = direction;
			if (NormalizationDirection == NormalizationDirection.PerRow)
			{
				MinValuePerDirection = new double[data.GetLength(0)];
				MaxValuePerDirection = new double[data.GetLength(0)];

				for (int row = 0; row < data.GetLength(0); row++)
				{
					MinValuePerDirection[row] = double.MaxValue;
					MaxValuePerDirection[row] = double.MinValue;

					for (int col = 0; col < data.GetLength(1); col++)
					{
						if (data[row, col] < MinValuePerDirection[row])
						{
							MinValuePerDirection[row] = data[row, col];
						}

						if (data[row, col] > MaxValuePerDirection[row])
						{
							MaxValuePerDirection[row] = data[row, col];
						}
					}
				}

				// TODO: Set scaling ratio, maybe?
			}
            else if (NormalizationDirection == NormalizationDirection.PerColumn)
            {
                MinValuePerDirection = new double[data.GetLength(1)];
				MaxValuePerDirection = new double[data.GetLength(1)];
				ScalingRatio = new double[data.GetLength(1)];

				for (int col = 0; col < data.GetLength(1); col++)
				{
					MinValuePerDirection[col] = double.MaxValue;
					MaxValuePerDirection[col] = double.MinValue;

					for (int row = 0; row < data.GetLength(0); row++)
					{
						if (data[row, col] < MinValuePerDirection[col])
						{
							MinValuePerDirection[col] = data[row, col];
						}

						if (data[row, col] > MaxValuePerDirection[col])
						{
							MaxValuePerDirection[col] = data[row, col];
						}
					}

					ScalingRatio[col] = MaxValuePerDirection[col] - MinValuePerDirection[col];
				}
			}
		}

		public double[,] Normalize(double[,] data)
		{
			double[,] scaledData = new double[data.GetLength(0), data.GetLength(1)];
			if (NormalizationDirection == NormalizationDirection.PerRow)
			{
				for (int row = 0; row < data.GetLength(0); row++)
				{
					for (int col = 0; col < data.GetLength(1); col++)
					{
						scaledData[row, col] = (data[row, col] - MinValuePerDirection[row]) / (MaxValuePerDirection[row] - MinValuePerDirection[row]);
					}
				}
				return scaledData;
			}
			else if (NormalizationDirection == NormalizationDirection.PerColumn)
			{
				for (int col = 0; col < data.GetLength(1); col++)
				{
					for (int row = 0; row < data.GetLength(0); row++)
					{
						scaledData[row, col] = (data[row, col] - MinValuePerDirection[col]) / (MaxValuePerDirection[col] - MinValuePerDirection[col]);
					}
				}
				return scaledData;
			}
			else
			{
				throw new ArgumentException("NormalizationDirection should be either PerRow or PerColumn");
			}
		}

		public double[,] Denormalize(double[,] scaledData)
		{
            if (NormalizationDirection == NormalizationDirection.PerRow)
            {
                double[,] data = new double[scaledData.GetLength(0), scaledData.GetLength(1)];
				for (int row = 0; row < scaledData.GetLength(0); row++)
				{
					for (int col = 0; col < scaledData.GetLength(1); col++)
					{
						data[row, col] = scaledData[row, col] * (MaxValuePerDirection[row] - MinValuePerDirection[row]) + MinValuePerDirection[row];
					}
				}
				return data;
			}
            else if (NormalizationDirection == NormalizationDirection.PerColumn)
            {
                double[,] data = new double[scaledData.GetLength(0), scaledData.GetLength(1)];
				for (int col = 0; col < scaledData.GetLength(1); col++)
				{
					for (int row = 0; row < scaledData.GetLength(0); row++)
					{
						data[row, col] = scaledData[row, col] * (MaxValuePerDirection[col] - MinValuePerDirection[col]) + MinValuePerDirection[col];
					}
				}
				return data;
			}
			else
			{
                throw new ArgumentException("NormalizationDirection should be either PerRow or PerColumn");
            }
        }
	}
}
