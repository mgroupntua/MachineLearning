using System;

namespace MGroup.MachineLearning.Preprocessing
{
    /// <summary>
    /// Normalize the data using the
    /// Z-score normalization so that
    /// their values have zero mean and unit variance.
    /// </summary>
    [Serializable]
	public class ZScoreNormalization : INormalization
	{
		public double[] MeanValuePerDirection { get; private set; }
		public double[] StandardDeviationPerDirection { get; private set; }
        public NormalizationDirection NormalizationDirection { get; private set; }

        public double[] ScalingRatio => StandardDeviationPerDirection;

        public void Initialize(double[,] data, NormalizationDirection direction)
        {
            NormalizationDirection = direction;
            if (NormalizationDirection == NormalizationDirection.PerRow)
            {
                MeanValuePerDirection = new double[data.GetLength(0)];
				StandardDeviationPerDirection = new double[data.GetLength(0)];
				for (int row = 0; row < data.GetLength(0); row++)
				{
					for (int col = 0; col < data.GetLength(1); col++)
					{
						MeanValuePerDirection[row] += data[row, col];
					}
					MeanValuePerDirection[row] = MeanValuePerDirection[row] / data.GetLength(1);
				}

				for (int row = 0; row < data.GetLength(0); row++)
				{
					for (int col = 0; col < data.GetLength(1); col++)
					{
						StandardDeviationPerDirection[row] += Math.Pow(data[row, col] - MeanValuePerDirection[row], 2);
					}
					StandardDeviationPerDirection[row] = Math.Sqrt(StandardDeviationPerDirection[row] / (data.GetLength(1) - 1));
				}
			}
            else if (NormalizationDirection == NormalizationDirection.PerColumn)
            {
                MeanValuePerDirection = new double[data.GetLength(1)];
				StandardDeviationPerDirection = new double[data.GetLength(1)];
				for (int col = 0; col < data.GetLength(1); col++)
				{
					for (int row = 0; row < data.GetLength(0); row++)
					{
						MeanValuePerDirection[col] += data[row, col];
					}
					MeanValuePerDirection[col] = MeanValuePerDirection[col] / data.GetLength(0);
				}

				for (int col = 0; col < data.GetLength(1); col++)
				{
					for (int row = 0; row < data.GetLength(0); row++)
					{
						StandardDeviationPerDirection[col] += Math.Pow(data[row, col] - MeanValuePerDirection[col], 2);
					}
					StandardDeviationPerDirection[col] = Math.Sqrt(StandardDeviationPerDirection[col] / (data.GetLength(0) - 1));
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
						scaledData[row, col] = (data[row, col] - MeanValuePerDirection[row]) / StandardDeviationPerDirection[row];
					}
				}
			}
            else if (NormalizationDirection == NormalizationDirection.PerColumn)
            {
                for (int col = 0; col < data.GetLength(1); col++)
				{
					for (int row = 0; row < data.GetLength(0); row++)
					{
						scaledData[row, col] = (data[row, col] - MeanValuePerDirection[col]) / StandardDeviationPerDirection[col];
					}
				}
			}
            else
            {
                throw new ArgumentException("NormalizationDirection should be either PerRow or PerColumn");
            }

            return (scaledData);
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
						data[row, col] = scaledData[row, col] * StandardDeviationPerDirection[row] + MeanValuePerDirection[row];
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
						data[row, col] = scaledData[row, col] * StandardDeviationPerDirection[col] + MeanValuePerDirection[col];
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
