using System;

namespace MGroup.MachineLearning.Preprocessing
{
    /// <summary>
    /// No normalization takes place.
    /// </summary>
    [Serializable]
	public class NullNormalization : INormalization
	{
		public double[] ScalingRatio { get; private set; }

		public void Initialize(double[,] data, NormalizationDirection direction)
		{
			ScalingRatio = new double[data.GetLength(1)];
			for (int i = 0; i < data.GetLength(1); i++)
			{
				ScalingRatio[i] = 1;
			}
		}

		public double[,] Normalize(double[,] data)
		{
			return data;
		}

		public double[,] Denormalize(double[,] data)
		{
			return data;
		}
	}
}
