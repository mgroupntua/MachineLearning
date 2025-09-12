namespace MGroup.MachineLearning.Interfaces
{
	using System;
	using System.Collections.Generic;
	using System.Text;

	using MGroup.MachineLearning.Utilities;

	/// <summary>
	/// Surrogate that maps a double[] input to a double[] output.
	/// </summary>
	public interface ISurrogateModel2DTo2D
	{
		/// <summary>
		/// The names of error metrics for each dataset (training, test, evaluation).
		/// </summary>
		IReadOnlyList<string> ErrorNames { get; }

		/// <summary>
		/// Trains the surrogate model, tests its various parts and returns the corresponding errors.
		/// </summary>
		/// <param name="inputDataset">
		/// 2D array with the inputs of the surrogate. GetLength(0) must be equal to the number of combined samples 
		/// (training, test, evaluation).
		/// </param>
		/// <param name="outputDataset">
		/// 2D array with the outputs of the surrogate. GetLength(0) must be equal to the number of combined samples 
		/// (training, test, evaluation).
		/// </param>
		/// <param name="splitter">
		/// Determines how to split the input/output datasets into training, test and evaluation sets. 
		/// If null, the surrogate will decide how to split the datasets.
		/// </param>
		/// <returns>Error metrics for the datasets</returns>
		Dictionary<string, double> TrainAndEvaluate(double[,] inputDataset, double[,] outputDataset, DatasetSplitter splitter);
	}
}
