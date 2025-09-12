namespace MGroup.MachineLearning.TensorFlow
{
	using System;
	using System.Collections.Generic;
	using System.Diagnostics;
	using System.IO;
	using System.Text;
	using System.Text.RegularExpressions;
	using System.Timers;

	using MGroup.MachineLearning.Interfaces;
	using MGroup.MachineLearning.Preprocessing;
	using MGroup.MachineLearning.TensorFlow.Keras.Optimizers;
	using MGroup.MachineLearning.TensorFlow.KerasLayers;
	using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
	using MGroup.MachineLearning.Utilities;

	using Tensorflow;
	using Tensorflow.Clustering;
	using Tensorflow.Keras.Losses;
	using Tensorflow.Operations.Initializers;

	/// <summary>
	/// Machine learning surrogate that uses convolutional autoencoders to compress ouputs to a latent space and then a 
	/// feed-forward neural network to map inputs to the latent space.
	/// </summary>
	public class CaeFffnSurrogate : ISurrogateModel2DTo2D
	{
		private const TF_DataType DataType = TF_DataType.TF_DOUBLE;

		private readonly int _caeBatchSize;
		private readonly int _caeNumEpochs;
		private readonly float _caeLearningRate;
		private readonly int _caeKernelSize;
		private readonly int _caeStrides;
		private readonly ConvolutionPaddingType _caePadding;

		/// <summary>
		/// Another convolutional layer (with linear activation) from the last hidden layer to the output space will be 
		/// automatically added to the end.
		/// </summary>
		private readonly int[] _decoderFiltersWithoutOutput = { 32, 64, 128 };
		private readonly int[] _encoderFilters = { 128, 64, 32, 16 };

		private readonly int _ffnnBatchSize = 20;
		private readonly int _ffnnNumEpochs = 3000;
		private readonly int _ffnnNumHiddenLayers = 6;
		private readonly int _ffnnHiddenLayerSize = 64;
		private readonly float _ffnnLearningRate = 1E-4f;
		private readonly int _latentSpaceSize = 8;
		private readonly DatasetSplitter _splitter;
		private readonly int? _tfSeed;
		private readonly Func<StreamWriter> _initOutputStream;

		private ConvolutionalAutoencoder _cae;
		private FeedForwardNeuralNetwork _ffnn;

		//Delete
		private ConvolutionalNeuralNetwork _encoder;

		/// <summary>
		/// <inheritdoc/>
		/// </summary>
		public IReadOnlyList<string> ErrorNames => new string[] { "CAE error", "Surrogate error" };

		/// <summary>
		/// Creates a new instance of <see cref="CaeFffnSurrogate"/> with the specified settings
		/// </summary>
		/// <param name="caeBatchSize">The batch size when training the CAE.</param>
		/// <param name="caeNumEpochs">The number of epochs when training the CAE.</param>
		/// <param name="caeLearningRate">The learning rate when training the CAE.</param>
		/// <param name="caeKernelSize">The size of the 1D convulution kernel.</param>
		/// <param name="caeStrides">The strides for the convolutional kernel.</param>
		/// <param name="caePadding">The padding for the convolutional kernel.</param>
		/// <param name="decoderFiltersWithoutOutput">Size of each decoder layer of the CAE, except the ouput layer.</param>
		/// <param name="encoderFilters">Size of each encoder layer of the CAE, except the input layer</param>
		/// <param name="ffnnBatchSize">The batch size when training the FFNN.</param>
		/// <param name="ffnnNumEpochs">The number of epochs when training the FFNN.</param>
		/// <param name="ffnnNumHiddenLayers">The number of FFNN layers, except input and output.</param>
		/// <param name="ffnnHiddenLayerSize">The size of each FFNN layer, except input and output.</param>
		/// <param name="ffnnLearningRate">The learning rate when training the FFNN.</param>
		/// <param name="latentSpaceDim">
		/// The size of the latent space of the surrogate. It is equal to the output size of the FFNN and the input size of 
		/// the decoder.
		/// </param>
		/// <param name="splitter">
		/// Determines how to split the input/output datasets into training, test and evaluation sets.
		/// </param>
		/// <param name="tfSeed">A seed value for TensorFlow.Net, in order to reproduce runs.</param>
		/// <param name="initOutputStream">
		/// Output stream where logs collected during training and testing will be written to.
		/// </param>
		/// <exception cref="ArgumentException">Thrown if some of the settings are not compatible.</exception>
		public CaeFffnSurrogate(int caeBatchSize, int caeNumEpochs, float caeLearningRate, int caeKernelSize, int caeStrides,
			ConvolutionPaddingType caePadding, int[] decoderFiltersWithoutOutput, int[] encoderFilters,
			int ffnnBatchSize, int ffnnNumEpochs, int ffnnNumHiddenLayers, int ffnnHiddenLayerSize, float ffnnLearningRate,
			int latentSpaceDim, DatasetSplitter splitter, int? tfSeed, Func<StreamWriter> initOutputStream)
		{
			#region DEBUG
			//TODO: remove the option to set stride altogether. This is not a traditional convolution
			if (caeStrides != 1)
			{
				throw new ArgumentException("CAE stride must be 1");
			}
			#endregion

			_caeBatchSize = caeBatchSize;
			_caeNumEpochs = caeNumEpochs;
			_caeLearningRate = caeLearningRate;
			_caeKernelSize = caeKernelSize;
			_caeStrides = caeStrides;
			_caePadding = caePadding;
			_decoderFiltersWithoutOutput = decoderFiltersWithoutOutput;
			_encoderFilters = encoderFilters;
			_ffnnBatchSize = ffnnBatchSize;
			_ffnnNumEpochs = ffnnNumEpochs;
			_ffnnNumHiddenLayers = ffnnNumHiddenLayers;
			_ffnnHiddenLayerSize = ffnnHiddenLayerSize;
			_ffnnLearningRate = ffnnLearningRate;
			_latentSpaceSize = latentSpaceDim;
			_splitter = splitter;
			_tfSeed = tfSeed;

			_initOutputStream = initOutputStream;
		}

		/// <summary>
		/// Predicts the response of the original model for the provided parameter values. 
		/// </summary>
		/// <param name="input">1D array with parameter values of the original model.</param>
		/// <returns>The predicted response as an 1D array.</returns>
		public double[] Predict(double[] input)
		{
			double[,] ffnnInput = input.AddEmptyDimensions(true, false);
			double[,] ffnnPrediction = _ffnn.EvaluateResponses(ffnnInput);
			double[,,,] surrogatePrediction = _cae.MapReduced2DToFull4D(ffnnPrediction);
			double[] output = surrogatePrediction.RemoveEmptyDimensions(0, 1, 2);
			return output;
		}

		/// <summary>
		/// <inheritdoc/>
		/// </summary>
		/// <param name="inputDataset"><inheritdoc/></param>
		/// <param name="outputDataset"><inheritdoc/></param>
		/// <param name="splitter"><inheritdoc/></param>
		/// <returns><inheritdoc/></returns>
		/// <exception cref="ArgumentException"><inheritdoc/></exception>
		public Dictionary<string, double> TrainAndEvaluate(double[,] inputDataset, double[,] outputDataset, 
			DatasetSplitter? splitter)
		{
			if (splitter == null)
			{
				splitter = _splitter;
			}

			int numTotalSamples = inputDataset.GetLength(0);
			if (outputDataset.GetLength(0) != numTotalSamples)
			{
				throw new ArgumentException(
					"The first dimension of the input and ouput dataset arrays must be the same and equal to the number of " +
					$"samples, but were {numTotalSamples} and {outputDataset.GetLength(0)} respectively instead");
			}

			// Define the networks
			double[,] parameters = inputDataset;
			double[,] solutionVectors = outputDataset;
			int parameterSpaceDim = parameters.GetLength(1);
			int solutionSpaceDim = solutionVectors.GetLength(1);
			BuildAutoEncoder(solutionSpaceDim);
			BuildFfnn(parameterSpaceDim);

			// Split the input and output into train-test sets
			splitter.SetupSplittingRules(numTotalSamples);
			(double[,] trainSolutions, double[,] testSolutions, _) = splitter.SplitDataset(solutionVectors);
			(double[,] trainParameters, double[,] testParameters, _) = splitter.SplitDataset(parameters);
			double[,,,] caeTrainX = trainSolutions.AddEmptyDimensions(false, true, true, false);
			double[,,,] caeTestX = testSolutions.AddEmptyDimensions(false, true, true, false);

			// Train
			TrainCae(caeTrainX);
			TrainFfnn(caeTrainX, trainParameters, testParameters);

			// Evaluate
			var result = new Dictionary<string, double>();
			result["CAE error"] = TestCae(caeTestX);
			result["Surrogate error"] = TestFullSurrogate(testSolutions, testParameters);
			return result;
		}

		private void BuildAutoEncoder(int solutionSpaceDim)
		{
			// Encoder layers
			var encoderLayers = new List<INetworkLayer>();
			encoderLayers.Add(new InputLayer(new int[] { 1, 1, solutionSpaceDim}));
			for (int i = 0; i < _encoderFilters.Length; ++i)
			{
				encoderLayers.Add(new Convolutional2DLayer(_encoderFilters[i], (_caeKernelSize, 1), ActivationType.RelU,
					strides: (_caeStrides, 1), padding: _caePadding.GetNameForTensorFlow(), dilationRate:1));
			}
			encoderLayers.Add(new FlattenLayer());
			encoderLayers.Add(new DenseLayer(_latentSpaceSize, ActivationType.Linear)); // Output layer. Activation: f(x) = x

			// Decoder layers
			var decoderLayers = new List<INetworkLayer>();
			//decoderLayers.Add(new InputLayer(new int[] { _latentSpaceSize })); // CAE class does this itself
			int denseLayerSize = _decoderFiltersWithoutOutput[0] / 2; // In python code, it did not divide over 2. 
			decoderLayers.Add(new DenseLayer(denseLayerSize, ActivationType.RelU)); // This is 16 in the paper
			decoderLayers.Add(new ReshapeLayer(new int[] { 1, 1, denseLayerSize }));
			for (int i = 0; i < _decoderFiltersWithoutOutput.Length; ++i)
			{
				decoderLayers.Add(new Convolutional2DTransposeLayer(_decoderFiltersWithoutOutput[i], (_caeKernelSize, 1),
					ActivationType.RelU, strides: (_caeStrides, 1), padding: _caePadding.GetNameForTensorFlow(), dilationRate:1));
			}

			// Output layer. Activation: f(x) = x
			decoderLayers.Add(new Convolutional2DTransposeLayer(solutionSpaceDim, (_caeKernelSize, 1), 
				ActivationType.Linear, strides: (_caeStrides, 1), padding: _caePadding.GetNameForTensorFlow(), dilationRate: 1)); 

			INormalization normalizationX = new NullNormalization();
			var optimizer = new Adam(dataType: DataType, learning_rate: _caeLearningRate);
			ILossFunc lossFunction = KerasApi.keras.losses.MeanSquaredError();
			_cae = new ConvolutionalAutoencoder(normalizationX, optimizer, lossFunction, encoderLayers.ToArray(),
				decoderLayers.ToArray(), _caeNumEpochs, _caeBatchSize, _tfSeed, shuffleTrainingData: true);
		}

		private void BuildFfnn(int parameterSpaceDim)
		{
			var layers = new List<INetworkLayer>();
			layers.Add(new InputLayer(new int[] { parameterSpaceDim }));
			for (int i = 0; i < _ffnnNumHiddenLayers; i++) 
			{
				layers.Add(new DenseLayer(_ffnnHiddenLayerSize, ActivationType.RelU));
			};
			layers.Add(new DenseLayer(_latentSpaceSize, ActivationType.Linear)); // Output layer. Activation: f(x) = x

			INormalization normalizationX = new NullNormalization();
			INormalization normalizationY = new NullNormalization();
			var optimizer = new Adam(dataType: DataType, learning_rate: _ffnnLearningRate);
			ILossFunc lossFunction = KerasApi.keras.losses.MeanSquaredError();
			_ffnn = new FeedForwardNeuralNetwork(normalizationX, normalizationY, optimizer, lossFunction, layers.ToArray(),
				_ffnnNumEpochs, _ffnnBatchSize, _tfSeed, shuffleTrainingData:true);
		}

		private void TrainCae(double[,,,] trainSolutions)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Training convolutional autoencoder:");
			watch.Start();
			_cae.Train(trainSolutions); // python code also used the test data as validation set here.
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine();

			writer.Close();
		}

		private void TrainFfnn(double[,,,] trainSolutions, double[,] trainParameters, double[,] testParameters)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Prepare FFNN output data using convolutional encoder");
			watch.Start();

			// RemoveEmptyDimensions() did not work correctly for [160, 1, 1, 8]. It produced [160,1]. Must test all add/remove empty dimension methods
			//double[,] ffnnTrainY = _cae
			//	.MapFullToReduced(trainSolutions)
			//	.RemoveEmptyDimensions(1, 2); 

			double[,] ffnnTrainY = _cae.MapFull4DToReduced2D(trainSolutions);
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);

			writer.WriteLine("Training feed forward neural network:");
			watch.Restart();
			_ffnn.Train(trainParameters, ffnnTrainY); // python code also used the encoded test data as validation set here.
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);

			writer.Close();
		}

		private double TestCae(double[,,,] testSolutions)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Testing convolutional autoencoder:");
			watch.Start();
			double[,,,] caePredictions = _cae.EvaluateResponses(testSolutions);
			double error = ErrorMetrics.CalculateMeanNorm2Error(testSolutions, caePredictions);
			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine($"Mean error = 1/numSamples * sumOverSamples( norm2(CAE(u) - u) / norm2(u) = {error}");
			writer.WriteLine();

			writer.Close();
			return error;
		}

		private double TestFullSurrogate(double[,] testSolutions, double[,] testParameters)
		{
			var watch = new Stopwatch();
			StreamWriter writer = _initOutputStream();

			writer.WriteLine("Testing convolutional autoencoder:");
			watch.Start();
			double[,] ffnnPredictions = _ffnn.EvaluateResponses(testParameters);
			double[,,,] surrogatePredictions = _cae.MapReduced2DToFull4D(ffnnPredictions);
			double error = ErrorMetrics.CalculateMeanNorm2Error(
				testSolutions.AddEmptyDimensions(false, true, true, false), surrogatePredictions);
			
			//double[,,,] surrogatePredictions = _cae.MapReducedToFull(
			//	ffnnPredictions.AddEmptyDimensions(false, true, false, true));
			//double error = ErrorMetrics.CalculateMeanNorm2Error(
			//	testSolutions.AddEmptyDimensions(false, true, false, true), surrogatePredictions);

			watch.Stop();
			writer.WriteLine("Ellapsed ms: " + watch.ElapsedMilliseconds);
			writer.WriteLine($"Mean error = 1/numSamples * sumOverSamples( norm2(surrogate(theta) - u) / norm2(u) = {error}");

			writer.Close();
			return error;
		}

		/// <summary>
		/// Helper class to facilitate the creation of <see cref="CaeFffnSurrogate"/>
		/// </summary>
		public class Builder
		{
			/// <summary>
			/// Creates a new instance of <see cref="Builder"/> with default settings.
			/// </summary>
			public Builder()
			{
				//GetOutputStream = () =>
				//{
				//	var writer = new StreamWriter(Console.OpenStandardOutput());
				//	writer.AutoFlush = true;
				//	Console.SetOut(writer);
				//	return writer;
				//};
				GetOutputStream = () => new DebugTextWriter();

				Splitter = new DatasetSplitter();
				Splitter.MinTestSetPercentage = 0.2;
				Splitter.MinValidationSetPercentage = 0.0;
				Splitter.SetOrderToContiguous(DataSubsetType.Training, DataSubsetType.Test);
			}

			/// <summary>
			/// The batch size when training the CAE.
			/// </summary>
			public int CaeBatchSize { get; set; } = 10;

			/// <summary>
			/// The number of epochs when training the CAE.
			/// </summary>
			public int CaeNumEpochs { get; set; } = 40;

			/// <summary>
			/// The learning rate when training the CAE.
			/// </summary>
			public float CaeLearningRate { get; set; } = 5E-4f;

			/// <summary>
			/// The size of the 1D convulution kernel.
			/// </summary>
			public int CaeKernelSize { get; set; } = 5;

			/// <summary>
			/// The strides for the convolutional kernel.
			/// </summary>
			public int CaeStrides { get; set; } = 1;

			/// <summary>
			/// The padding for the convolutional kernel.
			/// </summary>
			public ConvolutionPaddingType CaePadding { get; set; } = ConvolutionPaddingType.Same;

			/// <summary>
			/// Size of each decoder layer of the CAE, except the ouput layer. Another convolutional layer (with linear 
			/// activation) from the last hidden layer to the output space will be automatically added to the end.
			/// </summary>
			public int[] DecoderFiltersWithoutOutput { get; set; } = { 32, 64, 128 };

			/// <summary>
			/// Size of each encoder layer of the CAE, except the input layer.
			/// </summary>
			public int[] EncoderFilters { get; set; } = { 128, 64, 32, 16 };

			/// <summary>
			/// The batch size when training the FFNN.
			/// </summary>
			public int FfnnBatchSize { get; set; } = 20;

			/// <summary>
			/// The number of epochs when training the FFNN.
			/// </summary>
			public int FfnnNumEpochs { get; set; } = 3000;

			/// <summary>
			/// The number of FFNN layers, except input and output.
			/// </summary>
			public int FfnnNumHiddenLayers { get; set; } = 6;

			/// <summary>
			/// The size of each FFNN layer, except input and output.
			/// </summary>
			public int FfnnHiddenLayerSize { get; set; } = 64;

			/// <summary>
			/// The learning rate when training the FFNN.
			/// </summary>
			public float FfnnLearningRate { get; set; } = 1E-4f;

			/// <summary>
			/// Function that provides the output stream where logs collected during training and testing will be written to.
			/// </summary>
			public Func<StreamWriter> GetOutputStream { get; set; }

			/// <summary>
			/// The size of the latent space of the surrogate. It is equal to the output size of the FFNN and the input size of 
			/// the decoder.
			/// </summary>
			public int LatentSpaceDim { get; set; } = 8;

			/// <summary>
			/// Determines how to split the input/output datasets into training, test and evaluation sets.
			/// </summary>
			public DatasetSplitter Splitter { get; set; }

			/// <summary>
			/// A seed value for TensorFlow.Net, in order to reproduce runs.
			/// </summary>
			public int? TensorFlowSeed { get; set; } = null;

			/// <summary>
			/// Creates a new instance of <see cref="CaeFffnSurrogate"/> based on the properties of this object.
			/// </summary>
			/// <returns>A new instance of <see cref="CaeFffnSurrogate"/></returns>
			public CaeFffnSurrogate BuildSurrogate()
			{
				return new CaeFffnSurrogate(CaeBatchSize, CaeNumEpochs, CaeLearningRate, CaeKernelSize, CaeStrides, CaePadding, 
					DecoderFiltersWithoutOutput, EncoderFilters, FfnnBatchSize, FfnnNumEpochs, FfnnNumHiddenLayers, 
					FfnnHiddenLayerSize, FfnnLearningRate, LatentSpaceDim, Splitter, TensorFlowSeed, GetOutputStream);
			}
		}
	}
}
