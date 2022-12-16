using System;
using Xunit;
using static Tensorflow.KerasApi;
using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
using MGroup.MachineLearning.Preprocessing;
using MGroup.MachineLearning.TensorFlow.KerasLayers;
using Tensorflow.Keras.Losses;

namespace MGroup.MachineLearning.Tests
{
	public class FeedForwardNeuralNetworkTest
    {
		//learning the polynomial : f(x)=x^2 in x -> [-1 , 1], f'(x)=2x
		static double[,] trainX = { { -1 }, { -0.9 }, { -0.8 }, { -0.7 }, { -0.6 }, { -0.5 }, { -0.4 }, { -0.3 }, { -0.2 }, { -0.1 }, { 0 }, { 0.1 }, { 0.2 }, { 0.3 }, { 0.4 }, { 0.5 }, { 0.6 }, { 0.7 }, { 0.8 }, { 0.9 } };

		static double[,] trainY = CalculateFunction(trainX);

		static double[,] testX = { { -0.7 }, { -0.3 }, { 0.3 }, { 0.7 } };

		static double[,] testY = CalculateFunction(testX);

		static double[][,] testGradient = { new double[1, 1] { { 2 * -0.7 } }, new double[1, 1] { { 2 * -0.3 } }, new double[1, 1] { { 2 * 0.3 } }, new double[1, 1] { { 2 * 0.7 } } };

		private static double[,] CalculateFunction(double[,] trainX)
		{
			var trainY = new double[trainX.GetLength(0), trainX.GetLength(1)];
			for (int i = 0; i < trainX.GetLength(0); i++)
			{
				trainY[i, 0] = Math.Pow(trainX[i, 0], 2);
			}
			return trainY;
		}

		[Fact]
		public static void WithoutNormalizationWithAdam() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new NullNormalization(), new NullNormalization(),
		new TensorFlow.Keras.Optimizers.Adam(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.05f),
		keras.losses.MeanSquaredError(), new INetworkLayer[]
		{
					new InputLayer(new int[1]{1}),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(1, ActivationType.Linear),
		},
		200));

		[Fact(Skip = "INeuralNetwork.EvaluateResponseGradients doesn't work for SGD optimizer")]
		public static void WithoutNormalizationWithSGD() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new NullNormalization(), new NullNormalization(),
				new TensorFlow.Keras.Optimizers.SGD(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.1f),
				keras.losses.MeanSquaredError(), new INetworkLayer[]
				{
					new InputLayer(new int[1]{1}),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(1, ActivationType.Linear),
				},
				200));

		[Fact]
		public static void WithoutNormalizationWithRMSProp() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new NullNormalization(), new NullNormalization(),
				new TensorFlow.Keras.Optimizers.RMSProp(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.1f),
				keras.losses.MeanSquaredError(), new INetworkLayer[]
				{
					new InputLayer(new int[1]{1}),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(1, ActivationType.Linear),
				},
				200));

		[Fact]
        public static void MinMaxNormalizationWithAdam() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new MinMaxNormalization(), new MinMaxNormalization(),
                new TensorFlow.Keras.Optimizers.Adam(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.05f),
                keras.losses.MeanSquaredError(), new INetworkLayer[]
                {
					new InputLayer(new int[1]{1}),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(1, ActivationType.Linear),
				},
				200));

		[Fact(Skip = "INeuralNetwork.EvaluateResponseGradients doesn't work for SGD optimizer")]
		public static void MinMaxNormalizationWithSGD() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new MinMaxNormalization(), new MinMaxNormalization(),
                new TensorFlow.Keras.Optimizers.SGD(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.1f),
                keras.losses.MeanSquaredError(), new INetworkLayer[]
                {
					new InputLayer(new int[1]{1}),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(1, ActivationType.Linear),
				},
				200));

        [Fact]
        public static void MinMaxNormalizationWithRMSProp() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new MinMaxNormalization(), new MinMaxNormalization(),
                new TensorFlow.Keras.Optimizers.RMSProp(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.1f),
                keras.losses.MeanSquaredError(), new INetworkLayer[]
                {
					new InputLayer(new int[1]{1}),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(1, ActivationType.Linear),
				},
				200));

        [Fact]
        public static void ZScoreNormalizationWithAdam() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new ZScoreNormalization(), new ZScoreNormalization(),
                new TensorFlow.Keras.Optimizers.Adam(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.05f),
                keras.losses.MeanSquaredError(), new INetworkLayer[]
                {
					new InputLayer(new int[1]{1}),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(1, ActivationType.Linear),
				},
				200));

        [Fact (Skip = "INeuralNetwork.EvaluateResponseGradients doesn't work for SGD optimizer")]
        public static void ZScoreNormalizationWithSGD() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new ZScoreNormalization(), new ZScoreNormalization(),
                new TensorFlow.Keras.Optimizers.SGD(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.05f),
                keras.losses.MeanSquaredError(), new INetworkLayer[]
                {
					new InputLayer(new int[1]{1}),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(1, ActivationType.Linear),
				},
				200));

        [Fact]
        public static void ZScoreNormalizationWithRMSProp() => TestFeedForwardNeuralNetwork(new FeedForwardNeuralNetwork(new ZScoreNormalization(), new ZScoreNormalization(),
                new TensorFlow.Keras.Optimizers.RMSProp(dataType: Tensorflow.TF_DataType.TF_DOUBLE, learning_rate: 0.1f),
				keras.losses.MeanSquaredError(), new INetworkLayer[]
                {
					new InputLayer(new int[1]{1}),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(50, ActivationType.SoftMax),
					new DenseLayer(1, ActivationType.Linear),
				},
				200));

        private static void TestFeedForwardNeuralNetwork(FeedForwardNeuralNetwork neuralNetwork)
        {
            neuralNetwork.Train(trainX, trainY);
            //var responses = neuralNetwork.EvaluateResponses(testX);
            var gradients = neuralNetwork.EvaluateResponseGradients(testX);
			//CheckResponseAccuracy(testY, responses);
			var loss = neuralNetwork.ValidateNetwork(testX,testY);
			Assert.True(loss < 0.03);
            CheckResponseGradientAccuracy(testGradient, gradients);
		}

        private static void CheckResponseAccuracy(double[,] data, double[,] prediction)
        {
            var deviation = new double[prediction.GetLength(0), prediction.GetLength(1)];
            var norm = 0d;
            for (int i = 0; i < prediction.GetLength(0); i++)
            {
                for (int j = 0; j < prediction.GetLength(1); j++)
                {
					deviation[i, j] = (data[i, j] - prediction[i, j]);
                    norm += Math.Pow(deviation[i, j], 2);
                }
            }
			norm = norm / prediction.GetLength(0);
			Assert.True(norm < 0.05, $"Response norm was above threshold (norm value: {norm})");
        }

        private static void CheckResponseGradientAccuracy(double[][,] dataGradient, double[][,] gradient)
        {
            var norm = new double[gradient.GetLength(0)];
            var normTotal = 0d;
            for (int k = 0; k < gradient.GetLength(0); k++)
            {
                var deviation = new double[gradient[k].GetLength(0), gradient[k].GetLength(1)];
                for (int i = 0; i < gradient[k].GetLength(0); i++)
                {
                    for (int j = 0; j < gradient[k].GetLength(1); j++)
                    {
						deviation[i, j] = (dataGradient[k][i, j] - gradient[k][i, j]);
                        norm[k] += Math.Pow(deviation[i, j], 2);
                    }
                }
                normTotal += norm[k];
            }
			normTotal = normTotal / gradient.GetLength(0);

			Assert.True(normTotal < 0.3, $"Gradient norm was above threshold (norm value: {norm})");
        }
    }
}
