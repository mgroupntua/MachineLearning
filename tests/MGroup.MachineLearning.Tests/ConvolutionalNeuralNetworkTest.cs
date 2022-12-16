using System;
using Xunit;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;
using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
using MGroup.MachineLearning.Preprocessing;
using MGroup.MachineLearning.TensorFlow.KerasLayers;
using Tensorflow;

namespace MGroup.MachineLearning.Tests
{
	public class ConvolutionalNeuralNetworkTest
	{

		[Fact]
		public static void ConvolutionalNeuralNetworkWithAdam() => TestConvolutionalNeuralNetwork(new ConvolutionalNeuralNetwork(new NullNormalization(), new NullNormalization(),
				new TensorFlow.Keras.Optimizers.Adam(dataType: Tensorflow.TF_DataType.TF_DOUBLE),
				keras.losses.SparseCategoricalCrossentropy(from_logits: true), new INetworkLayer[]
				{
							new InputLayer(new int[]{28, 28, 1}),
							//new RescalingLayer(scale: 1/255),
							new Convolutional2DLayer(filters: 32, kernelSize: (5, 5), ActivationType.RelU),
							new MaxPooling2DLayer(strides: (2,2)),
							new Convolutional2DLayer(filters: 64, kernelSize: (3, 3), ActivationType.RelU),
							new MaxPooling2DLayer(strides: (2,2)),
							new FlattenLayer(),
							new DenseLayer(1024, ActivationType.RelU),
							//new DropoutLayer(0.5, seed: 1),
							new DenseLayer(10, ActivationType.SoftMax),
				},
				batchSize: 32, epochs: 50, seed: 1, classification: true));

		private static void TestConvolutionalNeuralNetwork(ConvolutionalNeuralNetwork neuralNetwork)
		{
			(double[,,,] trainX, double[,] trainY, double[,,,] testX, double[,] testY) = PrepareData();
			neuralNetwork.Train(trainX, trainY);
			var accuracy = neuralNetwork.ValidateNetwork(testX, testY);
			Assert.True(accuracy > 0.75);
			//CheckResponseGradientAccuracy(testGradient, gradients);
		}

		//private static void CheckResponseGradientAccuracy(double[][,] dataGradient, double[][,] gradient)
		//{
		//	var norm = new double[gradient.GetLength(0)];
		//	var normTotal = 0d;
		//	for (int k = 0; k < gradient.GetLength(0); k++)
		//	{
		//		var deviation = new double[gradient[k].GetLength(0), gradient[k].GetLength(1)];
		//		for (int i = 0; i < gradient[k].GetLength(0); i++)
		//		{
		//			for (int j = 0; j < gradient[k].GetLength(1); j++)
		//			{
		//				deviation[i, j] = (dataGradient[k][i, j] - gradient[k][i, j]);
		//				norm[k] += Math.Pow(deviation[i, j], 2);
		//			}
		//		}
		//		normTotal += norm[k];
		//	}
		//	normTotal = normTotal / gradient.GetLength(0);

		//	Assert.True(normTotal < 0.3, $"Gradient norm was above threshold (norm value: {norm})");
		//}

		public static (double[,,,] trainX, double[,] trainY, double[,,,] testX, double[,] testY) PrepareData()
		{
			((var trainXnd, var trainYnd), (var testXnd, var testYnd)) = keras.datasets.mnist.load_data();
			trainXnd = trainXnd["::600"];
			trainYnd = trainYnd["::600"];
			testXnd = testXnd["::100"];
			testYnd = testYnd["::100"];
			//trainYnd = np_utils.to_categorical(trainYnd, 10);
			//testYnd = np_utils.to_categorical(testYnd, 10);
			trainXnd = tf.expand_dims(trainXnd).numpy();
			trainYnd = tf.expand_dims(trainYnd).numpy();
			testXnd = tf.expand_dims(testXnd).numpy();
			testYnd = tf.expand_dims(testYnd).numpy();
			trainXnd = tf.convert_to_tensor(trainXnd, dtype: TF_DataType.TF_DOUBLE).numpy();
			trainYnd = tf.convert_to_tensor(trainYnd, dtype: TF_DataType.TF_DOUBLE).numpy();
			testXnd = tf.convert_to_tensor(testXnd, dtype: TF_DataType.TF_DOUBLE).numpy();
			testYnd = tf.convert_to_tensor(testYnd, dtype: TF_DataType.TF_DOUBLE).numpy();

			(trainXnd, testXnd) = (trainXnd / 255.0f, testXnd / 255.0f);
			double[,,,] trainX = new double[trainXnd.GetShape().as_int_list()[0], trainXnd.GetShape().as_int_list()[1], trainXnd.GetShape().as_int_list()[2], trainXnd.GetShape().as_int_list()[3]];
			double[,] trainY = new double[trainYnd.GetShape().as_int_list()[0], trainYnd.GetShape().as_int_list()[1]];
			double[,,,] testX = new double[testXnd.GetShape().as_int_list()[0], testXnd.GetShape().as_int_list()[1], testXnd.GetShape().as_int_list()[2], trainXnd.GetShape().as_int_list()[3]];
			double[,] testY = new double[testYnd.GetShape().as_int_list()[0], testYnd.GetShape().as_int_list()[1]];
			for (int k = 0; k < trainX.GetLength(2); k++)
			{
				for (int j = 0; j < trainX.GetLength(1); j++)
				{
					for (int i = 0; i < trainX.GetLength(0); i++)
					{
						trainX[i, j, k, 0] = trainXnd[i, j, k, 0];
					}
					for (int i = 0; i < testX.GetLength(0); i++)
					{
						testX[i, j, k, 0] = testXnd[i, j, k, 0];
					}
				}
			}
			for (int i = 0; i < trainY.GetLength(0); i++)
			{
				for (int j = 0; j < trainY.GetLength(1); j++)
				{
					trainY[i, j] = trainYnd[i, j];
				}
			}
			for (int i = 0; i < testY.GetLength(0); i++)
			{
				for (int j = 0; j < testY.GetLength(1); j++)
				{
					testY[i, j] = testYnd[i, j];
				}
			}

			return (trainX, trainY, testX, testY);
		}
	}
}
