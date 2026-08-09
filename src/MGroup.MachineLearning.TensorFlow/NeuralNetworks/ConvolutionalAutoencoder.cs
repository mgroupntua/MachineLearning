namespace MGroup.MachineLearning.TensorFlow.NeuralNetworks
{
	using System;
	using System.IO;
	using System.Linq;
	using System.Text;

	using MGroup.MachineLearning.Preprocessing;
	using MGroup.MachineLearning.TensorFlow.KerasLayers;
	using MGroup.MachineLearning.Utilities;

	using Tensorflow;
	using Tensorflow.Keras.Engine;
	using Tensorflow.Keras.Losses;
	using Tensorflow.Keras.Optimizers;
	using Tensorflow.NumPy;

	using static Tensorflow.Binding;
	using static Tensorflow.KerasApi;

	public class ConvolutionalAutoencoder //: INeuralNetwork
	{
		private Keras.Model autoencoderModel;
		private Keras.Model encoderModel;
		private Keras.Model decoderModel;
		private NDArray trainX, testX;
		private bool classification;

		public ConvolutionalAutoencoder(INormalization normalization, OptimizerV2 optimizer, ILossFunc lossFunc, 
			INetworkLayer[] encoderLayers, INetworkLayer[] decoderLayers, int epochs, int batchSize = -1, int? seed = 1, 
			bool classification = false, bool shuffleTrainingData = false)
		{
			BatchSize = batchSize;
			Epochs = epochs;
			Seed = seed;
			Normalization = normalization;
			Optimizer = optimizer;
			LossFunction = lossFunc;
			EncoderLayer = encoderLayers;
			DecoderLayer = decoderLayers;
			this.classification = classification;
			ShuffleTrainingData = shuffleTrainingData;
			if (seed != null)
			{
				tf.set_random_seed(seed.Value);
			}
		}

		/// <summary>
		/// This constructor can be used for objects that will load their properties from external files.
		/// </summary>
		public ConvolutionalAutoencoder()
		{
		}

		public int? Seed { get; }

		public bool ShuffleTrainingData { get; }

		public int BatchSize { get; }

		public int Epochs { get; }

		public INetworkLayer[] EncoderLayer { get; private set; }

		public INetworkLayer[] DecoderLayer { get; private set; }

		public INetworkLayer[] AutoencoderLayer { get; private set; }

		public INormalization Normalization { get; private set; }

		public OptimizerV2 Optimizer { get; }

		public ILossFunc LossFunction { get; }

		public Layer[] Layer { get; private set; }

		public void Train(double[,,,] stimuli) => Train(stimuli, null);

		public void Train(double[,,,] trainX, double[,,,] testX = null)
		{
			tf.enable_eager_execution();

			PrepareData(trainX, testX);

			CreateModel();

			autoencoderModel.compile(loss: LossFunction, optimizer: Optimizer, metrics: new[] { "accuracy" });

			autoencoderModel.fit(this.trainX, this.trainX, batch_size: BatchSize, epochs: Epochs, shuffle: ShuffleTrainingData);

			if (testX != null)
			{
				autoencoderModel.evaluate(this.testX, this.testX, batch_size: BatchSize);
			}
		}

		public double[,,,] EvaluateResponses(double[,,,] stimuli)
		{
			var npData = np.array(stimuli);
			var result = ((Tensor)autoencoderModel.Apply(npData, training: false)).numpy();
			var responses = new double[result.shape[0], result.shape[1], result.shape[2], result.shape[3]];
			for (int i = 0; i < result.shape[0]; i++)
			{
				for (int j = 0; j < result.shape[1]; j++)
				{
					for (int k = 0; k < result.shape[2]; k++)
					{
						for (int l = 0; l < result.shape[3]; l++)
						{
							responses[i, j, k, l] = result[i, j, k, l];
						}
					}
				}
			}
			return responses;
		}

		public double[,] MapFull4DToReduced2D(double[,,,] initialStimuli)
		{
			var npData = np.array(initialStimuli);
			Tensor prediction = encoderModel.Apply(npData, training: false);
			var result = ((Tensor)prediction).numpy();

			if (result.rank == 2)
			{
				Array array = result.ToMultiDimArray<double>();
				return (double[,])array;
			}
			else
			{
				throw new NotImplementedException();
			}
			
		}

		public double[,,,] MapFullToReduced(double[,,,] initialStimuli)
		{
			var npData = np.array(initialStimuli);
			Tensor prediction = encoderModel.Apply(npData, training: false);
			var result = ((Tensor)prediction).numpy();

			if (result.rank > 4)
			{
				throw new Exception("Applying the encoder model should not have returned a tensor with rank > 4.");
			}

			Array array = result.ToMultiDimArray<double>();
			if (result.rank == 4)
			{
				return (double[,,,])array;
			}
			else
			{
				var makeDimEmpty = new bool[4];
				int numEmptyDimensions = 0;
				for (int d = 0; d < 4; ++d)
				{
					if (initialStimuli.GetLength(d) == 1)
					{
						makeDimEmpty[d] = true;
						++numEmptyDimensions;
					}
				}
				if (numEmptyDimensions + result.rank != 4)
				{
					throw new Exception(
						$"Input array has rank ri={initialStimuli.Rank} and e={numEmptyDimensions} empty dimensions, while" +
						$"output TensorFlow tensor has rank ro={result.rank}. It must hold ro+e=ri, but it did not.");
				}


				if (result.rank == 3)
				{
					var array3D = (double[,,])array;
					return array3D.AddEmptyDimensions(makeDimEmpty[0], makeDimEmpty[1], makeDimEmpty[2], makeDimEmpty[3]);
				}
				else if (result.rank == 2)
				{
					var array2D = (double[,])array;
					return array2D.AddEmptyDimensions(makeDimEmpty[0], makeDimEmpty[1], makeDimEmpty[2], makeDimEmpty[3]);
				}
				else // result.rank == 1
				{
					var array1D = (double[])array;
					return array1D.AddEmptyDimensions(makeDimEmpty[0], makeDimEmpty[1], makeDimEmpty[2], makeDimEmpty[3]);
				}
			}

			//var responses = new double[result.shape[0], result.shape[1], result.shape[2], result.shape[3]];
			//for (int i = 0; i < result.shape[0]; i++)
			//{
			//	for (int j = 0; j < result.shape[1]; j++)
			//	{
			//		for (int k = 0; k < result.shape[2]; k++)
			//		{
			//			for (int l = 0; l < result.shape[3]; l++)
			//			{
			//				responses[i, j, k, l] = result[i, j, k, l];
			//			}
			//		}
			//	}
			//}
			//return responses;
		}

		public double[,,,] MapReducedToFull(double[,,,] reducedStimuli)
		{
			var npData = np.array(reducedStimuli);
			var result = ((Tensor)decoderModel.Apply(npData, training: false)).numpy();
			var responses = new double[result.shape[0], result.shape[1], result.shape[2], result.shape[3]];
			for (int i = 0; i < result.shape[0]; i++)
			{
				for (int j = 0; j < result.shape[1]; j++)
				{
					for (int k = 0; k < result.shape[2]; k++)
					{
						for (int l = 0; l < result.shape[3]; l++)
						{
							responses[i, j, k, l] = result[i, j, k, l];
						}
					}
				}
			}
			return responses;
		}

		public double[,,,] MapReduced2DToFull4D(double[,] reducedStimuli)
		{
			var npData = np.array(reducedStimuli);
			var result = ((Tensor)decoderModel.Apply(npData, training: false)).numpy();
			var responses = new double[result.shape[0], result.shape[1], result.shape[2], result.shape[3]];
			for (int i = 0; i < result.shape[0]; i++)
			{
				for (int j = 0; j < result.shape[1]; j++)
				{
					for (int k = 0; k < result.shape[2]; k++)
					{
						for (int l = 0; l < result.shape[3]; l++)
						{
							responses[i, j, k, l] = result[i, j, k, l];
						}
					}
				}
			}
			return responses;
		}

		//public double[][,] EvaluateResponseGradients(double[,] stimuli)
		//{
		//	stimuli = NormalizationX.Normalize(stimuli);

		//	var responseGradients = new double[stimuli.GetLength(0)][,];
		//	for (int k = 0; k < stimuli.GetLength(0); k++)
		//	{
		//		var sample = new double[1, stimuli.GetLength(1)];
		//		for (int i = 0; i < stimuli.GetLength(1); i++)
		//		{
		//			sample[0, i] = stimuli[k, i];
		//		}

		//		var ratioX = NormalizationX.ScalingRatio;

		//		var npSample = np.array(sample);
		//		using var tape = tf.GradientTape(persistent: true);
		//		{
		//			tape.watch(npSample);
		//			Tensor pred = model.Apply(npSample, training: false);
		//			var ratioY = NormalizationY.ScalingRatio;

		//			var numRowsGrad = pred.shape.dims[1];
		//			var numColsGrad = npSample.GetShape().as_int_list()[1];
		//			var slicedPred = new Tensor();
		//			responseGradients[k] = new double[numRowsGrad, numColsGrad];
		//			for (int i = 0; i < numRowsGrad; i++)
		//			{
		//				slicedPred = tf.slice<int, int>(pred, new int[] { 0, i }, new int[] { 1, 1 });
		//				var slicedGrad = tape.gradient(slicedPred, npSample).ToArray<double>();
		//				for (int j = 0; j < numColsGrad; j++)
		//				{
		//					responseGradients[k][i, j] = ratioY[i] / ratioX[j] * slicedGrad[j];
		//				}
		//			}
		//		}
		//	}

		//	return responseGradients;
		//}

		public double ValidateNetwork(double[,,,] testX)
		{
			var predX = EvaluateResponses(testX);
			var predXnp = np.array(predX);
			var testXnp = np.array(testX);
			var accuracy = new Tensor(0);
			if (classification == false)
			{
				accuracy = LossFunction.Call(testXnp, predXnp);
			}
			else
			{
				var correct_prediction = tf.equal(tf.math.argmax(predXnp, 1), tf.cast(tf.squeeze(testXnp), tf.int64));
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
			}
			return accuracy.ToArray<double>()[0];
		}

		public void SaveNetwork(string netPath, string weightsPath, string normalizationPath)
		{
			autoencoderModel.save_weights(weightsPath);

			using (Stream stream = File.Open(normalizationPath, false ? FileMode.Append : FileMode.Create))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				binaryFormatter.Serialize(stream, new INormalization[] { Normalization });
			}

			using (Stream stream = File.Open(netPath, false ? FileMode.Append : FileMode.Create))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				binaryFormatter.Serialize(stream, AutoencoderLayer);
			}
		}

		public void LoadNetwork(string netPath, string weightsPath, string normalizationPath)
		{
			using (Stream stream = File.Open(normalizationPath, FileMode.Open, FileAccess.Read, FileShare.Read))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				var normalization = (INormalization[])binaryFormatter.Deserialize(stream);
				Normalization = normalization[0];
			}

			using (Stream stream = File.Open(netPath, FileMode.Open, FileAccess.Read, FileShare.Read))
			{
				var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				AutoencoderLayer = (INetworkLayer[])binaryFormatter.Deserialize(stream);
			}

			CreateModel();

			autoencoderModel.load_weights(weightsPath);
		}

		private void PrepareData(double[,,,] trainX, double[,,,] testX = null)
		{
			this.trainX = np.array(trainX);
			if (testX != null)
			{
				this.testX = np.array(testX);
			}
		}

		private void CreateModel()
		{
			#region DEBUG
			//Delete all lines referencing "text" variable, when cleaning this
			var text = new System.Text.StringBuilder();
			text.AppendLine("CAE layers and tensors:");
			text.AppendLine("Encoder:");
			#endregion

			keras.backend.clear_session();
			keras.backend.set_floatx(TF_DataType.TF_DOUBLE);
			if (!(EncoderLayer[0] is KerasLayers.InputLayer))
			{
				throw new NotImplementedException($"First layer must be of type IInputLayer");
			}

			//var inputs = keras.Input(shape: (this.trainX.shape[1], this.trainX.shape[2], this.trainX.shape[3]), dtype: TF_DataType.TF_DOUBLE);
			AutoencoderLayer = new INetworkLayer[EncoderLayer.Length + DecoderLayer.Length];
			AutoencoderLayer[0] = EncoderLayer[0];

			var inputsEncoder = keras.Input(shape: ((KerasLayers.InputLayer)EncoderLayer[0]).InputShape, dtype: TF_DataType.TF_DOUBLE); //.as_int_list()[0]
			var outputsEncoder = inputsEncoder;
			LogTensor(outputsEncoder, text);
			for (int i = 1; i < EncoderLayer.Length; i++)
			{
				outputsEncoder = EncoderLayer[i].BuildLayer(outputsEncoder);
				LogTensor(outputsEncoder, text);
				AutoencoderLayer[i] = EncoderLayer[i];
			}

			encoderModel = new Keras.Model(inputsEncoder, outputsEncoder, "encoder");

			encoderModel.summary();

			var inputsDecoder = keras.Input(shape: encoderModel.Layers[encoderModel.Layers.Count - 1].output_shape.as_int_list().Skip(1).Take(3).ToArray(), dtype: TF_DataType.TF_DOUBLE); //.as_int_list()[0]
			var outputsDecoder = inputsDecoder;
			text.AppendLine("Decoder:");
			LogTensor(outputsDecoder, text);
			for (int i = 0; i < DecoderLayer.Length; i++)
			{
				outputsDecoder = DecoderLayer[i].BuildLayer(outputsDecoder);
				LogTensor(outputsDecoder, text);
				AutoencoderLayer[EncoderLayer.Length + i] = DecoderLayer[i];
			}
			System.Diagnostics.Debug.WriteLine(text.ToString());

			decoderModel = new Keras.Model(inputsDecoder, outputsDecoder, "decoder");

			decoderModel.summary();

			//var shape = ((KerasLayers.InputLayer)EncoderLayer[0]).InputShape;
			//var shape1 = new int[4];
			//shape1[0] = 1;
			//shape1[1] = shape[0];
			//shape1[2] = shape[1];
			//shape1[3] = shape[2];
			//var inputsAutoencoder = keras.Input(shape: shape1, dtype: TF_DataType.TF_DOUBLE);
			keras.backend.clear_session();
			var inputsAutoencoder = keras.Input(shape: ((KerasLayers.InputLayer)EncoderLayer[0]).InputShape, dtype: TF_DataType.TF_DOUBLE);
			var outputsAutoencoder = encoderModel.Apply(inputsAutoencoder);
			outputsAutoencoder = decoderModel.Apply(outputsAutoencoder);

			autoencoderModel = new Keras.Model(inputsAutoencoder, outputsAutoencoder, "autoencoder");

			autoencoderModel.summary();
		}

		#region DEBUG
		private static void LogTensor(Tensor tensor, StringBuilder text)
		{
			text.Append($"Layer: name={tensor.KerasHistory.Layer.Name}, shape={tensor.KerasHistory.Layer.output_shape}");
			text.AppendLine($". Tensor: name={tensor.name}, shape={tensor.shape}");
		}
		#endregion
	}
}
