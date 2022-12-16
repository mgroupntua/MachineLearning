using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras;
using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;

namespace MGroup.MachineLearning.TensorFlow.KerasLayers
{
	[Serializable]
	public class DenseLayer : INetworkLayer
	{
		public int Neurons { get; }
		public ActivationType ActivationType { get; }

		public DenseLayer(int neurons, ActivationType activationType)
		{
			Neurons = neurons;
			ActivationType = activationType;
		}

		public Tensors BuildLayer(Tensors output) => new Dense(new DenseArgs()
		{
			Units = Neurons,
			Activation = GetActivationByName(ActivationType),
			DType = TF_DataType.TF_DOUBLE
		}).Apply(output);

		private Activation GetActivationByName(ActivationType activation)
		{
			return activation switch
			{
				ActivationType.Linear => KerasApi.keras.activations.Linear,
				ActivationType.RelU => KerasApi.keras.activations.Relu,
				ActivationType.Sigmoid => KerasApi.keras.activations.Sigmoid,
				ActivationType.TanH => KerasApi.keras.activations.Tanh,
				ActivationType.SoftMax => KerasApi.keras.activations.Softmax,
				_ => throw new Exception($"Activation '{activation}' not found"),
			};
		}
	}
}
