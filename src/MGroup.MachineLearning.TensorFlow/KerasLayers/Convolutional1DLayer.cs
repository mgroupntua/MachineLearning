using System;
using System.Collections.Generic;
using System.Text;

using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;

namespace MGroup.MachineLearning.TensorFlow.KerasLayers
{
	[Serializable]
	public class Convolutional1DLayer : INetworkLayer
	{
		public int Filters { get; }
		public int KernelSize { get; }
		public ActivationType ActivationType { get; }
		public int Rank { get; }
		public int DilationRate { get; }
		public int Strides { get; }

		public Convolutional1DLayer(int filters, int kernelSize, ActivationType activationType, int rank = 1, int dilationRate = 1, int strides = 1)
		{
			Filters = filters;
			KernelSize = kernelSize;
			ActivationType = activationType;
			Rank = rank;
			DilationRate = dilationRate;
			Strides = strides;
		}

		public Tensors BuildLayer(Tensors inputs) => new Conv1D(new Conv1DArgs()
		{
			Rank = this.Rank,
			Filters = this.Filters,
			KernelSize = this.KernelSize,
			Activation = GetActivationByName(ActivationType),
			DilationRate = this.DilationRate,
			Strides = this.Strides,
			DType = TF_DataType.TF_DOUBLE,
		}).Apply(inputs);

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
