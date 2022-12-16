using System;

using Tensorflow;
using Tensorflow.Keras;

namespace MGroup.MachineLearning.TensorFlow.KerasLayers
{

	public enum ActivationType
	{
		NotSet,
		Linear,
		RelU,
		Sigmoid,
		TanH,
		SoftMax
	}

	public interface INetworkLayer
	{
		Tensors BuildLayer(Tensors output);
	}
}
