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
	public class FlattenLayer : INetworkLayer
	{

		public FlattenLayer()
		{
		}

		public Tensors BuildLayer(Tensors output) => new Flatten(new FlattenArgs()
		{
			DType = TF_DataType.TF_DOUBLE
		}).Apply(output);

	}
}
