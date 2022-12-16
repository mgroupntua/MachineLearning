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
	public class MaxPooling2DLayer : INetworkLayer
	{
		public (int, int)? PoolSize { get; }
		public (int, int)? Strides { get; }

		public MaxPooling2DLayer((int, int)? poolSize = null, (int, int)? strides = null)
		{
			PoolSize = poolSize ?? (2, 2);
			Strides = strides;
		}

		public Tensors BuildLayer(Tensors output) => new MaxPooling2D(new MaxPooling2DArgs()
		{
			PoolSize = this.PoolSize,
			Strides = this.Strides,
			DType = TF_DataType.TF_DOUBLE
		}).Apply(output);

	}
}
