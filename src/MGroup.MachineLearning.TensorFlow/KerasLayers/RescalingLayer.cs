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
	public class RescalingLayer : INetworkLayer
	{
		public double Scale { get; }
		public double Offset { get; }
		public (int, int, int)? InputShape { get; }

		public RescalingLayer(double scale, double offset = 0, (int, int, int)? inputShape = null)
		{
			Scale = scale;
			Offset = offset;
			InputShape = inputShape;
		}

		public Tensors BuildLayer(Tensors output) => new Rescaling(new RescalingArgs()
		{
			Scale = (float)this.Scale,
			Offset = (float)this.Offset,
			DType = TF_DataType.TF_DOUBLE,
			//InputShape = this.InputShape,
		}).Apply(output);

	}
}
