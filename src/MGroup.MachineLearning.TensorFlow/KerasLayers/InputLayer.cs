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
	public class InputLayer : INetworkLayer
	{
		public int[] InputShape { get; }
		public string Name { get; }
		public bool Sparse { get; }
		public bool Ragged { get; }

		public InputLayer(int[] inputShape, string name = null, bool sparse = false, bool ragged = false)
		{
			InputShape = inputShape;
			Name = name;
			Sparse = sparse;
			Ragged = ragged;
		}

		public Tensors BuildLayer(Tensors output) => new Tensorflow.Keras.Layers.InputLayer(new InputLayerArgs()
		{
			InputShape = InputShape,
			Name = Name,
			Sparse = Sparse,
			Ragged = Ragged,
			DType = TF_DataType.TF_DOUBLE
		}).Apply(output);

	}
}
