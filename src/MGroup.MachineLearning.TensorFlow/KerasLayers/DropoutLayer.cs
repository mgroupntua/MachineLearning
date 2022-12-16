using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras;
using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using System.Xml.Linq;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Utils;

namespace MGroup.MachineLearning.TensorFlow.KerasLayers
{
	[Serializable]
	public class DropoutLayer : INetworkLayer
	{
		public double Rate { get; }
		public (int, int, int)? NoiseShape { get; }
		public int? Seed { get; }

		public DropoutLayer(double rate, (int, int, int)? noiseShape = null, int? seed = null)
		{
			Rate = rate;
			NoiseShape = noiseShape;
			Seed = seed;
		}

		public Tensors BuildLayer(Tensors output) => new Dropout(new DropoutArgs()
		{
			Rate = (float)this.Rate,
			NoiseShape = NoiseShape,
			Seed = Seed,
			DType = TF_DataType.TF_DOUBLE,
		}).Apply(output);

		public class Dropout : Layer
		{
			private DropoutArgs args;

			public Dropout(DropoutArgs args)
				: base(args)
			{
				this.args = args;
			}

			protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
			{
				Tensors inputs2 = inputs;
				if (!training.HasValue)
				{
					training = false;
				}

				return tf_utils.smart_cond(training.Value, () => false_fn(inputs2), () => array_ops.identity(inputs2));
			}

			private Tensor get_noise_shape(Tensor inputs)
			{
				_ = args.NoiseShape == null;
				return null;
			}

			private static Tensor _get_noise_shape(Tensor x, Tensor noise_shape)
			{
				if (noise_shape == null)
				{
					return array_ops.shape(x);
				}

				return noise_shape;
			}

			private Tensor false_fn(Tensors inputs)
			{
				Tensor rate2 = Binding.tf.constant(args.Rate, dtype: inputs.dtype);

				string name2 = name;
				Tensor x2 = inputs;
				Tensor noise_shape2 = get_noise_shape(x2);

				return Binding.tf_with(ops.name_scope(name2, "dropout", x2), delegate (ops.NameScope scope)
				{
					name2 = (string)scope;
					x2 = ops.convert_to_tensor(x2, TF_DataType.DtInvalid, "x");
					if (!x2.dtype.is_floating())
					{
						throw new NotImplementedException("x has to be a floating point tensor since it's going to" + $" be scaled. Got a {x2.dtype} tensor instead.");
					}

					Tensor tensor = 1 - rate2;
					Tensor tensor2 = 1 / tensor;
					Tensor y = ops.convert_to_tensor(tensor2, x2.dtype);
					Tensor tensor3 = gen_math_ops.mul(x2, y);
					noise_shape2 = _get_noise_shape(x2, noise_shape2);
					Tensor x3 = random_ops.random_uniform(noise_shape2, 0, null, seed: args.Seed, dtype: x2.dtype) >= rate2;
					tensor3 = x2 * tensor2 * math_ops.cast(x3, x2.dtype);
					if (!Binding.tf.executing_eagerly())
					{
						tensor3.shape = x2.shape;
					}
					return tensor3;
				});
			}
		}
	}
}
