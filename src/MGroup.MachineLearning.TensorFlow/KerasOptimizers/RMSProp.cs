using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Optimizers;
using Tensorflow;
using Tensorflow.Operations.Initializers;
using Tensorflow.Keras.Utils;

namespace MGroup.MachineLearning.TensorFlow.Keras.Optimizers
{
	public class RMSProp : OptimizerV2
	{
		private double epsilon = 1E-07f;
		private bool centered;
		private RMSpropArgs args;
		private TF_DataType dataType;

		protected override string _name => "RMSprop";

		public RMSProp(TF_DataType dataType = TF_DataType.TF_FLOAT, float learning_rate = 0.001f, float rho = 0.9f, float momentum = 0.0f, float epsilon = 1E-07f, bool centered = false, string name = "RMSprop")
			: base(new OptimizerV2Args())
		{
			this.args = args;
			this.dataType = dataType;
			_set_hyper("learning_rate", learning_rate);
			_set_hyper("rho", rho);
			_set_hyper("momentum", momentum);
			this.epsilon = epsilon;
			this.centered = centered;
		}

		protected override void _create_slots(IVariableV1[] var_list)
		{
			var dataTypeZeros = new Zeros(dtype: dataType);
			IVariableV1[] array = var_list;
			foreach (IVariableV1 var in array)
			{
				add_slot(var, "rms", dataTypeZeros);
			}

			if (_momentum)
			{
				array = var_list;
				foreach (IVariableV1 var2 in array)
				{
					add_slot(var2, "momentum", dataTypeZeros);
				}
			}

			if (centered)
			{
				array = var_list;
				foreach (IVariableV1 var3 in array)
				{
					add_slot(var3, "mg", dataTypeZeros);
				}
			}
		}

		private ResourceVariable add_weight(string name, Shape shape, TF_DataType dtype = TF_DataType.TF_FLOAT, IInitializer initializer = null, bool trainable = false, VariableSynchronization synchronization = VariableSynchronization.Auto, VariableAggregation aggregation = VariableAggregation.None)
		{
			if (initializer == null)
			{
				initializer = Binding.tf.zeros_initializer;
			}

			if (dtype == TF_DataType.DtInvalid)
			{
				dtype = TF_DataType.TF_FLOAT;
			}

			return _add_variable_with_custom_getter(new VariableArgs
			{
				Name = name,
				Shape = shape,
				Getter = new Func<VariableArgs, IVariableV1>(base_layer_utils.make_variable),
				DType = dtype,
				Overwrite = true,
				Initializer = initializer,
				Trainable = trainable,
				UseResource = true,
				Synchronization = synchronization,
				Aggregation = aggregation
			}) as ResourceVariable;
		}

		private void my_prepare_local(DeviceDType device_dtype, Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
		{
			if (_initial_decay > 0f)
			{
				throw new NotImplementedException("Initial decay is more than zero.");
			}

			if (dataType != device_dtype.DType)
			{
				throw new ArgumentException($"Device data type is not the same as Adam data type (device: {device_dtype.DType}, Adam: {dataType}). Initialize Adam with the proper data type ({device_dtype.DType}) at the constructor.", nameof(device_dtype));
			}

			double lr = (float)_get_hyper("learning_rate", TF_DataType.TF_FLOAT);
			var lrVariable = add_weight("learning_rate", new int[0], device_dtype.DType, Binding.tf.constant_initializer(lr, device_dtype.DType), trainable: false, VariableSynchronization.Auto, VariableAggregation.OnlyFirstReplica);
			Tensor value = array_ops.identity(lrVariable);
			_apply_state[device_dtype]["lr_t"] = value;
		}

		protected override void _prepare_local(DeviceDType device_dtype, Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
		{
			if (dataType != device_dtype.DType)
			{
				throw new ArgumentException($"Device data type is not the same as Adam data type (device: {device_dtype.DType}, Adam: {dataType}). Initialize Adam with the proper data type ({device_dtype.DType}) at the constructor.", nameof(device_dtype));
			}

			double rho = (float)_get_hyper("rho", TF_DataType.TF_FLOAT);
			var rhoVariable = add_weight("rho", new int[0], device_dtype.DType, Binding.tf.constant_initializer(rho, device_dtype.DType), trainable: false, VariableSynchronization.Auto, VariableAggregation.OnlyFirstReplica);
			double momentum = (float)_get_hyper("momentum", TF_DataType.TF_FLOAT);
			var momentumVariable = add_weight("momentum", new int[0], device_dtype.DType, Binding.tf.constant_initializer(momentum, device_dtype.DType), trainable: false, VariableSynchronization.Auto, VariableAggregation.OnlyFirstReplica);

			my_prepare_local(device_dtype, _apply_state);
			Tensor tensor = array_ops.identity(rhoVariable);
			_apply_state[device_dtype]["neg_lr_t"] = -_apply_state[device_dtype]["lr_t"];
			_apply_state[device_dtype]["epsilon"] = ops.convert_to_tensor(epsilon, device_dtype.DType);
			_apply_state[device_dtype]["rho"] = tensor;
			_apply_state[device_dtype]["momentum"] = array_ops.identity(momentumVariable);
			_apply_state[device_dtype]["one_minus_rho"] = 1f - tensor;
		}

		protected override Operation _resource_apply_dense(IVariableV1 var, Tensor grad, Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
		{
			Dictionary<string, Tensor> dictionary = null;
			foreach (KeyValuePair<DeviceDType, Dictionary<string, Tensor>> item in _apply_state)
			{
				if (item.Key.DType == var.dtype.as_base_dtype() && item.Key.Device == var.Device)
				{
					dictionary = item.Value;
					break;
				}
			}

			IVariableV1 variableV = get_slot(var, "rms");
			if (_momentum)
			{
				throw new NotImplementedException("");
			}

			Tensor value = dictionary["rho"] * variableV.AsTensor() + dictionary["one_minus_rho"] * math_ops.square(grad);
			value = state_ops.assign(variableV, value, validate_shape: true, _use_locking);
			Tensor x = value;
			if (centered)
			{
				throw new NotImplementedException("");
			}

			Tensor value2 = var.AsTensor() - dictionary["lr_t"] * grad / (math_ops.sqrt(x) + dictionary["epsilon"]);
			return state_ops.assign(var, value2, validate_shape: true, _use_locking).op;
		}
	}
}
