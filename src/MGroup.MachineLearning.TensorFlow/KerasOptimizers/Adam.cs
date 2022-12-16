using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Keras.Utils;
using Tensorflow.Operations.Initializers;

namespace MGroup.MachineLearning.TensorFlow.Keras.Optimizers
{
    //
    // Summary:
    //     Optimizer that implements the Adam algorithm. Adam optimization is a stochastic
    //     gradient descent method that is based on adaptive estimation of first-order and
    //     second-order moments.
    public class Adam : OptimizerV2
    {
        private double epsilon = 1E-07f;
        private bool amsgrad;
        private TF_DataType dataType;

        protected override string _name => "Adam";

        public Adam(TF_DataType dataType = TF_DataType.TF_FLOAT, float learning_rate = 0.001f, float beta_1 = 0.9f, float beta_2 = 0.999f, float epsilon = 1E-07f, bool amsgrad = false, string name = "Adam")
            : base(new OptimizerV2Args())
        {
            this.dataType = dataType;
            _set_hyper("learning_rate", learning_rate);
            _set_hyper("beta_1", beta_1);
            _set_hyper("beta_2", beta_2);
            this.epsilon = epsilon;
            this.amsgrad = amsgrad;
        }

        protected override void _create_slots(IVariableV1[] var_list)
        {
            var dataTypeZeros = new Zeros(dtype: dataType);
            IVariableV1[] array = var_list;
            foreach (IVariableV1 var in array)
            {
                add_slot(var, "m", dataTypeZeros);
            }

            array = var_list;
            foreach (IVariableV1 var2 in array)
            {
                add_slot(var2, "v", dataTypeZeros);
            }

            if (amsgrad)
            {
                array = var_list;
                foreach (IVariableV1 var3 in array)
                {
                    add_slot(var3, "vhat", dataTypeZeros);
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

        protected override void _prepare_local(DeviceDType device_dtype, Dictionary<DeviceDType, Dictionary<string, Tensor>> apply_state)
        {
            if (dataType != device_dtype.DType)
            {
                throw new ArgumentException($"Device data type is not the same as Adam data type (device: {device_dtype.DType}, Adam: {dataType}). Initialize Adam with the proper data type ({device_dtype.DType}) at the constructor.", nameof(device_dtype));
            }

            double beta1 = (float)_get_hyper("beta_1", TF_DataType.TF_FLOAT);
            var beta1Variable = add_weight("beta_1", new int[0], device_dtype.DType, Binding.tf.constant_initializer(beta1, device_dtype.DType), trainable: false, VariableSynchronization.Auto, VariableAggregation.OnlyFirstReplica);
            double beta2 = (float)_get_hyper("beta_2", TF_DataType.TF_FLOAT);
            var beta2Variable = add_weight("beta_2", new int[0], device_dtype.DType, Binding.tf.constant_initializer(beta2, device_dtype.DType), trainable: false, VariableSynchronization.Auto, VariableAggregation.OnlyFirstReplica);

            my_prepare_local(device_dtype, apply_state);
            TF_DataType dType = device_dtype.DType;
            _ = device_dtype.Device;
            Tensor y = math_ops.cast(base.iterations + 1, dType);
            Tensor tensor = array_ops.identity(beta1Variable);
            Tensor tensor2 = array_ops.identity(beta2Variable);
            Tensor tensor3 = math_ops.pow(tensor, y);
            Tensor tensor4 = math_ops.pow(tensor2, y);
            Tensor value = apply_state[device_dtype]["lr_t"] * (math_ops.sqrt(1 - tensor4) / (1 - tensor3));
            apply_state[device_dtype]["lr"] = value;
            apply_state[device_dtype]["epsilon"] = ops.convert_to_tensor(epsilon, device_dtype.DType);
            apply_state[device_dtype]["beta_1_t"] = tensor;
            apply_state[device_dtype]["beta_1_power"] = tensor3;
            apply_state[device_dtype]["one_minus_beta_1_t"] = 1 - tensor;
            apply_state[device_dtype]["beta_2_t"] = tensor2;
            apply_state[device_dtype]["beta_2_power"] = tensor4;
            apply_state[device_dtype]["one_minus_beta_2_t"] = 1 - tensor2;
        }

        protected override Operation _resource_apply_dense(IVariableV1 var, Tensor grad, Dictionary<DeviceDType, Dictionary<string, Tensor>> apply_state)
        {
            string device = var.Device;
            TF_DataType tF_DataType = var.dtype.as_base_dtype();
            string var_device = device;
            TF_DataType var_dtype = tF_DataType;
            Dictionary<string, Tensor> dictionary = apply_state.FirstOrDefault<KeyValuePair<DeviceDType, Dictionary<string, Tensor>>>((KeyValuePair<DeviceDType, Dictionary<string, Tensor>> x) => x.Key.Device == var_device && x.Key.DType == var_dtype)!.Value ?? _fallback_apply_state(var_device, var_dtype);
            IVariableV1 variableV = get_slot(var, "m");
            IVariableV1 variableV2 = get_slot(var, "v");
            if (!amsgrad)
            {
                return gen_training_ops.resource_apply_adam(var.Handle, variableV.Handle, variableV2.Handle, dictionary["beta_1_power"], dictionary["beta_2_power"], dictionary["lr_t"], dictionary["beta_1_t"], dictionary["beta_2_t"], dictionary["epsilon"], grad, _use_locking);
            }

            throw new NotImplementedException("");
        }
    }
}
