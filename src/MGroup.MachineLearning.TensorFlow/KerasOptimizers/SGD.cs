using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Optimizers;
using Tensorflow;
using Tensorflow.Keras.Utils;

namespace MGroup.MachineLearning.TensorFlow.Keras.Optimizers
{
    public class SGD : OptimizerV2
    {
        private bool nesterov;
        private TF_DataType dataType;

        protected override string _name => "SGD";

        public SGD(TF_DataType dataType = TF_DataType.TF_FLOAT, float learning_rate = 0.001f, float momentum = 0f, bool nesterov = false, float decay = 0f)
            : base(new OptimizerV2Args())
        {
            this.dataType = dataType;
            _set_hyper("learning_rate", learning_rate);
            _set_hyper("decay", decay);
            _momentum = momentum > 0f;
            _set_hyper("momentum", momentum);
            this.nesterov = nesterov;
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

            double momentum = (float)_get_hyper("momentum", TF_DataType.TF_FLOAT);
            var momentumVariable = add_weight("momentum", new int[0], device_dtype.DType, Binding.tf.constant_initializer(momentum, device_dtype.DType), trainable: false, VariableSynchronization.Auto, VariableAggregation.OnlyFirstReplica);

            my_prepare_local(device_dtype, _apply_state);
            _apply_state[device_dtype]["momentum"] = array_ops.identity(momentumVariable);
        }

        protected override Operation _resource_apply_dense(IVariableV1 var, Tensor grad, Dictionary<DeviceDType, Dictionary<string, Tensor>> _apply_state)
        {
            IVariableV1 var2 = var;
            if (_momentum)
            {
                throw new NotImplementedException("_resource_apply_dense");
            }

            DeviceDType key = _apply_state.Keys.FirstOrDefault((DeviceDType x) => x.Device == var2.Device && x.DType == var2.dtype.as_base_dtype());
            return gen_training_ops.resource_apply_gradient_descent(var2.Handle, _apply_state[key]["lr_t"], grad, _use_locking);
        }
    }
}
