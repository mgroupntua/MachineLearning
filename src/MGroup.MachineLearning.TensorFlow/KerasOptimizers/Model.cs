using System.Linq;
using System.Reflection;
using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;

namespace MGroup.MachineLearning.TensorFlow.Keras
{
    public class Model : Functional
    {
        public Model(Tensors inputs, Tensors outputs, string name = null) : base(inputs, outputs, name)
        {
            var i = inputs.dtype;
            var fieldInfos = typeof(Layer).GetFields(BindingFlags.Instance | BindingFlags.NonPublic).Where(x => x.Name == "args");

            foreach (var fieldInfo in fieldInfos)
            {
                var value = fieldInfo.GetValue(this) as ModelArgs;
                if (value == null) continue;
                value.DType = i;
                fieldInfo.SetValue(this, value);
            }
        }
    }
}
