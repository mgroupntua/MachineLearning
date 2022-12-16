using System;
using System.Collections.Generic;
using System.Text;

using MGroup.MachineLearning.TensorFlow.KerasLayers;

namespace MGroup.MachineLearning
{
	// for each double[] => rows: input quantity, columns: input dimension
	public interface INeuralNetwork
	{
        INetworkLayer[] NeuralNetworkLayer { get; }

        void Train(double[,] stimuli, double[,] responses);
        double[,] EvaluateResponses(double[,] data);
        double[][,] EvaluateResponseGradients(double[,] stimuli);

        void LoadNetwork(string netPath, string weightsPath, string normalizationPath);
        void SaveNetwork(string netPath, string weightsPath, string normalizationPath);
    }
}
