#include "neuron.h"

double Neuron::eta = 0.15; // overall net learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]
double Neuron::leaky = 0;


void Neuron::updateInputWeights(Layer &prevLayer)
{
	// The weights to be updated are in the Connection container
	// in the nuerons in the preceding layer

	for(unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		
		double newDeltaWeight = 
				// Individual input, magnified by the gradient and train rate:
                // 0.0 slow learner
                // 0.2 medium learner
                // 1.0 reckless learner
				eta
				* neuron.getOutputVal()
				* m_gradient
				// Also add momentum = a fraction of the previous delta weight
                // 0.0 no momentum
                // 0.5 moderate momentum
				+ alpha
				* oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
		// if(n<=5){cout<<	"newDeltaWeight = "<<newDeltaWeight<<endl;	}
	}
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVals)
{
	double delta = targetVals - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
	// tanh - output range [-1.0..1.0]
	// return tanh(x);
	// return x>=0? x : leaky * x;
	return (1 / (1 + exp(-1 * x)));

    /// \frac{d}{dx} tanh x =1 - tanh^2x
}

double Neuron::transferFunctionDerivative(double x)
{
	// tanh derivative
	// return 1.0 - x * x;
	// return x>=0 ? 1 : leaky;
	return (1 / (1 + exp(-1 * x))) * (1-(1 / (1 + exp(-1 * x))));
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

	for(unsigned n = 0 ; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() * 
				 prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    // c表示connections, numOutputs为输出的数量（即下一层的长度）
	for(unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());

        // 初始化权重填入随机值
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}