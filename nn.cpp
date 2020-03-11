#include "nn.h"

using namespace std;

double NN::m_recentAverageSmoothingFactor = 100.0; 

NN::NN(string trainF, string testF, string predictOutF){
    // cout<<"constrution---"<<endl;
    trainFile = trainF;
    testFile = testF;
    predictOutFile = predictOutF;
    featuresNum = 0;
    vector<int> shape = {7,1};
    init(shape);
}

void NN::showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << " ";
}

void NN::train(){

    int i,j;
    vector<double> resultVals, targetVals;

    for(i = 0; i < maxIterTimes; i++){
        // targetVals.clear();
        for (j = 0; j < trainDataSet.size(); j++) {
        // for (j = 0; j < 2000; j++) {    
            // cout<<"开始处理 第"<<j+1<<" 组数据"<<endl;
            feedForward(trainDataSet[j].features); // 参数为一个向量
            getResults(resultVals); // 读取计算结果
            // showVectorVals("Outputs:", resultVals);
            targetVals.push_back(trainDataSet[j].label);
            // cout<<"targetVals.size() = "<<targetVals.size()<<endl;

            //可以去掉
            assert(targetVals.size() == m_layers.back().size()-1);
            // showVectorVals("Targets:", targetVals);
            // 标签为一个向量,二分类的结果实际上只有一个值
            backProp(targetVals);
            targetVals.clear(); // 考虑把targetVals改成int而不是vector
        }
        // cout<<"第 "<<i+1<<" 个epoch训练结束"<<endl;
        // cout<<"m_recentAverageError = "<<m_recentAverageError<<endl;

        // 暂时以整个数据集遍历一次为一个epoch
    
    }

}


void NN::feedForward(const vector<double> &inputVals){

    // Check the num of inputVals euqal to neuronnum expect bias
    // 输入的featrues的长度要和第一层的宽度一致
    // 可以去掉
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign {latch} the input values into the input neurons
    // 将输入值塞到第一层中

	for(int i = 0; i < inputVals.size(); ++i){
		m_layers[0][i].setOutputVal(inputVals[i]); 
        // cout<<"输入 第"<<i<<"个值为"<<inputVals[i]<<endl;
	}

	// Forward propagate
	for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
		Layer &prevLayer = m_layers[layerNum - 1];
		for(int n = 0; n < m_layers[layerNum].size() - 1; ++n){
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
    // cout<<"填入成功----"<<endl;

}


void NN::getResults(vector<double> &resultVals) const{
    resultVals.clear();

	for(int n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
        // cout<<"前向计算结果 ： "<<m_layers.back()[n].getOutputVal()<<endl;
	}
}

void NN::backProp(const std::vector<double> &targetVals){
	// Calculate overal net error (RMS of output neuron errors)
    // cout<<"开始反向传播----"<<endl;
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for(int n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta *delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = sqrt(m_error); // RMS
    // cout<<"RMS = "<<m_error<<endl;
	// Implement a recent average measurement:

	m_recentAverageError = 
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);
	// Calculate output layer gradients
    
    // cout<<"开始梯度计算----"<<endl;
	for(int n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}
    
	// Calculate gradients on hidden layers
    // cout<<"开始计算隐藏层---"<<endl;
	for(int layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for(int n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights
    // cout<<"开始更新权重---"<<endl;
	for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for(int n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void NN::storeModel(){

    ofstream fout(weightParamFile.c_str());
    if (!fout.is_open()) {
        cout << "打开模型参数文件失败" << endl;
    }

    for(int i = 0; i< m_layers.size();i++){
        for(int j = 0; j< m_layers[i].size();j++){
            fout << m_layers[i][j].getOutputVal() << " ";
        }
    }
}

void NN::predict(){

    double sigVal;
    int predictVal;
    vector<double> resultVals;
    loadTestData();
    for (int j = 0; j < testDataSet.size(); j++) {
        feedForward(testDataSet[j].features);
        getResults(resultVals); 
        // sigVal = sigmoidCalc(wxbCalc(testDataSet[j]));
        sigVal = resultVals[0];
        predictVal = sigVal >= predictTrueThresh ? 1 : 0;
        predictVec.push_back(predictVal);
    }

    storePredict(predictVec);
}

bool NN::init(vector<int>& shape){
    // 先处理一部分变量并加载数据集
    // cout<<"start init----"<<endl;
    trainDataSet.clear();
    bool status = loadTrainData();
    if (status != true) {
        return false;
    }
    featuresNum = trainDataSet[0].features.size();

    // 神经网络的初始化：包括网络的构造以及权重的初始化

    shape.insert(shape.begin(),featuresNum);
    cout<<"start init nn ----"<<endl;
    int numLayers = shape.size();
	for(int layerNum = 0; layerNum < numLayers; ++layerNum){
		m_layers.push_back(Layer());
		// numOutputs of layer[i] is the numInputs of layer[i+1]
		// numOutputs of last layer is 0
		int numOutputs = layerNum == shape.size() - 1 ? 0 :shape[layerNum + 1];

		// We have made a new Layer, now fill it ith neurons, and
		// add a bias neuron to the layer:
        int neuronNum = 0;
		for(neuronNum = 0; neuronNum <= shape[layerNum]; ++neuronNum){
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			// cout << "Made a Neuron!" << endl;
		}
        // cout<< "layer : "<< layerNum << " neuronNum : "<< neuronNum<<endl;
		// Force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}

    return true;
}

bool NN::loadTrainData(){
    ifstream infile(trainFile.c_str());
    string line;

    if (!infile) {
        cout << "打开训练文件失败" << endl;
        exit(0);
    }

    while (infile) {
        getline(infile, line);
        if (line.size() > featuresNum) {
            stringstream sin(line);
            char ch;
            double dataV;
            int i;
            vector<double> feature;
            i = 0;

            while (sin) {
                char c = sin.peek();
                if (int(c) != -1) {
                    sin >> dataV;
                    feature.push_back(dataV);
                    sin >> ch;
                    i++;
                } else {
                    cout << "训练文件数据格式不正确，出错行为" << (trainDataSet.size() + 1) << "行" << endl;
                    return false;
                }
            }
            int ftf;
            ftf = (int)feature.back();
            feature.pop_back();
            trainDataSet.push_back(Data(feature, ftf));
        }
    }
    infile.close();
    return true;
}

bool NN::loadTestData()
{
    ifstream infile(testFile.c_str());
    string lineTitle;

    if (!infile) {
        cout << "打开测试文件失败" << endl;
        exit(0);
    }

    while (infile) {
        vector<double> feature;
        string line;
        getline(infile, line);
        if (line.size() > featuresNum) {
            stringstream sin(line);
            double dataV;
            int i;
            char ch;
            i = 0;
            while (i < featuresNum && sin) {
                char c = sin.peek();
                if (int(c) != -1) {
                    sin >> dataV;
                    feature.push_back(dataV);
                    sin >> ch;
                    i++;
                } else {
                    cout << "测试文件数据格式不正确" << endl;
                    return false;
                }
            }
            testDataSet.push_back(Data(feature, 0));
        }
    }

    infile.close();
    return true;
}

int NN::storePredict(vector<int> &predict)
{
    string line;
    int i;

    ofstream fout(predictOutFile.c_str());
    if (!fout.is_open()) {
        cout << "打开预测结果文件失败" << endl;
    }
    for (i = 0; i < predict.size(); i++) {
        fout << predict[i] << endl;
    }
    fout.close();
    return 0;
}

