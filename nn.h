#ifndef __NN
#define __NN

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include "neuron.h"
using namespace std;

struct Data {
    vector<double> features;
    int label;
    Data(vector<double> f, int l) : features(f), label(l)
    {}
};

class Neuron;
class NN{

public:

    NN(string trainF, string testF, string predictOutF); // 构造函数没有返回值
    void train();
    void storeModel();
    void predict();

private:
    vector<Data> trainDataSet;
    vector<Data> testDataSet;
    vector<int> predictVec;
    string trainFile;
    string testFile;
    string predictOutFile;
    string weightParamFile = "modelweight.txt";

private:
    bool init(vector<int>& shape);
    bool loadTrainData();
    bool loadTestData();
    int storePredict(vector<int> &predict);
    void feedForward(const vector<double> &inputVals);
    void getResults(vector<double> &resultVals) const;
    void backProp(const std::vector<double> &targetVals);
    void showVectorVals(string label, vector<double> &v);


private:
    int featuresNum;
    vector<Layer> m_layers;
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
    const int maxIterTimes = 500;
    const double predictTrueThresh = 0.5;
// 50 100 200 500
};

#endif