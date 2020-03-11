# 从零开始的NN

## 第一阶段

main:

* construction

* train
  * feedForward
  * getResults
  * backProp

* storeModel

* predict


## 第二阶段

### train

* feedFoward : Neuron::setOutputVal, Neuron::feedForward 填充第一层并进行前向计算
* getResults : 从最后一层读取计算结果
* targetVals.push_back : 读取计算结果
* backProp : 反向


