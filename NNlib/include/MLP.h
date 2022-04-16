#ifndef NNLIB_AND_TEST_EXAMPLE_MLP_H
#define NNLIB_AND_TEST_EXAMPLE_MLP_H


#include "Model.h"
#include "layer.h"

#include <iostream>

class MLP : public Model {
public:
    MLP(int numInputNodes); // TODO: implement it
    void addLayer(int numNodes, Layer::ActivationFunction activationFunction);
    void printMLP();
    void trainExample(std::vector<double> features, std::vector<double> labels);

private:
    int numInputNodes;
};


#endif //NNLIB_AND_TEST_EXAMPLE_MLP_H
