#ifndef NNLIB_AND_TEST_EXAMPLE_MLP_H
#define NNLIB_AND_TEST_EXAMPLE_MLP_H


#include "Model.h"
#include "layer.h"
#include "lossfunction/FullLoss.h"

#include <iostream>

class MLP : public Model {
public:
    MLP(int numInputNodes, FullLoss fullLoss) : numInputNodes{numInputNodes}, fullLoss{fullLoss}
    {}

    Eigen::VectorXd forwardPass(Eigen::VectorXd input);
    void setParameters(double alpha);

    void addLayer(int numNodes, Layer::ActivationFunction activationFunction);
    void printMLP();
    void trainExample(Eigen::VectorXd features, Eigen::VectorXd labels);

    friend std::ostream& operator<<(std::ostream &os, const MLP &mlp);
protected:
    std::vector<Layer> layers;
    double alpha;

    FullLoss fullLoss;

    int numInputNodes;
};


#endif //NNLIB_AND_TEST_EXAMPLE_MLP_H
