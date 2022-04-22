#ifndef NNLIB_AND_TEST_EXAMPLE_MLP_H
#define NNLIB_AND_TEST_EXAMPLE_MLP_H

#include <memory>
#include <iostream>

#include "Model.h"
#include "layer/Layer.h"
#include "lossfunction/LossFunction.h"

class MLP : public Model {
public:
    MLP(int numInputNodes, const std::shared_ptr<LossFunction>& lossFunctionPtr) : numInputNodes{numInputNodes},
                                                                            lossFunctionPtr{lossFunctionPtr}
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

    std::shared_ptr<LossFunction> lossFunctionPtr;

    int numInputNodes;
};


#endif //NNLIB_AND_TEST_EXAMPLE_MLP_H
