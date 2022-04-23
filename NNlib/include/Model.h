#ifndef NNLIB_AND_TEST_EXAMPLE_MODEL_H
#define NNLIB_AND_TEST_EXAMPLE_MODEL_H

#include "layer/Layer.h"
#include "lossfunction/LossFunction.h"

#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <string>
#include <memory>

class Model {
public:
    Model(int numInputNodes, const std::shared_ptr<LossFunction>& lossFunctionPtr) : numInputNodes{numInputNodes},
                                                                                   lossFunctionPtr{lossFunctionPtr}
    {}

    Eigen::VectorXd forwardPass(Eigen::VectorXd input);

//    void addLayer(int numNodes, Layer::ActivationFunction activationFunction);
    void trainExample(Eigen::VectorXd features, Eigen::VectorXd labels);

    friend std::ostream& operator<<(std::ostream &os, const Model &mlp);
protected:
    std::vector<Layer> layers;

    std::shared_ptr<LossFunction> lossFunctionPtr;

    const int numInputNodes;
};


#endif //NNLIB_AND_TEST_EXAMPLE_MODEL_H
