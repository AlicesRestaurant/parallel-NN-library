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

    Eigen::MatrixXd forwardPass(Eigen::MatrixXd input);
    double calcLoss(Eigen::MatrixXd input, const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& groundTruths);

    template<class LayerType, class... Args>
    void addLayer(Args... args) {
        layerPtrs.push_back(std::make_shared<LayerType>(args...));
    }

    void trainExample(Eigen::VectorXd features, Eigen::VectorXd label, double alpha);
    void trainBatch(Eigen::MatrixXd features, Eigen::MatrixXd labels, double alpha);

    friend std::ostream& operator<<(std::ostream &os, const Model &mlp);
protected:
    std::vector<std::shared_ptr<Layer>> layerPtrs;

    std::shared_ptr<LossFunction> lossFunctionPtr;

    const int numInputNodes;
};


#endif //NNLIB_AND_TEST_EXAMPLE_MODEL_H
