//
// Created by vityha on 22.04.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_ACTIVATIONLAYER_H
#define NNLIB_AND_TEST_EXAMPLE_ACTIVATIONLAYER_H

#include "layer/Layer.h"

class ActivationLayer : public Layer {
public:
    ActivationLayer(int nodesNumber);

    Eigen::MatrixXd forwardPropagate(const Eigen::MatrixXd &bottomData) override;

    Eigen::MatrixXd calculateGradientsWrtInputs(const Eigen::MatrixXd &topDerivatives) override;
    Eigen::MatrixXd calculateGradientsWrtWeights(const Eigen::MatrixXd &topDerivatives) override;

    void updateWeights(const Eigen::MatrixXd &newWeights) override;
    Eigen::MatrixXd getWeights() const override;

protected:
    virtual Eigen::MatrixXd calculateActivations(const Eigen::MatrixXd &inputs) = 0;

    virtual Eigen::MatrixXd calculateDerivatives(const Eigen::MatrixXd &topDerivatives) = 0;
};

#endif //NNLIB_AND_TEST_EXAMPLE_ACTIVATIONLAYER_H
