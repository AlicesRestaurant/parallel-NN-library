//
// Created by vityha on 22.04.22.
//

#include "layer/SigmoidActivationLayer.h"

#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

Eigen::MatrixXd SigmoidActivationLayer::calculateActivations(const Eigen::MatrixXd &inputs) {
    return inputs.unaryExpr([](double x) { return 1 / (1 + exp(-x)); });
}

Eigen::MatrixXd SigmoidActivationLayer::calculateDerivatives(const Eigen::MatrixXd &topDerivatives) {
    MatrixXd activationDerivatives = calculateActivations(layerInputs).cwiseProduct(
            MatrixXd::Ones(layerInputs.rows(), layerInputs.cols()) - calculateActivations(layerInputs));
    return topDerivatives.cwiseProduct(activationDerivatives);
}