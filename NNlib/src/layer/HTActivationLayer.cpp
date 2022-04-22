//
// Created by vityha on 22.04.22.
//

#include "layer/HTActivationLayer.h"

#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

Eigen::MatrixXd HTActivationLayer::calculateActivations(const Eigen::MatrixXd &inputs) {
    return inputs.unaryExpr([](double x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); });
}

Eigen::MatrixXd HTActivationLayer::calculateGradientsWrtInputs(const Eigen::MatrixXd &topDerivatives) {
    MatrixXd activationDerivatives = layerInputs.unaryExpr(
            [](double x) { return 2 / pow((exp(x) + exp(-x)), 2); });
    return topDerivatives.cwiseProduct(activationDerivatives);
}