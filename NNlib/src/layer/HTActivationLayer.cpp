//
// Created by vityha on 22.04.22.
//

#include "layer/HTActivationLayer.h"

#include <Eigen/Dense>


MatrixType HTActivationLayer::calculateActivations(const MatrixType &inputs) {
    return inputs.unaryExpr([](double x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); });
}

MatrixType HTActivationLayer::calculateDerivatives(const MatrixType &topDerivatives) {
    MatrixType activationDerivatives = layerInputs.unaryExpr(
            [](double x) { return 2 / pow((exp(x) + exp(-x)), 2); });
    return topDerivatives.cwiseProduct(activationDerivatives);
}