//
// Created by vityha on 22.04.22.
//

#include "layer/SigmoidActivationLayer.h"
#include "matrix/MatrixType.h"

#include <Eigen/Dense>


MatrixType SigmoidActivationLayer::calculateActivations(const MatrixType &inputs) {
    return inputs.unaryExpr([](double x) { return 1 / (1 + exp(-x)); });
}

MatrixType SigmoidActivationLayer::calculateDerivatives(const MatrixType &topDerivatives) {
    MatrixType activationDerivatives = calculateActivations(layerInputs).cwiseProduct(
            MatrixType::Ones(layerInputs.rows(), layerInputs.cols()) - calculateActivations(layerInputs));
    return topDerivatives.cwiseProduct(activationDerivatives);
}