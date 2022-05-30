//
// Created by vityha on 22.04.22.
//

#include "layer/ActivationLayer.h"
#include "layer/Layer.h"
#include "matrix/MatrixType.h"

#include <Eigen/Dense>


// Initialization

ActivationLayer::ActivationLayer(int nodesNumber) :
Layer(nodesNumber, Layer::LayerType::Activation) {}


// Forward propagation

MatrixType ActivationLayer::forwardPropagate(const MatrixType &inputs) {
    layerInputs = inputs;
    return calculateActivations(inputs);
}

// Backward propagation


MatrixType ActivationLayer::calculateGradientsWrtInputs(const MatrixType &topDerivatives) {
    return calculateDerivatives(topDerivatives);
}

MatrixType ActivationLayer::calculateGradientsWrtWeights(const MatrixType &topDerivatives) {
    return MatrixType();
}

// Weights

void ActivationLayer::updateWeights(const MatrixType &newWeights) {
}

MatrixType ActivationLayer::getWeights() const {
    return {};
}

