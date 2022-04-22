//
// Created by vityha on 22.04.22.
//

#include "layer/ActivationLayer.h"

#include "layer/Layer.h"

#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

// Initialization

ActivationLayer::ActivationLayer(int nodesNumber) :
Layer(nodesNumber, Layer::LayerType::Activation) {}


// Forward propagation

MatrixXd ActivationLayer::forwardPropagate(const MatrixXd &inputs) {
    layerInputs = inputs;
    return calculateActivations(inputs);
}

// Backward propagation


Eigen::MatrixXd ActivationLayer::calculateGradientsWrtInputs(const MatrixXd &topDerivatives) {
    return calculateDerivatives(topDerivatives);
}

Eigen::MatrixXd ActivationLayer::calculateGradientsWrtWeights(const MatrixXd &topDerivatives) {
    return {};
}

// Weights

void ActivationLayer::updateWeights(const MatrixXd &newWeights) {
}

MatrixXd ActivationLayer::getWeights() const {
}

