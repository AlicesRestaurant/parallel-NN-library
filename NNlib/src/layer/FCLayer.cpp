//
// Created by vityha on 22.04.22.
//

#include "layer/FCLayer.h"

#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

// Initialization

FCLayer::FCLayer(size_t nodesNumber, size_t layerInputsNumber, double minWeight, double maxWeight) :
        Layer{nodesNumber, Layer::LayerType::FC},
        weights(MatrixType::Random(nodesNumber, layerInputsNumber + 1))
{
    weights = (weights.array() + 1) / 2 * (maxWeight - minWeight) + minWeight;
}

// Forward propagation

MatrixType FCLayer::forwardPropagate(const MatrixType &inputs) {
    MatrixType inputsWithBias(inputs.rows() + 1, inputs.cols());
    inputsWithBias << MatrixType::Ones(1, inputs.cols()), inputs;
    layerInputs = inputsWithBias;
    return weights * layerInputs;;
}

// Backward propagation

MatrixType FCLayer::calculateGradientsWrtInputs(const MatrixType &topDerivatives) {
    using Eigen::placeholders::last, Eigen::placeholders::all;
    MatrixType bottomDerivatives = weights(all, Eigen::seq(1, last)).transpose() *
                                                                        topDerivatives;
    return bottomDerivatives;
}

MatrixType FCLayer::calculateGradientsWrtWeights(const MatrixType &topDerivatives) {
    int examplesNum = layerInputs.cols();
#if 0
    MatrixType batchDerivativesByWeights(weights.rows(), weights.cols());
    for (int i = 0; i < examplesNum; i++) {
        batchDerivativesByWeights += topDerivatives.col(i) * layerInputs.col(i).transpose();
    }
    return batchDerivativesByWeights;
#else
    return topDerivatives * layerInputs.transpose();
#endif
}

// Weights

void FCLayer::updateWeights(const MatrixType &newWeights) {
    weights = newWeights;
}

MatrixType FCLayer::getWeights() const {
    return weights;
}