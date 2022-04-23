//
// Created by vityha on 22.04.22.
//

#include "layer/FCLayer.h"

#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

// Initialization

FCLayer::FCLayer(int nodesNumber, int layerInputsNumber) :
                                                Layer{nodesNumber, Layer::LayerType::FC},
                                                weights{MatrixXd::Random(nodesNumber, layerInputsNumber + 1)}
{}

// Forward propagation

MatrixXd FCLayer::forwardPropagate(const MatrixXd &inputs) {
    MatrixXd inputsWithBias(inputs.rows() + 1, inputs.cols());
    inputsWithBias << MatrixXd::Ones(1, inputs.cols()), inputs;
    layerInputs = inputsWithBias;
    return weights * layerInputs;;
}

// Backward propagation

MatrixXd FCLayer::calculateGradientsWrtInputs(const MatrixXd &topDerivatives) {
    int examplesNum = layerInputs.cols();
    MatrixXd meanBatchDerivativesByWeights;
    for (int i = 0; i < examplesNum; i++) {
        meanBatchDerivativesByWeights += topDerivatives.col(i) * layerInputs.col(i).transpose();
    }
    meanBatchDerivativesByWeights = meanBatchDerivativesByWeights / examplesNum;
    MatrixXd bottomDerivatives = weights.transpose() * topDerivatives;
    return bottomDerivatives;
}

MatrixXd FCLayer::calculateGradientsWrtWeights(const MatrixXd &topDerivatives) {
    int examplesNum = layerInputs.cols();
    MatrixXd meanBatchDerivativesByWeights;
    for (int i = 0; i < examplesNum; i++) {
        meanBatchDerivativesByWeights += topDerivatives.col(i) * layerInputs.col(i).transpose();
    }
    meanBatchDerivativesByWeights = meanBatchDerivativesByWeights / examplesNum;
    return meanBatchDerivativesByWeights;
}

// Weights

void FCLayer::updateWeights(const MatrixXd &newWeights) {
    weights = newWeights;
}

MatrixXd FCLayer::getWeights() const {
    return weights;
}