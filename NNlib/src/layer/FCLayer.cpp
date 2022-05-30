//
// Created by vityha on 22.04.22.
//

#include "layer/FCLayer.h"
#include "matrix/MatrixType.h"

#include <Eigen/Dense>

// Initialization

FCLayer::FCLayer(size_t nodesNumber, size_t layerInputsNumber, double minWeight, double maxWeight) :
        Layer{nodesNumber, Layer::LayerType::FC},
        weights(MatrixType::Random(nodesNumber, layerInputsNumber + 1))
{
#ifdef USE_EIGEN
    weights = weights.unaryExpr([maxWeight, minWeight] (double x) {
#else
    weights.unaryExprInPlace([maxWeight, minWeight] (double x) {
#endif
        return (x + 1) / 2 * (maxWeight - minWeight) + minWeight;
    });
}

// Forward propagation

MatrixType FCLayer::forwardPropagate(const MatrixType &inputs) {
#ifdef USE_EIGEN
    MatrixType inputsWithBias(inputs.rows() + 1, inputs.cols());
    inputsWithBias << MatrixType::Ones(1, inputs.cols()), inputs;
    layerInputs = inputsWithBias;
#else
    layerInputs = MatrixType(inputs.rows() + 1, inputs.cols());
    for (size_t i = 0; i < layerInputs.cols(); ++i) {
        layerInputs(0, i) = 1;
    }

    for (size_t i = 1; i < layerInputs.rows(); ++i) {
        for (size_t j = 0; j < layerInputs.cols(); ++j) {
            layerInputs(i, j) = inputs(i - 1, j);
        }
    }
#endif
    return weights * layerInputs;;
}

// Backward propagation

MatrixType FCLayer::calculateGradientsWrtInputs(const MatrixType &topDerivatives) {
#ifdef USE_EIGEN
    using Eigen::placeholders::last, Eigen::placeholders::all;
    MatrixType bottomDerivatives = weights(all, Eigen::seq(1, last)).transpose() *
                                                                        topDerivatives;
#else
    MatrixType bottomDerivatives = weights.subblock(0, weights.rows(), 1, weights.cols()).transpose() *
                                   topDerivatives;
#endif
    return bottomDerivatives;
}

MatrixType FCLayer::calculateGradientsWrtWeights(const MatrixType &topDerivatives) {
#if 0
    int examplesNum = layerInputs.cols();
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