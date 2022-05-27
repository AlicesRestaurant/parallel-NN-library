//
// Created by vityha on 22.04.22.
//

#include "layer/SMActivationLayer.h"
#include "matrix/MatrixType.h"

#include <Eigen/Dense>


MatrixType SMActivationLayer::calculateActivations(const MatrixType &inputs) {
#ifdef USE_EIGEN
    VectorXd denominators = inputs.unaryExpr(std::ref(exp)).colwise().sum().unaryExpr([](double x) { return 1 / x; });
    return inputs.unaryExpr(std::ref(exp)) * denominators.asDiagonal();
#else
    MatrixD denominators = inputs.unaryExpr(std::ref(exp))
            .colReduce([] (double a, double b) {return a + b;}, 0)
            .unaryExpr([](double x) { return 1 / x; });
    MatrixD activations(inputs.unaryExpr(std::ref(exp)));
    for (size_t i = 0; i < denominators.rows(); ++i) {
        activations.subblock(0, activations.rows(), i, 1 + i) *= denominators(i, 0);
    }
    return activations;
#endif
}

MatrixType SMActivationLayer::calculateDerivatives(const MatrixType &topDerivatives) {
    MatrixType derivativesByActivationInputs(nodesNumber, layerInputs.cols());
    MatrixType layerOutputs = calculateActivations(layerInputs);
    for (int exampleNum = 0; exampleNum < layerInputs.cols(); exampleNum++) {
        for (int activationInputNum = 0; activationInputNum < nodesNumber; activationInputNum++) {
            double sumOfDerivativeByActivationInputParts = 0;
            for (int layerOutputNum = 0; layerOutputNum < nodesNumber; layerOutputNum++) {
                double activationDerivative = 0;
                if (activationInputNum == layerOutputNum) {
                    activationDerivative = layerOutputs(layerOutputNum, exampleNum) * (1 - layerOutputs(activationInputNum, exampleNum));
                } else {
                    activationDerivative = layerOutputs(layerOutputNum, exampleNum) * (-layerOutputs(activationInputNum, exampleNum));
                }
                sumOfDerivativeByActivationInputParts += activationDerivative * topDerivatives(layerOutputNum, exampleNum);
            }
            derivativesByActivationInputs(activationInputNum, exampleNum) = sumOfDerivativeByActivationInputParts;
        }
    }
    return derivativesByActivationInputs;
}