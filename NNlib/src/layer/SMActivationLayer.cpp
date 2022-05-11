//
// Created by vityha on 22.04.22.
//

#include "layer/SMActivationLayer.h"

#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

Eigen::MatrixXd SMActivationLayer::calculateActivations(const Eigen::MatrixXd &inputs) {
    VectorXd denominators = inputs.unaryExpr(std::ref(exp)).colwise().sum().unaryExpr([](double x) { return 1 / x; });
    return inputs.unaryExpr(std::ref(exp)) * denominators.asDiagonal();
}

Eigen::MatrixXd SMActivationLayer::calculateDerivatives(const Eigen::MatrixXd &topDerivatives) {
    MatrixXd derivativesByActivationInputs(nodesNumber, layerInputs.cols());
    MatrixXd layerOutputs = calculateActivations(layerInputs);
    for (int exampleNum = 0; exampleNum < layerInputs.cols(); exampleNum++) {
        for (int activationInputNum = 0; activationInputNum < nodesNumber; activationInputNum++) {
            double sumOfDerivativeByActivationInputParts = 0;
            for (int layerOutputNum = 0; layerOutputNum < nodesNumber; layerOutputNum++) {
                double activationDerivative = 0;
                if (activationInputNum == layerOutputNum) {
                    activationDerivative = layerOutputs.coeff(layerOutputNum, exampleNum) * (1 - layerOutputs.coeff(activationInputNum, exampleNum));
                } else {
                    activationDerivative = layerOutputs.coeff(layerOutputNum, exampleNum) * (-layerOutputs.coeff(activationInputNum, exampleNum));
                }
                sumOfDerivativeByActivationInputParts += activationDerivative * topDerivatives.coeff(layerOutputNum, exampleNum);
            }
            derivativesByActivationInputs(activationInputNum, exampleNum) = sumOfDerivativeByActivationInputParts;
        }
    }
    return derivativesByActivationInputs;
}