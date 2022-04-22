//
// Created by vityha on 29.03.22.
//

#include "Layer.h"

#include <tuple>

using Eigen::VectorXd;
using Eigen::MatrixXd;

using matrixTuple = std::tuple<MatrixXd, MatrixXd>;

// Initialization

Layer::Layer(int layerInputsNumber, int nodesNumber, ActivationFunction activationFunction) : nodesNumber{nodesNumber},
                                                                                              activationFunction{
                                                                                                      activationFunction} {
    weights.resize(nodesNumber, layerInputsNumber + 1);
    weights = MatrixXd::Random(nodesNumber, layerInputsNumber + 1);
}

// Forward propagation

MatrixXd Layer::forwardPropagate(const MatrixXd &inputs) {
    MatrixXd inputsWithBias(inputs.rows() + 1, inputs.cols());
    inputsWithBias << MatrixXd::Ones(1, inputs.cols()), inputs;
    layerInputs = inputsWithBias;
    MatrixXd activationInputs = calculateActivationInputs();
    return calculateActivations(activationInputs);
}

MatrixXd Layer::calculateActivationInputs() {
    return weights * layerInputs;
}

MatrixXd Layer::calculateActivations(const MatrixXd &inputs) {
    MatrixXd outputs;
    switch (activationFunction) {
        case ActivationFunction::Sigmoid:
            outputs = sigmoid(inputs);
            break;
        case ActivationFunction::HyperbolicTangent:
            outputs = hyperbolicTangent(inputs);
            break;
        case ActivationFunction::SoftMax:
            outputs = softMax(inputs);
            break;
    }
    return outputs;
}

// Backward propagation

matrixTuple Layer::backPropagate(const MatrixXd &topDerivatives) {
    MatrixXd activationInputs = calculateActivationInputs();
    MatrixXd derivativesByActivationInputs = calculateDerivativesByActivationInputs(activationInputs, topDerivatives);
    int examplesNum = layerInputs.cols();
    MatrixXd meanBatchDerivativesByWeights;
    for (int i = 0; i < examplesNum; i++){
        meanBatchDerivativesByWeights += derivativesByActivationInputs.col(i) * layerInputs.col(i).transpose();
    }
    meanBatchDerivativesByWeights = meanBatchDerivativesByWeights / examplesNum;
    MatrixXd bottomDerivatives = weights.transpose() * derivativesByActivationInputs;
    return std::make_tuple(bottomDerivatives, meanBatchDerivativesByWeights);
}

MatrixXd
Layer::calculateDerivativesByActivationInputs(const MatrixXd &activationInputs, const MatrixXd &topDerivatives) {
    MatrixXd derivatives;
    switch (activationFunction) {
        case ActivationFunction::Sigmoid:
            derivatives = derivativesSigmoid(activationInputs, topDerivatives);
            break;
        case ActivationFunction::HyperbolicTangent:
            derivatives = derivativesHyperbolicTangent(activationInputs, topDerivatives);
            break;
        case ActivationFunction::SoftMax:
            derivatives = derivativesSoftMax(activationInputs, topDerivatives);
            break;
    }
    return derivatives;
}

// Weights

void Layer::updateWeights(const MatrixXd &newWeights) {
    weights = newWeights;
}

MatrixXd Layer::getWeights() const {
    return weights;
}

// Structure

void Layer::setNodesNumber(int number) {
    nodesNumber = number;
}

int Layer::getNodesNumber() const {
    return nodesNumber;
}

// Activation function

void Layer::setActivationFunction(const ActivationFunction &newActivationFunction) {
    activationFunction = newActivationFunction;
}

Layer::ActivationFunction Layer::getActivationFunction() const {
    return activationFunction;
}

std::string Layer::getActivationFunctionName() const {
    switch (activationFunction) {
        case ActivationFunction::HyperbolicTangent:
            return "HyperbolicTangent";
        case ActivationFunction::Sigmoid:
            return "Sigmoid";
        case ActivationFunction::SoftMax:
            return "SoftMax";
    }
}

// Different activation functions

MatrixXd Layer::hyperbolicTangent(const MatrixXd &inputs) {
    return inputs.unaryExpr([](double x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); });
}

MatrixXd Layer::softMax(const MatrixXd &inputs) {
    VectorXd denominators = inputs.unaryExpr(std::ref(exp)).colwise().sum().unaryExpr([](double x) { return 1 / x; });
    return inputs.unaryExpr(std::ref(exp)) * denominators.asDiagonal();
}

MatrixXd Layer::sigmoid(const MatrixXd &inputs) {
    return inputs.unaryExpr([](double x) { return 1 / (1 + exp(-x)); });
}

// Different derivatives by activation inputs through activation functions

MatrixXd Layer::derivativesHyperbolicTangent(const MatrixXd &activationInputs, const MatrixXd &topDerivatives) {
    MatrixXd activationDerivatives = activationInputs.unaryExpr(
            [](double x) { return 2 / pow((exp(x) + exp(-x)), 2); });
    return topDerivatives.cwiseProduct(activationDerivatives);
}

MatrixXd Layer::derivativesSigmoid(const MatrixXd &activationInputs, const MatrixXd &topDerivatives) {
    MatrixXd activationDerivatives = sigmoid(activationInputs).cwiseProduct(
            MatrixXd::Ones(activationInputs.rows(), activationInputs.cols()) - sigmoid(activationInputs));
    return topDerivatives.cwiseProduct(activationDerivatives);
}

MatrixXd Layer::derivativesSoftMax(const MatrixXd &activationInputs, const MatrixXd &topDerivatives) {
    MatrixXd derivativesByActivationInputs(nodesNumber, activationInputs.cols());
    MatrixXd layerOutputs = softMax(activationInputs);
    for (int exampleNum = 0; exampleNum < activationInputs.cols(); exampleNum++) {
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