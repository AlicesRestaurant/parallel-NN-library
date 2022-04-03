//
// Created by vityha on 29.03.22.
//

#include "layer.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Initialization

Layer::Layer(int layerInputsNumber, int nodesNumber){
    weights.resize(nodesNumber, layerInputsNumber + 1);
    this->nodesNumber = nodesNumber;
}

// Forward propagation

VectorXd Layer::forwardPropagate(const VectorXd &inputs) {
    VectorXd inputsWithBias(inputs.size() + 1);
    inputsWithBias << 1;
    inputsWithBias << inputs;
    layerInputs = std::move(inputsWithBias);
    VectorXd activationInputs = calculateActivationInputs();
    return calculateActivations(activationInputs);
}

VectorXd Layer::calculateActivationInputs() {
    return weights * layerInputs;
}

VectorXd Layer::calculateActivations(const VectorXd &inputs) {
    VectorXd outputs;
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

VectorXd Layer::backPropagate(const VectorXd &topDerivatives, double alpha) {
    VectorXd activationInputs = calculateActivationInputs();
    VectorXd derivativesByActivationInputs = calculateDerivativesByActivationInputs(activationInputs, topDerivatives);
    MatrixXd derivativesByWeights = derivativesByActivationInputs * layerInputs.transpose();
    VectorXd bottomDerivatives = weights.transpose() * derivativesByActivationInputs;
    updateWeights(weights - alpha * derivativesByWeights);
    return bottomDerivatives;
}

VectorXd Layer::calculateDerivativesByActivationInputs(const VectorXd &activationInputs, const VectorXd &topDerivatives) {
    VectorXd derivatives;
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

void Layer::updateWeights(const VectorXd &newWeights) {
    weights = newWeights;
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

// Different activation functions

VectorXd Layer::hyperbolicTangent(const VectorXd &inputs) {
    return inputs.unaryExpr([](double x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); });
}

VectorXd Layer::softMax(const VectorXd &inputs) {
    double denominator = inputs.unaryExpr(std::ref(exp)).sum();
    return inputs.unaryExpr([denominator](double x) { return exp(x) / denominator; });
}

VectorXd Layer::sigmoid(const VectorXd &inputs) {
    return inputs.unaryExpr([](double x) { return 1 / (1 + exp(x)); });
}

// Different derivatives by activation inputs through activation functions

VectorXd Layer::derivativesHyperbolicTangent(const VectorXd &activationInputs, const VectorXd &topDerivatives) {
    VectorXd activationDerivatives = activationInputs.unaryExpr([](double x) { return 2 / pow((exp(x) + exp(-x)), 2); });
    return topDerivatives.cwiseProduct(activationDerivatives);
}

VectorXd Layer::derivativesSigmoid(const VectorXd &activationInputs, const VectorXd &topDerivatives) {
    VectorXd activationDerivatives = sigmoid(activationInputs).cwiseProduct(VectorXd::Ones(activationInputs.size()) - sigmoid(activationInputs));
    return topDerivatives.cwiseProduct(activationDerivatives);
}

VectorXd Layer::derivativesSoftMax(const VectorXd &activationInputs, const VectorXd &topDerivatives) {
    VectorXd derivativesByActivationInputs(nodesNumber);
    VectorXd layerOutputs = softMax(activationInputs);
    for (int i = 0; i < nodesNumber; i++) {
        double sumOfDerivativeParts = 0;
        for (int j = 0; j < nodesNumber; j++) {
            double activationDerivative = 0;
            if (i == j) {
                activationDerivative = layerOutputs[j] * (1 - layerOutputs[i]);
            } else {
                activationDerivative = layerOutputs[j] * (-layerOutputs[i]);
            }
            sumOfDerivativeParts += activationDerivative * topDerivatives[j];
        }
        derivativesByActivationInputs[i] = sumOfDerivativeParts;
    }
    return derivativesByActivationInputs;
}