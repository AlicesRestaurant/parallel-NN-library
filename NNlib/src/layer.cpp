//
// Created by vityha on 29.03.22.
//

#include "layer.h"

// Output and derivatives calculations

VectorXd Layer::forwardPropagate(const VectorXd &inputs){
    derivatives = calculateDerivatives(inputs);
    return calculateOutputs(inputs);
}

VectorXd Layer::calculateOutputs(const VectorXd &inputs) {
    VectorXd activationInputs = weights * inputs;
    return calculateActivations(activationInputs);
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

void Layer::calculateDerivatives(const VectorXd &inputs){
    switch (activationFunction) {
        case ActivationFunction::Sigmoid:
            derivatives = sigmoidDerivatives(inputs);
            break;
        case ActivationFunction::HyperbolicTangent:
            derivatives = hyperbolicTangentDerivatives(inputs);
            break;
        case ActivationFunction::SoftMax:
            derivatives = softMaxDerivatives(inputs);
            break;
    }
}

// Weights

void Layer::updateWeights(const VectorXd &newWeights) {
    weights = newWeights;
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

// Different derivatives

VectorXd Layer::hyperbolicTangentDerivatives(const VectorXd &inputs){
    return inputs.unaryExpr([](double x) { return 2 / pow((exp(x) + exp(-x)), 2); });
}

VectorXd Layer::sigmoidDerivatives(const VectorXd &inputs){
    return sigmoid(inputs).cwiseProduct(VectorXd::Ones(inputs.size()) - sigmoid(inputs));
}

VectorXd softMaxDerivatives(const VectorXd &inputs){

}