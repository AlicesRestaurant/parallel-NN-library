//
// Created by vityha on 29.03.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_LAYER_H
#define NNLIB_AND_TEST_EXAMPLE_LAYER_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

class Layer {
public:
    enum class ActivationFunction{
        HyperbolicTangent,
        Sigmoid,
        SoftMax
    };

    Layer(int layerInputsNumber, int nodesNumber, ActivationFunction activationFunction);

    Eigen::MatrixXd forwardPropagate(const Eigen::MatrixXd &bottomData);
    // Returns matrix of derivatives with respect to layer inputs and
    // mean of derivatives with respect to weights
    // for use in Optimizer
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> backPropagate(const Eigen::MatrixXd &topDerivatives);

    void updateWeights(const Eigen::MatrixXd &newWeights);
    Eigen::MatrixXd getWeights() const;

    void setActivationFunction(const ActivationFunction &newActivationFunction);
    ActivationFunction getActivationFunction() const;
    std::string getActivationFunctionName() const;

    void setNodesNumber(int number);
    int getNodesNumber() const;


protected:
    Eigen::MatrixXd weights;
    ActivationFunction activationFunction;
    Eigen::MatrixXd layerInputs;
    int nodesNumber;

    Eigen::MatrixXd calculateActivationInputs();
    Eigen::MatrixXd calculateActivations(const Eigen::MatrixXd &inputs);
    Eigen::MatrixXd calculateDerivativesByActivationInputs(const Eigen::MatrixXd &activationInputs, const Eigen::MatrixXd &topDerivatives);

    Eigen::MatrixXd hyperbolicTangent(const Eigen::MatrixXd &inputs);
    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &inputs);
    Eigen::MatrixXd softMax(const Eigen::MatrixXd &inputs);

    Eigen::MatrixXd derivativesHyperbolicTangent(const Eigen::MatrixXd &activationInputs, const Eigen::MatrixXd &topDerivatives);
    Eigen::MatrixXd derivativesSigmoid(const Eigen::MatrixXd &activationInputs, const Eigen::MatrixXd &topDerivatives);
    Eigen::MatrixXd derivativesSoftMax(const Eigen::MatrixXd &activationInputs, const Eigen::MatrixXd &topDerivatives);
};

#endif //NNLIB_AND_TEST_EXAMPLE_LAYER_H
