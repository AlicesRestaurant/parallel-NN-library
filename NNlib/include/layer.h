//
// Created by vityha on 29.03.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_LAYER_H
#define NNLIB_AND_TEST_EXAMPLE_LAYER_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

// TODO: addNewLayer() in MLP and test

class Layer {
public:
    enum class ActivationFunction{
        HyperbolicTangent,
        Sigmoid,
        SoftMax
    };

    Layer(int layerInputsNumber, int nodesNumber, ActivationFunction activationFunction);

    Eigen::VectorXd forwardPropagate(const Eigen::VectorXd &bottomData);
    Eigen::VectorXd backPropagate(const Eigen::VectorXd &topDerivatives, double alpha);

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
    Eigen::VectorXd layerInputs;
    int nodesNumber;

    Eigen::VectorXd calculateActivationInputs();
    Eigen::VectorXd calculateActivations(const Eigen::VectorXd &inputs);
    Eigen::VectorXd calculateDerivativesByActivationInputs(const Eigen::VectorXd &activationInputs, const Eigen::VectorXd &topDerivatives);

    Eigen::VectorXd hyperbolicTangent(const Eigen::VectorXd &inputs);
    Eigen::VectorXd sigmoid(const Eigen::VectorXd &inputs);
    Eigen::VectorXd softMax(const Eigen::VectorXd &inputs);

    Eigen::VectorXd derivativesHyperbolicTangent(const Eigen::VectorXd &activationInputs, const Eigen::VectorXd &topDerivatives);
    Eigen::VectorXd derivativesSigmoid(const Eigen::VectorXd &activationInputs, const Eigen::VectorXd &topDerivatives);
    Eigen::VectorXd derivativesSoftMax(const Eigen::VectorXd &activationInputs, const Eigen::VectorXd &topDerivatives);
};

#endif //NNLIB_AND_TEST_EXAMPLE_LAYER_H
