//
// Created by vityha on 29.03.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_LAYER_H
#define NNLIB_AND_TEST_EXAMPLE_LAYER_H

#include <Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Layer {
public:
    enum class ActivationFunction{
        HyperbolicTangent,
        Sigmoid,
        SoftMax
    };

    VectorXd forwardPropagate(const VectorXd &inputs);

    void updateWeights(const VectorXd &newWeights);

    void setActivationFunction(const ActivationFunction &newActivationFunction);
    ActivationFunction getActivationFunction() const;


protected:
    MatrixXd weights;
    ActivationFunction activationFunction;
    VectorXd &derivatives;

    VectorXd calculateOutputs(const VectorXd &inputs);
    VectorXd calculateActivations(const VectorXd &inputs);
    void calculateDerivatives(const VectorXd &inputs);

    VectorXd hyperbolicTangent(const VectorXd &inputs);
    VectorXd sigmoid(const VectorXd &inputs);
    VectorXd softMax(const VectorXd &inputs);

    VectorXd hyperbolicTangentDerivatives(const VectorXd &inputs);
    VectorXd sigmoidDerivatives(const VectorXd &inputs);
    VectorXd softMaxDerivatives(const VectorXd &inputs);

};

#endif //NNLIB_AND_TEST_EXAMPLE_LAYER_H
