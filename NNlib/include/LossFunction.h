//
// Created by vityha on 11.04.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_LOSSFUNCTION_H
#define NNLIB_AND_TEST_EXAMPLE_LOSSFUNCTION_H

#include <Eigen/Dense>

class LossFunction {
public:
    enum class FunctionType{
        MSE
    };

    // gets and returns one or many examples depending on loss function type
    Eigen::VectorXd forwardPropagate(const Eigen::VectorXd &bottomData);

    // returns derivatives
    Eigen::VectorXd backPropagate(const Eigen::VectorXd &labels);

private:
    FunctionType functionType;
};

#endif //NNLIB_AND_TEST_EXAMPLE_LOSSFUNCTION_H
