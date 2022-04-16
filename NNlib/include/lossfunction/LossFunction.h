#ifndef NNLIB_AND_TEST_EXAMPLE_LOSSFUNCTION_H
#define NNLIB_AND_TEST_EXAMPLE_LOSSFUNCTION_H

#include <Eigen/Core>

class LossFunction {
public:
    // calculate and return loss
    // NOTE: each row corresponds to a single example
    virtual double forwardPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) = 0;

    // calculate and return loss gradient of loss with respect to bottomData
    // NOTE: each row corresponds to a single example
    virtual Eigen::MatrixXd backPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) = 0;
};


#endif //NNLIB_AND_TEST_EXAMPLE_LOSSFUNCTION_H
