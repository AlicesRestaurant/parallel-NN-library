#ifndef NNLIB_AND_TEST_EXAMPLE_SOFTMAXLOSSFUNCTION_H
#define NNLIB_AND_TEST_EXAMPLE_SOFTMAXLOSSFUNCTION_H


#include "LossFunction.h"

class SoftMaxLossFunction : public LossFunction {
    double forwardPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) override;
    Eigen::MatrixXd backPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) override;
};


#endif //NNLIB_AND_TEST_EXAMPLE_SOFTMAXLOSSFUNCTION_H