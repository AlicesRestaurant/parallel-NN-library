#ifndef NNLIB_AND_TEST_EXAMPLE_SOFTMAXLOSSFUNCTION_H
#define NNLIB_AND_TEST_EXAMPLE_SOFTMAXLOSSFUNCTION_H

#include "lossfunction/LossFunction.h"

#include <Eigen/Dense>

// Note: the invoker of forwardPropagate() and backPropagate() need to guarantee that each column of labels is one-hot
class SoftMaxLossFunction : public LossFunction {
public:
    double forwardPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) override;
    Eigen::MatrixXd backPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) override;
};


#endif //NNLIB_AND_TEST_EXAMPLE_SOFTMAXLOSSFUNCTION_H
