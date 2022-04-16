#ifndef NNLIB_AND_TEST_EXAMPLE_MSELOSSFUNCTION_H
#define NNLIB_AND_TEST_EXAMPLE_MSELOSSFUNCTION_H

#include "lossfunction/LossFunction.h"

class MSELossFunction : public LossFunction {
    double forwardPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) override;
    Eigen::MatrixXd backPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) override;
};


#endif //NNLIB_AND_TEST_EXAMPLE_MSELOSSFUNCTION_H
