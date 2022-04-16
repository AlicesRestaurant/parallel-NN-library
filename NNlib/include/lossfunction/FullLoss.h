#ifndef NNLIB_AND_TEST_EXAMPLE_FULLLOSS_H
#define NNLIB_AND_TEST_EXAMPLE_FULLLOSS_H

#include <memory>

#include <Eigen/Dense>

#include "lossfunction/LossFunction.h"
#include "MSELossFunction.h"

class FullLoss {
public:
    FullLoss(std::shared_ptr<LossFunction> lossFunctionPtr): lossFunctionPtr{lossFunctionPtr} {}

    // calculate and return loss
    double forwardPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels);

    // calculate and return loss
    // calculate and output gradient of loss with respect to bottomData
    Eigen::MatrixXd backPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels);

private:
    std::shared_ptr<LossFunction> lossFunctionPtr;
};

#endif //NNLIB_AND_TEST_EXAMPLE_FULLLOSS_H
