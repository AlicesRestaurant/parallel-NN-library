#include "lossfunction/FullLoss.h"

#include <Eigen/Core>

double FullLoss::forwardPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) {
    return lossFunctionPtr->forwardPropagate(bottomData, labels);
}

Eigen::MatrixXd FullLoss::backPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) {
    return lossFunctionPtr->backPropagate(bottomData, labels);
}