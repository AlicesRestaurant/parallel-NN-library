#include <Eigen/Core>

#include "lossfunction/FullLoss.h"

double FullLoss::forwardPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) {
    return lossFunctionPtr->forwardPropagate(bottomData, labels);
}

Eigen::MatrixXd FullLoss::backPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) {
    return lossFunctionPtr->backPropagate(bottomData, labels);
}