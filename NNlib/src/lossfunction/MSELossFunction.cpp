#include "lossfunction/MSELossFunction.h"

double MSELossFunction::forwardPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) {
    return (bottomData - labels).array().square().sum() / bottomData.rows();
}

Eigen::MatrixXd MSELossFunction::backPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) {
    return 2.0 / bottomData.rows() * (bottomData - labels);
}
