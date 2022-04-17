#include "lossfunction/SoftMaxLossFunction.h"

#include <Eigen/Core>

double forwardPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) {
    Eigen::MatrixXd exponentiated = bottomData.array().exp();
    Eigen::MatrixXd probabilities = (labels.array() * bottomData.array()).rowwise().sum().array() / exponentiated.rowwise().sum().array();
    return - 1.0 / bottomData.rows() * probabilities.array().log().sum();
}

Eigen::MatrixXd backPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) {
    return 1.0 / bottomData.rows() * ((bottomData.array().colwise() / bottomData.rowwise().sum().array()).matrix() - labels);
}
