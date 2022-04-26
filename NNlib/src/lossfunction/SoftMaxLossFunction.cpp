#include "lossfunction/SoftMaxLossFunction.h"

#include <Eigen/Core>

double SoftMaxLossFunction::forwardPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) {
    Eigen::MatrixXd exponentiated = bottomData.array().exp();
    Eigen::MatrixXd probabilities = (labels.array() * exponentiated.array()).colwise().sum().array() /
            exponentiated.colwise().sum().array();

    return - 1.0 / bottomData.cols() * probabilities.array().log().sum();
}

Eigen::MatrixXd SoftMaxLossFunction::backPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) {
    // TODO: derive the formula analytically
    Eigen::MatrixXd bottomDataExp = bottomData.array().exp();
    auto grad = 1.0 / bottomData.cols() * ((bottomDataExp.array().rowwise() / bottomDataExp.colwise().sum().array()).matrix() - labels);
    return grad;
}
