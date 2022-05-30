#include "lossfunction/SoftMaxLossFunction.h"
#include "matrix/MatrixType.h"

#include <functional> // cref
#include <cmath> // exp

#include <Eigen/Core>

double SoftMaxLossFunction::forwardPropagate(const MatrixType &bottomData, const MatrixType &labels) {
#ifdef USE_EIGEN
    Eigen::MatrixXd exponentiated = bottomData.array().exp();
    Eigen::MatrixXd probabilities = (labels.array() * exponentiated.array()).colwise().sum().array() /
            exponentiated.colwise().sum().array();

    return - 1.0 / bottomData.cols() * probabilities.array().log().sum();
#else
    MatrixType exponentiated = bottomData.unaryExpr([] (double el) {return exp(el);});
    MatrixD probabilities = labels.cwiseProduct(exponentiated)
            .colReduce([] (double el1, double el2) {return el1 + el2;}, 0)
            / exponentiated.colReduce([] (double el1, double el2) {return el1 + el2;}, 0);

    return - 1.0 / bottomData.cols() * probabilities.unaryExpr(log).sum();
#endif
}

MatrixType SoftMaxLossFunction::backPropagate(const MatrixType &bottomData, const MatrixType &labels) {
    // TODO: derive the formula analytically
#ifdef USE_EIGEN
    Eigen::MatrixXd bottomDataExp = bottomData.array().exp();
    auto grad = 1.0 / bottomData.cols() * ((bottomDataExp.array().rowwise() / bottomDataExp.colwise().sum().array()).matrix() - labels);
#else
    MatrixType bottomDataExp = bottomData.unaryExpr([] (double el) {return exp(el);});
    MatrixD bottomDataExpSum = bottomDataExp.colReduce([] (double el1, double el2) {return el1 + el2;}, 0);
    for (size_t rowIdx = 0; rowIdx < bottomDataExp.rows(); ++rowIdx) {
        bottomDataExp.row(rowIdx) /= bottomDataExpSum;
    }
    auto grad = 1.0 / bottomData.cols() *
                (bottomDataExp - labels);
#endif
    return grad;
}
