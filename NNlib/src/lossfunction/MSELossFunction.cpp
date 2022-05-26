#include "lossfunction/MSELossFunction.h"

double MSELossFunction::forwardPropagate(const MatrixType &bottomData, const MatrixType &labels) {
    return (bottomData - labels).unaryExpr([] (double x) {return x*x;}).sum() / bottomData.cols() / bottomData.rows();
}

MatrixType MSELossFunction::backPropagate(const MatrixType &bottomData, const MatrixType &labels) {
    return 2.0 / bottomData.cols() / bottomData.rows() * (bottomData - labels);
}
