#ifndef NNLIB_AND_TEST_EXAMPLE_MSELOSSFUNCTION_H
#define NNLIB_AND_TEST_EXAMPLE_MSELOSSFUNCTION_H

#include "matrix/MatrixType.h"

#include "lossfunction/LossFunction.h"

class MSELossFunction : public LossFunction {
    double forwardPropagate(const MatrixType &bottomData, const MatrixType &labels) override;
    MatrixType backPropagate(const MatrixType &bottomData, const MatrixType &labels) override;
};


#endif //NNLIB_AND_TEST_EXAMPLE_MSELOSSFUNCTION_H
