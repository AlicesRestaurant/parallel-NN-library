//
// Created by vityha on 22.04.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_SIGMOIDACTIVATIONLAYER_H
#define NNLIB_AND_TEST_EXAMPLE_SIGMOIDACTIVATIONLAYER_H


#include "layer/ActivationLayer.h"
#include "matrix/MatrixType.h"

#include <Eigen/Dense>

class SigmoidActivationLayer : public ActivationLayer {
public:
    SigmoidActivationLayer(int numNodes): ActivationLayer{numNodes} {}
protected:
    MatrixType calculateActivations(const MatrixType &inputs) override;
    MatrixType calculateDerivatives(const MatrixType &topDerivatives) override;
};


#endif //NNLIB_AND_TEST_EXAMPLE_SIGMOIDACTIVATIONLAYER_H
