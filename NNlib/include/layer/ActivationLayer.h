//
// Created by vityha on 22.04.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_ACTIVATIONLAYER_H
#define NNLIB_AND_TEST_EXAMPLE_ACTIVATIONLAYER_H

#include "layer/Layer.h"
#include "matrix/MatrixType.h"

class ActivationLayer : public Layer {
public:
    ActivationLayer(int nodesNumber);

    MatrixType forwardPropagate(const MatrixType &bottomData) override;

    MatrixType calculateGradientsWrtInputs(const MatrixType &topDerivatives) override;
    MatrixType calculateGradientsWrtWeights(const MatrixType &topDerivatives) override;

    void updateWeights(const MatrixType &newWeights) override;
    MatrixType getWeights() const override;

protected:
    virtual MatrixType calculateActivations(const MatrixType &inputs) = 0;

    virtual MatrixType calculateDerivatives(const MatrixType &topDerivatives) = 0;
};

#endif //NNLIB_AND_TEST_EXAMPLE_ACTIVATIONLAYER_H
