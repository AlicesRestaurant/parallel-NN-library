//
// Created by vityha on 22.04.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_FCLAYER_H
#define NNLIB_AND_TEST_EXAMPLE_FCLAYER_H

#include "Layer.h"

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

class FCLayer : public Layer {
public:
    FCLayer(size_t nodesNumber, size_t layerInputsNumber, double minWeight = -1, double maxWeight = 1);

    MatrixType forwardPropagate(const MatrixType &bottomData) override;

    MatrixType calculateGradientsWrtInputs(const MatrixType &topDerivatives) override;
    MatrixType calculateGradientsWrtWeights(const MatrixType &topDerivatives) override;

    void updateWeights(const MatrixType &newWeights) override;
    MatrixType getWeights() const override;

protected:
    MatrixType weights;
};

#endif //NNLIB_AND_TEST_EXAMPLE_FCLAYER_H
