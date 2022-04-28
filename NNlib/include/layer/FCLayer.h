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

    Eigen::MatrixXd forwardPropagate(const Eigen::MatrixXd &bottomData) override;

    Eigen::MatrixXd calculateGradientsWrtInputs(const Eigen::MatrixXd &topDerivatives) override;
    Eigen::MatrixXd calculateGradientsWrtWeights(const Eigen::MatrixXd &topDerivatives) override;

    void updateWeights(const Eigen::MatrixXd &newWeights) override;
    Eigen::MatrixXd getWeights() const override;

protected:
    Eigen::MatrixXd weights;
};

#endif //NNLIB_AND_TEST_EXAMPLE_FCLAYER_H
