//
// Created by vityha on 22.04.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_SIGMOIDACTIVATIONLAYER_H
#define NNLIB_AND_TEST_EXAMPLE_SIGMOIDACTIVATIONLAYER_H


#include "layer/ActivationLayer.h"

class SigmoidActivationLayer : public ActivationLayer {
protected:
    Eigen::MatrixXd calculateActivations(const Eigen::MatrixXd &inputs) override;

    Eigen::MatrixXd calculateGradientsWrtInputs(const Eigen::MatrixXd &topDerivatives) override;
};


#endif //NNLIB_AND_TEST_EXAMPLE_SIGMOIDACTIVATIONLAYER_H
