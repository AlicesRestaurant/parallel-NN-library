//
// Created by vityha on 22.04.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_HTACTIVATIONLAYER_H
#define NNLIB_AND_TEST_EXAMPLE_HTACTIVATIONLAYER_H

#include "layer/ActivationLayer.h"

class HTActivationLayer : public ActivationLayer {
public:
    HTActivationLayer(int numNodes): ActivationLayer{numNodes} {}
protected:
    Eigen::MatrixXd calculateActivations(const Eigen::MatrixXd &inputs) override;

    Eigen::MatrixXd calculateDerivatives(const Eigen::MatrixXd &topDerivatives) override;
};


#endif //NNLIB_AND_TEST_EXAMPLE_HTACTIVATIONLAYER_H
