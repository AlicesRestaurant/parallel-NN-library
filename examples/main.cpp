//
// Created by vityha on 11.04.22.
//

#include "MLP.h"

#include <memory>

using ActivationFunction = Layer::ActivationFunction;

int main() {
    // TODO: fix design problem with loss functions
    MLP mlp{2, FullLoss(std::shared_ptr<MSELossFunction>(new MSELossFunction()))};
    mlp.addLayer(3, ActivationFunction::Sigmoid);
    mlp.addLayer(3, ActivationFunction::HyperbolicTangent);
    std::cout << mlp;
    Eigen::VectorXd ins(2);
    ins << 1, 2;
    Eigen::VectorXd res = mlp.forwardPass(ins);
    std::cout << res;

    return 0;
}