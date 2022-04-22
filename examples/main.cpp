//
// Created by vityha on 11.04.22.
//

#include <memory>

#include "MLP.h"
#include "lossfunction/MSELossFunction.h"


int main() {
    MLP mlp{2, std::shared_ptr<MSELossFunction>(new MSELossFunction())};
//    mlp.addLayer(3, ActivationFunction::Sigmoid);
//    mlp.addLayer(3, ActivationFunction::HyperbolicTangent);
    std::cout << mlp;
    Eigen::VectorXd ins(2);
    ins << 1, 2;
    Eigen::VectorXd res = mlp.forwardPass(ins);
    std::cout << res;

    return 0;
}