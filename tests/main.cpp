//
// Created by vityha on 11.04.22.
//

#include "MLP.h"

using ActivationFunction = Layer::ActivationFunction;

int main() {
    MLP mlp{2};
    mlp.addLayer(3, ActivationFunction::Sigmoid);
    mlp.addLayer(3, ActivationFunction::HyperbolicTangent);
    mlp.printMLP();
    Eigen::VectorXd ins(2);
    ins << 1, 2;
    Eigen::VectorXd res = mlp.forwardPass(ins);
    std::cout << res;

    return 0;
}