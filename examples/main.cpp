//
// Created by vityha on 11.04.22.
//

#include <Model.h>
#include <lossfunction/MSELossFunction.h>
#include <Eigen/Core>

#include <iostream>
#include <memory>

int main() {
    Model model{2, std::shared_ptr<MSELossFunction>(new MSELossFunction())};
//    mlp.addLayer(3, ActivationFunction::Sigmoid);
//    mlp.addLayer(3, ActivationFunction::HyperbolicTangent);
    std::cout << model;
    Eigen::VectorXd ins(2);
    ins << 1, 2;
    Eigen::VectorXd res = model.forwardPass(ins);
    std::cout << res;

    return 0;
}