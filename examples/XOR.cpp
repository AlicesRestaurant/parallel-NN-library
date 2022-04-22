//
// Created by vityha on 19.04.22.
//

#include "MLP.h"

#include <Eigen/Dense>

using DynMatrix = Eigen::MatrixXd;
using DynVector = Eigen::VectorXd;

int main() {
#if 0
    //create batch
    DynMatrix features(2, 4);
    features << 0, 0, 1, 1,
            0, 1, 0, 1;
    DynVector labels;
    labels << 0, 1, 1, 0;

    //create model
    double alpha = 0.2;
    // TODO: initialize mlp
    MLP mlp;
    mlp.setParameters(alpha);
    mlp.addLayer(2, ActivationFunction::Sigmoid);
    mlp.addLayer(1, ActivationFunction::Sigmoid);

    //train model
    for (int i = 0; i < 10; i++) {
        mlp.trainBatch(features, labels);
    }

    //print info
    std::cout << mlp;

    //test
    std::cout << mlp.forwardPass(features);
#endif
    return 0;
}