//
// Created by vityha on 19.04.22.
//

#include <Model.h>
#include <lossfunction/MSELossFunction.h>
#include <layer/FCLayer.h>
#include <layer/SigmoidActivationLayer.h>
#include <Eigen/Dense>

using DynMatrix = Eigen::MatrixXd;
using DynVector = Eigen::VectorXd;

int main() {
    //create batch
    DynMatrix features(2, 4);
    features << 0, 0, 1, 1,
            0, 1, 0, 1;
    DynVector labels(4);
    labels << 0, 1, 1, 0;

    srand((unsigned int) 2);

    //create model
    double alpha = 1;

    // TODO: initialize mlp
    Model model{2, std::make_shared<MSELossFunction>()};
    model.addLayer<FCLayer>(2, 2);
    model.addLayer<SigmoidActivationLayer>(2);
    model.addLayer<FCLayer>(1, 2);
    model.addLayer<SigmoidActivationLayer>(1);

    //train model
    for (int i = 0; i < 10000; i++) {
        model.trainBatch(features, labels.transpose(), alpha);
    }

    //print info
    std::cout << model;

    //test
    std::cout << model.forwardPass(features);

    return 0;
}