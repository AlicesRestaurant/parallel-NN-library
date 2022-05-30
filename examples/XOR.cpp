//
// Created by vityha on 19.04.22.
//

#include <Model.h>
#include <lossfunction/MSELossFunction.h>
#include <lossfunction/SoftMaxLossFunction.h>
#include <layer/FCLayer.h>
#include <layer/SigmoidActivationLayer.h>
#include <layer/HTActivationLayer.h>
#include <Eigen/Dense>
#include <matrix/MatrixType.h>

int main() {
    //create batch
    MatrixType features(2, 4);
    MatrixType labels(4, 1);
#ifdef USE_EIGEN
    features << 0, 0, 1, 1,
            0, 1, 0, 1;
    labels << 0, 1, 1, 0;
#else
    features(0, 0) = 0;
    features(0, 1) = 0;
    features(0, 2) = 1;
    features(0, 3) = 1;
    features(1, 0) = 0;
    features(1, 1) = 1;
    features(1, 2) = 0;
    features(1, 3) = 1;

    labels(0, 0) = 0;
    labels(1, 0) = 1;
    labels(2, 0) = 1;
    labels(3, 0) = 0;
#endif

    srand((unsigned int) 2);

    //create model
    double alpha = 1;

    // TODO: initialize mlp
    Model model{2, std::make_shared<MSELossFunction>()};
    model.addLayer<FCLayer>(2, 2);
    model.addLayer<HTActivationLayer>(2);
    model.addLayer<FCLayer>(1, 2);
    model.addLayer<HTActivationLayer>(1);

    //train model
    for (int i = 0; i < 10000; i++) {
        model.trainBatch(features, labels.transpose(), alpha);
        if (i % 1000 == 0) {
            std::cout << "Loss: " << model.calcLoss(model.forwardPass(features), labels.transpose()) << std::endl;
        }
    }

    //print info
    std::cout << model;

    //test
    std::cout << model.forwardPass(features) << std::endl;

    std::cout << "Loss: " << model.calcLoss(model.forwardPass(features), labels.transpose()) << std::endl;

    return 0;
}