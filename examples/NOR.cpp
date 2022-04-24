#include <Model.h>
#include <lossfunction/MSELossFunction.h>
#include <layer/FCLayer.h>
#include <layer/SigmoidActivationLayer.h>
#include <Eigen/Core>

#include <iostream>
#include <memory>

int main() {
    Model model{2, std::make_shared<MSELossFunction>()};

    model.addLayer<FCLayer>(2, 2);
    model.addLayer<SigmoidActivationLayer>(2);

    model.addLayer<FCLayer>(1, 2);
    model.addLayer<SigmoidActivationLayer>(1);

    Eigen::MatrixXd features(4, 2);
    features << 0, 0,
                0, 1,
                1, 0,
                1, 1;
    features.transposeInPlace();

    Eigen::RowVectorXd labels(4);
    labels << 1, 0, 0, 0;

    double alpha = 1;

    for (int i = 0; i < 10000; ++i) {
        model.trainBatch(features, labels, alpha);
    }

    std::cout << "Features:" << std::endl;
    std::cout << features << std::endl;
    std::cout << std::endl;

    std::cout << "Ground truth:" << std::endl;
    std::cout << labels << std::endl;
    std::cout << std::endl;

    std::cout << "Prediction:" << std::endl;
    std::cout << model.forwardPass(features) << std::endl;
    std::cout << std::endl;
    
    return 0;
}