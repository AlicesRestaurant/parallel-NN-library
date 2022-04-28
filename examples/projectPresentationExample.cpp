#include <Model.h>
#include <layer/FCLayer.h>
#include <lossfunction/SoftMaxLossFunction.h>
#include <layer/SigmoidActivationLayer.h>
#include "lossfunction/SVMLossFunction.h"
#include "lossfunction/MSELossFunction.h"
#include <Eigen/Core>

using Eigen::MatrixXd;

int main() {
    Model model{1, std::make_shared<MSELossFunction>()};
    model.addLayer<FCLayer>(1, 1);

    MatrixXd trainX{{1, 2, 3}};
    MatrixXd trainY{{1, 2, 3}};

    for (int i = 0; i < 10000; ++i) {
        model.trainBatch(trainX, trainY, 0.01);
    }

    std::cout << model << std::endl;

    return 0;
}