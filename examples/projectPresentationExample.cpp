#include <Model.h>
#include <layer/FCLayer.h>
#include "lossfunction/MSELossFunction.h"
#include <matrix/MatrixType.h>
#include <Eigen/Core>

int main() {
    Model model{1, std::make_shared<MSELossFunction>()};
    model.addLayer<FCLayer>(1, 1);

    MatrixType trainX{{1, 2, 3}};
    MatrixType trainY{{1, 2, 3}};

    for (int i = 0; i < 10000; ++i) {
        model.trainBatch(trainX, trainY, 0.01);
    }

    std::cout << model << std::endl;

    return 0;
}