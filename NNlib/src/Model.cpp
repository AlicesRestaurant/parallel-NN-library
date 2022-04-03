#include "Model.h"

#include <Eigen/Dense>

#include "layer.h"

Eigen::VectorXd Model::forwardPass(Eigen::VectorXd input) {
    Eigen::VectorXd outputOfPrevLayer = input;
    for (int i = 0; i < layers.size(); ++i) {
        outputOfPrevLayer = layers[i].forwardPropagate(outputOfPrevLayer);
    }
    return outputOfPrevLayer;
}