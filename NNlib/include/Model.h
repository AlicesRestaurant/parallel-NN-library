#ifndef NNLIB_AND_TEST_EXAMPLE_MODEL_H
#define NNLIB_AND_TEST_EXAMPLE_MODEL_H


#include <vector>
#include <string>

#include <Eigen/Dense>

#include "layer.h"

class Model {
public:
    Eigen::VectorXd forwardPass(Eigen::VectorXd input);
protected:
    std::vector<Layer> layers;
};


#endif //NNLIB_AND_TEST_EXAMPLE_MODEL_H
