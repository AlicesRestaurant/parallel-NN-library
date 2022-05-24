#ifndef NNLIB_AND_TEST_EXAMPLE_MODEL_H
#define NNLIB_AND_TEST_EXAMPLE_MODEL_H

#include "layer/Layer.h"
#include "lossfunction/LossFunction.h"

#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cstddef>

class Model {
public:
    Model(size_t numInputNodes, const std::shared_ptr<LossFunction>& lossFunctionPtr) : numInputNodes{numInputNodes},
                                                                                   lossFunctionPtr{lossFunctionPtr}
    {}

    MatrixType forwardPass(MatrixType input);
    double calcLoss(const MatrixType& predictions, const MatrixType& groundTruths);

    template<class LayerType, class... Args>
    void addLayer(Args... args) {
        layerPtrs.push_back(std::make_shared<LayerType>(args...));
    }

    void trainBatch(MatrixType features, MatrixType labels, double alpha);

    friend std::ostream& operator<<(std::ostream &os, const Model &mlp);
protected:
    std::vector<std::shared_ptr<Layer>> layerPtrs;

    std::shared_ptr<LossFunction> lossFunctionPtr;

    const size_t numInputNodes;
};


#endif //NNLIB_AND_TEST_EXAMPLE_MODEL_H
