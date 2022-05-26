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
    Model(size_t numInputNodes, const std::shared_ptr<LossFunction> &lossFunctionPtr) : numInputNodes{numInputNodes},
                                                                                        lossFunctionPtr{
                                                                                                lossFunctionPtr} {}

    MatrixType forwardPass(MatrixType input);

    double calcLoss(const MatrixType &predictions, const MatrixType &groundTruths);

    template<class LayerType, class... Args>
    void addLayer(Args... args) {
        layerPtrs.push_back(std::make_shared<LayerType>(args...));
    }

    std::vector<size_t> getFCLayersIndices();

    MatrixType updLayerWeights(size_t layerIdx, MatrixType &layerWeightsGradients, double alpha);
    MatrixType updLayersWeights(std::vector<MatrixType> &layersWeightsGradients, double alpha);
    void setLayerWeights(size_t layerIdx, MatrixType &newWeights);
    MatrixType getLayerWeights(size_t layerIdx);

    void trainBatch(const MatrixType &features, const MatrixType &labels, double alpha);

    std::vector<MatrixType> calculateBatchLayersGradients(MatrixType &features, MatrixType &labels);

    friend std::ostream &operator<<(std::ostream &os, const Model &mlp);

protected:
    std::vector<std::shared_ptr<Layer>> layerPtrs;

    std::shared_ptr<LossFunction> lossFunctionPtr;

    const size_t numInputNodes;



};



#endif //NNLIB_AND_TEST_EXAMPLE_MODEL_H