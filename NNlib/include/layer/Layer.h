//
// Created by vityha on 29.03.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_LAYER_H
#define NNLIB_AND_TEST_EXAMPLE_LAYER_H

#include <matrix/MatrixType.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>

class Layer {
public:
    enum class LayerType {
        FC,
        Activation
    };

    explicit Layer(size_t nodesNumber, LayerType layerType);

    virtual MatrixType forwardPropagate(const MatrixType &bottomData) = 0;

    virtual MatrixType calculateGradientsWrtInputs(const MatrixType &topDerivatives) = 0;
    virtual MatrixType calculateGradientsWrtWeights(const MatrixType &topDerivatives) = 0;

    virtual void updateWeights(const MatrixType &newWeights) = 0;
    virtual MatrixType getWeights() const = 0;

    int getNodesNumber() const;

    LayerType getLayerType() const;

protected:
    MatrixType layerInputs;
    const size_t nodesNumber;
    const LayerType layerType;
};

#endif //NNLIB_AND_TEST_EXAMPLE_LAYER_H
