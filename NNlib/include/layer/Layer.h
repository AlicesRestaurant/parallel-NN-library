//
// Created by vityha on 29.03.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_LAYER_H
#define NNLIB_AND_TEST_EXAMPLE_LAYER_H

#include <Eigen/Dense>
#include <cmath>

class Layer {
public:
    enum class LayerType {
        FC,
        Activation
    };

    explicit Layer(int nodesNumber, LayerType layerType);

    virtual Eigen::MatrixXd forwardPropagate(const Eigen::MatrixXd &bottomData) = 0;

    virtual Eigen::MatrixXd calculateGradientsWrtInputs(const Eigen::MatrixXd &topDerivatives) = 0;
    virtual Eigen::MatrixXd calculateGradientsWrtWeights(const Eigen::MatrixXd &topDerivatives) = 0;

    virtual void updateWeights(const Eigen::MatrixXd &newWeights) = 0;
    virtual Eigen::MatrixXd getWeights() const = 0;

    int getNodesNumber() const;

    LayerType getLayerType() const;

protected:
    Eigen::MatrixXd layerInputs;
    const int nodesNumber;
    const LayerType layerType;
};

#endif //NNLIB_AND_TEST_EXAMPLE_LAYER_H
