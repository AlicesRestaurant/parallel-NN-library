//
// Created by vityha on 29.03.22.
//

#include "layer/Layer.h"

#include <cstddef>

// Initialization

Layer::Layer(size_t nodesNumber, LayerType layerType) : nodesNumber{nodesNumber}, layerType{layerType} {}

// Structure

int Layer::getNodesNumber() const {
    return nodesNumber;
}

Layer::LayerType Layer::getLayerType() const {
    return layerType;
}
