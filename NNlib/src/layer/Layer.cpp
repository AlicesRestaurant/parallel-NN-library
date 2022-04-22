//
// Created by vityha on 29.03.22.
//

#include "layer/Layer.h"

// Initialization

Layer::Layer(int nodesNumber, LayerType layerType) : nodesNumber{nodesNumber}, layerType{layerType} {}

// Structure

int Layer::getNodesNumber() const {
    return nodesNumber;
}

Layer::LayerType Layer::getLayerType() const {
    return layerType;
}
