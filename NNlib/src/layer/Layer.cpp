//
// Created by vityha on 29.03.22.
//

#include "layer/Layer.h"

// Initialization

Layer::Layer(int nodesNumber, LayerType layerType) : nodesNumber{nodesNumber}, layerType{layerType} {}

// Structure

void Layer::setNodesNumber(int number) {
    nodesNumber = number;
}

int Layer::getNodesNumber() const {
    return nodesNumber;
}

Layer::LayerType Layer::getLayerType() const {
    return layerType;
}
