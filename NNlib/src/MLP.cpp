#include "MLP.h"
#include "layer.h"

// Structure

void MLP::addLayer(int numNodes, Layer::ActivationFunction activationFunction) {
    int layerInputsNumber;
    if (layers.empty()) {
        layerInputsNumber = numInputNodes;
    } else {
        layerInputsNumber = layers[layers.size() - 1].getNodesNumber();
    }
    Layer newLayer{layerInputsNumber, numNodes, activationFunction};
    layers.push_back(std::move(newLayer));
}

// Training

void MLP::trainExample(Eigen::VectorXd features, Eigen::VectorXd labels) {
    Eigen::VectorXd layersOutputs = forwardPass(features);
    Eigen::VectorXd topDerivatives = fullLoss.backPropagate(layersOutputs, labels);
    for (int i=layers.size()-1;i>=0;i--){
        topDerivatives = layers[i].backPropagate(topDerivatives, alpha);
    }
}

// Printing

void MLP::printMLP(){
    std::cout << "input nodes number = " << numInputNodes
    << "\n" << "---------------------------" << "\n"
    << "layers (" << layers.size() << ") :" << "\n";
    for (int i = 0; i < layers.size(); i++){
        std::cout << i + 1 << "\n"
        << "---------------------------" << "\n"
        << "weights = " << "\n"
        << layers[i].getWeights() << "\n"
        << "type = " << layers[i].getActivationFunctionName() << "\n";
    }
}

// copied from Model by Bohdan Mahometa

Eigen::VectorXd MLP::forwardPass(Eigen::VectorXd input) {
    Eigen::VectorXd outputOfPrevLayer = input;
    for (int i = 0; i < layers.size(); ++i) {
        outputOfPrevLayer = layers[i].forwardPropagate(outputOfPrevLayer);
    }
    return outputOfPrevLayer;
}

void MLP::setParameters(double newAlpha){
    alpha = newAlpha;
}
