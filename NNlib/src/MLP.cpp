#include "MLP.h"

// Initialization

MLP::MLP(int numInputNodes) : numInputNodes{numInputNodes} {

}

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

void MLP::trainExample(Eigen::VectorXd features, Eigen::VectorXd labels){
    Eigen::VectorXd layersOutputs = forwardPass(features);
    lossFunction.forwardPropagate(layersOutputs);
    Eigen::VectorXd topDerivatives = lossFunction.backPropagate(labels);
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