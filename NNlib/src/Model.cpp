#include "Model.h"
#include "layer/Layer.h"

// Structure

//void MLP::addLayer(int numNodes, Layer::ActivationFunction activationFunction) {
//    int layerInputsNumber;
//    if (layers.empty()) {
//        layerInputsNumber = numInputNodes;
//    } else {
//        layerInputsNumber = layers.back().getNodesNumber();
//    }
//    layers.emplace_back(layerInputsNumber, numNodes, activationFunction);
//}

// Training

void Model::trainExample(Eigen::VectorXd features, Eigen::VectorXd labels) {
    Eigen::VectorXd layersOutputs = forwardPass(features);
    Eigen::VectorXd topDerivatives =
            lossFunctionPtr->backPropagate(layersOutputs.transpose(), labels.transpose())
                    .transpose();
    for (int i = layers.size() - 1; i >= 0; i--){
        topDerivatives = layers[i].calculateGradientsWrtInputs(topDerivatives);
    }
}

// Printing

std::ostream& operator<<(std::ostream &os, const Model &model) {
    os << "MLP {\n\tnumInputNodes = " << model.numInputNodes << ",\n";
    for (int i = 0; i < model.layers.size(); ++i){
        os << "\tlayer " << i + 1 << " = layer{\n"
           << "\t\tweights = \n";
        Eigen::MatrixXd weights = model.layers[i].getWeights();
        for (int rowIdx = 0; rowIdx < weights.rows(); ++rowIdx) {
            os << "\t\t\t" << weights.row(rowIdx) << '\n';
        }
        os << "\t}\n";
    }
    os << "}\n";
    return os;
}

// copied from Model by Bohdan Mahometa

Eigen::VectorXd Model::forwardPass(Eigen::VectorXd input) {
    Eigen::VectorXd outputOfPrevLayer = input;
    for (int i = 0; i < layers.size(); ++i) {
        outputOfPrevLayer = layers[i].forwardPropagate(outputOfPrevLayer);
    }
    return outputOfPrevLayer;
}
