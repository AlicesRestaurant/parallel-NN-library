#include "Model.h"
#include "layer/Layer.h"

// Training

void Model::trainExample(Eigen::VectorXd features, Eigen::VectorXd labels) {
    Eigen::VectorXd layersOutputs = forwardPass(features);
    Eigen::VectorXd topDerivatives =
            lossFunctionPtr->backPropagate(layersOutputs, labels);
    for (int i = layerPtrs.size() - 1; i >= 0; i--){
        topDerivatives = layerPtrs[i]->calculateGradientsWrtInputs(topDerivatives);
    }
}

void Model::trainBatch(Eigen::MatrixXd features, Eigen::MatrixXd labels, double alpha) {
    Eigen::VectorXd layersOutputs = forwardPass(features);
    Eigen::VectorXd topDerivatives =
            lossFunctionPtr->backPropagate(layersOutputs, labels);
    layerPtrs.back()->updateWeights(layerPtrs.back()->getWeights() - alpha * layerPtrs.back()->calculateGradientsWrtWeights(topDerivatives));
    for (int i = layerPtrs.size() - 1; i > 0; --i) {
        topDerivatives = layerPtrs[i]->calculateGradientsWrtInputs(topDerivatives);
        layerPtrs.back()->updateWeights(layerPtrs[i-1]->getWeights() - alpha * layerPtrs[i-1]->calculateGradientsWrtWeights(topDerivatives));
    }
}

// Printing

std::ostream& operator<<(std::ostream &os, const Model &model) {
    os << "MLP {\n\tnumInputNodes = " << model.numInputNodes << ",\n";
    for (int i = 0; i < model.layerPtrs.size(); ++i){
        os << "\tlayer " << i + 1 << " = layer{\n"
           << "\t\tweights = \n";
        Eigen::MatrixXd weights = model.layerPtrs[i]->getWeights();
        for (int rowIdx = 0; rowIdx < weights.rows(); ++rowIdx) {
            os << "\t\t\t" << weights.row(rowIdx) << '\n';
        }
        os << "\t}\n";
    }
    os << "}\n";
    return os;
}

Eigen::VectorXd Model::forwardPass(Eigen::VectorXd input) {
    Eigen::VectorXd outputOfPrevLayer = input;
    for (int i = 0; i < layerPtrs.size(); ++i) {
        outputOfPrevLayer = layerPtrs[i]->forwardPropagate(outputOfPrevLayer);
    }
    return outputOfPrevLayer;
}
