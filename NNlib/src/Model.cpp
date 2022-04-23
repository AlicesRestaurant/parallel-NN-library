#include "Model.h"
#include "layer/Layer.h"

// Forward

Eigen::MatrixXd Model::forwardPass(Eigen::MatrixXd input) {
    Eigen::MatrixXd outputOfPrevLayer = input;
    for (int i = 0; i < layerPtrs.size(); ++i) {
        outputOfPrevLayer = layerPtrs[i]->forwardPropagate(outputOfPrevLayer);
    }
    return outputOfPrevLayer;
}

double Model::calcLoss(Eigen::MatrixXd input, const Eigen::MatrixXd& predictions, const Eigen::MatrixXd& groundTruths) {
    return lossFunctionPtr->forwardPropagate(predictions, groundTruths);
}


// Training

void Model::trainExample(Eigen::VectorXd features, Eigen::VectorXd label, double alpha) {
    Model::trainBatch(features, label, alpha);
}

void Model::trainBatch(Eigen::MatrixXd features, Eigen::MatrixXd labels, double alpha) {
    Eigen::MatrixXd layersOutputs = forwardPass(features);
    Eigen::MatrixXd topDerivatives =
            lossFunctionPtr->backPropagate(layersOutputs, labels);
    if (layerPtrs.back()->getLayerType() == Layer::LayerType::FC) {
        layerPtrs.back()->updateWeights(layerPtrs.back()->getWeights() -
                                        alpha * layerPtrs.back()->calculateGradientsWrtWeights(topDerivatives));
    }
    for (int i = layerPtrs.size() - 1; i > 0; --i) {
        topDerivatives = layerPtrs[i]->calculateGradientsWrtInputs(topDerivatives);
        if (layerPtrs[i - 1]->getLayerType() == Layer::LayerType::FC) {
            layerPtrs[i - 1]->updateWeights(layerPtrs[i - 1]->getWeights() -
                                            alpha * layerPtrs[i - 1]->calculateGradientsWrtWeights(topDerivatives));
        }
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
