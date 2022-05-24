#include "Model.h"
#include "layer/Layer.h"

#include <cstddef>
#include <iostream>

#define DEBUG

// Forward

MatrixType Model::forwardPass(MatrixType input) {
    MatrixType outputOfPrevLayer = input;
    for (size_t i = 0; i < layerPtrs.size(); ++i) {
        outputOfPrevLayer = layerPtrs[i]->forwardPropagate(outputOfPrevLayer);
    }
    return outputOfPrevLayer;
}

double Model::calcLoss(const MatrixType& predictions, const MatrixType& groundTruths) {
    return lossFunctionPtr->forwardPropagate(predictions, groundTruths);
}


// Training

void Model::trainBatch(MatrixType features, MatrixType labels, double alpha) {
    MatrixType layersOutputs = forwardPass(features);
    MatrixType topDerivatives =
            lossFunctionPtr->backPropagate(layersOutputs, labels);
    if (layerPtrs.back()->getLayerType() == Layer::LayerType::FC) {
        layerPtrs.back()->updateWeights(layerPtrs.back()->getWeights() -
                                        alpha * layerPtrs.back()->calculateGradientsWrtWeights(topDerivatives));
    }
    for (size_t i = layerPtrs.size() - 1; i > 0; --i) {
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
    for (size_t i = 0; i < model.layerPtrs.size(); ++i){
        os << "\tlayer " << i + 1 << " = layer{\n"
           << "\t\tweights = \n";
        MatrixType weights = model.layerPtrs[i]->getWeights();
        for (size_t rowIdx = 0; rowIdx < weights.rows(); ++rowIdx) {
            os << "\t\t\t" << weights.row(rowIdx) << '\n';
        }
        os << "\t}\n";
    }
    os << "}\n";
    return os;
}
