#include "Model.h"
#include "layer/Layer.h"
#include "utils/random.h"

#include <cstddef>
#include <iostream>
#include <vector>
#include <algorithm>


#define DEBUG

// Layer operations

std::vector<size_t> Model::getFCLayersIndices() {
    std::vector<size_t> FCLayersIdx;
    for (size_t i = 0; i < layerPtrs.size(); ++i) {
        if (layerPtrs[i]->getLayerType() == Layer::LayerType::FC) {
            FCLayersIdx.emplace_back(i);
        }
    }
    return FCLayersIdx;
}

MatrixType Model::updLayerWeights(size_t layerIdx, const MatrixType &layerWeightsGradients, double alpha) {
    layerPtrs[layerIdx]->updateWeights(layerPtrs[layerIdx]->getWeights() -
                                       alpha * layerWeightsGradients);
    return layerPtrs[layerIdx]->getWeights();
}

MatrixType Model::updLayersWeights(std::vector<MatrixType> &layersWeightsGradients, double alpha) {
    std::vector<size_t> FCLayersIdx = getFCLayersIndices();
    for (size_t i = 0; i < FCLayersIdx.size(); ++i) {
        size_t layerIdx = FCLayersIdx[i];
        updLayerWeights(layerIdx, layersWeightsGradients[i], alpha);
    }
}

void Model::setLayerWeights(size_t layerIdx, const MatrixType &newWeights) {
    layerPtrs[layerIdx]->updateWeights(newWeights);
}

MatrixType Model::getLayerWeights(size_t layerIdx) {
    return layerPtrs[layerIdx]->getWeights();
}

// Forward

MatrixType Model::forwardPass(const MatrixType &input) {
    MatrixType outputOfPrevLayer = input;
    for (size_t i = 0; i < layerPtrs.size(); ++i) {
        outputOfPrevLayer = layerPtrs[i]->forwardPropagate(outputOfPrevLayer);
    }
    return outputOfPrevLayer;
}

double Model::calcLoss(const MatrixType &predictions, const MatrixType &groundTruths) {
    return lossFunctionPtr->forwardPropagate(predictions, groundTruths);
}


// Training
void Model::trainBatch(const MatrixType &features, const MatrixType &labels, double alpha) {
    MatrixType layersOutputs = forwardPass(features);
    MatrixType topDerivatives =
            lossFunctionPtr->backPropagate(layersOutputs, labels);
    for (size_t i = layerPtrs.size() - 1; i > 0; --i) {
        MatrixType tempDerivatives = layerPtrs[i]->calculateGradientsWrtInputs(topDerivatives);
        if (layerPtrs[i]->getLayerType() == Layer::LayerType::FC) {
            layerPtrs[i]->updateWeights(layerPtrs[i]->getWeights() -
                                            alpha * layerPtrs[i]->calculateGradientsWrtWeights(topDerivatives));
        }
        topDerivatives = std::move(tempDerivatives);
    }
    if (layerPtrs.front()->getLayerType() == Layer::LayerType::FC) {
        layerPtrs.front()->updateWeights(layerPtrs.front()->getWeights() -
                                    alpha * layerPtrs.front()->calculateGradientsWrtWeights(topDerivatives));
    }
}

std::vector<MatrixType> Model::calculateBatchLayersGradients(const MatrixType &features, const MatrixType &labels) {
    std::vector<MatrixType> batchGradients;

    MatrixType layersOutputs = forwardPass(features);
    MatrixType topDerivatives =
            lossFunctionPtr->backPropagate(layersOutputs, labels);
    if (layerPtrs.back()->getLayerType() == Layer::LayerType::FC) {
        batchGradients.emplace_back(layerPtrs.back()->calculateGradientsWrtWeights(topDerivatives));
    }
    for (size_t i = layerPtrs.size() - 1; i > 0; --i) {
        topDerivatives = layerPtrs[i]->calculateGradientsWrtInputs(topDerivatives);
        if (layerPtrs[i - 1]->getLayerType() == Layer::LayerType::FC) {
            batchGradients.emplace_back(layerPtrs[i - 1]->calculateGradientsWrtWeights(topDerivatives));
        }
    }

    // reverse due to reverse order of gradients in backprop
    //  relatively to layer indices
    std::reverse(batchGradients.begin(), batchGradients.end());

    return batchGradients;
}



// Printing

std::ostream &operator<<(std::ostream &os, const Model &model) {
    os << "MLP {\n\tnumInputNodes = " << model.numInputNodes << ",\n";
    for (size_t i = 0; i < model.layerPtrs.size(); ++i) {
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



