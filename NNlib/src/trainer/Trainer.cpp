//
// Created by vityha on 24.05.22.
//

#include "trainer/Trainer.h"
#include "utils/random.h"

void Trainer::trainDataset(MatrixType features, MatrixType labels, int numberIterations) {
    // get indices of FC layers
    std::vector<size_t> FCLayersIdx = std::move(model->getFCLayersIndices());

    std::vector<MatrixType> batchesFeatures = splitMatrixInBatches(features, batchSize),
            batchesLabels = splitMatrixInBatches(labels, batchSize);
    size_t numBatches = batchesFeatures.size();

    // n times
    for (int s = 0; s < numberIterations; ++s) {
        // generate random number
        int batchNumber = generateRandInt(0, numBatches-1);

        // do calculation
        std::vector<MatrixType> layersGradients(
                std::move(model->calculateBatchGradients(batchesFeatures[batchNumber], batchesLabels[batchNumber])));

        // update
        for (size_t i = 0; i < FCLayersIdx.size(); ++i) {
            size_t layerIdx = FCLayersIdx[i];
            model->updLayerWeights(layerIdx, layersGradients[i], learningRate);
        }
    }
}

// Additional operations

std::vector<MatrixType> Trainer::splitMatrixInBatches(MatrixType &mat, size_t batchSize) {
    size_t quotient = mat.cols() / batchSize;
    size_t remainder = mat.cols() % batchSize;
    size_t numBatches = quotient + (remainder > 0);

    std::vector<MatrixType> batchesMatrices;
    batchesMatrices.reserve(numBatches);
    size_t startCol, endCol = 0;
    for (size_t i = 0; i < numBatches; i++) {
        startCol = endCol;
        endCol += quotient + (i < remainder);

        std::vector<size_t> rowsIdx(mat.rows());
        std::iota(rowsIdx.begin(), rowsIdx.end(), 0);
        std::vector<size_t> colsIdx(endCol - startCol);
        std::iota(colsIdx.begin(), colsIdx.end(), startCol);

        batchesMatrices.emplace_back(std::move(mat(rowsIdx, colsIdx)));
    }

    return batchesMatrices;
}