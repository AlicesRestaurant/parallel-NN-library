//
// Created by vityha on 24.05.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_TRAINER_H
#define NNLIB_AND_TEST_EXAMPLE_TRAINER_H

#include <utility>

#include "matrix/MatrixType.h"
#include "Model.h"

class Trainer {
public:
    Trainer(std::shared_ptr<Model> model, size_t batchSize, double learningRate) : model(std::move(model)), learningRate(learningRate), batchSize(batchSize) {}

    virtual void trainDataset(MatrixType features, MatrixType labels, int numberIterations);

protected:
    std::shared_ptr<Model> model;
    double learningRate = 1;
    size_t batchSize = 1;

    std::vector<MatrixType> splitMatrixInBatches(MatrixType &mat, size_t batchSize);
};


#endif //NNLIB_AND_TEST_EXAMPLE_TRAINER_H
