//
// Created by vityha on 24.05.22.
//

#include "trainer/DistributedTrainer.h"
#include "matrix/MatrixType.h"
#include "utils/random.h"

#define SEQ

void DistributedTrainer::trainDataset(MatrixType features, MatrixType labels, int numberIterations) {
    std::vector<size_t> FCLayersIdx = std::move(model->getFCLayersIndices());

#ifdef SEQ

    std::vector<MatrixType> batchesFeatures = splitMatrixInBatches(features, batchSize),
            batchesLabels = splitMatrixInBatches(labels, batchSize);
    size_t numBatches = batchesFeatures.size();

    std::vector<int> batchesNumbers;
    batchesNumbers.reserve(numProcessors);

    for (int s = 0; s < numberIterations; ++s) {
        // subtract 1 because generates including max
        batchesNumbers = std::move(generateRandIntSeq(0, numBatches - 1, numProcessors));

        std::vector<std::vector<MatrixType>> allGradients;
        allGradients.reserve(numProcessors);
        for (size_t i = 0; i < numProcessors; ++i) {
            std::vector<MatrixType> batchGradients = std::move(model->calculateBatchGradients(batchesFeatures[batchesNumbers[i]],
                                                                                              batchesLabels[batchesNumbers[i]]));

            allGradients.emplace_back(std::move(batchGradients));
        }

        std::vector<MatrixType> totalGradients;
        totalGradients.reserve(FCLayersIdx.size());
        std::vector<MatrixType> layerGradients(numProcessors);
        for (size_t i = 0; i < FCLayersIdx.size(); ++i) {
            for (size_t j = 0; j < numProcessors; ++j) {
                layerGradients[j] = allGradients[j][i];
            }
            totalGradients.emplace_back(std::reduce(layerGradients.begin(), layerGradients.end(),
                                                    MatrixType{layerGradients[0].rows(),
                                                               layerGradients[0].cols()}, std::plus<>()));
            // find mean of processors layer gradients
            totalGradients[i] = totalGradients[i] / (double) numProcessors;
        }

        // replace loop with model function call
        // why don't work
        // model->updLayersWeights(totalGradients, learningRate);

        for (size_t i = 0; i < FCLayersIdx.size(); ++i) {
            size_t layerIdx = FCLayersIdx[i];
            model->updLayerWeights(layerIdx, totalGradients[i], learningRate);
        }
    }

#else

    if (comm->rank() == 0) {
        // divide in equal batches => convert in array of matrices
        std::vector<MatrixType> batchesFeatures = splitMatrixInBatches(features, batchSize),
                batchesLabels = splitMatrixInBatches(labels, batchSize);
        size_t numBatches = batchesFeatures.size();

        std::vector<int> batchesNumbers;
        batchesNumbers.reserve(numProcessors);

        // 100 times
        for (int s = 0; s < numberIterations; ++s) {
            // generate 6 random numbers in array indices interval
            batchesNumbers = std::move(generateRandIntSeq(0, numBatches, numProcessors));

            // send asynchronously random arrays to processors
            // wait for sends
            std::vector<mpi::request> reqs(2 * (numProcessors - 1));
            for (size_t k = 1; k < numProcessors; ++k) {
                reqs[2 * (k - 1)] = comm->isend(k, 0, batchesFeatures[batchesNumbers[k]]);
                reqs[2 * (k - 1) + 1] = comm->isend(k, 0, batchesLabels[batchesNumbers[k]]);
            }
            mpi::wait_all(reqs.begin(), reqs.end());

            // do own calculation
            std::vector<MatrixType> ownGradients(
                    std::move(model->calculateBatchGradients(batchesFeatures[0], batchesLabels[0])));

            // recv async result -> sync with gather
            // wait for recvs
            std::vector<MatrixType> totalGradients;
            totalGradients.reserve(FCLayersIdx.size());
            std::vector<MatrixType> layerGradients;
            layerGradients.reserve(numProcessors);
            for (size_t i = 0; i < FCLayersIdx.size(); ++i) {
                mpi::gather(*comm, ownGradients[i], layerGradients, 0);
                totalGradients.emplace_back(std::reduce(layerGradients.begin(), layerGradients.end(),
                                                        MatrixType{layerGradients[0].rows(),
                                                                   layerGradients[0].cols()}, std::plus<>()));
            }

            // broadcast and update
            for (size_t i = 0; i < FCLayersIdx.size(); ++i) {
                size_t layerIdx = FCLayersIdx[i];
                MatrixType updatedWeights = model->updLayerWeights(layerIdx, totalGradients[i], learningRate);
                mpi::broadcast(*comm, updatedWeights, 0);
            }
        }
    } else {
        // 100 times
        for (int s = 0; s < numberIterations; ++s) {
            // recv synchronously batch
            MatrixType batchFeatures, batchLabels;
            comm->recv(0, 0, batchFeatures);
            comm->recv(0, 0, batchLabels);

            // train batch
            std::vector<MatrixType> ownGradients(std::move(model->calculateBatchGradients(batchFeatures, batchLabels)));

            // send asynchronously result
            std::vector<MatrixType> dummy;
            for (size_t i = 0; i < FCLayersIdx.size(); ++i) {
                mpi::gather(*comm, ownGradients[i], dummy, 0);
            }

            // recv updated weights
            for (size_t i = 0; i < FCLayersIdx.size(); ++i) {
                MatrixType updatedWeights;
                mpi::broadcast(*comm, updatedWeights, 0);
                size_t layerIdx = FCLayersIdx[i];
                model->setLayerWeights(layerIdx, updatedWeights);
            }
        }
    }

#endif

}