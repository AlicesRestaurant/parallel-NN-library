//
// Created by vityha on 24.05.22.
//

#include "trainer/DistributedTrainer.h"
#include "matrix/MatrixType.h"
#include "utils/random.h"

//#define SEQ
//#define DEBUG

void DistributedTrainer::trainDataset(MatrixType &features, MatrixType &labels, int numberIterations) {
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
        for (size_t p = 0; p < numProcessors; ++p) {
            std::vector<MatrixType> layersGradients = std::move(
                    model->calculateBatchLayersGradients(batchesFeatures[batchesNumbers[p]],
                                                         batchesLabels[batchesNumbers[p]]));

            allGradients.emplace_back(std::move(layersGradients));
        }

        std::vector<MatrixType> meanLayersGradients;
        meanLayersGradients.reserve(FCLayersIdx.size());

        std::vector<MatrixType> layerGradients(numProcessors);
        for (size_t numLayer = 0; numLayer < FCLayersIdx.size(); ++numLayer) {
            for (size_t p = 0; p < numProcessors; ++p) {
                layerGradients[p] = allGradients[p][numLayer];
            }

            // changed init to really empty matrix
            MatrixType emptyMatrix = MatrixType::Constant(layerGradients[0].rows(), layerGradients[0].cols(), 0);
            meanLayersGradients.emplace_back(std::reduce(layerGradients.begin(), layerGradients.end(),
                                                         emptyMatrix, std::plus<>()));

            // find mean of processors layer gradients
            meanLayersGradients[numLayer] = meanLayersGradients[numLayer] / (double) numProcessors;
        }

        // replace loop with model function call
        // why don't work
        // model->updLayersWeights(meanLayersGradients, learningRate);

        for (size_t numLayer = 0; numLayer < FCLayersIdx.size(); ++numLayer) {
            size_t layerIdx = FCLayersIdx[numLayer];
            model->updLayerWeights(layerIdx, meanLayersGradients[numLayer], learningRate);
        }

//        std::cout << s << "\n";
    }

#else

    if (comm->rank() == 0) {

        // divide in equal batches => convert in vector of matrices
        std::vector<MatrixType> batchesFeatures = splitMatrixInBatches(features, batchSize),
                batchesLabels = splitMatrixInBatches(labels, batchSize);
        size_t numBatches = batchesFeatures.size();

        std::vector<int> batchesNumbers;
        batchesNumbers.reserve(numProcessors);

        // n times
        for (int numIter = 0; numIter < numberIterations; ++numIter) {
            // generate k random numbers in array indices interval
            batchesNumbers = std::move(generateRandIntSeq(0, numBatches - 1, numProcessors));

#ifdef DEBUG

            std::cout << "iter: " << numIter << '\n'
            << "--------------------" << '\n'
            << "batches numbers: " << '\n';
            for (auto &num: batchesNumbers) {
                std::cout << num << "\n";
            }
            std::cout << "--------------------" << '\n';

#endif

            // send asynchronously random arrays to processors
            // wait for sends
            std::vector<mpi::request> reqs(2 * (numProcessors - 1));
            for (size_t p = 1; p < numProcessors; ++p) {
                reqs[2 * (p - 1)] = comm->isend(p, 0, batchesFeatures[batchesNumbers[p]]);
                reqs[2 * (p - 1) + 1] = comm->isend(p, 0, batchesLabels[batchesNumbers[p]]);
            }
            mpi::wait_all(reqs.begin(), reqs.end());

            // do own calculation
            // pick index for 0 rank from batches numbers
            std::vector<MatrixType> ownLayersGradients(
                    std::move(model->calculateBatchLayersGradients(batchesFeatures[batchesNumbers[0]],
                                                                   batchesLabels[batchesNumbers[0]])));

            // recv async result -> sync with gather
            // wait for recvs
            std::vector<MatrixType> meanLayersGradients;
            meanLayersGradients.reserve(FCLayersIdx.size());
            // need to fill layerGradients to serialize incoming MatrixD
            std::vector<MatrixType> layerGradients(numProcessors, MatrixType{0, 0});
            for (size_t numLayer = 0; numLayer < FCLayersIdx.size(); ++numLayer) {
                mpi::gather(*comm, ownLayersGradients[numLayer], layerGradients, 0);

#ifdef DEBUG

                if (numLayer == 0 && numIter == 1) {
                    std::cout << layerGradients.size() << '\n';
                    std::cout << ownLayersGradients[numLayer] << '\n';
                    for (auto &layerGradient: layerGradients) {
                        std::cout << layerGradient << '\n';
                    }
                }

#endif

                MatrixType emptyMatrix = MatrixType::Constant(layerGradients[0].rows(), layerGradients[0].cols(), 0);
                meanLayersGradients.emplace_back(std::reduce(layerGradients.begin(), layerGradients.end(),
                                                             emptyMatrix, std::plus<>()));
                meanLayersGradients[numLayer] = meanLayersGradients[numLayer] / (double) numProcessors;

#ifdef DEBUG

                if (numLayer == 0 && numIter == 1) {
                    std::cout << meanLayersGradients[numLayer] << '\n';
                }

#endif

            }

            // update and broadcast
            for (size_t numLayer = 0; numLayer < FCLayersIdx.size(); ++numLayer) {
                size_t layerIdx = FCLayersIdx[numLayer];
                MatrixType updatedWeights = model->updLayerWeights(layerIdx, meanLayersGradients[numLayer],
                                                                   learningRate);
                mpi::broadcast(*comm, updatedWeights, 0);
            }
        }
    } else {
        // n times
        for (int numIter = 0; numIter < numberIterations; ++numIter) {
            // recv synchronously batch
            // init with 0 to init MatrixD::ViewData
            // otherwise, error
            MatrixType batchFeatures{0, 0}, batchLabels{0, 0};
            comm->recv(0, 0, batchFeatures);
            comm->recv(0, 0, batchLabels);

            // train batch
            std::vector<MatrixType> ownGradients(
                    std::move(model->calculateBatchLayersGradients(batchFeatures, batchLabels)));

            // send asynchronously result
            std::vector<MatrixType> dummy;
            for (size_t numLayer = 0; numLayer < FCLayersIdx.size(); ++numLayer) {
                mpi::gather(*comm, ownGradients[numLayer], dummy, 0);
            }

            // recv updated weights from root and update
            for (size_t numLayer = 0; numLayer < FCLayersIdx.size(); ++numLayer) {
                MatrixType updatedWeights{0, 0};
                mpi::broadcast(*comm, updatedWeights, 0);
                size_t layerIdx = FCLayersIdx[numLayer];
                model->setLayerWeights(layerIdx, updatedWeights);
            }
        }
    }

#endif

}