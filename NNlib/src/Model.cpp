#include "Model.h"
#include "layer/Layer.h"

#include <cstddef>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/mpi.hpp>

namespace mpi = boost::mpi;

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
        MatrixType tempDerivatives = layerPtrs.back()->calculateGradientsWrtInputs(topDerivatives);
        layerPtrs.back()->updateWeights(layerPtrs.back()->getWeights() -
                                        alpha * layerPtrs.back()->calculateGradientsWrtWeights(topDerivatives));
        topDerivatives = tempDerivatives;
    } else {
        topDerivatives = layerPtrs.back()->calculateGradientsWrtInputs(topDerivatives);
    }
    for (size_t i = layerPtrs.size() - 2; i > 0; --i) {
        MatrixType tempDerivatives = layerPtrs[i]->calculateGradientsWrtInputs(topDerivatives);
        if (layerPtrs[i]->getLayerType() == Layer::LayerType::FC) {
            layerPtrs[i]->updateWeights(layerPtrs[i]->getWeights() -
                                            alpha * layerPtrs[i]->calculateGradientsWrtWeights(topDerivatives));
        }
        topDerivatives = tempDerivatives;
    }
}

//Eigen::MatrixXd forwardPropagateBatch

int generateRandInt(int min, int max) {
    std::random_device r;
    std::mt19937 gen(r());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

std::vector<int> generateRandIntSeq(int min, int max, int length) {
    std::vector<int> batchesNumbers;
    batchesNumbers.reserve(length);

    while (batchesNumbers.size() != length) {
        batchesNumbers.emplace_back(generateRandInt(min, max)); // create new random number
        std::sort(begin(batchesNumbers), end(batchesNumbers)); // sort before call to unique
        auto last = std::unique(begin(batchesNumbers), end(batchesNumbers));
        batchesNumbers.erase(last, end(batchesNumbers));       // erase duplicates
    }

    return batchesNumbers;
}

std::vector<Eigen::MatrixXd> Model::calculateBatchGradients(Eigen::MatrixXd features, Eigen::MatrixXd labels) {
    std::vector<Eigen::MatrixXd> batchGradients;
    Eigen::MatrixXd layersOutputs = forwardPass(features);
    Eigen::MatrixXd topDerivatives =
            lossFunctionPtr->backPropagate(layersOutputs, labels);
    batchGradients.emplace_back(topDerivatives);
    for (size_t i = layerPtrs.size() - 1; i > 0; --i) {
        topDerivatives = layerPtrs[i]->calculateGradientsWrtInputs(topDerivatives);
        batchGradients.emplace_back(topDerivatives);
    }
}

void Model::trainDataset(Eigen::MatrixXd features, Eigen::MatrixXd labels, int batchSize,
                         int numberIterations, double alpha) {
    // get indices of FC layers
    std::vector<size_t> FCLayersIdx;
    for (size_t i = 0; i < layerPtrs.size(); ++i) {
        if (layerPtrs[i]->getLayerType() == Layer::LayerType::FC) {
            FCLayersIdx.emplace_back(i);
        }
    }

    // TODO: implement not MPI
    if (!getMPIExecution()) {
        return;
    }

    if (comm.rank() == 0) {
        // divide in equal batches => convert in array of matrices
        size_t quotient = features.cols() / batchSize;
        size_t remainder = features.cols() % batchSize;
        size_t numBatches = quotient + (remainder > 0);

        Eigen::MatrixXd batchesFeatures[numBatches], batchesLabels[numBatches];
        size_t startCol, endCol = 0;
        for (size_t i = 0; i < numBatches; i++) {
            startCol = endCol;
            endCol += quotient + (i < remainder);
            batchesFeatures[i] = features(Eigen::placeholders::all, Eigen::seq(startCol, endCol - 1));
            batchesLabels[i] = features(Eigen::placeholders::all, Eigen::seq(startCol, endCol - 1));
        }

        // generate 6 random numbers in array indices interval
        std::vector<int> batchesNumbers;
        batchesNumbers.reserve(numProcessors);
        batchesNumbers = std::move(generateRandIntSeq(0, numBatches, numProcessors));

        // 100 times
        for (int s = 0; s < numberIterations; ++s) {
            // send asynchronously random arrays to processors
            // wait for sends
            mpi::request reqs[2 * (numProcessors - 1)];
            for (size_t k = 1; k < numProcessors; ++k) {
                reqs[2 * (k - 1)] = comm.isend(k, 0, batchesFeatures[batchesNumbers[k]]);
                reqs[2 * (k - 1) + 1] = comm.isend(k, 0, batchesLabels[batchesNumbers[k]]);
            }
            mpi::wait_all(reqs, reqs + 2 * (numProcessors - 1));

            // do own calculation
            std::vector<Eigen::MatrixXd> ownGradients(
                    std::move(calculateBatchGradients(batchesFeatures[0], batchesLabels[0])));

            // recv async result -> sync with gather
            // wait for recvs
            std::vector<Eigen::MatrixXd> totalGradients;
            totalGradients.reserve(FCLayersIdx.size());
            std::vector<Eigen::MatrixXd> layerGradients;
            layerGradients.reserve(numProcessors);
            for (size_t i = 0; i < FCLayersIdx.size(); ++i) {
                mpi::gather(comm, ownGradients[i], layerGradients, 0);
                totalGradients.emplace_back(std::reduce(layerGradients.begin(), layerGradients.end(),
                                                        Eigen::MatrixXd{layerGradients[0].rows(),
                                                                        layerGradients[0].cols()}, std::plus<>()));
            }

            // broadcast and update
            for (size_t i = 0; i < FCLayersIdx.size(); ++i) {
                size_t layerIdx = FCLayersIdx[i];
                mpi::broadcast(comm, totalGradients[i], 0);
                layerPtrs[layerIdx]->updateWeights(layerPtrs[layerIdx]->getWeights() -
                                                   alpha * totalGradients[i]);
            }
        }
    } else {
        // 100 times
        for (int s = 0; s < numberIterations; ++s) {
            // recv synchronously batch
            Eigen::MatrixXd batchFeatures, batchLabels;
            comm.recv(0, 0, batchFeatures);
            comm.recv(0, 0, batchLabels);

            // train batch
            std::vector<Eigen::MatrixXd> ownGradients(std::move(calculateBatchGradients(batchFeatures, batchLabels)));

            // send asynchronously result
            std::vector<Eigen::MatrixXd> dummy;
            for (size_t i = 0; i < FCLayersIdx.size(); ++i) {
                mpi::gather(comm, ownGradients[i], dummy, 0);
            }

            // recv updated weights
            for (size_t i = 0; i < FCLayersIdx.size(); ++i) {
                Eigen::MatrixXd updatedWeights;
                mpi::broadcast(comm, updatedWeights, 0);
                size_t layerIdx = FCLayersIdx[i];
                layerPtrs[layerIdx]->updateWeights(updatedWeights);
            }
        }
    }
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
