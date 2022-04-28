#include <Model.h>
#include <layer/FCLayer.h>
#include <layer/SigmoidActivationLayer.h>
#include <lossfunction/SoftMaxLossFunction.h>
#include "lossfunction/SVMLossFunction.h"
#include <Eigen/Core>

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdint> // for std::uint32_t and others
#include <memory>

using Eigen::MatrixXd;

void readImages(MatrixXd &outImages, const std::string &fname);
void readLabels(MatrixXd &outLabels, const std::string &fname);
void fitNN(const MatrixXd &trainImages, const MatrixXd &trainLabels, const MatrixXd &testImages, const MatrixXd &testLabels,
           unsigned long numIters, double alpha);
template <class T>
void endswap(T *objp)
{
    unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
    std::reverse(memp, memp + sizeof(T));
}

constexpr const char* prefix = "./data/mnist/";
constexpr const char* trainImagesFilename = "train-images.idx3-ubyte";
constexpr const char* trainLabelsFilename = "train-labels.idx1-ubyte";
constexpr const char* testImagesFilename = "t10k-images.idx3-ubyte";
constexpr const char* testLabelsFilename = "t10k-labels.idx1-ubyte";

constexpr unsigned long NUM_ITERS = 500ul;
constexpr double ALPHA = 1e-5;

int main() {
    MatrixXd trainImages, trainLabels, testImages, testLabels;
    std::string prefixStr{prefix};

    readImages(trainImages, prefixStr + trainImagesFilename);
    readLabels(trainLabels, prefixStr + trainLabelsFilename);
    readImages(testImages, prefixStr + testImagesFilename);
    readLabels(testLabels, prefixStr + testLabelsFilename);

    fitNN(trainImages, trainLabels,
          testImages, testLabels, NUM_ITERS, ALPHA);

    return 0;
}

void fitNN(const MatrixXd &trainImages, const MatrixXd &trainLabels, const MatrixXd &testImages, const MatrixXd &testLabels,
           unsigned long numIters, double alpha) {
    size_t numInputsNeurons = trainImages.rows();
    size_t numOutputNeurons = trainLabels.rows();
    Model model{numInputsNeurons, std::make_shared<SVMLossFunction>()};
    model.addLayer<FCLayer>(numOutputNeurons, numInputsNeurons, -1.0/255, 1.0/255);

    for (unsigned long i = 0; i < numIters; ++i) {
        if (i % 100ul == 0) {
            std::cout << "Loss on train:\t" << model.calcLoss(model.forwardPass(trainImages), trainLabels) << '\n';
            std::cout << "Loss on test:\t" << model.calcLoss(model.forwardPass(testImages), testLabels) << '\n' << std::endl;
        }
        model.trainBatch(trainImages, trainLabels, alpha);
    }

    auto predicted = model.forwardPass(testImages);
    size_t numCorrect = 0;
    for (size_t colIdx = 0; colIdx < testImages.cols(); ++colIdx) {
        size_t idx;
        predicted.col(colIdx).maxCoeff(&idx);
        if (testLabels(idx, colIdx) == 1) {
            ++numCorrect;
        }
    }
    std::cout << "Error Rate: " << 100.0 * (testImages.cols() - numCorrect) / testImages.cols() << '%' <<  std::endl;
    std::cout << "Model: " << model << std::endl;
}

void readImages(MatrixXd &outImages, const std::string &fname) {
    std::ifstream ifs(fname, std::ios::out | std::ios::binary);

    if (!ifs) {
        std::cerr << "Couldn't open file " + fname << std::endl;
        exit(EXIT_FAILURE);
    }

    std::uint32_t magicNumber;
    ifs.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    endswap(&magicNumber);
    assert(magicNumber == 2051);

    std::uint32_t numImages;
    ifs.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    endswap(&numImages);

    std::uint32_t numRows, numCols;
    ifs.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    ifs.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
    endswap(&numRows);
    endswap(&numCols);

    outImages.resize(numRows * numCols, numImages);
    for (std::uint_fast32_t imgIdx = 0; imgIdx < numImages; ++imgIdx) {
        for (std::uint_fast32_t i = 0; i < numRows; ++i) {
            for (std::uint_fast32_t j = 0; j < numCols; ++j) {
                unsigned char byte;
                ifs.read(reinterpret_cast<char*>(&byte), sizeof(byte));
                outImages(i * numRows + j, imgIdx) = byte;
            }
        }
    }
}


void readLabels(MatrixXd &outLabels, const std::string &fname) {
    std::ifstream ifs(fname, std::ios::out | std::ios::binary);

    if (!ifs) {
        std::cerr << "Couldn't open file " + fname << std::endl;
        exit(EXIT_FAILURE);
    }

    std::uint32_t magicNumber;
    ifs.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    endswap(&magicNumber);
    assert(magicNumber == 2049);

    std::uint32_t numLabels;
    ifs.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
    endswap(&numLabels);

    outLabels.resize(10, numLabels);
    outLabels.setZero();
    for (std::uint_fast32_t lblIdx = 0; lblIdx < numLabels; ++lblIdx) {
        unsigned char byte;
        ifs.read(reinterpret_cast<char*>(&byte), sizeof(byte));
        outLabels(byte, lblIdx) = 1;
    }
}


