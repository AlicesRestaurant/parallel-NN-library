#ifndef NNLIB_AND_TEST_EXAMPLE_MODEL_H
#define NNLIB_AND_TEST_EXAMPLE_MODEL_H

#include "layer/Layer.h"
#include "lossfunction/LossFunction.h"

#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cstddef>
#include <boost/mpi.hpp>

namespace mpi = boost::mpi;

class Model {
public:
    Model(size_t numInputNodes, const std::shared_ptr<LossFunction> &lossFunctionPtr) : numInputNodes{numInputNodes},
                                                                                        lossFunctionPtr{
                                                                                                lossFunctionPtr} {}

    MatrixType forwardPass(MatrixType input);
    double calcLoss(const MatrixType& predictions, const MatrixType& groundTruths);

    template<class LayerType, class... Args>
    void addLayer(Args... args) {
        layerPtrs.push_back(std::make_shared<LayerType>(args...));
    }

    void trainBatch(MatrixType features, MatrixType labels, double alpha);


    void trainDataset(Eigen::MatrixXd features, Eigen::MatrixXd labels, int batchSize, int numberProcessors,
                      int numberIterations, double alpha);

    std::vector<Eigen::MatrixXd> calculateBatchGradients(Eigen::MatrixXd features, Eigen::MatrixXd labels);

    static void setMPIExecution(bool mpiExec) {
        mpiExecution = mpiExec;
    }

    static bool getMPIExecution() {
        return mpiExecution;
    }

    static void setNumberProcessors(size_t nProcessors) {
        numProcessors = nProcessors;
    }

    static size_t getNumberProcessors() {
        return numProcessors;
    }

    static void setCommunicator(mpi::communicator c) {
        comm = c;
    }

    static mpi::communicator &getCommunicator() {
        return comm;
    }

    friend std::ostream &operator<<(std::ostream &os, const Model &mlp);

protected:
    std::vector<std::shared_ptr<Layer>> layerPtrs;

    std::shared_ptr<LossFunction> lossFunctionPtr;

    const size_t numInputNodes;

    static bool mpiExecution;
    static size_t numProcessors;
    static mpi::communicator comm;
};


#endif //NNLIB_AND_TEST_EXAMPLE_MODEL_H
