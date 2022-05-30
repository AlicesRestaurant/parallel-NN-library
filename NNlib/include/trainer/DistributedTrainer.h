//
// Created by vityha on 24.05.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_DISTRIBUTEDTRAINER_H
#define NNLIB_AND_TEST_EXAMPLE_DISTRIBUTEDTRAINER_H

#include "trainer/Trainer.h"
#include "Model.h"

#include <memory>
#include <utility>
#include <boost/mpi.hpp>

namespace mpi = boost::mpi;

class DistributedTrainer : public Trainer {
public:
    DistributedTrainer(std::shared_ptr<Model> model, size_t batchSize, double learningRate, std::shared_ptr<mpi::communicator> comm, size_t numProcessors) : Trainer(std::move(model), batchSize, learningRate), comm(std::move(comm)), numProcessors(numProcessors) {}
    void trainDataset(MatrixType &features, MatrixType &labels, int numberIterations) override;

    void setNumberProcessors(size_t nProcessors) {
        numProcessors = nProcessors;
    }

    size_t getNumberProcessors() const {
        return numProcessors;
    }

    void setCommunicator(std::shared_ptr<mpi::communicator> &c) {
        comm = c;
    }

    std::shared_ptr<mpi::communicator> &getCommunicator() {
        return comm;
    }
protected:
    std::shared_ptr<mpi::communicator> comm;
    size_t numProcessors = 4;
};


#endif //NNLIB_AND_TEST_EXAMPLE_DISTRIBUTEDTRAINER_H
