#include <Model.h>
#include <lossfunction/MSELossFunction.h>
#include <layer/FCLayer.h>
#include <layer/SigmoidActivationLayer.h>
#include "trainer/DistributedTrainer.h"

#include <Eigen/Core>
#include <boost/mpi.hpp>

#include <iostream>
#include <memory>

namespace mpi = boost::mpi;

int main() {
    mpi::environment env;
    mpi::communicator comm;

    if (comm.rank() == 0) {
        //
    }

    Eigen::MatrixXd features(4, 2);
    features << 0, 0,
                0, 1,
                1, 0,
                1, 1;
    features.transposeInPlace();

    Eigen::MatrixXd labels(1, 4);
    labels << 1, 0, 0, 0;

    double alpha = 1;

    size_t batchSize = 2;
    int numberIters = 2000;
    size_t numProcessors = comm.size();

    if (comm.rank() == 0) {
//        std::cout << "before train \n" << model << "\n";

        Model seqModel{2, std::make_shared<MSELossFunction>()};

        seqModel.addLayer<FCLayer>(2, 2);
        seqModel.addLayer<SigmoidActivationLayer>(2);

        seqModel.addLayer<FCLayer>(1, 2);
        seqModel.addLayer<SigmoidActivationLayer>(1);

        Trainer seqTrainer{std::make_shared<Model>(seqModel), batchSize, alpha};

        seqTrainer.trainDataset(features, labels, numberIters);

        std::cout << "Seq model prediction:" << std::endl;
        std::cout << seqModel.forwardPass(features) << std::endl;
        std::cout << std::endl;

//        std::cout << "after train \n" << model << "\n";
    }

    Model distModel{2, std::make_shared<MSELossFunction>()};

    distModel.addLayer<FCLayer>(2, 2);
    distModel.addLayer<SigmoidActivationLayer>(2);

    distModel.addLayer<FCLayer>(1, 2);
    distModel.addLayer<SigmoidActivationLayer>(1);

    DistributedTrainer distTrainer{std::make_shared<Model>(distModel), batchSize, alpha, std::make_shared<mpi::communicator>(comm), numProcessors};

    distTrainer.trainDataset(features, labels, numberIters);

    if (comm.rank() == 0) {
        std::cout << "Dist model prediction:" << std::endl;
        std::cout << distModel.forwardPass(features) << std::endl;
        std::cout << std::endl;

        std::cout << "Features:" << std::endl;
        std::cout << features << std::endl;
        std::cout << std::endl;

        std::cout << "Ground truth:" << std::endl;
        std::cout << labels << std::endl;
        std::cout << std::endl;
    }
    
    return 0;
}