#include <Model.h>
#include <layer/SigmoidActivationLayer.h>
#include <layer/FCLayer.h>
#include <lossfunction/MSELossFunction.h>
#include <Eigen/Core>

#include <fstream>
#include <string>

void readAutoInsurSweden(const std::string &path, std::vector<double> &outX, std::vector<double> &outY);

constexpr const char *pathToAutoInsurSweden = "./data/AutoInsurSweden.txt";
constexpr unsigned long numIters = 5000000;

int main() {
    std::vector<double> vecX, vecY;
    readAutoInsurSweden(pathToAutoInsurSweden, vecX, vecY);

    Eigen::Map<Eigen::RowVectorXd> X(&vecX[0], vecX.size());
    Eigen::Map<Eigen::RowVectorXd> Y(&vecY[0], vecY.size());

    Model model{1, std::make_shared<MSELossFunction>()};
    model.addLayer<FCLayer>(30, 1);
    model.addLayer<SigmoidActivationLayer>(30);
    model.addLayer<FCLayer>(30, 1);
    model.addLayer<SigmoidActivationLayer>(30);
    model.addLayer<FCLayer>(1, 30);

    double alpha = 0.00001;

    for (unsigned long i = 0; i < numIters; ++i) {
        model.trainBatch(X, Y, alpha);
        if (i % 10000 == 0) {
            std::cout << "MSE Loss on iteration #" << i << " =\t" << model.calcLoss(model.forwardPass(X), Y) << std::endl;
        }
    }

    Eigen::MatrixXd range = Eigen::RowVectorXd::LinSpaced(130 * 10 + 1, 0.0, 130.0);
    Eigen::MatrixXd predictions = model.forwardPass(range);

    Eigen::MatrixXd stacked(2, predictions.cols());
    stacked << range, predictions;

    std::cout << "\nActual\n"
    << "---------------------\n"
    << stacked << '\n'
    << "---------------------\n"
    << "Predicted\n";

    return 0;
}

void readAutoInsurSweden(const std::string &path, std::vector<double> &outX, std::vector<double> &outY) {
    std::ifstream ifs{path};
    if (!ifs) {
        std::cerr << "File couldn't be open" << std::endl;
        exit(EXIT_FAILURE);
    }
    int i = 0;
    while (!ifs.eof()) {
        int x;
        std::string yStr;
        ifs >> x >> yStr;
        outX.push_back(x);

        auto iter = yStr.find(',');
        if (iter != std::string::npos) {
            yStr.replace(iter, 1, 1, '.');
        }
        outY.push_back(std::stod(yStr));
    }
}
