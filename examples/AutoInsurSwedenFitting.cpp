#include <Model.h>
#include <layer/SigmoidActivationLayer.h>
#include <layer/FCLayer.h>
#include <lossfunction/MSELossFunction.h>
#include <Eigen/Core>

#include <fstream>
#include <string>
#include <algorithm> // for std::random_shuffle()
#include <cmath> // for std::round()

using Eigen::MatrixXd;

void readAutoInsurSweden(const std::string &path, std::vector<double> &outX, std::vector<double> &outY);
void divideDataIntoTrainAndTest(const MatrixXd &X, const MatrixXd &Y, size_t numTrainExamples,
                                MatrixXd &outTrainX, MatrixXd &outTrainY,
                                MatrixXd &outTestX, MatrixXd &outTestY);
void runExample(const MatrixXd &trainX, const MatrixXd &trainY, const MatrixXd &testX, const MatrixXd &testY,
                unsigned long trainIterations, double alpha);

constexpr const char *pathToAutoInsurSweden = "./data/AutoInsurSweden.txt";
constexpr unsigned long numIters = 20000000;
constexpr double alphaVal = 1e-5;

int main() {
    std::vector<double> vecX, vecY;
    readAutoInsurSweden(pathToAutoInsurSweden, vecX, vecY);

    Eigen::Map<Eigen::RowVectorXd> X(&vecX[0], vecX.size());
    Eigen::Map<Eigen::RowVectorXd> Y(&vecY[0], vecY.size());

    MatrixXd trainX, trainY, testX, testY;

    divideDataIntoTrainAndTest(X, Y, static_cast<unsigned long>(std::round(0.8 * X.cols())), trainX, trainY, testX, testY);

    std::cout << "trainX:\n" << trainX << '\n'
    << "trainY:\n" << trainY << '\n'
    << "testX:\n" << testX << '\n'
    << "testY:\n" << testY << '\n' << std::endl;

    runExample(trainX, trainY, testX, testY, numIters, alphaVal);

    return 0;
}

void runExample(const MatrixXd &trainX, const MatrixXd &trainY, const MatrixXd &testX, const MatrixXd &testY,
                unsigned long trainIterations, double alpha) {
    Model model{1, std::make_shared<MSELossFunction>()};
    model.addLayer<FCLayer>(10, 1);
    model.addLayer<SigmoidActivationLayer>(10);
    model.addLayer<FCLayer>(10, 1);
    model.addLayer<SigmoidActivationLayer>(10);
    model.addLayer<FCLayer>(1, 10);

    for (unsigned long i = 0; i < trainIterations; ++i) {
        model.trainBatch(trainX, trainY, alpha);
        if (i % 20000 == 0) {
            std::cout << "Iteration #" << i << "\n";
            std::cout << "MSE Loss on train dataset (iteration #" << i << ") =\t"
                                            << model.calcLoss(model.forwardPass(trainX), trainY) << '\n';
            std::cout << "MSE Loss on test dataset =\t"
                      << model.calcLoss(model.forwardPass(testX), testY) << '\n' << std::endl;
        }
    }
    std::cout << "Final MSE Loss on train dataset =\t"
              << model.calcLoss(model.forwardPass(trainX), trainY) << '\n';
    std::cout << "Final MSE Loss on test dataset =\t"
              << model.calcLoss(model.forwardPass(testX), testY) << '\n' << std::endl;

    MatrixXd range = Eigen::RowVectorXd::LinSpaced(130 * 10 + 1, 0.0, 130.0);
    MatrixXd predictions = model.forwardPass(range);

    std::cout << "range\n"
              << "---------------------\n"
              << range << '\n'
              << "---------------------\n"
              << predictions << '\n'
              << "---------------------\n"
              << "Predicted\ns" << std::endl;

    std::cout << model << std::endl;
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

void divideDataIntoTrainAndTest(const MatrixXd &X, const MatrixXd &Y, size_t numTrainExamples,
                MatrixXd &outTrainX, MatrixXd &outTrainY,
                MatrixXd &outTestX, MatrixXd &outTestY) {
    MatrixXd stack(X.rows() + Y.rows(), X.cols());
    stack << X, Y;

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(stack.cols());
    perm.setIdentity();
    std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
    stack = stack * perm; // permute columns

    auto trainIndices = Eigen::seq(0, numTrainExamples - 1);
    auto testIndices = Eigen::seq(numTrainExamples, Eigen::placeholders::last);
    auto XIndices = Eigen::seq(0, X.rows() - 1);
    auto YIndices = Eigen::seq(X.rows(), Eigen::placeholders::last);

    outTrainX = stack(XIndices, trainIndices);
    outTrainY = stack(YIndices, trainIndices);
    outTestX = stack(XIndices, testIndices);
    outTestY = stack(YIndices, testIndices);
}
