#ifndef NNLIB_AND_TEST_EXAMPLE_SVMLOSSFUNCTION_H
#define NNLIB_AND_TEST_EXAMPLE_SVMLOSSFUNCTION_H

#include <lossfunction/LossFunction.h>

#include <Eigen/Dense>

#include <cstddef>

class SVMLossFunction : public LossFunction {
public:
    double forwardPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) override {
        double lossValue = 0;
        for (size_t colIdx = 0; colIdx < bottomData.cols(); ++colIdx) {
            size_t groundTruthIdx = getGroundTruthIdxFromOneHot(labels.col(colIdx));
            // TODO: possibility to do without copying???
            Eigen::VectorXd curColumnCopy{bottomData.col(colIdx)};
            curColumnCopy = (curColumnCopy.array() - bottomData(colIdx, groundTruthIdx) + 1).max(0);
            curColumnCopy(groundTruthIdx) = 0;
            lossValue += curColumnCopy.sum();
        }
        lossValue /= bottomData.cols();
        return lossValue;
    }
    Eigen::MatrixXd backPropagate(const Eigen::MatrixXd &bottomData, const Eigen::MatrixXd &labels) override {
        Eigen::MatrixXd gradientMatrix(bottomData.rows(), bottomData.cols());
        for (size_t colIdx = 0; colIdx < bottomData.cols(); ++colIdx) {
            size_t groundTruthIdx = getGroundTruthIdxFromOneHot(labels.col(colIdx));
            double val = bottomData(groundTruthIdx, colIdx); // TODO: will there be aliasing if we don't copy?
            gradientMatrix.col(colIdx) = bottomData.col(colIdx).unaryExpr([val] (double el) {
                return (el > val - 1) ? 1.0 : 0.0;
            });
            gradientMatrix(groundTruthIdx, colIdx) = 0;
            gradientMatrix(groundTruthIdx, colIdx) = -gradientMatrix.col(colIdx).sum();
        }
        gradientMatrix /= bottomData.cols();
    }
protected:
    size_t getGroundTruthIdxFromOneHot(const Eigen::VectorXd &oneHot) {
        size_t groundTruthIdx = 0;
        for (size_t rowIdx = 0; rowIdx < oneHot.rows(); ++rowIdx) {
            if (oneHot(rowIdx) == 1) {
                groundTruthIdx = rowIdx;
                break;
            }
        }
        return groundTruthIdx;
    }
};


#endif //NNLIB_AND_TEST_EXAMPLE_SVMLOSSFUNCTION_H
