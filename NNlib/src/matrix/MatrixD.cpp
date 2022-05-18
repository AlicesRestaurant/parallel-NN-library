//
// Created by vityha on 11.05.22.
//

#include "matrix/MatrixD.h"

#include <thread>

// Printing

std::ostream &operator<<(std::ostream &os, const MatrixD &matrix) {
    os << "[";
    for (size_t i = 0; i < matrix.rows(); i++) {
        os << "[";
        for (size_t j = 0; j < matrix.cols(); j++) {
            os << matrix(i, j);
            if (j != matrix.cols() - 1) {
                os << " ";
            }
        }
        os << "]";
        if (i != matrix.rows() - 1) {
            os << "\n";
        }
    }
    os << "]\n";

    return os;
}

// Multiplication

MatrixD MatrixD::operator*(const MatrixD &B) {
    return primitiveMultiplication(B);
}

MatrixD MatrixD::primitiveMultiplication(const MatrixD &B) {
    const MatrixD &mA = (*this);
    const MatrixD &mB = B;
    MatrixD mC(mA.rows(), mB.cols(), std::vector<double>(mA.rows() * mB.cols()));

    size_t curNumOfProcs = numOfProcs;
    if (curNumOfProcs > mA.rows()) {
        curNumOfProcs = mA.rows();
    }

    size_t firstRow, lastRow;
    std::vector<std::thread> threads;
    threads.reserve(curNumOfProcs);

    for (size_t i = 0; i < curNumOfProcs; i++) {
        if (i == curNumOfProcs - 1) {
            firstRow = (mA.rows() / curNumOfProcs) * i;
            lastRow = mA.rows() - 1;
        } else {
            firstRow = (mA.rows() / curNumOfProcs) * i;
            lastRow = (mA.rows() / curNumOfProcs) * (i + 1) - 1;
        }
        if (parallelExecution) {
            threads.emplace_back(rowsMatrixMultiplication, firstRow, lastRow, std::ref(mA), std::ref(mB), std::ref(mC));
        } else {
            rowsMatrixMultiplication(firstRow, lastRow, mA, mB, mC);
        }
    }

    for (auto &th: threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    return mC;
}

void
MatrixD::rowsMatrixMultiplication(size_t firstRow, size_t lastRow, const MatrixD &A, const MatrixD &B, MatrixD &C) {
    for (size_t iA = firstRow; iA <= lastRow; iA++) {
        for (size_t jB = 0; jB < B.cols(); jB++) {
            double sum = 0;
            for (int k = 0; k < A.cols(); k++) {
                sum += A(iA, k) * B(k, jB);
            }
            C(iA, jB) = sum;
        }
    }
}

// Parallel parameters

bool MatrixD::parallelExecution = false;
size_t MatrixD::numOfProcs = 6;

void MatrixD::setParallelExecution(bool parExecution) {
    parallelExecution = parExecution;
}

void MatrixD::setNumberProcessors(size_t numProcessors) {
    numOfProcs = numProcessors;
}