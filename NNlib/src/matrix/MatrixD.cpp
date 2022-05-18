//
// Created by vityha on 11.05.22.
//

#include "matrix/MatrixD.h"

#include <thread>
#include <functional> // ref, cref
#include <string>
#include <stdexcept> // runtime_error

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
    if (this->cols() != B.rows()) {
        throw std::runtime_error(
                "Number of columns of the first matrix does not match number of rows of the second." +
                std::to_string(this->cols()) + " != " + std::to_string(B.rows()));
    }
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

    std::vector<std::thread> threads;
    threads.reserve(curNumOfProcs);
    size_t quotient = mA.rows() / curNumOfProcs;
    size_t remainder = mA.rows() % curNumOfProcs;

    size_t startRow, endRow = 0;
    for (size_t i = 0; i < curNumOfProcs; ++i) {
        startRow = endRow;
        endRow += quotient + (i < remainder);
        if (parallel) {
            threads.emplace_back(rowsMatrixMultiplication, startRow, endRow,
                                 std::cref(mA), std::cref(mB), std::ref(mC));
        } else {
            rowsMatrixMultiplication(startRow, endRow, mA, mB, mC);
        }
    }

    for (auto &th: threads) {
        th.join();
    }

    return mC;
}

void
MatrixD::rowsMatrixMultiplication(size_t startRow, size_t endRow, const MatrixD &A, const MatrixD &B, MatrixD &C) {
    for (size_t iA = startRow; iA < endRow; ++iA) {
        for (size_t jB = 0; jB < B.cols(); ++jB) {
            double sum = 0;
            for (size_t k = 0; k < A.cols(); ++k) {
                sum += A(iA, k) * B(k, jB);
            }
            C(iA, jB) = sum;
        }
    }
}

// Comparison operators
bool operator==(const MatrixD& left, const MatrixD& right) {
    if (left.rows() != right.rows() || left.cols() != right.cols()) {
        return false;
    }
    for (size_t rowIdx = 0; rowIdx < left.rows(); ++rowIdx) {
        for (size_t colIdx = 0; colIdx < left.cols(); ++colIdx) {
            if (left(rowIdx, colIdx) != right(rowIdx, colIdx)) {
                return false;
            }
        }
    }
    return true;
}

bool operator!=(const MatrixD& left, const MatrixD& right) {
    return !(left == right);
}
