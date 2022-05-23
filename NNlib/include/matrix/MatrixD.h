//
// Created by vityha on 11.05.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_MATRIXD_H
#define NNLIB_AND_TEST_EXAMPLE_MATRIXD_H

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <utility> // std::move

class MatrixD {
public:
    static bool parallelExecution;
    static size_t numOfProcs;

    MatrixD(size_t nRows, size_t nCols) : nRows(nRows), nCols(nCols), data(nRows * nCols) {}
    MatrixD(size_t nRows, size_t nCols, std::vector<double> &data) : nRows(nRows), nCols(nCols), data(data) {}
    MatrixD(size_t nRows, size_t nCols, std::vector<double> &&data) : nRows(nRows), nCols(nCols), data(std::move(data)) {}

    // data is in row-major way
    double operator()(size_t i, size_t j) const {
        if (j >= nCols || i >= nRows)
            throw std::out_of_range("matrix indices out of range");
        return data[nCols * i + j];
    }

    double &operator()(size_t i, size_t j) {
        if (j >= nCols || i >= nRows)
            throw std::out_of_range("matrix indices out of range");
        return data[nCols * i + j];
    }

    friend std::ostream &operator<<(std::ostream &os, const MatrixD &matrix);

    size_t rows() const {
        return nRows;
    }

    size_t cols() const {
        return nCols;
    }

    static void setParallelExecution(bool parExecution);

    static bool getParallelExecution() {
        return parallelExecution;
    }

    static void setNumberProcessors(size_t numProcessors);

    static size_t getNumberProcessors() {
        return numOfProcs;
    }
private:
    std::vector<double> data;
    size_t nRows, nCols;
};

MatrixD primitiveMultiplication(const MatrixD &left, const MatrixD &right);
void rowsMatrixMultiplication(size_t startRow, size_t endRow, const MatrixD &A, const MatrixD &B, MatrixD &C);

MatrixD operator*(const MatrixD &left, const MatrixD &right);
bool operator==(const MatrixD& left, const MatrixD& right);
bool operator!=(const MatrixD& left, const MatrixD& right);

#endif //NNLIB_AND_TEST_EXAMPLE_MATRIXD_H
