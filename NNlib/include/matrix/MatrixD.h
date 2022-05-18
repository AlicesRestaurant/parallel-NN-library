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
    MatrixD(size_t nRows, size_t nCols) : nRows(nRows), nCols(nCols), data(nRows * nCols) {}

    MatrixD(size_t nRows, size_t nCols, std::vector<double> &data) : nRows(nRows), nCols(nCols), data(data) {}

    MatrixD(size_t nRows, size_t nCols, std::vector<double> &&data) : nRows(nRows), nCols(nCols), data(std::move(data)) {}

    // data is in column-major way
    double operator()(size_t y, size_t x) const {
        if (x >= nCols || y >= nRows)
            throw std::out_of_range("matrix indices out of range");
        return data[nCols * y + x];
    }

    double &operator()(size_t y, size_t x) {
        if (x >= nCols || y >= nRows)
            throw std::out_of_range("matrix indices out of range");
        return data[nCols * y + x];
    }

    MatrixD operator*(const MatrixD &B);

    friend std::ostream &operator<<(std::ostream &os, const MatrixD &matrix);

    size_t rows() const {
        return nRows;
    }

    size_t cols() const {
        return nCols;
    }

private:
    std::vector<double> data;
    size_t nRows, nCols;
    size_t numOfProcs = 6;
    bool parallel = true;

    MatrixD primitiveMultiplication(const MatrixD &B);

    static void
    rowsMatrixMultiplication(size_t startRow, size_t endRow, const MatrixD &A, const MatrixD &B, MatrixD &C);
};

bool operator==(const MatrixD& left, const MatrixD& right);
bool operator!=(const MatrixD& left, const MatrixD& right);

#endif //NNLIB_AND_TEST_EXAMPLE_MATRIXD_H
