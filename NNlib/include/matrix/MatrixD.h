//
// Created by vityha on 11.05.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_MATRIXD_H
#define NNLIB_AND_TEST_EXAMPLE_MATRIXD_H

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <iostream>

class MatrixD {
public:
    static bool parallelExecution;
    static size_t numOfProcs;

    MatrixD(size_t xDim, size_t yDim) : xDim(xDim), yDim(yDim) {
        data.reserve(xDim * yDim);
    }

    MatrixD(size_t xDim, size_t yDim, std::vector<double> &data) : xDim(xDim), yDim(yDim), data(data) {}

    MatrixD(size_t xDim, size_t yDim, std::vector<double> &&data) : xDim(xDim), yDim(yDim), data(data) {}

    // data is in column-major way
    const double &operator()(unsigned int x, unsigned int y) const {
        if (x >= xDim || y >= yDim)
            throw std::out_of_range("matrix indices out of range");
        return data[xDim * y + x];
    }

    double &operator()(unsigned int x, unsigned int y) {
        if (x >= xDim || y >= yDim)
            throw std::out_of_range("matrix indices out of range");
        return data[xDim * y + x];
    }

    MatrixD operator*(const MatrixD &B);

    friend std::ostream &operator<<(std::ostream &os, const MatrixD &matrix);

    size_t rows() const {
        return xDim;
    }

    size_t cols() const {
        return yDim;
    }

    static void setParallelExecution(bool parExecution);

    static void setNumberProcessors(size_t numProcessors);
private:
    std::vector<double> data;
    size_t xDim, yDim;

    MatrixD primitiveMultiplication(const MatrixD &B);

    static void
    rowsMatrixMultiplication(size_t firstRow, size_t lastRow, const MatrixD &A, const MatrixD &B, MatrixD &C);
};


#endif //NNLIB_AND_TEST_EXAMPLE_MATRIXD_H
