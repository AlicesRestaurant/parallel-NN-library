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
    MatrixD(size_t xDim, size_t yDim, std::vector<double> &data) : xDim(xDim), yDim(yDim), data(data) {}

    double &operator()(unsigned int x, unsigned int y){
        if (x >= xDim || y >= yDim)
            throw std::out_of_range("matrix indices out of range");
        return data[xDim * y + x];
    }

    friend std::ostream &operator<<(std::ostream &os, const MatrixD &matrix) {
        os << "[";
        for (size_t i = 0; i < matrix.rows(); i++) {
            os << "[";
            for (size_t j = 0; j < matrix.cols(); j++) {
                os << matrix(i, j);
                if (j != matrix.cols() - 1) {
                    os << " ";
                }
            }
            os << "]\n";
        }
        os << "]";

        return os;
    }

    size_t rows() const {
        return xDim + 1;
    }

    size_t cols() const {
        return yDim + 1;
    }

private:
    std::vector<double> data;
    size_t xDim, yDim;
};


#endif //NNLIB_AND_TEST_EXAMPLE_MATRIXD_H
