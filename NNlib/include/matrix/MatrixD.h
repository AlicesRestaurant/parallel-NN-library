//
// Created by vityha on 11.05.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_MATRIXD_H
#define NNLIB_AND_TEST_EXAMPLE_MATRIXD_H

#include <vector>
#include <stdexcept>
#include <cstddef>

class MatrixD {
public:
    MatrixD(size_t xDim, size_t yDim, std::vector<double> &data) : xDim(xDim), yDim(yDim), data(data) {}

    double& operator()(unsigned int x, unsigned int y)
    {
        if (x >= xDim || y >= yDim)
            throw std::out_of_range("matrix indices out of range");
        return data[xDim * y + x];
    }

private:
    std::vector<double> data;
    size_t xDim, yDim;
};


#endif //NNLIB_AND_TEST_EXAMPLE_MATRIXD_H
