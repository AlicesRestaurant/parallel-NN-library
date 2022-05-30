#ifndef NNLIB_AND_TEST_EXAMPLE_CUDACODE_H
#define NNLIB_AND_TEST_EXAMPLE_CUDACODE_H

#include "matrix/MatrixD.h"

#include <vector>

double *copy_to_device(const std::vector<double> &cont);

ViewOfData<MatrixD::ContainerType> cudaMatrixMultiplication(const ViewOfData<MatrixD::ContainerType> &viewOfData1,
                                                            const ViewOfData<MatrixD::ContainerType> &viewOfData2);

#endif //NNLIB_AND_TEST_EXAMPLE_CUDACODE_H

