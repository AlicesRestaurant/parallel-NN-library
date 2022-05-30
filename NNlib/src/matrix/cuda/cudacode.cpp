#include "matrix/cuda/cudacode.h"
#include "matrix/MatrixD.h"

#include <iostream>

__global__ void matrix_mul_device(double *dev_d1, double *dev_d2, double *dev_res, size_t start1, size_t fullWidth1, size_t fullHeight1, size_t width1, size_t height1,
                                  size_t start2, size_t fullWidth2, size_t fullHeight2, size_t width2, size_t height2) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height1 || col >= width2) {
        return;
    }
    double summ = 0;
    for (size_t i = 0; i < width1; ++i) {
        summ += dev_d1[start1 + i + row * fullWidth1] * dev_d2[start2 + col + i * fullWidth2];
    }
    dev_res[row * width1 + col] = summ;
}

double *copy_to_device(const MatrixD::ContainerType &cont) {
    double *ptr;
    cudaMalloc(&ptr, cont.size());
    cudaMemcpy(ptr, &cont[0], cont.size() * sizeof(double), cudaMemcpyHostToDevice);
    return ptr;
}

ViewOfData<MatrixD::ContainerType> cudaMatrixMultiplication(const ViewOfData<MatrixD::ContainerType> &viewOfData1,
                              const ViewOfData<MatrixD::ContainerType> &viewOfData2) {
    const MatrixD::ContainerType &d1 = viewOfData1.getData();
    const MatrixD::ContainerType &d2 = viewOfData2.getData();
    double *dev_d1, *dev_d2, *dev_res;
    cudaMalloc(&dev_d1, viewOfData1.getFullWidth() * viewOfData1.getFullHeight() * sizeof(double));
    cudaMalloc(&dev_d2, viewOfData2.getFullWidth() * viewOfData2.getFullHeight() * sizeof(double));
    cudaMalloc(&dev_res, viewOfData1.getHeight() * viewOfData2.getWidth() * sizeof(double));
    cudaMemcpy(dev_d1, &d1[0], d1.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_d2, &d2[0], d2.size() * sizeof(double), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((viewOfData1.getHeight() + 31) / 32, (viewOfData2.getWidth() + 31) / 32);
    matrix_mul_device<<<blocksPerGrid, threadsPerBlock>>>(dev_d1, dev_d2, dev_res, viewOfData1.getStart(), viewOfData1.getFullWidth(), viewOfData1.getFullHeight(), viewOfData1.getWidth(), viewOfData1.getHeight(),
                                                          viewOfData2.getStart(), viewOfData2.getFullWidth(), viewOfData2.getFullHeight(), viewOfData2.getWidth(), viewOfData2.getHeight());
    cudaFree(dev_d1);
    cudaFree(dev_d2);
    MatrixD::ContainerType res(viewOfData1.getHeight() * viewOfData2.getWidth());
    cudaMemcpy(&res[0], dev_res, res.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dev_res);
    return ViewOfData<MatrixD::ContainerType>(std::move(res), viewOfData1.getHeight(), viewOfData2.getWidth());
}

