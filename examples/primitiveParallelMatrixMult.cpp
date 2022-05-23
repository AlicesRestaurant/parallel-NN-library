//
// Created by vityha on 17.05.22.
//

#include "matrix/MatrixD.h"

#include <Eigen/Dense>

#include <iostream>
#include <cstddef>
#include <vector>
#include <thread>

//divide num of rows by k processors
//start threads with (numberOfRows, numberOfProcessor, ref(A), ref(B), ref(C))
//multiply in each thread
//write result in C

//#define PARALLEL

void multiply(size_t firstRow, size_t lastRow, MatrixD &A, MatrixD &B, MatrixD &C) {
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

int main() {
    size_t mat_m = 5, mat_n = 4;

    std::vector<double> vA{1, 5, 8, 6,
                           1, 3, 7, 6,
                           1, 3, 4, 6,
                           1, 2, 7, 6,
                           7, 2, 3, 4};
    std::vector<double> vB{1, 3, 8, 6,
                           1, 3, 9, 6,
                           1, 1, 1, 1,
                           1, 2, 7, 6};
    std::vector<double> vC(mat_m * mat_n);
    MatrixD mA(mat_m, mat_n, vA);
    MatrixD mB(mat_n, mat_n, vB);
    MatrixD mC(mat_m, mat_n, vC);

    std::cout << mA(2, 2) << '\n';

    size_t numOfProcs = 6;
    if (numOfProcs > mA.rows()) {
        numOfProcs = mA.rows();
    }

    size_t firstRow, lastRow;
    std::vector<std::thread> threads;
    threads.reserve(numOfProcs);

    for (size_t i = 0; i < numOfProcs; i++) {
        if (i == numOfProcs - 1) {
            firstRow = (mA.rows() / numOfProcs) * i;
            lastRow = mA.rows() - 1;
        } else {
            firstRow = (mA.rows() / numOfProcs) * i;
            lastRow = (mA.rows() / numOfProcs) * (i + 1) - 1;
        }
#ifdef PARALLEL
        threads.emplace_back(multiply, firstRow, lastRow, std::ref(mA), std::ref(mB), std::ref(mC));
#else
        multiply(firstRow, lastRow, mA, mB, mC);
#endif
    }

    for (auto &th: threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    std::cout << mC;

//    compare with Eigen

    Eigen::MatrixXd eA = Eigen::Map<Eigen::Matrix<double, 5, 4>>(vA.data());
    Eigen::MatrixXd eB = Eigen::Map<Eigen::Matrix<double, 4, 4>>(vB.data());

    std::cout << eA * eB;

    return 0;
}