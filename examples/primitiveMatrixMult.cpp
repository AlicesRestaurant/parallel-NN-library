//
// Created by vityha on 17.05.22.
//

#include "matrix/MatrixD.h"

#include <iostream>
#include <cstddef>
#include <vector>
#include <thread>

//divide num of rows by k processors
//start threads with (numberOfRows, numberOfProcessor, ref(A), ref(B), ref(C))
//multiply in each thread
//write result in C

void multiply(size_t numOfRows, size_t numOfProc, MatrixD &A, MatrixD &B, MatrixD &C) {
    for (int iA = (numOfProc - 1) * numOfRows; iA < numOfProc * numOfRows; iA++) {
        for (int jB = 0; jB < B.cols(); jB++) {
            double sum = 0;
            for (int k = 0; k < A.cols(); k++) {
                sum += A(iA, k) * B(k, jB);
            }
            C(iA, jB) = sum;
        }
    }
}

int main() {
    size_t mat_n = 4;

    std::vector<double> v1{1, 5, 8, 6,
                           1, 3, 7, 6,
                           1, 3, 4, 6,
                           1, 2, 7, 6};
    std::vector<double> v2{1, 3, 8, 6,
                           1, 3, 9, 6,
                           1, 1, 1, 1,
                           1, 2, 7, 6};
    std::vector<double> vC(mat_n * mat_n);
    MatrixD m1(mat_n, mat_n, v1);
    MatrixD m2(mat_n, mat_n, v2);
    MatrixD mC(mat_n, mat_n, vC);

    size_t numOfProc = 3;
    size_t numOfRows;
    std::vector<std::thread> threads;
    threads.reserve(3);

    for (size_t i = 0; i < numOfProc; i++) {
        if (i == numOfProc - 1) {
            numOfRows = m1.rows() / numOfProc + m1.rows() % numOfProc;
        }
        else {
            numOfRows = m1.rows() / numOfProc;
        }
        threads.emplace_back(multiply, numOfRows, i, std::ref(m1), std::ref(m2), std::ref(mC));
    }

    for (auto& th: threads) {
        if (th.joinable()){
            th.join();
        }
    }

    std::cout << mC;

    return 0;
}