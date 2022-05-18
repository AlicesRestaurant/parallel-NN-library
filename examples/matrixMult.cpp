//
// Created by vityha on 18.05.22.
//

#include "matrix/MatrixD.h"

#include <Eigen/Dense>

#include <iostream>
#include <cstddef>
#include <vector>

int main() {
    size_t mat_m = 5, mat_n = 4;

    std::vector<double> v1{1, 5, 8, 6,
                           1, 3, 7, 6,
                           1, 3, 4, 6,
                           1, 2, 7, 6,
                           7, 2, 3, 4};
    std::vector<double> v2{1, 3, 8, 6,
                           1, 3, 9, 6,
                           1, 1, 1, 1,
                           1, 2, 7, 6};
    std::vector<double> vC(mat_m * mat_n);
    MatrixD mA(mat_m, mat_n, v1);
    MatrixD mB(mat_n, mat_n, v2);
    MatrixD mC(mat_m, mat_n, vC);

    std::cout << mA * mB;

    return 0;
}