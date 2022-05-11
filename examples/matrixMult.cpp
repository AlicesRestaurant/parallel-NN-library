//
// Created by vityha on 11.05.22.
//

#include "matrix/MatrixD.h"

#include <iostream>
#include <cstddef>
#include <vector>

void partition(double *&A00, double *&A01, double *&A10, double *&A11, size_t mat_n, size_t cur_n) {
    A01 = A00 + cur_n / 2;
    A10 = A00 + mat_n * cur_n / 2;
    A11 = A00 + (mat_n + 1) * cur_n / 2;
}

void add(size_t mat_n, size_t mat_T_n, size_t cur_n, double *&C, double *&T) {
    if (cur_n == 1) {
        *C = (*C) + (*T);
        return;
    }

    double *C00 = C, *C01, *C10, *C11;
    double *T00 = T, *T01, *T10, *T11;
    partition(C00, C01, C10, C11, mat_n, cur_n);
    partition(T00, T01, T10, T11, mat_T_n, cur_n);

    add(mat_n, mat_T_n, cur_n / 2, C00, T00);
    add(mat_n, mat_T_n, cur_n / 2, C01, T01);
    add(mat_n, mat_T_n, cur_n / 2, C10, T10);
    add(mat_n, mat_T_n, cur_n / 2, C11, T11);
}

void multiply(size_t mat_n, size_t cur_n, double *&A, double *&B, double *&C) {
    if (cur_n == 1) {
        *C = (*A) * (*B);
        return;
    }

    std::vector<double> vT(cur_n * cur_n);
    MatrixD T(cur_n, cur_n, vT);

    double *A00 = A, *A01, *A10, *A11;
    double *B00 = B, *B01, *B10, *B11;
    double *C00 = C, *C01, *C10, *C11;
    double *T00 = &T(0, 0), *T01, *T10, *T11;
    partition(A00, A01, A10, A11, mat_n, cur_n);
    partition(B00, B01, B10, B11, mat_n, cur_n);
    partition(C00, C01, C10, C11, mat_n, cur_n);
    partition(T00, T01, T10, T11, mat_n, cur_n);

    multiply(mat_n, cur_n / 2, A00, B00, C00);
    multiply(mat_n, cur_n / 2, A00, B01, C01);
    multiply(mat_n, cur_n / 2, A10, B00, C10);
    multiply(mat_n, cur_n / 2, A10, B01, C11);
    multiply(mat_n, cur_n / 2, A01, B10, T00);
    multiply(mat_n, cur_n / 2, A01, B11, T01);
    multiply(mat_n, cur_n / 2, A11, B10, T10);
    multiply(mat_n, cur_n / 2, A11, B11, T11);

    add(mat_n, cur_n, cur_n, C00, T00);
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

    double *A = &m1(0, 0), *B = &m2(0, 0), *C = &mC(0, 0);

    multiply(mat_n, mat_n, A, B, C);

    std::cout << ' ';

//    for (size_t i = 0; i < mat_n * mat_n; i++) {
//        if (i % mat_n == 0) {
//            std::cout << '\n';
//        }
//        std::cout << *(C + i) << ' ';
//    }

//    double *p1 = &m1(1, 0);
//
//    std::cout << p1 + 1 << '\n' << p1 + 5;

    return 0;
}