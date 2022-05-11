//
// Created by vityha on 11.05.22.
//

#include "matrix/MatrixD.h"

#include <iostream>

int main() {
    std::vector<double> v1 = {1, 2, 3, 4};
    MatrixD m1(2, 2, v1);

    double *p1 = &m1(1,0);

    std::cout << *(p1+1) << '\n' << *(p1+2);

    return 0;
}