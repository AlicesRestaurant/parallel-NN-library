//
// Created by vityha on 18.05.22.
//

#include "matrix/MatrixD.h"
#include "time_measurement.h"

#include <Eigen/Dense>

#include <iostream>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <limits>
#include <random>

struct gen_rand {
    double factor;
public:
    gen_rand(double range = 1.0) : factor(range / RAND_MAX) {}

    double operator()() {
        return rand() * factor;
    }
};

int main() {
    size_t mat_m = 1e+03, mat_n = 1e+03;
    size_t numElements = 1e+06;
    double range = 1000;

    std::vector<double> v1;
    std::vector<double> v2;
    std::vector<double> vC(mat_m * mat_n);
    v1.reserve(numElements);
    v2.reserve(numElements);
    std::generate_n(std::back_inserter(v1), numElements, gen_rand(range));
    std::generate_n(std::back_inserter(v2), numElements, gen_rand(range));

//    for (auto v: v1) {
//        std::cout << v << ",";
//    }
//    std::cout << "\n";

    MatrixD mA(mat_m, mat_n, v1);
    MatrixD mB(mat_n, mat_n, v2);
    MatrixD mC(mat_m, mat_n, vC);

    MatrixD::setNumberProcessors(6);

    MatrixD::setParallelExecution(true);

    auto startPar = get_current_time_fenced();

    mC = mA * mB;

    auto finPar = get_current_time_fenced();

    std::cout << mC(0, 0) << '\n';

    MatrixD::setParallelExecution(false);

    auto startSeq = get_current_time_fenced();

    mC = mA * mB;

    auto finSeq = get_current_time_fenced();

    std::cout << "parallel (us): " << to_us(finPar - startPar) << '\n'
              << "seq (us): " << to_us(finSeq - startSeq) << '\n';

    std::cout << mC(0, 0) << '\n';

    return 0;
}