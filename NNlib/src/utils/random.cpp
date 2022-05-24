//
// Created by vityha on 24.05.22.
//

#include "utils/random.h"

#include <algorithm>
#include <random>

int generateRandInt(int min, int max) {
    std::random_device r;
    std::mt19937 gen(r());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

std::vector<int> generateRandIntSeq(int min, int max, int length) {
    std::vector<int> batchesNumbers;
    batchesNumbers.reserve(length);

    while (batchesNumbers.size() != length) {
        batchesNumbers.emplace_back(generateRandInt(min, max)); // create new random number
        std::sort(begin(batchesNumbers), end(batchesNumbers)); // sort before call to unique
        auto last = std::unique(begin(batchesNumbers), end(batchesNumbers));
        batchesNumbers.erase(last, end(batchesNumbers));       // erase duplicates
    }

    return batchesNumbers;
}
