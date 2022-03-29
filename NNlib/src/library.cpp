#include "library.h"

#include <iostream>

#include <Eigen/Dense>

void hello() {
    float data[] = {1, 2, 3, 4, 5, 6};
    Eigen::Map<Eigen::Vector3f> v1(data);
    v1(0) = 10;
    for(auto &i: data){
        std::cout << "i = " << i << '\n';
    }
}
