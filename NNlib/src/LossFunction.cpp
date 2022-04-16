//
// Created by vityha on 11.04.22.
//

#include "LossFunction.h"

// Forward propagate

Eigen::VectorXd LossFunction::forwardPropagate(const Eigen::VectorXd &layersOutputs){
    return calculateFunctionOutput(layersOutputs);
}



Eigen::VectorXd LossFunction::calculateFunctionOutput(const Eigen::VectorXd &layersOutputs){
    switch(functionType){
        case FunctionType::MSE:

    }
}

// Backward propagate


// Different functions and derivatives

Eigen::VectorXd LossFunction::MSE(const Eigen::VectorXd &layersOutputs){
    double sumSquaredErrors;
    for (int i=0;i<bottomData.size();i++){}
}