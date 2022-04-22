cmake_minimum_required(VERSION 3.21)

# Set useful variables
set(PROJECT_NAME NNlib_and_test_example)
set(LIBRARY_NAME NNlib)
set(EXECUTABLE_NAME test_example)

project("${PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 17)

# Create the library
add_library("${LIBRARY_NAME}" NNlib/src/layer/Layer.cpp NNlib/include/layer/Layer.h
        NNlib/src/Model.cpp NNlib/include/Model.h
        NNlib/src/MLP.cpp NNlib/include/MLP.h
        NNlib/src/lossfunction/LossFunction.cpp NNlib/include/lossfunction/LossFunction.h
        NNlib/src/lossfunction/MSELossFunction.cpp NNlib/include/lossfunction/MSELossFunction.h
        NNlib/src/regularization/Regularization.cpp NNlib/src/regularization/Regularization.cpp
        NNlib/include/regularization/Regularization.h NNlib/src/regularization/NoRegularization.cpp
        NNlib/include/regularization/NoRegularization.h NNlib/src/regularization/L2Regularization.cpp
        NNlib/include/regularization/L2Regularization.h NNlib/src/lossfunction/SoftMaxLossFunction.cpp
        NNlib/include/lossfunction/SoftMaxLossFunction.h NNlib/include/layer/FCLayer.h NNlib/src/layer/FCLayer.cpp
        NNlib/include/layer/ActivationLayer.h NNlib/src/layer/ActivationLayer.cpp NNlib/src/layer/HTActivationLayer.cpp
        NNlib/include/layer/HTActivationLayer.h
        NNlib/src/layer/SMActivationLayer.cpp NNlib/include/layer/SMActivationLayer.h
        NNlib/src/layer/SigmoidActivationLayer.cpp NNlib/include/layer/SigmoidActivationLayer.h)
target_include_directories("${LIBRARY_NAME}" PUBLIC NNlib/include)

# Set installation prefix of Eigen3
set(Eigen3_DIR ./Eigen3/share/eigen3/cmake)

# Link Eigen3 with the our library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries("${LIBRARY_NAME}" Eigen3::Eigen)

# Create test example and link with our library
add_executable("${EXECUTABLE_NAME}" NNlib/include/MLP.h examples/main.cpp examples/XOR.cpp)
target_link_libraries("${EXECUTABLE_NAME}" "${LIBRARY_NAME}")

# Add testing executable
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest/CMakeLists.txt")
    add_subdirectory(googletest)
    add_executable(
            NNlib_test
            tests/test.cpp
    )
    target_link_libraries(
            NNlib_test
            gtest_main
    )
else()
    message(WARNING "Clone Gtest into project's root directory to be able to run Google tests:\n\
git clone https://github.com/google/googletest.git")
endif()
