cmake_minimum_required(VERSION 3.21)

# Set useful variables
set(PROJECT_NAME NNlib_and_test_example)
set(LIBRARY_NAME NNlib)
set(EXECUTABLE_NAME test_example)

project("${PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 17)

# Create the library
add_library("${LIBRARY_NAME}" NNlib/src/library.cpp NNlib/include/library.h NNlib/src/layer.cpp NNlib/include/layer.h
        NNlib/src/Model.cpp NNlib/include/Model.h NNlib/src/MLP.cpp NNlib/include/MLP.h)
target_include_directories("${LIBRARY_NAME}" PUBLIC NNlib/include)

# Set installation prefix of Eigen3
set(Eigen3_DIR ./Eigen3/share/eigen3/cmake)

# Link Eigen3 with the our library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries("${LIBRARY_NAME}" Eigen3::Eigen)

# Create test example and link with our library
add_executable("${EXECUTABLE_NAME}" main.cpp NNlib/include/layer.h)
target_link_libraries("${EXECUTABLE_NAME}" "${LIBRARY_NAME}")
