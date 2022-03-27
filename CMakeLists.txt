cmake_minimum_required(VERSION 3.21)

# Set useful variables
set(PROJECT_NAME NNlib_and_test_example)
set(LIBRARY_NAME NNlib)
set(EXECUTABLE_NAME test_example)

project("${PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 17)

# Create the library
add_library("${LIBRARY_NAME}" NNlib/src/library.cpp NNlib/include/library.h)
target_include_directories("${LIBRARY_NAME}" PUBLIC NNlib/include)

# Create test example and link with the library
add_executable("${EXECUTABLE_NAME}" main.cpp)
target_link_libraries("${EXECUTABLE_NAME}" "${LIBRARY_NAME}")
