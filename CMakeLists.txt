cmake_minimum_required(VERSION 3.21)

# Set useful variables
set(PROJECT_NAME NNlib_and_test_example)
set(LIBRARY_NAME NNlib)

project("${PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 17)

# Create the library
add_library("${LIBRARY_NAME}" NNlib/src/layer/Layer.cpp NNlib/include/layer/Layer.h
        NNlib/src/Model.cpp NNlib/include/Model.h
        NNlib/src/MLP.cpp NNlib/include/MLP.h
        NNlib/src/lossfunction/LossFunction.cpp NNlib/include/lossfunction/LossFunction.h
        NNlib/src/lossfunction/MSELossFunction.cpp NNlib/include/lossfunction/MSELossFunction.h
#        NNlib/src/regularization/Regularization.cpp NNlib/include/regularization/Regularization.h
#        NNlib/src/regularization/NoRegularization.cpp NNlib/include/regularization/NoRegularization.h
#        NNlib/src/regularization/L2Regularization.cpp NNlib/include/regularization/L2Regularization.h
#        NNlib/src/lossfunction/SoftMaxLossFunction.cpp NNlib/include/lossfunction/SoftMaxLossFunction.h
        NNlib/include/layer/FCLayer.h NNlib/src/layer/FCLayer.cpp
        NNlib/include/layer/ActivationLayer.h NNlib/src/layer/ActivationLayer.cpp
#        NNlib/src/layer/HTActivationLayer.cpp NNlib/include/layer/HTActivationLayer.h
#        NNlib/src/layer/SMActivationLayer.cpp NNlib/include/layer/SMActivationLayer.h
        NNlib/src/layer/SigmoidActivationLayer.cpp NNlib/include/layer/SigmoidActivationLayer.h
#        NNlib/src/lossfunction/SVMLossFunction.cpp NNlib/include/lossfunction/SVMLossFunction.h
        NNlib/src/matrix/MatrixD.cpp NNlib/include/matrix/MatrixD.h
        NNlib/include/matrix/MatrixType.h NNlib/include/utils/random.h NNlib/src/utils/random.cpp
        NNlib/src/trainer/Trainer.cpp NNlib/include/trainer/Trainer.h
        NNlib/src/trainer/DistributedTrainer.cpp NNlib/include/trainer/DistributedTrainer.h)
target_include_directories("${LIBRARY_NAME}" PUBLIC NNlib/include)
#if(MSVC)
#    add_compile_options(/FI"NNlib/include/matrix/MatrixType.h")
#else()
#    # GCC or Clang
##    add_compile_options(-include matrix/MatrixType.h)
#endif()


# Set installation prefix of Eigen3
set(Eigen3_DIR ./Eigen3/share/eigen3/cmake)

# Link Eigen3 with the our library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries("${LIBRARY_NAME}" Eigen3::Eigen)

# Link MPI with library
find_package(MPI REQUIRED)
target_include_directories(${LIBRARY_NAME} PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(${LIBRARY_NAME} ${MPI_CXX_LIBRARIES})

find_package(Boost 1.71.0 COMPONENTS mpi serialization REQUIRED)
target_include_directories(${LIBRARY_NAME} PRIVATE ${Boost_INCLUDE_DIR})
target_link_libraries(${LIBRARY_NAME} ${Boost_LIBRARIES})

# Link Threads with library
find_package(Threads REQUIRED)
target_link_libraries("${LIBRARY_NAME}" ${CMAKE_THREAD_LIBS_INIT})

# Create test examples and link with our library
function(addTestExample target_name_of_example)
    # example of usage: addTestExample(targetName src/main.cpp src/my_utils.h)
    add_executable(${ARGV})
    target_link_libraries("${target_name_of_example}" "${LIBRARY_NAME}")
    target_link_libraries("${target_name_of_example}" ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries("${target_name_of_example}" Eigen3::Eigen)

    target_include_directories(${target_name_of_example} PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(${target_name_of_example} ${MPI_CXX_LIBRARIES})

    target_include_directories(${target_name_of_example} PRIVATE ${Boost_INCLUDE_DIR})
    target_link_libraries(${target_name_of_example} ${Boost_LIBRARIES})
endfunction()

addTestExample(XOR examples/XOR.cpp)
addTestExample(NOR examples/NOR.cpp)
addTestExample(AutoInsurSwedenFitting examples/AutoInsurSwedenFitting.cpp)
addTestExample(linearClassifierMNIST examples/linearClassifierMNIST.cpp)
addTestExample(partitionMatrixMult examples/partitionMatrixMult.cpp)
addTestExample(projectPresentationExample examples/projectPresentationExample.cpp)
addTestExample(primitiveMatrixMult examples/primitiveParallelMatrixMult.cpp)
addTestExample(matrixMult examples/matrixMult.cpp examples/time_measurement.h)

# Add libraries to example



# Add testing executable
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest/CMakeLists.txt")
    add_subdirectory(googletest)
    add_executable(
            NNlib_test
            tests/test.cpp
    )
    target_link_libraries(
            NNlib_test
            gtest_main
    )
    target_link_libraries(NNlib_test ${LIBRARY_NAME})
else ()
    message(WARNING "Clone Gtest into project's root directory to be able to run Google tests:\n\
git clone https://github.com/google/googletest.git")
endif ()
