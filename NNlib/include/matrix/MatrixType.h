#ifndef NNLIB_AND_TEST_EXAMPLE_MATRIXTYPE_H
#define NNLIB_AND_TEST_EXAMPLE_MATRIXTYPE_H

#include "MatrixD.h"

#include <Eigen/Core>
#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace mpi = boost::mpi;

//#define USE_EIGEN

#ifdef USE_EIGEN
typedef Eigen::MatrixXd MatrixType;
#else
typedef MatrixD MatrixType;
#endif

// Customization of boost

namespace boost::serialization {
    template<class Archive>
    void serialize(Archive &ar,
                   Eigen::MatrixXd &matrix,
                   const unsigned int /* aVersion */) {
        Eigen::Index rows = matrix.rows();
        Eigen::Index cols = matrix.cols();
        ar & (rows);
        ar & (cols);
        if (rows != matrix.rows() || cols != matrix.cols())
            matrix.resize(rows, cols);
        if (matrix.size() != 0)
            ar & boost::serialization::make_array(matrix.data(), rows * cols);
    }
} // namespace boost::serialization

#endif //NNLIB_AND_TEST_EXAMPLE_MATRIXTYPE_H
