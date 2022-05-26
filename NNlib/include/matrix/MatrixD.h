//
// Created by vityha on 11.05.22.
//

#ifndef NNLIB_AND_TEST_EXAMPLE_MATRIXD_H
#define NNLIB_AND_TEST_EXAMPLE_MATRIXD_H

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <vector>
#include <stdexcept>
#include <cstddef> // size_t
#include <cstdlib> // std::rand()
#include <iostream>
#include <utility> // std::move, std::swap
#include <initializer_list>

class MatrixD {
public:
    static bool parallelExecution;
    static size_t numThreads;

    MatrixD() = default;
    MatrixD(size_t nRows, size_t nCols) : nRows(nRows), nCols(nCols), data(nRows * nCols) {}
    MatrixD(size_t nRows, size_t nCols, std::vector<double> &data) : nRows(nRows), nCols(nCols), data(data) {}
    MatrixD(size_t nRows, size_t nCols, std::vector<double> &&data) : nRows(nRows), nCols(nCols), data(std::move(data)) {}
    MatrixD(size_t nRows, size_t nCols, double val) : nRows(nRows), nCols(nCols), data(nRows * nCols, val) {}
    MatrixD(std::initializer_list<std::initializer_list<double>> l) {
        if (l.size() == 0) {
            nRows = nCols = 0;
            return;
        }
        size_t listNumCols = (*(l.begin())).size();
        for (const auto &innerList : l) {
            assert(innerList.size() == listNumCols);
        }
        if (listNumCols == 0) {
            nRows = nCols = 0;
            return;
        }
        nRows = l.size();
        nCols = listNumCols;
        data.reserve(nRows * nCols);
        for (const auto &innerList : l) {
            data.insert(data.end(), innerList);
        }
    }

    static MatrixD Random(size_t nRows, size_t nCols) {
        MatrixD res(nRows, nCols);
        for (size_t i = 0; i < res.data.size(); ++i) {
            res.data[i] = static_cast<double>(std::rand()) / RAND_MAX;
        }
        return res;
    }

    static MatrixD Constant(size_t nRows, size_t nCols, double val) {
        return MatrixD(nRows, nCols, val);
    }

    static MatrixD Ones(size_t nRows, size_t nCols) {
        return Constant(nRows, nCols, 1);
    }

    // data is in row-major way
    double operator()(size_t i, size_t j) const {
        if (j >= nCols || i >= nRows)
            throw std::out_of_range("matrix indices out of range");
        return data[nCols * i + j];
    }

    double &operator()(size_t i, size_t j) {
        if (j >= nCols || i >= nRows)
            throw std::out_of_range("matrix indices out of range");
        return data[nCols * i + j];
    }

    MatrixD row(size_t row) const {
        assert(row < nRows);
        return MatrixD(1, nCols, std::vector<double>(data.begin() + row * nCols, data.begin() + (row + 1) * (nCols)));
    }

    MatrixD &transposeInPlace() {
        std::vector<double> new_data(nRows * nCols);
        for (size_t i = 0; i < nRows; ++i) {
            for (size_t j = 0; j < nCols; ++j) {
                new_data[j * nRows + i] = this->operator()(i, j);
            }
        }
        data = std::move(new_data);
        std::swap(nRows, nCols);
        return *this;
    }

    MatrixD transpose() const {
        MatrixD res(*this);
        return res.transposeInPlace();
    }

    MatrixD subblock(size_t startRow, size_t endRow, size_t startCol, size_t endCol) const {
        assert(startRow < endRow && endRow <= nRows);
        assert(startCol < endCol && endCol <= nCols);
        MatrixD res{endRow - startRow, endCol - startCol};
        for (size_t i = 0; i < res.nRows; ++i) {
            for (size_t j = 0; j < res.nCols; ++j) {
                res(i, j) = this->operator()(startRow + i, startCol + j);
            }
        }
        return res;
    }

    MatrixD operator()(const std::vector<size_t> &rowsIndices, const std::vector<size_t> &colsIndices);

    MatrixD cwiseProduct(const MatrixD &other) const {
        MatrixD resMat(*this);
        resMat.cwiseProductInPlace(other);
        return resMat;
    }

    MatrixD &cwiseProductInPlace(const MatrixD &other) {
        this->cwiseBinaryOperationInPlace([] (double el1, double el2) {return el1 * el2;}, other);
        return *this;
    }

    template<class CustomOp> // TODO: add check for non matching dimentions
    MatrixD &cwiseBinaryOperationInPlace(const CustomOp &operation, const MatrixD &other) {
        for (size_t x = 0; x < this->nCols; ++x) {
            for (size_t y = 0; y < this->nRows; ++y) {
                double &el = this->operator()(y, x);
                el = operation(el, other(y, x));
            }
        }
        return *this;
    }

    template<class CustomOp>
    MatrixD &unaryExprInPlace(const CustomOp &operation) {
        MatrixD &thisMatrix = *this;
        for (size_t x = 0; x < this->nCols; ++x) {
            for (size_t y = 0; y < this->nRows; ++y) {
                double &val = thisMatrix(y, x);
                val = operation(val);
            }
        }
        return thisMatrix;
    }

    template<class CustomOp>
    MatrixD unaryExpr(const CustomOp &operation) const {
        MatrixD copyMatrix = *this;
        copyMatrix.unaryExprInPlace<CustomOp>(operation);
        return copyMatrix;
    }

    MatrixD &operator+=(double scalar) {
        this->unaryExprInPlace([scalar] (double el) {
            return el + scalar;
        });
        return *this;
    }

    MatrixD &operator-=(double scalar) {
        return (*this) += -scalar;
    }

    friend std::ostream &operator<<(std::ostream &os, const MatrixD &matrix);

    size_t rows() const {
        return nRows;
    }

    size_t cols() const {
        return nCols;
    }

    double sum() {
        double summ = 0;
        for (size_t i = 0; i < this->nRows; ++i) {
            for (size_t j = 0; j < this->nCols; ++j) {
                summ += this->operator()(i, j);
            }
        }
        return summ;
    }

    static void setParallelExecution(bool parExecution) {
        parallelExecution = parExecution;
    }

    static bool getParallelExecution() {
        return parallelExecution;
    }

    static void setNumberThreads(size_t nThreads) {
        numThreads = nThreads;
    }

    static size_t getNumberThreads() {
        return numThreads;
    }
private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & data;
        ar & nRows;
        ar & nCols;
    }

    std::vector<double> data;
    size_t nRows, nCols;
};

MatrixD primitiveMultiplication(const MatrixD &left, const MatrixD &right);
void rowsMatrixMultiplication(size_t startRow, size_t endRow, const MatrixD &A, const MatrixD &B, MatrixD &C);

MatrixD operator+(const MatrixD &left, const MatrixD &right);
MatrixD operator-(const MatrixD &left, const MatrixD &right);
MatrixD operator+(const MatrixD &left, double scalar);
MatrixD operator-(const MatrixD &left, double scalar);
MatrixD operator*(double scalar, const MatrixD &mat);
MatrixD operator/(double scalar, const MatrixD &mat);
MatrixD operator/(const MatrixD &mat, double scalar);

MatrixD operator*(const MatrixD &left, const MatrixD &right);

bool operator==(const MatrixD& left, const MatrixD& right);
bool operator!=(const MatrixD& left, const MatrixD& right);

template<class CustomOp> // TODO: throw error when sizes do not match
MatrixD cwiseBinaryOperation(const CustomOp &operation, const MatrixD &mat1, const MatrixD &mat2) {
    MatrixD resMat(mat1.rows(), mat1.cols());
    for (size_t x = 0; x < mat1.cols(); ++x) {
        for (size_t y = 0; y < mat1.rows(); ++y) {
            resMat(y, x) = operation(mat1(y, x), mat2(y, x));
        }
    }
    return resMat;
}

#endif //NNLIB_AND_TEST_EXAMPLE_MATRIXD_H
