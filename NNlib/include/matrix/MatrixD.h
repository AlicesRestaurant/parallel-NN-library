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
#include <iterator>
#include <memory> // for shared_ptr

template<typename Container>
class ViewOfData {
public:
    class Iterator {
    public:
        Iterator(ViewOfData &enclosing, typename Container::iterator iter): enclosing(enclosing), iter(iter) {}
        typename Container::value_type &operator*() {
            return *iter;
        }
        Iterator& operator++() {
            iter += enclosing.step;
            return *this;
        }
        Iterator operator++(int) {
            Iterator saved = *this;
            iter += enclosing.step;
            return saved;
        }
        Iterator& operator+=(size_t incr) {
            iter += enclosing.step * incr;
            return *this;
        }
        Iterator& operator-=(int) {
            iter -= enclosing.step;
            return *this;
        }
        Iterator& operator==(Iterator other) {
            return iter == other.iter;
        }
        Iterator& operator!=(Iterator other) {
            return iter != other.iter;
        }
    private:
        ViewOfData &enclosing;
        typename Container::iterator iter;
    };

    ViewOfData() = default;
    ViewOfData(const std::shared_ptr<Container> &cPtr): dataPtr(cPtr), start(0), step(1), nSteps(cPtr->size()) {}
    ViewOfData(const Container &c, size_t start, size_t step, size_t nSteps): dataPtr(std::make_shared<Container>(c)),
                                                                            start(start),
                                                                            step(step),
                                                                            nSteps(nSteps)
    {}
    ViewOfData(const ViewOfData &c, size_t start, size_t step, size_t nSteps): dataPtr(c.dataPtr),
                                                        start(c.start * start + c.step),
                                                        step(c.start * step),
                                                        nSteps(nSteps) {}
    ViewOfData(const std::shared_ptr<Container> &cPtr, size_t oldStart, size_t oldStep, size_t oldNSteps):
    dataPtr(std::make_shared<Container>()),
    start(0),
    step(1),
    nSteps(cPtr->size())
    {
        (*dataPtr).reserve(oldNSteps);
        for (size_t i = 0; i < oldNSteps; ++i) {
            (*dataPtr).push_back((*cPtr)[oldStart + i * oldStep]);
        }
        start = 0;
        step = 1;
        nSteps = dataPtr->size();
    }
    Iterator begin() {
        return Iterator(*this, dataPtr->begin() + start);
    }
    Iterator end() {
        return Iterator(*this, dataPtr->begin() + start + step * nSteps);
    }
    typename Container::value_type &operator[](size_t idx) {
        return (*dataPtr)[start + step * idx];
    }
    typename Container::value_type operator[](size_t idx) const {
        return (*dataPtr)[start + step * idx];
    }
    [[nodiscard]] size_t size() const {
        return nSteps;
    }
    [[nodiscard]] const Container &getData() const {
        return *dataPtr;
    }
    [[nodiscard]] size_t getStart() const {
        return start;
    }
    [[nodiscard]] size_t getStep() const {
        return step;
    }
    [[nodiscard]] size_t getNSteps() const {
        return nSteps;
    }
private:
    std::shared_ptr<Container> dataPtr;
    size_t start{};
    size_t step{};
    size_t nSteps{};

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & (*dataPtr);
        ar & start;
        ar & step;
        ar & nSteps;
    }
};

class MatrixD {
public:
    using ContainerType = std::vector<double>;
    static bool parallelExecution;
    static size_t numThreads;

public:
    MatrixD() = default; //TODO
    MatrixD(const MatrixD &other): nRows(other.nRows), nCols(other.nCols), data(other.data.getData(), other.data.getStart(),
                                                                                other.data.getStep(), other.data.getNSteps())
                                                                                {}
    MatrixD(size_t nRows, size_t nCols) : nRows(nRows),
                                        nCols(nCols),
                                        data(std::make_shared<ContainerType>(nRows * nCols)) {}
    MatrixD(size_t nRows, size_t nCols, std::vector<double> &data) : nRows(nRows),
                                                                    nCols(nCols),
                                                                    data(std::make_shared<ContainerType>(data)) {}
    MatrixD(size_t nRows, size_t nCols, std::vector<double> &&data) :
                                                                    nRows(nRows),
                                                                    nCols(nCols),
                                                                    data(std::make_shared<ContainerType>(std::move(data))) {}
    MatrixD(size_t nRows, size_t nCols, double val) :
                                                nRows(nRows),
                                                nCols(nCols),
                                                data(std::make_shared<ContainerType>(nRows * nCols, val)) {}
    MatrixD(std::initializer_list<std::initializer_list<double>> l);
private:
    MatrixD(size_t nRows, size_t nCols, ViewOfData<ContainerType> c): nRows(nRows), nCols(nCols), data(c) {}
public:

    static MatrixD Random(size_t nRows, size_t nCols) {
        MatrixD res(nRows, nCols);
        for (size_t i = 0; i < res.data.size(); ++i) {
            res.data[i] = static_cast<double>(std::rand()) / RAND_MAX * 2 - 1;
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
    virtual double operator()(size_t i, size_t j) const {
        if (j >= nCols || i >= nRows)
            throw std::out_of_range("matrix indices out of range");
        return data[nCols * i + j];
    }

    virtual double &operator()(size_t i, size_t j) {
        if (j >= nCols || i >= nRows)
            throw std::out_of_range("matrix indices out of range");
        return data[nCols * i + j];
    }

    virtual MatrixD row(size_t row) const {
        assert(row < nRows);
        return MatrixD(1, nCols, ViewOfData<ContainerType>(data, row * nCols, 1, nCols));
    }

    virtual MatrixD &transposeInPlace();

    virtual MatrixD transpose() const {
        MatrixD res(*this);
        return res.transposeInPlace();
    }

    virtual MatrixD subblock(size_t startRow, size_t endRow, size_t startCol, size_t endCol) const {
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

    virtual MatrixD operator()(const std::vector<size_t> &rowsIndices, const std::vector<size_t> &colsIndices);

    virtual MatrixD cwiseProduct(const MatrixD &other) const {
        MatrixD resMat(*this);
        resMat.cwiseProductInPlace(other);
        return resMat;
    }

    virtual MatrixD &cwiseProductInPlace(const MatrixD &other) {
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
        for (size_t x = 0; x < nCols; ++x) {
            for (size_t y = 0; y < nRows; ++y) {
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

    virtual MatrixD &operator+=(double scalar) {
        unaryExprInPlace([scalar] (double el) {
            return el + scalar;
        });
        return *this;
    }

    virtual MatrixD &operator-=(double scalar) {
        return (*this) += -scalar;
    }

    virtual MatrixD &operator*=(double scalar) {
        unaryExprInPlace([scalar] (double el) {
            return el * scalar;
        });
        return *this;
    }

    friend std::ostream &operator<<(std::ostream &os, const MatrixD &matrix);

    virtual size_t rows() const {
        return nRows;
    }

    virtual size_t cols() const {
        return nCols;
    }

    virtual double sum() {
        double summ = 0;
        for (size_t i = 0; i < this->nRows; ++i) {
            for (size_t j = 0; j < this->nCols; ++j) {
                summ += this->operator()(i, j);
            }
        }
        return summ;
    }

    template <class BinaryCommutativeOperator>
    MatrixD rowReduce(const BinaryCommutativeOperator &func, double init_val) {
        MatrixD res(nRows, 1);
        for (size_t i = 0; i < nRows; ++i) {
            double val = init_val;
            for (size_t j = 0; j < nCols; ++j) {
                val = func(val, this->operator()(i, j));
            }
            res(i, 0) = val;
        }
        return res;
    }

    template <class BinaryCommutativeOperator>
    MatrixD colReduce(const BinaryCommutativeOperator &func, double init_val) {
        MatrixD res(1, nCols);
        for (size_t j = 0; j < this->nCols; ++j) {
            double val = init_val;
            for (size_t i = 0; i < this->nRows; ++i) {
                val = func(val, this->operator()(i, j));
            }
            res(0, j) = val;
        }
        return res;
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

    ViewOfData<ContainerType> data;
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
