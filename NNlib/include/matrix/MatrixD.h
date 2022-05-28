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
#include <limits> // for numeric_limits<>

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
    ViewOfData(const ViewOfData&) = default; // Copy inner array
    ViewOfData(ViewOfData&&) = default; // Move inner array
    ViewOfData &operator=(const ViewOfData&) = default; // Copy inner array
    ViewOfData &operator=(ViewOfData&&) = default; // Move inner array
    // Construct view on passed container
    ViewOfData(const Container &cont, size_t h, size_t w):
                dataPtr(std::make_shared<Container>(cont)),
                start(0),
                height(h),
                fullHeight(h),
                width(w),
                fullWidth(w) {}
    // Construct view on passed container
    ViewOfData(const Container &&cont, size_t h, size_t w):
            dataPtr(std::make_shared<Container>(std::move(cont))),
            start(0),
            height(h),
            fullHeight(h),
            width(w),
            fullWidth(w) {}
    // Construct view on view
    ViewOfData(const ViewOfData &on, size_t startI, size_t startJ, size_t width, size_t height):
    dataPtr(on.dataPtr),
    start(startI * on.fullWidth + startJ),
    height(height),
    width(width),
    fullHeight(on.fullHeight),
    fullWidth(on.fullWidth)
    {}
    // Return new view on copy of the data
    ViewOfData copy() const {
        Container con(width * height);
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                con[i * width + j] = operator()(i, j);
            }
        }
        return ViewOfData(con, height, width);
    }
//    Iterator begin() {
//        return Iterator(*this, dataPtr->begin() + start);
//    }
//    Iterator end() {
//        return Iterator(*this, dataPtr->begin() + start + width + height * fullWidth);
//    }
    typename Container::value_type &operator()(size_t i, size_t j) {
        return (*dataPtr)[start + j + i * fullWidth];
    }
    typename Container::value_type operator()(size_t i, size_t j) const {
        return (*dataPtr)[start + j + i * fullWidth];
    }
    long getNumViews() {
        return dataPtr.use_count();
    }
//    [[nodiscard]] size_t size() const {
//        return nSteps;
//    }
//    [[nodiscard]] const Container &getData() const {
//        return *dataPtr;
//    }
//    [[nodiscard]] size_t getStart() const {
//        return start;
//    }
    [[nodiscard]] size_t getWidth() const {
        return width;
    }
    [[nodiscard]] size_t getHeight() const {
        return height;
    }
private:
    std::shared_ptr<Container> dataPtr;
    size_t start;
    size_t fullWidth;
    size_t fullHeight;
    size_t width;
    size_t height;

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & (*dataPtr);
        ar & start;
        ar & fullWidth;
        ar & fullHeight;
        ar & width;
        ar & height;
    }
};

class MatrixD {
public:
    using ContainerType = std::vector<double>;
    static bool parallelExecution;
    static size_t numThreads;

public:
    MatrixD() = default; //TODO
    MatrixD(const MatrixD& other): nRows(other.nRows), nCols(other.nCols), data(other.data.copy()) {} // Copy content
    MatrixD(MatrixD&& other): nRows(other.nRows), nCols(other.nCols), data(other.data) {
        if (other.data.getNumViews() > 1) {
            data = other.data.copy();
        } else {
            data = std::move(other.data);
        }
    } // move if only one view
    MatrixD &operator=(const MatrixD& other) {
        nRows = other.nRows;
        nCols = other.nCols;
        data = other.data.copy();
        return *this;
    } // Copy content
    MatrixD &operator=(MatrixD&& other) {
        nRows = other.nRows;
        nCols = other.nCols;
        if (other.data.getNumViews() > 1) {
            data = other.data.copy();
        } else {
            data = std::move(other.data);
        }
        return *this;
    }; // move if only one view
    MatrixD(size_t nRows, size_t nCols) : nRows(nRows),
                                        nCols(nCols),
                                        data(ContainerType(nRows * nCols), nRows, nCols) {}
    MatrixD(size_t nRows, size_t nCols, const ContainerType &container) : nRows(nRows),
                                                                    nCols(nCols),
                                                                    data(container, nRows, nCols) {}
    MatrixD(size_t nRows, size_t nCols, ContainerType &&container) :
                                                                    nRows(nRows),
                                                                    nCols(nCols),
                                                                    data(std::move(container), nRows, nCols) {}
    MatrixD(size_t nRows, size_t nCols, double val) :
                                                nRows(nRows),
                                                nCols(nCols),
                                                data(ContainerType(nRows * nCols, val), nRows, nCols) {}
    MatrixD(std::initializer_list<std::initializer_list<double>> l);
private:
    MatrixD(ViewOfData<ContainerType> view): nRows(view.getHeight()), nCols(view.getWidth()), data(view) {}
public:

    static MatrixD Random(size_t nRows, size_t nCols) {
        MatrixD res(nRows, nCols);
        for (size_t i = 0; i < res.nRows; ++i) {
            for (size_t j = 0; j < res.nCols; ++j) {
                res.data(i, j) = static_cast<double>(std::rand()) / RAND_MAX * 2 - 1;
            }
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
        return data(i, j);
    }

    virtual double &operator()(size_t i, size_t j) {
        if (j >= nCols || i >= nRows)
            throw std::out_of_range("matrix indices out of range");
        return data(i, j);
    }

    virtual MatrixD &transposeInPlace();

    virtual MatrixD transpose() const {
        MatrixD res(*this);
        return res.transposeInPlace();
    }

    virtual MatrixD subblock(size_t startRow, size_t endRow, size_t startCol, size_t endCol) {
        assert(startRow < endRow && endRow <= nRows);
        assert(startCol < endCol && endCol <= nCols);
        return MatrixD(ViewOfData<ContainerType>(data, startRow, startCol, endCol - startCol, endRow - startRow));
    }

    virtual MatrixD row(size_t rowIdx) {
        assert(0 <= rowIdx && rowIdx < nRows);
        return subblock(rowIdx, rowIdx + 1, 0, nCols);
    }

    virtual MatrixD col(size_t colIdx) {
        assert(0 <= colIdx && colIdx < nCols);
        return subblock(0, nRows, colIdx, colIdx + 1);
    }

    virtual void maxCoeff(size_t *iIdx = nullptr, size_t *jIdx = nullptr) {
        double maxEl = std::numeric_limits<double>::lowest();
        size_t maxI = -1;
        size_t maxJ = -1;
        for (size_t i = 0; i < nRows; ++i) {
            for (size_t j = 0; j < nCols; ++j) {
                double curEl;
                if (maxEl < (curEl = operator()(i, j))) {
                    maxEl = curEl;
                    maxI = i;
                    maxJ = j;
                }
            }
        }
        if (iIdx != nullptr) {
            *iIdx = maxI;
        }
        if (jIdx != nullptr) {
            *jIdx = maxJ;
        }
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

    virtual MatrixD &operator/=(MatrixD right) {
        cwiseBinaryOperationInPlace([] (double el1, double el2) {
            return el1 / el2;
        }, right);
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
MatrixD operator/(const MatrixD &left, const MatrixD &right);

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
