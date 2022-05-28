#include <gtest/gtest.h>
#include <Model.h>
#include <lossfunction/MSELossFunction.h>
#include <matrix/MatrixD.h>

#include <memory>
#include <cmath> // for std::log(), std::exp()
#include <vector>

TEST(TestSuiteName, TestName) {
    EXPECT_STRNE("hello", "world");
    EXPECT_EQ(1 + 1, 2);
}

TEST(ModelTest, Xor) {
    Model model{2, std::make_shared<MSELossFunction>()};
}

//TEST(LossTest, SoftMaxLossFunctionTest) {
//    using Eigen::MatrixXd;
//    SoftMaxLossFunction loss;
//    double actual = loss.forwardPropagate(MatrixXd{{2, 3},
//                                                   {1, -3}},
//                                          MatrixXd{{0, 1},
//                                                   {1, 0}});
//    using std::exp, std::log;
//    double expected = - 1.0 / 2 * (log(exp(1) / (exp(2) + exp(1))) +
//            log(exp(3) / (exp(3) + exp(-3))));
//    EXPECT_DOUBLE_EQ(actual, expected);
//}

TEST(MatrixTest, Constructor) {
    MatrixD m1{3, 2};
    ASSERT_EQ(m1(0, 0), double{});
    ASSERT_EQ(m1(0, 1), double{});
    ASSERT_EQ(m1(1, 0), double{});
    ASSERT_EQ(m1(1, 1), double{});
    ASSERT_EQ(m1(2, 0), double{});
    ASSERT_EQ(m1(2, 1), double{});

    MatrixD m2(4, 2, std::vector<double>{1, 2,
                                         3, 4,
                                         5, 6,
                                         7, 8});
    MatrixD m3(4, 2, std::vector<double>{1, 2,
                                         4, 4,
                                         5, 6,
                                         7, 8});
    ASSERT_NE(m2, m3);
    m3(1, 0) = 3;
    ASSERT_EQ(m2, m3);

    ASSERT_NE(MatrixD(1, 1), m2);


    MatrixD m4{{1}};
    MatrixD m5{{1, 2}};
    MatrixD m6{{3}, {4}};
    MatrixD m7{{1, 2}, {3, 4}};
    MatrixD m8{};

    EXPECT_EQ(m4.rows(), 1);
    EXPECT_EQ(m4.cols(), 1);
    EXPECT_EQ(m5.rows(), 1);
    EXPECT_EQ(m5.cols(), 2);
    EXPECT_EQ(m6.rows(), 2);
    EXPECT_EQ(m6.cols(), 1);
    EXPECT_EQ(m7.rows(), 2);
    EXPECT_EQ(m7.cols(), 2);
    EXPECT_EQ(m8.rows(), 0);
    EXPECT_EQ(m8.cols(), 0);

    MatrixD m9{{}, {}}; // reduces to {}
    EXPECT_EQ(m9.rows(), 0);
    EXPECT_EQ(m9.rows(), 0);

    EXPECT_EQ(m4(0, 0), 1);

    EXPECT_EQ(m5(0, 0), 1);
    EXPECT_EQ(m5(0, 1), 2);

    EXPECT_EQ(m6(0, 0), 3);
    EXPECT_EQ(m6(1, 0), 4);

    EXPECT_EQ(m7(0, 0), 1);
    EXPECT_EQ(m7(0, 1), 2);
    EXPECT_EQ(m7(1, 0), 3);
    EXPECT_EQ(m7(1, 1), 4);

    EXPECT_EQ(m8, MatrixD());
    EXPECT_EQ(m9, MatrixD());
}

TEST(MatrixTest, Multiplication_1) {
    size_t mat_m = 5, mat_n = 4;
    std::vector<double> v1{1, 5, 8, 6,
                           1, 3, 7, 6,
                           1, 3, 4, 6,
                           1, 2, 7, 6,
                           7, 2, 3, 4};
    std::vector<double> v2{1, 3, 8, 6,
                           1, 3, 9, 6,
                           1, 1, 1, 1,
                           1, 2, 7, 6};
    std::vector<double> v3{20, 38, 103, 80,
                           17, 31, 84, 67,
                           14, 28, 81, 64,
                           16, 28, 75, 61,
                           16, 38, 105, 81};
    MatrixD mA(mat_m, mat_n, v1);
    MatrixD mB(mat_n, mat_n, v2);
    MatrixD mC(mat_m, mat_n, v3);
    EXPECT_EQ(mA * mB, mC);
}

TEST(MatrixTest, Multiplication_2) {
    std::vector<double> v1{-2};
    std::vector<double> v2{5};
    std::vector<double> v3{-10};
    MatrixD mA(1, 1, v1);
    MatrixD mB(1, 1, v2);
    MatrixD mC(1, 1, v3);
    EXPECT_EQ(mA * mB, mC);
}

TEST(MatrixTest, Multiplication_3) {
    std::vector<double> v1{-2, 0.5};
    std::vector<double> v2{4, 6};
    std::vector<double> v3{-5.0}; // TODO: Fix operator== with doubles
    MatrixD mA(1, 2, std::move(v1));
    MatrixD mB(2, 1, std::move(v2));
    MatrixD mC(1, 1, std::move(v3));
    EXPECT_EQ(mA * mB, mC);
}

TEST(MatrixTest, Transpose) {
    std::vector<double> v1{1, 5, 8, 6,
                           1, 3, 7, 6,
                           1, 3, 4, 6,
                           1, 2, 7, 6,
                           7, 2, 3, 4};
    std::vector<double> v2{1, 3, 8, 6,
                           1, 3, 9, 6,
                           1, 1, 1, 1,
                           1, 2, 7, 6};
    std::vector<double> v3{20, 38, 103, 80};
    MatrixD mA(5, 4, v1);
    MatrixD mB(4, 4, v2);
    MatrixD mC(1, 4, v3);
    mA.transposeInPlace();
    mA.transposeInPlace();
    mA.transposeInPlace();
    mB.transposeInPlace();
    mC.transposeInPlace();
    EXPECT_EQ(mA.rows(), 4);
    EXPECT_EQ(mA.cols(), 5);
    EXPECT_EQ(mB.rows(), 4);
    EXPECT_EQ(mB.cols(), 4);
    EXPECT_EQ(mC.rows(), 4);
    EXPECT_EQ(mC.cols(), 1);

    EXPECT_EQ(mA, mA.transposeInPlace().transposeInPlace());
    EXPECT_EQ(mB, mB.transposeInPlace().transposeInPlace());
    EXPECT_EQ(mC, mC.transposeInPlace().transposeInPlace());
    EXPECT_EQ(mA, MatrixD(4, 5, std::vector<double>{1, 1, 1, 1, 7,
                                            5, 3, 3, 2, 2,
                                            8, 7, 4, 7, 3,
                                            6, 6, 6, 6, 4}));
    EXPECT_EQ(mB, MatrixD(4, 4, std::vector<double>{1, 1, 1, 1,
                                                    3, 3, 1, 2,
                                                    8, 9, 1, 7,
                                                    6, 6, 1, 6}));
    EXPECT_EQ(mC, MatrixD(4, 1, std::vector<double>{20,
                                                    38,
                                                    103,
                                                    80}));
}

TEST(MatrixTest, CreatingMethods) {
    MatrixD m = MatrixD::Constant(2, 3, 3.52);
    MatrixD expected_m = MatrixD{{3.52, 3.52, 3.52},
                                 {3.52, 3.52, 3.52}};
    EXPECT_EQ(m, expected_m);

    MatrixD m1 = MatrixD::Ones(2, 4);
    MatrixD expected_m1 = MatrixD{{1, 1, 1, 1},
                                  {1, 1, 1, 1}};
    EXPECT_EQ(m1, expected_m1);

    for (int n = 0; n < 10; ++n) {
        MatrixD m2 = MatrixD::Ones(5, 1);
        EXPECT_EQ(m2.rows(), 5);
        EXPECT_EQ(m2.cols(), 1);
        for (size_t i = 0; i < m2.rows(); ++i) {
            for (size_t j = 0; j < m2.cols(); ++j) {
                ASSERT_LE(m2(i, j), 1);
                ASSERT_GE(m2(i, j), 0);
            }
        }
    }
}

TEST(MatrixTest, Subblock) {
    MatrixD m{{1, 2, 3},
              {4, 5, 6},
              {7, 8, 9},
              {11, 12, 13}};
    ASSERT_EQ(m.subblock(0, m.rows(), 0, m.cols()), m);

    ASSERT_EQ(m.subblock(0, 1, 0, 1), MatrixD{{1}});

    MatrixD s = m.subblock(1, 3, 1, 2);
    MatrixD expected{{5},
                     {8}};
    ASSERT_EQ(s, expected);
}

TEST(MatrixTest, CwiseProduct) {
    MatrixD m1{{1, 2, 3},
               {4, 5, 6}};
    MatrixD m2{{1, 0, 0},
               {1, 1, 1}};
    MatrixD res{{1, 0, 0},
                {4, 5, 6}};
    EXPECT_EQ(m1.cwiseProduct(m2), res);
}

TEST(MatrixTest, CwiseProductInPlace) {
    MatrixD m1{{-1, 2}, {5, 6}};
    MatrixD m2{{1, 0}, {1, 1}};
    m1.cwiseProductInPlace(m2);
    MatrixD res{{-1, 0}, {5, 6}};
    EXPECT_EQ(m1, res);
}

TEST(MatrixTest, CwiseBinaryOperationInPlace) {
    MatrixD m1{{-1, 2, 1}, {1, 5, 6}};
    MatrixD m2{{1, 0, 5}, {1, 1, 1}};
    MatrixD res{{-1, 1, -2}, {0, 2, 2.5}};
    m1.cwiseBinaryOperationInPlace([] (double e1, double e2) {return (e1 - e2) / 2;}, m2);
    EXPECT_EQ(m1, res);
}

TEST(MatrixTest, unaryExprInPlace) {
    MatrixD m1{{-1, 2, 1}, {1, 5, 6}};
    MatrixD res{{-1, 8, 1}, {1, 125, 216}};
    EXPECT_EQ(m1.unaryExpr([] (double x) {return x*x*x;}), res);
    EXPECT_EQ(m1.unaryExprInPlace([] (double x) {return x*x*x;}), res);
}

TEST(MatrixTest, ArithmeticOperators) {
    MatrixD m1{{-1, 2, 1}, {1, 5, 6}};
    MatrixD res{{0, 3, 2}, {2, 6, 7}};
    EXPECT_EQ(m1 + 1, res);
    EXPECT_EQ(m1 + 0.5, res - 0.5);
    EXPECT_EQ((m1 + res) / 2 + (res - m1) / 2, res);
}

TEST(MatrixTest, Aggregate) {
    MatrixD m1{{1}};
    EXPECT_EQ(m1.sum(), 1);
    MatrixD m2{{1, 2}};
    EXPECT_EQ(m2.sum(), 3);
    MatrixD m3{{1, 2},
               {-1, -2}};
    EXPECT_EQ(m3.sum(), 0);
}

TEST(MatrixTest, Reduce) {
    MatrixD m1{{1, 2}, {3, 4}, {5, 6}};
    MatrixD res1 = m1.rowReduce([] (double x1, double x2) {return x1 + x2;}, 0);
    MatrixD expected1{{3}, {7}, {11}};
    EXPECT_EQ(res1, expected1);

    MatrixD res2 = m1.colReduce([] (double x1, double x2) {return x1 * x2;}, 1);
    MatrixD expected2{{15, 48}};
    EXPECT_EQ(res2, expected2);

    expected2 *= -2;
    MatrixD expected3{{-30, -96}};
    EXPECT_EQ(expected2, expected3);
}

TEST(MatrixTest, Subblock2) {
    MatrixD m1{{1, 2, 3},
               {4, 5, 6},
               {7, 8, 9}};
    MatrixD expected1{{2, 3},
                      {5, 6},
                      {8, 9}};
    MatrixD expected12{{5}};
    EXPECT_EQ(m1.subblock(1, 2, 1, 2), expected12);

    m1.subblock(0, 3, 0, 3) *= 2;
    MatrixD expected13{{2, 4, 6},
                       {8, 10, 12},
                       {14, 16, 18}};
    EXPECT_EQ(m1, expected13);
}