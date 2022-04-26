#include <gtest/gtest.h>
#include <Model.h>
#include <lossfunction/MSELossFunction.h>
#include <lossfunction/SoftMaxLossFunction.h>

#include <memory>
#include <cmath> // for std::log(), std::exp()

TEST(TestSuiteName, TestName) {
    EXPECT_STRNE("hello", "world");
    EXPECT_EQ(1 + 1, 2);
}

TEST(ModelTest, Xor) {
    Model model{2, std::make_shared<MSELossFunction>()};
}

TEST(LossTest, SoftMaxLossFunctionTest) {
    using Eigen::MatrixXd;
    SoftMaxLossFunction loss;
    double actual = loss.forwardPropagate(MatrixXd{{2, 3},
                                                   {1, -3}},
                                          MatrixXd{{0, 1},
                                                   {1, 0}});
    using std::exp, std::log;
    double expected = - 1.0 / 2 * (log(exp(1) / (exp(2) + exp(1))) +
            log(exp(3) / (exp(3) + exp(-3))));
    EXPECT_DOUBLE_EQ(actual, expected);
}