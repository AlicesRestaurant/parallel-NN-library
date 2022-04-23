#include <gtest/gtest.h>
#include <Model.h>
#include <lossfunction/MSELossFunction.h>

#include <memory>

TEST(TestSuiteName, TestName) {
    EXPECT_STRNE("hello", "world");
    EXPECT_EQ(1 + 1, 2);
}

TEST(ModelTest, Xor) {
    Model model{2, std::make_shared<MSELossFunction>()};
}
