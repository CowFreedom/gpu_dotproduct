#include <gtest/gtest.h>
#include <random>
#include <memory>
#include <iostream>

#include <helpers.h>
#include <dotprod/gpu/dp.h>

using TestGPUTypes = ::testing::Types<ParameterizedFixtureType<double,12243600>,ParameterizedFixtureType<float,123600>>;

template<typename T>
class DotProductGPUTest : public DotProductTest<T,DeviceType::GPU> {
	
};

TYPED_TEST_SUITE(DotProductGPUTest, TestGPUTypes);

TYPED_TEST(DotProductGPUTest, DotProduct) {
	auto res_control=avx::dp<0,0>(TypeParam::vector_size,this->v,this->w);
	auto res=gpu::dp<0,0>(TypeParam::vector_size,this->v,this->w,true);
	EXPECT_EQ(res,res_control);
}

TYPED_TEST(DotProductGPUTest, DotProductSQ) {

	auto res_control=avx::dp_sq<0,0>(TypeParam::vector_size,this->v);
	auto res=gpu::dp<0,0>(TypeParam::vector_size,this->v,true);
	EXPECT_EQ(res,res_control);
}