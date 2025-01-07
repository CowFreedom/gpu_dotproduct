
#include <gtest/gtest.h>
#include <random>
#include <memory>
#include <iostream>

#include <helpers.h>
#include <dotprod/extensions/avx2.h>

using TestAVX2Types = ::testing::Types<ParameterizedFixtureType<double,1123127>,ParameterizedFixtureType<float,31123127>>;

template<typename T>
class DotProductAVX2Test : public DotProductTest<T,DeviceType::CPU> {
	
};

TYPED_TEST_SUITE(DotProductAVX2Test, TestAVX2Types);

TYPED_TEST(DotProductAVX2Test, DotProduct) {
	auto res_control=dot_product_control(TypeParam::vector_size,this->v,this->w);	
	auto res=avx2::dp<0,0>(TypeParam::vector_size,this->v,this->w);
	EXPECT_EQ(res,res_control);
}

TYPED_TEST(DotProductAVX2Test, DotProductParallel) {
	auto res_control=dot_product_control(TypeParam::vector_size,this->v,this->w);
	size_t n_threads=40;
	size_t block_size=50000;
	auto res=dotprod::dp_parallel(n_threads,block_size,TypeParam::vector_size,avx2::dp<0,0>,this->v,this->w);
	EXPECT_EQ(res,res_control);
}
