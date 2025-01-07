#include <gtest/gtest.h>
#include <random>
#include <memory>
#include <iostream>

#include <helpers.h>
#include <dotprod/extensions/avx.h>

using TestAVXTypes = ::testing::Types<ParameterizedFixtureType<double,(1<<27)>,ParameterizedFixtureType<float,(1<<27)>>;

template<typename T>
class DotProductAVXTest : public DotProductTest<T,DeviceType::CPU> {
	
};

TYPED_TEST_SUITE(DotProductAVXTest, TestAVXTypes);

TYPED_TEST(DotProductAVXTest, DotProduct) {
	auto res_control=dot_product_control(TypeParam::vector_size,this->v,this->w);
	
	auto res=avx::dp<0,0>(TypeParam::vector_size,this->v,this->w);
	EXPECT_EQ(res,res_control);
}

TYPED_TEST(DotProductAVXTest, DotProductParallel) {
	auto res_control=dot_product_control(TypeParam::vector_size,this->v,this->w);
	size_t n_threads=40;
	size_t block_size=50000;
	auto res=dotprod::dp_parallel(n_threads,block_size,TypeParam::vector_size,avx::dp<0,0>,this->v,this->w);
	EXPECT_EQ(res,res_control);
}



template<typename T>
class DotProductSQAVXTest : public DotProductSQTest<T> {
	
};

using TestAVXSQTypes = ::testing::Types<ParameterizedFixtureType<double,(1<<27)>,ParameterizedFixtureType<float,(1<<20)>>;

TYPED_TEST_SUITE(DotProductSQAVXTest, TestAVXSQTypes);

TYPED_TEST(DotProductSQAVXTest, DotProduct) {
	auto res_control=dot_product_control(TypeParam::vector_size,this->v,this->v);

	auto res=avx::dp_sq<0,0>(TypeParam::vector_size,this->v);
	EXPECT_EQ(res,res_control);
}
