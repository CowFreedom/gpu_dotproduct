#include <gtest/gtest.h>
#include <random>
#include <memory>
#include <iostream>

#include <helpers.h>

/*
TYPED_TEST_SUITE(DotProductTest, TestTypes);

TYPED_TEST(DotProductTest, DotProduct) {
	if constexpr (std::is_integral<TypeParam::value_type>::value){
		//TypeParam* ptr=reinterpret_cast<TypeParam*>(&(this->w.v));
		auto res_control=dot_product_control(TypeParam::vector_size,this->v,this->w);
	 }
}

TYPED_TEST(DotProductTest, VectorAlignment) {
	//std::cout<<(reinterpret_cast<uintptr_t>(this->w) & 63)<<"\n\n";
	EXPECT_EQ(reinterpret_cast<uintptr_t>(this->w) & 63,false);
}

*/