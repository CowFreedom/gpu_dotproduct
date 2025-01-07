#pragma once
#include <gtest/gtest.h>
#include <dotprod/util/helpers.h>
#include <random>
#include <memory>
#include <iostream>

enum class DeviceType{
	CPU,
	GPU
};

template<typename T, size_t N>
class ParameterizedFixtureType{
	
	public:
	using value_type=T;
	static const size_t vector_size=N;
};

template<class T>
T dot_product_control(size_t n, T* a, T* b){
	T sum=T(0);
	for (size_t i=0;i<n;i++){
		sum+=(*a)*(*b);
		a++;
		b++;
	}
	return sum;
}
 
template<typename T, DeviceType DEVICE>
class DotProductTest : public ::testing::Test {
	protected:
	void SetUp() override {
	    std::random_device dev;	
		rng=std::mt19937(dev());
		dist=std::uniform_int_distribution<int>(-20,20); // distribution in range [1, 6]
		redraw_elements();
	}

	void TearDown() override {
			if (DEVICE == DeviceType::GPU){
			#ifdef __CUDACC__ 
				CHECK_CUDA_ERROR(cudaFreeHost(v));
				CHECK_CUDA_ERROR(cudaFreeHost(w));
			//#else
				
			#endif
			}
			else{
				::operator delete[](v, std::align_val_t(64));
				::operator delete[](w, std::align_val_t(64));
			}			
   }

   ~DotProductTest(){
	   if (v != nullptr){
		//	::operator delete(v, std::align_val_t{32});		   
	   }
	   if (w != nullptr){
			//::operator delete(w, std::align_val_t{32});	   				   
	   }

   }
//
	typename T::value_type* v;
	typename T::value_type* w;

    std::mt19937 rng;	
	std::uniform_int_distribution<int> dist;
	
	void redraw_elements(){	
	//Clause due to MSVC Bug https://developercommunity.visualstudio.com/t/using-c17-new-stdalign-val-tn-syntax-results-in-er/528320
	
	if (DEVICE == DeviceType::GPU){
		#ifdef __CUDACC__ 
			CHECK_CUDA_ERROR(cudaMallocHost(&v,sizeof(typename T::value_type) * T::vector_size));
			CHECK_CUDA_ERROR(cudaMallocHost(&w,sizeof(typename T::value_type) * T::vector_size));
		//#else
		//	#error Only CUDA is supported as GPU language
		#endif
	}
	else{
	#ifdef _MSC_VER 
		v = (typename T::value_type *) operator new[](sizeof(typename T::value_type) * T::vector_size, (std::align_val_t)(64));
		w = (typename T::value_type *) operator new[](sizeof(typename T::value_type) * T::vector_size, (std::align_val_t)(64));
	#else
		v=new(std::align_val_t{ 64 }) typename T::value_type[T::vector_size];
		w=new(std::align_val_t{ 64 }) typename T::value_type[T::vector_size];
	#endif		
	}
		

		for (size_t i=0;i<T::vector_size;i++){
			v[i]=dist(rng);
			w[i]=dist(rng);
		}
	}
};

template<typename T>
class DotProductSQTest : public ::testing::Test {
	protected:
	void SetUp() override {
	    std::random_device dev;	
		rng=std::mt19937(dev());
		dist=std::uniform_int_distribution<int>(-1,1); // smaller distribution range to avoid the sum of squares go to infinity
		redraw_elements();
	}

	void TearDown() override {
			::operator delete[](v, std::align_val_t(64));
   }

   ~DotProductSQTest(){
	   if (v != nullptr){
		//	::operator delete(v, std::align_val_t{32});		   
	   }
   }
   
	typename T::value_type* v;

    std::mt19937 rng;	
	std::uniform_int_distribution<int> dist;
	
	void redraw_elements(){	
	//Clause due to MSVC Bug https://developercommunity.visualstudio.com/t/using-c17-new-stdalign-val-tn-syntax-results-in-er/528320
	#ifdef _MSC_VER 
		v = (typename T::value_type *) operator new[](sizeof(typename T::value_type) * T::vector_size, (std::align_val_t)(64));
	#else
		v=new(std::align_val_t{ 64 }) typename T::value_type[T::vector_size];
	#endif
		for (size_t i=0;i<T::vector_size;i++){
			v[i]=dist(rng);
		}
	}
};

using TestTypes = ::testing::Types<ParameterizedFixtureType<int,1000>>;