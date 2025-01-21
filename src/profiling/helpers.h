#pragma once
#include <benchmark/benchmark.h>
#include <dotprod/util/helpers.h>
#include <random>
#include <memory>
#include <iostream>
//#include <windows.h>

constexpr int PERFORMANCE_TEST_LOWER_LIMIT=64;
constexpr int PERFORMANCE_TEST_UPPER_LIMIT=1<<27;

enum class DeviceType{
	CPU,
	GPU
};

template<typename T, DeviceType DEVICE>
T* initialize_aligned_memory_randomly(size_t n){
	T* arr;
	if (DEVICE == DeviceType::GPU){
	#ifdef __CUDACC__
		CHECK_CUDA_ERROR(cudaMallocHost(&arr,sizeof(T) * n));
	#endif
	}else{
	#ifdef _MSC_VER 
		arr= (T *) operator new[](sizeof(T) * n, (std::align_val_t)(64));
	#else
		arr= new(std::align_val_t{ 64 }) T[n];
	#endif
	}
	std::random_device dev;	
	auto rng=std::mt19937(dev());
	auto dist=std::uniform_int_distribution<std::mt19937::result_type>(0,100); // distribution in range [1, 6]
	for (size_t i=0;i<n;i++){
		arr[i]=dist(rng);		
	}
	return arr;
}

template<typename T, DeviceType DEVICE>
void release_aligned_memory_randomly(T* r){
	if (DEVICE == DeviceType::GPU){
		#ifdef __CUDACC__ 
		CHECK_CUDA_ERROR(cudaFreeHost(r));	
		#endif
	}
	else{
	::operator delete[](r, std::align_val_t(64));
	}
}

template<class T>
T dp_control(size_t n, T* a, T* b){
	T sum=T(0);
	for (size_t i=0;i<n;i++){
		sum+=(*a)*(*b);
		a++;
		b++;
	}
	return sum;
}
 

//see https://ashvardanian.com/posts/google-benchmark/
template<typename T, T (*F)(size_t, T*,T*), DeviceType DEVICE>
static void bm_vector_dotproduct(benchmark::State& state) {	
	T* v;
	T* w;
	T sum=T(0);
	auto count = static_cast<size_t>(state.range(0));
	if (state.thread_index() == 0) {
		v = initialize_aligned_memory_randomly<T,DEVICE>(count);
		w = initialize_aligned_memory_randomly<T,DEVICE>(count);
	}
	if (DEVICE == DeviceType::GPU){
		auto start = std::chrono::high_resolution_clock::now();
		for (auto _ : state) {
			v[0]++;
			w[0]++;
			benchmark::DoNotOptimize(sum+=F(count,v,w));	
		}
		auto end = std::chrono::high_resolution_clock::now();

		auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

		state.SetIterationTime(elapsed_seconds.count());
	}
	else{
		for (auto _ : state) {
			v[0]++;
			w[0]++;
			benchmark::DoNotOptimize(sum+=F(count,v,w));	
		}
	}


	if (state.thread_index() == 0) {
		release_aligned_memory_randomly<T,DEVICE>(v);
		release_aligned_memory_randomly<T,DEVICE>(w);
	}
}

//see https://ashvardanian.com/posts/google-benchmark/
template<typename T, T (*F)(size_t,T*), DeviceType DEVICE>
static void bm_vector_dotproduct(benchmark::State& state) {	
	T* v;
	T sum=T(0);
	auto count = static_cast<size_t>(state.range(0));
	if (state.thread_index() == 0) {
		v = initialize_aligned_memory_randomly<T,DEVICE>(count);
	}
	
	if (DEVICE == DeviceType::GPU){
		auto start = std::chrono::high_resolution_clock::now();
		for (auto _ : state) {
			v[0]++;
			benchmark::DoNotOptimize(sum+=F(count,v));	
		}
		auto end = std::chrono::high_resolution_clock::now();

		auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

		state.SetIterationTime(elapsed_seconds.count());
	}
	else{
		for (auto _ : state) {
			v[0]++;
			benchmark::DoNotOptimize(sum+=F(count,v));	
		}
	}	

	if (state.thread_index() == 0) {
		release_aligned_memory_randomly<T,DEVICE>(v);
	}
	
}
