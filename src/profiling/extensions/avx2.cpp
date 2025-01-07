#include <benchmark/benchmark.h>
#include <helpers.h>
#include <dotprod/util/helpers.h>
#include <dotprod/extensions/avx2.h>
#include <thread>
#include <chrono>



BENCHMARK(bm_vector_dotproduct<double,avx2::dp<0,0>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,avx2::dp<0,0>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->Unit(benchmark::kMillisecond);



template<typename T>
T dp_avx2_parallel(size_t n, T* x, T* y){
	size_t n_threads=40;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx2::dp<0,0>,x,y);
}

template<typename T>
T dp_avx2_parallel_40_threads(size_t n, T* x, T* y){
	size_t n_threads=40;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx2::dp<0,0>,x,y);
}

template<typename T>
T dp_avx2_parallel_60_threads(size_t n, T* x, T* y){
	size_t n_threads=60;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx2::dp<0,0>,x,y);
}



BENCHMARK(bm_vector_dotproduct<double,dp_avx2_parallel_40_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx2_parallel_60_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);