#include <benchmark/benchmark.h>
#include <helpers.h>
#include <dotprod/util/helpers.h>
#include <dotprod/extensions/avx.h>
#include <thread>
#include <chrono>



BENCHMARK(bm_vector_dotproduct<double,avx::dp<0,0>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,avx::dp<0,0>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,avx::dp_sq<0,0>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,avx::dp_sq<0,0>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->Unit(benchmark::kMillisecond);



template<typename T>
T dp_avx_parallel(size_t n, T* x, T* y){
	
	size_t n_threads=40;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
	//return dotprod::dp_parallel<T>(n, avx::dp<0,0>, x, y);
}

template<typename T>
T dp_avx_parallel_2_threads(size_t n, T* x, T* y){
	
	size_t n_threads=2;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_3_threads(size_t n, T* x, T* y){
	
	size_t n_threads=3;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_4_threads(size_t n, T* x, T* y){
	
	size_t n_threads=4;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_5_threads(size_t n, T* x, T* y){
	
	size_t n_threads=5;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_6_threads(size_t n, T* x, T* y){
	
	size_t n_threads=6;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_7_threads(size_t n, T* x, T* y){
	
	size_t n_threads=7;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_8_threads(size_t n, T* x, T* y){
	
	size_t n_threads=8;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_9_threads(size_t n, T* x, T* y){
	
	size_t n_threads=9;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_10_threads(size_t n, T* x, T* y){
	
	size_t n_threads=10;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_11_threads(size_t n, T* x, T* y){
	
	size_t n_threads=11;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_12_threads(size_t n, T* x, T* y){
	
	size_t n_threads=12;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_20_threads(size_t n, T* x, T* y){
	
	size_t n_threads=20;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_40_threads(size_t n, T* x, T* y){
	
	size_t n_threads=40;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_60_threads(size_t n, T* x, T* y){
	
	size_t n_threads=60;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

template<typename T>
T dp_avx_parallel_80_threads(size_t n, T* x, T* y){
	
	size_t n_threads=80;
	size_t block_size=50000;
	return dotprod::dp_parallel(n_threads,block_size,n,avx::dp<0,0>,x,y);
}

BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_40_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_60_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);

//BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
//BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
/*
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_2_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_3_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_4_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_5_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_6_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_7_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_8_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_9_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_10_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_11_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_12_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_20_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_40_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_60_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<double,dp_avx_parallel_80_threads<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_2_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_3_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_4_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_5_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_6_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_7_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_8_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_9_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_10_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_11_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_12_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_20_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_40_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_60_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_avx_parallel_80_threads<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseRealTime()->Unit(benchmark::kMillisecond);
*/
template<typename T>
T dp_sq_avx_parallel(size_t n, T* x){
	return dotprod::dp_parallel<T>(n, avx::dp_sq<0,0>, x);
}

//BENCHMARK(bm_vector_dotproduct<double,dp_sq_avx_parallel<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(1<<29, 1<<30)->Unit(benchmark::kSecond)->UseRealTime();

