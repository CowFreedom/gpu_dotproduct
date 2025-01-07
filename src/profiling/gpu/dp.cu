#include <benchmark/benchmark.h>
#include <dotprod/gpu/dp.h>
#include <helpers.h>


template<size_t STRIDE_X, size_t STRIDE_Y, class T>
T dp_gpu(size_t n, T* x, T* y){
	return gpu::dp<0,0,T>(n,x,y);
}

template<size_t STRIDE_X, size_t STRIDE_Y, class T>
T dp_gpu_sq(size_t n, T* x){
	return gpu::dp<0,0,T>(n,x);
}


BENCHMARK(bm_vector_dotproduct<double,dp_gpu<0,0,double>,DeviceType::GPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseManualTime()->Unit(benchmark::kMillisecond);

BENCHMARK(bm_vector_dotproduct<float,dp_gpu<0,0,float>,DeviceType::GPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseManualTime()->Unit(benchmark::kMillisecond);

BENCHMARK(bm_vector_dotproduct<double,dp_gpu_sq<0,0,double>,DeviceType::GPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseManualTime()->Unit(benchmark::kMillisecond);

BENCHMARK(bm_vector_dotproduct<float,dp_gpu_sq<0,0,float>,DeviceType::GPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->UseManualTime()->Unit(benchmark::kMillisecond);

