#include <benchmark/benchmark.h>
#include <dotprod/dotprod.h>
#include <helpers.h>

BENCHMARK_MAIN();

BENCHMARK(bm_vector_dotproduct<double,dp_control<double>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->Unit(benchmark::kMillisecond);
BENCHMARK(bm_vector_dotproduct<float,dp_control<float>,DeviceType::CPU>)->Threads(1)->RangeMultiplier(2)->Range(PERFORMANCE_TEST_LOWER_LIMIT, PERFORMANCE_TEST_UPPER_LIMIT)->Unit(benchmark::kMillisecond);
