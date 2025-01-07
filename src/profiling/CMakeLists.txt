set (DOTPROD_PROFILE_SOURCE
"dotprod.cpp"
"extensions/avx.cpp"
"extensions/avx2.cpp"
"gpu/dp.cu"
)

if (BUILD_CUDA_LIBRARY)
	set(DOTPROD_PROFILE_CUDA_SOURCE  gpu/dp.cu)
endif()	

message(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_executable(dotprod_profiling ${DOTPROD_PROFILE_SOURCE} ${DOTPROD_PROFILE_CUDA_SOURCE} helpers.h)

target_include_directories(dotprod_profiling PRIVATE /)

target_link_libraries(dotprod_profiling PRIVATE dotprod_lib benchmark::benchmark)
target_compile_definitions (dotprod_profiling PRIVATE ${DOTPROD_DEFINITIONS})

