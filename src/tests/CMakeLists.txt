set (DOTPROD_TEST_SOURCE
#"dotprod.cpp" 
"extensions/avx.cpp"
"extensions/avx2.cpp")

if (BUILD_CUDA_LIBRARY)
	set(DOTPROD_CUDA_SOURCE  gpu/dp.cu)
endif()	


add_executable(dotprod_test ${DOTPROD_TEST_SOURCE} ${DOTPROD_CUDA_SOURCE} helpers.h)
target_include_directories(dotprod_test PRIVATE "${PROJECT_SOURCE_DIR}/src/tests")

target_link_libraries(dotprod_test PRIVATE dotprod_lib gtest_main)
target_compile_definitions (dotprod_test PRIVATE ${DOTPROD_DEFINITIONS})

if (BUILD_CUDA_LIBRARY)
	set_source_files_properties(dotprod_test PROPERTIES LANGUAGE CUDA)
endif()	

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( dotprod_test
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME dotprod_unit_test
	COMMAND dotprod_test)