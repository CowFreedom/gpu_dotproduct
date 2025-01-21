cmake_minimum_required (VERSION 3.25.2)

project(dotprod
	VERSION 1.0
	DESCRIPTION "A vector dot product as optimized SIMD version.")
	
set (CMAKE_CXX_STANDARD 20)

if (MSVC)

else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()


option(BUILD_CUDA_LIBRARY "Build the additional CUDA library" ON)

include(CheckLanguage)
check_language(CUDA)	

if (CMAKE_CUDA_COMPILER AND BUILD_CUDA_LIBRARY)
	enable_language(CUDA)	
	set(CMAKE_CUDA_STANDARD 20)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
	message(STATUS "CUDA libraries have been activated!")
	set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
	set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
	set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)	

else()
	option(BUILD_CUDA_LIBRARY "Build the additional CUDA library" OFF)
	message(STATUS "Compiling C++ code only.")
	set (CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

list (APPEND DOTPROD_DEFINITIONS "DOTPROD_DEBUG_BUILD")

message(STATUS ${PROJECT_NAME})

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark
  GIT_TAG        v1.9.0
)

FetchContent_MakeAvailable(googletest googlebenchmark)


if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
	add_subdirectory(src/tests)
	add_subdirectory(src/profiling)
endif()

#Determine sources and headers

set (DOTPROD_SOURCE src/dotprod/dotprod.cpp	
  src/dotprod/extensions/avx2.cpp
  src/dotprod/extensions/avx.cpp				
						)			
						
set (DOTPROD_SOURCE_HEADERS
  src/dotprod/dotprod.h	
  src/dotprod/util/helpers.h	
  src/dotprod/extensions/avx2.h	
  src/dotprod/extensions/avx.h	
)

if (BUILD_CUDA_LIBRARY)
	set(DOTPROD_CUDA_SOURCE  src/dotprod/gpu/dp.cu)
	set(DOTPROD_CUDA_SOURCE_HEADERS  src/dotprod/gpu/dp.h)
endif()		 

source_group ("Source Files" FILES ${DOTPROD_SOURCE} ${DOTPROD_CUDA_SOURCE})
source_group ("Header Files" FILES ${DOTPROD_SOURCE_HEADERS} ${DOTPROD_CUDA_SOURCE_HEADERS})

add_library (dotprod_lib STATIC
  ${DOTPROD_SOURCE}
  ${DOTPROD_SOURCE_HEADERS}
  ${DOTPROD_CUDA_SOURCE}
  ${DOTPROD_CUDA_SOURCE_HEADERS} 
  )
  
target_include_directories (dotprod_lib PUBLIC
  src
  src/ext
)
  

target_compile_definitions (dotprod_lib PRIVATE ${DOTPROD_DEFINITIONS})

add_executable (dotprod_exe src/dotprod/bin/dotprod.cpp)

target_include_directories (dotprod_exe PUBLIC
  src
  src/ext
)

#option(DOTPROD_TEST_SOURCE "Enable tests" OFF)



