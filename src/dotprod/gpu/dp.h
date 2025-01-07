#pragma once

#include <dotprod/util/helpers.h>
#include <dotprod/extensions/avx.h>
#include <iostream>
#include <string>
namespace gpu{
	
	template<class T>
	struct VECTORIZED_CVT;
	
	template<>
	struct VECTORIZED_CVT<double>{
		using type=double4;
	};

	template<>
	struct VECTORIZED_CVT<float>{
		using type=float4;
	};
	
	namespace helper{
		
	

		template<class T>
		__device__ T get_zero_vector();
		
		template<>
		__device__ double4 get_zero_vector() { 
			return make_double4(0.0, 0.0, 0.0, 0.0); 
		}

		template<>
		__device__ float4 get_zero_vector() { 
			return make_float4(0.0, 0.0, 0.0, 0.0); 
		}

		__device__
		__forceinline__ void fma(double4* x, double4* y, double4& value){
			value.x=::fma((*x).x,(*y).x,value.x); //vectorized fma would be better
			value.y=::fma((*x).y,(*y).y,value.y); //vectorized fma would be better
			value.z=::fma((*x).z,(*y).z,value.z); //vectorized fma would be better
			value.w=::fma((*x).w,(*y).w,value.w); //vectorized fma would be better
		}

		__device__
		__forceinline__ void fma(float4* x, float4* y, float4& value){
			value.x=::fma((*x).x,(*y).x,value.x); //vectorized fma would be better
			value.y=::fma((*x).y,(*y).y,value.y); //vectorized fma would be better
			value.z=::fma((*x).z,(*y).z,value.z); //vectorized fma would be better
			value.w=::fma((*x).w,(*y).w,value.w); //vectorized fma would be better
		}

		template<int OFFSET>
		__device__
		__forceinline__ void shared_memory_add_assignment_of_vector(double4* sdata){
				sdata[threadIdx.x].x+=sdata[threadIdx.x+OFFSET].x;
				sdata[threadIdx.x].y+=sdata[threadIdx.x+OFFSET].y;
				sdata[threadIdx.x].z+=sdata[threadIdx.x+OFFSET].z;
				sdata[threadIdx.x].w+=sdata[threadIdx.x+OFFSET].w;
		}

		template<int OFFSET>
		__device__
		__forceinline__ void shared_memory_add_assignment_of_vector(float4* sdata){
				sdata[threadIdx.x].x+=sdata[threadIdx.x+OFFSET].x;
				sdata[threadIdx.x].y+=sdata[threadIdx.x+OFFSET].y;
				sdata[threadIdx.x].z+=sdata[threadIdx.x+OFFSET].z;
				sdata[threadIdx.x].w+=sdata[threadIdx.x+OFFSET].w;
		}

			
		//no syncthreads needed because I am in a warp
		template<size_t THREADS_PER_BLOCK, typename VECTORIZED_T>
		__device__
		__forceinline__ void reduce_warp(VECTORIZED_T* sdata){
			if(THREADS_PER_BLOCK>=64){		
					shared_memory_add_assignment_of_vector<32>(sdata);
				__syncthreads();
			}
			if(THREADS_PER_BLOCK>=32){
					shared_memory_add_assignment_of_vector<16>(sdata);
				__syncthreads();
			}
			if(THREADS_PER_BLOCK>=16){
					shared_memory_add_assignment_of_vector<8>(sdata);
				__syncthreads();
			}
			if(THREADS_PER_BLOCK>=8){
				shared_memory_add_assignment_of_vector<4>(sdata);
				__syncthreads();
			}
			if(THREADS_PER_BLOCK>=4){
					shared_memory_add_assignment_of_vector<2>(sdata);
				__syncthreads();
			}
			if(THREADS_PER_BLOCK>=2){
					shared_memory_add_assignment_of_vector<1>(sdata);
				__syncthreads();
			}			
		}

		double vectorized_type_to_scalar_sum(double4& x){
			return x.x+x.y+x.z+x.w;
		}
		
		float vectorized_type_to_scalar_sum(float4& x){
			return x.x+x.y+x.z+x.w;
		}
	}
	

	template<size_t THREADS_PER_BLOCK, typename VECTORIZED_T>
	__global__
	void reduce_k(size_t n, VECTORIZED_T* x, VECTORIZED_T* y, VECTORIZED_T* r){
		constexpr size_t memsize=(THREADS_PER_BLOCK<=64)?64:THREADS_PER_BLOCK;
		using T=decltype(x[0].x);
		static __shared__ VECTORIZED_T sdata[memsize];
		VECTORIZED_T value = helper::get_zero_vector<VECTORIZED_T>();
		sdata[threadIdx.x] = helper::get_zero_vector<VECTORIZED_T>();
		const int stride = gridDim.x*blockDim.x;

		for (int i=blockDim.x*blockIdx.x+threadIdx.x; i<n; i+=stride){
			helper::fma(x+i,y+i,value);	
		}
		sdata[threadIdx.x]=value;
		__syncthreads();

		if constexpr (THREADS_PER_BLOCK>=1024){
			if (threadIdx.x < 512) { 
				helper::shared_memory_add_assignment_of_vector<512>(sdata);
			}
			__syncthreads(); 
		}
		if constexpr (THREADS_PER_BLOCK>=512){
			if (threadIdx.x < 256) { 
				helper::shared_memory_add_assignment_of_vector<256>(sdata);
			}
			__syncthreads(); 
		}

		if constexpr (THREADS_PER_BLOCK>=256){
			if (threadIdx.x < 128) { 
				helper::shared_memory_add_assignment_of_vector<128>(sdata);
			}
			__syncthreads(); 
		}	

		if constexpr (THREADS_PER_BLOCK>=128){
			if (threadIdx.x < 64) { 
				helper::shared_memory_add_assignment_of_vector<64>(sdata);
			}
			__syncthreads(); 
		}		
		if (threadIdx.x<32){
			helper::reduce_warp<THREADS_PER_BLOCK,VECTORIZED_T>(sdata);
			if (threadIdx.x == 0){ //If statement not needed because all threads because all threads write the same value
				r[blockIdx.x]=sdata[0];
			}
		}
	}
	
	
	//stream block size must be power of two
	template<size_t MAX_STREAMS, size_t MIN_N_PER_STREAM_SIZE, size_t MAX_BLOCKS_PER_STREAM, size_t THREADS_PER_BLOCK, typename T> requires dotprod::IsPositivePowerOfTwo<MAX_BLOCKS_PER_STREAM> && dotprod::IsPositivePowerOfTwo<THREADS_PER_BLOCK>
	__host__
	T reduce(size_t n, T* x_pinned, T* y_pinned){
		if (n == 0){
			return T(0);
		}
		using VECTORIZED_T=typename VECTORIZED_CVT<T>::type;
		int VECTORIZED_ELEMENTS=sizeof(VECTORIZED_T)/sizeof(T);

		T res=T(0);
		int n_power_of_two=1u<<dotprod::log2_int(n); //todo change
		uint64_t streams_needed_no_clamp=(n_power_of_two+static_cast<long long>(MIN_N_PER_STREAM_SIZE-1LL))/MIN_N_PER_STREAM_SIZE;
		size_t streams_needed=(streams_needed_no_clamp<MAX_STREAMS)?streams_needed_no_clamp:MAX_STREAMS; //AND only works if MAX_STREAMS is power of two
		int n_per_stream=1u<<dotprod::log2_int(n_power_of_two/streams_needed);
		int n_vectorized_per_stream=n_per_stream/VECTORIZED_ELEMENTS;
		int n_gpu=n_per_stream*streams_needed;
		size_t n_cpu=n-n_gpu;
		int blocks_used=(((n_vectorized_per_stream+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK)<MAX_BLOCKS_PER_STREAM)?(n_vectorized_per_stream+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK:MAX_BLOCKS_PER_STREAM;		
		VECTORIZED_T* r_d;

		VECTORIZED_T* r_h = new VECTORIZED_T[MAX_STREAMS*MAX_BLOCKS_PER_STREAM]; 
		CHECK_CUDA_ERROR(cudaMalloc((void**)&r_d, sizeof(VECTORIZED_T)*streams_needed*blocks_used));

		cudaStream_t streams[MAX_STREAMS];
		for (size_t i=0;i<streams_needed;i++){
			size_t bytes_to_copy=sizeof(T)*n_per_stream;
			VECTORIZED_T* _x_d;
			VECTORIZED_T* _y_d;
			VECTORIZED_T* _r_d=r_d+i*blocks_used;
			cudaStreamCreate (&(streams[i]));
			CHECK_CUDA_ERROR(cudaMallocAsync((void**)&_x_d, bytes_to_copy, streams[i]));
			CHECK_CUDA_ERROR(cudaMallocAsync((void**)&_y_d, bytes_to_copy, streams[i]));
			CHECK_CUDA_ERROR(cudaMemcpyAsync (_x_d, reinterpret_cast<VECTORIZED_T*>(&(x_pinned[i*n_per_stream])), bytes_to_copy, cudaMemcpyHostToDevice, streams[i] ));
			CHECK_CUDA_ERROR(cudaMemcpyAsync (_y_d, reinterpret_cast<VECTORIZED_T*>(&(y_pinned[i*n_per_stream])), bytes_to_copy, cudaMemcpyHostToDevice, streams[i] ));
			reduce_k<THREADS_PER_BLOCK,VECTORIZED_T><<<dim3(blocks_used,1,1),dim3(THREADS_PER_BLOCK,1,1),0,streams[i]>>>(n_vectorized_per_stream,_x_d,_y_d,_r_d);
			cudaFreeAsync(_x_d,streams[i]);
			cudaFreeAsync(_y_d,streams[i]);
			
		}
		res+=avx::dp<0,0>(n_cpu,x_pinned+n_gpu,y_pinned+n_gpu);
		for (size_t i=0;i<streams_needed;i++){
			CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
			CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
		}
		
		CHECK_CUDA_ERROR(cudaMemcpy(r_h, r_d, sizeof(VECTORIZED_T)*streams_needed*blocks_used, cudaMemcpyDeviceToHost));			
		for (int i=0;i<streams_needed*blocks_used;i++){
			res+=helper::vectorized_type_to_scalar_sum(r_h[i]);
		}
		delete[] r_h;
		cudaFree(r_d);
		return res;
	}
	

	template<size_t STRIDE_A, size_t STRIDE_B,typename T>
	inline T dp(size_t n, T* x_pinned, T* y_pinned, bool host_memory_is_pinned=false){
		constexpr size_t MAX_STREAMS=64;
		constexpr size_t MIN_N_PER_STREAM_SIZE=1<<12;
		constexpr size_t MAX_BLOCKS_PER_STREAM=32;
		constexpr size_t THREADS_PER_BLOCK=1024;

		return reduce<MAX_STREAMS,MIN_N_PER_STREAM_SIZE,MAX_BLOCKS_PER_STREAM,THREADS_PER_BLOCK,T>(n,x_pinned,y_pinned);
	}
	
	//stream block size must be power of two
	template<size_t MAX_STREAMS, size_t MIN_N_PER_STREAM_SIZE, size_t MAX_BLOCKS_PER_STREAM, size_t THREADS_PER_BLOCK, typename T> requires dotprod::IsPositivePowerOfTwo<MAX_BLOCKS_PER_STREAM> && dotprod::IsPositivePowerOfTwo<THREADS_PER_BLOCK>
	__host__
	T reduce(size_t n, T* x_pinned){
		if (n == 0){
			return T(0);
		}
		using VECTORIZED_T=typename VECTORIZED_CVT<T>::type;
		int VECTORIZED_ELEMENTS=sizeof(VECTORIZED_T)/sizeof(T);

		T res=T(0);
		int n_power_of_two=1u<<dotprod::log2_int(n); //todo change
		uint64_t streams_needed_no_clamp=(n_power_of_two+static_cast<long long>(MIN_N_PER_STREAM_SIZE-1LL))/MIN_N_PER_STREAM_SIZE;
		size_t streams_needed=(streams_needed_no_clamp<MAX_STREAMS)?streams_needed_no_clamp:MAX_STREAMS; //AND only works if MAX_STREAMS is power of two
		int n_per_stream=1u<<dotprod::log2_int(n_power_of_two/streams_needed);
		int n_vectorized_per_stream=n_per_stream/VECTORIZED_ELEMENTS;
		int n_gpu=n_per_stream*streams_needed;
		size_t n_cpu=n-n_gpu;
		int blocks_used=(((n_vectorized_per_stream+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK)<MAX_BLOCKS_PER_STREAM)?(n_vectorized_per_stream+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK:MAX_BLOCKS_PER_STREAM;		
		VECTORIZED_T* r_d;

		VECTORIZED_T* r_h = new VECTORIZED_T[MAX_STREAMS*MAX_BLOCKS_PER_STREAM]; 
		CHECK_CUDA_ERROR(cudaMalloc((void**)&r_d, sizeof(VECTORIZED_T)*streams_needed*blocks_used));

		cudaStream_t streams[MAX_STREAMS];
		for (size_t i=0;i<streams_needed;i++){
			size_t bytes_to_copy=sizeof(T)*n_per_stream;
			VECTORIZED_T* _x_d;
			VECTORIZED_T* _r_d=r_d+i*blocks_used;
			cudaStreamCreate (&(streams[i]));
			CHECK_CUDA_ERROR(cudaMallocAsync((void**)&_x_d, bytes_to_copy, streams[i]));
			CHECK_CUDA_ERROR(cudaMemcpyAsync (_x_d, reinterpret_cast<VECTORIZED_T*>(&(x_pinned[i*n_per_stream])), bytes_to_copy, cudaMemcpyHostToDevice, streams[i] ));
			reduce_k<THREADS_PER_BLOCK,VECTORIZED_T><<<dim3(blocks_used,1,1),dim3(THREADS_PER_BLOCK,1,1),0,streams[i]>>>(n_vectorized_per_stream,_x_d,_x_d,_r_d);
			cudaFreeAsync(_x_d,streams[i]);
			
		}
		res+=avx::dp_sq<0,0>(n_cpu,x_pinned+n_gpu);
		for (size_t i=0;i<streams_needed;i++){
			CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
			CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
		}
		
		CHECK_CUDA_ERROR(cudaMemcpy(r_h, r_d, sizeof(VECTORIZED_T)*streams_needed*blocks_used, cudaMemcpyDeviceToHost));			
		for (int i=0;i<streams_needed*blocks_used;i++){
			res+=helper::vectorized_type_to_scalar_sum(r_h[i]);
		}
		delete[] r_h;
		cudaFree(r_d);
		return res;
	}
	

	template<size_t STRIDE_A, size_t STRIDE_B,typename T>
	inline T dp(size_t n, T* x_pinned, bool host_memory_is_pinned=false){
		constexpr size_t MAX_STREAMS=64;
		constexpr size_t MIN_N_PER_STREAM_SIZE=1<<12;
		constexpr size_t MAX_BLOCKS_PER_STREAM=32;
		constexpr size_t THREADS_PER_BLOCK=1024;

		return reduce<MAX_STREAMS,MIN_N_PER_STREAM_SIZE,MAX_BLOCKS_PER_STREAM,THREADS_PER_BLOCK,T>(n,x_pinned);
	}	

}
