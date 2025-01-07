#pragma once


#include <thread>
#include <vector>
#include <functional>
#include <iostream>
#include <concepts>

namespace dotprod{
	//https://stackoverflow.com/questions/10437843/restrict-integer-template-parameter
	template<size_t X>
	concept IsPositivePowerOfTwo = std::is_integral<decltype(X)>::value && ((X & (X-1)) == 0) && (X != 0);
	
	
	template<class T>
	T dp_parallel(size_t n_threads, size_t block_size, size_t n, T (*f)(size_t, T*, T*), T* x, T* y){
		if (n == 0){
			return T(0);
		}
		size_t blocks_used=(n+block_size-1)/block_size;
		size_t blocks_per_thread=blocks_used/n_threads;
		int blocks_per_thread_remainder=blocks_used%n_threads;
		size_t threads_used=(blocks_per_thread==0)?blocks_per_thread_remainder:n_threads;
	
		std::vector<std::thread> ts(threads_used-1);
		std::vector<T> res(threads_used-1);
		
		auto dp=[&f](T& res, size_t n, T* a, T* b){
			res=f(n,a,b);
		};
		
		for (size_t i=0;i<threads_used-1;i++){
			size_t step=(blocks_per_thread_remainder>0)?block_size*(blocks_per_thread+1):block_size*blocks_per_thread;
			ts[i]=std::thread(dp,std::ref(res[i]),step,x,y);
			x+=step;
			y+=step;
			blocks_per_thread_remainder--;
			n-=step;
		}

		T sum=f(n,x,y);
		
		for (auto& t:ts){
			t.join();
		}
		for (auto s:res){
			sum+=s;
		}
		return sum;			
	}

	template<class T>
	T dp_parallel(size_t n, T (*f)(size_t, T*, T*), T* x, T* y){
		size_t n_threads=12;
		size_t block_size=50000;
		return dp_parallel(n_threads,block_size,n,f,x,y);
	}
/*	
template<class T>
	T dp_parallel(size_t n_threads, size_t block_size, size_t n, T (*f)(size_t, T*), T* x){
		
		size_t blocks_used=(n+block_size-1)/block_size;
		size_t blocks_used_remainder=n%block_size;
		size_t blocks_per_thread=blocks_used/n_threads;
		int blocks_per_thread_remainder=blocks_used%n_threads;
		size_t threads_used=(blocks_per_thread==0)?blocks_per_thread_remainder:n_threads;
		
		std::vector<std::thread> ts(threads_used-1);
		std::vector<T> res(threads_used-1);
		
		auto dp=[&f](T& res, size_t n, T* a){
			res=f(n,a);
		};
		
		for (size_t i=0;i<threads_used-1;i++){
			size_t step=(blocks_per_thread_remainder>0)?block_size*(blocks_per_thread+1):block_size*blocks_per_thread;
			ts[i]=std::thread(dp,std::ref(res[i]),step,x);
			x+=step;
			blocks_per_thread_remainder--;
			n-=step;
		}
		
		T sum=f(n,x);
		//ts[ts.size()-1]=std::thread(dp,std::ref(res[ts.size()-1]),n,x);

		for (auto& t:ts){
		//	t.join();
		}
		for (auto s:res){
			sum+=s;
		}
		return sum;
			
	}

	template<class T>
	T dp_parallel(size_t n, T (*f)(size_t, T*), T* x){
		size_t n_threads=1;
		size_t block_size=50000;
		return dp_parallel(n_threads,block_size,n,f,x);
	}*/
	
	inline int log2_int(uint32_t v) {
	#ifdef _MSC_VER
		unsigned long lz = 0;
		if (_BitScanReverse(&lz, v))
			return lz;
		return 0;
	#else
		return 31 - __builtin_clz(v);
	#endif
	}	
}

namespace gpu::helper{
	
	#ifdef __CUDACC__

	void check(cudaError_t err, const char* const func, const char* const file, const int line){
		if (err != cudaSuccess)
		{
			std::cerr << "CUDA Runtime Error at: " << file << ":" << line<<"\n";
			std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
			std::exit(EXIT_FAILURE);
		}
	}
	#define CHECK_CUDA_ERROR(val) ::gpu::helper::check((val), #val, __FILE__, __LINE__)		
	#endif
}
