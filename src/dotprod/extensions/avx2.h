#pragma once

#if defined(_M_ARM) || defined(_M_ARM64) || defined(_M_HYBRID_X86_ARM64) || defined(_M_ARM64EC) || __arm__ || __aarch64__
	#error AVX2 not supported on ARM platform
#endif

#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <iostream>

namespace avx2{
	
	template<size_t STRIDE_A, size_t STRIDE_B>
	inline double dp(size_t n, double* a, double* b);
	template<>
	inline double dp<0,0>(size_t n, double* a, double* b){
		size_t blocks_end = n-static_cast<size_t>(n&15);
		double acc = 0;
		size_t i = 0;
		
		if (blocks_end != n){
		
			__m256d sum1 = _mm256_setzero_pd();
			__m256d sum2 = _mm256_setzero_pd();
			__m256d sum3 = _mm256_setzero_pd();
			__m256d sum4 = _mm256_setzero_pd();
			
			for (i = 0;i < blocks_end;i += 16){
				__m256d x = _mm256_loadu_pd(a);
				__m256d y = _mm256_loadu_pd(a+4);
				__m256d z = _mm256_loadu_pd(a+8);			
				__m256d w = _mm256_loadu_pd(a+12);
				__m256d k = _mm256_loadu_pd(b);		
				__m256d r = _mm256_loadu_pd(b+4);	
				__m256d p = _mm256_loadu_pd(b+8);	
				__m256d m = _mm256_loadu_pd(b+12);	
				
				sum1 = _mm256_fmadd_pd(x,k,sum1);
				sum2 = _mm256_fmadd_pd(y,r,sum2);
				sum3 = _mm256_fmadd_pd(z,p,sum3);
				sum4 = _mm256_fmadd_pd(w,m,sum4);		
				
				a += 16;
				b += 16;
			}
		
			__m256d sum1_plus_sum2 = _mm256_add_pd(sum1,sum2);
			__m256d sum2_plus_sum3 = _mm256_add_pd(sum3,sum4);
			
			__m256d sum1_sum2_plus_sum3_sum4 = _mm256_add_pd(sum1_plus_sum2,sum2_plus_sum3);
			
			__m128d sum1_sum2_plus_sum3_sum4_low = _mm256_castpd256_pd128(sum1_sum2_plus_sum3_sum4);
			__m128d sum1_sum2_plus_sum3_sum4_high = _mm256_extractf128_pd(sum1_sum2_plus_sum3_sum4,1);
			
			__m128d sum = _mm_add_pd(sum1_sum2_plus_sum3_sum4_low,sum1_sum2_plus_sum3_sum4_high);
			__m128d shuffled_sum = _mm_shuffle_pd(sum,sum,0b01);
			
			__m128d result = _mm_add_pd(sum,shuffled_sum); //result is stored in lower and higher register halves
			
			acc+=_mm_cvtsd_f64(result);
		}
		
		blocks_end = n-static_cast<size_t>(n&7);
		
		if (blocks_end != n){
			__m256d sum1 = _mm256_setzero_pd();
			__m256d sum2 = _mm256_setzero_pd();
			for (;i < blocks_end;i += 8){
				__m256d x = _mm256_loadu_pd(a);
				__m256d y = _mm256_loadu_pd(a+4);
				__m256d z = _mm256_loadu_pd(b);		
				__m256d w = _mm256_loadu_pd(b+4);	
				sum1 = _mm256_fmadd_pd(x,z,sum1);
				sum2 = _mm256_fmadd_pd(y,w,sum2);				
				
				//a += 8;
				b += 8;
			}	
			
			__m256d sum1_plus_sum2 = _mm256_add_pd(sum1,sum2);

			__m128d sum1_plus_sum2_low = _mm256_castpd256_pd128(sum1_plus_sum2);
			__m128d sum1_plus_sum2_high = _mm256_extractf128_pd(sum1_plus_sum2,1);
			__m128d sum = _mm_add_pd(sum1_plus_sum2_low,sum1_plus_sum2_high);
			__m128d shuffled_sum = _mm_shuffle_pd(sum,sum,0b01);
			
			__m128d result = _mm_add_pd(sum,shuffled_sum); //result is stored in lower and higher register halves
			
			acc += _mm_cvtsd_f64(result);				
		}
		
		while (i<n){
			acc=std::fma(*a,*b,acc);
			a++;
			b++;
			i++;
		}			
		return acc;
	}
	
	template<size_t STRIDE_A, size_t STRIDE_B>
	inline float dp(size_t n, float* a, float* b);
	
	template<>
	inline float dp<0,0>(size_t n, float* a, float* b){
		size_t blocks_end=n-static_cast<size_t>(n&31);
		float acc=0;
		size_t i=0;
		if (blocks_end != n){
			__m256 sum1 = _mm256_setzero_ps();
			__m256 sum2 = _mm256_setzero_ps();
			__m256 sum3 = _mm256_setzero_ps();
			__m256 sum4 = _mm256_setzero_ps();
			
			for (i=0;i<blocks_end;i+=32){
				__m256 x =_mm256_loadu_ps(a);
				__m256 y =_mm256_loadu_ps(a+8);
				__m256 z =_mm256_loadu_ps(a+16);			
				__m256 w =_mm256_loadu_ps(a+24);
				__m256 k =_mm256_loadu_ps(b);		
				__m256 r =_mm256_loadu_ps(b+8);	
				__m256 p =_mm256_loadu_ps(b+16);	
				__m256 m =_mm256_loadu_ps(b+24);	
				
				sum1 = _mm256_fmadd_ps(x,k,sum1);
				sum2 = _mm256_fmadd_ps(y,r,sum2);
				sum3 = _mm256_fmadd_ps(z,p,sum3);
				sum4 = _mm256_fmadd_ps(w,m,sum4);
				
				a+=32;
				b+=32;
			}				

			__m256 xk_plus_yr=_mm256_add_ps(sum1,sum2);
			__m256 zp_plus_wm=_mm256_add_ps(sum3,sum4);
			
			__m256 xk_yr_plus_zp_wm=_mm256_add_ps(xk_plus_yr,zp_plus_wm);
			
			__m128 xk_yr_plus_zp_wm_low=_mm256_castps256_ps128(xk_yr_plus_zp_wm);
			__m128 xk_yr_plus_zp_wm_high=_mm256_extractf128_ps(xk_yr_plus_zp_wm,1);
			
			__m128 sum=_mm_add_ps(xk_yr_plus_zp_wm_low,xk_yr_plus_zp_wm_high);
			__m128 shuffled_sum=_mm_shuffle_ps(sum,sum,_MM_SHUFFLE(0,1,2,3));
			
			__m128 sum_final=_mm_add_ps(sum,shuffled_sum); 
			__m128 shuffled_sum_final=_mm_shuffle_ps(sum_final,sum_final,_MM_SHUFFLE(2,3,0,1));
			__m128 result=_mm_add_ps(sum_final,shuffled_sum_final); 		
			
			acc+=_mm_cvtss_f32(result);		
				
			
		}
		
		while (i<n){
			acc=std::fma(*a,*b,acc);
			a++;
			b++;
			i++;
		}			
		return acc;
	}	
	
}