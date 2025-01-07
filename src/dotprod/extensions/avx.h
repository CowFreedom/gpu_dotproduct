#pragma once

#include <immintrin.h>
#include <cstdint>

namespace avx{
	
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
				
				__m256d xk = _mm256_mul_pd(x,k);			
				__m256d yr = _mm256_mul_pd(y,r);			
				__m256d zp = _mm256_mul_pd(z,p);			
				__m256d wm = _mm256_mul_pd(w,m);	
				
				sum1 = _mm256_add_pd(sum1,xk);
				sum2 = _mm256_add_pd(sum2,yr);
				sum3 = _mm256_add_pd(sum3,zp);
				sum4 = _mm256_add_pd(sum4,wm);		
				
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
			
			for (;i < blocks_end;i += 8){
				__m256d x = _mm256_loadu_pd(a);
				__m256d y = _mm256_loadu_pd(a+4);
				__m256d z = _mm256_loadu_pd(b);		
				__m256d w = _mm256_loadu_pd(b+4);	
				__m256d xz = _mm256_mul_pd(x,z);
				__m256d yw = _mm256_mul_pd(y,w);			
				__m256d xz_plus_yw = _mm256_add_pd(xz,yw);

				sum1 = _mm256_add_pd(sum1,xz_plus_yw);		
				
				a += 8;
				b += 8;
			}	
			__m128d sum1_low = _mm256_castpd256_pd128(sum1);
			__m128d sum1_high = _mm256_extractf128_pd(sum1,1);
			__m128d sum = _mm_add_pd(sum1_low,sum1_high);
			__m128d shuffled_sum = _mm_shuffle_pd(sum,sum,0b01);
			
			__m128d result = _mm_add_pd(sum,shuffled_sum); //result is stored in lower and higher register halves
			
			acc += _mm_cvtsd_f64(result);				
		}
		
		while (i<n){
			acc += (*a)*(*b);
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
		
		for (i=0;i<blocks_end;i+=32){
			__m256 x=_mm256_loadu_ps(a);
			__m256 y=_mm256_loadu_ps(a+8);
			__m256 z=_mm256_loadu_ps(a+16);			
			__m256 w=_mm256_loadu_ps(a+24);
			__m256 k=_mm256_loadu_ps(b);		
			__m256 r=_mm256_loadu_ps(b+8);	
			__m256 p=_mm256_loadu_ps(b+16);	
			__m256 m=_mm256_loadu_ps(b+24);	
			
			__m256 xk= _mm256_mul_ps(x,k);			
			__m256 yr= _mm256_mul_ps(y,r);			
			__m256 zp= _mm256_mul_ps(z,p);			
			__m256 wm= _mm256_mul_ps(w,m);	

			__m256 xk_plus_yr=_mm256_add_ps(xk,yr);
			__m256 zp_plus_wm=_mm256_add_ps(zp,wm);
			
			__m256 xk_yr_plus_zp_wm=_mm256_add_ps(xk_plus_yr,zp_plus_wm);
			
			__m128 xk_yr_plus_zp_wm_low=_mm256_castps256_ps128(xk_yr_plus_zp_wm);
			__m128 xk_yr_plus_zp_wm_high=_mm256_extractf128_ps(xk_yr_plus_zp_wm,1);
			
			__m128 sum=_mm_add_ps(xk_yr_plus_zp_wm_low,xk_yr_plus_zp_wm_high);
			__m128 shuffled_sum=_mm_shuffle_ps(sum,sum,_MM_SHUFFLE(0,1,2,3));
			
			__m128 sum2=_mm_add_ps(sum,shuffled_sum); 
			__m128 shuffled_sum2=_mm_shuffle_ps(sum2,sum2,_MM_SHUFFLE(2,3,0,1));
			__m128 result=_mm_add_ps(sum2,shuffled_sum2); 		
			
			acc+=_mm_cvtss_f32(result);		
			
			a+=32;
			b+=32;
			
		}
		
		blocks_end=n-static_cast<size_t>(n&15);
		
		for (;i<blocks_end;i+=16){
			__m256 x=_mm256_loadu_ps(a);
			__m256 y=_mm256_loadu_ps(a+8);
			__m256 z=_mm256_loadu_ps(b);		
			__m256 w=_mm256_loadu_ps(b+8);	
			__m256 xz= _mm256_mul_ps(x,z);
			__m256 yw= _mm256_mul_ps(y,w);			
			__m256 xz_plus_yw=_mm256_add_ps(xz,yw);
			__m128 xz_plus_yw_low=_mm256_castps256_ps128(xz_plus_yw);
			__m128 xz_plus_yw_high=_mm256_extractf128_ps(xz_plus_yw,1);
			
			__m128 sum=_mm_add_ps(xz_plus_yw_low,xz_plus_yw_high);
			__m128 shuffled_sum=_mm_shuffle_ps(sum,sum,_MM_SHUFFLE(0,1,2,3));
			
			__m128 sum2=_mm_add_ps(sum,shuffled_sum); 
			__m128 shuffled_sum2=_mm_shuffle_ps(sum2,sum2,_MM_SHUFFLE(2,3,0,1));
			__m128 result=_mm_add_ps(sum2,shuffled_sum2); 		
			
			acc+=_mm_cvtss_f32(result);		
			
			a+=16;
			b+=16;
			
		}
		
		while (i<n){
			acc+=(*a)*(*b);
			a++;
			b++;
			i++;
		}			
		return acc;
	}	

	template<size_t STRIDE_A, size_t STRIDE_B>
	inline double dp_sq(size_t n, double* a);

	template<>
	inline double dp_sq<0,0>(size_t n, double* a){
		size_t blocks_end = n-static_cast<size_t>(n&31);
		double acc = 0.0;
		size_t i = 0;
		
		if (blocks_end != n){
			
			__m256d sum1= _mm256_setzero_pd();
			__m256d sum2= _mm256_setzero_pd();
			__m256d sum3= _mm256_setzero_pd();
			__m256d sum4= _mm256_setzero_pd();
			__m256d sum5= _mm256_setzero_pd();
			__m256d sum6= _mm256_setzero_pd();
			__m256d sum7= _mm256_setzero_pd();		
			__m256d sum8= _mm256_setzero_pd();	

			for (;i<blocks_end;i+=32){
				__m256d x=_mm256_loadu_pd(a);
				__m256d y=_mm256_loadu_pd(a+4);	
				__m256d z=_mm256_loadu_pd(a+8);	
				__m256d w=_mm256_loadu_pd(a+12);	
				__m256d k=_mm256_loadu_pd(a+16);	
				__m256d r=_mm256_loadu_pd(a+20);	
				__m256d p=_mm256_loadu_pd(a+24);	
				__m256d m=_mm256_loadu_pd(a+28);	
				
				__m256d xx= _mm256_mul_pd(x,x);
				__m256d yy= _mm256_mul_pd(y,y);
				__m256d zz= _mm256_mul_pd(z,z);
				__m256d ww= _mm256_mul_pd(w,w);
				__m256d kk= _mm256_mul_pd(k,k);
				__m256d rr= _mm256_mul_pd(r,r);
				__m256d pp= _mm256_mul_pd(p,p);		
				__m256d mm= _mm256_mul_pd(m,m);		
				
				sum1= _mm256_add_pd(sum1,xx); //no fma in avx (only avx2)
				sum2= _mm256_add_pd(sum2,yy);
				sum3= _mm256_add_pd(sum3,zz);
				sum4= _mm256_add_pd(sum4,ww);
				sum5= _mm256_add_pd(sum5,kk);
				sum6= _mm256_add_pd(sum6,rr);
				sum7= _mm256_add_pd(sum7,pp);		
				sum8= _mm256_add_pd(sum8,mm);	
				
				a+=32;
			}
			
			__m256d xx_plus_yy=_mm256_add_pd(sum1,sum2);
			__m256d zz_plus_ww=_mm256_add_pd(sum3,sum4);
			__m256d kk_plus_rr=_mm256_add_pd(sum5,sum6);
			__m256d pp_plus_mm=_mm256_add_pd(sum7,sum8);
			
			__m256d xx_yy_plus_zz_ww=_mm256_add_pd(xx_plus_yy,zz_plus_ww);
			__m256d kk_rr_plus_pp_mm=_mm256_add_pd(kk_plus_rr,pp_plus_mm);
			
			__m256d xx_yy_zz_ww_plus_kk_rr_pp_mm=_mm256_add_pd(xx_yy_plus_zz_ww,kk_rr_plus_pp_mm);			
			
			__m128d xx_yy_zz_ww_plus_kk_rr_pp_mm_low=_mm256_castpd256_pd128(xx_yy_zz_ww_plus_kk_rr_pp_mm);
			__m128d xx_yy_zz_ww_plus_kk_rr_pp_mm_high=_mm256_extractf128_pd(xx_yy_zz_ww_plus_kk_rr_pp_mm,1);
			
			__m128d sum=_mm_add_pd(xx_yy_zz_ww_plus_kk_rr_pp_mm_low,xx_yy_zz_ww_plus_kk_rr_pp_mm_high);
			__m128d shuffled_sum=_mm_shuffle_pd(sum,sum,0b01);
			
			__m128d result=_mm_add_pd(sum,shuffled_sum); //result is stored in lower and higher register halves
			
			acc+=_mm_cvtsd_f64(result);//	
		
		}
		while (i<n){
			acc+=(*a)*(*a);
			a++;
			i++;
		}	
		return acc;
	}
	
	template<size_t STRIDE_A, size_t STRIDE_B>
	inline float dp_sq(size_t n, float* a);	
	
	template<>
	inline float dp_sq<0,0>(size_t n, float* a){
		size_t blocks_end=n-static_cast<size_t>(n&63);
		double acc=0;
		size_t i=0;
		
		for (;i<blocks_end;i+=64){
			__m256 x=_mm256_loadu_ps(a);
			__m256 y=_mm256_loadu_ps(a+8);	
			__m256 z=_mm256_loadu_ps(a+16);	
			__m256 w=_mm256_loadu_ps(a+24);	
			__m256 k=_mm256_loadu_ps(a+32);	
			__m256 r=_mm256_loadu_ps(a+40);	
			__m256 p=_mm256_loadu_ps(a+48);	
			__m256 m=_mm256_loadu_ps(a+56);	
			
			__m256 xx= _mm256_mul_ps(x,x);
			__m256 yy= _mm256_mul_ps(y,y);
			__m256 zz= _mm256_mul_ps(z,z);
			__m256 ww= _mm256_mul_ps(w,w);
			__m256 kk= _mm256_mul_ps(k,k);
			__m256 rr= _mm256_mul_ps(r,r);
			__m256 pp= _mm256_mul_ps(p,p);		
			__m256 mm= _mm256_mul_ps(m,m);		

			__m256 xx_plus_yy=_mm256_add_ps(xx,yy);
			__m256 zz_plus_ww=_mm256_add_ps(zz,ww);
			__m256 kk_plus_rr=_mm256_add_ps(kk,rr);
			__m256 pp_plus_mm=_mm256_add_ps(pp,mm);
			
			__m256 xx_yy_plus_zz_ww=_mm256_add_ps(xx_plus_yy,zz_plus_ww);
			__m256 kk_rr_plus_pp_mm=_mm256_add_ps(kk_plus_rr,pp_plus_mm);
			
			__m256 xx_yy_zz_ww_plus_kk_rr_pp_mm=_mm256_add_ps(xx_yy_plus_zz_ww,kk_rr_plus_pp_mm);			
			
			__m128 xx_yy_zz_ww_plus_kk_rr_pp_mm_low=_mm256_castps256_ps128(xx_yy_zz_ww_plus_kk_rr_pp_mm);
			__m128 xx_yy_zz_ww_plus_kk_rr_pp_mm_high=_mm256_extractf128_ps(xx_yy_zz_ww_plus_kk_rr_pp_mm,1);
			
			__m128 sum=_mm_add_ps(xx_yy_zz_ww_plus_kk_rr_pp_mm_low,xx_yy_zz_ww_plus_kk_rr_pp_mm_high);
			__m128 shuffled_sum=_mm_shuffle_ps(sum,sum,_MM_SHUFFLE(0,1,2,3));
			
			__m128 sum2=_mm_add_ps(sum,shuffled_sum); 
			__m128 shuffled_sum2=_mm_shuffle_ps(sum2,sum2,_MM_SHUFFLE(2,3,0,1));
			__m128 result=_mm_add_ps(sum2,shuffled_sum2); 		
			
			acc+=_mm_cvtss_f32(result);					
			
			a+=64;
			
		}
		
		while (i<n){
			acc+=(*a)*(*a);
			a++;
			i++;
		}			
		return acc;
	}
	
	
}