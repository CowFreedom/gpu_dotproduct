#pragma once 

#if defined(__AVX2__defined)
	#include <dotprod/extensions/avx2.h>
#endif

#if defined(__AVX__defined)
	#include <dotprod/extensions/avx.h>
#endif

