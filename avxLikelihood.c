#include <unistd.h>

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include "axml.h"
#include <stdint.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>

const union __attribute__ ((aligned (BYTE_ALIGNMENT)))
{
  uint64_t i[4];
  __m256d m;
  
} absMask_AVX = {{0x7fffffffffffffffULL, 0x7fffffffffffffffULL, 0x7fffffffffffffffULL, 0x7fffffffffffffffULL}};



static inline __m256d hadd4(__m256d v, __m256d u)
{ 
  __m256d
    a, b;
  
  v = _mm256_hadd_pd(v, v);
  a = _mm256_permute2f128_pd(v, v, 1);
  v = _mm256_add_pd(a, v);

  u = _mm256_hadd_pd(u, u);
  b = _mm256_permute2f128_pd(u, u, 1);
  u = _mm256_add_pd(b, u);

  v = _mm256_mul_pd(v, u);	
  
  return v;
}

static inline __m256d hadd3(__m256d v)
{ 
  __m256d
    a;
  
  v = _mm256_hadd_pd(v, v);
  a = _mm256_permute2f128_pd(v, v, 1);
  v = _mm256_add_pd(a, v);
  
  return v;
}


void  newviewGTRGAMMA_AVX(int tipCase,
			 double *x1, double *x2, double *x3,
			 double *extEV, double *tipVector,
			 int *ex3, unsigned char *tipX1, unsigned char *tipX2,
			 const int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling
			 )
{
 
  int  
    i, 
    k, 
    scale, 
    addScale = 0;
 
  __m256d 
    minlikelihood_avx = _mm256_set1_pd( minlikelihood ),
    twoto = _mm256_set1_pd(twotothe256);
 

  switch(tipCase)
    {
    case TIP_TIP:
      {
	double 
	  *uX1, 
	  umpX1[1024] __attribute__ ((aligned (BYTE_ALIGNMENT))), 
	  *uX2, 
	  umpX2[1024] __attribute__ ((aligned (BYTE_ALIGNMENT)));

	for (i = 1; i < 16; i++)
	  {
	    __m256d 
	      tv = _mm256_load_pd(&(tipVector[i * 4]));

	    int 
	      j;
	    
	    for (j = 0; j < 4; j++)
	      for (k = 0; k < 4; k++)
		{		 
		  __m256d 
		    left1 = _mm256_load_pd(&left[j * 16 + k * 4]);		  		  		  

		  left1 = _mm256_mul_pd(left1, tv);		  
		  left1 = hadd3(left1);
		  		  		  
		  _mm256_store_pd(&umpX1[i * 64 + j * 16 + k * 4], left1);
		}
	  
	    for (j = 0; j < 4; j++)
	      for (k = 0; k < 4; k++)
		{		 
		  __m256d 
		    left1 = _mm256_load_pd(&right[j * 16 + k * 4]);		  		  		  

		  left1 = _mm256_mul_pd(left1, tv);		  
		  left1 = hadd3(left1);
		  		  		  
		  _mm256_store_pd(&umpX2[i * 64 + j * 16 + k * 4], left1);
		}	    
	  }   	
	  

	for(i = 0; i < n; i++)
	  {	    		 	    
	    uX1 = &umpX1[64 * tipX1[i]];
	    uX2 = &umpX2[64 * tipX2[i]];		  
	    
	    for(k = 0; k < 4; k++)
	      {
		__m256d	   
		  xv = _mm256_setzero_pd();
	       
		int 
		  l;
		
		for(l = 0; l < 4; l++)
		  {	       	     				      	      																	   
		    __m256d
		      x1v =  _mm256_mul_pd(_mm256_load_pd(&uX1[k * 16 + l * 4]), _mm256_load_pd(&uX2[k * 16 + l * 4]));
		
		    __m256d 
		      evv = _mm256_load_pd(&extEV[l * 4]);
						  
		    xv = _mm256_add_pd(xv, _mm256_mul_pd(x1v, evv));
		  }
		
		_mm256_store_pd(&x3[16 * i + 4 * k], xv);
	      }	         	   	    
	  }
      }
      break;
    case TIP_INNER:
      {
	double 
	  *uX1, 
	  umpX1[1024] __attribute__ ((aligned (BYTE_ALIGNMENT)));

	for (i = 1; i < 16; i++)
	  {
	    __m256d 
	      tv = _mm256_load_pd(&(tipVector[i*4]));

	    int 
	      j;
	    
	    for (j = 0; j < 4; j++)
	      for (k = 0; k < 4; k++)
		{		 
		  __m256d 
		    left1 = _mm256_load_pd(&left[j * 16 + k * 4]);		  		  		  

		  left1 = _mm256_mul_pd(left1, tv);		  
		  left1 = hadd3(left1);
		  		  		  
		  _mm256_store_pd(&umpX1[i * 64 + j * 16 + k * 4], left1);
		}	 	   
	  }   	
	
	for(i = 0; i < n; i++)
	  { 
	    __m256d
	      xv[4];	    	   
	    
	    scale = 1;
	    uX1 = &umpX1[64 * tipX1[i]];

	    for(k = 0; k < 4; k++)
	      {
		__m256d	   		 
		  xvr = _mm256_load_pd(&(x2[i * 16 + k * 4]));

		int 
		  l;

		xv[k]  = _mm256_setzero_pd();
		  
		for(l = 0; l < 4; l++)
		  {	       	     				      	      															
		    __m256d  
		      x1v = _mm256_load_pd(&uX1[k * 16 + l * 4]),		     
		      x2v = _mm256_mul_pd(xvr, _mm256_load_pd(&right[k * 16 + l * 4]));			    
			
		    x2v = hadd3(x2v);
		    x1v = _mm256_mul_pd(x1v, x2v);			
		
		    __m256d 
		      evv = _mm256_load_pd(&extEV[l * 4]);
						  
		    xv[k] = _mm256_add_pd(xv[k], _mm256_mul_pd(x1v, evv));
		  }
		    
		if(scale)
		  {
		    __m256d 	     
		      v1 = _mm256_and_pd(xv[k], absMask_AVX.m);

		    v1 = _mm256_cmp_pd(v1,  minlikelihood_avx, _CMP_LT_OS);
		    
		    if(_mm256_movemask_pd( v1 ) != 15)
		      scale = 0;
		  }
	      }	    

	    if(scale)
	      {
		xv[0] = _mm256_mul_pd(xv[0], twoto);
		xv[1] = _mm256_mul_pd(xv[1], twoto);
		xv[2] = _mm256_mul_pd(xv[2], twoto);
		xv[3] = _mm256_mul_pd(xv[3], twoto);
		addScale += wgt[i];
	      }

	    _mm256_store_pd(&x3[16 * i],      xv[0]);
	    _mm256_store_pd(&x3[16 * i + 4],  xv[1]);
	    _mm256_store_pd(&x3[16 * i + 8],  xv[2]);
	    _mm256_store_pd(&x3[16 * i + 12], xv[3]);
	  }
      }
      break;
    case INNER_INNER:
      {
	for(i = 0; i < n; i++)
	  {	
	    __m256d
	      xv[4];
	    
	    scale = 1;

	    for(k = 0; k < 4; k++)
	      {
		__m256d	   
		 
		  xvl = _mm256_load_pd(&(x1[i * 16 + k * 4])),
		  xvr = _mm256_load_pd(&(x2[i * 16 + k * 4]));

		int 
		  l;

		xv[k] = _mm256_setzero_pd();

		for(l = 0; l < 4; l++)
		  {	       	     				      	      															
		    __m256d 
		      x1v = _mm256_mul_pd(xvl, _mm256_load_pd(&left[k * 16 + l * 4])),
		      x2v = _mm256_mul_pd(xvr, _mm256_load_pd(&right[k * 16 + l * 4]));			    
			
		    x1v = hadd4(x1v, x2v);			
		
		    __m256d 
		      evv = _mm256_load_pd(&extEV[l * 4]);
						  
		    xv[k] = _mm256_add_pd(xv[k], _mm256_mul_pd(x1v, evv));
		  }
		
		if(scale)
		  {
		    __m256d 	     
		      v1 = _mm256_and_pd(xv[k], absMask_AVX.m);

		    v1 = _mm256_cmp_pd(v1,  minlikelihood_avx, _CMP_LT_OS);
		    
		    if(_mm256_movemask_pd( v1 ) != 15)
		      scale = 0;
		  }
	      }

	     if(scale)
	      {
		xv[0] = _mm256_mul_pd(xv[0], twoto);
		xv[1] = _mm256_mul_pd(xv[1], twoto);
		xv[2] = _mm256_mul_pd(xv[2], twoto);
		xv[3] = _mm256_mul_pd(xv[3], twoto);
		addScale += wgt[i];
	      }
		
	    _mm256_store_pd(&x3[16 * i],      xv[0]);
	    _mm256_store_pd(&x3[16 * i + 4],  xv[1]);
	    _mm256_store_pd(&x3[16 * i + 8],  xv[2]);
	    _mm256_store_pd(&x3[16 * i + 12], xv[3]);
	  }
      }
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;
  
}


/*
static void  newviewGTRGAMMA_AVX2(int tipCase,
			 double *x1, double *x2, double *x3,
			 double *extEV, double *tipVector,
			 int *ex3, unsigned char *tipX1, unsigned char *tipX2,
			 const int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling
			 )
{
 
  int  
    i, 
    k, 
    scale, 
    addScale = 0;
 
  __m256d 
    minlikelihood_avx = _mm256_set1_pd( minlikelihood ),
    twoto = _mm256_set1_pd(twotothe256);
 

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for(i = 0; i < n; i++)
	  {
	    __m256d
	      xvl = _mm256_load_pd(&(tipVector[4 * tipX1[i]])),
	      xvr = _mm256_load_pd(&(tipVector[4 * tipX2[i]]));
				  
	    for(k = 0; k < 4; k++)
	      {
		__m256d	   
		  xv = _mm256_setzero_pd();

		int 
		  l;

		for(l = 0; l < 4; l++)
		  {	       	     				      	      															
		    __m256d 
		      x1v = _mm256_mul_pd(xvl, _mm256_load_pd(&left[k * 16 + l * 4])),
		      x2v = _mm256_mul_pd(xvr, _mm256_load_pd(&right[k * 16 + l * 4]));			    
			
		    x1v = hadd4(x1v, x2v);			
		
		    __m256d 
		      evv = _mm256_load_pd(&extEV[l * 4]);
						  
		    xv = _mm256_add_pd(xv, _mm256_mul_pd(x1v, evv));
		  }
		
		_mm256_store_pd(&x3[16 * i + 4 * k], xv);
	      }	         	   	    
	  }
      }
      break;
    case TIP_INNER:
      {
	for(i = 0; i < n; i++)
	  { 
	    __m256d
	      xv[4];
	    
	    __m256d
	      xvl = _mm256_load_pd(&(tipVector[4 * tipX1[i]]));
	    
	    scale = 1;

	    for(k = 0; k < 4; k++)
	      {
		__m256d	   		 
		  xvr = _mm256_load_pd(&(x2[i * 16 + k * 4]));

		int 
		  l;

		xv[k]  = _mm256_setzero_pd();
		  
		for(l = 0; l < 4; l++)
		  {	       	     				      	      															
		    __m256d 
		      x1v = _mm256_mul_pd(xvl, _mm256_load_pd(&left[k * 16 + l * 4])),
		      x2v = _mm256_mul_pd(xvr, _mm256_load_pd(&right[k * 16 + l * 4]));			    
			
		    x1v = hadd4(x1v, x2v);			
		
		    __m256d 
		      evv = _mm256_load_pd(&extEV[l * 4]);
						  
		    xv[k] = _mm256_add_pd(xv[k], _mm256_mul_pd(x1v, evv));
		  }
		    
		if(scale)
		  {
		    __m256d 	     
		      v1 = _mm256_and_pd(xv[k], absMask_AVX.m);

		    v1 = _mm256_cmp_pd(v1,  minlikelihood_avx, _CMP_LT_OS);
		    
		    if(_mm256_movemask_pd( v1 ) != 15)
		      scale = 0;
		  }
	      }	    

	    if(scale)
	      {
		xv[0] = _mm256_mul_pd(xv[0], twoto);
		xv[1] = _mm256_mul_pd(xv[1], twoto);
		xv[2] = _mm256_mul_pd(xv[2], twoto);
		xv[3] = _mm256_mul_pd(xv[3], twoto);
		addScale += wgt[i];
	      }

	    _mm256_store_pd(&x3[16 * i],      xv[0]);
	    _mm256_store_pd(&x3[16 * i + 4],  xv[1]);
	    _mm256_store_pd(&x3[16 * i + 8],  xv[2]);
	    _mm256_store_pd(&x3[16 * i + 12], xv[3]);
	  }
      }
      break;
    case INNER_INNER:
      {
	for(i = 0; i < n; i++)
	  {	
	    __m256d
	      xv[4];
	    
	    scale = 1;

	    for(k = 0; k < 4; k++)
	      {
		__m256d	   
		 
		  xvl = _mm256_load_pd(&(x1[i * 16 + k * 4])),
		  xvr = _mm256_load_pd(&(x2[i * 16 + k * 4]));

		int 
		  l;

		xv[k] = _mm256_setzero_pd();

		for(l = 0; l < 4; l++)
		  {	       	     				      	      															
		    __m256d 
		      x1v = _mm256_mul_pd(xvl, _mm256_load_pd(&left[k * 16 + l * 4])),
		      x2v = _mm256_mul_pd(xvr, _mm256_load_pd(&right[k * 16 + l * 4]));			    
			
		    x1v = hadd4(x1v, x2v);			
		
		    __m256d 
		      evv = _mm256_load_pd(&extEV[l * 4]);
						  
		    xv[k] = _mm256_add_pd(xv[k], _mm256_mul_pd(x1v, evv));
		  }
		
		if(scale)
		  {
		    __m256d 	     
		      v1 = _mm256_and_pd(xv[k], absMask_AVX.m);

		    v1 = _mm256_cmp_pd(v1,  minlikelihood_avx, _CMP_LT_OS);
		    
		    if(_mm256_movemask_pd( v1 ) != 15)
		      scale = 0;
		  }
	      }

	     if(scale)
	      {
		xv[0] = _mm256_mul_pd(xv[0], twoto);
		xv[1] = _mm256_mul_pd(xv[1], twoto);
		xv[2] = _mm256_mul_pd(xv[2], twoto);
		xv[3] = _mm256_mul_pd(xv[3], twoto);
		addScale += wgt[i];
	      }
		
	    _mm256_store_pd(&x3[16 * i],      xv[0]);
	    _mm256_store_pd(&x3[16 * i + 4],  xv[1]);
	    _mm256_store_pd(&x3[16 * i + 8],  xv[2]);
	    _mm256_store_pd(&x3[16 * i + 12], xv[3]);
	  }
      }
      break;
    default:
      assert(0);
    }

  if(useFastScaling)
    *scalerIncrement = addScale;
  
}
*/

void newviewGTRCAT_AVX(int tipCase,  double *EV,  int *cptr,
			   double *x1_start, double *x2_start,  double *x3_start, double *tipVector,
			   int *ex3, unsigned char *tipX1, unsigned char *tipX2,
			   int n,  double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double
    *le,
    *ri,
    *x1,
    *x2, 
    *x3;
    
  int 
    i, 
    j, 
    scale, 
    addScale = 0;
   
  __m256d 
    minlikelihood_avx = _mm256_set1_pd( minlikelihood ),
    twoto = _mm256_set1_pd(twotothe256);
  
  switch(tipCase)
    {
    case TIP_TIP:      
      for (i = 0; i < n; i++)
	{	 
	  int 
	    l;
	  
	  le = &left[cptr[i] * 16];
	  ri = &right[cptr[i] * 16];

	  x1 = &(tipVector[4 * tipX1[i]]);
	  x2 = &(tipVector[4 * tipX2[i]]);
	  
	  __m256d	   
	    vv = _mm256_setzero_pd();
	   	   	    
	  for(l = 0; l < 4; l++)
	    {	       	     				      	      															
	      __m256d 
		x1v = _mm256_mul_pd(_mm256_load_pd(x1), _mm256_load_pd(&le[l * 4])),
		x2v = _mm256_mul_pd(_mm256_load_pd(x2), _mm256_load_pd(&ri[l * 4]));			    
			
	      x1v = hadd4(x1v, x2v);			
		
	      __m256d 
		evv = _mm256_load_pd(&EV[l * 4]);
						
	      vv = _mm256_add_pd(vv, _mm256_mul_pd(x1v, evv));						      	
	    }	  		  

	  _mm256_store_pd(&x3_start[4 * i], vv);	    	   	    
	}
      break;
    case TIP_INNER:      
      for (i = 0; i < n; i++)
	{
	  int 
	    l;

	  x1 = &(tipVector[4 * tipX1[i]]);
	  x2 = &x2_start[4 * i];	 
	  
	  le =  &left[cptr[i] * 16];
	  ri =  &right[cptr[i] * 16];

	  __m256d	   
	    vv = _mm256_setzero_pd();
	  
	  for(l = 0; l < 4; l++)
	    {	       	     				      	      															
	      __m256d 
		x1v = _mm256_mul_pd(_mm256_load_pd(x1), _mm256_load_pd(&le[l * 4])),
		x2v = _mm256_mul_pd(_mm256_load_pd(x2), _mm256_load_pd(&ri[l * 4]));			    
			
	      x1v = hadd4(x1v, x2v);			
		
	      __m256d 
		evv = _mm256_load_pd(&EV[l * 4]);
				
	      /* vv = _mm256_fmadd_pd(x1v, evv, vv);*/
	      vv = _mm256_add_pd(vv, _mm256_mul_pd(x1v, evv));
	    }	  		  
	  
	  
	  __m256d 	     
	    v1 = _mm256_and_pd(vv, absMask_AVX.m);

	  v1 = _mm256_cmp_pd(v1,  minlikelihood_avx, _CMP_LT_OS);
	    
	  if(_mm256_movemask_pd( v1 ) == 15)
	    {	     	      
	      vv = _mm256_mul_pd(vv, twoto);	      
	      addScale += wgt[i];
	    }       
	  
	  _mm256_store_pd(&x3_start[4 * i], vv);	 	  	  
	}
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
	{
	  int 
	    l;

	  x1 = &x1_start[4 * i];
	  x2 = &x2_start[4 * i];
	  
	  
	  le =  &left[cptr[i] * 16];
	  ri =  &right[cptr[i] * 16];

	  __m256d	   
	    vv = _mm256_setzero_pd();
	  
	  for(l = 0; l < 4; l++)
	    {	       	     				      	      															
	      __m256d 
		x1v = _mm256_mul_pd(_mm256_load_pd(x1), _mm256_load_pd(&le[l * 4])),
		x2v = _mm256_mul_pd(_mm256_load_pd(x2), _mm256_load_pd(&ri[l * 4]));			    
			
	      x1v = hadd4(x1v, x2v);			
		
	      __m256d 
		evv = _mm256_load_pd(&EV[l * 4]);
						
	      vv = _mm256_add_pd(vv, _mm256_mul_pd(x1v, evv));						      	
	    }	  		  

	 
	  __m256d 	     
	    v1 = _mm256_and_pd(vv, absMask_AVX.m);

	  v1 = _mm256_cmp_pd(v1,  minlikelihood_avx, _CMP_LT_OS);
	    
	  if(_mm256_movemask_pd( v1 ) == 15)
	    {	
	      vv = _mm256_mul_pd(vv, twoto);	      
	      addScale += wgt[i];
	    }	

	  _mm256_store_pd(&x3_start[4 * i], vv);
	  	  
	}
      break;
    default:
      assert(0);
    }

  
  *scalerIncrement = addScale;
}

void newviewGTRCATPROT_AVX(int tipCase, double *extEV,
			       int *cptr,
			       double *x1, double *x2, double *x3, double *tipVector,
			       int *ex3, unsigned char *tipX1, unsigned char *tipX2,
			       int n, double *left, double *right, int *wgt, int *scalerIncrement, const boolean useFastScaling)
{
  double
    *le, *ri, *v, *vl, *vr;

  int i, l, j, scale, addScale = 0;

  switch(tipCase)
    {
    case TIP_TIP:
      {
	for (i = 0; i < n; i++)
	  {	   
	    le = &left[cptr[i] * 400];
	    ri = &right[cptr[i] * 400];

	    vl = &(tipVector[20 * tipX1[i]]);
	    vr = &(tipVector[20 * tipX2[i]]);
	    v  = &x3[20 * i];	    	    	   	    

	    __m256d vv[5];
	    
	    vv[0] = _mm256_setzero_pd();
	    vv[1] = _mm256_setzero_pd();
	    vv[2] = _mm256_setzero_pd();
	    vv[3] = _mm256_setzero_pd();
	    vv[4] = _mm256_setzero_pd();	   	    

	    for(l = 0; l < 20; l++)
	      {	       
		__m256d 
		  x1v = _mm256_setzero_pd(),
		  x2v = _mm256_setzero_pd();	
				
		double 
		  *ev = &extEV[l * 20],
		  *lv = &le[l * 20],
		  *rv = &ri[l * 20];														
		
		x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[0]), _mm256_load_pd(&lv[0])));
		x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[4]), _mm256_load_pd(&lv[4])));
		x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[8]), _mm256_load_pd(&lv[8])));
		x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[12]), _mm256_load_pd(&lv[12])));
		x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[16]), _mm256_load_pd(&lv[16])));

		x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[0]), _mm256_load_pd(&rv[0])));			    
		x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[4]), _mm256_load_pd(&rv[4])));				    
		x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[8]), _mm256_load_pd(&rv[8])));			    
		x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[12]), _mm256_load_pd(&rv[12])));				    
		x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[16]), _mm256_load_pd(&rv[16])));	

		x1v = hadd4(x1v, x2v);			
		
		__m256d 
		  evv[5];
	    	
		evv[0] = _mm256_load_pd(&ev[0]);
		evv[1] = _mm256_load_pd(&ev[4]);
		evv[2] = _mm256_load_pd(&ev[8]);
		evv[3] = _mm256_load_pd(&ev[12]);
		evv[4] = _mm256_load_pd(&ev[16]);		
		
		vv[0] = _mm256_add_pd(vv[0], _mm256_mul_pd(x1v, evv[0]));
		vv[1] = _mm256_add_pd(vv[1], _mm256_mul_pd(x1v, evv[1]));
		vv[2] = _mm256_add_pd(vv[2], _mm256_mul_pd(x1v, evv[2]));
		vv[3] = _mm256_add_pd(vv[3], _mm256_mul_pd(x1v, evv[3]));
		vv[4] = _mm256_add_pd(vv[4], _mm256_mul_pd(x1v, evv[4]));				      	
	      }	  

	    _mm256_store_pd(&v[0], vv[0]);
	    _mm256_store_pd(&v[4], vv[1]);
	    _mm256_store_pd(&v[8], vv[2]);
	    _mm256_store_pd(&v[12], vv[3]);
	    _mm256_store_pd(&v[16], vv[4]);
	  }
      }
      break;
    case TIP_INNER:      	
      for (i = 0; i < n; i++)
	{
	  le = &left[cptr[i] * 400];
	  ri = &right[cptr[i] * 400];
	  
	  vl = &(tipVector[20 * tipX1[i]]);
	  vr = &x2[20 * i];
	  v  = &x3[20 * i];	   
	  
	  __m256d vv[5];
	  
	  vv[0] = _mm256_setzero_pd();
	  vv[1] = _mm256_setzero_pd();
	  vv[2] = _mm256_setzero_pd();
	  vv[3] = _mm256_setzero_pd();
	  vv[4] = _mm256_setzero_pd();
	  
	  for(l = 0; l < 20; l++)
	    {	       
	      __m256d 
		x1v = _mm256_setzero_pd(),
		x2v = _mm256_setzero_pd();	
	      
	      double 
		*ev = &extEV[l * 20],
		*lv = &le[l * 20],
		*rv = &ri[l * 20];														
	      
	      x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[0]), _mm256_load_pd(&lv[0])));
	      x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[4]), _mm256_load_pd(&lv[4])));
	      x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[8]), _mm256_load_pd(&lv[8])));
	      x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[12]), _mm256_load_pd(&lv[12])));
	      x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[16]), _mm256_load_pd(&lv[16])));
	      
	      x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[0]), _mm256_load_pd(&rv[0])));			    
	      x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[4]), _mm256_load_pd(&rv[4])));				    
	      x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[8]), _mm256_load_pd(&rv[8])));			    
	      x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[12]), _mm256_load_pd(&rv[12])));				    
	      x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[16]), _mm256_load_pd(&rv[16])));

	      x1v = hadd4(x1v, x2v);			
	      
	      __m256d 
		evv[5];
	      
	      evv[0] = _mm256_load_pd(&ev[0]);
	      evv[1] = _mm256_load_pd(&ev[4]);
	      evv[2] = _mm256_load_pd(&ev[8]);
	      evv[3] = _mm256_load_pd(&ev[12]);
	      evv[4] = _mm256_load_pd(&ev[16]);		
	      
	      vv[0] = _mm256_add_pd(vv[0], _mm256_mul_pd(x1v, evv[0]));
	      vv[1] = _mm256_add_pd(vv[1], _mm256_mul_pd(x1v, evv[1]));
	      vv[2] = _mm256_add_pd(vv[2], _mm256_mul_pd(x1v, evv[2]));
	      vv[3] = _mm256_add_pd(vv[3], _mm256_mul_pd(x1v, evv[3]));
	      vv[4] = _mm256_add_pd(vv[4], _mm256_mul_pd(x1v, evv[4]));				      	
	    }	  

	   	     
	  __m256d minlikelihood_avx = _mm256_set1_pd( minlikelihood );
	  
	  scale = 1;
	  
	  for(l = 0; scale && (l < 20); l += 4)
	    {	       
	      __m256d 
		v1 = _mm256_and_pd(vv[l / 4], absMask_AVX.m);
	      v1 = _mm256_cmp_pd(v1,  minlikelihood_avx, _CMP_LT_OS);
	      
	      if(_mm256_movemask_pd( v1 ) != 15)
		scale = 0;
	    }	    	  	  

	  if(scale)
	    {
	      __m256d 
		twoto = _mm256_set1_pd(twotothe256);
	      
	      for(l = 0; l < 20; l += 4)
		vv[l / 4] = _mm256_mul_pd(vv[l / 4] , twoto);		    		 
	  
	      if(useFastScaling)
		addScale += wgt[i];
	      else
		ex3[i]  += 1;	      
	    }

	  _mm256_store_pd(&v[0], vv[0]);
	  _mm256_store_pd(&v[4], vv[1]);
	  _mm256_store_pd(&v[8], vv[2]);
	  _mm256_store_pd(&v[12], vv[3]);
	  _mm256_store_pd(&v[16], vv[4]);	       
	}
      break;
    case INNER_INNER:
      for(i = 0; i < n; i++)
	{
	  le = &left[cptr[i] * 400];
	  ri = &right[cptr[i] * 400];

	  vl = &x1[20 * i];
	  vr = &x2[20 * i];
	  v = &x3[20 * i];

	  __m256d vv[5];
	  
	  vv[0] = _mm256_setzero_pd();
	  vv[1] = _mm256_setzero_pd();
	  vv[2] = _mm256_setzero_pd();
	  vv[3] = _mm256_setzero_pd();
	  vv[4] = _mm256_setzero_pd();
	  
	  for(l = 0; l < 20; l++)
	    {	       
	      __m256d 
		x1v = _mm256_setzero_pd(),
		x2v = _mm256_setzero_pd();	
	      
	      double 
		*ev = &extEV[l * 20],
		*lv = &le[l * 20],
		*rv = &ri[l * 20];														
	      
	      x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[0]), _mm256_load_pd(&lv[0])));
	      x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[4]), _mm256_load_pd(&lv[4])));
	      x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[8]), _mm256_load_pd(&lv[8])));
	      x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[12]), _mm256_load_pd(&lv[12])));
	      x1v = _mm256_add_pd(x1v, _mm256_mul_pd(_mm256_load_pd(&vl[16]), _mm256_load_pd(&lv[16])));
	      
	      x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[0]), _mm256_load_pd(&rv[0])));			    
	      x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[4]), _mm256_load_pd(&rv[4])));				    
	      x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[8]), _mm256_load_pd(&rv[8])));			    
	      x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[12]), _mm256_load_pd(&rv[12])));				    
	      x2v = _mm256_add_pd(x2v,  _mm256_mul_pd(_mm256_load_pd(&vr[16]), _mm256_load_pd(&rv[16])));

	      x1v = hadd4(x1v, x2v);			
	      
	      __m256d 
		evv[5];
	      
	      evv[0] = _mm256_load_pd(&ev[0]);
	      evv[1] = _mm256_load_pd(&ev[4]);
	      evv[2] = _mm256_load_pd(&ev[8]);
	      evv[3] = _mm256_load_pd(&ev[12]);
	      evv[4] = _mm256_load_pd(&ev[16]);		
	      
	      vv[0] = _mm256_add_pd(vv[0], _mm256_mul_pd(x1v, evv[0]));
	      vv[1] = _mm256_add_pd(vv[1], _mm256_mul_pd(x1v, evv[1]));
	      vv[2] = _mm256_add_pd(vv[2], _mm256_mul_pd(x1v, evv[2]));
	      vv[3] = _mm256_add_pd(vv[3], _mm256_mul_pd(x1v, evv[3]));
	      vv[4] = _mm256_add_pd(vv[4], _mm256_mul_pd(x1v, evv[4]));				      	
	    }	  

	   	     
	  __m256d minlikelihood_avx = _mm256_set1_pd( minlikelihood );
	  
	  scale = 1;
	  
	  for(l = 0; scale && (l < 20); l += 4)
	    {	       
	      __m256d 
		v1 = _mm256_and_pd(vv[l / 4], absMask_AVX.m);
	      v1 = _mm256_cmp_pd(v1,  minlikelihood_avx, _CMP_LT_OS);
	      
	      if(_mm256_movemask_pd( v1 ) != 15)
		scale = 0;
	    }	    	  	  

	  if(scale)
	    {
	      __m256d 
		twoto = _mm256_set1_pd(twotothe256);
	      
	      for(l = 0; l < 20; l += 4)
		vv[l / 4] = _mm256_mul_pd(vv[l / 4] , twoto);		    		 
	  
	      if(useFastScaling)
		addScale += wgt[i];
	      else
		ex3[i]  += 1;	      
	    }

	  _mm256_store_pd(&v[0], vv[0]);
	  _mm256_store_pd(&v[4], vv[1]);
	  _mm256_store_pd(&v[8], vv[2]);
	  _mm256_store_pd(&v[12], vv[3]);
	  _mm256_store_pd(&v[16], vv[4]);
	 
	}
      break;
    default:
      assert(0);
    }
  
  if(useFastScaling)
    *scalerIncrement = addScale;
}
