/*  RAxML-VI-HPC (version 2.2) a program for sequential and parallel estimation of phylogenetic trees
 *  Copyright August 2006 by Alexandros Stamatakis
 *
 *  Partially derived from
 *  fastDNAml, a program for estimation of phylogenetic trees from sequences by Gary J. Olsen
 *
 *  and
 *
 *  Programs of the PHYLIP package by Joe Felsenstein.
 *
 *  This program is free software; you may redistribute it and/or modify its
 *  under the terms of the GNU General Public License as published by the Free
 *  Software Foundation; either version 2 of the License, or (at your option)
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 *  for more details.
 *
 *
 *  For any other enquiries send an Email to Alexandros Stamatakis
 *  Alexandros.Stamatakis@epfl.ch
 *
 *  When publishing work that is based on the results from RAxML-VI-HPC please cite:
 *
 *  Alexandros Stamatakis:"RAxML-VI-HPC: maximum likelihood-based phylogenetic analyses with
 *  thousands of taxa and mixed models".
 *  Bioinformatics 2006; doi: 10.1093/bioinformatics/btl446
 */

#ifndef WIN32
#include <unistd.h>
#endif



#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "axml.h"

#ifdef __SIM_SSE3
#include <xmmintrin.h>
#include <pmmintrin.h>
/*#include <tmmintrin.h>*/
#endif

#ifdef _USE_PTHREADS
extern volatile double *reductionBuffer;
extern volatile double *reductionBufferTwo;
extern volatile int NumberOfThreads;
#endif

#ifdef _FINE_GRAIN_MPI
extern double *globalResult;
#endif


extern const unsigned int mask32[32];

/*******************/

static inline boolean isGap(unsigned int *x, int pos)
{
  return (x[pos / 32] & mask32[pos % 32]);
}

static void sumCAT_SAVE(int tipCase, double *sum, double *x1_start, double *x2_start, double *tipVector,
			unsigned char *tipX1, unsigned char *tipX2, int n, double *x1_gapColumn, double *x2_gapColumn, unsigned int *x1_gap, unsigned int *x2_gap)
{
  int i;
  double 
    *x1, 
    *x2,    
    *x1_ptr = x1_start,
    *x2_ptr = x2_start;

  switch(tipCase)
    {
    case TIP_TIP:
      for (i = 0; i < n; i++)
	{
	  x1 = &(tipVector[4 * tipX1[i]]);
	  x2 = &(tipVector[4 * tipX2[i]]);

	  _mm_store_pd( &sum[i*4 + 0], _mm_mul_pd( _mm_load_pd( &x1[0] ), _mm_load_pd( &x2[0] )));
	  _mm_store_pd( &sum[i*4 + 2], _mm_mul_pd( _mm_load_pd( &x1[2] ), _mm_load_pd( &x2[2] )));
	}
      break;
    case TIP_INNER:
      for (i = 0; i < n; i++)
	{
	  x1 = &(tipVector[4 * tipX1[i]]);
	  if(isGap(x2_gap, i))
	    x2 = x2_gapColumn;
	  else
	    {
	      x2 = x2_ptr;
	      x2_ptr += 4;
	    }

	  _mm_store_pd( &sum[i*4 + 0], _mm_mul_pd( _mm_load_pd( &x1[0] ), _mm_load_pd( &x2[0] )));
	  _mm_store_pd( &sum[i*4 + 2], _mm_mul_pd( _mm_load_pd( &x1[2] ), _mm_load_pd( &x2[2] )));
	}
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
	{
	  if(isGap(x1_gap, i))
	    x1 = x1_gapColumn;
	  else
	    {
	      x1 = x1_ptr;
	      x1_ptr += 4;
	    }

	  if(isGap(x2_gap, i))
	    x2 = x2_gapColumn;
	  else
	    {
	      x2 = x2_ptr;
	      x2_ptr += 4;
	    }

	  _mm_store_pd( &sum[i*4 + 0], _mm_mul_pd( _mm_load_pd( &x1[0] ), _mm_load_pd( &x2[0] )));
	  _mm_store_pd( &sum[i*4 + 2], _mm_mul_pd( _mm_load_pd( &x1[2] ), _mm_load_pd( &x2[2] )));

	}    
      break;
    default:
      assert(0);
    }
}


static void sumGAMMA_GAPPED_SAVE(int tipCase, double *sumtable, double *x1_start, double *x2_start, double *tipVector,
				 unsigned char *tipX1, unsigned char *tipX2, int n, 
				 double *x1_gapColumn, double *x2_gapColumn, unsigned int *x1_gap, unsigned int *x2_gap)
{
  double 
    *x1, 
    *x2, 
    *sum,
    *x1_ptr = x1_start,
    *x2_ptr = x2_start;
  
  int i, j, k; 

  switch(tipCase)
    {
    case TIP_TIP:     
      for (i = 0; i < n; i++)
	{
	  x1 = &(tipVector[4 * tipX1[i]]);
	  x2 = &(tipVector[4 * tipX2[i]]);
	  sum = &sumtable[i * 16];
#ifndef __SIM_SSE3
	  for(j = 0; j < 4; j++)	    
	    for(k = 0; k < 4; k++)
	      sum[j * 4 + k] = x1[k] * x2[k];
#else
	  for(j = 0; j < 4; j++)	    
	    for(k = 0; k < 4; k+=2)
	      _mm_store_pd( &sum[j*4 + k], _mm_mul_pd( _mm_load_pd( &x1[k] ), _mm_load_pd( &x2[k] )));
#endif
	}
      break;
    case TIP_INNER:
      for (i = 0; i < n; i++)
	{
	  x1  = &(tipVector[4 * tipX1[i]]);
	  
	  if(x2_gap[i / 32] & mask32[i % 32])
	    x2 = x2_gapColumn;
	  else
	    {
	      x2  = x2_ptr;
	      x2_ptr += 16;
	    }
	  
	  sum = &sumtable[16 * i];
#ifndef __SIM_SSE3
	  for(j = 0; j < 4; j++)
	    for(k = 0; k < 4; k++)
	      sum[j * 4 + k] = x1[k] * x2[j * 4 + k];
#else
	  for(j = 0; j < 4; j++)	    
	    for(k = 0; k < 4; k+=2)
	      _mm_store_pd( &sum[j*4 + k], _mm_mul_pd( _mm_load_pd( &x1[k] ), _mm_load_pd( &x2[j * 4 + k] )));
#endif
	}
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
	{
	  if(x1_gap[i / 32] & mask32[i % 32])
	    x1 = x1_gapColumn;
	  else
	    {
	      x1  = x1_ptr;
	      x1_ptr += 16;
	    }
	  
	  if(x2_gap[i / 32] & mask32[i % 32])
	    x2 = x2_gapColumn;
	  else
	    {
	      x2  = x2_ptr;
	      x2_ptr += 16;
	    }

	  sum = &sumtable[16 * i];
	  
#ifndef __SIM_SSE3
	  for(j = 0; j < 4; j++)
	    for(k = 0; k < 4; k++)
	      sum[j * 4 + k] = x1[j * 4 + k] * x2[j * 4 + k];
#else
	   for(j = 0; j < 4; j++)	    
	    for(k = 0; k < 4; k+=2)
	      _mm_store_pd( &sum[j*4 + k], _mm_mul_pd( _mm_load_pd( &x1[j * 4 + k] ), _mm_load_pd( &x2[j * 4 + k] )));
#endif
	}
      break;
    default:
      assert(0);
    }
}




static void sumGAMMA(int tipCase, double *sumtable, double *x1_start, double *x2_start, double *tipVector,
		     unsigned char *tipX1, unsigned char *tipX2, int n)
{
  double *x1, *x2, *sum;
  int i, j, k;

  /* C-OPT once again switch over possible configurations at inner node */

  switch(tipCase)
    {
    case TIP_TIP:
      /* C-OPT main for loop overt alignment length */
      for (i = 0; i < n; i++)
	{
	  x1 = &(tipVector[4 * tipX1[i]]);
	  x2 = &(tipVector[4 * tipX2[i]]);
	  sum = &sumtable[i * 16];
#ifndef __SIM_SSE3
	  for(j = 0; j < 4; j++)	    
	    for(k = 0; k < 4; k++)
	      sum[j * 4 + k] = x1[k] * x2[k];
#else
	  for(j = 0; j < 4; j++)	    
	    for(k = 0; k < 4; k+=2)
	      _mm_store_pd( &sum[j*4 + k], _mm_mul_pd( _mm_load_pd( &x1[k] ), _mm_load_pd( &x2[k] )));
#endif
	}
      break;
    case TIP_INNER:
      for (i = 0; i < n; i++)
	{
	  x1  = &(tipVector[4 * tipX1[i]]);
	  x2  = &x2_start[16 * i];
	  sum = &sumtable[16 * i];
#ifndef __SIM_SSE3
	  for(j = 0; j < 4; j++)
	    for(k = 0; k < 4; k++)
	      sum[j * 4 + k] = x1[k] * x2[j * 4 + k];
#else
	  for(j = 0; j < 4; j++)	    
	    for(k = 0; k < 4; k+=2)
	      _mm_store_pd( &sum[j*4 + k], _mm_mul_pd( _mm_load_pd( &x1[k] ), _mm_load_pd( &x2[j * 4 + k] )));
#endif
	}
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
	{
	  x1  = &x1_start[16 * i];
	  x2  = &x2_start[16 * i];
	  sum = &sumtable[16 * i];
#ifndef __SIM_SSE3
	  for(j = 0; j < 4; j++)
	    for(k = 0; k < 4; k++)
	      sum[j * 4 + k] = x1[j * 4 + k] * x2[j * 4 + k];
#else
	   for(j = 0; j < 4; j++)	    
	    for(k = 0; k < 4; k+=2)
	      _mm_store_pd( &sum[j*4 + k], _mm_mul_pd( _mm_load_pd( &x1[j * 4 + k] ), _mm_load_pd( &x2[j * 4 + k] )));
#endif
	}
      break;
    default:
      assert(0);
    }
}


static void sumCAT(int tipCase, double *sum, double *x1_start, double *x2_start, double *tipVector,
		   unsigned char *tipX1, unsigned char *tipX2, int n)
{
  int i;
  double 
    *x1, 
    *x2;

  switch(tipCase)
    {
    case TIP_TIP:
      for (i = 0; i < n; i++)
	{
	  x1 = &(tipVector[4 * tipX1[i]]);
	  x2 = &(tipVector[4 * tipX2[i]]);

	  _mm_store_pd( &sum[i*4 + 0], _mm_mul_pd( _mm_load_pd( &x1[0] ), _mm_load_pd( &x2[0] )));
	  _mm_store_pd( &sum[i*4 + 2], _mm_mul_pd( _mm_load_pd( &x1[2] ), _mm_load_pd( &x2[2] )));
	}
      break;
    case TIP_INNER:
      for (i = 0; i < n; i++)
	{
	  x1 = &(tipVector[4 * tipX1[i]]);
	  x2 = &x2_start[4 * i];

	  _mm_store_pd( &sum[i*4 + 0], _mm_mul_pd( _mm_load_pd( &x1[0] ), _mm_load_pd( &x2[0] )));
	  _mm_store_pd( &sum[i*4 + 2], _mm_mul_pd( _mm_load_pd( &x1[2] ), _mm_load_pd( &x2[2] )));
	}
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
	{
	  x1 = &x1_start[4 * i];
	  x2 = &x2_start[4 * i];

	  _mm_store_pd( &sum[i*4 + 0], _mm_mul_pd( _mm_load_pd( &x1[0] ), _mm_load_pd( &x2[0] )));
	  _mm_store_pd( &sum[i*4 + 2], _mm_mul_pd( _mm_load_pd( &x1[2] ), _mm_load_pd( &x2[2] )));

	}    
      break;
    default:
      assert(0);
    }
}

static void coreGTRGAMMA(const int upper, double *sumtable,
			 volatile double *ext_dlnLdlz,  volatile double *ext_d2lnLdlz2, double *EIGN, double *gammaRates, double lz, int *wrptr)
{
  double 
    dlnLdlz = 0.0,
    d2lnLdlz2 = 0.0,
    ki, 
    kisqr,  
    inv_Li, 
    dlnLidlz, 
    d2lnLidlz2,  
    *sum, 
    diagptable0[16] __attribute__ ((aligned (16))),
    diagptable1[16] __attribute__ ((aligned (16))),
    diagptable2[16] __attribute__ ((aligned (16)));    
    
  int     
    i, 
    j, 
    l;
  
  for(i = 0; i < 4; i++)
    {
      ki = gammaRates[i];
      kisqr = ki * ki;
      
      diagptable0[i * 4] = 1.0;
      diagptable1[i * 4] = 0.0;
      diagptable2[i * 4] = 0.0;

      for(l = 1; l < 4; l++)
	{
	  diagptable0[i * 4 + l] = EXP(EIGN[l-1] * ki * lz);
	  diagptable1[i * 4 + l] = EIGN[l-1] * ki;
	  diagptable2[i * 4 + l] = EIGN[l-1] * EIGN[l-1] * kisqr;
	}
    }

  for (i = 0; i < upper; i++)
    { 
      __m128d a0 = _mm_setzero_pd();
      __m128d a1 = _mm_setzero_pd();
      __m128d a2 = _mm_setzero_pd();

      sum = &sumtable[i * 16];         

      for(j = 0; j < 4; j++)
	{	 	  	
	  double 	   
	    *d0 = &diagptable0[j * 4],
	    *d1 = &diagptable1[j * 4],
	    *d2 = &diagptable2[j * 4];
  	 	 
	  for(l = 0; l < 4; l+=2)
	    {
	      __m128d tmpv = _mm_mul_pd(_mm_load_pd(&d0[l]), _mm_load_pd(&sum[j * 4 + l]));
	      a0 = _mm_add_pd(a0, tmpv);
	      a1 = _mm_add_pd(a1, _mm_mul_pd(tmpv, _mm_load_pd(&d1[l])));
	      a2 = _mm_add_pd(a2, _mm_mul_pd(tmpv, _mm_load_pd(&d2[l])));
	    }	 	  
	}

      a0 = _mm_hadd_pd(a0, a0);
      a1 = _mm_hadd_pd(a1, a1);
      a2 = _mm_hadd_pd(a2, a2);

      _mm_storel_pd(&inv_Li, a0);     
      _mm_storel_pd(&dlnLidlz, a1);
      _mm_storel_pd(&d2lnLidlz2, a2); 

      inv_Li = 1.0 / inv_Li;
     
      dlnLidlz   *= inv_Li;
      d2lnLidlz2 *= inv_Li;     

      dlnLdlz   += wrptr[i] * dlnLidlz;
      d2lnLdlz2 += wrptr[i] * (d2lnLidlz2 - dlnLidlz * dlnLidlz);
    }

 
  *ext_dlnLdlz   = dlnLdlz;
  *ext_d2lnLdlz2 = d2lnLdlz2; 
}



static void coreGTRCAT(int upper, int numberOfCategories, double *sum,
			   volatile double *d1, volatile double *d2, double *wrptr, double *wr2ptr,
			   double *rptr, double *EIGN, int *cptr, double lz)
{
  int i;
  double
    *d, *d_start,
    inv_Li, dlnLidlz, d2lnLidlz2,
    dlnLdlz = 0.0,
    d2lnLdlz2 = 0.0;
  double e1[4] __attribute__ ((aligned (16)));
  double e2[4] __attribute__ ((aligned (16)));
  double dd1, dd2, dd3;

  __m128d
    e1v[2],
    e2v[2];

  e1[0] = 0.0;
  e2[0] = 0.0;
  e1[1] = EIGN[0];
  e2[1] = EIGN[0] * EIGN[0];
  e1[2] = EIGN[1];
  e2[2] = EIGN[1] * EIGN[1];
  e1[3] = EIGN[2];
  e2[3] = EIGN[2] * EIGN[2];

  e1v[0]= _mm_load_pd(&e1[0]);
  e1v[1]= _mm_load_pd(&e1[2]);

  e2v[0]= _mm_load_pd(&e2[0]);
  e2v[1]= _mm_load_pd(&e2[2]);

  d = d_start = (double *)malloc_aligned(numberOfCategories * 4 * sizeof(double));

  dd1 = EIGN[0] * lz;
  dd2 = EIGN[1] * lz;
  dd3 = EIGN[2] * lz;

  for(i = 0; i < numberOfCategories; i++)
    {
      d[i * 4 + 0] = 1.0;
      d[i * 4 + 1] = EXP(dd1 * rptr[i]);
      d[i * 4 + 2] = EXP(dd2 * rptr[i]);
      d[i * 4 + 3] = EXP(dd3 * rptr[i]);
    }

  for (i = 0; i < upper; i++)
    {
      double *s = &sum[4 * i];
      d = &d_start[4 * cptr[i]];  
      
      __m128d tmp_0v =_mm_mul_pd(_mm_load_pd(&d[0]),_mm_load_pd(&s[0]));
      __m128d tmp_1v =_mm_mul_pd(_mm_load_pd(&d[2]),_mm_load_pd(&s[2]));

      __m128d inv_Liv    = _mm_add_pd(tmp_0v, tmp_1v);      
            	  
      __m128d dlnLidlzv   = _mm_add_pd(_mm_mul_pd(tmp_0v, e1v[0]), _mm_mul_pd(tmp_1v, e1v[1]));	  
      __m128d d2lnLidlz2v = _mm_add_pd(_mm_mul_pd(tmp_0v, e2v[0]), _mm_mul_pd(tmp_1v, e2v[1]));


      inv_Liv   = _mm_hadd_pd(inv_Liv, inv_Liv);
      dlnLidlzv = _mm_hadd_pd(dlnLidlzv, dlnLidlzv);
      d2lnLidlz2v = _mm_hadd_pd(d2lnLidlz2v, d2lnLidlz2v);                 
 
      _mm_storel_pd(&inv_Li, inv_Liv);     
      _mm_storel_pd(&dlnLidlz, dlnLidlzv);                 
      _mm_storel_pd(&d2lnLidlz2, d2lnLidlz2v);      

      inv_Li = 1.0/inv_Li;

      dlnLidlz   *= inv_Li;
      d2lnLidlz2 *= inv_Li;

      dlnLdlz   += wrptr[i] * dlnLidlz;
      d2lnLdlz2 += wr2ptr[i] * (d2lnLidlz2 - dlnLidlz * dlnLidlz);
    }

  *d1 = dlnLdlz;
  *d2 = d2lnLdlz2;

  free(d_start);
}


static void sumGAMMAPROT_GAPPED_SAVE(int tipCase, double *sumtable, double *x1, double *x2, double *tipVector,
				     unsigned char *tipX1, unsigned char *tipX2, int n, 
				     double *x1_gapColumn, double *x2_gapColumn, unsigned int *x1_gap, unsigned int *x2_gap)
{
  int i, l, k;
  double 
    *left, 
    *right, 
    *sum,
    *x1_ptr = x1,
    *x2_ptr = x2,
    *x1v,
    *x2v;

  switch(tipCase)
    {
    case TIP_TIP:
      for(i = 0; i < n; i++)
	{
	  left  = &(tipVector[20 * tipX1[i]]);
	  right = &(tipVector[20 * tipX2[i]]);

	  for(l = 0; l < 4; l++)
	    {
	      sum = &sumtable[i * 80 + l * 20];

	      for(k = 0; k < 20; k+=2)
		{
		  __m128d sumv = _mm_mul_pd(_mm_load_pd(&left[k]), _mm_load_pd(&right[k]));
		  
		  _mm_store_pd(&sum[k], sumv);		 
		}

	    }
	}
      break;
    case TIP_INNER:
      for(i = 0; i < n; i++)
	{
	  left = &(tipVector[20 * tipX1[i]]);
	   
	  if(x2_gap[i / 32] & mask32[i % 32])
	    x2v = x2_gapColumn;
	  else
	    {
	      x2v = x2_ptr;
	      x2_ptr += 80;
	    }
	  
	  for(l = 0; l < 4; l++)
	    {
	      right = &(x2v[l * 20]);
	      sum = &sumtable[i * 80 + l * 20];

	      for(k = 0; k < 20; k+=2)
		{
		  __m128d sumv = _mm_mul_pd(_mm_load_pd(&left[k]), _mm_load_pd(&right[k]));
		  
		  _mm_store_pd(&sum[k], sumv);		 
		}
	    }
	}
      break;
    case INNER_INNER:
      for(i = 0; i < n; i++)
	{
	  if(x1_gap[i / 32] & mask32[i % 32])
	    x1v = x1_gapColumn;
	  else
	    {
	      x1v  = x1_ptr;
	      x1_ptr += 80;
	    }

	  if(x2_gap[i / 32] & mask32[i % 32])
	    x2v = x2_gapColumn;
	  else
	    {
	      x2v  = x2_ptr;
	      x2_ptr += 80;
	    }

	  for(l = 0; l < 4; l++)
	    {
	      left  = &(x1v[l * 20]);
	      right = &(x2v[l * 20]);
	      sum   = &(sumtable[i * 80 + l * 20]);

	      for(k = 0; k < 20; k+=2)
		{
		  __m128d sumv = _mm_mul_pd(_mm_load_pd(&left[k]), _mm_load_pd(&right[k]));
		  
		  _mm_store_pd(&sum[k], sumv);		 
		}
	    }
	}
      break;
    default:
      assert(0);
    }
}


static void sumGAMMAPROT(int tipCase, double *sumtable, double *x1, double *x2, double *tipVector,
			 unsigned char *tipX1, unsigned char *tipX2, int n)
{
  int i, l, k;
  double *left, *right, *sum;

  switch(tipCase)
    {
    case TIP_TIP:
      for(i = 0; i < n; i++)
	{
	  left  = &(tipVector[20 * tipX1[i]]);
	  right = &(tipVector[20 * tipX2[i]]);

	  for(l = 0; l < 4; l++)
	    {
	      sum = &sumtable[i * 80 + l * 20];
#ifdef __SIM_SSE3
	      for(k = 0; k < 20; k+=2)
		{
		  __m128d sumv = _mm_mul_pd(_mm_load_pd(&left[k]), _mm_load_pd(&right[k]));
		  
		  _mm_store_pd(&sum[k], sumv);		 
		}
#else
	      for(k = 0; k < 20; k++)
		sum[k] = left[k] * right[k];
#endif
	    }
	}
      break;
    case TIP_INNER:
      for(i = 0; i < n; i++)
	{
	  left = &(tipVector[20 * tipX1[i]]);

	  for(l = 0; l < 4; l++)
	    {
	      right = &(x2[80 * i + l * 20]);
	      sum = &sumtable[i * 80 + l * 20];
#ifdef __SIM_SSE3
	      for(k = 0; k < 20; k+=2)
		{
		  __m128d sumv = _mm_mul_pd(_mm_load_pd(&left[k]), _mm_load_pd(&right[k]));
		  
		  _mm_store_pd(&sum[k], sumv);		 
		}
#else
	      for(k = 0; k < 20; k++)
		sum[k] = left[k] * right[k];
#endif
	    }
	}
      break;
    case INNER_INNER:
      for(i = 0; i < n; i++)
	{
	  for(l = 0; l < 4; l++)
	    {
	      left  = &(x1[80 * i + l * 20]);
	      right = &(x2[80 * i + l * 20]);
	      sum   = &(sumtable[i * 80 + l * 20]);

#ifdef __SIM_SSE3
	      for(k = 0; k < 20; k+=2)
		{
		  __m128d sumv = _mm_mul_pd(_mm_load_pd(&left[k]), _mm_load_pd(&right[k]));
		  
		  _mm_store_pd(&sum[k], sumv);		 
		}
#else
	      for(k = 0; k < 20; k++)
		sum[k] = left[k] * right[k];
#endif
	    }
	}
      break;
    default:
      assert(0);
    }
}


static void sumGTRCATPROT(int tipCase, double *sumtable, double *x1, double *x2, double *tipVector,
			  unsigned char *tipX1, unsigned char *tipX2, int n)
{
  int i, l;
  double *sum, *left, *right;

  switch(tipCase)
    {
    case TIP_TIP:
      for (i = 0; i < n; i++)
	{
	  left  = &(tipVector[20 * tipX1[i]]);
	  right = &(tipVector[20 * tipX2[i]]);
	  sum = &sumtable[20 * i];
#ifdef __SIM_SSE3
	  for(l = 0; l < 20; l+=2)
	    {
	      __m128d sumv = _mm_mul_pd(_mm_load_pd(&left[l]), _mm_load_pd(&right[l]));
	      
	      _mm_store_pd(&sum[l], sumv);		 
	    }
#else
	  for(l = 0; l < 20; l++)
	    sum[l] = left[l] * right[l];
#endif
	}
      break;
    case TIP_INNER:
      for (i = 0; i < n; i++)
	{
	  left = &(tipVector[20 * tipX1[i]]);
	  right = &x2[20 * i];
	  sum = &sumtable[20 * i];
#ifdef __SIM_SSE3
	  for(l = 0; l < 20; l+=2)
	    {
	      __m128d sumv = _mm_mul_pd(_mm_load_pd(&left[l]), _mm_load_pd(&right[l]));
	      
	      _mm_store_pd(&sum[l], sumv);		 
	    }
#else
	  for(l = 0; l < 20; l++)
	    sum[l] = left[l] * right[l];
#endif
	}
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
	{
	  left  = &x1[20 * i];
	  right = &x2[20 * i];
	  sum = &sumtable[20 * i];
#ifdef __SIM_SSE3
	  for(l = 0; l < 20; l+=2)
	    {
	      __m128d sumv = _mm_mul_pd(_mm_load_pd(&left[l]), _mm_load_pd(&right[l]));
	      
	      _mm_store_pd(&sum[l], sumv);		 
	    }
#else
	  for(l = 0; l < 20; l++)
	    sum[l] = left[l] * right[l];
#endif
	}
      break;
    default:
      assert(0);
    }
}


static void sumGTRCATPROT_SAVE(int tipCase, double *sumtable, double *x1, double *x2, double *tipVector,
			       unsigned char *tipX1, unsigned char *tipX2, int n, 
			       double *x1_gapColumn, double *x2_gapColumn, unsigned int *x1_gap, unsigned int *x2_gap)
{
  int 
    i, 
    l;
  
  double 
    *sum, 
    *left, 
    *right,
    *left_ptr = x1,
    *right_ptr = x2;

  switch(tipCase)
    {
    case TIP_TIP:
      for (i = 0; i < n; i++)
	{
	  left  = &(tipVector[20 * tipX1[i]]);
	  right = &(tipVector[20 * tipX2[i]]);
	  sum = &sumtable[20 * i];

	  for(l = 0; l < 20; l+=2)
	    {
	      __m128d sumv = _mm_mul_pd(_mm_load_pd(&left[l]), _mm_load_pd(&right[l]));
	      
	      _mm_store_pd(&sum[l], sumv);		 
	    }

	}
      break;
    case TIP_INNER:
      for (i = 0; i < n; i++)
	{
	  left = &(tipVector[20 * tipX1[i]]);       
	   
	  if(isGap(x2_gap, i))
	    right = x2_gapColumn;
	  else
	    {
	      right = right_ptr;
	      right_ptr += 20;
	    }
	  
	  sum = &sumtable[20 * i];

	  for(l = 0; l < 20; l+=2)
	    {
	      __m128d sumv = _mm_mul_pd(_mm_load_pd(&left[l]), _mm_load_pd(&right[l]));
	      
	      _mm_store_pd(&sum[l], sumv);		 
	    }

	}
      break;
    case INNER_INNER:
      for (i = 0; i < n; i++)
	{	 
	  if(isGap(x1_gap, i))
	    left = x1_gapColumn;
	  else
	    {
	      left = left_ptr;
	      left_ptr += 20;
	    }

	  if(isGap(x2_gap, i))
	    right = x2_gapColumn;
	  else
	    {
	      right = right_ptr;
	      right_ptr += 20;
	    }

	  sum = &sumtable[20 * i];

	  for(l = 0; l < 20; l+=2)
	    {
	      __m128d sumv = _mm_mul_pd(_mm_load_pd(&left[l]), _mm_load_pd(&right[l]));
	      
	      _mm_store_pd(&sum[l], sumv);		 
	    }
	}
      break;
    default:
      assert(0);
    }
}


static void coreGTRGAMMAPROT(double *gammaRates, double *EIGN, double *sumtable, int upper, int *wrptr,
			      volatile double *ext_dlnLdlz,  volatile double *ext_d2lnLdlz2, double lz)
{
  double  *sum, 
    diagptable0[80] __attribute__ ((aligned (16))),
    diagptable1[80] __attribute__ ((aligned (16))),
    diagptable2[80] __attribute__ ((aligned (16)));    
  int     i, j, l;
  double  dlnLdlz = 0;
  double d2lnLdlz2 = 0;
  double ki, kisqr; 
  double inv_Li, dlnLidlz, d2lnLidlz2;

  for(i = 0; i < 4; i++)
    {
      ki = gammaRates[i];
      kisqr = ki * ki;
      
      diagptable0[i * 20] = 1.0;
      diagptable1[i * 20] = 0.0;
      diagptable2[i * 20] = 0.0;

      for(l = 1; l < 20; l++)
	{
	  diagptable0[i * 20 + l] = EXP(EIGN[l-1] * ki * lz);
	  diagptable1[i * 20 + l] = EIGN[l-1] * ki;
	  diagptable2[i * 20 + l] = EIGN[l-1] * EIGN[l-1] * kisqr;
	}
    }

  for (i = 0; i < upper; i++)
    { 
      __m128d a0 = _mm_setzero_pd();
      __m128d a1 = _mm_setzero_pd();
      __m128d a2 = _mm_setzero_pd();

      sum = &sumtable[i * 80];         

      for(j = 0; j < 4; j++)
	{	 	  	
	  double 	   
	    *d0 = &diagptable0[j * 20],
	    *d1 = &diagptable1[j * 20],
	    *d2 = &diagptable2[j * 20];
  	 	 
	  for(l = 0; l < 20; l+=2)
	    {
	      __m128d tmpv = _mm_mul_pd(_mm_load_pd(&d0[l]), _mm_load_pd(&sum[j * 20 +l]));
	      a0 = _mm_add_pd(a0, tmpv);
	      a1 = _mm_add_pd(a1, _mm_mul_pd(tmpv, _mm_load_pd(&d1[l])));
	      a2 = _mm_add_pd(a2, _mm_mul_pd(tmpv, _mm_load_pd(&d2[l])));
	    }	 	  
	}

      a0 = _mm_hadd_pd(a0, a0);
      a1 = _mm_hadd_pd(a1, a1);
      a2 = _mm_hadd_pd(a2, a2);

      _mm_storel_pd(&inv_Li, a0);
      _mm_storel_pd(&dlnLidlz, a1);
      _mm_storel_pd(&d2lnLidlz2, a2);

      inv_Li = 1.0 / inv_Li;

      dlnLidlz   *= inv_Li;
      d2lnLidlz2 *= inv_Li;

      dlnLdlz   += wrptr[i] * dlnLidlz;
      d2lnLdlz2 += wrptr[i] * (d2lnLidlz2 - dlnLidlz * dlnLidlz);
    }

  *ext_dlnLdlz   = dlnLdlz;
  *ext_d2lnLdlz2 = d2lnLdlz2;
}



static void coreGTRCATPROT(double *EIGN, double lz, int numberOfCategories, double *rptr, int *cptr, int upper,
			   double *wrptr, double *wr2ptr,  volatile double *ext_dlnLdlz,  volatile double *ext_d2lnLdlz2, double *sumtable)
{
  int i, l;
  double *d1, *d_start, *sum;
  double 
    e[20] __attribute__ ((aligned (16))), 
    s[20] __attribute__ ((aligned (16))), 
    dd[20] __attribute__ ((aligned (16)));
  double inv_Li, dlnLidlz, d2lnLidlz2;
  double  dlnLdlz = 0.0;
  double  d2lnLdlz2 = 0.0;

  d1 = d_start = (double *)malloc_aligned(numberOfCategories * 20 * sizeof(double));

  e[0] = 0.0;
  s[0] = 0.0; 

  for(l = 1; l < 20; l++)
    {
      e[l]  = EIGN[l-1] * EIGN[l-1];
      s[l]  = EIGN[l-1];
      dd[l] = s[l] * lz;
    }

  for(i = 0; i < numberOfCategories; i++)
    {      
      d1[20 * i] = 1.0;
      for(l = 1; l < 20; l++)
	d1[20 * i + l] = EXP(dd[l] * rptr[i]);
    }

  for (i = 0; i < upper; i++)
    {
      __m128d a0 = _mm_setzero_pd();
      __m128d a1 = _mm_setzero_pd();
      __m128d a2 = _mm_setzero_pd();

      d1 = &d_start[20 * cptr[i]];
      sum = &sumtable[20 * i];
          
      for(l = 0; l < 20; l+=2)
	{	  
	  __m128d tmpv = _mm_mul_pd(_mm_load_pd(&d1[l]), _mm_load_pd(&sum[l]));
	  
	  a0 = _mm_add_pd(a0, tmpv);
	  __m128d sv = _mm_load_pd(&s[l]);	  
	  
	  a1 = _mm_add_pd(a1, _mm_mul_pd(tmpv, sv));
	  __m128d ev = _mm_load_pd(&e[l]);	  

	  a2 = _mm_add_pd(a2, _mm_mul_pd(tmpv, ev));
	}

      a0 = _mm_hadd_pd(a0, a0);
      a1 = _mm_hadd_pd(a1, a1);
      a2 = _mm_hadd_pd(a2, a2);

      _mm_storel_pd(&inv_Li, a0);     
      _mm_storel_pd(&dlnLidlz, a1);                 
      _mm_storel_pd(&d2lnLidlz2, a2);
      
      inv_Li = 1.0/inv_Li;

      dlnLidlz   *= inv_Li;
      d2lnLidlz2 *= inv_Li;

      dlnLdlz  += wrptr[i] * dlnLidlz;
      d2lnLdlz2 += wr2ptr[i] * (d2lnLidlz2 - dlnLidlz * dlnLidlz);
    }

  *ext_dlnLdlz   = dlnLdlz;
  *ext_d2lnLdlz2 = d2lnLdlz2;

  free(d_start);
}



static void getVects(tree *tr, unsigned char **tipX1, unsigned char **tipX2, double **x1_start, double **x2_start, int *tipCase, int model,
		     double **x1_gapColumn, double **x2_gapColumn, unsigned int **x1_gap, unsigned int **x2_gap)
{
  int    
    rateHet,
    states = tr->partitionData[model].states,
    pNumber, 
    qNumber;

  if(tr->rateHetModel == CAT)
    rateHet = 1;
  else
    rateHet = 4;
    

  if(tr->multiGene)
    {
      assert(tr->executeModel[model]);
      pNumber = tr->td[model].ti[0].pNumber;
      qNumber = tr->td[model].ti[0].qNumber;
    }
  else
    {
      pNumber = tr->td[0].ti[0].pNumber;
      qNumber = tr->td[0].ti[0].qNumber;
    }

  *x1_start = (double*)NULL,
  *x2_start = (double*)NULL;
  *tipX1 = (unsigned char*)NULL,
  *tipX2 = (unsigned char*)NULL;

  if(isTip(pNumber, tr->mxtips) || isTip(qNumber, tr->mxtips))
    {
      if(!( isTip(pNumber, tr->mxtips) && isTip(qNumber, tr->mxtips)) )
	{
	  *tipCase = TIP_INNER;
	  if(isTip(qNumber, tr->mxtips))
	    {
	      *tipX1 = tr->partitionData[model].yVector[qNumber];
	      *x2_start = tr->partitionData[model].xVector[pNumber - tr->mxtips - 1];
	      
	      if(tr->saveMemory)
		{
		  *x2_gap = &(tr->partitionData[model].gapVector[pNumber * tr->partitionData[model].gapVectorLength]);
		  *x2_gapColumn   = &tr->partitionData[model].gapColumn[(pNumber - tr->mxtips - 1) * states * rateHet];  
		}
	    }
	  else
	    {
	      *tipX1 = tr->partitionData[model].yVector[pNumber];
	      *x2_start = tr->partitionData[model].xVector[qNumber - tr->mxtips - 1];
	      
	      if(tr->saveMemory)
		{
		  *x2_gap = &(tr->partitionData[model].gapVector[qNumber * tr->partitionData[model].gapVectorLength]);
		  *x2_gapColumn   = &tr->partitionData[model].gapColumn[(qNumber - tr->mxtips - 1) * states * rateHet];
		}
	    }
	}
      else
	{
	  *tipCase = TIP_TIP;
	  *tipX1 = tr->partitionData[model].yVector[pNumber];
	  *tipX2 = tr->partitionData[model].yVector[qNumber];
	}
    }
  else
    {
      *tipCase = INNER_INNER;

      *x1_start = tr->partitionData[model].xVector[pNumber - tr->mxtips - 1];
      *x2_start = tr->partitionData[model].xVector[qNumber - tr->mxtips - 1];
      
      if(tr->saveMemory)
	{
	  *x1_gap = &(tr->partitionData[model].gapVector[pNumber * tr->partitionData[model].gapVectorLength]);
	  *x1_gapColumn   = &tr->partitionData[model].gapColumn[(pNumber - tr->mxtips - 1) * states * rateHet]; 
      
	  *x2_gap = &(tr->partitionData[model].gapVector[qNumber * tr->partitionData[model].gapVectorLength]);
	  *x2_gapColumn   = &tr->partitionData[model].gapColumn[(qNumber - tr->mxtips - 1) * states * rateHet]; 
	}
    }

}




void makenewzIterative(tree *tr)
{
  int 
    model, 
    tipCase;

  double
    *x1_start = (double*)NULL,
    *x2_start = (double*)NULL;

  
  unsigned char
    *tipX1,
    *tipX2;

  double
    *x1_gapColumn = (double*)NULL,
    *x2_gapColumn = (double*)NULL;
  
  unsigned int
    *x1_gap = (unsigned int*)NULL,
    *x2_gap = (unsigned int*)NULL;			      

 
  if(tr->multiGene)
    newviewIterativeMulti(tr);
  else
    newviewIterative(tr);

  for(model = 0; model < tr->NumberOfModels; model++)
    {
      if(tr->executeModel[model])
	{
	  int 
	    width = tr->partitionData[model].width,
	    states = tr->partitionData[model].states;

	  
	  getVects(tr, &tipX1, &tipX2, &x1_start, &x2_start, &tipCase, model, &x1_gapColumn, &x2_gapColumn, &x1_gap, &x2_gap);
	 

	  switch(tr->partitionData[model].dataType)
	    {
	    
	    case DNA_DATA:
	      if(tr->rateHetModel == CAT)
		{
		  if(tr->saveMemory)
		    sumCAT_SAVE(tipCase, tr->partitionData[model].sumBuffer, x1_start, x2_start, tr->partitionData[model].tipVector, tipX1, tipX2,
				width, x1_gapColumn, x2_gapColumn, x1_gap, x2_gap);
		  else
		    sumCAT(tipCase, tr->partitionData[model].sumBuffer, x1_start, x2_start, tr->partitionData[model].tipVector, tipX1, tipX2,
			   width);
		}
	      else
		{
		  if(tr->saveMemory)
		    sumGAMMA_GAPPED_SAVE(tipCase, tr->partitionData[model].sumBuffer, x1_start, x2_start, tr->partitionData[model].tipVector, tipX1, tipX2,
					 width, x1_gapColumn, x2_gapColumn, x1_gap, x2_gap);
		  else
		    sumGAMMA(tipCase, tr->partitionData[model].sumBuffer, x1_start, x2_start, tr->partitionData[model].tipVector, tipX1, tipX2,
			     width);
		}
	      break;		
	    case AA_DATA:
	      if(tr->rateHetModel == CAT)
		{
		  if(tr->saveMemory)
		    sumGTRCATPROT_SAVE(tipCase, tr->partitionData[model].sumBuffer, x1_start, x2_start, tr->partitionData[model].tipVector,
				       tipX1, tipX2, width, x1_gapColumn, x2_gapColumn, x1_gap, x2_gap);
		  else	      	      
		    sumGTRCATPROT(tipCase, tr->partitionData[model].sumBuffer, x1_start, x2_start, tr->partitionData[model].tipVector,
				  tipX1, tipX2, width);
		}
	      else
		{
		  if(tr->saveMemory)
		    sumGAMMAPROT_GAPPED_SAVE(tipCase, tr->partitionData[model].sumBuffer, x1_start, x2_start, tr->partitionData[model].tipVector, tipX1, tipX2,
					     width, x1_gapColumn, x2_gapColumn, x1_gap, x2_gap);
		  else
		    sumGAMMAPROT(tipCase, tr->partitionData[model].sumBuffer, x1_start, x2_start, tr->partitionData[model].tipVector,
				 tipX1, tipX2, width);
		}
	      break;		
	    default:
	      assert(0);
	    }
	}
    }
}





void execCore(tree *tr, volatile double *_dlnLdlz, volatile double *_d2lnLdlz2)
{
  int model, branchIndex;
  double lz;

  _dlnLdlz[0]   = 0.0;
  _d2lnLdlz2[0] = 0.0;

  for(model = 0; model < tr->NumberOfModels; model++)
    {
      if(tr->executeModel[model])
	{
	  int 
	    width = tr->partitionData[model].width,
	    states = tr->partitionData[model].states;
	  
	  double 
	    *sumBuffer       = (double*)NULL;
	 

	  volatile double
	    dlnLdlz   = 0.0,
	    d2lnLdlz2 = 0.0;
	  
	  
	  sumBuffer = tr->partitionData[model].sumBuffer;


	  if(tr->multiBranch)
	    {
	      branchIndex = model;
	      lz = tr->coreLZ[model];
	      _dlnLdlz[model]   = 0.0;
	      _d2lnLdlz2[model] = 0.0;
	    }
	  else
	    {
	      branchIndex = 0;
	      lz = tr->coreLZ[0];
	    }

	  switch(tr->partitionData[model].dataType)
	    {
	   
	    case DNA_DATA:
	      if(tr->rateHetModel == CAT)
		coreGTRCAT(width, tr->partitionData[model].numberOfCategories, sumBuffer,
			   &dlnLdlz, &d2lnLdlz2, tr->partitionData[model].wr, tr->partitionData[model].wr2,
			   tr->partitionData[model].perSiteRates, tr->partitionData[model].EIGN,  tr->partitionData[model].rateCategory, lz);
	      else 
		coreGTRGAMMA(width, sumBuffer,
			     &dlnLdlz, &d2lnLdlz2, tr->partitionData[model].EIGN, tr->partitionData[model].gammaRates, lz,
			     tr->partitionData[model].wgt);
		
	      break;		    
	    case AA_DATA:
	      if(tr->rateHetModel == CAT)
		coreGTRCATPROT(tr->partitionData[model].EIGN, lz, tr->partitionData[model].numberOfCategories,  tr->partitionData[model].perSiteRates,
			       tr->partitionData[model].rateCategory, width,
			       tr->partitionData[model].wr, tr->partitionData[model].wr2,
			       &dlnLdlz, &d2lnLdlz2,
			       sumBuffer);
	      else
		coreGTRGAMMAPROT(tr->partitionData[model].gammaRates, tr->partitionData[model].EIGN,
				 sumBuffer, width, tr->partitionData[model].wgt,
				 &dlnLdlz, &d2lnLdlz2, lz);
	      break;		   
	    default:
	      assert(0);
	    }

	  _dlnLdlz[branchIndex]   = _dlnLdlz[branchIndex]   + dlnLdlz;
	  _d2lnLdlz2[branchIndex] = _d2lnLdlz2[branchIndex] + d2lnLdlz2;
	}
    }

}




static void topLevelMakenewz(tree *tr, double *z0, int _maxiter, double *result)
{
  double   z[NUM_BRANCHES], zprev[NUM_BRANCHES], zstep[NUM_BRANCHES];
  volatile double  dlnLdlz[NUM_BRANCHES], d2lnLdlz2[NUM_BRANCHES];
  int i, maxiter[NUM_BRANCHES], numBranches, model;
  boolean firstIteration = TRUE;
  boolean outerConverged[NUM_BRANCHES];
  boolean loopConverged;



  if(tr->multiBranch)
    numBranches = tr->NumberOfModels;
  else
    numBranches = 1;



  for(i = 0; i < numBranches; i++)
    {
      z[i] = z0[i];
      maxiter[i] = _maxiter;
      outerConverged[i] = FALSE;
      tr->curvatOK[i]       = TRUE;
    }

  do
    {
      for(i = 0; i < numBranches; i++)
	{
	  if(outerConverged[i] == FALSE && tr->curvatOK[i] == TRUE)
	    {
	      tr->curvatOK[i] = FALSE;

	      zprev[i] = z[i];

	      zstep[i] = (1.0 - zmax) * z[i] + zmin;
	    }
	}

      for(i = 0; i < numBranches; i++)
	{
	  if(outerConverged[i] == FALSE && tr->curvatOK[i] == FALSE)
	    {
	      double lz;

	      if (z[i] < zmin) z[i] = zmin;
	      else if (z[i] > zmax) z[i] = zmax;
	      lz    = log(z[i]);

	      tr->coreLZ[i] = lz;
	    }
	}

      if(tr->multiBranch)
	{
	  for(model = 0; model < tr->NumberOfModels; model++)
	    {
	      if(tr->executeModel[model])
		tr->executeModel[model] = !tr->curvatOK[model];

	    }
	}
      else
	{
	  for(model = 0; model < tr->NumberOfModels; model++)
	    tr->executeModel[model] = !tr->curvatOK[0];
	}


#ifdef _USE_PTHREADS
      if(firstIteration)
	{
	  masterBarrier(THREAD_MAKENEWZ_FIRST, tr);
	  firstIteration = FALSE;
	}
      else
	masterBarrier(THREAD_MAKENEWZ, tr);

      if(!tr->multiBranch)
	{
	  dlnLdlz[0] = 0.0;
	  d2lnLdlz2[0] = 0.0;
	  for(i = 0; i < NumberOfThreads; i++)
	    {
	      dlnLdlz[0]   += reductionBuffer[i];
	      d2lnLdlz2[0] += reductionBufferTwo[i];
	    }
	}
      else
	{
	  int j;
	  for(j = 0; j < tr->NumberOfModels; j++)
	    {
	      dlnLdlz[j] = 0.0;
	      d2lnLdlz2[j] = 0.0;
	      for(i = 0; i < NumberOfThreads; i++)
		{
		  dlnLdlz[j]   += reductionBuffer[i * tr->NumberOfModels + j];
		  d2lnLdlz2[j] += reductionBufferTwo[i * tr->NumberOfModels + j];
		}
	    }
	}
#else
#ifdef _FINE_GRAIN_MPI
      if(firstIteration)
	{
	  masterBarrierMPI(THREAD_MAKENEWZ_FIRST, tr);
	  firstIteration = FALSE;
	}
      else
	masterBarrierMPI(THREAD_MAKENEWZ, tr);

      if(!tr->multiBranch)
	{
	  int 
	    model;	  	  
	  	
	  dlnLdlz[0]   = globalResult[0];
	  d2lnLdlz2[0] = globalResult[1];	  	  	 
	}
      else
	{
	  int
	    model;
	  
	  for(model = 0; model < tr->NumberOfModels; model++)
	    {	     	     
	      dlnLdlz[model]   = globalResult[model * 2 + 0];
	      d2lnLdlz2[model] = globalResult[model * 2 + 1];	    
	    }
	}

#else
      if(firstIteration)
	{
	  makenewzIterative(tr);
	  firstIteration = FALSE;
	}
      execCore(tr, dlnLdlz, d2lnLdlz2);
#endif
#endif
     

      for(i = 0; i < numBranches; i++)
	{
	  if(outerConverged[i] == FALSE && tr->curvatOK[i] == FALSE)
	    {
	      if ((d2lnLdlz2[i] >= 0.0) && (z[i] < zmax))
		zprev[i] = z[i] = 0.37 * z[i] + 0.63;  /*  Bad curvature, shorten branch */
	      else
		tr->curvatOK[i] = TRUE;
	    }
	}

      for(i = 0; i < numBranches; i++)
	{
	  if(tr->curvatOK[i] == TRUE && outerConverged[i] == FALSE)
	    {
	      if (d2lnLdlz2[i] < 0.0)
		{
		  double tantmp = -dlnLdlz[i] / d2lnLdlz2[i];
		  if (tantmp < 100)
		    {
		      z[i] *= EXP(tantmp);
		      if (z[i] < zmin)
			z[i] = zmin;

		      if (z[i] > 0.25 * zprev[i] + 0.75)
			z[i] = 0.25 * zprev[i] + 0.75;
		    }
		  else
		    z[i] = 0.25 * zprev[i] + 0.75;
		}
	      if (z[i] > zmax) z[i] = zmax;

	      maxiter[i] = maxiter[i] - 1;
	      if(maxiter[i] > 0 && (ABS(z[i] - zprev[i]) > zstep[i]))
		outerConverged[i] = FALSE;
	      else
		outerConverged[i] = TRUE;
	    }
	}

      loopConverged = TRUE;
      for(i = 0; i < numBranches; i++)
	loopConverged = loopConverged && outerConverged[i];
    }
  while (!loopConverged);


  for(model = 0; model < tr->NumberOfModels; model++)
    tr->executeModel[model] = TRUE;

  for(i = 0; i < numBranches; i++)
    result[i] = z[i];
}














void makenewzGeneric(tree *tr, nodeptr p, nodeptr q, double *z0, int maxiter, double *result, boolean mask)
{
  int i;
  boolean originalExecute[NUM_BRANCHES];

  if(tr->multiGene)
    {
      int sum = 0;

      for(i = 0; i < tr->numBranches; i++)      
	{           
	  if(mask)
	    {
	      if(tr->partitionConverged[i])
		tr->executeModel[i] = FALSE;
	      else
		tr->executeModel[i] = TRUE;		
	    }
	}

      assert(tr->numBranches == tr->NumberOfModels);      
      for(i = 0; i < tr->numBranches; i++)   
	sum += tr->executeModel[i];
      assert(sum == 1);
      assert(mask);
      
      for(i = 0; i < tr->NumberOfModels; i++)
	{	 
	  if(tr->executeModel[i])
	    {
	      tr->td[i].ti[0].pNumber = p->number;
	      tr->td[i].ti[0].qNumber = q->number;
	      tr->td[i].ti[0].qz[i] =  z0[i];
	      tr->td[i].count = 1;

	      if(!p->xs[i])
		computeTraversalInfoMulti(p, &(tr->td[i].ti[0]), &(tr->td[i].count), tr->mxtips, i); 
	      if(!q->xs[i])
		computeTraversalInfoMulti(q, &(tr->td[i].ti[0]), &(tr->td[i].count), tr->mxtips, i);	     	     
	    }	
	  else
	    tr->td[i].count = 0;
	}  
    }
  else
    {
      tr->td[0].ti[0].pNumber = p->number;
      tr->td[0].ti[0].qNumber = q->number;
      for(i = 0; i < tr->numBranches; i++)
	{
	  originalExecute[i] =  tr->executeModel[i];
	  tr->td[0].ti[0].qz[i] =  z0[i];
	  if(mask)
	    {
	      if(tr->partitionConverged[i])
		tr->executeModel[i] = FALSE;
	      else
		tr->executeModel[i] = TRUE;
	    }
	}
      
      tr->td[0].count = 1;
      
      if(!p->x)
	computeTraversalInfo(p, &(tr->td[0].ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);
      if(!q->x)
	computeTraversalInfo(q, &(tr->td[0].ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches);
    }


  
  topLevelMakenewz(tr, z0, maxiter, result);
 

  for(i = 0; i < tr->numBranches; i++)
      tr->executeModel[i] = TRUE;
}



void makenewzGenericDistance(tree *tr, int maxiter, double *z0, double *result, int taxon1, int taxon2)
{
  int i;

  assert(taxon1 != taxon2);
  assert(0 < taxon1 && taxon1 <= tr->mxtips);
  assert(0 < taxon2 && taxon2 <= tr->mxtips);

  tr->td[0].ti[0].pNumber = taxon1;
  tr->td[0].ti[0].qNumber = taxon2;
  tr->td[0].ti[0].tipCase = TIP_TIP;

  for(i = 0; i < tr->numBranches; i++)
    tr->td[0].ti[0].qz[i] =  defaultz;      /* TODO why defaultz ? */

  tr->td[0].count = 1;

  topLevelMakenewz(tr, z0, maxiter, result);
}



