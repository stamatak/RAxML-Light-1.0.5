/*  RAxML-VI-HPC (version 2.2) a program for sequential and parallel estimation of phylogenetic trees 
 *  Copyright August 2006 by Alexandros Stamatakis
 *
 *  Partially derived from
 *  fastDNAml, a program for estimation of phylogenetic trees from sequences by Gary J. Olsen
 *  
 *  and 
 *
 *  Programs of the PHYLIP package by Joe Felsenstein. 
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
 *  Alexandros Stamatakis:"RAxML-VI-HPC: maximum likelihood-based phylogenetic analyses with thousands of taxa and mixed models". 
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
extern volatile int NumberOfThreads;
#endif

extern const unsigned int mask32[32];





void calcDiagptable(double z, int data, int numberOfCategories, double *rptr, double *EIGN, double *diagptable)
{
  int i, l;
  double lz;

  if (z < zmin) 
    lz = log(zmin);
  else
    lz = log(z);

  switch(data)
    {     
    case BINARY_DATA:
      {
	double lz1;
	
	lz1 = EIGN[0] * lz;
	for(i = 0; i <  numberOfCategories; i++)
	  {		 
	    diagptable[2 * i] = 1.0;
	    diagptable[2 * i + 1] = EXP(rptr[i] * lz1);	   	    
	  }
      }
      break;
    case DNA_DATA:
      {
        double lz1, lz2, lz3;
        lz1 = EIGN[0] * lz;
        lz2 = EIGN[1] * lz;
        lz3 = EIGN[2] * lz;

        for(i = 0; i <  numberOfCategories; i++)
        {		 	    
          diagptable[4 * i] = 1.0;
          diagptable[4 * i + 1] = EXP(rptr[i] * lz1);
          diagptable[4 * i + 2] = EXP(rptr[i] * lz2);
          diagptable[4 * i + 3] = EXP(rptr[i] * lz3);	    
        }
      }
      break;
    case AA_DATA:
      {
        double lza[19];

        for(l = 0; l < 19; l++)      
          lza[l] = EIGN[l] * lz; 

        for(i = 0; i <  numberOfCategories; i++)
        {	      	       
          diagptable[i * 20] = 1.0;

          for(l = 1; l < 20; l++)
            diagptable[i * 20 + l] = EXP(rptr[i] * lza[l - 1]);     	          
        }
      }
      break;   
    default:
      assert(0);
  }
}



static double evaluateGTRGAMMAPROT_perSite (int *ex1, int *ex2, int *wptr,
    double *x1, double *x2,  					   
    unsigned char *tipX1, int n, 
    int *perSiteAA,
    siteAAModels *siteProtModel)
{
  double   
    sum = 0.0, term;        
  int     i, j, l;   
  double  *left, *right;              

  if(tipX1)
  {               
    for (i = 0; i < n; i++) 
    {
      double 
        *tipVector  = siteProtModel[perSiteAA[i]].tipVector,
        *diagptable = siteProtModel[perSiteAA[i]].left;

      __m128d tv = _mm_setzero_pd();
      left = &(tipVector[20 * tipX1[i]]);	  	  

      for(j = 0, term = 0.0; j < 4; j++)
      {
        double 
          *d = &diagptable[j * 20];

        right = &(x2[80 * i + 20 * j]);

        for(l = 0; l < 20; l+=2)
        {
          __m128d mul = _mm_mul_pd(_mm_load_pd(&left[l]), _mm_load_pd(&right[l]));
          tv = _mm_add_pd(tv, _mm_mul_pd(mul, _mm_load_pd(&d[l])));		   
        }		 		
      }
      tv = _mm_hadd_pd(tv, tv);
      _mm_storel_pd(&term, tv);

      term = LOG(0.25 * term);

      sum += wptr[i] * term;
    }    	        
  }              
  else
  {
    for (i = 0; i < n; i++) 
    {
      double 	   
        *diagptable = siteProtModel[perSiteAA[i]].left;	  	 	             

      __m128d tv = _mm_setzero_pd();	 	  	  

      for(j = 0, term = 0.0; j < 4; j++)
      {
        double 
          *d = &diagptable[j * 20];

        left  = &(x1[80 * i + 20 * j]);
        right = &(x2[80 * i + 20 * j]);

        for(l = 0; l < 20; l+=2)
        {
          __m128d mul = _mm_mul_pd(_mm_load_pd(&left[l]), _mm_load_pd(&right[l]));
          tv = _mm_add_pd(tv, _mm_mul_pd(mul, _mm_load_pd(&d[l])));		   
        }		 		
      }
      tv = _mm_hadd_pd(tv, tv);
      _mm_storel_pd(&term, tv);	  

      term = LOG(0.25 * term);

      sum += wptr[i] * term;
    }
  }

  return  sum;
}


static double evaluateGTRGAMMAPROT_GAPPED_SAVE (int *ex1, int *ex2, int *wptr,
    double *x1, double *x2,  
    double *tipVector, 
    unsigned char *tipX1, int n, double *diagptable, const boolean fastScaling,
    double *x1_gapColumn, double *x2_gapColumn, unsigned int *x1_gap, unsigned int *x2_gap)					   
{
  double   sum = 0.0, term;        
  int     i, j, l;   
  double  
    *left, 
    *right,
    *x1_ptr = x1,
    *x2_ptr = x2,
    *x1v,
    *x2v;              

  if(tipX1)
  {               
    for (i = 0; i < n; i++) 
    {
      if(x2_gap[i / 32] & mask32[i % 32])
        x2v = x2_gapColumn;
      else
      {
        x2v = x2_ptr;
        x2_ptr += 80;
      }

      __m128d tv = _mm_setzero_pd();
      left = &(tipVector[20 * tipX1[i]]);	  	  

      for(j = 0, term = 0.0; j < 4; j++)
      {
        double *d = &diagptable[j * 20];
        right = &(x2v[20 * j]);
        for(l = 0; l < 20; l+=2)
        {
          __m128d mul = _mm_mul_pd(_mm_load_pd(&left[l]), _mm_load_pd(&right[l]));
          tv = _mm_add_pd(tv, _mm_mul_pd(mul, _mm_load_pd(&d[l])));		   
        }		 		
      }

      tv = _mm_hadd_pd(tv, tv);
      _mm_storel_pd(&term, tv);


      if(fastScaling)
        term = LOG(0.25 * term);
      else
        term = LOG(0.25 * term) + (ex2[i] * LOG(minlikelihood));	   

      sum += wptr[i] * term;
    }    	        
  }              
  else
  {
    for (i = 0; i < n; i++) 
    {
      if(x1_gap[i / 32] & mask32[i % 32])
        x1v = x1_gapColumn;
      else
      {
        x1v = x1_ptr;
        x1_ptr += 80;
      }

      if(x2_gap[i / 32] & mask32[i % 32])
        x2v = x2_gapColumn;
      else
      {
        x2v = x2_ptr;
        x2_ptr += 80;
      }

      __m128d tv = _mm_setzero_pd();	 	  	  

      for(j = 0, term = 0.0; j < 4; j++)
      {
        double *d = &diagptable[j * 20];
        left  = &(x1v[20 * j]);
        right = &(x2v[20 * j]);

        for(l = 0; l < 20; l+=2)
        {
          __m128d mul = _mm_mul_pd(_mm_load_pd(&left[l]), _mm_load_pd(&right[l]));
          tv = _mm_add_pd(tv, _mm_mul_pd(mul, _mm_load_pd(&d[l])));		   
        }		 		
      }
      tv = _mm_hadd_pd(tv, tv);
      _mm_storel_pd(&term, tv);	  

      if(fastScaling)
        term = LOG(0.25 * term);
      else
        term = LOG(0.25 * term) + ((ex1[i] + ex2[i])*LOG(minlikelihood));

      sum += wptr[i] * term;
    }         
  }

  return  sum;
}



static double evaluateGTRGAMMAPROT (int *ex1, int *ex2, int *wptr,
    double *x1, double *x2,  
    double *tipVector, 
    unsigned char *tipX1, int n, double *diagptable, const boolean fastScaling)
{
  double   sum = 0.0, term;        
  int     i, j, l;   
  double  *left, *right;              

  if(tipX1)
  {               
    for (i = 0; i < n; i++) 
    {

      __m128d tv = _mm_setzero_pd();
      left = &(tipVector[20 * tipX1[i]]);	  	  

      for(j = 0, term = 0.0; j < 4; j++)
      {
        double *d = &diagptable[j * 20];
        right = &(x2[80 * i + 20 * j]);
        for(l = 0; l < 20; l+=2)
        {
          __m128d mul = _mm_mul_pd(_mm_load_pd(&left[l]), _mm_load_pd(&right[l]));
          tv = _mm_add_pd(tv, _mm_mul_pd(mul, _mm_load_pd(&d[l])));		   
        }		 		
      }
      tv = _mm_hadd_pd(tv, tv);
      _mm_storel_pd(&term, tv);


      if(fastScaling)
        term = LOG(0.25 * term);
      else
        term = LOG(0.25 * term) + (ex2[i] * LOG(minlikelihood));	   

      sum += wptr[i] * term;
    }    	        
  }              
  else
  {
    for (i = 0; i < n; i++) 
    {	  	 	             
      __m128d tv = _mm_setzero_pd();	 	  	  

      for(j = 0, term = 0.0; j < 4; j++)
      {
        double *d = &diagptable[j * 20];
        left  = &(x1[80 * i + 20 * j]);
        right = &(x2[80 * i + 20 * j]);

        for(l = 0; l < 20; l+=2)
        {
          __m128d mul = _mm_mul_pd(_mm_load_pd(&left[l]), _mm_load_pd(&right[l]));
          tv = _mm_add_pd(tv, _mm_mul_pd(mul, _mm_load_pd(&d[l])));		   
        }		 		
      }
      tv = _mm_hadd_pd(tv, tv);
      _mm_storel_pd(&term, tv);	  

      if(fastScaling)
        term = LOG(0.25 * term);
      else
        term = LOG(0.25 * term) + ((ex1[i] + ex2[i])*LOG(minlikelihood));

      sum += wptr[i] * term;
    }
  }

  return  sum;
}



static double evaluateGTRCATPROT (int *ex1, int *ex2, int *cptr, int *wptr,
    double *x1, double *x2, double *tipVector,
    unsigned char *tipX1, int n, double *diagptable_start, const boolean fastScaling)
{
  double   sum = 0.0, term;
  double  *diagptable,  *left, *right;
  int     i, l;                           

  if(tipX1)
  {                 
    for (i = 0; i < n; i++) 
    {	       	
      left = &(tipVector[20 * tipX1[i]]);
      right = &(x2[20 * i]);

      diagptable = &diagptable_start[20 * cptr[i]];	           	 
#ifdef __SIM_SSE3
      __m128d tv = _mm_setzero_pd();	    

      for(l = 0; l < 20; l+=2)
      {
        __m128d lv = _mm_load_pd(&left[l]);
        __m128d rv = _mm_load_pd(&right[l]);
        __m128d mul = _mm_mul_pd(lv, rv);
        __m128d dv = _mm_load_pd(&diagptable[l]);

        tv = _mm_add_pd(tv, _mm_mul_pd(mul, dv));		   
      }		 		

      tv = _mm_hadd_pd(tv, tv);
      _mm_storel_pd(&term, tv);
#else  
      for(l = 0, term = 0.0; l < 20; l++)
        term += left[l] * right[l] * diagptable[l];	 	  	  
#endif	    

      term = LOG(term);

      sum += wptr[i] * term;
    }      
  }    
  else
  {

    for (i = 0; i < n; i++) 
    {		       	      	      
      left  = &x1[20 * i];
      right = &x2[20 * i];

      diagptable = &diagptable_start[20 * cptr[i]];	  	
#ifdef __SIM_SSE3
      __m128d tv = _mm_setzero_pd();	    

      for(l = 0; l < 20; l+=2)
      {
        __m128d lv = _mm_load_pd(&left[l]);
        __m128d rv = _mm_load_pd(&right[l]);
        __m128d mul = _mm_mul_pd(lv, rv);
        __m128d dv = _mm_load_pd(&diagptable[l]);

        tv = _mm_add_pd(tv, _mm_mul_pd(mul, dv));		   
      }		 		

      tv = _mm_hadd_pd(tv, tv);
      _mm_storel_pd(&term, tv);
#else  
      for(l = 0, term = 0.0; l < 20; l++)
        term += left[l] * right[l] * diagptable[l];	
#endif

      term = LOG(term);	 

      sum += wptr[i] * term;      
    }
  }

  return  sum;         
} 

static inline boolean isGap(unsigned int *x, int pos)
{
  return (x[pos / 32] & mask32[pos % 32]);
}


static double evaluateGTRCATPROT_SAVE (int *ex1, int *ex2, int *cptr, int *wptr,
    double *x1, double *x2, double *tipVector,
    unsigned char *tipX1, int n, double *diagptable_start, const boolean fastScaling,
    double *x1_gapColumn, double *x2_gapColumn, unsigned int *x1_gap, unsigned int *x2_gap)
{
  double   
    sum = 0.0, 
  term,
  *diagptable,  
  *left, 
  *right,
  *left_ptr = x1,
  *right_ptr = x2;

  int     
    i, 
    l;                           

  if(tipX1)
  {                 
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

      diagptable = &diagptable_start[20 * cptr[i]];	           	 

      __m128d tv = _mm_setzero_pd();	    

      for(l = 0; l < 20; l+=2)
      {
        __m128d lv = _mm_load_pd(&left[l]);
        __m128d rv = _mm_load_pd(&right[l]);
        __m128d mul = _mm_mul_pd(lv, rv);
        __m128d dv = _mm_load_pd(&diagptable[l]);

        tv = _mm_add_pd(tv, _mm_mul_pd(mul, dv));		   
      }		 		

      tv = _mm_hadd_pd(tv, tv);
      _mm_storel_pd(&term, tv);


      term = LOG(term);

      sum += wptr[i] * term;
    }      
  }    
  else
  {

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

      diagptable = &diagptable_start[20 * cptr[i]];	  	

      __m128d tv = _mm_setzero_pd();	    

      for(l = 0; l < 20; l+=2)
      {
        __m128d lv = _mm_load_pd(&left[l]);
        __m128d rv = _mm_load_pd(&right[l]);
        __m128d mul = _mm_mul_pd(lv, rv);
        __m128d dv = _mm_load_pd(&diagptable[l]);

        tv = _mm_add_pd(tv, _mm_mul_pd(mul, dv));		   
      }		 		

      tv = _mm_hadd_pd(tv, tv);
      _mm_storel_pd(&term, tv);

      term = LOG(term);	 

      sum += wptr[i] * term;      
    }
  }

  return  sum;         
} 


static double evaluateGTRCAT_SAVE (int *ex1, int *ex2, int *cptr, int *wptr,
    double *x1_start, double *x2_start, double *tipVector, 		      
    unsigned char *tipX1, int n, double *diagptable_start, const boolean fastScaling,
    double *x1_gapColumn, double *x2_gapColumn, unsigned int *x1_gap, unsigned int *x2_gap)
{
  double  sum = 0.0, term;       
  int     i;

  double  *diagptable, 
          *x1, 
          *x2,
          *x1_ptr = x1_start,
          *x2_ptr = x2_start;

  if(tipX1)
  {           
    for (i = 0; i < n; i++) 
    {	
      double t[2] __attribute__ ((aligned (BYTE_ALIGNMENT)));
      __m128d x1v1, x1v2, x2v1, x2v2, dv1, dv2;

      x1 = &(tipVector[4 * tipX1[i]]);

      if(isGap(x2_gap, i))
        x2 = x2_gapColumn;
      else
      {
        x2 = x2_ptr;
        x2_ptr += 4;
      }

      diagptable = &diagptable_start[4 * cptr[i]];

      x1v1 =  _mm_load_pd(&x1[0]);
      x1v2 =  _mm_load_pd(&x1[2]);
      x2v1 =  _mm_load_pd(&x2[0]);
      x2v2 =  _mm_load_pd(&x2[2]);
      dv1  =  _mm_load_pd(&diagptable[0]);
      dv2  =  _mm_load_pd(&diagptable[2]);

      x1v1 = _mm_mul_pd(x1v1, x2v1);
      x1v1 = _mm_mul_pd(x1v1, dv1);

      x1v2 = _mm_mul_pd(x1v2, x2v2);
      x1v2 = _mm_mul_pd(x1v2, dv2);

      x1v1 = _mm_add_pd(x1v1, x1v2);

      _mm_store_pd(t, x1v1);

      term = LOG(t[0] + t[1]);



      sum += wptr[i] * term;
    }	
  }               
  else
  {
    for (i = 0; i < n; i++) 
    { 
      double t[2] __attribute__ ((aligned (BYTE_ALIGNMENT)));
      __m128d x1v1, x1v2, x2v1, x2v2, dv1, dv2;

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

      diagptable = &diagptable_start[4 * cptr[i]];	

      x1v1 =  _mm_load_pd(&x1[0]);
      x1v2 =  _mm_load_pd(&x1[2]);
      x2v1 =  _mm_load_pd(&x2[0]);
      x2v2 =  _mm_load_pd(&x2[2]);
      dv1  =  _mm_load_pd(&diagptable[0]);
      dv2  =  _mm_load_pd(&diagptable[2]);

      x1v1 = _mm_mul_pd(x1v1, x2v1);
      x1v1 = _mm_mul_pd(x1v1, dv1);

      x1v2 = _mm_mul_pd(x1v2, x2v2);
      x1v2 = _mm_mul_pd(x1v2, dv2);

      x1v1 = _mm_add_pd(x1v1, x1v2);

      _mm_store_pd(t, x1v1);


      term = LOG(t[0] + t[1]);

      sum += wptr[i] * term;
    }    
  }

  return  sum;         
} 


static double evaluateGTRGAMMA_GAPPED_SAVE(int *ex1, int *ex2, int *wptr,
    double *x1_start, double *x2_start, 
    double *tipVector, 
    unsigned char *tipX1, const int n, double *diagptable, const boolean fastScaling,
    double *x1_gapColumn, double *x2_gapColumn, unsigned int *x1_gap, unsigned int *x2_gap)
{
  double   sum = 0.0, term;    
  int     i, j;
  double  
    *x1, 
    *x2,
    *x1_ptr = x1_start,
    *x2_ptr = x2_start;



  if(tipX1)
  {        


    for (i = 0; i < n; i++)
    {
      double t[2] __attribute__ ((aligned (BYTE_ALIGNMENT)));
      __m128d termv, x1v, x2v, dv;

      x1 = &(tipVector[4 * tipX1[i]]);	 
      if(x2_gap[i / 32] & mask32[i % 32])
        x2 = x2_gapColumn;
      else
      {
        x2 = x2_ptr;	 
        x2_ptr += 16;
      }


      termv = _mm_set1_pd(0.0);	    	   

      for(j = 0; j < 4; j++)
      {
        x1v = _mm_load_pd(&x1[0]);
        x2v = _mm_load_pd(&x2[j * 4]);
        dv   = _mm_load_pd(&diagptable[j * 4]);

        x1v = _mm_mul_pd(x1v, x2v);
        x1v = _mm_mul_pd(x1v, dv);

        termv = _mm_add_pd(termv, x1v);

        x1v = _mm_load_pd(&x1[2]);
        x2v = _mm_load_pd(&x2[j * 4 + 2]);
        dv   = _mm_load_pd(&diagptable[j * 4 + 2]);

        x1v = _mm_mul_pd(x1v, x2v);
        x1v = _mm_mul_pd(x1v, dv);

        termv = _mm_add_pd(termv, x1v);
      }

      _mm_store_pd(t, termv);	  	 

      if(fastScaling)
        term = LOG(0.25 * (t[0] + t[1]));
      else
        term = LOG(0.25 * (t[0] + t[1])) + (ex2[i] * LOG(minlikelihood));	  

      sum += wptr[i] * term;
    }     
  }
  else
  {        

    for (i = 0; i < n; i++) 
    {

      double t[2] __attribute__ ((aligned (BYTE_ALIGNMENT)));
      __m128d termv, x1v, x2v, dv;

      if(x1_gap[i / 32] & mask32[i % 32])
        x1 = x1_gapColumn;
      else
      {
        x1 = x1_ptr; 	  	  
        x1_ptr += 16;
      }

      if(x2_gap[i / 32] & mask32[i % 32])
        x2 = x2_gapColumn;
      else
      {
        x2 = x2_ptr;
        x2_ptr += 16;
      }

      termv = _mm_set1_pd(0.0);	  	 

      for(j = 0; j < 4; j++)
      {
        x1v = _mm_load_pd(&x1[j * 4]);
        x2v = _mm_load_pd(&x2[j * 4]);
        dv   = _mm_load_pd(&diagptable[j * 4]);

        x1v = _mm_mul_pd(x1v, x2v);
        x1v = _mm_mul_pd(x1v, dv);

        termv = _mm_add_pd(termv, x1v);

        x1v = _mm_load_pd(&x1[j * 4 + 2]);
        x2v = _mm_load_pd(&x2[j * 4 + 2]);
        dv   = _mm_load_pd(&diagptable[j * 4 + 2]);

        x1v = _mm_mul_pd(x1v, x2v);
        x1v = _mm_mul_pd(x1v, dv);

        termv = _mm_add_pd(termv, x1v);
      }

      _mm_store_pd(t, termv);

      if(fastScaling)
        term = LOG(0.25 * (t[0] + t[1]));
      else
        term = LOG(0.25 * (t[0] + t[1])) + ((ex1[i] + ex2[i]) * LOG(minlikelihood));	  

      sum += wptr[i] * term;
    }                      	
  }

  return sum;
} 


static double evaluateGTRGAMMA(int *ex1, int *ex2, int *wptr,
    double *x1_start, double *x2_start, 
    double *tipVector, 
    unsigned char *tipX1, const int n, double *diagptable, const boolean fastScaling)
{
  double   sum = 0.0, term;    
  int     i, j;
#ifndef __SIM_SSE3  
  int k;
#endif
  double  *x1, *x2;             



  if(tipX1)
  {          	
    for (i = 0; i < n; i++)
    {
#ifdef __SIM_SSE3
      double t[2] __attribute__ ((aligned (BYTE_ALIGNMENT)));
      __m128d termv, x1v, x2v, dv;
#endif
      x1 = &(tipVector[4 * tipX1[i]]);	 
      x2 = &x2_start[16 * i];	 

#ifdef __SIM_SSE3	
      termv = _mm_set1_pd(0.0);	    	   

      for(j = 0; j < 4; j++)
      {
        x1v = _mm_load_pd(&x1[0]);
        x2v = _mm_load_pd(&x2[j * 4]);
        dv   = _mm_load_pd(&diagptable[j * 4]);

        x1v = _mm_mul_pd(x1v, x2v);
        x1v = _mm_mul_pd(x1v, dv);

        termv = _mm_add_pd(termv, x1v);

        x1v = _mm_load_pd(&x1[2]);
        x2v = _mm_load_pd(&x2[j * 4 + 2]);
        dv   = _mm_load_pd(&diagptable[j * 4 + 2]);

        x1v = _mm_mul_pd(x1v, x2v);
        x1v = _mm_mul_pd(x1v, dv);

        termv = _mm_add_pd(termv, x1v);
      }

      _mm_store_pd(t, termv);


      if(fastScaling)
        term = LOG(0.25 * (t[0] + t[1]));
      else
        term = LOG(0.25 * (t[0] + t[1])) + (ex2[i] * LOG(minlikelihood));	  
#else
      for(j = 0, term = 0.0; j < 4; j++)
        for(k = 0; k < 4; k++)
          term += x1[k] * x2[j * 4 + k] * diagptable[j * 4 + k];	          	  	  	    	    	  

      if(fastScaling)
        term = LOG(0.25 * term);
      else
        term = LOG(0.25 * term) + ex2[i] * LOG(minlikelihood);	 
#endif

      sum += wptr[i] * term;
    }     
  }
  else
  {        
    for (i = 0; i < n; i++) 
    {
#ifdef __SIM_SSE3
      double t[2] __attribute__ ((aligned (BYTE_ALIGNMENT)));
      __m128d termv, x1v, x2v, dv;
#endif

      x1 = &x1_start[16 * i];
      x2 = &x2_start[16 * i];	  	  

#ifdef __SIM_SSE3	
      termv = _mm_set1_pd(0.0);	  	 

      for(j = 0; j < 4; j++)
      {
        x1v = _mm_load_pd(&x1[j * 4]);
        x2v = _mm_load_pd(&x2[j * 4]);
        dv   = _mm_load_pd(&diagptable[j * 4]);

        x1v = _mm_mul_pd(x1v, x2v);
        x1v = _mm_mul_pd(x1v, dv);

        termv = _mm_add_pd(termv, x1v);

        x1v = _mm_load_pd(&x1[j * 4 + 2]);
        x2v = _mm_load_pd(&x2[j * 4 + 2]);
        dv   = _mm_load_pd(&diagptable[j * 4 + 2]);

        x1v = _mm_mul_pd(x1v, x2v);
        x1v = _mm_mul_pd(x1v, dv);

        termv = _mm_add_pd(termv, x1v);
      }

      _mm_store_pd(t, termv);

      if(fastScaling)
        term = LOG(0.25 * (t[0] + t[1]));
      else
        term = LOG(0.25 * (t[0] + t[1])) + ((ex1[i] + ex2[i]) * LOG(minlikelihood));	  
#else 
      for(j = 0, term = 0.0; j < 4; j++)
        for(k = 0; k < 4; k++)
          term += x1[j * 4 + k] * x2[j * 4 + k] * diagptable[j * 4 + k];

      if(fastScaling)
        term = LOG(0.25 * term);
      else
        term = LOG(0.25 * term) + (ex1[i] + ex2[i]) * LOG(minlikelihood);
#endif

      sum += wptr[i] * term;
    }                      	
  }

  return sum;
} 


static double evaluateGTRCAT (int *ex1, int *ex2, int *cptr, int *wptr,
    double *x1_start, double *x2_start, double *tipVector, 		      
    unsigned char *tipX1, int n, double *diagptable_start, const boolean fastScaling)
{
  double  sum = 0.0, term;       
  int     i;
#ifndef __SIM_SSE3
  int j;  
#endif
  double  *diagptable, *x1, *x2;                      	    

  if(tipX1)
  {           
    for (i = 0; i < n; i++) 
    {	
#ifdef __SIM_SSE3
      double t[2] __attribute__ ((aligned (BYTE_ALIGNMENT)));
      __m128d x1v1, x1v2, x2v1, x2v2, dv1, dv2;
#endif
      x1 = &(tipVector[4 * tipX1[i]]);
      x2 = &x2_start[4 * i];

      diagptable = &diagptable_start[4 * cptr[i]];

#ifdef __SIM_SSE3	    	  
      x1v1 =  _mm_load_pd(&x1[0]);
      x1v2 =  _mm_load_pd(&x1[2]);
      x2v1 =  _mm_load_pd(&x2[0]);
      x2v2 =  _mm_load_pd(&x2[2]);
      dv1  =  _mm_load_pd(&diagptable[0]);
      dv2  =  _mm_load_pd(&diagptable[2]);

      x1v1 = _mm_mul_pd(x1v1, x2v1);
      x1v1 = _mm_mul_pd(x1v1, dv1);

      x1v2 = _mm_mul_pd(x1v2, x2v2);
      x1v2 = _mm_mul_pd(x1v2, dv2);

      x1v1 = _mm_add_pd(x1v1, x1v2);

      _mm_store_pd(t, x1v1);


      term = LOG(t[0] + t[1]);

#else
      for(j = 0, term = 0.0; j < 4; j++)
        term += x1[j] * x2[j] * diagptable[j];


      term = LOG(term);

#endif	    
      sum += wptr[i] * term;
    }	
  }               
  else
  {
    for (i = 0; i < n; i++) 
    { 
#ifdef __SIM_SSE3
      double t[2] __attribute__ ((aligned (BYTE_ALIGNMENT)));
      __m128d x1v1, x1v2, x2v1, x2v2, dv1, dv2;
#endif
      x1 = &x1_start[4 * i];
      x2 = &x2_start[4 * i];

      diagptable = &diagptable_start[4 * cptr[i]];	

#ifdef __SIM_SSE3	  
      x1v1 =  _mm_load_pd(&x1[0]);
      x1v2 =  _mm_load_pd(&x1[2]);
      x2v1 =  _mm_load_pd(&x2[0]);
      x2v2 =  _mm_load_pd(&x2[2]);
      dv1  =  _mm_load_pd(&diagptable[0]);
      dv2  =  _mm_load_pd(&diagptable[2]);

      x1v1 = _mm_mul_pd(x1v1, x2v1);
      x1v1 = _mm_mul_pd(x1v1, dv1);

      x1v2 = _mm_mul_pd(x1v2, x2v2);
      x1v2 = _mm_mul_pd(x1v2, dv2);

      x1v1 = _mm_add_pd(x1v1, x1v2);

      _mm_store_pd(t, x1v1);


      term = LOG(t[0] + t[1]);

#else

      for(j = 0, term = 0.0; j < 4; j++)
        term += x1[j] * x2[j] * diagptable[j];     


      term = LOG(term);

#endif
      sum += wptr[i] * term;
    }    
  }

  return  sum;         
} 

static double evaluateGTRCAT_BINARY (int *ex1, int *ex2, int *cptr, int *wptr,
				     double *x1_start, double *x2_start, double *tipVector, 		      
				     unsigned char *tipX1, int n, double *diagptable_start)
{
  double  
    sum = 0.0, 
    term, 
    *diagptable, 
    *x1, 
    *x2; 
  
  int     
    i;
 
  if(tipX1)
    {          
      for (i = 0; i < n; i++) 
	{
	  double 
	    t[2] __attribute__ ((aligned (16)));	    		   	  

	  x1 = &(tipVector[2 * tipX1[i]]);
	  x2 = &(x2_start[2 * i]);
	  
	  diagptable = &(diagptable_start[2 * cptr[i]]);	    	    	  
	
	  _mm_store_pd(t, _mm_mul_pd(_mm_load_pd(x1), _mm_mul_pd(_mm_load_pd(x2), _mm_load_pd(diagptable))));
	  	  
	  term = LOG(t[0] + t[1]);     

	  sum += wptr[i] * term;
	}	
    }               
  else
    {
      for (i = 0; i < n; i++) 
	{	
	  double 
	    t[2] __attribute__ ((aligned (16)));	    		   	  
	          	
	  x1 = &x1_start[2 * i];
	  x2 = &x2_start[2 * i];
	  
	  diagptable = &diagptable_start[2 * cptr[i]];		  
	  
	  _mm_store_pd(t, _mm_mul_pd(_mm_load_pd(x1), _mm_mul_pd(_mm_load_pd(x2), _mm_load_pd(diagptable))));
	  	  
	  term = LOG(t[0] + t[1]);	 	     
	  
	  sum += wptr[i] * term;
	}	   
    }
       
  return  sum;         
} 

static double evaluateGTRGAMMA_BINARY(int *ex1, int *ex2, int *wptr,
				      double *x1_start, double *x2_start, 
				      double *tipVector, 
				      unsigned char *tipX1, const int n, double *diagptable)
{
  double   sum = 0.0, term;    
  int     i, j;

  double  *x1, *x2;             

  if(tipX1)
    {          
      for (i = 0; i < n; i++)
	{

	  double t[2] __attribute__ ((aligned (16)));
	  __m128d termv, x1v, x2v, dv;

	  x1 = &(tipVector[2 * tipX1[i]]);	 
	  x2 = &x2_start[8 * i];	          	  	
	
	  termv = _mm_set1_pd(0.0);	    	   
	  
	  for(j = 0; j < 4; j++)
	    {
	      x1v = _mm_load_pd(&x1[0]);
	      x2v = _mm_load_pd(&x2[j * 2]);
	      dv   = _mm_load_pd(&diagptable[j * 2]);
	      
	      x1v = _mm_mul_pd(x1v, x2v);
	      x1v = _mm_mul_pd(x1v, dv);
	      
	      termv = _mm_add_pd(termv, x1v);	      	      
	    }
	  
	  _mm_store_pd(t, termv);	        
	  	 
	  term = LOG(0.25 * (t[0] + t[1]));	  

	  
	  sum += wptr[i] * term;
	}	  
    }
  else
    {         
      for (i = 0; i < n; i++) 
	{

	  double t[2] __attribute__ ((aligned (16)));
	  __m128d termv, x1v, x2v, dv;
      	  	 	  	  
	  x1 = &x1_start[8 * i];
	  x2 = &x2_start[8 * i];
	  	  	
	  termv = _mm_set1_pd(0.0);	    	   
	  
	  for(j = 0; j < 4; j++)
	    {
	      x1v = _mm_load_pd(&x1[j * 2]);
	      x2v = _mm_load_pd(&x2[j * 2]);
	      dv   = _mm_load_pd(&diagptable[j * 2]);
	      
	      x1v = _mm_mul_pd(x1v, x2v);
	      x1v = _mm_mul_pd(x1v, dv);
	      
	      termv = _mm_add_pd(termv, x1v);	      	      
	    }
	  
	  _mm_store_pd(t, termv);	  	  
	 
	  term = LOG(0.25 * (t[0] + t[1]));	 	  

	  sum += wptr[i] * term;
	}                      	
    }

  return sum;
} 


double evaluateIterative(tree *tr,  boolean writeVector)
{
  double 
    *pz = tr->td[0].ti[0].qz,
    result = 0.0;   

  int 
    rateHet = tr->discreteRateCategories,
            pNumber = tr->td[0].ti[0].pNumber, 
            qNumber = tr->td[0].ti[0].qNumber, 
            model,
            /* recom */
            slot = -1;
  /* E recom */


  newviewIterative(tr);  

  for(model = 0; model < tr->NumberOfModels; model++)
  {    
    int 	    
      width = tr->partitionData[model].width;

    if(tr->executeModel[model] && width > 0)
    {	
      int 
        rateHet,	  
        states = tr->partitionData[model].states;

      double 
        z, 
        partitionLikelihood = 0.0, 
        *_vector = (double*)NULL;;

      int    
        *ex1 = (int*)NULL, 
        *ex2 = (int*)NULL;

      unsigned int
        *x1_gap = (unsigned int*)NULL,
        *x2_gap = (unsigned int*)NULL;

      double 
        *x1_start   = (double*)NULL, 
        *x2_start   = (double*)NULL,
        *diagptable = (double*)NULL,  
        *x1_gapColumn = (double*)NULL,
        *x2_gapColumn = (double*)NULL;

      unsigned char 
        *tip = (unsigned char*)NULL;


      if(tr->rateHetModel == CAT)
        rateHet = 1;
      else
        rateHet = 4;

      diagptable = tr->partitionData[model].left;

      if(isTip(pNumber, tr->mxtips) || isTip(qNumber, tr->mxtips))
      {	        	    
        if(isTip(qNumber, tr->mxtips))
        {			  		 
          /* recom */
          if(tr->useRecom)
          {
            slot = tr->td[0].ti[0].slot_p;
            x2_start = tr->partitionData[model].xVector[slot];		 		
          }
          /* E recom */
          else	      	      
            x2_start = tr->partitionData[model].xVector[pNumber - tr->mxtips -1];		  

          tip      = tr->partitionData[model].yVector[qNumber];	 

          if(tr->saveMemory)
          {
            x2_gap         = &(tr->partitionData[model].gapVector[pNumber * tr->partitionData[model].gapVectorLength]);
            x2_gapColumn   = &(tr->partitionData[model].gapColumn[(pNumber - tr->mxtips - 1) * states * rateHet]);
          }
        }           
        else
        {		 
          /* recom */
          if(tr->useRecom)
          {
            slot = tr->td[0].ti[0].slot_q;
            x2_start = tr->partitionData[model].xVector[slot];		  		 
          }
          /* E recom */
          else         	      
            x2_start = tr->partitionData[model].xVector[qNumber - tr->mxtips - 1];		  		  

          tip = tr->partitionData[model].yVector[pNumber];

          if(tr->saveMemory)
          {
            x2_gap         = &(tr->partitionData[model].gapVector[qNumber * tr->partitionData[model].gapVectorLength]);
            x2_gapColumn   = &(tr->partitionData[model].gapColumn[(qNumber - tr->mxtips - 1) * states * rateHet]);
          }	      
        }
      }
      else
      {  
        /* recom */
        if(tr->useRecom)
        {
          slot = tr->td[0].ti[0].slot_p;
          x1_start = tr->partitionData[model].xVector[slot];

          slot = tr->td[0].ti[0].slot_q;
          x2_start = tr->partitionData[model].xVector[slot];	     	    
        }
        /* E recom */
        else	  
        {
          x1_start = tr->partitionData[model].xVector[pNumber - tr->mxtips - 1];
          x2_start = tr->partitionData[model].xVector[qNumber - tr->mxtips - 1];
        }

        if(tr->saveMemory)
        {
          x1_gap = &(tr->partitionData[model].gapVector[pNumber * tr->partitionData[model].gapVectorLength]);
          x2_gap = &(tr->partitionData[model].gapVector[qNumber * tr->partitionData[model].gapVectorLength]);
          x1_gapColumn   = &tr->partitionData[model].gapColumn[(pNumber - tr->mxtips - 1) * states * rateHet];
          x2_gapColumn   = &tr->partitionData[model].gapColumn[(qNumber - tr->mxtips - 1) * states * rateHet];
        }	  
      }

      if(tr->multiBranch)
        z = pz[model];
      else
        z = pz[0];


      switch(tr->partitionData[model].dataType)
      { 
      case BINARY_DATA:
	if(tr->rateHetModel == CAT)
          {
            calcDiagptable(z, BINARY_DATA, tr->partitionData[model].numberOfCategories, tr->partitionData[model].perSiteRates, tr->partitionData[model].EIGN, diagptable);

	    /*f(tr->saveMemory)
	      assert(0);
	      else*/
	    partitionLikelihood =  evaluateGTRCAT_BINARY(ex1, ex2, tr->partitionData[model].rateCategory, tr->partitionData[model].wgt,
							 x1_start, x2_start, tr->partitionData[model].tipVector, 
							 tip, width, diagptable);
	  }
	else
	  {
	    calcDiagptable(z, BINARY_DATA, 4, tr->partitionData[model].gammaRates, tr->partitionData[model].EIGN, diagptable);

            /*(tr->saveMemory)
	      assert(0);
	      else*/
	    partitionLikelihood = evaluateGTRGAMMA_BINARY(ex1, ex2, tr->partitionData[model].wgt,
							  x1_start, x2_start, tr->partitionData[model].tipVector,
							  tip, width, diagptable); 
	  }
	break;
        case DNA_DATA:
          if(tr->rateHetModel == CAT)
          {
            calcDiagptable(z, DNA_DATA, tr->partitionData[model].numberOfCategories, 
                tr->partitionData[model].perSiteRates, tr->partitionData[model].EIGN, diagptable);

            if(tr->saveMemory)
              partitionLikelihood =  evaluateGTRCAT_SAVE(ex1, ex2, tr->partitionData[model].rateCategory, tr->partitionData[model].wgt,
                  x1_start, x2_start, tr->partitionData[model].tipVector, 
                  tip, width, diagptable, TRUE, x1_gapColumn, x2_gapColumn, x1_gap, x2_gap);
            else
              partitionLikelihood =  evaluateGTRCAT(ex1, ex2, tr->partitionData[model].rateCategory, tr->partitionData[model].wgt,
                  x1_start, x2_start, tr->partitionData[model].tipVector, 
                  tip, width, diagptable, TRUE);
          }
          else
          {
            calcDiagptable(z, DNA_DATA, 4, tr->partitionData[model].gammaRates, tr->partitionData[model].EIGN, diagptable);

            if(tr->saveMemory)		   
              partitionLikelihood =  evaluateGTRGAMMA_GAPPED_SAVE(ex1, ex2, tr->partitionData[model].wgt,
                  x1_start, x2_start, tr->partitionData[model].tipVector,
                  tip, width, diagptable, TRUE,
                  x1_gapColumn, x2_gapColumn, x1_gap, x2_gap);		    
            else
              partitionLikelihood =  evaluateGTRGAMMA(ex1, ex2, tr->partitionData[model].wgt,
                  x1_start, x2_start, tr->partitionData[model].tipVector,
                  tip, width, diagptable, TRUE); 


          }
          break;	  	   		   
        case AA_DATA:	
          if(tr->rateHetModel == CAT)
          {
            calcDiagptable(z, AA_DATA, tr->partitionData[model].numberOfCategories, 
                tr->partitionData[model].perSiteRates, tr->partitionData[model].EIGN, diagptable);

            if(tr->saveMemory)
              partitionLikelihood = evaluateGTRCATPROT_SAVE(ex1, ex2, tr->partitionData[model].rateCategory, tr->partitionData[model].wgt,
                  x1_start, x2_start, tr->partitionData[model].tipVector,
                  tip, width, diagptable, TRUE,  x1_gapColumn, x2_gapColumn, x1_gap, x2_gap);
            else
              partitionLikelihood = evaluateGTRCATPROT(ex1, ex2, tr->partitionData[model].rateCategory, tr->partitionData[model].wgt,
                  x1_start, x2_start, tr->partitionData[model].tipVector,
                  tip, width, diagptable, TRUE);		  
          }
          else
          {
            if(tr->estimatePerSiteAA)
            { 
              int 
                p;

              for(p = 0; p < (NUM_PROT_MODELS - 2); p++)			    
                calcDiagptable(z, AA_DATA, 4, tr->partitionData[model].gammaRates, tr->siteProtModel[p].EIGN, tr->siteProtModel[p].left);

              partitionLikelihood = evaluateGTRGAMMAPROT_perSite(ex1, 
                  ex2,
                  tr->partitionData[model].wgt,
                  x1_start, 
                  x2_start,
                  tip,
                  width, 
                  tr->partitionData[model].perSiteAAModel,
                  tr->siteProtModel);					   

            }
            else
            {
              calcDiagptable(z, AA_DATA, 4, tr->partitionData[model].gammaRates, tr->partitionData[model].EIGN, diagptable);

              if(tr->saveMemory)
                partitionLikelihood = evaluateGTRGAMMAPROT_GAPPED_SAVE(ex1, ex2, tr->partitionData[model].wgt,
                    x1_start, x2_start, tr->partitionData[model].tipVector,
                    tip, width, diagptable, TRUE,
                    x1_gapColumn, x2_gapColumn, x1_gap, x2_gap);

              else
                partitionLikelihood = evaluateGTRGAMMAPROT(ex1, ex2, tr->partitionData[model].wgt,
                    x1_start, x2_start, tr->partitionData[model].tipVector,
                    tip, width, diagptable, TRUE);
            }	      
          }
          break;	      		    
        default:
          assert(0);	    
      }	

      if(width > 0)
      {
        assert(partitionLikelihood < 0.0);	  
        partitionLikelihood += (tr->partitionData[model].globalScaler[pNumber] + tr->partitionData[model].globalScaler[qNumber]) * LOG(minlikelihood);
      }		

      result += partitionLikelihood;	  
      tr->perPartitionLH[model] = partitionLikelihood; 	  
    }
    else
    {
      if(width == 0)	    
        tr->perPartitionLH[model] = 0.0;	   
    }
  }

  return result;
}




double evaluateGeneric (tree *tr, nodeptr p)
{
  volatile double result;
  nodeptr q = p->back; 
  int i;


  {
    tr->td[0].ti[0].pNumber = p->number;
    tr->td[0].ti[0].qNumber = q->number;          

    for(i = 0; i < tr->numBranches; i++)    
      tr->td[0].ti[0].qz[i] =  q->z[i];

    /* recom */
    if(tr->useRecom)
    {
      int 
        slot = -1,
             count = 0;

      computeTraversalInfoStlen(p, tr->mxtips, tr->rvec, &count);
      computeTraversalInfoStlen(q, tr->mxtips, tr->rvec, &count);


      if(!isTip(q->number, tr->mxtips))
      {
        getxVector(tr->rvec, q->number, &slot, tr->mxtips);
        tr->td[0].ti[0].slot_q = slot;
      }
      if(!isTip(p->number, tr->mxtips))
      {
        getxVector(tr->rvec, p->number, &slot, tr->mxtips);
        tr->td[0].ti[0].slot_p = slot;
      }
    }
    /* E recom */
    tr->td[0].count = 1;
    if(needsRecomp(tr->useRecom, tr->rvec, p, tr->mxtips))
      computeTraversalInfo(p, &(tr->td[0].ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches, tr->rvec, tr->useRecom);
    if(needsRecomp(tr->useRecom, tr->rvec, q, tr->mxtips))
      computeTraversalInfo(q, &(tr->td[0].ti[0]), &(tr->td[0].count), tr->mxtips, tr->numBranches, tr->rvec, tr->useRecom);  

#ifdef _USE_PTHREADS 
    {
      int j;

      masterBarrier(THREAD_EVALUATE, tr); 
      if(tr->NumberOfModels == 1)
      {
        for(i = 0, result = 0.0; i < NumberOfThreads; i++)          
          result += reductionBuffer[i];  	  	     

        tr->perPartitionLH[0] = result;
      }
      else
      {
        volatile double partitionResult;

        result = 0.0;

        for(j = 0; j < tr->NumberOfModels; j++)
        {
          for(i = 0, partitionResult = 0.0; i < NumberOfThreads; i++)          	      
            partitionResult += reductionBuffer[i * tr->NumberOfModels + j];
          result += partitionResult;
          tr->perPartitionLH[j] = partitionResult;
        }
      }
    }  
#else
#ifdef _FINE_GRAIN_MPI
    masterBarrierMPI(THREAD_EVALUATE, tr);

    {
      int model = 0;

      for(model = 0, result = 0.0; model < tr->NumberOfModels; model++)
        result += tr->perPartitionLH[model];		  
    }
#else
    result = evaluateIterative(tr, FALSE);
#endif   
#endif
  }

  /* TODOFER if you need to unpin the root, do it here */
  if(tr->useRecom)
  {
    unpinNode(tr->rvec, p->number, tr->mxtips);
    unpinNode(tr->rvec, q->number, tr->mxtips);
  }

  tr->likelihood = result;    

  return result;
}





double evaluateGenericInitrav (tree *tr, nodeptr p)
{
  volatile double result;   


  {
    determineFullTraversal(p, tr);

#ifdef _USE_PTHREADS 
    {
      int i, j;

      masterBarrier(THREAD_EVALUATE, tr);    

      if(tr->NumberOfModels == 1)
      {
        for(i = 0, result = 0.0; i < NumberOfThreads; i++)          
          result += reductionBuffer[i];  	  	     

        tr->perPartitionLH[0] = result;
      }
      else
      {
        volatile double partitionResult;

        result = 0.0;

        for(j = 0; j < tr->NumberOfModels; j++)
        {
          for(i = 0, partitionResult = 0.0; i < NumberOfThreads; i++)          	      
            partitionResult += reductionBuffer[i * tr->NumberOfModels + j];
          result +=  partitionResult;
          tr->perPartitionLH[j] = partitionResult;
        }
      }

    }
#else
#ifdef _FINE_GRAIN_MPI
    masterBarrierMPI(THREAD_EVALUATE, tr);
    {
      int model = 0;

      for(model = 0, result = 0.0; model < tr->NumberOfModels; model++)
        result += tr->perPartitionLH[model];	

    }     
#else
    result = evaluateIterative(tr, FALSE);
#endif
#endif

  }


  tr->likelihood = result;         

  return result;
}


void onlyInitrav(tree *tr, nodeptr p)
{   

  {
    determineFullTraversal(p, tr);  

#ifdef _USE_PTHREADS  
    masterBarrier(THREAD_NEWVIEW, tr);  	 
#else
#ifdef _FINE_GRAIN_MPI
    masterBarrierMPI(THREAD_NEWVIEW, tr);
#else
    newviewIterative(tr);   
#endif   
#endif
  }
}








