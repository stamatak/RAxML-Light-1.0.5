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
 *  Alexandros Stamatakis:"RAxML-VI-HPC: maximum likelihood-based phylogenetic analyses with thousands of taxa and mixed models". 
 *  Bioinformatics 2006; doi: 10.1093/bioinformatics/btl446
 */

#ifndef WIN32
#include <sys/times.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h> 
#endif

#include <math.h>
#include <time.h> 
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>



#include "axml.h"

extern int Thorough;
extern int optimizeRateCategoryInvocations;
extern infoList iList;
extern char seq_file[1024];
extern char resultFileName[1024];
extern char tree_file[1024];
extern char run_id[128];
extern FILE *INFILE;
extern double masterTime;
extern double accumulatedTime;

extern checkPointState ckp;
extern partitionLengths pLengths[MAX_MODEL];
extern char binaryCheckpointName[1024];
extern char binaryCheckpointInputName[1024];

boolean initrav (tree *tr, nodeptr p)
{ 
  nodeptr  q;

  if (!isTip(p->number, tr->rdta->numsp)) 
  {      
    q = p->next;

    do 
    {	   
      if (! initrav(tr, q->back))  return FALSE;		   
      q = q->next;	
    } 
    while (q != p);  

    newviewGeneric(tr, p);
  }

  return TRUE;
} 












boolean update(tree *tr, nodeptr p)
{       
  nodeptr  q; 
  boolean smoothedPartitions[NUM_BRANCHES];
  int i;
  double   z[NUM_BRANCHES], z0[NUM_BRANCHES];
  double _deltaz;

  q = p->back;   

  for(i = 0; i < tr->numBranches; i++)
    z0[i] = q->z[i];    

  if(tr->numBranches > 1)
    makenewzGeneric(tr, p, q, z0, newzpercycle, z, TRUE);  
  else
    makenewzGeneric(tr, p, q, z0, newzpercycle, z, FALSE);

  for(i = 0; i < tr->numBranches; i++)    
    smoothedPartitions[i]  = tr->partitionSmoothed[i];

  for(i = 0; i < tr->numBranches; i++)
  {         
    if(!tr->partitionConverged[i])
    {	  
      _deltaz = deltaz;

      if(ABS(z[i] - z0[i]) > _deltaz)  
      {	      
        smoothedPartitions[i] = FALSE;       
      }	 



      p->z[i] = q->z[i] = z[i];	 
    }
  }

  for(i = 0; i < tr->numBranches; i++)    
    tr->partitionSmoothed[i]  = smoothedPartitions[i];

  return TRUE;
}




boolean smooth (tree *tr, nodeptr p)
{
  nodeptr  q;

  if (! update(tr, p))               return FALSE; /*  Adjust branch */
  if (! isTip(p->number, tr->rdta->numsp)) 
  {                                  /*  Adjust descendants */
    q = p->next;
    while (q != p) 
    {
      if (! smooth(tr, q->back))   return FALSE;
      q = q->next;
    }	

    if(tr->multiBranch && !tr->useRecom)		  
      newviewGenericMasked(tr, p);	
    else
      newviewGeneric(tr, p);     
  }

  return TRUE;
} 

static boolean allSmoothed(tree *tr)
{
  int i;
  boolean result = TRUE;

  for(i = 0; i < tr->numBranches; i++)
  {
    if(tr->partitionSmoothed[i] == FALSE)
      result = FALSE;
    else
      tr->partitionConverged[i] = TRUE;
  }

  return result;
}



boolean smoothTree (tree *tr, int maxtimes)
{
  nodeptr  p, q;   
  int i, count = 0;

  p = tr->start;
  for(i = 0; i < tr->numBranches; i++)
    tr->partitionConverged[i] = FALSE;

  while (--maxtimes >= 0) 
  {    
    for(i = 0; i < tr->numBranches; i++)	
      tr->partitionSmoothed[i] = TRUE;		

    if (! smooth(tr, p->back))       return FALSE;
    if (!isTip(p->number, tr->rdta->numsp)) 
    {
      q = p->next;
      while (q != p) 
      {
        if (! smooth(tr, q->back))   return FALSE;
        q = q->next;
      }
    }

    count++;

    if (allSmoothed(tr)) 
      break;      
  }

  for(i = 0; i < tr->numBranches; i++)
    tr->partitionConverged[i] = FALSE;



  return TRUE;
} 



boolean localSmooth (tree *tr, nodeptr p, int maxtimes)
{ 
  nodeptr  q;
  int i;

  if (isTip(p->number, tr->rdta->numsp)) return FALSE;

  for(i = 0; i < tr->numBranches; i++)	
    tr->partitionConverged[i] = FALSE;	

  while (--maxtimes >= 0) 
  {     
    for(i = 0; i < tr->numBranches; i++)	
      tr->partitionSmoothed[i] = TRUE;

    q = p;
    do 
    {
      if (! update(tr, q)) return FALSE;
      q = q->next;
    } 
    while (q != p);

    if (allSmoothed(tr)) 
      break;
  }

  for(i = 0; i < tr->numBranches; i++)
  {
    tr->partitionSmoothed[i] = FALSE; 
    tr->partitionConverged[i] = FALSE;
  }

  return TRUE;
}





static void resetInfoList(void)
{
  int i;

  iList.valid = 0;

  for(i = 0; i < iList.n; i++)    
  {
    iList.list[i].node = (nodeptr)NULL;
    iList.list[i].likelihood = unlikely;
  }    
}

void initInfoList(int n)
{
  int i;

  iList.n = n;
  iList.valid = 0;
  iList.list = (bestInfo *)malloc(sizeof(bestInfo) * n);

  for(i = 0; i < n; i++)
  {
    iList.list[i].node = (nodeptr)NULL;
    iList.list[i].likelihood = unlikely;
  }
}

void freeInfoList(void)
{ 
  free(iList.list);   
}


void insertInfoList(nodeptr node, double likelihood)
{
  int i;
  int min = 0;
  double min_l =  iList.list[0].likelihood;

  for(i = 1; i < iList.n; i++)
  {
    if(iList.list[i].likelihood < min_l)
    {
      min = i;
      min_l = iList.list[i].likelihood;
    }
  }

  if(likelihood > min_l)
  {
    iList.list[min].likelihood = likelihood;
    iList.list[min].node = node;
    iList.valid += 1;
  }

  if(iList.valid > iList.n)
    iList.valid = iList.n;
}


boolean smoothRegion (tree *tr, nodeptr p, int region)
{ 
  nodeptr  q;

  if (! update(tr, p))               return FALSE; /*  Adjust branch */

  if(region > 0)
  {
    if (!isTip(p->number, tr->rdta->numsp)) 
    {                                 
      q = p->next;
      while (q != p) 
      {
        if (! smoothRegion(tr, q->back, --region))   return FALSE;
        q = q->next;
      }	

      newviewGeneric(tr, p);
    }
  }

  return TRUE;
}

boolean regionalSmooth (tree *tr, nodeptr p, int maxtimes, int region)
{
  nodeptr  q;
  int i;

  if (isTip(p->number, tr->rdta->numsp)) return FALSE;            /* Should be an error */

  for(i = 0; i < tr->numBranches; i++)
    tr->partitionConverged[i] = FALSE;

  while (--maxtimes >= 0) 
  {	
    for(i = 0; i < tr->numBranches; i++)	  
      tr->partitionSmoothed[i] = TRUE;

    q = p;
    do 
    {
      if (! smoothRegion(tr, q, region)) return FALSE;
      q = q->next;
    } 
    while (q != p);

    if (allSmoothed(tr)) 
      break;
  }

  for(i = 0; i < tr->numBranches; i++)
    tr->partitionSmoothed[i] = FALSE;
  for(i = 0; i < tr->numBranches; i++)
    tr->partitionConverged[i] = FALSE;

  return TRUE;
} /* localSmooth */





nodeptr  removeNodeBIG (tree *tr, nodeptr p, int numBranches)
{  
  double   zqr[NUM_BRANCHES], result[NUM_BRANCHES];
  nodeptr  q, r;
  int i;

  q = p->next->back;
  r = p->next->next->back;

  for(i = 0; i < numBranches; i++)
    zqr[i] = q->z[i] * r->z[i];        

  makenewzGeneric(tr, q, r, zqr, iterations, result, FALSE);   

  for(i = 0; i < numBranches; i++)        
    tr->zqr[i] = result[i];

  hookup(q, r, result, numBranches); 

  p->next->next->back = p->next->back = (node *) NULL;

  return  q; 
}

nodeptr  removeNodeRestoreBIG (tree *tr, nodeptr p)
{
  nodeptr  q, r;

  q = p->next->back;
  r = p->next->next->back;  

  newviewGeneric(tr, q);
  newviewGeneric(tr, r);

  hookup(q, r, tr->currentZQR, tr->numBranches);

  p->next->next->back = p->next->back = (node *) NULL;

  return  q;
}


boolean insertBIG (tree *tr, nodeptr p, nodeptr q, int numBranches)
{
  nodeptr  r, s;
  int i;

  r = q->back;
  s = p->back;

  for(i = 0; i < numBranches; i++)
    tr->lzi[i] = q->z[i];

  if(Thorough)
  { 
    double  zqr[NUM_BRANCHES], zqs[NUM_BRANCHES], zrs[NUM_BRANCHES], lzqr, lzqs, lzrs, lzsum, lzq, lzr, lzs, lzmax;      
    double defaultArray[NUM_BRANCHES];	
    double e1[NUM_BRANCHES], e2[NUM_BRANCHES], e3[NUM_BRANCHES];
    double *qz;

    qz = q->z;

    for(i = 0; i < numBranches; i++)
      defaultArray[i] = defaultz;

    makenewzGeneric(tr, q, r, qz, iterations, zqr, FALSE);           
    makenewzGeneric(tr, q, s, defaultArray, iterations, zqs, FALSE);                  
    makenewzGeneric(tr, r, s, defaultArray, iterations, zrs, FALSE);


    for(i = 0; i < numBranches; i++)
    {
      lzqr = (zqr[i] > zmin) ? log(zqr[i]) : log(zmin); 
      lzqs = (zqs[i] > zmin) ? log(zqs[i]) : log(zmin);
      lzrs = (zrs[i] > zmin) ? log(zrs[i]) : log(zmin);
      lzsum = 0.5 * (lzqr + lzqs + lzrs);

      lzq = lzsum - lzrs;
      lzr = lzsum - lzqs;
      lzs = lzsum - lzqr;
      lzmax = log(zmax);

      if      (lzq > lzmax) {lzq = lzmax; lzr = lzqr; lzs = lzqs;} 
      else if (lzr > lzmax) {lzr = lzmax; lzq = lzqr; lzs = lzrs;}
      else if (lzs > lzmax) {lzs = lzmax; lzq = lzqs; lzr = lzrs;}          

      e1[i] = exp(lzq);
      e2[i] = exp(lzr);
      e3[i] = exp(lzs);
    }
    hookup(p->next,       q, e1, numBranches);
    hookup(p->next->next, r, e2, numBranches);
    hookup(p,             s, e3, numBranches);      		  
  }
  else
  {       
    double  z[NUM_BRANCHES]; 

    for(i = 0; i < numBranches; i++)
    {
      z[i] = sqrt(q->z[i]);      

      if(z[i] < zmin) 
        z[i] = zmin;
      if(z[i] > zmax)
        z[i] = zmax;
    }

    hookup(p->next,       q, z, tr->numBranches);
    hookup(p->next->next, r, z, tr->numBranches);	                         
  }

  newviewGeneric(tr, p);

  if(Thorough)
  {     
    localSmooth(tr, p, smoothings);   
    for(i = 0; i < numBranches; i++)
    {
      tr->lzq[i] = p->next->z[i];
      tr->lzr[i] = p->next->next->z[i];
      tr->lzs[i] = p->z[i];            
    }
  }           

  return  TRUE;
}

boolean insertRestoreBIG (tree *tr, nodeptr p, nodeptr q)
{
  nodeptr  r, s;

  r = q->back;
  s = p->back;

  if(Thorough)
  {                        
    hookup(p->next,       q, tr->currentLZQ, tr->numBranches);
    hookup(p->next->next, r, tr->currentLZR, tr->numBranches);
    hookup(p,             s, tr->currentLZS, tr->numBranches);      		  
  }
  else
  {       
    double  z[NUM_BRANCHES];
    int i;

    for(i = 0; i < tr->numBranches; i++)
    {
      double zz;
      zz = sqrt(q->z[i]);     
      if(zz < zmin) 
        zz = zmin;
      if(zz > zmax)
        zz = zmax;
      z[i] = zz;
    }

    hookup(p->next,       q, z, tr->numBranches);
    hookup(p->next->next, r, z, tr->numBranches);
  }   

  newviewGeneric(tr, p);

  return  TRUE;
}


static void restoreTopologyOnly(tree *tr, bestlist *bt)
{ 
  nodeptr p = tr->removeNode;
  nodeptr q = tr->insertNode;
  double qz[NUM_BRANCHES], pz[NUM_BRANCHES], p1z[NUM_BRANCHES], p2z[NUM_BRANCHES];
  nodeptr p1, p2, r, s;
  double currentLH = tr->likelihood;
  int i;

  p1 = p->next->back;
  p2 = p->next->next->back;

  for(i = 0; i < tr->numBranches; i++)
  {
    p1z[i] = p1->z[i];
    p2z[i] = p2->z[i];
  }

  hookup(p1, p2, tr->currentZQR, tr->numBranches);

  p->next->next->back = p->next->back = (node *) NULL;             
  for(i = 0; i < tr->numBranches; i++)
  {
    qz[i] = q->z[i];
    pz[i] = p->z[i];           
  }

  r = q->back;
  s = p->back;

  if(Thorough)
  {                        
    hookup(p->next,       q, tr->currentLZQ, tr->numBranches);
    hookup(p->next->next, r, tr->currentLZR, tr->numBranches);
    hookup(p,             s, tr->currentLZS, tr->numBranches);      		  
  }
  else
  { 	
    double  z[NUM_BRANCHES];	
    for(i = 0; i < tr->numBranches; i++)
    {
      z[i] = sqrt(q->z[i]);      
      if(z[i] < zmin)
        z[i] = zmin;
      if(z[i] > zmax)
        z[i] = zmax;
    }
    hookup(p->next,       q, z, tr->numBranches);
    hookup(p->next->next, r, z, tr->numBranches);
  }     

  tr->likelihood = tr->bestOfNode;

  saveBestTree(bt, tr);

  tr->likelihood = currentLH;

  hookup(q, r, qz, tr->numBranches);

  p->next->next->back = p->next->back = (nodeptr) NULL;

  if(Thorough)    
    hookup(p, s, pz, tr->numBranches);          

  hookup(p->next,       p1, p1z, tr->numBranches); 
  hookup(p->next->next, p2, p2z, tr->numBranches);      
}


boolean testInsertBIG (tree *tr, nodeptr p, nodeptr q)
{
  double  qz[NUM_BRANCHES], pz[NUM_BRANCHES];
  nodeptr  r;
  boolean doIt = TRUE;
  double startLH = tr->endLH;
  int i;

  r = q->back; 
  for(i = 0; i < tr->numBranches; i++)
  {
    qz[i] = q->z[i];
    pz[i] = p->z[i];
  }



  if(doIt)
  {     
    if (! insertBIG(tr, p, q, tr->numBranches))       return FALSE;         

    evaluateGeneric(tr, p->next->next);       

    if(tr->likelihood > tr->bestOfNode)
    {
      tr->bestOfNode = tr->likelihood;
      tr->insertNode = q;
      tr->removeNode = p;   
      for(i = 0; i < tr->numBranches; i++)
      {
        tr->currentZQR[i] = tr->zqr[i];           
        tr->currentLZR[i] = tr->lzr[i];
        tr->currentLZQ[i] = tr->lzq[i];
        tr->currentLZS[i] = tr->lzs[i];      
      }
    }

    if(tr->likelihood > tr->endLH)
    {			  
      tr->insertNode = q;
      tr->removeNode = p;   
      for(i = 0; i < tr->numBranches; i++)
        tr->currentZQR[i] = tr->zqr[i];      
      tr->endLH = tr->likelihood;                      
    }        

    hookup(q, r, qz, tr->numBranches);

    p->next->next->back = p->next->back = (nodeptr) NULL;

    if(Thorough)
    {
      nodeptr s = p->back;
      hookup(p, s, pz, tr->numBranches);      
    } 

    if((tr->doCutoff) && (tr->likelihood < startLH))
    {
      tr->lhAVG += (startLH - tr->likelihood);
      tr->lhDEC++;
      if((startLH - tr->likelihood) >= tr->lhCutoff)
        return FALSE;	    
      else
        return TRUE;
    }
    else
      return TRUE;
  }
  else
    return TRUE;  
}




void addTraverseBIG(tree *tr, nodeptr p, nodeptr q, int mintrav, int maxtrav)
{  
  if (--mintrav <= 0) 
  {              
    if (! testInsertBIG(tr, p, q))  return;

  }

  if ((!isTip(q->number, tr->rdta->numsp)) && (--maxtrav > 0)) 
  {    
    addTraverseBIG(tr, p, q->next->back, mintrav, maxtrav);
    addTraverseBIG(tr, p, q->next->next->back, mintrav, maxtrav);    
  }
} 





int rearrangeBIG(tree *tr, nodeptr p, int mintrav, int maxtrav)   
{  
  double   p1z[NUM_BRANCHES], p2z[NUM_BRANCHES], q1z[NUM_BRANCHES], q2z[NUM_BRANCHES];
  nodeptr  p1, p2, q, q1, q2;
  int      mintrav2, i;  
  boolean doP = TRUE, doQ = TRUE;

  if (maxtrav < 1 || mintrav > maxtrav)  return 0;
  q = p->back;



  if (!isTip(p->number, tr->rdta->numsp) && doP) 
  {     
    p1 = p->next->back;
    p2 = p->next->next->back;


    if(!isTip(p1->number, tr->rdta->numsp) || !isTip(p2->number, tr->rdta->numsp))
    {
      for(i = 0; i < tr->numBranches; i++)
      {
        p1z[i] = p1->z[i];
        p2z[i] = p2->z[i];	   	   
      }

      if (! removeNodeBIG(tr, p,  tr->numBranches)) return badRear;

      if (!isTip(p1->number, tr->rdta->numsp)) 
      {
        addTraverseBIG(tr, p, p1->next->back,
            mintrav, maxtrav);         
        addTraverseBIG(tr, p, p1->next->next->back,
            mintrav, maxtrav);          
      }

      if (!isTip(p2->number, tr->rdta->numsp)) 
      {
        addTraverseBIG(tr, p, p2->next->back,
            mintrav, maxtrav);
        addTraverseBIG(tr, p, p2->next->next->back,
            mintrav, maxtrav);          
      }

      hookup(p->next,       p1, p1z, tr->numBranches); 
      hookup(p->next->next, p2, p2z, tr->numBranches);	   	    	    
      newviewGeneric(tr, p);	   	    
    }
  }  

  if (!isTip(q->number, tr->rdta->numsp) && maxtrav > 0 && doQ) 
  {
    q1 = q->next->back;
    q2 = q->next->next->back;

    /*if (((!q1->tip) && (!q1->next->back->tip || !q1->next->next->back->tip)) ||
      ((!q2->tip) && (!q2->next->back->tip || !q2->next->next->back->tip))) */
    if (
        (
         ! isTip(q1->number, tr->rdta->numsp) && 
         (! isTip(q1->next->back->number, tr->rdta->numsp) || ! isTip(q1->next->next->back->number, tr->rdta->numsp))
        )
        ||
        (
         ! isTip(q2->number, tr->rdta->numsp) && 
         (! isTip(q2->next->back->number, tr->rdta->numsp) || ! isTip(q2->next->next->back->number, tr->rdta->numsp))
        )
       )
    {

      for(i = 0; i < tr->numBranches; i++)
      {
        q1z[i] = q1->z[i];
        q2z[i] = q2->z[i];
      }

      if (! removeNodeBIG(tr, q, tr->numBranches)) return badRear;

      mintrav2 = mintrav > 2 ? mintrav : 2;

      if (/*! q1->tip*/ !isTip(q1->number, tr->rdta->numsp)) 
      {
        addTraverseBIG(tr, q, q1->next->back,
            mintrav2 , maxtrav);
        addTraverseBIG(tr, q, q1->next->next->back,
            mintrav2 , maxtrav);         
      }

      if (/*! q2->tip*/ ! isTip(q2->number, tr->rdta->numsp)) 
      {
        addTraverseBIG(tr, q, q2->next->back,
            mintrav2 , maxtrav);
        addTraverseBIG(tr, q, q2->next->next->back,
            mintrav2 , maxtrav);          
      }	   

      hookup(q->next,       q1, q1z, tr->numBranches); 
      hookup(q->next->next, q2, q2z, tr->numBranches);

      newviewGeneric(tr, q); 	   
    }
  } 

  return  1;
} 





double treeOptimizeRapid(tree *tr, int mintrav, int maxtrav, analdef *adef, bestlist *bt)
{
  int i, index,
      *perm = (int*)NULL;   

  nodeRectifier(tr);



  if (maxtrav > tr->mxtips - 3)  
    maxtrav = tr->mxtips - 3;  



  resetInfoList();

  resetBestTree(bt);

  tr->startLH = tr->endLH = tr->likelihood;

  if(tr->doCutoff)
  {
    if(tr->bigCutoff)
    {	  
      if(tr->itCount == 0)    
        tr->lhCutoff = 0.5 * (tr->likelihood / -1000.0);    
      else    		 
        tr->lhCutoff = 0.5 * ((tr->lhAVG) / ((double)(tr->lhDEC))); 	  
    }
    else
    {
      if(tr->itCount == 0)    
        tr->lhCutoff = tr->likelihood / -1000.0;    
      else    		 
        tr->lhCutoff = (tr->lhAVG) / ((double)(tr->lhDEC));   
    }    

    tr->itCount = tr->itCount + 1;
    tr->lhAVG = 0;
    tr->lhDEC = 0;
  }

  /*
     printf("DoCutoff: %d\n", tr->doCutoff);
     printf("%d %f %f %f\n", tr->itCount, tr->lhAVG, tr->lhDEC, tr->lhCutoff);

     printf("%d %d\n", mintrav, maxtrav);
     */

  for(i = 1; i <= tr->mxtips + tr->mxtips - 2; i++)
  {           
    tr->bestOfNode = unlikely;          

    if(adef->permuteTreeoptimize)
      index = perm[i];
    else
      index = i;     

    if(rearrangeBIG(tr, tr->nodep[index], mintrav, maxtrav))
    {    
      if(Thorough)
      {
        if(tr->endLH > tr->startLH)                 	
        {			   	     
          restoreTreeFast(tr);	 	 
          tr->startLH = tr->endLH = tr->likelihood;	 
          saveBestTree(bt, tr);
        }
        else
        { 		  
          if(tr->bestOfNode != unlikely)		    	     
            restoreTopologyOnly(tr, bt);		    
        }	   
      }
      else
      {
        insertInfoList(tr->nodep[index], tr->bestOfNode);	    
        if(tr->endLH > tr->startLH)                 	
        {		      
          restoreTreeFast(tr);	  	      
          tr->startLH = tr->endLH = tr->likelihood;	  	 	  	  	  	  	  	  
        }	    	  
      }
    }     
  }     

  if(!Thorough)
  {           
    Thorough = 1;  

    for(i = 0; i < iList.valid; i++)
    { 	  
      tr->bestOfNode = unlikely;

      if(rearrangeBIG(tr, iList.list[i].node, mintrav, maxtrav))
      {	  
        if(tr->endLH > tr->startLH)                 	
        {	 	     
          restoreTreeFast(tr);	 	 
          tr->startLH = tr->endLH = tr->likelihood;	 
          saveBestTree(bt, tr);
        }
        else
        { 

          if(tr->bestOfNode != unlikely)
          {	     
            restoreTopologyOnly(tr, bt);
          }	
        }      
      }
    }       

    Thorough = 0;
  }

  if(adef->permuteTreeoptimize)
    free(perm);

  return tr->startLH;     
}




boolean testInsertRestoreBIG (tree *tr, nodeptr p, nodeptr q)
{    
  if(Thorough)
  {
    if (! insertBIG(tr, p, q, tr->numBranches))       return FALSE;    

    evaluateGeneric(tr, p->next->next);               
  }
  else
  {
    if (! insertRestoreBIG(tr, p, q))       return FALSE;

    {
      nodeptr x, y;
      x = p->next->next;
      y = p->back;

      if(! isTip(x->number, tr->rdta->numsp) && isTip(y->number, tr->rdta->numsp))
      {
        while ((! x->x)) 
        {
          if (! (x->x))
            newviewGeneric(tr, x);		     
        }
      }

      if(isTip(x->number, tr->rdta->numsp) && !isTip(y->number, tr->rdta->numsp))
      {
        while ((! y->x)) 
        {		  
          if (! (y->x))
            newviewGeneric(tr, y);
        }
      }

      if(!isTip(x->number, tr->rdta->numsp) && !isTip(y->number, tr->rdta->numsp))
      {
        while ((! x->x) || (! y->x)) 
        {
          if (! (x->x))
            newviewGeneric(tr, x);
          if (! (y->x))
            newviewGeneric(tr, y);
        }
      }				      	

    }

    tr->likelihood = tr->endLH;
  }

  return TRUE;
} 

void restoreTreeFast(tree *tr)
{
  removeNodeRestoreBIG(tr, tr->removeNode);    
  testInsertRestoreBIG(tr, tr->removeNode, tr->insertNode);
}

static void myfwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
{
  size_t  
    bytes_written = fwrite(ptr, size, nmemb, stream);

  assert(bytes_written == nmemb);
}

static void myfread(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
  size_t
    bytes_read;

  bytes_read = fread(ptr, size, nmemb, stream);

  assert(bytes_read == nmemb);
}




static void writeTree(tree *tr, FILE *f)
{
  int 
    x = tr->mxtips + 3 * (tr->mxtips - 1);

  nodeptr
    base = tr->nodeBaseAddress;

  myfwrite(&(tr->start->number), sizeof(int), 1, f);
  myfwrite(&base, sizeof(nodeptr), 1, f);
  myfwrite(tr->nodeBaseAddress, sizeof(node), x, f);

}

int ckpCount = 0;

static void writeCheckpoint(tree *tr)
{
  int   
    model; 

  char 
    extendedName[2048],
    buf[64];

  FILE 
    *f;

  strcpy(extendedName,  binaryCheckpointName);
  strcat(extendedName, "_");
  sprintf(buf, "%d", ckpCount);
  strcat(extendedName, buf);  

  ckpCount++;

  f = myfopen(extendedName, "w"); 

  /* cdta */   


  ckp.accumulatedTime = accumulatedTime + (gettime() - masterTime);

  /* printf("Acc time: %f\n", ckp.accumulatedTime); */

  myfwrite(&ckp, sizeof(checkPointState), 1, f);

  myfwrite(tr->tree0, sizeof(char), tr->treeStringLength, f);
  myfwrite(tr->tree1, sizeof(char), tr->treeStringLength, f);

  myfwrite(tr->cdta->rateCategory, sizeof(int), tr->rdta->sites + 1, f);
  myfwrite(tr->cdta->patrat, sizeof(double), tr->rdta->sites + 1, f);
  myfwrite(tr->cdta->patratStored, sizeof(double), tr->rdta->sites + 1, f);
  myfwrite(tr->wr,  sizeof(double), tr->rdta->sites + 1, f);
  myfwrite(tr->wr2,  sizeof(double), tr->rdta->sites + 1, f);


  /* pInfo */

  for(model = 0; model < tr->NumberOfModels; model++)
  {
    int 
      dataType = tr->partitionData[model].dataType;

    myfwrite(&(tr->partitionData[model].numberOfCategories), sizeof(int), 1, f);
    myfwrite(tr->partitionData[model].perSiteRates, sizeof(double), tr->maxCategories, f);
    myfwrite(tr->partitionData[model].EIGN, sizeof(double), pLengths[dataType].eignLength, f);
    myfwrite(tr->partitionData[model].EV, sizeof(double),  pLengths[dataType].evLength, f);
    myfwrite(tr->partitionData[model].EI, sizeof(double),  pLengths[dataType].eiLength, f);  

    myfwrite(tr->partitionData[model].frequencies, sizeof(double),  pLengths[dataType].frequenciesLength, f);
    myfwrite(tr->partitionData[model].tipVector, sizeof(double),  pLengths[dataType].tipVectorLength, f);  
    myfwrite(tr->partitionData[model].substRates, sizeof(double),  pLengths[dataType].substRatesLength, f);    
    myfwrite(&(tr->partitionData[model].alpha), sizeof(double), 1, f);
  }



  writeTree(tr, f);

  fclose(f); 

  printBothOpen("\nCheckpoint written to: %s likelihood: %f\n", extendedName, tr->likelihood);
}

static void readTree(tree *tr, FILE *f, analdef *adef)
{
  int 
    nodeNumber,   
    x = tr->mxtips + 3 * (tr->mxtips - 1);

  nodeptr
    base = tr->nodeBaseAddress;



  nodeptr
    startAddress;

  myfread(&nodeNumber, sizeof(int), 1, f);

  tr->start = tr->nodep[nodeNumber];

  /*printf("Start: %d %d\n", tr->start->number, nodeNumber);*/

  myfread(&startAddress, sizeof(nodeptr), 1, f);

  /*printf("%u %u\n", (size_t)startAddress, (size_t)tr->nodeBaseAddress);*/



  myfread(tr->nodeBaseAddress, sizeof(node), x, f);

  {
    int i;    

    size_t         
      offset;

    boolean 
      addIt;

    if(startAddress > tr->nodeBaseAddress)
    {
      addIt = FALSE;
      offset = (size_t)startAddress - (size_t)tr->nodeBaseAddress;
    }
    else
    {
      addIt = TRUE;
      offset = (size_t)tr->nodeBaseAddress - (size_t)startAddress;
    }       

    for(i = 0; i < x; i++)
    {      	
      if(addIt)
      {	    
        tr->nodeBaseAddress[i].next = (nodeptr)((size_t)tr->nodeBaseAddress[i].next + offset);	
        tr->nodeBaseAddress[i].back = (nodeptr)((size_t)tr->nodeBaseAddress[i].back + offset);
      }
      else
      {

        tr->nodeBaseAddress[i].next = (nodeptr)((size_t)tr->nodeBaseAddress[i].next - offset);	
        tr->nodeBaseAddress[i].back = (nodeptr)((size_t)tr->nodeBaseAddress[i].back - offset);	   
      } 
    }

  }

  evaluateGenericInitrav(tr, tr->start);  

  printBothOpen("RAxML Restart with likelihood: %1.50f\n", tr->likelihood);
}


static void readCheckpoint(tree *tr, analdef *adef)
{
  int   
    model; 

  FILE 
    *f = myfopen(binaryCheckpointInputName, "r");

  /* cdta */   

  myfread(&ckp, sizeof(checkPointState), 1, f);

  tr->ntips = tr->mxtips;

  tr->startLH    = ckp.tr_startLH;
  tr->endLH      = ckp.tr_endLH;
  tr->likelihood = ckp.tr_likelihood;
  tr->bestOfNode = ckp.tr_bestOfNode;

  tr->lhCutoff   = ckp.tr_lhCutoff;
  tr->lhAVG      = ckp.tr_lhAVG;
  tr->lhDEC      = ckp.tr_lhDEC;
  tr->itCount    = ckp.tr_itCount;
  Thorough       = ckp.Thorough;

  accumulatedTime = ckp.accumulatedTime;

  /* printf("Accumulated time so far: %f\n", accumulatedTime); */

  optimizeRateCategoryInvocations = ckp.optimizeRateCategoryInvocations;


  myfread(tr->tree0, sizeof(char), tr->treeStringLength, f);
  myfread(tr->tree1, sizeof(char), tr->treeStringLength, f);

  if(tr->searchConvergenceCriterion)
  {
    int bCounter = 0;

    if((ckp.state == FAST_SPRS && ckp.fastIterations > 0) ||
        (ckp.state == SLOW_SPRS && ckp.thoroughIterations > 0))
    { 

#ifdef _DEBUG_CHECKPOINTING    
      printf("parsing Tree 0\n");
#endif

      treeReadTopologyString(tr->tree0, tr);   

      bitVectorInitravSpecial(tr->bitVectors, tr->nodep[1]->back, tr->mxtips, tr->vLength, tr->h, 0, BIPARTITIONS_RF, (branchInfo *)NULL,
          &bCounter, 1, FALSE, FALSE);

      assert(bCounter == tr->mxtips - 3);
    }

    bCounter = 0;

    if((ckp.state == FAST_SPRS && ckp.fastIterations > 1) ||
        (ckp.state == SLOW_SPRS && ckp.thoroughIterations > 1))
    {

#ifdef _DEBUG_CHECKPOINTING
      printf("parsing Tree 1\n");
#endif

      treeReadTopologyString(tr->tree1, tr); 

      bitVectorInitravSpecial(tr->bitVectors, tr->nodep[1]->back, tr->mxtips, tr->vLength, tr->h, 1, BIPARTITIONS_RF, (branchInfo *)NULL,
          &bCounter, 1, FALSE, FALSE);

      assert(bCounter == tr->mxtips - 3);
    }
  }

  myfread(tr->cdta->rateCategory, sizeof(int), tr->rdta->sites + 1, f);
  myfread(tr->cdta->patrat, sizeof(double), tr->rdta->sites + 1, f);
  myfread(tr->cdta->patratStored, sizeof(double), tr->rdta->sites + 1, f);
  myfread(tr->wr,  sizeof(double), tr->rdta->sites + 1, f);
  myfread(tr->wr2,  sizeof(double), tr->rdta->sites + 1, f);


  /* pInfo */

  for(model = 0; model < tr->NumberOfModels; model++)
  {
    int 
      dataType = tr->partitionData[model].dataType;

    myfread(&(tr->partitionData[model].numberOfCategories), sizeof(int), 1, f);
    myfread(tr->partitionData[model].perSiteRates, sizeof(double), tr->maxCategories, f);
    myfread(tr->partitionData[model].EIGN, sizeof(double), pLengths[dataType].eignLength, f);
    myfread(tr->partitionData[model].EV, sizeof(double),  pLengths[dataType].evLength, f);
    myfread(tr->partitionData[model].EI, sizeof(double),  pLengths[dataType].eiLength, f);  

    myfread(tr->partitionData[model].frequencies, sizeof(double),  pLengths[dataType].frequenciesLength, f);
    myfread(tr->partitionData[model].tipVector, sizeof(double),  pLengths[dataType].tipVectorLength, f);  
    myfread(tr->partitionData[model].substRates, sizeof(double),  pLengths[dataType].substRatesLength, f);  
    myfread(&(tr->partitionData[model].alpha), sizeof(double), 1, f);
    makeGammaCats(tr->partitionData[model].alpha, tr->partitionData[model].gammaRates, 4);
  }

#ifdef _FINE_GRAIN_MPI
  masterBarrierMPI(THREAD_COPY_INIT_MODEL, tr);
#endif

#ifdef _USE_PTHREADS
  masterBarrier(THREAD_COPY_INIT_MODEL, tr);
#endif

  updatePerSiteRates(tr, FALSE);  

  readTree(tr, f, adef);

  fclose(f); 

}


void restart(tree *tr, analdef *adef)
{  
  readCheckpoint(tr, adef);

  switch(ckp.state)
  {
    case REARR_SETTING:      
      break;
    case FAST_SPRS:
      break;
    case SLOW_SPRS:
      break;
    default:
      assert(0);
  }
}

int determineRearrangementSetting(tree *tr,  analdef *adef, bestlist *bestT, bestlist *bt)
{
  const 
    int MaxFast = 26;

  int 
    i,   
    maxtrav = 5, 
    bestTrav = 5;

  double 
    startLH = tr->likelihood; 

  boolean 
    impr   = TRUE,
           cutoff = tr->doCutoff;

  if(adef->useCheckpoint)
  {
    assert(ckp.state == REARR_SETTING);

    maxtrav = ckp.maxtrav;
    bestTrav = ckp.bestTrav;
    startLH  = ckp.startLH;
    impr     = ckp.impr;

    cutoff = ckp.cutoff;

    adef->useCheckpoint = FALSE;
  }

  tr->doCutoff = FALSE;      

  resetBestTree(bt);    

#ifdef _DEBUG_CHECKPOINTING
  printBothOpen("MAXTRAV: %d\n", maxtrav);
#endif

  while(impr && maxtrav < MaxFast)
  {	
    recallBestTree(bestT, 1, tr);     
    nodeRectifier(tr);            

    ckp.optimizeRateCategoryInvocations = optimizeRateCategoryInvocations;

    ckp.cutoff = cutoff;
    ckp.state = REARR_SETTING;     
    ckp.maxtrav = maxtrav;
    ckp.bestTrav = bestTrav;
    ckp.startLH  = startLH;
    ckp.impr = impr;

    ckp.tr_startLH  = tr->startLH;
    ckp.tr_endLH    = tr->endLH;
    ckp.tr_likelihood = tr->likelihood;
    ckp.tr_bestOfNode = tr->bestOfNode;

    ckp.tr_lhCutoff = tr->lhCutoff;
    ckp.tr_lhAVG    = tr->lhAVG;
    ckp.tr_lhDEC    = tr->lhDEC;      
    ckp.tr_itCount  = tr->itCount;


    writeCheckpoint(tr);    

    if (maxtrav > tr->mxtips - 3)  
      maxtrav = tr->mxtips - 3;    

    tr->startLH = tr->endLH = tr->likelihood;

    for(i = 1; i <= tr->mxtips + tr->mxtips - 2; i++)
    {                	         
      tr->bestOfNode = unlikely;

      if(rearrangeBIG(tr, tr->nodep[i], 1, maxtrav))
      {	     
        if(tr->endLH > tr->startLH)                 	
        {		 	 	      
          restoreTreeFast(tr);	        	  	 	  	      
          tr->startLH = tr->endLH = tr->likelihood;		 
        }	         	       	
      }
    }

    treeEvaluate(tr, 0.25);
    saveBestTree(bt, tr); 

#ifdef _DEBUG_CHECKPOINTING
    printBothOpen("TRAV: %d lh %f\n", maxtrav, tr->likelihood);
#endif

    if(tr->likelihood > startLH)
    {	 
      startLH = tr->likelihood; 	  	  	  
      printLog(tr, adef, FALSE);	  
      bestTrav = maxtrav;	 
      impr = TRUE;
    }
    else	
      impr = FALSE;	



    if(tr->doCutoff)
    {
      tr->lhCutoff = (tr->lhAVG) / ((double)(tr->lhDEC));       

      tr->itCount =  tr->itCount + 1;
      tr->lhAVG = 0;
      tr->lhDEC = 0;
    }

    maxtrav += 5;


  }

  recallBestTree(bt, 1, tr);

  tr->doCutoff = cutoff; 

#ifdef _DEBUG_CHECKPOINTING
  printBothOpen("BestTrav %d\n", bestTrav);
#endif

  return bestTrav;     
}





void computeBIGRAPID (tree *tr, analdef *adef, boolean estimateModel) 
{   
  int
    i,
    impr, 
    bestTrav,
    treeVectorLength = 0,
    rearrangementsMax = 0, 
    rearrangementsMin = 0,    
    thoroughIterations = 0,
    fastIterations = 0;

  double 
    lh, 
    previousLh, 
    difference, 
    epsilon;              

  bestlist 
    *bestT, 
    *bt;        

  if(tr->searchConvergenceCriterion)   
    treeVectorLength = 1;

  bestT = (bestlist *) malloc(sizeof(bestlist));
  bestT->ninit = 0;
  initBestTree(bestT, 1, tr->mxtips);

  bt = (bestlist *) malloc(sizeof(bestlist));      
  bt->ninit = 0;
  initBestTree(bt, 20, tr->mxtips); 

  initInfoList(50);

  difference = 10.0;
  epsilon = 0.01;    

  Thorough = 0;     

  if(!adef->useCheckpoint)
  {
    if(estimateModel)
      modOpt(tr, adef, FALSE, 10.0, FALSE);
    else
      treeEvaluate(tr, 2);  
  }


  printLog(tr, adef, FALSE); 

  saveBestTree(bestT, tr);

  if(!adef->initialSet)   
  {
    if((!adef->useCheckpoint) || (adef->useCheckpoint && ckp.state == REARR_SETTING))
    {
      bestTrav = adef->bestTrav = determineRearrangementSetting(tr, adef, bestT, bt);     	  
      printBothOpen("\nBest rearrangement radius: %d\n", bestTrav);
    }
  }
  else
  {
    bestTrav = adef->bestTrav = adef->initial;       
    printBothOpen("\nUser-defined rearrangement radius: %d\n", bestTrav);
  }



  if(!(adef->useCheckpoint && (ckp.state == FAST_SPRS || ckp.state == SLOW_SPRS)))
  {      
    if(estimateModel)
      modOpt(tr, adef, FALSE, 5.0, FALSE);
    else
      treeEvaluate(tr, 1);   
  }

  saveBestTree(bestT, tr); 
  impr = 1;
  if(tr->doCutoff)
    tr->itCount = 0;

  if(adef->useCheckpoint && ckp.state == FAST_SPRS)
    goto START_FAST_SPRS;

  if(adef->useCheckpoint && ckp.state == SLOW_SPRS)
    goto START_SLOW_SPRS;

  while(impr)
  {              
START_FAST_SPRS:
    if(adef->useCheckpoint && ckp.state == FAST_SPRS)
    {
      optimizeRateCategoryInvocations = ckp.optimizeRateCategoryInvocations;   	


      impr = ckp.impr;
      Thorough = ckp.Thorough;
      bestTrav = ckp.bestTrav;
      treeVectorLength = ckp.treeVectorLength;
      rearrangementsMax = ckp.rearrangementsMax;
      rearrangementsMin = ckp.rearrangementsMin;
      thoroughIterations = ckp.thoroughIterations;
      fastIterations = ckp.fastIterations;


      lh = ckp.lh;
      previousLh = ckp.previousLh;
      difference = ckp.difference;
      epsilon    = ckp.epsilon;                    


      tr->likelihood = ckp.tr_likelihood;

      tr->lhCutoff = ckp.tr_lhCutoff;
      tr->lhAVG    = ckp.tr_lhAVG;
      tr->lhDEC    = ckp.tr_lhDEC;   	 
      tr->itCount = ckp.tr_itCount;
      tr->doCutoff = ckp.tr_doCutoff;

      adef->useCheckpoint = FALSE;
    }
    else
      recallBestTree(bestT, 1, tr); 

    {              
      ckp.state = FAST_SPRS;  
      ckp.optimizeRateCategoryInvocations = optimizeRateCategoryInvocations;              


      ckp.impr = impr;
      ckp.Thorough = Thorough;
      ckp.bestTrav = bestTrav;
      ckp.treeVectorLength = treeVectorLength;
      ckp.rearrangementsMax = rearrangementsMax;
      ckp.rearrangementsMin = rearrangementsMin;
      ckp.thoroughIterations = thoroughIterations;
      ckp.fastIterations = fastIterations;


      ckp.lh = lh;
      ckp.previousLh = previousLh;
      ckp.difference = difference;
      ckp.epsilon    = epsilon; 


      ckp.bestTrav = bestTrav;       
      ckp.impr = impr;

      ckp.tr_startLH  = tr->startLH;
      ckp.tr_endLH    = tr->endLH;
      ckp.tr_likelihood = tr->likelihood;
      ckp.tr_bestOfNode = tr->bestOfNode;

      ckp.tr_lhCutoff = tr->lhCutoff;
      ckp.tr_lhAVG    = tr->lhAVG;
      ckp.tr_lhDEC    = tr->lhDEC;       
      ckp.tr_itCount  = tr->itCount;
      ckp.tr_doCutoff = tr->doCutoff;

      writeCheckpoint(tr); 
    }


    if(tr->searchConvergenceCriterion)
    {
      int bCounter = 0;	  	      	 	  	  	

      if(fastIterations > 1)
        cleanupHashTable(tr->h, (fastIterations % 2));		

      bitVectorInitravSpecial(tr->bitVectors, tr->nodep[1]->back, tr->mxtips, tr->vLength, tr->h, fastIterations % 2, BIPARTITIONS_RF, (branchInfo *)NULL,
          &bCounter, 1, FALSE, FALSE);	    

      {
        char 
          *buffer = (char*)calloc(tr->treeStringLength, sizeof(char));
#ifdef _DEBUG_CHECKPOINTING
        printf("Storing tree in slot %d\n", fastIterations % 2);
#endif

        Tree2String(buffer, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, FALSE, adef, SUMMARIZE_LH, FALSE, FALSE);

        if(fastIterations % 2 == 0)	      
          memcpy(tr->tree0, buffer, tr->treeStringLength * sizeof(char));
        else
          memcpy(tr->tree1, buffer, tr->treeStringLength * sizeof(char));	    

        free(buffer);
      }


      assert(bCounter == tr->mxtips - 3);	    	   

      if(fastIterations > 0)
      {
        double rrf = convergenceCriterion(tr->h, tr->mxtips);

        if(rrf <= 0.01) /* 1% cutoff */
        {
          printBothOpen("ML fast search converged at fast SPR cycle %d with stopping criterion\n", fastIterations);
          printBothOpen("Relative Robinson-Foulds (RF) distance between respective best trees after one succseful SPR cycle: %f%s\n", rrf, "%");
          cleanupHashTable(tr->h, 0);
          cleanupHashTable(tr->h, 1);
          goto cleanup_fast;
        }
        else		    
          printBothOpen("ML search convergence criterion fast cycle %d->%d Relative Robinson-Foulds %f\n", fastIterations - 1, fastIterations, rrf);
      }
    }


    fastIterations++;	


    treeEvaluate(tr, 1.0);  


    saveBestTree(bestT, tr);           

    printLog(tr, adef, FALSE);         
    printResult(tr, adef, FALSE);    

    lh = previousLh = tr->likelihood;

    treeOptimizeRapid(tr, 1, bestTrav, adef, bt);   

    impr = 0;

    for(i = 1; i <= bt->nvalid; i++)
    {	    		  	   
      recallBestTree(bt, i, tr);

      treeEvaluate(tr, 0.25);



      difference = ((tr->likelihood > previousLh)? 
          tr->likelihood - previousLh: 
          previousLh - tr->likelihood); 	    
      if(tr->likelihood > lh && difference > epsilon)
      {
        impr = 1;	       
        lh = tr->likelihood;	       	     
        saveBestTree(bestT, tr);

      }	   	   
    }
#ifdef _DEBUG_CHECKPOINTING
    printBothOpen("FAST LH: %f\n", lh);
#endif


  }

  if(tr->searchConvergenceCriterion)
  {
    cleanupHashTable(tr->h, 0);
    cleanupHashTable(tr->h, 1);
  }

cleanup_fast:  
  Thorough = 1;
  impr = 1;

  recallBestTree(bestT, 1, tr); 

  {
    evaluateGenericInitrav(tr, tr->start);
#ifdef _DEBUG_CHECKPOINTING
    printBothOpen("After Fast SPRs Final %f\n", tr->likelihood);   
#endif
  }


  if(estimateModel)
    modOpt(tr, adef, FALSE, 1.0, FALSE);
  else
    treeEvaluate(tr, 1.0);

  while(1)
  {	 
START_SLOW_SPRS:
    if(adef->useCheckpoint && ckp.state == SLOW_SPRS)
    {
      optimizeRateCategoryInvocations = ckp.optimizeRateCategoryInvocations;   




      impr = ckp.impr;
      Thorough = ckp.Thorough;
      bestTrav = ckp.bestTrav;
      treeVectorLength = ckp.treeVectorLength;
      rearrangementsMax = ckp.rearrangementsMax;
      rearrangementsMin = ckp.rearrangementsMin;
      thoroughIterations = ckp.thoroughIterations;
      fastIterations = ckp.fastIterations;


      lh = ckp.lh;
      previousLh = ckp.previousLh;
      difference = ckp.difference;
      epsilon    = ckp.epsilon;                    


      tr->likelihood = ckp.tr_likelihood;

      tr->lhCutoff = ckp.tr_lhCutoff;
      tr->lhAVG    = ckp.tr_lhAVG;
      tr->lhDEC    = ckp.tr_lhDEC;   	 
      tr->itCount = ckp.tr_itCount;
      tr->doCutoff = ckp.tr_doCutoff;

      adef->useCheckpoint = FALSE;
    }
    else
      recallBestTree(bestT, 1, tr);

    {              
      ckp.state = SLOW_SPRS;  
      ckp.optimizeRateCategoryInvocations = optimizeRateCategoryInvocations;              


      ckp.impr = impr;
      ckp.Thorough = Thorough;
      ckp.bestTrav = bestTrav;
      ckp.treeVectorLength = treeVectorLength;
      ckp.rearrangementsMax = rearrangementsMax;
      ckp.rearrangementsMin = rearrangementsMin;
      ckp.thoroughIterations = thoroughIterations;
      ckp.fastIterations = fastIterations;


      ckp.lh = lh;
      ckp.previousLh = previousLh;
      ckp.difference = difference;
      ckp.epsilon    = epsilon; 


      ckp.bestTrav = bestTrav;       
      ckp.impr = impr;

      ckp.tr_startLH  = tr->startLH;
      ckp.tr_endLH    = tr->endLH;
      ckp.tr_likelihood = tr->likelihood;
      ckp.tr_bestOfNode = tr->bestOfNode;

      ckp.tr_lhCutoff = tr->lhCutoff;
      ckp.tr_lhAVG    = tr->lhAVG;
      ckp.tr_lhDEC    = tr->lhDEC;     
      ckp.tr_itCount  = tr->itCount;
      ckp.tr_doCutoff = tr->doCutoff;

      writeCheckpoint(tr); 
    }

    if(impr)
    {	    
      printResult(tr, adef, FALSE);
      rearrangementsMin = 1;
      rearrangementsMax = adef->stepwidth;	

      if(tr->searchConvergenceCriterion)
      {
        int bCounter = 0;	      

        if(thoroughIterations > 1)
          cleanupHashTable(tr->h, (thoroughIterations % 2));		

        bitVectorInitravSpecial(tr->bitVectors, tr->nodep[1]->back, tr->mxtips, tr->vLength, tr->h, thoroughIterations % 2, BIPARTITIONS_RF, (branchInfo *)NULL,
            &bCounter, 1, FALSE, FALSE);	    


        {
          char 
            *buffer = (char*)calloc(tr->treeStringLength, sizeof(char));

#ifdef _DEBUG_CHECKPOINTING		
          printf("Storing tree in slot %d\n", thoroughIterations % 2);
#endif

          Tree2String(buffer, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, FALSE, adef, SUMMARIZE_LH, FALSE, FALSE);

          if(thoroughIterations % 2 == 0)	      
            memcpy(tr->tree0, buffer, tr->treeStringLength * sizeof(char));
          else
            memcpy(tr->tree1, buffer, tr->treeStringLength * sizeof(char));	    

          free(buffer);
        }

        assert(bCounter == tr->mxtips - 3);

        if(thoroughIterations > 0)
        {
          double rrf = convergenceCriterion(tr->h, tr->mxtips);

          if(rrf <= 0.01) /* 1% cutoff */
          {
            printBothOpen("ML search converged at thorough SPR cycle %d with stopping criterion\n", thoroughIterations);
            printBothOpen("Relative Robinson-Foulds (RF) distance between respective best trees after one succseful SPR cycle: %f%s\n", rrf, "%");
            goto cleanup;
          }
          else		    
            printBothOpen("ML search convergence criterion thorough cycle %d->%d Relative Robinson-Foulds %f\n", thoroughIterations - 1, thoroughIterations, rrf);
        }
      }



      thoroughIterations++;	  
    }			  			
    else
    {		       	   
      rearrangementsMax += adef->stepwidth;
      rearrangementsMin += adef->stepwidth; 	        	      
      if(rearrangementsMax > adef->max_rearrange)	     	     	 
        goto cleanup; 	   
    }
    treeEvaluate(tr, 1.0);

    previousLh = lh = tr->likelihood;	      
    saveBestTree(bestT, tr);     
    printLog(tr, adef, FALSE);
    treeOptimizeRapid(tr, rearrangementsMin, rearrangementsMax, adef, bt);

    impr = 0;			      		            

    for(i = 1; i <= bt->nvalid; i++)
    {		 
      recallBestTree(bt, i, tr);	 	    	    	

      treeEvaluate(tr, 0.25);	    	 

      difference = ((tr->likelihood > previousLh)? 
          tr->likelihood - previousLh: 
          previousLh - tr->likelihood); 	    
      if(tr->likelihood > lh && difference > epsilon)
      {
        impr = 1;	       
        lh = tr->likelihood;	  	     
        saveBestTree(bestT, tr);
      }	   	   
    }  

#ifdef _DEBUG_CHECKPOINTING
    printBothOpen("SLOW LH: %f\n", lh);              
#endif
  }

cleanup: 

  {
    evaluateGenericInitrav(tr, tr->start);

#ifdef _DEBUG_CHECKPOINTING
    printBothOpen("After SLOW SPRs Final %f\n", tr->likelihood);   
#endif
  }

  if(tr->searchConvergenceCriterion)
  {
    freeBitVectors(tr->bitVectors, 2 * tr->mxtips);
    free(tr->bitVectors);
    freeHashTable(tr->h);
    free(tr->h);
  }

  freeBestTree(bestT);
  free(bestT);
  freeBestTree(bt);
  free(bt);
  freeInfoList();  
  printLog(tr, adef, FALSE);
  printResult(tr, adef, TRUE);



}




boolean treeEvaluate (tree *tr, double smoothFactor)       /* Evaluate a user tree */
{
  boolean result;

  result = smoothTree(tr, (int)((double)smoothings * smoothFactor));

  assert(result); 

  evaluateGeneric(tr, tr->start);   


  return TRUE;
}

static void setupBranches(tree *tr, nodeptr p,  branchInfo *bInf)
{
  int    
    countBranches = tr->branchCounter;

  if(isTip(p->number, tr->mxtips))    
  {      
    p->bInf       = &bInf[countBranches];
    p->back->bInf = &bInf[countBranches];               	      

    bInf[countBranches].oP = p;
    bInf[countBranches].oQ = p->back;

    tr->branchCounter =  tr->branchCounter + 1;
    return;
  }
  else
  {
    nodeptr q;
    assert(p == p->next->next->next);

    p->bInf       = &bInf[countBranches];
    p->back->bInf = &bInf[countBranches];

    bInf[countBranches].oP = p;
    bInf[countBranches].oQ = p->back;      

    tr->branchCounter =  tr->branchCounter + 1;      

    q = p->next;

    while(q != p)
    {
      setupBranches(tr, q->back, bInf);	
      q = q->next;
    }

    return;
  }
}



