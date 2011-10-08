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
#include <stdint.h>
#include <limits.h>
#include <mpi.h>
#include "axml.h"

extern const unsigned int mask32[32];

extern int processes;
extern int processID;
extern double *globalResult;

/* after some testing this seems to be optimal */

#define TRAVERSAL_LENGTH 5

#define fixedSize     sizeof(jobDescr)
#define traversalSize sizeof(traversalInfo)
#define messageSize   (fixedSize + TRAVERSAL_LENGTH * traversalSize)

char broadcastBuffer[messageSize];

static void sendMergedMessage(jobDescr *job, tree *tr)
{  
  /* alloc local or global, if local: heap or stack???? */
  
  /*char broadcastBuffer[messageSize];*/
  
  memcpy(broadcastBuffer, job, fixedSize);
  
  memcpy(&broadcastBuffer[fixedSize], &(tr->td[0].ti[0]), traversalSize * MIN(job->length, TRAVERSAL_LENGTH));
      
  MPI_Bcast(broadcastBuffer, messageSize, MPI_BYTE, 0, MPI_COMM_WORLD); 
}

static void receiveMergedMessage(jobDescr *job, tree *tr)
{    
  /*char broadcastBuffer[messageSize];            */
  
  MPI_Bcast(broadcastBuffer, messageSize, MPI_BYTE, 0, MPI_COMM_WORLD);
  
  memcpy(job, broadcastBuffer, fixedSize);
  
  memcpy(&(tr->td[0].ti[0]), &broadcastBuffer[fixedSize], traversalSize * MIN(job->length, TRAVERSAL_LENGTH));       
}

static void sendTraversalDescriptor(jobDescr *job, tree *tr)
{            
  size_t    
    length        = job->length - TRAVERSAL_LENGTH,   
    myMessageSize   = length * traversalSize;

  MPI_Bcast((char*)(&(tr->td[0].ti[TRAVERSAL_LENGTH])), myMessageSize, MPI_BYTE, 0, MPI_COMM_WORLD);
}

static void receiveTraversalDescriptor(jobDescr *job, tree *tr)
{

  size_t 
    length        = job->length - TRAVERSAL_LENGTH,      
    myMessageSize   = length * traversalSize;   

  MPI_Bcast((char*)(&(tr->td[0].ti[TRAVERSAL_LENGTH])), myMessageSize, MPI_BYTE, 0, MPI_COMM_WORLD); 
}




static void setupLocalStuff(tree *localTree)
{
  size_t
    model,
    j,
    i,
    globalCounter = 0,
    localCounter  = 0,
    offset,
    countOffset,
    myLength = 0;
    
  if(localTree->manyPartitions)
    for(model = 0, offset = 0, countOffset = 0; model < (size_t)localTree->NumberOfModels; model++)
      {            
	if(localTree->partitionData[model].width > 0)
	  {
	    localTree->partitionData[model].sumBuffer    = &localTree->sumBuffer[offset];      
	    localTree->partitionData[model].perSiteLL    = &localTree->perSiteLLPtr[countOffset];          
	    
	    localTree->partitionData[model].wgt          = &localTree->wgtPtr[countOffset];      
	    localTree->partitionData[model].rateCategory = &localTree->rateCategoryPtr[countOffset];     
	    
	    countOffset += localTree->partitionData[model].width;
	    
	    offset += (size_t)(localTree->discreteRateCategories) * (size_t)(localTree->partitionData[model].states) * (size_t)(localTree->partitionData[model].width);      
	  }
      }
  else
    for(model = 0, offset = 0, countOffset = 0; model < (size_t)localTree->NumberOfModels; model++)
      {            
	localTree->partitionData[model].sumBuffer    = &localTree->sumBuffer[offset];      
	localTree->partitionData[model].perSiteLL    = &localTree->perSiteLLPtr[countOffset];          
	
	localTree->partitionData[model].wgt          = &localTree->wgtPtr[countOffset];      
	localTree->partitionData[model].rateCategory = &localTree->rateCategoryPtr[countOffset];     
	
	countOffset += localTree->partitionData[model].width;
	
	offset += (size_t)(localTree->discreteRateCategories) * (size_t)(localTree->partitionData[model].states) * (size_t)(localTree->partitionData[model].width);      
      }


  myLength           = countOffset;

  if(localTree->manyPartitions)
    for(i = 0; i < (size_t)localTree->mxtips; i++)
      {
	for(model = 0, offset = 0, countOffset = 0; model < (size_t)localTree->NumberOfModels; model++)
	  {
	    if(localTree->partitionData[model].width > 0)
	      {
		localTree->partitionData[model].yVector[i+1]   = &localTree->y_ptr[i * myLength + countOffset];
		countOffset +=  localTree->partitionData[model].width;
	      }
	  }
	assert(countOffset == myLength);
      }
  else
    for(i = 0; i < (size_t)localTree->mxtips; i++)
      {
	for(model = 0, offset = 0, countOffset = 0; model < (size_t)localTree->NumberOfModels; model++)
	  {
	    localTree->partitionData[model].yVector[i+1]   = &localTree->y_ptr[i * myLength + countOffset];
	    countOffset +=  localTree->partitionData[model].width;
	}
	assert(countOffset == myLength);
      }
}

static void allocLikelihoodVectors(tree *tr)
{
  size_t
    i,    
    model; 
  
  /* 
     for(model = 0; model < (size_t)tr->NumberOfModels; model++)    
     printf("%d %d %d\n", processID, model, tr->partitionData[model].width);
  */

  for(i = 0; i < (size_t)tr->innerNodes; i++)    
    for(model = 0; model < (size_t)tr->NumberOfModels; model++)       
      tr->partitionData[model].xVector[i]   = (double*)NULL;       
}

static void memSaveInit(tree *tr)
{
  if(tr->saveMemory)
    {
      int model;
     
      for(model = 0; model < tr->NumberOfModels; model++)
	{
	  int 
	    i,
	    j,
	    undetermined = getUndetermined(tr->partitionData[model].dataType);
      
	  size_t
	    width =  tr->partitionData[model].width;

	  if(width > 0)
	    {
	      tr->partitionData[model].gapVectorLength = ((int)width / 32) + 1;
      
	      memset(tr->partitionData[model].gapVector, 0, tr->partitionData[model].initialGapVectorSize);

	      for(j = 1; j <= (size_t)(tr->mxtips); j++)
		for(i = 0; i < width; i++)
		  if(tr->partitionData[model].yVector[j][i] == undetermined)
		    tr->partitionData[model].gapVector[tr->partitionData[model].gapVectorLength * j + i / 32] |= mask32[i % 32];
	    }
	  else
	    {
	      tr->partitionData[model].gapVectorLength = 0;
	    }
	}
    }
}

static int sendBufferSizeInt(int numberOfModels)
{
  int
    size = 11;

  size += (size_t)numberOfModels * 9;

  return size;
}



void fineGrainWorker(tree *tr)
{
  int 
    sendBufferLength = 0,
    totalLength = 0,
    model,
    threadID,
    dataCounter,
    NumberOfThreads = processes,
    *sendBufferInt;

  double        
    *dlnLdlz,
    *d2lnLdlz2,
    *partialResult,
    *dummy;

  jobDescr
    job;



  MPI_Bcast(&(tr->NumberOfModels), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(tr->manyPartitions), 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(tr->manyPartitions)
    {
      tr->partitionAssignment = (int *)malloc(tr->NumberOfModels * sizeof(int));
      MPI_Bcast(tr->partitionAssignment, tr->NumberOfModels, MPI_INT, 0, MPI_COMM_WORLD);
    }

  

  sendBufferLength = sendBufferSizeInt(tr->NumberOfModels);
  sendBufferInt = (int*)malloc(sizeof(int) * sendBufferLength);

  threadID = processID;

  MPI_Bcast(&dataCounter, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(sendBufferInt, dataCounter, MPI_INT, 0, MPI_COMM_WORLD);
  dataCounter = 0;

  tr->saveMemory = sendBufferInt[dataCounter++];
  tr->useGappedImplementation = sendBufferInt[dataCounter++];
  tr->innerNodes = sendBufferInt[dataCounter++];
  tr->maxCategories = sendBufferInt[dataCounter++];
  tr->originalCrunchedLength = sendBufferInt[dataCounter++];  
  tr->mxtips = sendBufferInt[dataCounter++];
  tr->multiBranch = sendBufferInt[dataCounter++];
  tr->multiGene = sendBufferInt[dataCounter++];
  tr->numBranches = sendBufferInt[dataCounter++]; 
  tr->discreteRateCategories= sendBufferInt[dataCounter++]; 
  tr->rateHetModel = sendBufferInt[dataCounter++];
  
  tr->lhs                     = (double*)malloc(sizeof(double)   * tr->originalCrunchedLength);
  tr->executeModel            = (boolean*)malloc(sizeof(boolean) * tr->NumberOfModels);
  tr->perPartitionLH          = (double*)malloc(sizeof(double)   * tr->NumberOfModels);
  tr->storedPerPartitionLH    = (double*)malloc(sizeof(double)   * tr->NumberOfModels);

  tr->fracchanges = (double*)malloc(sizeof(double)   * tr->NumberOfModels);
  tr->partitionContributions = (double*)malloc(sizeof(double)   * tr->NumberOfModels);

  tr->partitionData = (pInfo*)malloc(sizeof(pInfo) * tr->NumberOfModels);

  tr->td[0].count = 0;
  tr->td[0].ti    = (traversalInfo *)malloc(sizeof(traversalInfo) * tr->mxtips);

  tr->cdta               = (cruncheddata*)malloc(sizeof(cruncheddata));
  tr->cdta->patrat       = (double*)malloc(sizeof(double) * tr->originalCrunchedLength);
  tr->cdta->patratStored = (double*)malloc(sizeof(double) * tr->originalCrunchedLength); 

  for(model = 0; model < tr->NumberOfModels; model++)
    {
      tr->partitionData[model].states       = sendBufferInt[dataCounter++];
      tr->partitionData[model].maxTipStates = sendBufferInt[dataCounter++];
      tr->partitionData[model].dataType     = sendBufferInt[dataCounter++];
      tr->partitionData[model].protModels   = sendBufferInt[dataCounter++];
      tr->partitionData[model].protFreqs    = sendBufferInt[dataCounter++];
      tr->partitionData[model].mxtips = sendBufferInt[dataCounter++];
      tr->partitionData[model].lower = sendBufferInt[dataCounter++];
      tr->partitionData[model].upper = sendBufferInt[dataCounter++];
      tr->partitionData[model].numberOfCategories = sendBufferInt[dataCounter++];
      tr->executeModel[model] = TRUE;
      tr->perPartitionLH[model] = 0.0;
      tr->storedPerPartitionLH[model] = 0.0;
      totalLength += (tr->partitionData[model].upper -  tr->partitionData[model].lower);
    }   

  assert(totalLength == tr->originalCrunchedLength);   

  allocNodex(tr, threadID, processes);

  setupLocalStuff(tr);

  {
    size_t
      model, 
      globalCounter, 
      localCounter, 
      i,
      j;

    int 
      *wgtBuf = (int *)malloc(sizeof(int) * tr->originalCrunchedLength),
      *catBuf = (int *)malloc(sizeof(int) * tr->originalCrunchedLength);
    
    unsigned char 
      *yBuffer = (unsigned char*)malloc(sizeof(unsigned char) * tr->originalCrunchedLength);
    
    MPI_Bcast(wgtBuf, tr->originalCrunchedLength, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(catBuf, tr->originalCrunchedLength, MPI_INT, 0, MPI_COMM_WORLD);

    if(tr->manyPartitions)
      for(model = 0, globalCounter = 0; model < (size_t)tr->NumberOfModels; model++)
	{
	  const boolean
	    mine = isThisMyPartition(tr, threadID, model, processes);
	  
	  size_t
	    width = (size_t)tr->partitionData[model].upper - (size_t)tr->partitionData[model].lower;
	  
	  if(mine)
	    {
	      memcpy(tr->partitionData[model].wgt,          &wgtBuf[globalCounter], sizeof(int) * width);
	      memcpy(tr->partitionData[model].rateCategory, &catBuf[globalCounter], sizeof(int) * width);
	    }
	  
	  globalCounter += width;		
	}      
    else
      for(model = 0, globalCounter = 0; model < (size_t)tr->NumberOfModels; model++)
	{
	  for(localCounter = 0, i = (size_t)tr->partitionData[model].lower;  i < (size_t)tr->partitionData[model].upper; i++)
	    {
	      if(i % (size_t)processes == (size_t)threadID)
		{
		  tr->partitionData[model].wgt[localCounter]          = wgtBuf[globalCounter];	      		
		  tr->partitionData[model].rateCategory[localCounter] = catBuf[globalCounter];	      			     
		  
		  localCounter++;
		}
	      globalCounter++;
	    }
	}      

    free(wgtBuf);
    free(catBuf);  

    for(j = 1; j <= (size_t)tr->mxtips; j++)
      {	      
	MPI_Bcast(yBuffer, tr->originalCrunchedLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);       
	
	if(tr->manyPartitions)	  
	  {
	    for(model = 0, globalCounter = 0; model < (size_t)tr->NumberOfModels; model++)	  
	      {
		const boolean
		  mine = isThisMyPartition(tr, threadID, model, processes);
		size_t
		  width = (size_t)tr->partitionData[model].upper - (size_t)tr->partitionData[model].lower;
		
		if(mine)	    
		  memcpy(tr->partitionData[model].yVector[j], &yBuffer[globalCounter], sizeof(unsigned char) * width);
		
		
		globalCounter += width;		   
	      }
	    assert(globalCounter == tr->originalCrunchedLength);
	  }
	else
	  for(model = 0, globalCounter = 0; model < (size_t)tr->NumberOfModels; model++)	  
	    {
	      for(localCounter = 0, i = (size_t)tr->partitionData[model].lower;  i < (size_t)tr->partitionData[model].upper; i++)	      
		{
		  if(i % (size_t)processes == (size_t)threadID)		  
		    {		    
		      tr->partitionData[model].yVector[j][localCounter] = yBuffer[globalCounter];      	            	
		      localCounter++;
		    }
		  globalCounter++;
		}
	    }
      }
    
    free(yBuffer);

  }

  MPI_Barrier(MPI_COMM_WORLD);

  allocLikelihoodVectors(tr);

  memSaveInit(tr);
  
  dlnLdlz       = (double*)malloc(sizeof(double) * tr->NumberOfModels);
  d2lnLdlz2     = (double*)malloc(sizeof(double) * tr->NumberOfModels);
  partialResult = (double*)malloc(sizeof(double) * tr->NumberOfModels * 2);
  dummy         = (double*)malloc(sizeof(double) * tr->NumberOfModels * 2); 

  while(1)
    {
      int 
	jobType,
	tid = processID,
	n = processes,
	i,   
	model,
	localCounter,
	globalCounter;              

      receiveMergedMessage(&job, tr);

      jobType = job.jobType;

      switch(jobType)
	{
	 case THREAD_COPY_RATE_CATS:
	   {
	     double
	       *b1 = (double *)malloc(sizeof(double) * tr->originalCrunchedLength),
	       *b2 = (double *)malloc(sizeof(double) * tr->originalCrunchedLength);

	     int 
	       *b_int = (int *)malloc(sizeof(int) * tr->originalCrunchedLength);
	     	    
	     MPI_Bcast(tr->cdta->patrat,         tr->originalCrunchedLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	     MPI_Bcast(tr->cdta->patratStored,   tr->originalCrunchedLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	     
	     for(model = 0; model < tr->NumberOfModels; model++)
	       {
		 MPI_Bcast(&(tr->partitionData[model].numberOfCategories), 1, MPI_INT, 0, MPI_COMM_WORLD);
		 MPI_Bcast(tr->partitionData[model].perSiteRates, tr->partitionData[model].numberOfCategories, MPI_DOUBLE, 0, MPI_COMM_WORLD);		 
	       }

	     MPI_Bcast(b_int, tr->originalCrunchedLength, MPI_INT,    0, MPI_COMM_WORLD);
	     MPI_Bcast(b1,    tr->originalCrunchedLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	     MPI_Bcast(b2,    tr->originalCrunchedLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	     for(model = 0; model < tr->NumberOfModels; model++)
	       {
		 if(tr->manyPartitions)
		   {
		     if(isThisMyPartition(tr, threadID, model, processes))
		       {
			 size_t 
			   start = (size_t)tr->partitionData[model].lower,
			   width = (size_t)tr->partitionData[model].upper - (size_t)tr->partitionData[model].lower;
			 
			 memcpy(tr->partitionData[model].rateCategory, &b_int[start], sizeof(int) * width);
			 memcpy(tr->partitionData[model].wr,           &b1[start],    sizeof(double) * width);
			 memcpy(tr->partitionData[model].wr2,          &b2[start],    sizeof(double) * width);		      
		       }
		   }
		 else		 
		   for(localCounter = 0, i = tr->partitionData[model].lower;  i < tr->partitionData[model].upper; i++)
		     {
		       if(i % n == tid)
			 {		 
			   tr->partitionData[model].rateCategory[localCounter] = b_int[i];
			   tr->partitionData[model].wr[localCounter]             = b1[i];
			   tr->partitionData[model].wr2[localCounter]            = b2[i];		 
			   
			   localCounter++;
			 }
		     }
	       } 

	     MPI_Barrier(MPI_COMM_WORLD);

	     free(b1);
	     free(b2);
	     free(b_int);
	   }
	   break; 
	case THREAD_COPY_INIT_MODEL:      	
     	 
	  for(model = 0; model < tr->NumberOfModels; model++)
	    {
	      const partitionLengths 
		*pl = getPartitionLengths(&(tr->partitionData[model]));
	      
	      MPI_Bcast(&(tr->partitionData[model].numberOfCategories), 1, MPI_INT, 0, MPI_COMM_WORLD);
	      
	      MPI_Bcast(tr->partitionData[model].EIGN,        pl->eignLength,        MPI_DOUBLE, 0, MPI_COMM_WORLD);
	      MPI_Bcast(tr->partitionData[model].EV,          pl->evLength,          MPI_DOUBLE, 0, MPI_COMM_WORLD);
	      MPI_Bcast(tr->partitionData[model].EI,          pl->eiLength,          MPI_DOUBLE, 0, MPI_COMM_WORLD);
	      MPI_Bcast(tr->partitionData[model].substRates,  pl->substRatesLength,  MPI_DOUBLE, 0, MPI_COMM_WORLD);	  
	      MPI_Bcast(tr->partitionData[model].frequencies, pl->frequenciesLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	      MPI_Bcast(tr->partitionData[model].tipVector,   pl->tipVectorLength,   MPI_DOUBLE, 0, MPI_COMM_WORLD);	 	     	 
	      
	      MPI_Bcast(&tr->partitionData[model].lower, 1  , MPI_INT, 0, MPI_COMM_WORLD);
	      MPI_Bcast(&tr->partitionData[model].upper, 1  , MPI_INT, 0, MPI_COMM_WORLD);
	      MPI_Bcast(&tr->partitionData[model].alpha, 1  , MPI_DOUBLE, 0, MPI_COMM_WORLD);
	      makeGammaCats(tr->partitionData[model].alpha, tr->partitionData[model].gammaRates, 4);
	    }
	  
	  MPI_Bcast(tr->cdta->patrat,       tr->originalCrunchedLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	  MPI_Bcast(tr->cdta->patratStored, tr->originalCrunchedLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	  
	  MPI_Barrier(MPI_COMM_WORLD);  
	  break;
	case THREAD_NEWVIEW:     	    
	  tr->td[0].count = job.length;
	  
	  if(job.length >= TRAVERSAL_LENGTH)
	    receiveTraversalDescriptor(&job, tr);
	    
	  newviewIterative(tr);	    	   
	
	  break;
	case THREAD_EVALUATE:
	  {
	    int    
	      model,
	      length = tr->td[0].count;
	
	    double	    
	      result;
              
	    length = job.length;
	    tr->td[0].count = length;

	    if(job.length >= TRAVERSAL_LENGTH)
	      receiveTraversalDescriptor(&job, tr);
	     	    	
	    evaluateIterative(tr, FALSE);	   
	
	    for(model = 0; model < tr->NumberOfModels; model++)
	      {
		partialResult[model]    = tr->perPartitionLH[model];
		tr->executeModel[model] = TRUE;
	      }

	    MPI_Reduce(partialResult, dummy, tr->NumberOfModels, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);	 	   
	  }
	  break;
	case THREAD_MAKENEWZ_FIRST:
	  {
	    int 
	      length,	      
	      model;		    
      	
	    length = job.length;
	    tr->td[0].count = length;

	    if(tr->multiBranch)
	      for(model = 0; model < tr->NumberOfModels; model++)
		{
		  tr->coreLZ[model]       = job.coreLZ[model];		
		  tr->executeModel[model] = job.executeModel[model];
		}
	    else
	      {
		tr->coreLZ[0]       = job.coreLZ[0];		
		tr->executeModel[0] = job.executeModel[0];
	      }

	    if(job.length >= TRAVERSAL_LENGTH)	     
	      receiveTraversalDescriptor(&job, tr);
      
	    makenewzIterative(tr);	
	    execCore(tr, dlnLdlz, d2lnLdlz2);
		   	 	
	    for(model = 0; model < tr->NumberOfModels; model++)
	      tr->executeModel[model] = TRUE;


	    if(tr->multiBranch)
	      {
		for(model = 0; model < tr->NumberOfModels; model++)
		  {		
		    partialResult[model * 2 + 0] = dlnLdlz[model];
		    partialResult[model * 2 + 1] = d2lnLdlz2[model];				
		  }
	      }
	    else
	      {
		partialResult[0] = dlnLdlz[0];
		partialResult[1] = d2lnLdlz2[0];
	      }


	    if(tr->multiBranch)
	      MPI_Reduce(partialResult, dummy, tr->NumberOfModels * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);			    
	    else
	      MPI_Reduce(partialResult, dummy, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	  }
	  break; 
	case THREAD_MAKENEWZ:
	  {
	    int 
	      model;
	    	   
	    if(tr->multiBranch)		    	
	      for(model = 0; model < tr->NumberOfModels; model++)
		{
		  tr->coreLZ[model]       = job.coreLZ[model];
		  tr->executeModel[model] = job.executeModel[model];
		}
	    else	   
	      {
		tr->coreLZ[0]       = job.coreLZ[0];		
		tr->executeModel[0] = job.executeModel[0];
	      }
	
	    execCore(tr, dlnLdlz, d2lnLdlz2);
	    	    
	    for(model = 0; model < tr->NumberOfModels; model++)
	      tr->executeModel[model] = TRUE;

	    if(tr->multiBranch)
	      for(model = 0; model < tr->NumberOfModels; model++)
		{
		  partialResult[model * 2 + 0] = dlnLdlz[model];
		  partialResult[model * 2 + 1] = d2lnLdlz2[model];		  
		}
	    else
	      {
		partialResult[0] = dlnLdlz[0];
		partialResult[1] = d2lnLdlz2[0];
	      }
	   
	    if(tr->multiBranch)	       
	      MPI_Reduce(partialResult, dummy, tr->NumberOfModels * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);	   	    
	    else
	      MPI_Reduce(partialResult, dummy, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);	     
	  }
	  break;
	case THREAD_OPT_RATE:
	  {	
	    double 	     
	      *buffer;

	    int
	      bufIndex = 0,
	      length = job.length,
	      model;

	    buffer = (double*)malloc(sizeof(double) * length);

	    MPI_Bcast(buffer, length, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
	
	    for(model = 0; model < tr->NumberOfModels; model++)	  	    
	      {
		const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));
		
		memcpy(tr->partitionData[model].EIGN,      &buffer[bufIndex],   pl->eignLength * sizeof(double));
		bufIndex += pl->eignLength;
		
		memcpy(tr->partitionData[model].EV,        &buffer[bufIndex],   pl->evLength * sizeof(double));		  
		bufIndex += pl->evLength;
		
		memcpy(tr->partitionData[model].EI,        &buffer[bufIndex],   pl->eiLength * sizeof(double));
		bufIndex += pl->eiLength;
		
		memcpy(tr->partitionData[model].tipVector, &buffer[bufIndex],   pl->tipVectorLength * sizeof(double));
		bufIndex += pl->tipVectorLength;
		
		tr->executeModel[model] = job.executeModel[model];
	      }	      			

	    evaluateIterative(tr, FALSE);
	     
	    for(model = 0; model < tr->NumberOfModels; model++)
	      partialResult[model] = tr->perPartitionLH[model];
	    
	    MPI_Reduce(partialResult, dummy, tr->NumberOfModels, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 
	    for(model = 0; model < tr->NumberOfModels; model++)
	      tr->executeModel[model] = TRUE;	  
	
	    free(buffer);
	  }
	  break;
	case THREAD_COPY_RATES:
	case THREAD_BROADCAST_RATE:
	  {	
	    double 	     
	      *buffer;

	    int
	      bufIndex = 0,
	      length = job.length,
	      model;

	    buffer = (double*)malloc(sizeof(double) * length);

	    MPI_Bcast(buffer, length, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
	
	    for(model = 0; model < tr->NumberOfModels; model++)	  	    
	      {
		const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));
		
		memcpy(tr->partitionData[model].EIGN,      &buffer[bufIndex],   pl->eignLength * sizeof(double));
		bufIndex += pl->eignLength;
		
		memcpy(tr->partitionData[model].EV,        &buffer[bufIndex],   pl->evLength * sizeof(double));		  
		bufIndex += pl->evLength;
		
		memcpy(tr->partitionData[model].EI,        &buffer[bufIndex],   pl->eiLength * sizeof(double));
		bufIndex += pl->eiLength;
		
		memcpy(tr->partitionData[model].tipVector, &buffer[bufIndex],   pl->tipVectorLength * sizeof(double));
		bufIndex += pl->tipVectorLength;
		
		if(jobType == THREAD_BROADCAST_RATE)
		  tr->executeModel[model] = job.executeModel[model];
	      }	      				   	  
	
	    free(buffer);
	  }
	  break;
	case THREAD_RATE_CATS:
	  { 
	    int 
	      model,
	      i,
	      localCounter,
	      sendBufferSize;
	    
	    double 
	      *localDummy = (double*)NULL,
	      *patBufSend,
	      *patStoredBufSend,
	      *lhsBufSend;
	   
	    if(tr->manyPartitions)
	      sendBufferSize = tr->originalCrunchedLength;
	    else
	      sendBufferSize = (tr->originalCrunchedLength / n) + 1;

	    patBufSend = (double *)malloc(sendBufferSize * sizeof(double));
	    patStoredBufSend =  (double *)malloc(sendBufferSize * sizeof(double));
	    lhsBufSend = (double *)malloc(sendBufferSize * sizeof(double));
 
	    tr->lower_spacing = job.lower_spacing;
	    tr->upper_spacing = job.upper_spacing;
	    
	    if(job.length >= TRAVERSAL_LENGTH)	  
	      receiveTraversalDescriptor(&job, tr);
	    
	    tr->td[0].count = job.length;

	    optRateCatPthreads(tr, tr->lower_spacing, tr->upper_spacing, tr->lhs, n, tid);
    
	    for(model = 0, localCounter = 0; model < tr->NumberOfModels; model++)
	      {               
	
		if(tr->manyPartitions)
		  {
		    size_t
		      start = (size_t)tr->partitionData[model].lower,
		      width = (size_t)tr->partitionData[model].upper - (size_t)tr->partitionData[model].lower;
		    
		    if(isThisMyPartition(tr, tid, model, n))
		      {
			memcpy(&patBufSend[start],       &tr->cdta->patrat[start],       sizeof(double) * width);
			memcpy(&patStoredBufSend[start], &tr->cdta->patratStored[start], sizeof(double) * width);
			memcpy(&lhsBufSend[start],       &tr->lhs[start],                sizeof(double) * width);
		      }		 
		  }
		else
		  {
		    for(i = tr->partitionData[model].lower;  i < tr->partitionData[model].upper; i++)
		      if(i % n == tid)
			{
			  patBufSend[localCounter] = tr->cdta->patrat[i];
			  patStoredBufSend[localCounter] = tr->cdta->patratStored[i];
			  lhsBufSend[localCounter] = tr->lhs[i];
			  localCounter++;
			}
		  }
	      }
   
	    MPI_Gather(patBufSend,       sendBufferSize, MPI_DOUBLE, localDummy, sendBufferSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Gather(patStoredBufSend, sendBufferSize, MPI_DOUBLE, localDummy, sendBufferSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Gather(lhsBufSend,       sendBufferSize, MPI_DOUBLE, localDummy, sendBufferSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    
	    free(patBufSend);
	    free(patStoredBufSend);
	    free(lhsBufSend);
	  }
	  break;
	case THREAD_NEWVIEW_MASKED:
	  {
	    tr->td[0].count = job.length;

	    if(job.length >= TRAVERSAL_LENGTH)	   
	      receiveTraversalDescriptor(&job, tr);
	   	 	    
	    for(model = 0; model < tr->NumberOfModels; model++)	  	    	  
	      tr->executeModel[model] = job.executeModel[model];
	    
	    newviewIterative(tr);	    	   
	    
	    for(model = 0; model < tr->NumberOfModels; model++)	
	      tr->executeModel[model] = TRUE;	          
	  }
	  break; 
	case THREAD_OPT_ALPHA:
	  {	
	    double 	     
	      *buffer;

	    int
	      bufIndex = 0,
	      length = job.length,
	      model;

	    buffer = (double*)malloc(sizeof(double) * length);

	    MPI_Bcast(buffer, length, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
	
	    for(model = 0; model < tr->NumberOfModels; model++)	  	    
	      {
		tr->partitionData[model].alpha = buffer[bufIndex++];
		makeGammaCats(tr->partitionData[model].alpha, tr->partitionData[model].gammaRates, 4);				
		tr->executeModel[model] = job.executeModel[model];
	      }	      			

	    evaluateIterative(tr, FALSE);
	     
	    for(model = 0; model < tr->NumberOfModels; model++)
	      partialResult[model] = tr->perPartitionLH[model];
	    
	    MPI_Reduce(partialResult, dummy, tr->NumberOfModels, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 
	    for(model = 0; model < tr->NumberOfModels; model++)
	      tr->executeModel[model] = TRUE;	  
	
	    free(buffer);
	  }
	  break;
	case THREAD_COPY_ALPHA:
	  {
	    double 	     
	      *buffer;

	    int
	      bufIndex = 0,
	      length = job.length,
	      model;

	    buffer = (double*)malloc(sizeof(double) * length);

	    MPI_Bcast(buffer, length, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
	
	    for(model = 0; model < tr->NumberOfModels; model++)	  	    
	      {
		tr->partitionData[model].alpha = buffer[bufIndex++];
		makeGammaCats(tr->partitionData[model].alpha, tr->partitionData[model].gammaRates, 4);					       
	      }	      				    	  
	
	    free(buffer);
	  }
	  break;
	case EXIT_GRACEFULLY:
	  goto endIT;
	default:
	  assert(0);
	}            
    }
 endIT:;
}



void startFineGrainMpi(tree *tr, analdef *adef)
{
  
  
  int
    sendBufferLength = 0,
    *sendBufferInt,
    threadID,
    totalLength = 0,
    dataCounter = 0,
    model = 0,
    NumberOfThreads = processes;     
  
  MPI_Bcast(&(tr->NumberOfModels), 1, MPI_INT, 0, MPI_COMM_WORLD);  
  MPI_Bcast(&(tr->manyPartitions), 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(tr->manyPartitions)
    MPI_Bcast(tr->partitionAssignment, tr->NumberOfModels, MPI_INT, 0, MPI_COMM_WORLD);

  sendBufferLength = sendBufferSizeInt(tr->NumberOfModels);
  sendBufferInt = (int*)malloc(sizeof(int) * sendBufferLength);

  threadID = processID;  

  assert(tr->NumberOfModels == NUM_BRANCHES);

  sendBufferInt[dataCounter++] = tr->saveMemory;
  sendBufferInt[dataCounter++] = tr->useGappedImplementation;
  sendBufferInt[dataCounter++] = tr->innerNodes;
  sendBufferInt[dataCounter++] = tr->maxCategories;
  sendBufferInt[dataCounter++] = tr->originalCrunchedLength; 
  sendBufferInt[dataCounter++] = tr->mxtips;
  sendBufferInt[dataCounter++] = tr->multiBranch;
  sendBufferInt[dataCounter++] = tr->multiGene;
  sendBufferInt[dataCounter++] = tr->numBranches;
  
  sendBufferInt[dataCounter++] = tr->discreteRateCategories; 
  sendBufferInt[dataCounter++] = tr->rateHetModel;
  
  for(model = 0; model < tr->NumberOfModels; model++)
    {
      sendBufferInt[dataCounter++] = tr->partitionData[model].states;
      sendBufferInt[dataCounter++] = tr->partitionData[model].maxTipStates;
      sendBufferInt[dataCounter++] = tr->partitionData[model].dataType;
      sendBufferInt[dataCounter++] = tr->partitionData[model].protModels;
      sendBufferInt[dataCounter++] = tr->partitionData[model].protFreqs;
      sendBufferInt[dataCounter++] = tr->partitionData[model].mxtips;
      sendBufferInt[dataCounter++] = tr->partitionData[model].lower;
      sendBufferInt[dataCounter++] = tr->partitionData[model].upper;      
      sendBufferInt[dataCounter++] = tr->partitionData[model].numberOfCategories;
      totalLength += (tr->partitionData[model].upper -  tr->partitionData[model].lower);

      assert(dataCounter <= sendBufferLength);
    }
  
  assert(totalLength == tr->originalCrunchedLength);
  
  MPI_Bcast(&dataCounter, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(sendBufferInt, dataCounter, MPI_INT, 0, MPI_COMM_WORLD);  
  

  
  allocNodex(tr, threadID, processes);
  setupLocalStuff(tr);
 
  
  {
    size_t
      model, 
      globalCounter, 
      localCounter, 
      i,
      j;

    unsigned char 
      *yBuffer = (unsigned char*)malloc(sizeof(unsigned char) * tr->originalCrunchedLength);
   
    MPI_Bcast(tr->cdta->aliaswgt,     tr->originalCrunchedLength, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tr->cdta->rateCategory, tr->originalCrunchedLength, MPI_INT, 0, MPI_COMM_WORLD);

    for(model = 0, globalCounter = 0; model < (size_t)tr->NumberOfModels; model++)
      {
	if(tr->manyPartitions)
	  {
	    const boolean
	      mine = isThisMyPartition(tr, threadID, model, processes);
	    size_t
	      width = (size_t)tr->partitionData[model].upper - (size_t)tr->partitionData[model].lower;
	    
	    if(mine)
	      {
		memcpy(tr->partitionData[model].wgt,          &tr->cdta->aliaswgt[globalCounter], sizeof(int) * width);
		memcpy(tr->partitionData[model].rateCategory, &tr->cdta->rateCategory[globalCounter], sizeof(int) * width);
	      }
	    
	    globalCounter += width;	
	  }
	else
	  {
	    for(localCounter = 0, i = (size_t)tr->partitionData[model].lower;  i < (size_t)tr->partitionData[model].upper; i++)
	      {
		if(i % (size_t)processes == (size_t)threadID)
		  {
		    tr->partitionData[model].wgt[localCounter]          = tr->cdta->aliaswgt[globalCounter];	      		
		    tr->partitionData[model].rateCategory[localCounter] = tr->cdta->rateCategory[globalCounter];	      			     
		    
		    localCounter++;
		  }
		globalCounter++;
	      }
	  }
      }            	      

    for(j = 1; j <= (size_t)tr->mxtips; j++)
      {
	for(i = 0; i < (size_t)tr->originalCrunchedLength; i++)
	  yBuffer[i] = tr->yVector[j][i];		

	MPI_Bcast(yBuffer, tr->originalCrunchedLength, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);       
	
	if(tr->manyPartitions)
	  {
	    for(model = 0, globalCounter = 0; model < (size_t)tr->NumberOfModels; model++)	 
	      {
		const boolean
		  mine = isThisMyPartition(tr, threadID, model, processes);
		
		size_t
		  width = (size_t)tr->partitionData[model].upper - (size_t)tr->partitionData[model].lower;
		
		if(mine)
		  memcpy(tr->partitionData[model].yVector[j], &yBuffer[globalCounter], sizeof(unsigned char) * width);
		
		globalCounter += width;	   
	      }
	  }
	else
	  {
	    for(model = 0, globalCounter = 0; model < (size_t)tr->NumberOfModels; model++)	 
	      {
		for(localCounter = 0, i = (size_t)tr->partitionData[model].lower;  i < (size_t)tr->partitionData[model].upper; i++)	      
		  {
		    if(i % (size_t)processes == (size_t)threadID)		  
		      {
			tr->partitionData[model].yVector[j][localCounter] = yBuffer[globalCounter];      	            
			localCounter++;
		      }
		    globalCounter++;
		  }
	      }
	  }
      }
    
    free(yBuffer);
  } 

  if(!adef->readBinaryFile)
    baseFrequenciesGTR(tr->rdta, tr->cdta, tr); 

  if(tr->rdta->y0)
    {    
      /*
	printf("Free on compressed alignment data\n");
      */

      free(tr->rdta->y0);
      tr->rdta->y0 = (unsigned char*)NULL;
    }
  
  MPI_Barrier(MPI_COMM_WORLD);

  allocLikelihoodVectors(tr);

  memSaveInit(tr);

  assert(processID == 0);
  
  if(processID == 0)
    globalResult = (double*)malloc(sizeof(double) * 2 * tr->NumberOfModels);

  /*
    printf("Starting comps\n"); 
  */
}

void masterBarrierMPI(int jobType, tree *tr)
{
  int
    tid = processID,
    n = processes,
    i,   
    model,
    localCounter,
    globalCounter;

  double
    *partialResult  = (double*)malloc(sizeof(double) * 2 * tr->NumberOfModels),
    *dlnLdlz        = (double*)malloc(sizeof(double) * tr->NumberOfModels),
    *d2lnLdlz2      = (double*)malloc(sizeof(double) * tr->NumberOfModels);

  jobDescr
    job;

  job.jobType = jobType;

  switch(jobType)
    {
    case THREAD_COPY_RATE_CATS:      
      {	
	job.length = 0;
	sendMergedMessage(&job, tr);
		
	MPI_Bcast(tr->cdta->patrat, tr->originalCrunchedLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(tr->cdta->patratStored, tr->originalCrunchedLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	for(model = 0; model < tr->NumberOfModels; model++)
	  { 
	    MPI_Bcast(&(tr->partitionData[model].numberOfCategories), 1, MPI_INT, 0, MPI_COMM_WORLD);
	    MPI_Bcast(tr->partitionData[model].perSiteRates, tr->partitionData[model].numberOfCategories, MPI_DOUBLE, 0, MPI_COMM_WORLD);	   
	  }

	
	MPI_Bcast(tr->cdta->rateCategory, tr->originalCrunchedLength, MPI_INT,    0, MPI_COMM_WORLD);
	MPI_Bcast(tr->wr,                 tr->originalCrunchedLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(tr->wr2,                tr->originalCrunchedLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	for(model = 0; model < tr->NumberOfModels; model++)
	  {
	    if(tr->manyPartitions)
	      {
		size_t
		  start = (size_t)tr->partitionData[model].lower,
		  width = (size_t)tr->partitionData[model].upper - (size_t)tr->partitionData[model].lower;
		
		if(isThisMyPartition(tr, tid, model, n))
		  {
		    memcpy(tr->partitionData[model].rateCategory, &tr->cdta->rateCategory[start], sizeof(int) * width);
		    memcpy(tr->partitionData[model].wr,           &tr->wr[start], sizeof(double) * width);
		    memcpy(tr->partitionData[model].wr2,          &tr->wr2[start], sizeof(double) * width);
		  }
	      }
	    else
	      {
		for(localCounter = 0, i = tr->partitionData[model].lower;  i < tr->partitionData[model].upper; i++)
		  {
		    if(i % n == tid)
		      {		 
			tr->partitionData[model].rateCategory[localCounter] = tr->cdta->rateCategory[i];
			tr->partitionData[model].wr[localCounter]             = tr->wr[i];
			tr->partitionData[model].wr2[localCounter]            = tr->wr2[i];		 
			
			localCounter++;
		      }
		  }
	      }
	  } 
	MPI_Barrier(MPI_COMM_WORLD);
      }
      break;
    case THREAD_COPY_INIT_MODEL:      
      {       
	job.length = 0;
	sendMergedMessage(&job, tr);
	      
       

	
	for(model = 0; model < tr->NumberOfModels; model++)
	  {
	    const partitionLengths 
	      *pl = getPartitionLengths(&(tr->partitionData[model]));	  

	    MPI_Bcast(&(tr->partitionData[model].numberOfCategories), 1, MPI_INT, 0, MPI_COMM_WORLD);
	    
	    MPI_Bcast(tr->partitionData[model].EIGN,        pl->eignLength,        MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(tr->partitionData[model].EV,          pl->evLength,          MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(tr->partitionData[model].EI,          pl->eiLength,          MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(tr->partitionData[model].substRates,  pl->substRatesLength,  MPI_DOUBLE, 0, MPI_COMM_WORLD);	  
	    MPI_Bcast(tr->partitionData[model].frequencies, pl->frequenciesLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    MPI_Bcast(tr->partitionData[model].tipVector,   pl->tipVectorLength,   MPI_DOUBLE, 0, MPI_COMM_WORLD);	 
	   	 
	    
	    MPI_Bcast(&tr->partitionData[model].lower, 1  , MPI_INT, 0, MPI_COMM_WORLD);
	    MPI_Bcast(&tr->partitionData[model].upper, 1  , MPI_INT, 0, MPI_COMM_WORLD);
	    MPI_Bcast(&tr->partitionData[model].alpha, 1  , MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    makeGammaCats(tr->partitionData[model].alpha, tr->partitionData[model].gammaRates, 4); 
	  }
			
	
	MPI_Bcast(tr->cdta->patrat,       tr->originalCrunchedLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(tr->cdta->patratStored, tr->originalCrunchedLength, MPI_DOUBLE, 0, MPI_COMM_WORLD);       

	MPI_Barrier(MPI_COMM_WORLD);  	
      }
      break;
    case THREAD_NEWVIEW:      
      {
	job.length = tr->td[0].count;       	
	sendMergedMessage(&job, tr);
	

	if(job.length >= TRAVERSAL_LENGTH)	
	  sendTraversalDescriptor(&job, tr);
		
	newviewIterative(tr);	
      }
      break;
    case THREAD_EVALUATE:
      {
	int    
	  model,
	  length = tr->td[0].count;	
       	
	job.length = length;       	
	sendMergedMessage(&job, tr);
	
	if(job.length >= TRAVERSAL_LENGTH)	 
	  sendTraversalDescriptor(&job, tr);
			
	evaluateIterative(tr, FALSE);
	
	for(model = 0; model < tr->NumberOfModels; model++)
	  partialResult[model] = tr->perPartitionLH[model];

	MPI_Reduce(partialResult, globalResult, tr->NumberOfModels, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);	

	for(model = 0; model < tr->NumberOfModels; model++)	 	  
	  tr->perPartitionLH[model] = globalResult[model];	 
      }
      break;
    case THREAD_MAKENEWZ_FIRST:
      {
	int 
	  length = tr->td[0].count,
	  model;		      		
	
	if(tr->multiBranch)
	  for(model = 0; model < tr->NumberOfModels; model++)
	    {
	      job.coreLZ[model]       = tr->coreLZ[model];
	      job.executeModel[model] = tr->executeModel[model];
	    }
	else
	  {
	    job.coreLZ[0]       = tr->coreLZ[0];
	    job.executeModel[0] = tr->executeModel[0];
	  }
	
	job.length = length;
	sendMergedMessage(&job, tr);
	
	if(job.length >= TRAVERSAL_LENGTH)	 
	  sendTraversalDescriptor(&job, tr);
      
	makenewzIterative(tr);	
	execCore(tr, dlnLdlz, d2lnLdlz2);

	if(tr->multiBranch)
	  {
	    for(model = 0; model < tr->NumberOfModels; model++)
	      {	    
		partialResult[model * 2 + 0] = dlnLdlz[model];
		partialResult[model * 2 + 1] = d2lnLdlz2[model];	    
	      }
	  }
	else
	  {
	    partialResult[0] = dlnLdlz[0];
	    partialResult[1] = d2lnLdlz2[0];
	  }
	 	
	if(tr->multiBranch)
	  MPI_Reduce(partialResult, globalResult, tr->NumberOfModels * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	else
	  MPI_Reduce(partialResult, globalResult, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      }
      break;
    case THREAD_MAKENEWZ:
      {
	int 
	  model;
	
	if(tr->multiBranch)	  
	  for(model = 0; model < tr->NumberOfModels; model++)
	    {
	      job.coreLZ[model]       = tr->coreLZ[model];
	      job.executeModel[model] = tr->executeModel[model];
	    }
	else
	  {
	    job.coreLZ[0]       = tr->coreLZ[0];
	    job.executeModel[0] = tr->executeModel[0]; 
	  }
	
	job.length = 0;
	sendMergedMessage(&job, tr);		

	execCore(tr, dlnLdlz, d2lnLdlz2);

	if(tr->multiBranch)
	  {
	    for(model = 0; model < tr->NumberOfModels; model++)
	      {	    
		partialResult[model * 2 + 0] = dlnLdlz[model];
		partialResult[model * 2 + 1] = d2lnLdlz2[model];	    
	      }
	  }
	else
	  {
	    partialResult[0] = dlnLdlz[0];
	    partialResult[1] = d2lnLdlz2[0];
	  }
	      
	if(tr->multiBranch)
	  MPI_Reduce(partialResult, globalResult, tr->NumberOfModels * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	else
	  MPI_Reduce(partialResult, globalResult, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      }
      break;    
    case THREAD_OPT_RATE:
      {	
	double 	
	  *buffer;

	int
	  bufIndex = 0,
	  length = 0,
	  model;       
	
	for(model = 0; model < tr->NumberOfModels; model++)	  	    
	  {
	    const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));
	    length += (pl->eignLength + pl->evLength + pl->eiLength + pl->tipVectorLength);
	    job.executeModel[model] = tr->executeModel[model];	 
	  }

	buffer = (double*)malloc(sizeof(double) * length);
	
	for(model = 0; model < tr->NumberOfModels; model++)	  	    
	  {
	    const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));
	    memcpy(&buffer[bufIndex],   tr->partitionData[model].EIGN,        pl->eignLength * sizeof(double));
	    bufIndex += pl->eignLength;
	    
	    memcpy(&buffer[bufIndex],   tr->partitionData[model].EV,          pl->evLength * sizeof(double));		  
	    bufIndex += pl->evLength;

	    memcpy(&buffer[bufIndex],   tr->partitionData[model].EI,          pl->eiLength * sizeof(double));
	    bufIndex += pl->eiLength;
	    
	    memcpy(&buffer[bufIndex],   tr->partitionData[model].tipVector,   pl->tipVectorLength * sizeof(double));
	    bufIndex += pl->tipVectorLength;
	  }
	  
	job.length = length;	
	sendMergedMessage(&job, tr);
		
	MPI_Bcast(buffer, length, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	evaluateIterative(tr, FALSE);
	     
	for(model = 0; model < tr->NumberOfModels; model++)
	  partialResult[model] = tr->perPartitionLH[model];

	
	MPI_Reduce(partialResult, globalResult, tr->NumberOfModels, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	for(model = 0; model < tr->NumberOfModels; model++)
	  tr->perPartitionLH[model] = globalResult[model];
	
	free(buffer);
      }
      break; 
    case THREAD_COPY_RATES:
    case THREAD_BROADCAST_RATE:
      {	
	double 	
	  *buffer;

	int
	  bufIndex = 0,
	  length = 0,
	  model;       
	
	for(model = 0; model < tr->NumberOfModels; model++)	  	    
	  {
	    const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));
	    length += (pl->eignLength + pl->evLength + pl->eiLength + pl->tipVectorLength);
	    if(jobType == THREAD_BROADCAST_RATE)
	      job.executeModel[model] = tr->executeModel[model];	 
	  }

	buffer = (double*)malloc(sizeof(double) * length);
	
	for(model = 0; model < tr->NumberOfModels; model++)	  	    
	  {
	    const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));
	    memcpy(&buffer[bufIndex],   tr->partitionData[model].EIGN,        pl->eignLength * sizeof(double));
	    bufIndex += pl->eignLength;
	    
	    memcpy(&buffer[bufIndex],   tr->partitionData[model].EV,          pl->evLength * sizeof(double));		  
	    bufIndex += pl->evLength;

	    memcpy(&buffer[bufIndex],   tr->partitionData[model].EI,          pl->eiLength * sizeof(double));
	    bufIndex += pl->eiLength;
	    
	    memcpy(&buffer[bufIndex],   tr->partitionData[model].tipVector,   pl->tipVectorLength * sizeof(double));
	    bufIndex += pl->tipVectorLength;
	  }
	  
	job.length = length;	
	sendMergedMessage(&job, tr);
		
	MPI_Bcast(buffer, length, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	
	free(buffer);
      }
      break;      
    case THREAD_RATE_CATS:
      {
	int
	  localCounter,
	  model,
	  i,
	  sendBufferSize,
	  recvBufferSize;

	double 
	  *patBufSend,
	  *patStoredBufSend,
	  *lhsBufSend,
	  *patBufRecv,
	  *patStoredBufRecv,
	  *lhsBufRecv;
     
	if(tr->manyPartitions)
	  sendBufferSize = tr->originalCrunchedLength;
	else
	  sendBufferSize = (tr->originalCrunchedLength / n) + 1;

	recvBufferSize = sendBufferSize * n;
	
	patBufSend = (double *)malloc(sendBufferSize * sizeof(double));
	patStoredBufSend =  (double *)malloc(sendBufferSize * sizeof(double));
	lhsBufSend = (double *)malloc(sendBufferSize * sizeof(double));
	patBufRecv = (double *)malloc(recvBufferSize * sizeof(double));
	patStoredBufRecv =  (double *)malloc(recvBufferSize * sizeof(double));
	lhsBufRecv = (double *)malloc(recvBufferSize * sizeof(double));
	

	job.length  = tr->td[0].count;
	job.lower_spacing = tr->lower_spacing;
	job.upper_spacing = tr->upper_spacing;
	sendMergedMessage(&job, tr);
	
	if(job.length >= TRAVERSAL_LENGTH)
	  sendTraversalDescriptor(&job, tr);
	
	optRateCatPthreads(tr, tr->lower_spacing, tr->upper_spacing, tr->lhs, n, tid);

	for(model = 0, localCounter = 0; model < tr->NumberOfModels; model++)
	   {               
	    
	     if(tr->manyPartitions)
	       {
		 size_t
		   start = (size_t)tr->partitionData[model].lower,
		   width = (size_t)tr->partitionData[model].upper - (size_t)tr->partitionData[model].lower;
		 
		 if(isThisMyPartition(tr, tid, model, n))
		   {
		     memcpy(&patBufSend[start],       &tr->cdta->patrat[start],       sizeof(double) * width);
		     memcpy(&patStoredBufSend[start], &tr->cdta->patratStored[start], sizeof(double) * width);
		     memcpy(&lhsBufSend[start],       &tr->lhs[start],                sizeof(double) * width);
		   }	
	       }
	     else
	       {
		 for(i = tr->partitionData[model].lower;  i < tr->partitionData[model].upper; i++)
		   if(i % n == tid)
		     {
		       patBufSend[localCounter] = tr->cdta->patrat[i];
		       patStoredBufSend[localCounter] = tr->cdta->patratStored[i];
		       lhsBufSend[localCounter] = tr->lhs[i];
		       localCounter++;
		     }
	       }
	   }
   
	MPI_Gather(patBufSend,       sendBufferSize, MPI_DOUBLE, patBufRecv,       sendBufferSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(patStoredBufSend, sendBufferSize, MPI_DOUBLE, patStoredBufRecv, sendBufferSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(lhsBufSend,       sendBufferSize, MPI_DOUBLE, lhsBufRecv,       sendBufferSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

	for(model = 0; model < tr->NumberOfModels; model++)
	   {   
	     if(tr->manyPartitions)
	       {
		 size_t
		   offset = (size_t)tr->partitionAssignment[model],
		   start  = (size_t)tr->partitionData[model].lower,
		   width  = (size_t)tr->partitionData[model].upper - (size_t)tr->partitionData[model].lower;
		 
		 memcpy(&tr->cdta->patrat[start],       &patBufRecv[offset * sendBufferSize + start],       sizeof(double) * width);
		 memcpy(&tr->cdta->patratStored[start], &patStoredBufRecv[offset * sendBufferSize + start], sizeof(double) * width);
		 memcpy(&tr->lhs[start],                &lhsBufRecv[offset * sendBufferSize + start],       sizeof(double) * width);
	       }
	     else
	       {
		 for(i = tr->partitionData[model].lower;  i < tr->partitionData[model].upper; i++)
		   {
		     int 
		       offset = i % n,
		       position = i / n;
		     
		     tr->cdta->patrat[i]       = patBufRecv[offset * sendBufferSize + position];
		     tr->cdta->patratStored[i] = patStoredBufRecv[offset * sendBufferSize + position];
		     tr->lhs[i]                = lhsBufRecv[offset * sendBufferSize + position];		     
		   }	   
	       }
	   }
	
	

	free(patBufSend);
	free(patStoredBufSend);
	free(lhsBufSend);
	free(patBufRecv);
	free(patStoredBufRecv);
	free(lhsBufRecv);

      }
      break;	
    case THREAD_NEWVIEW_MASKED:
      {		
	for(model = 0; model < tr->NumberOfModels; model++)	  	    	  
	  job.executeModel[model] = tr->executeModel[model];	        

	job.length = tr->td[0].count;
	sendMergedMessage(&job, tr);
	
	if(job.length >= TRAVERSAL_LENGTH)	 
	  sendTraversalDescriptor(&job, tr);
		
	newviewIterative(tr);	
      }
      break;        
    case THREAD_OPT_ALPHA:
      {
	double 	
	  *buffer;

	int
	  bufIndex = 0,
	  length = tr->NumberOfModels,
	  model;       
	
	for(model = 0; model < tr->NumberOfModels; model++)	  	    	  	   
	  job.executeModel[model] = tr->executeModel[model];	 	 

	buffer = (double*)malloc(sizeof(double) * length);
	
	for(model = 0; model < tr->NumberOfModels; model++)	 
	  buffer[bufIndex++] = tr->partitionData[model].alpha;	 
	  
	job.length = length;	
	sendMergedMessage(&job, tr);
		
	MPI_Bcast(buffer, length, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	evaluateIterative(tr, FALSE);
	     
	for(model = 0; model < tr->NumberOfModels; model++)
	  partialResult[model] = tr->perPartitionLH[model];
	
	MPI_Reduce(partialResult, globalResult, tr->NumberOfModels, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	for(model = 0; model < tr->NumberOfModels; model++)
	  tr->perPartitionLH[model] = globalResult[model];
	
	free(buffer);
      }      
      break;      
    case THREAD_COPY_ALPHA:
      {
	 double 	
	  *buffer;

	int
	  bufIndex = 0,
	  length = tr->NumberOfModels,
	  model;       		 	 

	buffer = (double*)malloc(sizeof(double) * length);
	
	for(model = 0; model < tr->NumberOfModels; model++)	 
	  buffer[bufIndex++] = tr->partitionData[model].alpha;	 
	  
	job.length = length;	
	sendMergedMessage(&job, tr);
		
	MPI_Bcast(buffer, length, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	
	free(buffer);  
      }
      break;
    case EXIT_GRACEFULLY:     
      {
	job.length = 0;
	sendMergedMessage(&job, tr);      
      }
      break;
    default:
      assert(0);
    }

  

  free(partialResult);
  free(dlnLdlz);
  free(d2lnLdlz2);
  
 
  
}
