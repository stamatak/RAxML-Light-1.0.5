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

#ifdef WIN32
#include <direct.h>
#endif

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
#include <stdarg.h>
#include <limits.h>

#ifdef  _FINE_GRAIN_MPI
#include <mpi.h>
#endif



#ifdef _USE_PTHREADS
#include <pthread.h>

#endif

#if ! (defined(__ppc) || defined(__powerpc__) || defined(PPC))
#include <xmmintrin.h>
/*
   special bug fix, enforces denormalized numbers to be flushed to zero,
   without this program is a tiny bit faster though.
#include <emmintrin.h> 
#define MM_DAZ_MASK    0x0040
#define MM_DAZ_ON    0x0040
#define MM_DAZ_OFF    0x0000
*/
#endif

#include "axml.h"
#include "globalVariables.h"


#define _PORTABLE_PTHREADS


/***************** UTILITY FUNCTIONS **************************/


void myBinFwrite(const void *ptr, size_t size, size_t nmemb)
{ 
  size_t  
    bytes_written = fwrite(ptr, size, nmemb, byteFile);

  assert(bytes_written == nmemb);
}


void myBinFread(void *ptr, size_t size, size_t nmemb)
{  
  size_t
    bytes_read;

  bytes_read = fread(ptr, size, nmemb, byteFile);

  assert(bytes_read == nmemb);
}


void *malloc_aligned(size_t size) 
{
  void 
    *ptr = (void *)NULL;

  int 
    res;


#if defined (__APPLE__)
  /* 
     presumably malloc on MACs always returns 
     a 16-byte aligned pointer
     */

  ptr = malloc(size);

  if(ptr == (void*)NULL) 
    assert(0);

#ifdef __AVX
  assert(0);
#endif


#else
  res = posix_memalign( &ptr, BYTE_ALIGNMENT, size );

  if(res != 0) 
    assert(0);
#endif 

  return ptr;
}




FILE *getNumberOfTrees(tree *tr, char *fileName, analdef *adef)
{
  FILE 
    *f = myfopen(fileName, "r");

  int 
    trees = 0,
          ch;

  while((ch = fgetc(f)) != EOF)
    if(ch == ';')
      trees++;

  assert(trees > 0);

  tr->numberOfTrees = trees;

  if(!adef->allInOne)   
    printBothOpen("\n\nFound %d trees in File %s\n\n", trees, fileName);


  rewind(f);

  return f;
}

static void printBoth(FILE *f, const char* format, ... )
{
  va_list args;
  va_start(args, format);
  vfprintf(f, format, args );
  va_end(args);

  va_start(args, format);
  vprintf(format, args );
  va_end(args);
}

void printBothOpen(const char* format, ... )
{
  FILE *f = myfopen(infoFileName, "ab");

  va_list args;
  va_start(args, format);
  vfprintf(f, format, args );
  va_end(args);

  va_start(args, format);
  vprintf(format, args );
  va_end(args);

  fclose(f);
}

void printBothOpenMPI(const char* format, ... )
{
#ifdef _WAYNE_MPI
  if(processID == 0)
#endif
  {
    FILE *f = myfopen(infoFileName, "ab");

    va_list args;
    va_start(args, format);
    vfprintf(f, format, args );
    va_end(args);

    va_start(args, format);
    vprintf(format, args );
    va_end(args);

    fclose(f);
  }
}


boolean getSmoothFreqs(int dataType)
{
  assert(MIN_MODEL < dataType && dataType < MAX_MODEL);

  return pLengths[dataType].smoothFrequencies;
}

const unsigned int *getBitVector(int dataType)
{
  assert(MIN_MODEL < dataType && dataType < MAX_MODEL);

  return pLengths[dataType].bitVector;
}


int getStates(int dataType)
{
  assert(MIN_MODEL < dataType && dataType < MAX_MODEL);

  return pLengths[dataType].states;
}

int getUndetermined(int dataType)
{
  assert(MIN_MODEL < dataType && dataType < MAX_MODEL);

  return pLengths[dataType].undetermined;
}



char getInverseMeaning(int dataType, unsigned char state)
{
  assert(MIN_MODEL < dataType && dataType < MAX_MODEL);

  return  pLengths[dataType].inverseMeaning[state];
}

partitionLengths *getPartitionLengths(pInfo *p)
{
  int 
    dataType  = p->dataType,
              states    = p->states,
              tipLength = p->maxTipStates;

  assert(states != -1 && tipLength != -1);

  assert(MIN_MODEL < dataType && dataType < MAX_MODEL);

  pLength.leftLength = pLength.rightLength = states * states;
  pLength.eignLength = states -1;
  pLength.evLength   = states * states;
  pLength.eiLength   = states * states - states;
  pLength.substRatesLength = (states * states - states) / 2;
  pLength.frequenciesLength = states;
  pLength.tipVectorLength   = tipLength * states;
  pLength.symmetryVectorLength = (states * states - states) / 2;
  pLength.frequencyGroupingLength = states;
  pLength.nonGTR = FALSE;

  return (&pLengths[dataType]); 
}



static boolean isCat(analdef *adef)
{
  if(adef->model == M_PROTCAT || adef->model == M_GTRCAT || adef->model == M_BINCAT || adef->model == M_32CAT || adef->model == M_64CAT)
    return TRUE;
  else
    return FALSE;
}








static void setRateHetAndDataIncrement(tree *tr, analdef *adef)
{
  int model;

  if(isCat(adef))
  {
    tr->rateHetModel = CAT;
    tr->discreteRateCategories = 1; 
  }
  else
  {
    tr->rateHetModel = GAMMA;
    tr->discreteRateCategories = 4;
  }



  for(model = 0; model < tr->NumberOfModels; model++)
  {
    int 
      states = -1,
             maxTipStates = getUndetermined(tr->partitionData[model].dataType) + 1;

    switch(tr->partitionData[model].dataType)
    {
      case DNA_DATA:
      case AA_DATA:	
        states = getStates(tr->partitionData[model].dataType);	 
        break;	
      default:
        assert(0);
    }

    tr->partitionData[model].states       = states;
    tr->partitionData[model].maxTipStates = maxTipStates;
  }
}


double gettime(void)
{
#ifdef WIN32
  time_t tp;
  struct tm localtm;
  tp = time(NULL);
  localtm = *localtime(&tp);
  return 60.0*localtm.tm_min + localtm.tm_sec;
#else
  struct timeval ttime;
  gettimeofday(&ttime , NULL);
  return ttime.tv_sec + ttime.tv_usec * 0.000001;
#endif
}

int gettimeSrand(void)
{
#ifdef WIN32
  time_t tp;
  struct tm localtm;
  tp = time(NULL);
  localtm = *localtime(&tp);
  return 24*60*60*localtm.tm_yday + 60*60*localtm.tm_hour + 60*localtm.tm_min  + localtm.tm_sec;
#else
  struct timeval ttime;
  gettimeofday(&ttime , NULL);
  return ttime.tv_sec + ttime.tv_usec;
#endif
}

double randum (long  *seed)
{
  long  sum, mult0, mult1, seed0, seed1, seed2, newseed0, newseed1, newseed2;
  double res;

  mult0 = 1549;
  seed0 = *seed & 4095;
  sum  = mult0 * seed0;
  newseed0 = sum & 4095;
  sum >>= 12;
  seed1 = (*seed >> 12) & 4095;
  mult1 =  406;
  sum += mult0 * seed1 + mult1 * seed0;
  newseed1 = sum & 4095;
  sum >>= 12;
  seed2 = (*seed >> 24) & 255;
  sum += mult0 * seed2 + mult1 * seed1;
  newseed2 = sum & 255;

  *seed = newseed2 << 24 | newseed1 << 12 | newseed0;
  res = 0.00390625 * (newseed2 + 0.000244140625 * (newseed1 + 0.000244140625 * newseed0));

  return res;
}

static int filexists(char *filename)
{
  FILE *fp;
  int res;
  fp = fopen(filename,"rb");

  if(fp)
  {
    res = 1;
    fclose(fp);
  }
  else
    res = 0;

  return res;
}


FILE *myfopen(const char *path, const char *mode)
{
  FILE *fp = fopen(path, mode);

  if(strcmp(mode,"r") == 0 || strcmp(mode,"rb") == 0)
  {
    if(fp)
      return fp;
    else
    {
      if(processID == 0)
        printf("The file %s you want to open for reading does not exist, exiting ...\n", path);
      errorExit(-1);
      return (FILE *)NULL;
    }
  }
  else
  {
    if(fp)
      return fp;
    else
    {
      if(processID == 0)
        printf("The file %s RAxML wants to open for writing or appending can not be opened [mode: %s], exiting ...\n",
            path, mode);
      errorExit(-1);
      return (FILE *)NULL;
    }
  }


}





/********************* END UTILITY FUNCTIONS ********************/


/******************************some functions for the likelihood computation ****************************/


boolean isTip(int number, int maxTips)
{
  assert(number > 0);

  if(number <= maxTips)
    return TRUE;
  else
    return FALSE;
}









void getxnode (nodeptr p)
{
  nodeptr  s;

  if ((s = p->next)->x || (s = s->next)->x)
  {
    p->x = s->x;
    s->x = 0;
  }

  assert(p->x);
}





void hookup (nodeptr p, nodeptr q, double *z, int numBranches)
{
  int i;

  p->back = q;
  q->back = p;

  for(i = 0; i < numBranches; i++)
    p->z[i] = q->z[i] = z[i];
}

void hookupDefault (nodeptr p, nodeptr q, int numBranches)
{
  int i;

  p->back = q;
  q->back = p;

  for(i = 0; i < numBranches; i++)
    p->z[i] = q->z[i] = defaultz;
}


/***********************reading and initializing input ******************/

static void getnums (rawdata *rdta)
{
  if (fscanf(INFILE, "%d %d", & rdta->numsp, & rdta->sites) != 2)
  {
    if(processID == 0)
      printf("ERROR: Problem reading number of species and sites\n");
    errorExit(-1);
  }

  if (rdta->numsp < 4)
  {
    if(processID == 0)
      printf("TOO FEW SPECIES\n");
    errorExit(-1);
  }

  if (rdta->sites < 1)
  {
    if(processID == 0)
      printf("TOO FEW SITES\n");
    errorExit(-1);
  }

  return;
}





boolean whitechar (int ch)
{
  return (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r');
}


static void uppercase (int *chptr)
{
  int  ch;

  ch = *chptr;
  if ((ch >= 'a' && ch <= 'i') || (ch >= 'j' && ch <= 'r')
      || (ch >= 's' && ch <= 'z'))
    *chptr = ch + 'A' - 'a';
}




static void getyspace (rawdata *rdta)
{
  size_t size = 4 * ((size_t)(rdta->sites / 4 + 1));
  int    i;
  unsigned char *y0;

  rdta->y = (unsigned char **) malloc((rdta->numsp + 1) * sizeof(unsigned char *));
  assert(rdta->y);   

  y0 = (unsigned char *) malloc(((size_t)(rdta->numsp + 1)) * size * sizeof(unsigned char));

  /*
     printf("Raw alignment data Assigning %Zu bytes\n", ((size_t)(rdta->numsp + 1)) * size * sizeof(unsigned char));

*/

  assert(y0);   

  rdta->y0 = y0;

  for (i = 0; i <= rdta->numsp; i++)
  {
    rdta->y[i] = y0;
    y0 += size;
  }

  return;
}


static unsigned int KISS32(void)
{
  static unsigned int 
    x = 123456789, 
      y = 362436069,
      z = 21288629,
      w = 14921776,
      c = 0;

  unsigned int t;

  x += 545925293;
  y ^= (y<<13); 
  y ^= (y>>17); 
  y ^= (y<<5);
  t = z + w + c; 
  z = w; 
  c = (t>>31); 
  w = t & 2147483647;

  return (x+y+w);
}

static boolean setupTree (tree *tr, analdef *adef)
{
  nodeptr  p0, p, q;
  int
    i,
    j,
    k,
    tips,
    inter; 

  if(!adef->readTaxaOnly)
  {
    tr->bigCutoff = FALSE;

    tr->patternPosition = (int*)NULL;
    tr->columnPosition = (int*)NULL;

    tr->maxCategories = MAX(4, adef->categories);

    tr->partitionContributions = (double *)malloc(sizeof(double) * tr->NumberOfModels);

    for(i = 0; i < tr->NumberOfModels; i++)
      tr->partitionContributions[i] = -1.0;

    tr->perPartitionLH = (double *)malloc(sizeof(double) * tr->NumberOfModels);
    tr->storedPerPartitionLH = (double *)malloc(sizeof(double) * tr->NumberOfModels);

    for(i = 0; i < tr->NumberOfModels; i++)
    {
      tr->perPartitionLH[i] = 0.0;
      tr->storedPerPartitionLH[i] = 0.0;
    }

    if(adef->grouping)
      tr->grouped = TRUE;
    else
      tr->grouped = FALSE;

    if(adef->constraint)
      tr->constrained = TRUE;
    else
      tr->constrained = FALSE;

    tr->treeID = 0;
  }

  tips  = tr->mxtips;
  inter = tr->mxtips - 1;

  if(!adef->readTaxaOnly)
  {
    tr->yVector      = (unsigned char **)  malloc((tr->mxtips + 1) * sizeof(unsigned char *));

    tr->fracchanges  = (double *)malloc(tr->NumberOfModels * sizeof(double));
    tr->likelihoods  = (double *)malloc(adef->multipleRuns * sizeof(double));
  }

  tr->numberOfTrees = -1;



  tr->treeStringLength = tr->mxtips * (nmlngth+128) + 256 + tr->mxtips * 2;

  tr->tree_string  = (char*)calloc(tr->treeStringLength, sizeof(char)); 
  tr->tree0 = (char*)calloc(tr->treeStringLength, sizeof(char));
  tr->tree1 = (char*)calloc(tr->treeStringLength, sizeof(char));


  /*TODO, must that be so long ?*/

  if(!adef->readTaxaOnly)
  {


    tr->td[0].count = 0;
    tr->td[0].ti    = (traversalInfo *)malloc(sizeof(traversalInfo) * tr->mxtips);


    for(i = 0; i < tr->NumberOfModels; i++)
      tr->fracchanges[i] = -1.0;
    tr->fracchange = -1.0;

    tr->constraintVector = (int *)malloc((2 * tr->mxtips) * sizeof(int));

    tr->nameList = (char **)malloc(sizeof(char *) * (tips + 1));
  }

  if (!(p0 = (nodeptr) malloc((tips + 3*inter) * sizeof(node))))
  {
    printf("ERROR: Unable to obtain sufficient tree memory\n");
    return  FALSE;
  }

  tr->nodeBaseAddress = p0;


  if (!(tr->nodep = (nodeptr *) malloc((2*tr->mxtips) * sizeof(nodeptr))))
  {
    printf("ERROR: Unable to obtain sufficient tree memory, too\n");
    return  FALSE;
  }

  tr->nodep[0] = (node *) NULL;    /* Use as 1-based array */

  for (i = 1; i <= tips; i++)
  {
    p = p0++;

    p->hash   =  KISS32(); /* hast table stuff */
    p->x      =  0;
    p->number =  i;
    p->next   =  p;
    p->back   = (node *)NULL;
    p->bInf   = (branchInfo *)NULL;




    tr->nodep[i] = p;
  }

  for (i = tips + 1; i <= tips + inter; i++)
  {
    q = (node *) NULL;
    for (j = 1; j <= 3; j++)
    {	 
      p = p0++;
      if(j == 1)
        p->x = 1;
      else
        p->x =  0;
      p->number = i;
      p->next   = q;
      p->bInf   = (branchInfo *)NULL;
      p->back   = (node *) NULL;
      p->hash   = 0;




      q = p;
    }
    p->next->next->next = p;
    tr->nodep[i] = p;
  }

  tr->likelihood  = unlikely;
  tr->start       = (node *) NULL;

  for(i = 0; i < NUM_BRANCHES; i++)
    tr->startVector[i]  = (node *) NULL;

  tr->ntips       = 0;
  tr->nextnode    = 0;

  if(!adef->readTaxaOnly)
  {
    for(i = 0; i < tr->numBranches; i++)
      tr->partitionSmoothed[i] = FALSE;
  }

  tr->bitVectors = (unsigned int **)NULL;

  tr->vLength = 0;

  tr->h = (hashtable*)NULL;


  return TRUE;
}


static void checkTaxonName(char *buffer, int len)
{
  int i;

  for(i = 0; i < len - 1; i++)
  {
    boolean valid;

    switch(buffer[i])
    {
      case '\0':
      case '\t':
      case '\n':
      case '\r':
      case ' ':
      case ':':
      case ',':
      case '(':
      case ')':
      case ';':
      case '[':
      case ']':
        valid = FALSE;
        break;
      default:
        valid = TRUE;
    }

    if(!valid)
    {
      printf("ERROR: Taxon Name \"%s\" is invalid at position %d, it contains illegal character %c\n", buffer, i, buffer[i]);
      printf("Illegal characters in taxon-names are: tabulators, carriage returns, spaces, \":\", \",\", \")\", \"(\", \";\", \"]\", \"[\"\n");
      printf("Exiting\n");
      exit(-1);
    }

  }
  assert(buffer[len - 1] == '\0');
}

static boolean getdata(analdef *adef, rawdata *rdta, tree *tr)
{
  int   
    i, 
    j, 
    basesread, 
    basesnew, 
    ch, my_i, meaning,
    len,
    meaningAA[256], 
    meaningDNA[256], 
    meaningBINARY[256],
    meaningGeneric32[256],
    meaningGeneric64[256];

  boolean  
    allread, 
    firstpass;

  char 
    buffer[nmlngth + 2];

  unsigned char
    genericChars32[32] = {'0', '1', '2', '3', '4', '5', '6', '7', 
      '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
      'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
      'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V'};  
  unsigned long 
    total = 0,
          gaps  = 0;

  for (i = 0; i < 256; i++)
  {      
    meaningAA[i]          = -1;
    meaningDNA[i]         = -1;
    meaningBINARY[i]      = -1;
    meaningGeneric32[i]   = -1;
    meaningGeneric64[i]   = -1;
  }

  /* generic 32 data */

  for(i = 0; i < 32; i++)
    meaningGeneric32[genericChars32[i]] = i;
  meaningGeneric32['-'] = getUndetermined(GENERIC_32);
  meaningGeneric32['?'] = getUndetermined(GENERIC_32);

  /* AA data */

  meaningAA['A'] =  0;  /* alanine */
  meaningAA['R'] =  1;  /* arginine */
  meaningAA['N'] =  2;  /*  asparagine*/
  meaningAA['D'] =  3;  /* aspartic */
  meaningAA['C'] =  4;  /* cysteine */
  meaningAA['Q'] =  5;  /* glutamine */
  meaningAA['E'] =  6;  /* glutamic */
  meaningAA['G'] =  7;  /* glycine */
  meaningAA['H'] =  8;  /* histidine */
  meaningAA['I'] =  9;  /* isoleucine */
  meaningAA['L'] =  10; /* leucine */
  meaningAA['K'] =  11; /* lysine */
  meaningAA['M'] =  12; /* methionine */
  meaningAA['F'] =  13; /* phenylalanine */
  meaningAA['P'] =  14; /* proline */
  meaningAA['S'] =  15; /* serine */
  meaningAA['T'] =  16; /* threonine */
  meaningAA['W'] =  17; /* tryptophan */
  meaningAA['Y'] =  18; /* tyrosine */
  meaningAA['V'] =  19; /* valine */
  meaningAA['B'] =  20; /* asparagine, aspartic 2 and 3*/
  meaningAA['Z'] =  21; /*21 glutamine glutamic 5 and 6*/

  meaningAA['X'] = 
    meaningAA['?'] = 
    meaningAA['*'] = 
    meaningAA['-'] = 
    getUndetermined(AA_DATA);

  /* DNA data */

  meaningDNA['A'] =  1;
  meaningDNA['B'] = 14;
  meaningDNA['C'] =  2;
  meaningDNA['D'] = 13;
  meaningDNA['G'] =  4;
  meaningDNA['H'] = 11;
  meaningDNA['K'] = 12;
  meaningDNA['M'] =  3;  
  meaningDNA['R'] =  5;
  meaningDNA['S'] =  6;
  meaningDNA['T'] =  8;
  meaningDNA['U'] =  8;
  meaningDNA['V'] =  7;
  meaningDNA['W'] =  9; 
  meaningDNA['Y'] = 10;

  meaningDNA['N'] = 
    meaningDNA['O'] = 
    meaningDNA['X'] = 
    meaningDNA['-'] = 
    meaningDNA['?'] = 
    getUndetermined(DNA_DATA);

  /* BINARY DATA */

  meaningBINARY['0'] = 1;
  meaningBINARY['1'] = 2;

  meaningBINARY['-'] = 
    meaningBINARY['?'] = 
    getUndetermined(BINARY_DATA);


  /*******************************************************************/

  basesread = basesnew = 0;

  allread = FALSE;
  firstpass = TRUE;
  ch = ' ';

  while (! allread)
  {
    for (i = 1; i <= tr->mxtips; i++)
    {
      if (firstpass)
      {
        ch = getc(INFILE);
        while(ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r')
          ch = getc(INFILE);

        my_i = 0;

        do
        {
          buffer[my_i] = ch;
          ch = getc(INFILE);
          my_i++;
          if(my_i >= nmlngth)
          {
            if(processID == 0)
            {
              printf("Taxon Name to long at taxon %d, adapt constant nmlngth in\n", i);
              printf("axml.h, current setting %d\n", nmlngth);
            }
            errorExit(-1);
          }
        }
        while(ch !=  ' ' && ch != '\n' && ch != '\t' && ch != '\r');

        while(ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r')
          ch = getc(INFILE);

        ungetc(ch, INFILE);

        buffer[my_i] = '\0';
        len = strlen(buffer) + 1;
        checkTaxonName(buffer, len);
        tr->nameList[i] = (char *)malloc(sizeof(char) * len);
        strcpy(tr->nameList[i], buffer);
      }

      j = basesread;

      while ((j < rdta->sites) && ((ch = getc(INFILE)) != EOF) && (ch != '\n') && (ch != '\r'))
      {
        uppercase(& ch);

        assert(tr->dataVector[j + 1] != -1);

        switch(tr->dataVector[j + 1])
        {
          case BINARY_DATA:
            meaning = meaningBINARY[ch];
            break;
          case DNA_DATA:
          case SECONDARY_DATA:
          case SECONDARY_DATA_6:
          case SECONDARY_DATA_7:
            /*
               still dealing with DNA/RNA here, hence just act if as they where DNA characters
               corresponding column merging for sec struct models will take place later
               */
            meaning = meaningDNA[ch];
            break;
          case AA_DATA:
            meaning = meaningAA[ch];
            break;
          case GENERIC_32:
            meaning = meaningGeneric32[ch];
            break;
          case GENERIC_64:
            meaning = meaningGeneric64[ch];
            break;
          default:
            assert(0);
        }

        if (meaning != -1)
        {
          j++;
          rdta->y[i][j] = ch;		 
        }
        else
        {
          if(!whitechar(ch))
          {
            printf("ERROR: Bad base (%c) at site %d of sequence %d\n",
                ch, j + 1, i);
            return FALSE;
          }
        }
      }

      if (ch == EOF)
      {
        printf("ERROR: End-of-file at site %d of sequence %d\n", j + 1, i);
        return  FALSE;
      }

      if (! firstpass && (j == basesread))
        i--;
      else
      {
        if (i == 1)
          basesnew = j;
        else
          if (j != basesnew)
          {
            printf("ERROR: Sequences out of alignment\n");
            printf("%d (instead of %d) residues read in sequence %d %s\n",
                j - basesread, basesnew - basesread, i, tr->nameList[i]);
            return  FALSE;
          }
      }
      while (ch != '\n' && ch != EOF && ch != '\r') ch = getc(INFILE);  /* flush line *//* PC-LINEBREAK*/
    }

    firstpass = FALSE;
    basesread = basesnew;
    allread = (basesread >= rdta->sites);
  }

  for(j = 1; j <= tr->mxtips; j++)
    for(i = 1; i <= rdta->sites; i++)
    {
      assert(tr->dataVector[i] != -1);

      switch(tr->dataVector[i])
      {
        case BINARY_DATA:
          meaning = meaningBINARY[rdta->y[j][i]];
          if(meaning == getUndetermined(BINARY_DATA))
            gaps++;
          break;

        case SECONDARY_DATA:
        case SECONDARY_DATA_6:
        case SECONDARY_DATA_7:
          assert(tr->secondaryStructurePairs[i - 1] != -1);
          assert(i - 1 == tr->secondaryStructurePairs[tr->secondaryStructurePairs[i - 1]]);
          /*
             don't worry too much about undetermined column count here for sec-struct, just count
             DNA/RNA gaps here and worry about the rest later-on, falling through to DNA again :-)
             */
        case DNA_DATA:
          meaning = meaningDNA[rdta->y[j][i]];
          if(meaning == getUndetermined(DNA_DATA))
            gaps++;
          break;

        case AA_DATA:
          meaning = meaningAA[rdta->y[j][i]];
          if(meaning == getUndetermined(AA_DATA))
            gaps++;
          break;

        case GENERIC_32:
          meaning = meaningGeneric32[rdta->y[j][i]];
          if(meaning == getUndetermined(GENERIC_32))
            gaps++;
          break;

        case GENERIC_64:
          meaning = meaningGeneric64[rdta->y[j][i]];
          if(meaning == getUndetermined(GENERIC_64))
            gaps++;
          break;
        default:
          assert(0);
      }

      total++;
      rdta->y[j][i] = meaning;
    }

  adef->gapyness = (double)gaps / (double)total;

  if(adef->writeBinaryFile)
  {
    int i;

    myBinFwrite(&(adef->gapyness), sizeof(double), 1);

    for(i = 1; i <= tr->mxtips; i++)
    {
      int 
        len = strlen(tr->nameList[i]) + 1;

      myBinFwrite(&len, sizeof(int), 1);
      myBinFwrite(tr->nameList[i], sizeof(char), len);

      /*printf("%d %s\n", len, tr->nameList[i]);*/
    }     
  }

  return  TRUE;
}



static void inputweights (rawdata *rdta)
{
  int i, w, fres;
  FILE *weightFile;
  int *wv = (int *)malloc(sizeof(int) *  rdta->sites);

  weightFile = myfopen(weightFileName, "rb");

  i = 0;

  while((fres = fscanf(weightFile,"%d", &w)) != EOF)
  {
    if(!fres)
    {
      if(processID == 0)
        printf("error reading weight file probably encountered a non-integer weight value\n");
      errorExit(-1);
    }
    wv[i] = w;
    i++;
  }

  if(i != rdta->sites)
  {
    if(processID == 0)
      printf("number %d of weights not equal to number %d of alignment columns\n", i, rdta->sites);
    errorExit(-1);
  }

  for(i = 1; i <= rdta->sites; i++)
    rdta->wgt[i] = wv[i - 1];

  fclose(weightFile);
  free(wv);
}



static void getinput(analdef *adef, rawdata *rdta, cruncheddata *cdta, tree *tr)
{
  int i;

  if(adef->readBinaryFile)
  {
    myBinFread(&(rdta->sites), sizeof(int), 1);
    myBinFread(&(rdta->numsp), sizeof(int), 1);
  }
  else
  {
    INFILE = myfopen(seq_file, "rb");

    getnums(rdta);

    if(adef->writeBinaryFile)
    {
      myBinFwrite(&(rdta->sites), sizeof(int), 1);
      myBinFwrite(&(rdta->numsp), sizeof(int), 1);
    }
  }


  tr->mxtips            = rdta->numsp;

  if(!adef->readTaxaOnly)
  {
    rdta->wgt             = (int *)    malloc((rdta->sites + 1) * sizeof(int));
    cdta->alias           = (int *)    malloc((rdta->sites + 1) * sizeof(int));
    cdta->aliaswgt        = (int *)    malloc((rdta->sites + 1) * sizeof(int));
    cdta->rateCategory    = (int *)    malloc((rdta->sites + 1) * sizeof(int));
    tr->model             = (int *)    calloc((rdta->sites + 1), sizeof(int));
    tr->initialDataVector  = (int *)    malloc((rdta->sites + 1) * sizeof(int));
    tr->extendedDataVector = (int *)    malloc((rdta->sites + 1) * sizeof(int));     
    cdta->patrat          = (double *) malloc((rdta->sites + 1) * sizeof(double));
    cdta->patratStored    = (double *) malloc((rdta->sites + 1) * sizeof(double));      
    tr->wr                = (double *) malloc((rdta->sites + 1) * sizeof(double)); 
    tr->wr2               = (double *) malloc((rdta->sites + 1) * sizeof(double)); 


    if(!adef->useWeightFile)
    {
      for (i = 1; i <= rdta->sites; i++)
        rdta->wgt[i] = 1;
    }
    else
    {
      assert(!adef->useSecondaryStructure);
      inputweights(rdta);
    }
  }

  tr->multiBranch = 0;
  tr->numBranches = 1;

  if(!adef->readTaxaOnly)
  {
    if(adef->useMultipleModel)
    {
      int ref;

      parsePartitions(adef, rdta, tr);

      for(i = 1; i <= rdta->sites; i++)
      {
        ref = tr->model[i];
        tr->initialDataVector[i] = tr->initialPartitionData[ref].dataType;
      }
    }
    else
    {
      int dataType = -1;

      tr->initialPartitionData  = (pInfo*)malloc(sizeof(pInfo));
      tr->initialPartitionData[0].partitionName = (char*)malloc(128 * sizeof(char));
      strcpy(tr->initialPartitionData[0].partitionName, "No Name Provided");

      tr->initialPartitionData[0].protModels = adef->proteinMatrix;
      tr->initialPartitionData[0].protFreqs  = adef->protEmpiricalFreqs;


      tr->NumberOfModels = 1;

      if(adef->model == M_PROTCAT || adef->model == M_PROTGAMMA)
        dataType = AA_DATA;
      if(adef->model == M_GTRCAT || adef->model == M_GTRGAMMA)
        dataType = DNA_DATA;
      if(adef->model == M_BINCAT || adef->model == M_BINGAMMA)
        dataType = BINARY_DATA;
      if(adef->model == M_32CAT || adef->model == M_32GAMMA)
        dataType = GENERIC_32;
      if(adef->model == M_64CAT || adef->model == M_64GAMMA)
        dataType = GENERIC_64;



      assert(dataType == BINARY_DATA || dataType == DNA_DATA || dataType == AA_DATA || 
          dataType == GENERIC_32  || dataType == GENERIC_64);

      tr->initialPartitionData[0].dataType = dataType;

      for(i = 0; i <= rdta->sites; i++)
      {
        tr->initialDataVector[i] = dataType;
        tr->model[i]      = 0;
      }
    }

    if(adef->useSecondaryStructure)
    {
      memcpy(tr->extendedDataVector, tr->initialDataVector, (rdta->sites + 1) * sizeof(int));

      tr->extendedPartitionData =(pInfo*)malloc(sizeof(pInfo) * tr->NumberOfModels);

      for(i = 0; i < tr->NumberOfModels; i++)
      {
        tr->extendedPartitionData[i].partitionName = (char*)malloc((strlen(tr->initialPartitionData[i].partitionName) + 1) * sizeof(char));
        strcpy(tr->extendedPartitionData[i].partitionName, tr->initialPartitionData[i].partitionName);
        tr->extendedPartitionData[i].dataType   = tr->initialPartitionData[i].dataType;

        tr->extendedPartitionData[i].protModels = tr->initialPartitionData[i].protModels;
        tr->extendedPartitionData[i].protFreqs  = tr->initialPartitionData[i].protFreqs;
      }

      parseSecondaryStructure(tr, adef, rdta->sites);

      tr->dataVector    = tr->extendedDataVector;
      tr->partitionData = tr->extendedPartitionData;
    }
    else
    {
      tr->dataVector    = tr->initialDataVector;
      tr->partitionData = tr->initialPartitionData;
    }

    tr->executeModel   = (boolean *)malloc(sizeof(boolean) * tr->NumberOfModels);

    for(i = 0; i < tr->NumberOfModels; i++)
      tr->executeModel[i] = TRUE;
    if(!adef->readBinaryFile)
      getyspace(rdta);
  } 

  setupTree(tr, adef);


  if(!adef->readTaxaOnly)
  {
    if(adef->readBinaryFile)
    {
      int i;

      myBinFread(&(adef->gapyness), sizeof(double), 1);

      for(i = 1; i <= tr->mxtips; i++)
      {
        int len;
        myBinFread(&len, sizeof(int), 1);
        tr->nameList[i] = (char*)malloc(sizeof(char) * len);
        myBinFread(tr->nameList[i], sizeof(char), len);

        /*printf("%d %s\n", len, tr->nameList[i]);*/
      }   
    }
    else
    {
      if(!getdata(adef, rdta, tr))
      {
        printf("Problem reading alignment file \n");
        errorExit(1);
      }
    }

    tr->nameHash = initStringHashTable(10 * tr->mxtips);
    for(i = 1; i <= tr->mxtips; i++)
      addword(tr->nameList[i], tr->nameHash, i);

    if(!adef->readBinaryFile)
      fclose(INFILE);
  }
}



static unsigned char buildStates(int secModel, unsigned char v1, unsigned char v2)
{
  unsigned char new = 0;

  switch(secModel)
  {
    case SECONDARY_DATA:
      new = v1;
      new = new << 4;
      new = new | v2;
      break;
    case SECONDARY_DATA_6:
      {
        int
          meaningDNA[256],
          i;

        const unsigned char
          allowedStates[6][2] = {{'A','T'}, {'C', 'G'}, {'G', 'C'}, {'G','T'}, {'T', 'A'}, {'T', 'G'}};

        const unsigned char
          finalBinaryStates[6] = {1, 2, 4, 8, 16, 32};

        unsigned char
          intermediateBinaryStates[6];

        int length = 6;

        for(i = 0; i < 256; i++)
          meaningDNA[i] = -1;

        meaningDNA['A'] =  1;
        meaningDNA['B'] = 14;
        meaningDNA['C'] =  2;
        meaningDNA['D'] = 13;
        meaningDNA['G'] =  4;
        meaningDNA['H'] = 11;
        meaningDNA['K'] = 12;
        meaningDNA['M'] =  3;
        meaningDNA['N'] = 15;
        meaningDNA['O'] = 15;
        meaningDNA['R'] =  5;
        meaningDNA['S'] =  6;
        meaningDNA['T'] =  8;
        meaningDNA['U'] =  8;
        meaningDNA['V'] =  7;
        meaningDNA['W'] =  9;
        meaningDNA['X'] = 15;
        meaningDNA['Y'] = 10;
        meaningDNA['-'] = 15;
        meaningDNA['?'] = 15;

        for(i = 0; i < length; i++)
        {
          unsigned char n1 = meaningDNA[allowedStates[i][0]];
          unsigned char n2 = meaningDNA[allowedStates[i][1]];

          new = n1;
          new = new << 4;
          new = new | n2;

          intermediateBinaryStates[i] = new;
        }

        new = v1;
        new = new << 4;
        new = new | v2;

        for(i = 0; i < length; i++)
        {
          if(new == intermediateBinaryStates[i])
            break;
        }
        if(i < length)
          new = finalBinaryStates[i];
        else
        {
          new = 0;
          for(i = 0; i < length; i++)
          {
            if(v1 & meaningDNA[allowedStates[i][0]])
            {
              /*printf("Adding %c%c\n", allowedStates[i][0], allowedStates[i][1]);*/
              new |= finalBinaryStates[i];
            }
            if(v2 & meaningDNA[allowedStates[i][1]])
            {
              /*printf("Adding %c%c\n", allowedStates[i][0], allowedStates[i][1]);*/
              new |= finalBinaryStates[i];
            }
          }
        }	
      }
      break;
    case SECONDARY_DATA_7:
      {
        int
          meaningDNA[256],
          i;

        const unsigned char
          allowedStates[6][2] = {{'A','T'}, {'C', 'G'}, {'G', 'C'}, {'G','T'}, {'T', 'A'}, {'T', 'G'}};

        const unsigned char
          finalBinaryStates[7] = {1, 2, 4, 8, 16, 32, 64};

        unsigned char
          intermediateBinaryStates[7];

        for(i = 0; i < 256; i++)
          meaningDNA[i] = -1;

        meaningDNA['A'] =  1;
        meaningDNA['B'] = 14;
        meaningDNA['C'] =  2;
        meaningDNA['D'] = 13;
        meaningDNA['G'] =  4;
        meaningDNA['H'] = 11;
        meaningDNA['K'] = 12;
        meaningDNA['M'] =  3;
        meaningDNA['N'] = 15;
        meaningDNA['O'] = 15;
        meaningDNA['R'] =  5;
        meaningDNA['S'] =  6;
        meaningDNA['T'] =  8;
        meaningDNA['U'] =  8;
        meaningDNA['V'] =  7;
        meaningDNA['W'] =  9;
        meaningDNA['X'] = 15;
        meaningDNA['Y'] = 10;
        meaningDNA['-'] = 15;
        meaningDNA['?'] = 15;


        for(i = 0; i < 6; i++)
        {
          unsigned char n1 = meaningDNA[allowedStates[i][0]];
          unsigned char n2 = meaningDNA[allowedStates[i][1]];

          new = n1;
          new = new << 4;
          new = new | n2;

          intermediateBinaryStates[i] = new;
        }

        new = v1;
        new = new << 4;
        new = new | v2;

        for(i = 0; i < 6; i++)
        {
          /* exact match */
          if(new == intermediateBinaryStates[i])
            break;
        }
        if(i < 6)
          new = finalBinaryStates[i];
        else
        {
          /* distinguish between exact mismatches and partial mismatches */

          for(i = 0; i < 6; i++)
            if((v1 & meaningDNA[allowedStates[i][0]]) && (v2 & meaningDNA[allowedStates[i][1]]))
              break;
          if(i < 6)
          {
            /* printf("partial mismatch\n"); */

            new = 0;
            for(i = 0; i < 6; i++)
            {
              if((v1 & meaningDNA[allowedStates[i][0]]) && (v2 & meaningDNA[allowedStates[i][1]]))
              {
                /*printf("Adding %c%c\n", allowedStates[i][0], allowedStates[i][1]);*/
                new |= finalBinaryStates[i];
              }
              else
                new |=  finalBinaryStates[6];
            }
          }
          else
            new = finalBinaryStates[6];
        }	
      }
      break;
    default:
      assert(0);
  }

  return new;

}



static void adaptRdataToSecondary(tree *tr, rawdata *rdta)
{
  int *alias = (int*)calloc(rdta->sites, sizeof(int));
  int i, j, realPosition;  

  for(i = 0; i < rdta->sites; i++)
    alias[i] = -1;

  for(i = 0, realPosition = 0; i < rdta->sites; i++)
  {
    int partner = tr->secondaryStructurePairs[i];
    if(partner != -1)
    {
      assert(tr->dataVector[i+1] == SECONDARY_DATA || tr->dataVector[i+1] == SECONDARY_DATA_6 || tr->dataVector[i+1] == SECONDARY_DATA_7);

      if(i < partner)
      {
        for(j = 1; j <= rdta->numsp; j++)
        {
          unsigned char v1 = rdta->y[j][i+1];
          unsigned char v2 = rdta->y[j][partner+1];

          assert(i+1 < partner+1);

          rdta->y[j][i+1] = buildStates(tr->dataVector[i+1], v1, v2);
        }
        alias[realPosition] = i;
        realPosition++;
      }
    }
    else
    {
      alias[realPosition] = i;
      realPosition++;
    }
  }

  assert(rdta->sites - realPosition == tr->numberOfSecondaryColumns / 2);

  rdta->sites = realPosition;

  for(i = 0; i < rdta->sites; i++)
  {
    assert(alias[i] != -1);
    tr->model[i+1]    = tr->model[alias[i]+1];
    tr->dataVector[i+1] = tr->dataVector[alias[i]+1];
    rdta->wgt[i+1] =  rdta->wgt[alias[i]+1];

    for(j = 1; j <= rdta->numsp; j++)
      rdta->y[j][i+1] = rdta->y[j][alias[i]+1];
  }

  free(alias);
}

static void sitesort(rawdata *rdta, cruncheddata *cdta, tree *tr, analdef *adef)
{
  int  gap, i, j, jj, jg, k, n, nsp;
  int  
    *index, 
    *category = (int*)NULL;

  boolean  flip, tied;
  unsigned char  **data;

  if(adef->useSecondaryStructure)
  {
    assert(tr->NumberOfModels > 1 && adef->useMultipleModel);

    adaptRdataToSecondary(tr, rdta);
  }

  if(adef->useMultipleModel)    
    category      = tr->model;


  index    = cdta->alias;
  data     = rdta->y;
  n        = rdta->sites;
  nsp      = rdta->numsp;
  index[0] = -1;


  if(adef->compressPatterns)
  {
    for (gap = n / 2; gap > 0; gap /= 2)
    {
      for (i = gap + 1; i <= n; i++)
      {
        j = i - gap;

        do
        {
          jj = index[j];
          jg = index[j+gap];
          if(adef->useMultipleModel)
          {		     		      
            assert(category[jj] != -1 &&
                category[jg] != -1);

            flip = (category[jj] > category[jg]);
            tied = (category[jj] == category[jg]);		     

          }
          else
          {
            flip = 0;
            tied = 1;
          }

          for (k = 1; (k <= nsp) && tied; k++)
          {
            flip = (data[k][jj] >  data[k][jg]);
            tied = (data[k][jj] == data[k][jg]);
          }

          if (flip)
          {
            index[j]     = jg;
            index[j+gap] = jj;
            j -= gap;
          }
        }
        while (flip && (j > 0));
      }
    }
  }
}


static void sitecombcrunch (rawdata *rdta, cruncheddata *cdta, tree *tr, analdef *adef)
{
  int  i, sitei, j, sitej, k;
  boolean  tied;
  int 
    *aliasModel = (int*)NULL,
    *aliasSuperModel = (int*)NULL;

  if(adef->useMultipleModel)
  {
    aliasSuperModel = (int*)malloc(sizeof(int) * (rdta->sites + 1));
    aliasModel      = (int*)malloc(sizeof(int) * (rdta->sites + 1));
  } 

  i = 0;
  cdta->alias[0]    = cdta->alias[1];
  cdta->aliaswgt[0] = 0;

  if(adef->mode == PER_SITE_LL)
  {
    int i;

    assert(0);

    tr->patternPosition = (int*)malloc(sizeof(int) * rdta->sites);
    tr->columnPosition  = (int*)malloc(sizeof(int) * rdta->sites);

    for(i = 0; i < rdta->sites; i++)
    {
      tr->patternPosition[i] = -1;
      tr->columnPosition[i]  = -1;
    }
  }



  i = 0;
  for (j = 1; j <= rdta->sites; j++)
  {
    sitei = cdta->alias[i];
    sitej = cdta->alias[j];
    if(!adef->compressPatterns)
      tied = 0;
    else
    {
      if(adef->useMultipleModel)
      {	     
        tied = (tr->model[sitei] == tr->model[sitej]);
        if(tied)
          assert(tr->dataVector[sitei] == tr->dataVector[sitej]);
      }
      else
        tied = 1;
    }

    for (k = 1; tied && (k <= rdta->numsp); k++)
      tied = (rdta->y[k][sitei] == rdta->y[k][sitej]);

    if (tied)
    {
      if(adef->mode == PER_SITE_LL)
      {
        tr->patternPosition[j - 1] = i;
        tr->columnPosition[j - 1] = sitej;
        /*printf("Pattern %d from column %d also at site %d\n", i, sitei, sitej);*/
      }


      cdta->aliaswgt[i] += rdta->wgt[sitej];
      if(adef->useMultipleModel)
      {
        aliasModel[i]      = tr->model[sitej];
        aliasSuperModel[i] = tr->dataVector[sitej];
      }
    }
    else
    {
      if (cdta->aliaswgt[i] > 0) i++;

      if(adef->mode == PER_SITE_LL)
      {
        tr->patternPosition[j - 1] = i;
        tr->columnPosition[j - 1] = sitej;
        /*printf("Pattern %d is from cloumn %d\n", i, sitej);*/
      }

      cdta->aliaswgt[i] = rdta->wgt[sitej];
      cdta->alias[i] = sitej;
      if(adef->useMultipleModel)
      {
        aliasModel[i]      = tr->model[sitej];
        aliasSuperModel[i] = tr->dataVector[sitej];
      }
    }
  }

  cdta->endsite = i;
  if (cdta->aliaswgt[i] > 0) cdta->endsite++;

  if(adef->mode == PER_SITE_LL)
  {
    assert(0);

    for(i = 0; i < rdta->sites; i++)
    {
      int p  = tr->patternPosition[i];
      int c  = tr->columnPosition[i];

      assert(p >= 0 && p < cdta->endsite);
      assert(c >= 1 && c <= rdta->sites);
    }
  }


  if(adef->useMultipleModel)
  {
    for(i = 0; i <= rdta->sites; i++)
    {
      tr->model[i]      = aliasModel[i];
      tr->dataVector[i] = aliasSuperModel[i];
    }
  }

  if(adef->useMultipleModel)
  {
    free(aliasModel);
    free(aliasSuperModel);
  }     
}


static boolean makeweights (analdef *adef, rawdata *rdta, cruncheddata *cdta, tree *tr)
{
  int  i;

  if(adef->readBinaryFile)
  { 
    myBinFread(cdta->alias,    sizeof(int), (rdta->sites + 1));
    myBinFread(cdta->aliaswgt, sizeof(int), (rdta->sites + 1));
    myBinFread(tr->model,      sizeof(int), (rdta->sites + 1));
    myBinFread(tr->dataVector, sizeof(int), (rdta->sites + 1));
    myBinFread(&(cdta->endsite), sizeof(int), 1);
  }
  else
  {
    for (i = 1; i <= rdta->sites; i++)
      cdta->alias[i] = i;

    sitesort(rdta, cdta, tr, adef);
    sitecombcrunch(rdta, cdta, tr, adef);

    if(adef->writeBinaryFile)
    {
      myBinFwrite(cdta->alias,    sizeof(int), (rdta->sites + 1));
      myBinFwrite(cdta->aliaswgt, sizeof(int), (rdta->sites + 1));
      myBinFwrite(tr->model,      sizeof(int), (rdta->sites + 1));
      myBinFwrite(tr->dataVector, sizeof(int), (rdta->sites + 1));
      myBinFwrite(&(cdta->endsite), sizeof(int), 1);
    }
  }

  return TRUE;
}




static boolean makevalues(rawdata *rdta, cruncheddata *cdta, tree *tr, analdef *adef)
{
  int  
    i, 
    j, 
    model, 
    modelCounter;

  unsigned char
    *y    = (unsigned char *)malloc(((size_t)rdta->numsp) * ((size_t)cdta->endsite) * sizeof(unsigned char));


  /*

     printf("compressed data Assigning %Zu bytes\n", ((size_t)rdta->numsp) * ((size_t)cdta->endsite) * sizeof(unsigned char));

*/

  if(adef->readBinaryFile)
    myBinFread(y, sizeof(unsigned char), ((size_t)rdta->numsp) * ((size_t)cdta->endsite) * sizeof(unsigned char));
  else
  {
    for (i = 1; i <= rdta->numsp; i++)
      for (j = 0; j < cdta->endsite; j++)   
        y[(((size_t)(i - 1)) * ((size_t)cdta->endsite)) + j] = rdta->y[i][cdta->alias[j]];

    /*
       printf("Free on raw data\n");
       */

    free(rdta->y0);
    free(rdta->y);

    if(adef->writeBinaryFile)
      myBinFwrite(y, sizeof(unsigned char), ((size_t)rdta->numsp) * ((size_t)cdta->endsite) * sizeof(unsigned char));
  }

  rdta->y0 = y;

  if(!adef->useMultipleModel)
    tr->NumberOfModels = 1;

  if(adef->useMultipleModel)
  {
    tr->partitionData[0].lower = 0;

    model        = tr->model[0];
    modelCounter = 0;

    i            = 1;

    while(i <  cdta->endsite)
    {
      if(tr->model[i] != model)
      {
        tr->partitionData[modelCounter].upper     = i;
        tr->partitionData[modelCounter + 1].lower = i;

        model = tr->model[i];	     
        modelCounter++;
      }
      i++;
    }


    tr->partitionData[tr->NumberOfModels - 1].upper = cdta->endsite;      

    for(i = 0; i < tr->NumberOfModels; i++)		  
      tr->partitionData[i].width      = tr->partitionData[i].upper -  tr->partitionData[i].lower;

    model        = tr->model[0];
    modelCounter = 0;
    tr->model[0] = modelCounter;
    i            = 1;

    while(i < cdta->endsite)
    {	 
      if(tr->model[i] != model)
      {
        model = tr->model[i];
        modelCounter++;
        tr->model[i] = modelCounter;
      }
      else
        tr->model[i] = modelCounter;
      i++;
    }      
  }
  else
  {
    tr->partitionData[0].lower = 0;
    tr->partitionData[0].upper = cdta->endsite;
    tr->partitionData[0].width =  tr->partitionData[0].upper -  tr->partitionData[0].lower;
  }

  tr->rdta       = rdta;
  tr->cdta       = cdta; 

  tr->originalCrunchedLength = tr->cdta->endsite;

  for(i = 0; i < rdta->numsp; i++)
    tr->yVector[i + 1] = &(rdta->y0[((size_t)tr->originalCrunchedLength) * ((size_t)i)]);

  return TRUE;
}




static void allocPartitions(tree *tr)
{
  int
    i,
    maxCategories = tr->maxCategories;

  for(i = 0; i < tr->NumberOfModels; i++)
  {
    const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[i]));

    size_t 
      k,
      width = tr->partitionData[i].width;      

    tr->partitionData[i].perSiteAAModel = (int *)malloc(sizeof(int) * width);
    for(k = 0; k < width; k++)
      tr->partitionData[i].perSiteAAModel[k] = WAG;

    tr->partitionData[i].wr = (double *)malloc(sizeof(double) * width);
    tr->partitionData[i].wr2 = (double *)malloc(sizeof(double) * width);     


    tr->partitionData[i].globalScaler    = (unsigned int *)calloc(2 * tr->mxtips, sizeof(unsigned int));  	         

    tr->partitionData[i].left              = (double *)malloc_aligned(pl->leftLength * (maxCategories + 1) * sizeof(double));
    tr->partitionData[i].right             = (double *)malloc_aligned(pl->rightLength * (maxCategories + 1) * sizeof(double));
    tr->partitionData[i].EIGN              = (double*)malloc(pl->eignLength * sizeof(double));
    tr->partitionData[i].EV                = (double*)malloc_aligned(pl->evLength * sizeof(double));
    tr->partitionData[i].EI                = (double*)malloc(pl->eiLength * sizeof(double));
    tr->partitionData[i].substRates        = (double *)malloc(pl->substRatesLength * sizeof(double));
    tr->partitionData[i].frequencies       = (double*)malloc(pl->frequenciesLength * sizeof(double));
    tr->partitionData[i].tipVector         = (double *)malloc_aligned(pl->tipVectorLength * sizeof(double));
    tr->partitionData[i].symmetryVector    = (int *)malloc(pl->symmetryVectorLength  * sizeof(int));
    tr->partitionData[i].frequencyGrouping = (int *)malloc(pl->frequencyGroupingLength  * sizeof(int));
    tr->partitionData[i].perSiteRates      = (double *)malloc(sizeof(double) * tr->maxCategories);

    tr->partitionData[i].nonGTR = FALSE;             

    tr->partitionData[i].gammaRates = (double*)malloc(sizeof(double) * 4);
    tr->partitionData[i].yVector = (unsigned char **)malloc(sizeof(unsigned char*) * (tr->mxtips + 1));


    tr->partitionData[i].xVector = (double **)malloc(sizeof(double*) * tr->innerNodes);      

    tr->partitionData[i].xSpaceVector = (size_t *)calloc(tr->innerNodes, sizeof(size_t));    

    tr->partitionData[i].mxtips  = tr->mxtips;




#if ! (defined(_USE_PTHREADS) || defined(_FINE_GRAIN_MPI))
    {
      int j;

      for(j = 1; j <= tr->mxtips; j++)
        tr->partitionData[i].yVector[j] = &(tr->yVector[j][tr->partitionData[i].lower]);
    }
#endif

  }
}

#if ! (defined(_USE_PTHREADS) || defined(_FINE_GRAIN_MPI))





static void allocNodex (tree *tr)
{
  size_t
    rateHet,
    i,   
    model,
    offset,
    memoryRequirements = 0;

  allocPartitions(tr);

  if(tr->useRecom)
    allocRecompVectorsInfo(tr);
  else
    tr->rvec = (recompVectors*)NULL;



  if(tr->rateHetModel == CAT)
    rateHet = 1;
  else
    rateHet = 4;


  for(model = 0; model < (size_t)tr->NumberOfModels; model++)
  {
    size_t width = tr->partitionData[model].upper - tr->partitionData[model].lower;

    memoryRequirements += (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width;  

    /* this seems to be gap-saving related code only */
    {
      int 
        undetermined, 
        j;

      tr->partitionData[model].gapVectorLength = ((int)width / 32) + 1;

      tr->partitionData[model].gapVector = (unsigned int*)calloc(tr->partitionData[model].gapVectorLength * 2 * tr->mxtips, sizeof(unsigned int));

      tr->partitionData[model].initialGapVectorSize = tr->partitionData[model].gapVectorLength * 2 * tr->mxtips * sizeof(int);

      tr->partitionData[model].gapColumn = (double *)malloc_aligned(((size_t)tr->innerNodes) *								      
          ((size_t)(tr->partitionData[model].states)) *
          rateHet *
          sizeof(double));		  		

      undetermined = getUndetermined(tr->partitionData[model].dataType);

      for(j = 1; j <= tr->mxtips; j++)
        for(i = 0; i < width; i++)
          if(tr->partitionData[model].yVector[j][i] == undetermined)
            tr->partitionData[model].gapVector[tr->partitionData[model].gapVectorLength * j + i / 32] |= mask32[i % 32];
    }

  }

  tr->perSiteLL       = (double *)malloc((size_t)tr->cdta->endsite * sizeof(double));
  assert(tr->perSiteLL != NULL);


  tr->sumBuffer  = (double *)malloc_aligned(memoryRequirements * sizeof(double));
  assert(tr->sumBuffer != NULL);


  assert(4 * sizeof(double) > sizeof(parsimonyVector));

  offset = 0;

  /* C-OPT for initial testing tr->NumberOfModels will be 1 */

  for(model = 0; model < (size_t)tr->NumberOfModels; model++)
  {
    size_t lower = tr->partitionData[model].lower;
    size_t width = tr->partitionData[model].upper - lower;

    /* TODO all of this must be reset/adapted when fixModelIndices is called ! */


    tr->partitionData[model].sumBuffer       = &tr->sumBuffer[offset];


    tr->partitionData[model].perSiteLL    = &tr->perSiteLL[lower];        


    tr->partitionData[model].wgt          = &tr->cdta->aliaswgt[lower];

    tr->partitionData[model].rateCategory = &tr->cdta->rateCategory[lower];

    offset += (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width;      
  }



  for(i = 0; i < tr->innerNodes; i++)
  {     
    for(model = 0; model < (size_t)tr->NumberOfModels; model++)	    		
      tr->partitionData[model].xVector[i]   = (double*)NULL;	      	   	 
  }

}

#endif


static void initAdef(analdef *adef)
{  
  adef->useSecondaryStructure  = FALSE;
  adef->bootstrapBranchLengths = FALSE;
  adef->model                  = M_GTRCAT;
  adef->max_rearrange          = 21;
  adef->stepwidth              = 5;
  adef->initial                = adef->bestTrav = 10;
  adef->initialSet             = FALSE;
  adef->restart                = FALSE;
  adef->mode                   = BIG_RAPID_MODE;
  adef->categories             = 25;
  adef->boot                   = 0;
  adef->rapidBoot              = 0;
  adef->useWeightFile          = FALSE;
  adef->checkpoints            = 0;
  adef->startingTreeOnly       = 0;
  adef->multipleRuns           = 1;
  adef->useMultipleModel       = FALSE;
  adef->likelihoodEpsilon      = 0.1;
  adef->constraint             = FALSE;
  adef->grouping               = FALSE;
  adef->randomStartingTree     = FALSE;
  adef->parsimonySeed          = 0;
  adef->proteinMatrix          = JTT;
  adef->protEmpiricalFreqs     = 0;
  adef->outgroup               = FALSE;
  adef->useInvariant           = FALSE;
  adef->permuteTreeoptimize    = FALSE;
  adef->useInvariant           = FALSE;
  adef->allInOne               = FALSE;
  adef->likelihoodTest         = FALSE;
  adef->perGeneBranchLengths   = FALSE;
  adef->generateBS             = FALSE;
  adef->bootStopping           = FALSE;
  adef->gapyness               = 0.0;
  adef->similarityFilterMode   = 0;
  adef->useExcludeFile         = FALSE;
  adef->userProteinModel       = FALSE;
  adef->externalAAMatrix       = (double*)NULL;
  adef->computeELW             = FALSE;
  adef->computeDistance        = FALSE;
  adef->thoroughInsertion      = FALSE;
  adef->compressPatterns       = TRUE; 
  adef->readTaxaOnly           = FALSE;
  adef->meshSearch             = 0;
  adef->useCheckpoint          = FALSE;
  adef->leaveDropMode          = FALSE;
  adef->slidingWindowSize      = 100;
  adef->writeBinaryFile        = FALSE;
  adef->readBinaryFile         = FALSE; 
}




static int modelExists(char *model, analdef *adef)
{
  int i;
  char thisModel[1024];

  /********** BINARY ********************/

  if(strcmp(model, "BINGAMMAI\0") == 0)
  {
    adef->model = M_BINGAMMA;
    adef->useInvariant = TRUE;
    return 1;
  }

  if(strcmp(model, "BINGAMMA\0") == 0)
  {
    adef->model = M_BINGAMMA;
    adef->useInvariant = FALSE;
    return 1;
  }

  if(strcmp(model, "BINCAT\0") == 0)
  {
    adef->model = M_BINCAT;
    adef->useInvariant = FALSE;
    return 1;
  }

  if(strcmp(model, "BINCATI\0") == 0)
  {
    adef->model = M_BINCAT;
    adef->useInvariant = TRUE;
    return 1;
  }

  /*********** 32 state ****************************/

  if(strcmp(model, "MULTIGAMMAI\0") == 0)
  {
    adef->model = M_32GAMMA;
    adef->useInvariant = TRUE;
    return 1;
  }

  if(strcmp(model, "MULTIGAMMA\0") == 0)
  {
    adef->model = M_32GAMMA;
    adef->useInvariant = FALSE;
    return 1;
  }

  if(strcmp(model, "MULTICAT\0") == 0)
  {
    adef->model = M_32CAT;
    adef->useInvariant = FALSE;
    return 1;
  }

  if(strcmp(model, "MULTICATI\0") == 0)
  {
    adef->model = M_32CAT;
    adef->useInvariant = TRUE;
    return 1;
  }

  /*********** 64 state ****************************/

  if(strcmp(model, "CODONGAMMAI\0") == 0)
  {
    adef->model = M_64GAMMA;
    adef->useInvariant = TRUE;
    return 1;
  }

  if(strcmp(model, "CODONGAMMA\0") == 0)
  {
    adef->model = M_64GAMMA;
    adef->useInvariant = FALSE;
    return 1;
  }

  if(strcmp(model, "CODONCAT\0") == 0)
  {
    adef->model = M_64CAT;
    adef->useInvariant = FALSE;
    return 1;
  }

  if(strcmp(model, "CODONCATI\0") == 0)
  {
    adef->model = M_64CAT;
    adef->useInvariant = TRUE;
    return 1;
  }


  /*********** DNA **********************/

  if(strcmp(model, "GTRGAMMAI\0") == 0)
  {
    adef->model = M_GTRGAMMA;
    adef->useInvariant = TRUE;
    return 1;
  }

  if(strcmp(model, "GTRGAMMA\0") == 0)
  {
    adef->model = M_GTRGAMMA;
    adef->useInvariant = FALSE;
    return 1;
  }

  if(strcmp(model, "GTRGAMMA_FLOAT\0") == 0)
  {
    adef->model = M_GTRGAMMA;
    adef->useInvariant = FALSE;      
    return 1;
  }

  if(strcmp(model, "GTRCAT\0") == 0)
  {
    adef->model = M_GTRCAT;
    adef->useInvariant = FALSE;
    return 1;
  }

  if(strcmp(model, "GTRCAT_FLOAT\0") == 0)
  {
    adef->model = M_GTRCAT;
    adef->useInvariant = FALSE;      
    return 1;

  }

  if(strcmp(model, "GTRCATI\0") == 0)
  {
    adef->model = M_GTRCAT;
    adef->useInvariant = TRUE;
    return 1;
  }




  /*************** AA GTR ********************/

  /* TODO empirical FREQS */

  if(strcmp(model, "PROTCATGTR\0") == 0)
  {
    adef->model = M_PROTCAT;
    adef->proteinMatrix = GTR;
    adef->useInvariant = FALSE;
    return 1;
  }

  if(strcmp(model, "PROTCATIGTR\0") == 0)
  {
    adef->model = M_PROTCAT;
    adef->proteinMatrix = GTR;
    adef->useInvariant = TRUE;
    return 1;
  }

  if(strcmp(model, "PROTGAMMAGTR\0") == 0)
  {
    adef->model = M_PROTGAMMA;
    adef->proteinMatrix = GTR;
    adef->useInvariant = FALSE;
    return 1;
  }

  if(strcmp(model, "PROTGAMMAIGTR\0") == 0)
  {
    adef->model = M_PROTGAMMA;
    adef->proteinMatrix = GTR;
    adef->useInvariant = TRUE;
    return 1;
  }

  /****************** AA ************************/

  for(i = 0; i < NUM_PROT_MODELS - 1; i++)
  {
    /* check CAT */

    strcpy(thisModel, "PROTCAT");
    strcat(thisModel, protModels[i]);

    if(strcmp(model, thisModel) == 0)
    {
      adef->model = M_PROTCAT;
      adef->proteinMatrix = i;
      return 1;
    }

    /* check CATF */

    strcpy(thisModel, "PROTCAT");
    strcat(thisModel, protModels[i]);
    strcat(thisModel, "F");

    if(strcmp(model, thisModel) == 0)
    {
      adef->model = M_PROTCAT;
      adef->proteinMatrix = i;
      adef->protEmpiricalFreqs = 1;
      return 1;
    }

    /* check CAT FLOAT */

    strcpy(thisModel, "PROTCAT");
    strcat(thisModel, protModels[i]);
    strcat(thisModel, "_FLOAT");

    if(strcmp(model, thisModel) == 0)
    {
      adef->model = M_PROTCAT;
      adef->proteinMatrix = i;

      return 1;
    }

    /* check CATF FLOAT */

    strcpy(thisModel, "PROTCAT");
    strcat(thisModel, protModels[i]);
    strcat(thisModel, "F");
    strcat(thisModel, "_FLOAT");

    if(strcmp(model, thisModel) == 0)
    {
      adef->model = M_PROTCAT;
      adef->proteinMatrix = i;
      adef->protEmpiricalFreqs = 1;

      return 1;
    }

    /* check CATI */

    strcpy(thisModel, "PROTCATI");
    strcat(thisModel, protModels[i]);

    if(strcmp(model, thisModel) == 0)
    {
      adef->model = M_PROTCAT;
      adef->proteinMatrix = i;
      adef->useInvariant = TRUE;
      return 1;
    }

    /* check CATIF */

    strcpy(thisModel, "PROTCATI");
    strcat(thisModel, protModels[i]);
    strcat(thisModel, "F");

    if(strcmp(model, thisModel) == 0)
    {
      adef->model = M_PROTCAT;
      adef->proteinMatrix = i;
      adef->protEmpiricalFreqs = 1;
      adef->useInvariant = TRUE;
      return 1;
    }


    /****************check GAMMA ************************/

    strcpy(thisModel, "PROTGAMMA");
    strcat(thisModel, protModels[i]);

    if(strcmp(model, thisModel) == 0)
    {
      adef->model = M_PROTGAMMA;
      adef->proteinMatrix = i;
      adef->useInvariant = FALSE;
      return 1;
    }

    /* check GAMMA FLOAT */

    strcpy(thisModel, "PROTGAMMA");
    strcat(thisModel, protModels[i]);
    strcat(thisModel, "_FLOAT");

    if(strcmp(model, thisModel) == 0)
    {
      adef->model = M_PROTGAMMA;
      adef->proteinMatrix = i;

      adef->useInvariant = FALSE;
      return 1;
    }


    /*check GAMMAI*/

    strcpy(thisModel, "PROTGAMMAI");
    strcat(thisModel, protModels[i]);

    if(strcmp(model, thisModel) == 0)
    {
      adef->model = M_PROTGAMMA;
      adef->proteinMatrix = i;
      adef->useInvariant = TRUE;
      return 1;
    }


    /* check GAMMAmodelF */

    strcpy(thisModel, "PROTGAMMA");
    strcat(thisModel, protModels[i]);
    strcat(thisModel, "F");

    if(strcmp(model, thisModel) == 0)
    {
      adef->model = M_PROTGAMMA;
      adef->proteinMatrix = i;
      adef->protEmpiricalFreqs = 1;
      adef->useInvariant = FALSE;
      return 1;
    }

    /* check GAMMAmodelF FLOAT*/

    strcpy(thisModel, "PROTGAMMA");
    strcat(thisModel, protModels[i]);
    strcat(thisModel, "F");
    strcat(thisModel, "_FLOAT");

    if(strcmp(model, thisModel) == 0)
    {
      adef->model = M_PROTGAMMA;
      adef->proteinMatrix = i;
      adef->protEmpiricalFreqs = 1;
      adef->useInvariant = FALSE;

      return 1;
    }

    /* check GAMMAImodelF */

    strcpy(thisModel, "PROTGAMMAI");
    strcat(thisModel, protModels[i]);
    strcat(thisModel, "F");

    if(strcmp(model, thisModel) == 0)
    {
      adef->model = M_PROTGAMMA;
      adef->proteinMatrix = i;
      adef->protEmpiricalFreqs = 1;
      adef->useInvariant = TRUE;
      return 1;
    }

  }

  /*********************************************************************************/



  return 0;
}



static int mygetopt(int argc, char **argv, char *opts, int *optind, char **optarg)
{
  static int sp = 1;
  register int c;
  register char *cp;

  if(sp == 1)
  {
    if(*optind >= argc || argv[*optind][0] != '-' || argv[*optind][1] == '\0')
      return -1;
  }
  else
  {
    if(strcmp(argv[*optind], "--") == 0)
    {
      *optind =  *optind + 1;
      return -1;
    }
  }

  c = argv[*optind][sp];
  if(c == ':' || (cp=strchr(opts, c)) == 0)
  {
    printf(": illegal option -- %c \n", c);
    if(argv[*optind][++sp] == '\0')
    {
      *optind =  *optind + 1;
      sp = 1;
    }
    return('?');
  }
  if(*++cp == ':')
  {
    if(argv[*optind][sp+1] != '\0')
    {
      *optarg = &argv[*optind][sp+1];
      *optind =  *optind + 1;
    }
    else
    {
      *optind =  *optind + 1;
      if(*optind >= argc)
      {
        printf(": option requires an argument -- %c\n", c);
        sp = 1;
        return('?');
      }
      else
      {
        *optarg = argv[*optind];
        *optind =  *optind + 1;
      }
    }
    sp = 1;
  }
  else
  {
    if(argv[*optind][++sp] == '\0')
    {
      sp = 1;
      *optind =  *optind + 1;
    }
    *optarg = 0;
  }
  return(c);
}

static void checkOutgroups(tree *tr, analdef *adef)
{
  if(adef->outgroup)
  {
    boolean found;
    int i, j;

    for(j = 0; j < tr->numberOfOutgroups; j++)
    {
      found = FALSE;
      for(i = 1; (i <= tr->mxtips) && !found; i++)
      {
        if(strcmp(tr->nameList[i], tr->outgroups[j]) == 0)
        {
          tr->outgroupNums[j] = i;
          printf("%d\n", i);
          found = TRUE;
        }
      }
      if(!found)
      {
        printf("Error, the outgroup name \"%s\" you specified can not be found in the alignment, exiting ....\n", tr->outgroups[j]);
        errorExit(-1);
      }
    }
  }

}

static void parseOutgroups(char *outgr, tree *tr)
{
  int count = 1, i, k;
  char name[nmlngth];

  i = 0;
  while(outgr[i] != '\0')
  {
    if(outgr[i] == ',')
      count++;
    i++;
  }

  tr->numberOfOutgroups = count;

  tr->outgroups = (char **)malloc(sizeof(char *) * count);

  for(i = 0; i < tr->numberOfOutgroups; i++)
    tr->outgroups[i] = (char *)malloc(sizeof(char) * nmlngth);

  tr->outgroupNums = (int *)malloc(sizeof(int) * count);

  i = 0;
  k = 0;
  count = 0;
  while(outgr[i] != '\0')
  {
    if(outgr[i] == ',')
    {
      name[k] = '\0';
      strcpy(tr->outgroups[count], name);
      count++;
      k = 0;
    }
    else
    {
      name[k] = outgr[i];
      k++;
    }
    i++;
  }

  name[k] = '\0';
  strcpy(tr->outgroups[count], name);

  for(i = 0; i < tr->numberOfOutgroups; i++)
    printf("%d %s \n", i, tr->outgroups[i]);


  printf("%s \n", name);
}


/*********************************** OUTGROUP STUFF END *********************************************************/


static void printVersionInfo(void)
{
  printf("\n\nThis is %s version %s released by Alexandros Stamatakis, Christian Goll, and Fernando Izquierdo-Carrasco (ole) in %s.\n\n",  programName, programVersion, programDate); 
}

static void printMinusFUsage(void)
{
  printf("\n");


  printf("              \"-f d\": new rapid hill-climbing \n");
  printf("                      DEFAULT: ON\n");

  printf("\n");

  printf("              \"-f o\": old and slower rapid hill-climbing without heuristic cutoff\n");

  printf("\n");

  printf("              DEFAULT for \"-f\": new rapid hill climbing\n");

  printf("\n");
}


static void printREADME(void)
{
  printVersionInfo();
  printf("\n");  
  printf("\nTo report bugs go to the RAxML google group at http://groups.google.com/group/raxml\n");
  printf("Please specify the exact invocation, details of the HW and operating system,\n");
  printf("as well as all error messages printed to screen.\n\n\n");

  printf("raxmlLight|raxmlLight-PTHREADS|raxmlLight-MPI|\n");
  printf("raxmlLight-AVX|raxmlLight-PTHREADS-AVX|raxmlLight-MPI-AVX\n");
  printf("      -s sequenceFileName| -G binarySequnceFile\n");
  printf("      -n outputFileName\n");
  printf("      -m substitutionModel\n");
  printf("      -t userStartingTree| -R binaryCheckpointFile\n");
  printf("      [-B]\n"); 
  printf("      [-c numberOfCategories]\n");
  printf("      [-D]\n");
  printf("      [-e likelihoodEpsilon] \n");
  printf("      [-f d|o]\n");   
  printf("      [-h]\n");
  printf("      [-i initialRearrangementSetting] \n");
  printf("      [-M]\n");
  printf("      [-o outGroupName1[,outGroupName2[,...]]] \n");
  printf("      [-P proteinModel]\n");
  printf("      [-q multipleModelFileName] \n");
#if (defined(_USE_PTHREADS) || (_FINE_GRAIN_MPI))
  printf("      [-Q]\n");
#endif
  printf("      [-r recomputationFraction]\n");
  printf("      [-S]\n");
  printf("      [-T numberOfThreads]\n");  
  printf("      [-v]\n"); 
  printf("      [-w outputDirectory] \n"); 
  printf("      [-X]\n");
  printf("\n");
  printf("      -B      Parse phylip file and conduct pattern compression, then store the output in a \n");
  printf("              binary file called sequenceFileName.binary that can be read via the \"-G\" option\n");
  printf("              ATTENTION: the \"-B\" option only works with the sequential version\n");
  printf("\n");
  printf("      -c      Specify number of distinct rate catgories for RAxML when modelOfEvolution\n");
  printf("              is set to GTRCAT\n");
  printf("              Individual per-site rates are categorized into numberOfCategories rate \n");
  printf("              categories to accelerate computations. \n");
  printf("\n");
  printf("              DEFAULT: 25\n");
  printf("\n");
  printf("      -D      ML search convergence criterion. This will break off ML searches if the relative \n");
  printf("              Robinson-Foulds distance between the trees obtained from two consecutive lazy SPR cycles\n");
  printf("              is smaller or equal to 1%s. Usage recommended for very large datasets in terms of taxa.\n", "%");
  printf("              On trees with more than 500 taxa this will yield execution time improvements of approximately 50%s\n",  "%");
  printf("              While yielding only slightly worse trees.\n");
  printf("\n");
  printf("              DEFAULT: OFF\n");    
  printf("\n");
  printf("      -e      set model optimization precision in log likelihood units for final\n");
  printf("              optimization of model parameters\n");
  printf("\n");
  printf("              DEFAULT: 0.1 \n"); 
  printf("\n");
  printf("      -f      select algorithm:\n");

  printMinusFUsage();

  printf("\n");
  printf("      -G      Read in a binary alignment file (instead of a text-based phylip file with \"-s\") that was previsouly\n");
  printf("              generated with the \"-B\" option. This can substantially save time spent in input parsing \n");
  printf("              for very large parallel runs\n");
  printf("\n");
  printf("      -h      Display this help message.\n");
  printf("\n");  
  printf("      -i      Initial rearrangement setting for the subsequent application of topological \n");
  printf("              changes phase\n");
  printf("\n");
  printf("      -m      Model of  Nucleotide or Amino Acid Substitution: \n");
  printf("\n"); 
  printf("              NUCLEOTIDES:\n\n");
  printf("                \"-m GTRCAT\"         : GTR + Optimization of substitution rates + Optimization of site-specific\n");
  printf("                                      evolutionary rates which are categorized into numberOfCategories distinct \n");
  printf("                                      rate categories for greater computational efficiency.\n");
  printf("                \"-m GTRGAMMA\"       : GTR + GAMMA model of rate heterogeneity. This uses 4 hard-coded discrete rates\n");
  printf("                                      to discretize the GAMMA distribution.\n");
  printf("\n");
  printf("              AMINO ACIDS:\n\n");
  printf("                \"-m PROTCATmatrixName[F]\"         : specified AA matrix + Optimization of substitution rates + Optimization of site-specific\n");
  printf("                                                    evolutionary rates which are categorized into numberOfCategories distinct \n");
  printf("                                                    rate categories for greater computational efficiency.\n");  
  printf("                \"-m PROTGAMMAmatrixName[F]\"       : specified AA matrix + GAMMA model of rate heterogeneity. This uses 4 hard-coded discrete rates\n");
  printf("                                                    to discretize the GAMMA distribution.\n");
  printf("\n");
  printf("                Available AA substitution models: DAYHOFF, DCMUT, JTT, MTREV, WAG, RTREV, CPREV, VT, BLOSUM62, MTMAM, LG, MTART, MTZOA,\n");
  printf("                PMB, HIVB, HIVW, JTTDCMUT, FLU, AUTO, GTR\n");
  printf("                With the optional \"F\" appendix you can specify if you want to use empirical base frequencies\n");
  printf("                Please note that for mixed models you can in addition specify the per-gene AA model in\n");
  printf("                the mixed model file (see manual for details). Also note that if you estimate AA GTR parameters on a partitioned\n");
  printf("                dataset, they will be linked (estimated jointly) across all partitions to avoid over-parametrization\n");
  printf("                When AUTO is used RAxML will conduct an ML estimate of all available pre-defined AA models (excluding GTR) every time the model parameters\n");
  printf("                are optimized during the tree search.\n");
  printf("                WARNING: we have not figured out yet how to best do this for partitioned analyses, so don't use AUTO for datasets with partitions\n");
  printf("\n");
  printf("      -M      Switch on estimation of individual per-partition branch lengths. Only has effect when used in combination with \"-q\"\n");
  printf("              Branch lengths for individual partitions will be printed to separate files\n");
  printf("              A weighted average of the branch lengths is computed by using the respective partition lengths\n");
  printf("\n"),
    printf("              DEFAULT: OFF\n");
  printf("\n");
  printf("      -n      Specifies the name of the output file.\n");
  printf("\n");
  printf("      -o      Specify the name of a single outgrpoup or a comma-separated list of outgroups, eg \"-o Rat\" \n");
  printf("              or \"-o Rat,Mouse\", in case that multiple outgroups are not monophyletic the first name \n");
  printf("              in the list will be selected as outgroup, don't leave spaces between taxon names!\n"); 
  printf("\n"); 
  printf("      -P      Specify the file name of a user-defined AA (Protein) substitution model. This file must contain\n");
  printf("              420 entries, the first 400 being the AA substitution rates (this must be a symmetric matrix) and the\n");
  printf("              last 20 are the empirical base frequencies\n");
  printf("\n");
  printf("      -q      Specify the file name which contains the assignment of models to alignment\n");
  printf("              partitions for multiple models of substitution. For the syntax of this file\n");
  printf("              please consult the manual.\n");  
  printf("\n");
#if (defined(_USE_PTHREADS) || (_FINE_GRAIN_MPI))
  printf("      -Q      Enable alternative data/load distribution algorithm for datasets with many partitions\n");
  printf("              In particular under CAT this can lead to parallel performance improvements of over 50 per cent\n");
#endif
  printf("\n");
  printf("      -r      Specify the fraction of ancestral node vectors (%f <= R < %f) that will be allocated in RAM\n", MIN_RECOM_FRACTION, MAX_RECOM_FRACTION);
  printf("\n");
  printf("      -R      read in a binary checkpoint file called RAxML_binaryCheckpoint.RUN_ID_number\n");
  printf("\n");
  printf("      -s      Specify the name of the alignment data file in PHYLIP format\n");
  printf("\n");
  printf("      -S      turn on memory saving option for gappy multi-gene alignments. For large and gappy datasets specify -S to save memory\n");
  printf("              This will produce slightly different likelihood values, may be a bit slower but can reduce memory consumption\n");
  printf("              from 70GB to 19GB on very large and gappy datasets\n");
  printf("\n");
  printf("      -t      Specify a user starting tree file name in Newick format\n");
  printf("\n");
  printf("      -T      PTHREADS VERSION ONLY! Specify the number of threads you want to run.\n");
  printf("              Make sure to set \"-T\" to at most the number of CPUs you have on your machine,\n");
  printf("              otherwise, there will be a huge performance decrease!\n");
  printf("\n");  
  printf("      -v      Display version information\n");
  printf("\n");
  printf("      -w      FULL (!) path to the directory into which RAxML shall write its output files\n");
  printf("\n");
  printf("              DEFAULT: current directory\n"); 
  printf("\n");
  printf("      -X      EXPERIMENTAL OPTION: This option will do a per-site estimate of protein substitution models\n");
  printf("              by looping over all given, fixed models LG, WAG, JTT, etc and using their respective base frequencies to independently\n");
  printf("              assign a prot subst. model to each site via ML optimization\n");
  printf("              At present this option only works with the GTR+GAMMA model, unpartitioned datasets, and in the sequential\n");
  printf("              version only.\n");
  printf("\n\n\n\n");

}




static void analyzeRunId(char id[128])
{
  int i = 0;

  while(id[i] != '\0')
  {    
    if(i >= 128)
    {
      printf("Error: run id after \"-n\" is too long, it has %d characters please use a shorter one\n", i);
      assert(0);
    }

    if(id[i] == '/')
    {
      printf("Error character %c not allowed in run ID\n", id[i]);
      assert(0);
    }


    i++;
  }

  if(i == 0)
  {
    printf("Error: please provide a string for the run id after \"-n\" \n");
    assert(0);
  }

}

static void get_args(int argc, char *argv[], analdef *adef, tree *tr)
{
  boolean
    bad_opt    =FALSE,
               resultDirSet = FALSE;

  char
    resultDir[1024] = "",
    aut[256],         
    *optarg,
    model[2048] = "",
    secondaryModel[2048] = "",
    multiStateModel[2048] = "",
    modelChar;

  double 
    likelihoodEpsilon,    
    wcThreshold,
    fastEPAthreshold;

  int  
    optind = 1,        
           c,
           nameSet = 0,
           alignmentSet = 0,
           multipleRuns = 0,
           constraintSet = 0,
           treeSet = 0,
           groupSet = 0,
           modelSet = 0,
           treesSet  = 0;

  boolean
    bSeedSet = FALSE,
             xSeedSet = FALSE,
             multipleRunsSet = FALSE;

  run_id[0] = 0;
  workdir[0] = 0;
  seq_file[0] = 0;
  tree_file[0] = 0;
  model[0] = 0;
  weightFileName[0] = 0;
  modelFileName[0] = 0;

  /*********** tr inits **************/

#ifdef _USE_PTHREADS
  NumberOfThreads = 0;
#endif




  tr->bootStopCriterion = -1;
  tr->wcThreshold = 0.03;
  tr->doCutoff = TRUE;
  tr->secondaryStructureModel = SEC_16; /* default setting */
  tr->searchConvergenceCriterion = FALSE;
  tr->catOnly = FALSE;
  tr->fastEPA_ML = FALSE;
  tr->fastEPA_MP = FALSE;
  tr->fastEPAthreshold = -1.0;
  tr->multiStateModel  = GTR_MULTI_STATE;
  tr->useGappedImplementation = FALSE;
  tr->saveMemory = FALSE;
  tr->estimatePerSiteAA = FALSE;

  /* recom */
  tr->useRecom = FALSE;
  tr->rvec = (recompVectors*)NULL;
  /* recom */


#if (defined(_USE_PTHREADS) || (_FINE_GRAIN_MPI))
  tr->manyPartitions = FALSE;
#endif

  /********* tr inits end*************/

#if (defined(_USE_PTHREADS) || (_FINE_GRAIN_MPI))
  while(!bad_opt &&
      ((c = mygetopt(argc,argv,"T:P:R:e:c:f:i:m:t:w:s:n:o:q:r:G:vhMSDBQX", &optind, &optarg))!=-1))
#else
    while(!bad_opt &&
        ((c = mygetopt(argc,argv,"T:P:R:e:c:f:i:m:t:w:s:n:o:q:r:G:vhMSDBX", &optind, &optarg))!=-1))
#endif
    {
      switch(c)
      {
        case 'X':
          tr->estimatePerSiteAA = TRUE;       
          break;
#if (defined(_USE_PTHREADS) || (_FINE_GRAIN_MPI))	
        case 'Q':
          tr->manyPartitions = TRUE;
          break;
#endif
        case 'G':
          {
            char byteFileName[1024] = "";
            adef->readBinaryFile = TRUE;
            strcpy(byteFileName, optarg);
            byteFile = fopen(byteFileName, "rb");
          }
          break;
        case 'B':	
          adef->writeBinaryFile = TRUE;       
          break;
        case 'S':
          tr->saveMemory = TRUE;
          break;
        case 'D':
          tr->searchConvergenceCriterion = TRUE;	
          break;
        case 'R':
          adef->useCheckpoint = TRUE;
          strcpy(binaryCheckpointInputName, optarg);
          break;     

        case 'M':
          adef->perGeneBranchLengths = TRUE;
          break;
        case 'P':
          strcpy(proteinModelFileName, optarg);
          adef->userProteinModel = TRUE;
          parseProteinModel(adef);
          break;      
        case 'T':
#ifdef _USE_PTHREADS
          sscanf(optarg,"%d", &NumberOfThreads);
#else
          if(processID == 0)
          {
            printf("Option -T does not have any effect with the sequential or parallel MPI version.\n");
            printf("It is used to specify the number of threads for the Pthreads-based parallelization\n");
          }
#endif
          break;                  
        case 'o':
          {
            char *outgroups;
            outgroups = (char*)malloc(sizeof(char) * (strlen(optarg) + 1));
            strcpy(outgroups, optarg);
            parseOutgroups(outgroups, tr);
            free(outgroups);
            adef->outgroup = TRUE;
          }
          break;

        case 'e':
          sscanf(optarg,"%lf", &likelihoodEpsilon);
          adef->likelihoodEpsilon = likelihoodEpsilon;
          break;
        case 'q':
          strcpy(modelFileName,optarg);
          adef->useMultipleModel = TRUE;
          break;

        case 'r':
          sscanf(optarg, "%f", &tr->vectorRecomFraction);
          if(tr->vectorRecomFraction < MIN_RECOM_FRACTION || tr->vectorRecomFraction >= MAX_RECOM_FRACTION)
          {
            printf("Recomputation fraction passed via -r must be greater or equal to %f\n", MIN_RECOM_FRACTION);
            printf("and smaller than %f .... exiting \n", MAX_RECOM_FRACTION);
            exit(0);
          }
          tr->useRecom = TRUE;
          break;

        case 'v':
          printVersionInfo();
          errorExit(0);

        case 'h':
          printREADME();
          errorExit(0);

        case 'c':
          sscanf(optarg, "%d", &adef->categories);
          break;     
        case 'f':
          sscanf(optarg, "%c", &modelChar);
          switch(modelChar)
          {	 
            case 'd':
              adef->mode = BIG_RAPID_MODE;
              tr->doCutoff = TRUE;
              break;	  
            case 'o':
              adef->mode = BIG_RAPID_MODE;
              tr->doCutoff = FALSE;
              break;	    	  	  	     
            default:
              {
                if(processID == 0)
                {
                  printf("Error select one of the following algorithms via -f :\n");
                  printMinusFUsage();
                }
                errorExit(-1);
              }
          }
          break;
        case 'i':
          sscanf(optarg, "%d", &adef->initial);
          adef->initialSet = TRUE;
          break;
        case 'n':
          strcpy(run_id,optarg);
          analyzeRunId(run_id);
          nameSet = 1;
          break;
        case 'w':
          strcpy(resultDir, optarg);
          resultDirSet = TRUE;
          break;
        case 't':
          strcpy(tree_file, optarg);
          adef->restart = TRUE;
          treeSet = 1;
          break;
        case 's':
          strcpy(seq_file, optarg);
          alignmentSet = 1;
          break;
        case 'm':
          strcpy(model,optarg);
          if(modelExists(model, adef) == 0)
          {
            if(processID == 0)
            {
              printf("Model %s does not exist\n\n", model);               
              printf("For DNA data use:    GTRCAT                or GTRGAMMA                or\n");	
              printf("For AA data use:     PROTCATmatrixName[F]  or PROTGAMMAmatrixName[F]  or\n");		
              printf("The AA substitution matrix can be one of the following: \n");
              printf("DAYHOFF, DCMUT, JTT, MTREV, WAG, RTREV, CPREV, VT, BLOSUM62, MTMAM, LG, MTART, MTZOA, PMB, HIVB, HIVW, JTTDCMUT, FLU, AUTO, GTR\n\n");
              printf("With the optional \"F\" appendix you can specify if you want to use empirical base frequencies\n");
              printf("Please note that for mixed models you can in addition specify the per-gene model in\n");
              printf("the mixed model file (see manual for details)\n");
            }
            errorExit(-1);
          }
          else
            modelSet = 1;
          break;
        default:
          errorExit(-1);
      }
    }



#ifdef _USE_PTHREADS
  if(NumberOfThreads < 2)
  {
    printf("\nThe number of threads is currently set to %d\n", NumberOfThreads);
    printf("Specify the number of threads to run via -T numberOfThreads\n");
    printf("NumberOfThreads must be set to an integer value greater than 1\n\n");
    errorExit(-1);
  }
#endif

#if (defined(_USE_PTHREADS) || (_FINE_GRAIN_MPI))
  if(adef->writeBinaryFile)
  {
    if(processID == 0)
      printf("\n Error, parsing a standard alignment file and writing a binary one is not allowed with Pthreads and MPI versions\n");
    errorExit(-1);
  }
#endif

  if(adef->restart && adef->useCheckpoint)
  {
    if(processID == 0)
      printf("\n Error, you must either specify a starting tree via \"-t\" or a checkpoint file via \"-R\"\n");
    errorExit(-1);
  }


  if(!modelSet)
  {
    if(processID == 0)
      printf("\n Error, you must specify a model of substitution with the \"-m\" option\n");
    errorExit(-1);
  }

  if(!nameSet)
  {
    if(processID == 0)
      printf("\n Error: please specify a name for this run with -n\n");
    errorExit(-1);
  }

  if(!adef->readBinaryFile)
  {
    if(!alignmentSet)
    {
      if(processID == 0)
        printf("\n Error: please specify an alignment for this run with -s\n");
      errorExit(-1);
    }
  }
  else
  {
    if(alignmentSet)
    {
      if(processID == 0)
        printf("\n Error: you can't specify a normal alignment with -s and a binary one with -G at the same time\n");
      errorExit(-1);
    }
  }

  {
#ifdef WIN32
    const 
      char *separator = "\\";
#else
    const 
      char *separator = "/";
#endif

    if(resultDirSet)
    {
      char 
        dir[1024] = "";

#ifndef WIN32
      if(resultDir[0] != separator[0])
        strcat(dir, separator);
#endif

      strcat(dir, resultDir);

      if(dir[strlen(dir) - 1] != separator[0]) 
        strcat(dir, separator);
      strcpy(workdir, dir);
    }
    else
    {
      char 
        dir[1024] = "",
        *result = getcwd(dir, sizeof(dir));

      assert(result != (char*)NULL);

      if(dir[strlen(dir) - 1] != separator[0]) 
        strcat(dir, separator);

      strcpy(workdir, dir);		
    }
  }


  if(adef->writeBinaryFile)
  {
    char byteFileName[1024] = "";

    strcpy(byteFileName, workdir);
    strcat(byteFileName, seq_file);
    strcat(byteFileName, ".binary");

    if(filexists(byteFileName))
    {
      printf("\n\nError: Binary compressed file %s you want to generate already exists ... exiting\n\n", byteFileName);
      exit(0);
    }

    byteFile = fopen(byteFileName, "wb");
  }

  return;
}




void errorExit(int e)
{

#ifdef _WAYNE_MPI
  MPI_Finalize();
#endif

  exit(e);

}



static void makeFileNames(void)
{
  int infoFileExists = 0;


  strcpy(permFileName,         workdir);
  strcpy(resultFileName,       workdir);
  strcpy(logFileName,          workdir);
  strcpy(checkpointFileName,   workdir);
  strcpy(infoFileName,         workdir);
  strcpy(randomFileName,       workdir);
  strcpy(bootstrapFileName,    workdir);
  strcpy(bipartitionsFileName, workdir);
  strcpy(bipartitionsFileNameBranchLabels, workdir);
  strcpy(ratesFileName,        workdir);
  strcpy(lengthFileName,       workdir);
  strcpy(lengthFileNameModel,  workdir);
  strcpy(perSiteLLsFileName,  workdir);
  strcpy(binaryCheckpointName, workdir);


  strcat(permFileName,         "RAxML_parsimonyTree.");
  strcat(resultFileName,       "RAxML_result.");
  strcat(logFileName,          "RAxML_log.");
  strcat(checkpointFileName,   "RAxML_checkpoint.");
  strcat(infoFileName,         "RAxML_info.");
  strcat(randomFileName,       "RAxML_randomTree.");
  strcat(bootstrapFileName,    "RAxML_bootstrap.");
  strcat(bipartitionsFileName, "RAxML_bipartitions.");
  strcat(bipartitionsFileNameBranchLabels, "RAxML_bipartitionsBranchLabels.");
  strcat(ratesFileName,        "RAxML_perSiteRates.");
  strcat(lengthFileName,       "RAxML_treeLength.");
  strcat(lengthFileNameModel,  "RAxML_treeLengthModel.");
  strcat(perSiteLLsFileName,   "RAxML_perSiteLLs."); 
  strcat(binaryCheckpointName, "RAxML_binaryCheckpoint.");


  strcat(permFileName,         run_id);
  strcat(resultFileName,       run_id);
  strcat(logFileName,          run_id);
  strcat(checkpointFileName,   run_id);
  strcat(infoFileName,         run_id);
  strcat(randomFileName,       run_id);
  strcat(bootstrapFileName,    run_id);
  strcat(bipartitionsFileName, run_id);
  strcat(bipartitionsFileNameBranchLabels, run_id);  
  strcat(ratesFileName,        run_id);
  strcat(lengthFileName,       run_id);
  strcat(lengthFileNameModel,  run_id);
  strcat(perSiteLLsFileName,   run_id);  
  strcat(binaryCheckpointName, run_id);


#ifdef _WAYNE_MPI  
  {
    char buf[64];

    strcpy(bootstrapFileNamePID, bootstrapFileName);
    strcat(bootstrapFileNamePID, ".PID.");
    sprintf(buf, "%d", processID);
    strcat(bootstrapFileNamePID, buf);
  }
#endif

  if(processID == 0)
  {
    infoFileExists = filexists(infoFileName);

    if(infoFileExists)
    {
      printf("RAxML output files with the run ID <%s> already exist \n", run_id);
      printf("in directory %s ...... exiting\n", workdir);

      exit(-1);
    }
  }
}









/***********************reading and initializing input ******************/


/********************PRINTING various INFO **************************************/


static void printModelAndProgramInfo(tree *tr, analdef *adef, int argc, char *argv[])
{
  if(processID == 0)
  {
    int i, model;
    FILE *infoFile = myfopen(infoFileName, "ab");
    char modelType[128];

    if(!adef->readTaxaOnly)
    {
      if(adef->useInvariant)
        strcpy(modelType, "GAMMA+P-Invar");
      else
        strcpy(modelType, "GAMMA");
    }

    printBoth(infoFile, "\n\nThis is %s version %s released by Alexandros Stamatakis, Christian Goll, and Fernando Izquierdo-Carrasco (ole) in %s.\n\n",  programName, programVersion, programDate);


    if(!adef->readTaxaOnly)
    {
      if(!adef->compressPatterns)
        printBoth(infoFile, "\nAlignment has %d columns\n\n",  tr->cdta->endsite);
      else
        printBoth(infoFile, "\nAlignment has %d distinct alignment patterns\n\n",  tr->cdta->endsite);

      if(adef->useInvariant)
        printBoth(infoFile, "Found %d invariant alignment patterns that correspond to %d columns \n", tr->numberOfInvariableColumns, tr->weightOfInvariableColumns);

      printBoth(infoFile, "Proportion of gaps and completely undetermined characters in this alignment: %3.2f%s\n", 100.0 * adef->gapyness, "%");
    }

    switch(adef->mode)
    {
      case THOROUGH_PARSIMONY:
        printBoth(infoFile, "\nRAxML more exhaustive parsimony search with a ratchet.\n");
        printBoth(infoFile, "For a faster and better implementation of MP searches please use TNT by Pablo Goloboff.\n\n");
        break;
      case DISTANCE_MODE:
        printBoth(infoFile, "\nRAxML Computation of pairwise distances\n\n");
        break;
      case TREE_EVALUATION :
        printBoth(infoFile, "\nRAxML Model Optimization up to an accuracy of %f log likelihood units\n\n", adef->likelihoodEpsilon);
        break;
      case  BIG_RAPID_MODE:
        if(adef->rapidBoot)
        {
          if(adef->allInOne)
            printBoth(infoFile, "\nRAxML rapid bootstrapping and subsequent ML search\n\n");
          else
            printBoth(infoFile,  "\nRAxML rapid bootstrapping algorithm\n\n");
        }
        else
          printBoth(infoFile, "\nRAxML rapid hill-climbing mode\n\n");
        break;
      case CALC_BIPARTITIONS:
        printBoth(infoFile, "\nRAxML Bipartition Computation: Drawing support values from trees in file %s onto tree in file %s\n\n",
            bootStrapFile, tree_file);
        break;
      case PER_SITE_LL:
        printBoth(infoFile, "\nRAxML computation of per-site log likelihoods\n");
        break;
      case PARSIMONY_ADDITION:
        printBoth(infoFile, "\nRAxML stepwise MP addition to incomplete starting tree\n\n");
        break;
      case CLASSIFY_ML:
        printBoth(infoFile, "\nRAxML classification algorithm\n\n");
        break;
      case GENERATE_BS:
        printBoth(infoFile, "\nRAxML BS replicate generation\n\n");
        break;
      case COMPUTE_ELW:
        printBoth(infoFile, "\nRAxML ELW test\n\n");
        break;
      case BOOTSTOP_ONLY:
        printBoth(infoFile, "\nRAxML a posteriori Bootstrap convergence assessment\n\n");
        break;
      case CONSENSUS_ONLY:
        if(adef->leaveDropMode)
          printBoth(infoFile, "\nRAxML rogue taxa computation by Andre Aberer (TUM)\n\n");
        else
          printBoth(infoFile, "\nRAxML consensus tree computation\n\n");
        break;
      case COMPUTE_LHS:
        printBoth(infoFile, "\nRAxML computation of likelihoods for a set of trees\n\n");
        break;
      case COMPUTE_BIPARTITION_CORRELATION:
        printBoth(infoFile, "\nRAxML computation of bipartition support correlation on two sets of trees\n\n");
        break;
      case COMPUTE_RF_DISTANCE:
        printBoth(infoFile, "\nRAxML computation of RF distances for all pairs of trees in a set of trees\n\n");
        break;
      case MORPH_CALIBRATOR:
        printBoth(infoFile, "\nRAxML morphological calibrator using Maximum Likelihood\n\n");
        break;
      case MORPH_CALIBRATOR_PARSIMONY:
        printBoth(infoFile, "\nRAxML morphological calibrator using Parsimony\n\n");
        break;	  
      case MESH_TREE_SEARCH:
        printBoth(infoFile, "\nRAxML experimental mesh tree search\n\n");
        break;
      case FAST_SEARCH:
        printBoth(infoFile, "\nRAxML experimental very fast tree search\n\n");
        break;
      case SH_LIKE_SUPPORTS:
        printBoth(infoFile, "\nRAxML computation of SH-like support values on a given tree\n\n");
        break;
      case EPA_ROGUE_TAXA:
        printBoth(infoFile, "\nRAxML experimental statistical rogue taxon identification algorithm\n\n");
        break;
      case EPA_SITE_SPECIFIC_BIAS:
        printBoth(infoFile, "\nRAxML exprimental site-specfific phylogenetic placement bias analysis algorithm\n\n");
        break;
      default:
        assert(0);
    }

    if(adef->mode != THOROUGH_PARSIMONY)
    { 
      if(!adef->readTaxaOnly)
      {
        if(adef->perGeneBranchLengths)
          printBoth(infoFile, "Using %d distinct models/data partitions with individual per partition branch length optimization\n\n\n", tr->NumberOfModels);
        else
          printBoth(infoFile, "Using %d distinct models/data partitions with joint branch length optimization\n\n\n", tr->NumberOfModels);
      }
    }


    /*
       if(adef->mode == BIG_RAPID_MODE)
       {
       if(adef->rapidBoot)
       {
       if(adef->allInOne)
       printBoth(infoFile, "\nExecuting %d rapid bootstrap inferences and thereafter a thorough ML search \n\n", adef->multipleRuns);
       else
       printBoth(infoFile, "\nExecuting %d rapid bootstrap inferences\n\n", adef->multipleRuns);
       }
       else
       {
       if(adef->boot)
       printBoth(infoFile, "Executing %d non-parametric bootstrap inferences\n\n", adef->multipleRuns);
       else
       {
       char treeType[1024];

       if(adef->restart)
       strcpy(treeType, "user-specifed");
       else
       {
       if(adef->randomStartingTree)
       strcpy(treeType, "distinct complete random");
       else
       strcpy(treeType, "distinct randomized MP");
       }

       printBoth(infoFile, "Executing %d inferences on the original alignment using %d %s trees\n\n",
       adef->multipleRuns, adef->multipleRuns, treeType);
       }
       }
       }
       */


    if(!adef->readTaxaOnly)
    {
      if(adef->mode != THOROUGH_PARSIMONY)
        printBoth(infoFile, "All free model parameters will be estimated by RAxML\n");

      if(adef->mode != THOROUGH_PARSIMONY)
      {
        if(tr->rateHetModel == GAMMA || tr->rateHetModel == GAMMA_I)
          printBoth(infoFile, "%s model of rate heteorgeneity, ML estimate of alpha-parameter\n\n", modelType);
        else
        {
          printBoth(infoFile, "ML estimate of %d per site rate categories\n\n", adef->categories);
          /*
             if(adef->mode != CLASSIFY_ML)
             printBoth(infoFile, "Likelihood of final tree will be evaluated and optimized under %s\n\n", modelType);
             */
        }

        /*
           if(adef->mode != CLASSIFY_ML)
           printBoth(infoFile, "%s Model parameters will be estimated up to an accuracy of %2.10f Log Likelihood units\n\n",
           modelType, adef->likelihoodEpsilon);
           */
      }

      for(model = 0; model < tr->NumberOfModels; model++)
      {
        printBoth(infoFile, "Partition: %d\n", model);
        printBoth(infoFile, "Alignment Patterns: %d\n", tr->partitionData[model].upper - tr->partitionData[model].lower);
        printBoth(infoFile, "Name: %s\n", tr->partitionData[model].partitionName);

        switch(tr->partitionData[model].dataType)
        {
          case DNA_DATA:
            printBoth(infoFile, "DataType: DNA\n");
            if(adef->mode != THOROUGH_PARSIMONY)
              printBoth(infoFile, "Substitution Matrix: GTR\n");
            break;
          case AA_DATA:
            assert(tr->partitionData[model].protModels >= 0 && tr->partitionData[model].protModels < NUM_PROT_MODELS);
            printBoth(infoFile, "DataType: AA\n");
            if(adef->mode != THOROUGH_PARSIMONY)
            {
              printBoth(infoFile, "Substitution Matrix: %s\n", (adef->userProteinModel)?"External user-specified model":protModels[tr->partitionData[model].protModels]);
              printBoth(infoFile, "%s Base Frequencies:\n", (tr->partitionData[model].protFreqs == 1)?"Empirical":"Fixed");
            }
            break;
          case BINARY_DATA:
            printBoth(infoFile, "DataType: BINARY/MORPHOLOGICAL\n");
            if(adef->mode != THOROUGH_PARSIMONY)
              printBoth(infoFile, "Substitution Matrix: Uncorrected\n");
            break;
          case SECONDARY_DATA:
            printBoth(infoFile, "DataType: SECONDARY STRUCTURE\n");
            if(adef->mode != THOROUGH_PARSIMONY)
              printBoth(infoFile, "Substitution Matrix: %s\n", secondaryModelList[tr->secondaryStructureModel]);
            break;
          case SECONDARY_DATA_6:
            printBoth(infoFile, "DataType: SECONDARY STRUCTURE 6 STATE\n");
            if(adef->mode != THOROUGH_PARSIMONY)
              printBoth(infoFile, "Substitution Matrix: %s\n", secondaryModelList[tr->secondaryStructureModel]);
            break;
          case SECONDARY_DATA_7:
            printBoth(infoFile, "DataType: SECONDARY STRUCTURE 7 STATE\n");
            if(adef->mode != THOROUGH_PARSIMONY)
              printBoth(infoFile, "Substitution Matrix: %s\n", secondaryModelList[tr->secondaryStructureModel]);
            break;
          case GENERIC_32:
            printBoth(infoFile, "DataType: Multi-State with %d distinct states in use (maximum 32)\n",tr->partitionData[model].states);		  
            switch(tr->multiStateModel)
            {
              case ORDERED_MULTI_STATE:
                printBoth(infoFile, "Substitution Matrix: Ordered Likelihood\n");
                break;
              case MK_MULTI_STATE:
                printBoth(infoFile, "Substitution Matrix: MK model\n");
                break;
              case GTR_MULTI_STATE:
                printBoth(infoFile, "Substitution Matrix: GTR\n");
                break;
              default:
                assert(0);
            }
            break;
          case GENERIC_64:
            printBoth(infoFile, "DataType: Codon\n");		  
            break;		
          default:
            assert(0);
        }
        printBoth(infoFile, "\n\n\n");
      }
    }

    printBoth(infoFile, "\n");

    printBoth(infoFile, "RAxML was called as follows:\n\n");
    for(i = 0; i < argc; i++)
      printBoth(infoFile,"%s ", argv[i]);
    printBoth(infoFile,"\n\n\n");

    fclose(infoFile);
  }
}

void printResult(tree *tr, analdef *adef, boolean finalPrint)
{
  FILE *logFile;
  char temporaryFileName[1024] = "", treeID[64] = "";

  strcpy(temporaryFileName, resultFileName);

  switch(adef->mode)
  {
    case MORPH_CALIBRATOR_PARSIMONY:
    case MESH_TREE_SEARCH:    
    case MORPH_CALIBRATOR:
      break;
    case TREE_EVALUATION:


      Tree2String(tr->tree_string, tr, tr->start->back, TRUE, TRUE, FALSE, FALSE, finalPrint, adef, SUMMARIZE_LH, FALSE, FALSE);

      logFile = myfopen(temporaryFileName, "wb");
      fprintf(logFile, "%s", tr->tree_string);
      fclose(logFile);

      if(adef->perGeneBranchLengths)
        printTreePerGene(tr, adef, temporaryFileName, "wb");


      break;
    case BIG_RAPID_MODE:
      if(!adef->boot)
      {
        if(adef->multipleRuns > 1)
        {
          sprintf(treeID, "%d", tr->treeID);
          strcat(temporaryFileName, ".RUN.");
          strcat(temporaryFileName, treeID);
        }


        if(finalPrint)
        {
          switch(tr->rateHetModel)
          {
            case GAMMA:
            case GAMMA_I:
              Tree2String(tr->tree_string, tr, tr->start->back, TRUE, TRUE, FALSE, FALSE, finalPrint, adef,
                  SUMMARIZE_LH, FALSE, FALSE);

              logFile = myfopen(temporaryFileName, "wb");
              fprintf(logFile, "%s", tr->tree_string);
              fclose(logFile);

              if(adef->perGeneBranchLengths)
                printTreePerGene(tr, adef, temporaryFileName, "wb");
              break;
            case CAT:
              /*Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, finalPrint, adef,
                NO_BRANCHES, FALSE, FALSE);*/



              Tree2String(tr->tree_string, tr, tr->start->back, TRUE, TRUE, FALSE, FALSE,
                  TRUE, adef, SUMMARIZE_LH, FALSE, FALSE);




              logFile = myfopen(temporaryFileName, "wb");
              fprintf(logFile, "%s", tr->tree_string);
              fclose(logFile);

              break;
            default:
              assert(0);
          }
        }
        else
        {
          Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, finalPrint, adef,
              NO_BRANCHES, FALSE, FALSE);
          logFile = myfopen(temporaryFileName, "wb");
          fprintf(logFile, "%s", tr->tree_string);
          fclose(logFile);
        }
      }
      break;
    default:
      printf("FATAL ERROR call to printResult from undefined STATE %d\n", adef->mode);
      exit(-1);
      break;
  }
}

void printBootstrapResult(tree *tr, analdef *adef, boolean finalPrint)
{
  FILE 
    *logFile;
#ifdef _WAYNE_MPI
  char 
    *fileName = bootstrapFileNamePID;
#else
  char 
    *fileName = bootstrapFileName;
#endif

  if(adef->mode == BIG_RAPID_MODE && (adef->boot || adef->rapidBoot))
  {
    if(adef->bootstrapBranchLengths)
    {
      Tree2String(tr->tree_string, tr, tr->start->back, TRUE, TRUE, FALSE, FALSE, finalPrint, adef, SUMMARIZE_LH, FALSE, FALSE);

      logFile = myfopen(fileName, "ab");
      fprintf(logFile, "%s", tr->tree_string);
      fclose(logFile);

      if(adef->perGeneBranchLengths)
        printTreePerGene(tr, adef, fileName, "ab");
    }
    else
    {
      Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, finalPrint, adef, NO_BRANCHES, FALSE, FALSE);

      logFile = myfopen(fileName, "ab");
      fprintf(logFile, "%s", tr->tree_string);
      fclose(logFile);
    }
  }
  else
  {
    printf("FATAL ERROR in  printBootstrapResult\n");
    exit(-1);
  }
}



void printBipartitionResult(tree *tr, analdef *adef, boolean finalPrint)
{
  if(processID == 0 || adef->allInOne)
  {
    FILE *logFile;

    Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, TRUE, finalPrint, adef, NO_BRANCHES, FALSE, FALSE);
    logFile = myfopen(bipartitionsFileName, "ab");
    fprintf(logFile, "%s", tr->tree_string);
    fclose(logFile);

    Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, finalPrint, adef, NO_BRANCHES, TRUE, FALSE);

    logFile = myfopen(bipartitionsFileNameBranchLabels, "ab");
    fprintf(logFile, "%s", tr->tree_string);
    fclose(logFile);

  }
}



void printLog(tree *tr, analdef *adef, boolean finalPrint)
{
  FILE *logFile;
  char temporaryFileName[1024] = "", checkPoints[1024] = "", treeID[64] = "";
  double lh, t;

  lh = tr->likelihood;
  t = gettime() - masterTime;

  strcpy(temporaryFileName, logFileName);
  strcpy(checkPoints,       checkpointFileName);

  switch(adef->mode)
  {
    case TREE_EVALUATION:
      logFile = myfopen(temporaryFileName, "ab");

      printf("%f %f\n", t, lh);
      fprintf(logFile, "%f %f\n", t, lh);

      fclose(logFile);
      break;
    case BIG_RAPID_MODE:
      if(adef->boot || adef->rapidBoot)
      {
        /* testing only printf("%f %f\n", t, lh);*/
        /* NOTHING PRINTED so far */
      }
      else
      {
        if(adef->multipleRuns > 1)
        {
          sprintf(treeID, "%d", tr->treeID);
          strcat(temporaryFileName, ".RUN.");
          strcat(temporaryFileName, treeID);

          strcat(checkPoints, ".RUN.");
          strcat(checkPoints, treeID);
        }


        if(!adef->checkpoints)
        {
          logFile = myfopen(temporaryFileName, "ab");

          fprintf(logFile, "%f %f\n", t, lh);

          fclose(logFile);
        }
        else
        {
          logFile = myfopen(temporaryFileName, "ab");

          fprintf(logFile, "%f %f %d\n", t, lh, tr->checkPointCounter);

          fclose(logFile);

          strcat(checkPoints, ".");

          sprintf(treeID, "%d", tr->checkPointCounter);
          strcat(checkPoints, treeID);

          Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, finalPrint, adef, NO_BRANCHES, FALSE, FALSE);

          logFile = myfopen(checkPoints, "ab");
          fprintf(logFile, "%s", tr->tree_string);
          fclose(logFile);

          tr->checkPointCounter++;
        }
      }
      break;
    case MORPH_CALIBRATOR_PARSIMONY:
    case MORPH_CALIBRATOR:
      break;
    default:
      assert(0);
  }
}



void printStartingTree(tree *tr, analdef *adef, boolean finalPrint)
{
  if(adef->boot)
  {
    /* not printing starting trees for bootstrap */
  }
  else
  {
    FILE *treeFile;
    char temporaryFileName[1024] = "", treeID[64] = "";

    Tree2String(tr->tree_string, tr, tr->start->back, FALSE, TRUE, FALSE, FALSE, finalPrint, adef, NO_BRANCHES, FALSE, FALSE);

    if(adef->randomStartingTree)
      strcpy(temporaryFileName, randomFileName);
    else
      strcpy(temporaryFileName, permFileName);

    if(adef->multipleRuns > 1)
    {
      sprintf(treeID, "%d", tr->treeID);
      strcat(temporaryFileName, ".RUN.");
      strcat(temporaryFileName, treeID);
    }

    treeFile = myfopen(temporaryFileName, "ab");
    fprintf(treeFile, "%s", tr->tree_string);
    fclose(treeFile);
  }
}

void writeInfoFile(analdef *adef, tree *tr, double t)
{

  {      
    switch(adef->mode)
    {
      case MESH_TREE_SEARCH:
        break;
      case TREE_EVALUATION:
        break;
      case BIG_RAPID_MODE:
        if(adef->boot || adef->rapidBoot)
        {
          if(!adef->initialSet)	
            printBothOpen("Bootstrap[%d]: Time %f seconds, bootstrap likelihood %f, best rearrangement setting %d\n", tr->treeID, t, tr->likelihood,  adef->bestTrav);		
          else	
            printBothOpen("Bootstrap[%d]: Time %f seconds, bootstrap likelihood %f\n", tr->treeID, t, tr->likelihood);		
        }
        else
        {
          int model;
          char modelType[128];

          switch(tr->rateHetModel)
          {
            case GAMMA_I:
              strcpy(modelType, "GAMMA+P-Invar");
              break;
            case GAMMA:
              strcpy(modelType, "GAMMA");
              break;
            case CAT:
              strcpy(modelType, "CAT");
              break;
            default:
              assert(0);
          }

          if(!adef->initialSet)		
            printBothOpen("Inference[%d]: Time %f %s-based likelihood %f, best rearrangement setting %d\n",
                tr->treeID, t, modelType, tr->likelihood,  adef->bestTrav);		 
          else		
            printBothOpen("Inference[%d]: Time %f %s-based likelihood %f\n",
                tr->treeID, t, modelType, tr->likelihood);		 

          {
            FILE *infoFile = myfopen(infoFileName, "ab");

            for(model = 0; model < tr->NumberOfModels; model++)
            {
              fprintf(infoFile, "alpha[%d]: %f ", model, tr->partitionData[model].alpha);


              if(tr->partitionData[model].dataType == DNA_DATA)
              {
                int 
                  k,
                  states = tr->partitionData[model].states,
                  rates = ((states * states - states) / 2);

                fprintf(infoFile, "rates[%d] ac ag at cg ct gt: ", model);
                for(k = 0; k < rates; k++)
                  fprintf(infoFile, "%f ", tr->partitionData[model].substRates[k]);
              }		 

            }

            fprintf(infoFile, "\n");
            fclose(infoFile);
          }
        }
        break;
      default:
        assert(0);
    }      
  }
}

static void printFreqs(int n, double *f, char **names)
{
  int k;

  for(k = 0; k < n; k++)
    printBothOpen("freq pi(%s): %f\n", names[k], f[k]);
}

static void printRatesDNA_BIN(int n, double *r, char **names)
{
  int i, j, c;

  for(i = 0, c = 0; i < n; i++)
  {
    for(j = i + 1; j < n; j++)
    {
      if(i == n - 2 && j == n - 1)
        printBothOpen("rate %s <-> %s: %f\n", names[i], names[j], 1.0);
      else
        printBothOpen("rate %s <-> %s: %f\n", names[i], names[j], r[c]);
      c++;
    }
  }
}

static void printRatesRest(int n, double *r, char **names)
{
  int i, j, c;

  for(i = 0, c = 0; i < n; i++)
  {
    for(j = i + 1; j < n; j++)
    {
      printBothOpen("rate %s <-> %s: %f\n", names[i], names[j], r[c]);
      c++;
    }
  }
}


void getDataTypeString(tree *tr, int model, char typeOfData[1024])
{
  switch(tr->partitionData[model].dataType)
  {
    case AA_DATA:
      strcpy(typeOfData,"AA");
      break;
    case DNA_DATA:
      strcpy(typeOfData,"DNA");
      break;
    case BINARY_DATA:
      strcpy(typeOfData,"BINARY/MORPHOLOGICAL");
      break;
    case SECONDARY_DATA:
      strcpy(typeOfData,"SECONDARY 16 STATE MODEL USING ");
      strcat(typeOfData, secondaryModelList[tr->secondaryStructureModel]);
      break;
    case SECONDARY_DATA_6:
      strcpy(typeOfData,"SECONDARY 6 STATE MODEL USING ");
      strcat(typeOfData, secondaryModelList[tr->secondaryStructureModel]);
      break;
    case SECONDARY_DATA_7:
      strcpy(typeOfData,"SECONDARY 7 STATE MODEL USING ");
      strcat(typeOfData, secondaryModelList[tr->secondaryStructureModel]);
      break;
    case GENERIC_32:
      strcpy(typeOfData,"Multi-State");
      break;
    case GENERIC_64:
      strcpy(typeOfData,"Codon"); 
      break;
    default:
      assert(0);
  }
}



void printModelParams(tree *tr, analdef *adef)
{
  int
    model;

  double
    *f = (double*)NULL,
    *r = (double*)NULL;

  for(model = 0; model < tr->NumberOfModels; model++)
  {

    char typeOfData[1024];

    getDataTypeString(tr, model, typeOfData);      

    printBothOpen("Model Parameters of Partition %d, Name: %s, Type of Data: %s\n",
        model, tr->partitionData[model].partitionName, typeOfData);
    printBothOpen("alpha: %f\n", tr->partitionData[model].alpha);





    f = tr->partitionData[model].frequencies;
    r = tr->partitionData[model].substRates;

    switch(tr->partitionData[model].dataType)
    {
      case AA_DATA:
        {
          char *freqNames[20] = {"A", "R", "N ","D", "C", "Q", "E", "G",
            "H", "I", "L", "K", "M", "F", "P", "S",
            "T", "W", "Y", "V"};

          printRatesRest(20, r, freqNames);
          printBothOpen("\n");
          printFreqs(20, f, freqNames);
        }
        break;
      case GENERIC_32:
        {
          char *freqNames[32] = {"0", "1", "2", "3", "4", "5", "6", "7", 
            "8", "9", "A", "B", "C", "D", "E", "F",
            "G", "H", "I", "J", "K", "L", "M", "N",
            "O", "P", "Q", "R", "S", "T", "U", "V"}; 

          printRatesRest(32, r, freqNames);
          printBothOpen("\n");
          printFreqs(32, f, freqNames);
        }
        break;
      case GENERIC_64:
        assert(0);
        break;
      case DNA_DATA:
        {
          char *freqNames[4] = {"A", "C", "G", "T"};

          printRatesDNA_BIN(4, r, freqNames);
          printBothOpen("\n");
          printFreqs(4, f, freqNames);
        }
        break;
      case SECONDARY_DATA_6:
        {
          char *freqNames[6] = {"AU", "CG", "GC", "GU", "UA", "UG"};

          printRatesRest(6, r, freqNames);
          printBothOpen("\n");
          printFreqs(6, f, freqNames);
        }
        break;
      case SECONDARY_DATA_7:
        {
          char *freqNames[7] = {"AU", "CG", "GC", "GU", "UA", "UG", "REST"};

          printRatesRest(7, r, freqNames);
          printBothOpen("\n");
          printFreqs(7, f, freqNames);
        }
        break;
      case SECONDARY_DATA:
        {
          char *freqNames[16] = {"AA", "AC", "AG", "AU", "CA", "CC", "CG", "CU",
            "GA", "GC", "GG", "GU", "UA", "UC", "UG", "UU"};

          printRatesRest(16, r, freqNames);
          printBothOpen("\n");
          printFreqs(16, f, freqNames);
        }
        break;
      case BINARY_DATA:
        {
          char *freqNames[2] = {"0", "1"};

          printRatesDNA_BIN(2, r, freqNames);
          printBothOpen("\n");
          printFreqs(2, f, freqNames);
        }
        break;
      default:
        assert(0);
    }

    printBothOpen("\n");
  }
}

static void finalizeInfoFile(tree *tr, analdef *adef)
{
  if(processID == 0)
  {
    double t;

    t = gettime() - masterTime;
    accumulatedTime = accumulatedTime + t;

    switch(adef->mode)
    {
      case MESH_TREE_SEARCH:
        break;
      case TREE_EVALUATION :
        printBothOpen("\n\nOverall Time for Tree Evaluation %f\n", t);
        printBothOpen("Final GAMMA  likelihood: %f\n", tr->likelihood);

        {
          int
            params,
            paramsBrLen;

          if(tr->NumberOfModels == 1)
          {
            if(adef->useInvariant)
            {
              params      = 1 /* INVAR */ + 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */;
              paramsBrLen = 1 /* INVAR */ + 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */ +
                (2 * tr->mxtips - 3);
            }
            else
            {
              params      = 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */;
              paramsBrLen = 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */ +
                (2 * tr->mxtips - 3);
            }
          }
          else
          {
            if(tr->multiBranch)
            {
              if(adef->useInvariant)
              {
                params      = tr->NumberOfModels * (1 /* INVAR */ + 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */);
                paramsBrLen = tr->NumberOfModels * (1 /* INVAR */ + 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */ +
                    (2 * tr->mxtips - 3));
              }
              else
              {
                params      = tr->NumberOfModels * (5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */);
                paramsBrLen = tr->NumberOfModels * (5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */ +
                    (2 * tr->mxtips - 3));
              }
            }
            else
            {
              if(adef->useInvariant)
              {
                params      = tr->NumberOfModels * (1 /* INVAR */ + 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */);
                paramsBrLen = tr->NumberOfModels * (1 /* INVAR */ + 5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */)
                  + (2 * tr->mxtips - 3);
              }
              else
              {
                params      = tr->NumberOfModels * (5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */);
                paramsBrLen = tr->NumberOfModels * (5 /* RATES */ + 3 /* freqs */ + 1 /* alpha */)
                  + (2 * tr->mxtips - 3);
              }

            }
          }

          if(tr->partitionData[0].dataType == DNA_DATA)
          {
            printBothOpen("Number of free parameters for AIC-TEST(BR-LEN): %d\n",    paramsBrLen);
            printBothOpen("Number of free parameters for AIC-TEST(NO-BR-LEN): %d\n", params);
          }

        }

        printBothOpen("\n\n");

        printModelParams(tr, adef);

        printBothOpen("Final tree written to:                 %s\n", resultFileName);
        printBothOpen("Execution Log File written to:         %s\n", logFileName);


        break;
      case  BIG_RAPID_MODE:
        if(adef->boot)
        {
          printBothOpen("\n\nOverall Time for %d Bootstraps %f\n", adef->multipleRuns, t);
          printBothOpen("\n\nAverage Time per Bootstrap %f\n", (double)(t/((double)adef->multipleRuns)));
          printBothOpen("All %d bootstrapped trees written to: %s\n", adef->multipleRuns, bootstrapFileName);
        }
        else
        {
          if(adef->multipleRuns > 1)
          {
            double avgLH = 0;
            double bestLH = unlikely;
            int i, bestI  = 0;

            for(i = 0; i < adef->multipleRuns; i++)
            {
              avgLH   += tr->likelihoods[i];
              if(tr->likelihoods[i] > bestLH)
              {
                bestLH = tr->likelihoods[i];
                bestI  = i;
              }
            }
            avgLH /= ((double)adef->multipleRuns);

            printBothOpen("\n\nOverall Time for %d Inferences %f\n", adef->multipleRuns, t);
            printBothOpen("Average Time per Inference %f\n", (double)(t/((double)adef->multipleRuns)));
            printBothOpen("Average Likelihood   : %f\n", avgLH);
            printBothOpen("\n");
            printBothOpen("Best Likelihood in run number %d: likelihood %f\n\n", bestI, bestLH);

            if(adef->checkpoints)
              printBothOpen("Checkpoints written to:                 %s.RUN.%d.* to %d.*\n", checkpointFileName, 0, adef->multipleRuns - 1);
            if(!adef->restart)
            {
              if(adef->randomStartingTree)
                printBothOpen("Random starting trees written to:       %s.RUN.%d to %d\n", randomFileName, 0, adef->multipleRuns - 1);
              else
                printBothOpen("Parsimony starting trees written to:    %s.RUN.%d to %d\n", permFileName, 0, adef->multipleRuns - 1);
            }
            printBothOpen("Final trees written to:                 %s.RUN.%d to %d\n", resultFileName,  0, adef->multipleRuns - 1);
            printBothOpen("Execution Log Files written to:         %s.RUN.%d to %d\n", logFileName, 0, adef->multipleRuns - 1);
            printBothOpen("Execution information file written to:  %s\n", infoFileName);
          }
          else
          {
            printBothOpen("\n\nOverall Time for 1 Inference %f\n", t);
            printBothOpen("\nOverall accumulated Time (in case of restarts): %f\n\n", accumulatedTime);
            printBothOpen("Likelihood   : %f\n", tr->likelihood);
            printBothOpen("\n\n");

            if(adef->checkpoints)
              printBothOpen("Checkpoints written to:                %s.*\n", checkpointFileName);
            if(!adef->restart)
            {
              if(adef->randomStartingTree)
                printBothOpen("Random starting tree written to:       %s\n", randomFileName);
              else
                printBothOpen("Parsimony starting tree written to:    %s\n", permFileName);
            }
            printBothOpen("Final tree written to:                 %s\n", resultFileName);
            printBothOpen("Execution Log File written to:         %s\n", logFileName);
            printBothOpen("Execution information file written to: %s\n",infoFileName);
          }
        }

        break;
      case CALC_BIPARTITIONS:
        printBothOpen("\n\nTime for Computation of Bipartitions %f\n", t);
        printBothOpen("Tree with bipartitions written to file:  %s\n", bipartitionsFileName);
        printBothOpen("Tree with bipartitions as branch labels written to file:  %s\n", bipartitionsFileNameBranchLabels);	  
        printBothOpen("Execution information file written to :  %s\n",infoFileName);
        break;
      case PER_SITE_LL:
        printBothOpen("\n\nTime for Optimization of per-site log likelihoods %f\n", t);
        printBothOpen("Per-site Log Likelihoods written to File %s in Tree-Puzzle format\n",  perSiteLLsFileName);
        printBothOpen("Execution information file written to :  %s\n",infoFileName);

        break;
      case PARSIMONY_ADDITION:
        printBothOpen("\n\nTime for MP stepwise addition %f\n", t);
        printBothOpen("Execution information file written to :  %s\n",infoFileName);
        printBothOpen("Complete parsimony tree written to:      %s\n", permFileName);
        break;
      default:
        assert(0);
    }
  }

}


/************************************************************************************/


#if (defined(_USE_PTHREADS) || (_FINE_GRAIN_MPI))




boolean isThisMyPartition(tree *localTree, int tid, int model, int numberOfThreads)
{ 
  if(localTree->partitionAssignment[model] == tid)
    return TRUE;
  else
    return FALSE;
}

static void computeFractionMany(tree *localTree, int tid, int n)
{
  int
    sites = 0;

  int
    i,
    model;

  for(model = 0; model < localTree->NumberOfModels; model++)
  {
    if(isThisMyPartition(localTree, tid, model, n))
    {	 
      localTree->partitionData[model].width = localTree->partitionData[model].upper - localTree->partitionData[model].lower;
      sites += localTree->partitionData[model].width;
    }
    else       	  
      localTree->partitionData[model].width = 0;       
  }


}



static void computeFraction(tree *localTree, int tid, int n)
{
  int
    i,
    model;

  for(model = 0; model < localTree->NumberOfModels; model++)
  {
    int width = 0;

    for(i = localTree->partitionData[model].lower; i < localTree->partitionData[model].upper; i++)
      if(i % n == tid)
        width++;

    localTree->partitionData[model].width = width;
  }
}



static void threadFixModelIndices(tree *tr, tree *localTree, int tid, int n)
{
  size_t
    model,
    j,
    i,
    globalCounter = 0,
    localCounter  = 0,
    offset,
    countOffset,
    myLength = 0,
    memoryRequirements = 0;

  for(model = 0; model < (size_t)localTree->NumberOfModels; model++)
  {
    localTree->partitionData[model].lower      = tr->partitionData[model].lower;
    localTree->partitionData[model].upper      = tr->partitionData[model].upper;
  }

  if(tr->manyPartitions)
    computeFractionMany(localTree, tid, n);
  else
    computeFraction(localTree, tid, n);

  for(model = 0, offset = 0, countOffset = 0; model < (size_t)localTree->NumberOfModels; model++)
  {           
    localTree->partitionData[model].sumBuffer       = &localTree->sumBuffer[offset];

    localTree->partitionData[model].perSiteLL    = &localTree->perSiteLLPtr[countOffset];          

    localTree->partitionData[model].wgt          = &localTree->wgtPtr[countOffset];

    localTree->partitionData[model].rateCategory = &localTree->rateCategoryPtr[countOffset];     

    countOffset += localTree->partitionData[model].width;

    offset += (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * (size_t)(localTree->partitionData[model].width);      
  }

  myLength           = countOffset;
  memoryRequirements = offset;


  /* figure in data */   


  for(i = 0; i < (size_t)localTree->mxtips; i++)
  {
    for(model = 0, offset = 0, countOffset = 0; model < (size_t)localTree->NumberOfModels; model++)
    {
      localTree->partitionData[model].yVector[i+1]   = &localTree->y_ptr[i * myLength + countOffset];
      countOffset +=  localTree->partitionData[model].width;
    }
    assert(countOffset == myLength);
  }

  for(i = 0; i < (size_t)localTree->innerNodes; i++)
  {
    for(model = 0, offset = 0, countOffset = 0; model < (size_t)localTree->NumberOfModels; model++)
    {
      size_t 
        width = localTree->partitionData[model].width;	  	  

      localTree->partitionData[model].xVector[i]   = (double*)NULL;

      countOffset += width;

      offset += (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width;

    }
    assert(countOffset == myLength);
  }

  if(tr->manyPartitions)
    for(model = 0, globalCounter = 0; model < (size_t)localTree->NumberOfModels; model++)
    {
      if(isThisMyPartition(localTree, tid, model, n))
      {
        assert(localTree->partitionData[model].upper - localTree->partitionData[model].lower == localTree->partitionData[model].width);

        for(localCounter = 0, i = (size_t)localTree->partitionData[model].lower;  i < (size_t)localTree->partitionData[model].upper; i++, localCounter++)
        {	    
          localTree->partitionData[model].wgt[localCounter]          = tr->cdta->aliaswgt[globalCounter];	      	     
          localTree->partitionData[model].rateCategory[localCounter] = tr->cdta->rateCategory[globalCounter];	      

          for(j = 1; j <= (size_t)localTree->mxtips; j++)
            localTree->partitionData[model].yVector[j][localCounter] = tr->yVector[j][globalCounter]; 	     

          globalCounter++;
        }
      }
      else
        globalCounter += (localTree->partitionData[model].upper - localTree->partitionData[model].lower);
    }
  else
    for(model = 0, globalCounter = 0; model < (size_t)localTree->NumberOfModels; model++)
    {
      for(localCounter = 0, i = (size_t)localTree->partitionData[model].lower;  i < (size_t)localTree->partitionData[model].upper; i++)
      {
        if(i % (size_t)n == (size_t)tid)
        {
          localTree->partitionData[model].wgt[localCounter]          = tr->cdta->aliaswgt[globalCounter];	      	     
          localTree->partitionData[model].rateCategory[localCounter] = tr->cdta->rateCategory[globalCounter];	      

          for(j = 1; j <= (size_t)localTree->mxtips; j++)
            localTree->partitionData[model].yVector[j][localCounter] = tr->yVector[j][globalCounter]; 	     

          localCounter++;
        }
        globalCounter++;
      }
    }

  for(model = 0; model < (size_t)localTree->NumberOfModels; model++)
  {
    int        
      undetermined = getUndetermined(localTree->partitionData[model].dataType);

    size_t
      width =  localTree->partitionData[model].width;

    if(width > 0)
    {
      localTree->partitionData[model].gapVectorLength = ((int)width / 32) + 1;

      memset(localTree->partitionData[model].gapVector, 0, localTree->partitionData[model].initialGapVectorSize);

      if(localTree->saveMemory)
      {
        for(j = 1; j <= (size_t)(localTree->mxtips); j++)
          for(i = 0; i < width; i++)
            if(localTree->partitionData[model].yVector[j][i] == undetermined)
              localTree->partitionData[model].gapVector[localTree->partitionData[model].gapVectorLength * j + i / 32] |= mask32[i % 32];
      }
    }
    else
    {
      localTree->partitionData[model].gapVectorLength = 0;
    }
  }
}


static void initPartition(tree *tr, tree *localTree, int tid)
{
  int model;

  localTree->threadID = tid; 

  if(tid > 0)
  {
    int totalLength = 0;

    localTree->rateHetModel            = tr->rateHetModel;
    localTree->saveMemory              = tr->saveMemory;
    localTree->useGappedImplementation = tr->useGappedImplementation;
    localTree->innerNodes              = tr->innerNodes;

    localTree->maxCategories           = tr->maxCategories;

    localTree->originalCrunchedLength  = tr->originalCrunchedLength;
    localTree->NumberOfModels          = tr->NumberOfModels;
    localTree->mxtips                  = tr->mxtips;
    localTree->multiBranch             = tr->multiBranch;
    localTree->numBranches             = tr->numBranches;
    localTree->lhs                     = (double*)malloc(sizeof(double)   * localTree->originalCrunchedLength);
    localTree->executeModel            = (boolean*)malloc(sizeof(boolean) * localTree->NumberOfModels);
    localTree->perPartitionLH          = (double*)malloc(sizeof(double)   * localTree->NumberOfModels);
    localTree->storedPerPartitionLH    = (double*)malloc(sizeof(double)   * localTree->NumberOfModels);

    localTree->fracchanges = (double*)malloc(sizeof(double)   * localTree->NumberOfModels);
    localTree->partitionContributions = (double*)malloc(sizeof(double)   * localTree->NumberOfModels);

    localTree->partitionData = (pInfo*)malloc(sizeof(pInfo) * localTree->NumberOfModels);

    /* extend for multi-branch */
    localTree->td[0].count = 0;  
    localTree->td[0].ti    = (traversalInfo *)malloc(sizeof(traversalInfo) * localTree->mxtips);

    localTree->cdta               = (cruncheddata*)malloc(sizeof(cruncheddata));
    localTree->cdta->patrat       = (double*)malloc(sizeof(double) * localTree->originalCrunchedLength);
    localTree->cdta->patratStored = (double*)malloc(sizeof(double) * localTree->originalCrunchedLength);      

    localTree->discreteRateCategories = tr->discreteRateCategories;     

    for(model = 0; model < localTree->NumberOfModels; model++)
    {
      localTree->partitionData[model].numberOfCategories    = tr->partitionData[model].numberOfCategories;
      localTree->partitionData[model].states     = tr->partitionData[model].states;
      localTree->partitionData[model].maxTipStates    = tr->partitionData[model].maxTipStates;
      localTree->partitionData[model].dataType   = tr->partitionData[model].dataType;
      localTree->partitionData[model].protModels = tr->partitionData[model].protModels;
      localTree->partitionData[model].protFreqs  = tr->partitionData[model].protFreqs;
      localTree->partitionData[model].mxtips     = tr->partitionData[model].mxtips;
      localTree->partitionData[model].lower      = tr->partitionData[model].lower;
      localTree->partitionData[model].upper      = tr->partitionData[model].upper;
      localTree->executeModel[model]             = TRUE;
      localTree->perPartitionLH[model]           = 0.0;
      localTree->storedPerPartitionLH[model]     = 0.0;
      totalLength += (localTree->partitionData[model].upper -  localTree->partitionData[model].lower);
    }

    assert(totalLength == localTree->originalCrunchedLength);
    /* recomp */
    localTree->useRecom = tr->useRecom;
    /* E recomp */
  }

  for(model = 0; model < localTree->NumberOfModels; model++)
    localTree->partitionData[model].width        = 0;
}




void allocNodex(tree *tr, int tid, int n)
{
  size_t 
    rateHet,
    model,
    memoryRequirements = 0,
    myLength = 0;

  if(tr->manyPartitions)
    computeFractionMany(tr, tid, n);
  else
    computeFraction(tr, tid, n);

  if(tr->useRecom && tid == 0)
    allocRecompVectorsInfo(tr);
  else
    tr->rvec = (recompVectors*)NULL;

  allocPartitions(tr);

  if(tr->rateHetModel == CAT)
    rateHet = 1;
  else
    rateHet = 4;



  for(model = 0; model < (size_t)tr->NumberOfModels; model++)
  {
    size_t 
      width = tr->partitionData[model].width;

    myLength += width;

    if(width > 0)
    {
      memoryRequirements += (size_t)(tr->discreteRateCategories) * (size_t)(tr->partitionData[model].states) * width;

      tr->partitionData[model].gapVectorLength = ((int)width / 32) + 1;

      tr->partitionData[model].gapVector = (unsigned int*)calloc(tr->partitionData[model].gapVectorLength * 2 * tr->mxtips, sizeof(unsigned int));	  

      tr->partitionData[model].initialGapVectorSize = tr->partitionData[model].gapVectorLength * 2 * tr->mxtips * sizeof(int);

      tr->partitionData[model].gapColumn = (double *)malloc_aligned(((size_t)tr->innerNodes) *								      
          ((size_t)(tr->partitionData[model].states)) *
          rateHet * sizeof(double));		              
    }
    else
    {
      tr->partitionData[model].gapVectorLength = 0;

      tr->partitionData[model].gapVector = (unsigned int*)NULL; 	  

      tr->partitionData[model].initialGapVectorSize = 0;

      tr->partitionData[model].gapColumn = (double*)NULL;
    }
  }

  if(tid == 0)
  {
    tr->perSiteLL       = (double *)malloc((size_t)tr->cdta->endsite * sizeof(double));
    assert(tr->perSiteLL != NULL);
  }



  tr->sumBuffer  = (double *)malloc_aligned(memoryRequirements * sizeof(double));
  assert(tr->sumBuffer != NULL);


  tr->y_ptr = (unsigned char *)malloc(myLength * (size_t)(tr->mxtips) * sizeof(unsigned char));
  assert(tr->y_ptr != NULL);

#ifdef  _FINE_GRAIN_MPI 
  printf("Process %d assigning %Zu bytes for partial alignment\n", processID, myLength * (size_t)(tr->mxtips) * sizeof(unsigned char));
#endif


  assert(4 * sizeof(double) > sizeof(parsimonyVector));

  tr->perSiteLLPtr     = (double*) malloc(myLength * sizeof(double));

  tr->wgtPtr           = (int*)    malloc(myLength * sizeof(int));
  assert(tr->wgtPtr != NULL);  

  tr->rateCategoryPtr  = (int*)    malloc(myLength * sizeof(int));
  assert(tr->rateCategoryPtr != NULL);


}


#ifdef _USE_PTHREADS



inline static void sendTraversalInfo(tree *localTree, tree *tr)
{
  /* the one below is a hack we are re-assigning the local pointer to the global one
     the memcpy version below is just for testing and preparing the
     fine-grained MPI BlueGene version */

  if(1)
  {
    localTree->td[0] = tr->td[0];
  }
  else
  {
    localTree->td[0].count = tr->td[0].count;
    memcpy(localTree->td[0].ti, tr->td[0].ti, localTree->td[0].count * sizeof(traversalInfo));
  }
}


static void collectDouble(double *dst, double *src, tree *tr, int n, int tid)
{
  int 
    model,
    i;
  if(tr->manyPartitions)
    for(model = 0; model < tr->NumberOfModels; model++)
    {
      if(isThisMyPartition(tr, tid, model, n))	
        for(i = tr->partitionData[model].lower; i < tr->partitionData[model].upper; i++)
          dst[i] = src[i];       
    }
  else
    for(model = 0; model < tr->NumberOfModels; model++)
    {
      for(i = tr->partitionData[model].lower; i < tr->partitionData[model].upper; i++)
      {
        if(i % n == tid)
          dst[i] = src[i];
      }
    }
}


static void broadcastPerSiteRates(tree *tr, tree *localTree)
{
  int
    i = 0,
      model = 0;  

  for(model = 0; model < localTree->NumberOfModels; model++)
  {
    localTree->partitionData[model].numberOfCategories = tr->partitionData[model].numberOfCategories;

    for(i = 0; i < localTree->partitionData[model].numberOfCategories; i++)
      localTree->partitionData[model].perSiteRates[i] = tr->partitionData[model].perSiteRates[i];
  }

}

static void execFunction(tree *tr, tree *localTree, int tid, int n)
{
  double volatile result;
  int
    i,
    currentJob,
    parsimonyResult,
    model,
    localCounter,
    globalCounter;

  currentJob = threadJob >> 16;



  switch(currentJob)
  {            
    case THREAD_INIT_PARTITION:

      localTree->estimatePerSiteAA = tr->estimatePerSiteAA;


      localTree->manyPartitions = tr->manyPartitions;
      if(localTree->manyPartitions && tid > 0)     
      {
        localTree->NumberOfModels = tr->NumberOfModels;
        localTree->partitionAssignment = (int*)malloc(sizeof(int) * localTree->NumberOfModels);
        memcpy(localTree->partitionAssignment, tr->partitionAssignment, localTree->NumberOfModels * sizeof(int));
      }

      initPartition(tr, localTree, tid);     
      allocNodex(localTree, tid, n);
      threadFixModelIndices(tr, localTree, tid, n);



      break;      
    case THREAD_EVALUATE:
      sendTraversalInfo(localTree, tr);
      result = evaluateIterative(localTree, FALSE);

      if(localTree->NumberOfModels > 1)
      {
        for(model = 0; model < localTree->NumberOfModels; model++)
          reductionBuffer[tid * localTree->NumberOfModels + model] = localTree->perPartitionLH[model];
      }
      else
        reductionBuffer[tid] = result;

      if(tid > 0)
      {
        for(model = 0; model < localTree->NumberOfModels; model++)
          localTree->executeModel[model] = TRUE;
      }
      break;
    case THREAD_NEWVIEW_MASKED:
      sendTraversalInfo(localTree, tr);
      memcpy(localTree->executeModel, tr->executeModel, sizeof(boolean) * localTree->NumberOfModels);
      newviewIterative(localTree);
      if(tid > 0)
      {
        for(model = 0; model < localTree->NumberOfModels; model++)
          localTree->executeModel[model] = TRUE;
      }
      break;
    case THREAD_NEWVIEW:
      sendTraversalInfo(localTree, tr);
      newviewIterative(localTree);
      break;
    case THREAD_MAKENEWZ_FIRST:
      {
        volatile double
          dlnLdlz[NUM_BRANCHES],
          d2lnLdlz2[NUM_BRANCHES];

        sendTraversalInfo(localTree, tr);
        if(tid > 0)
        {
          memcpy(localTree->coreLZ,   tr->coreLZ,   sizeof(double) *  localTree->numBranches);
          memcpy(localTree->executeModel, tr->executeModel, sizeof(boolean) * localTree->NumberOfModels);
        }

        makenewzIterative(localTree);	
        execCore(localTree, dlnLdlz, d2lnLdlz2);

        if(!tr->multiBranch)
        {
          reductionBuffer[tid]    = dlnLdlz[0];
          reductionBufferTwo[tid] = d2lnLdlz2[0];
        }
        else
        {
          for(i = 0; i < localTree->NumberOfModels; i++)
          {
            reductionBuffer[tid * localTree->NumberOfModels + i]    = dlnLdlz[i];
            reductionBufferTwo[tid * localTree->NumberOfModels + i] = d2lnLdlz2[i];
          }
        }

        if(tid > 0)
        {
          for(model = 0; model < localTree->NumberOfModels; model++)
            localTree->executeModel[model] = TRUE;
        }
      }
      break;
    case THREAD_MAKENEWZ:
      {
        volatile double
          dlnLdlz[NUM_BRANCHES],
          d2lnLdlz2[NUM_BRANCHES];

        memcpy(localTree->coreLZ,   tr->coreLZ,   sizeof(double) *  localTree->numBranches);
        memcpy(localTree->executeModel, tr->executeModel, sizeof(boolean) * localTree->NumberOfModels);

        execCore(localTree, dlnLdlz, d2lnLdlz2);

        if(!tr->multiBranch)
        {
          reductionBuffer[tid]    = dlnLdlz[0];
          reductionBufferTwo[tid] = d2lnLdlz2[0];
        }
        else
        {
          for(i = 0; i < localTree->NumberOfModels; i++)
          {
            reductionBuffer[tid * localTree->NumberOfModels + i]    = dlnLdlz[i];
            reductionBufferTwo[tid * localTree->NumberOfModels + i] = d2lnLdlz2[i];
          }
        }
        if(tid > 0)
        {
          for(model = 0; model < localTree->NumberOfModels; model++)
            localTree->executeModel[model] = TRUE;
        }
      }
      break;
    case THREAD_COPY_RATES:
      if(tid > 0)
      {
        for(model = 0; model < localTree->NumberOfModels; model++)
        {	      
          const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));

          memcpy(localTree->partitionData[model].EIGN,        tr->partitionData[model].EIGN,        pl->eignLength * sizeof(double));
          memcpy(localTree->partitionData[model].EV,          tr->partitionData[model].EV,          pl->evLength * sizeof(double));		  
          memcpy(localTree->partitionData[model].EI,          tr->partitionData[model].EI,          pl->eiLength * sizeof(double));
          memcpy(localTree->partitionData[model].tipVector,   tr->partitionData[model].tipVector,   pl->tipVectorLength * sizeof(double));	      	     	
        }
      }
      break;
    case THREAD_COPY_ALPHA:
      if(tid > 0)
      {
        for(model = 0; model < localTree->NumberOfModels; model++)
        {
          memcpy(localTree->partitionData[model].gammaRates, tr->partitionData[model].gammaRates, sizeof(double) * 4);
          localTree->partitionData[model].alpha = tr->partitionData[model].alpha;
        }
      }
      break;
    case THREAD_OPT_RATE:
      if(tid > 0)
      {
        memcpy(localTree->executeModel, tr->executeModel, localTree->NumberOfModels * sizeof(boolean));

        for(model = 0; model < localTree->NumberOfModels; model++)
        {
          const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));

          memcpy(localTree->partitionData[model].EIGN,        tr->partitionData[model].EIGN,        pl->eignLength * sizeof(double));
          memcpy(localTree->partitionData[model].EV,          tr->partitionData[model].EV,          pl->evLength * sizeof(double));		  
          memcpy(localTree->partitionData[model].EI,          tr->partitionData[model].EI,          pl->eiLength * sizeof(double));
          memcpy(localTree->partitionData[model].tipVector,   tr->partitionData[model].tipVector,   pl->tipVectorLength * sizeof(double));


        }
      }

      result = evaluateIterative(localTree, FALSE);


      if(localTree->NumberOfModels > 1)
      {
        for(model = 0; model < localTree->NumberOfModels; model++)
          reductionBuffer[tid * localTree->NumberOfModels + model] = localTree->perPartitionLH[model];
      }
      else
        reductionBuffer[tid] = result;


      if(tid > 0)
      {
        for(model = 0; model < localTree->NumberOfModels; model++)
          localTree->executeModel[model] = TRUE;
      }
      break;               
    case THREAD_BROADCAST_RATE:
      if(tid > 0)
      {
        memcpy(localTree->executeModel, tr->executeModel, localTree->NumberOfModels * sizeof(boolean));

        for(model = 0; model < localTree->NumberOfModels; model++)
        {
          const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));

          memcpy(localTree->partitionData[model].EIGN,        tr->partitionData[model].EIGN,        pl->eignLength * sizeof(double));
          memcpy(localTree->partitionData[model].EV,          tr->partitionData[model].EV,          pl->evLength * sizeof(double));		  
          memcpy(localTree->partitionData[model].EI,          tr->partitionData[model].EI,          pl->eiLength * sizeof(double));
          memcpy(localTree->partitionData[model].tipVector,   tr->partitionData[model].tipVector,   pl->tipVectorLength * sizeof(double));


        }
      }     
      break;               
    case THREAD_COPY_INIT_MODEL:
      if(tid > 0)
      {	  
        localTree->rateHetModel       = tr->rateHetModel;

        for(model = 0; model < localTree->NumberOfModels; model++)
        {
          const partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));

          memcpy(localTree->partitionData[model].EIGN,        tr->partitionData[model].EIGN,        pl->eignLength * sizeof(double));
          memcpy(localTree->partitionData[model].EV,          tr->partitionData[model].EV,          pl->evLength * sizeof(double));
          memcpy(localTree->partitionData[model].EI,          tr->partitionData[model].EI,          pl->eiLength * sizeof(double));
          memcpy(localTree->partitionData[model].substRates,  tr->partitionData[model].substRates,  pl->substRatesLength * sizeof(double));
          memcpy(localTree->partitionData[model].frequencies, tr->partitionData[model].frequencies, pl->frequenciesLength * sizeof(double));
          memcpy(localTree->partitionData[model].tipVector,   tr->partitionData[model].tipVector,   pl->tipVectorLength * sizeof(double));



          memcpy(localTree->partitionData[model].gammaRates, tr->partitionData[model].gammaRates, sizeof(double) * 4);
          localTree->partitionData[model].alpha = tr->partitionData[model].alpha;

          localTree->partitionData[model].lower      = tr->partitionData[model].lower;
          localTree->partitionData[model].upper      = tr->partitionData[model].upper; 

          localTree->partitionData[model].numberOfCategories      = tr->partitionData[model].numberOfCategories;
        }

        memcpy(localTree->cdta->patrat,        tr->cdta->patrat,      localTree->originalCrunchedLength * sizeof(double));
        memcpy(localTree->cdta->patratStored, tr->cdta->patratStored, localTree->originalCrunchedLength * sizeof(double));	  
      }     


      if(localTree->manyPartitions)
        for(model = 0; model < localTree->NumberOfModels; model++)
        {	  
          if(isThisMyPartition(localTree, tid, model, n))
          {
            int localIndex;

            for(i = localTree->partitionData[model].lower, localIndex = 0; i <  localTree->partitionData[model].upper; i++, localIndex++)	     	       
              localTree->partitionData[model].wgt[localIndex]          = tr->cdta->aliaswgt[i];				 					       
          }	  
        }
      else
        for(model = 0; model < localTree->NumberOfModels; model++)
        {
          int localIndex;
          for(i = localTree->partitionData[model].lower, localIndex = 0; i <  localTree->partitionData[model].upper; i++)
            if(i % n == tid)
            {
              localTree->partitionData[model].wgt[localIndex]          = tr->cdta->aliaswgt[i];				 		

              localIndex++;
            }	  
        }
      if(localTree->estimatePerSiteAA && tid > 0)   
      {
        int p;

        for(p = 0; p < NUM_PROT_MODELS - 2; p++)
        {
          memcpy(localTree->siteProtModel[p].EIGN,        tr->siteProtModel[p].EIGN,        sizeof(double) * 19);
          memcpy(localTree->siteProtModel[p].EV,          tr->siteProtModel[p].EV,          sizeof(double) * 400);                
          memcpy(localTree->siteProtModel[p].EI,          tr->siteProtModel[p].EI,          sizeof(double) * 380);
          memcpy(localTree->siteProtModel[p].substRates,  tr->siteProtModel[p].substRates,  sizeof(double) * 190);        
          memcpy(localTree->siteProtModel[p].frequencies, tr->siteProtModel[p].frequencies, sizeof(double) * 20);
          memcpy(localTree->siteProtModel[p].tipVector,   tr->siteProtModel[p].tipVector,   sizeof(double) * 460);
        }

        for(model = 0; model < localTree->NumberOfModels; model++)
        {
          int width = localTree->partitionData[model].width;

          for(i = 0; i < width; i++)
            localTree->partitionData[model].perSiteAAModel[i] = WAG;
        }	    
      }
      break;    
    case THREAD_RATE_CATS:
      sendTraversalInfo(localTree, tr);
      if(tid > 0)
      {
        localTree->lower_spacing = tr->lower_spacing;
        localTree->upper_spacing = tr->upper_spacing;
      }

      optRateCatPthreads(localTree, localTree->lower_spacing, localTree->upper_spacing, localTree->lhs, n, tid);

      if(tid > 0)
      {
        collectDouble(tr->cdta->patrat,       localTree->cdta->patrat,         localTree, n, tid);
        collectDouble(tr->cdta->patratStored, localTree->cdta->patratStored,   localTree, n, tid);
        collectDouble(tr->lhs,                localTree->lhs,                  localTree, n, tid);
      }
      break;
    case THREAD_COPY_RATE_CATS:
      if(tid > 0)
      {	  
        memcpy(localTree->cdta->patrat,       tr->cdta->patrat,         localTree->originalCrunchedLength * sizeof(double));
        memcpy(localTree->cdta->patratStored, tr->cdta->patratStored,   localTree->originalCrunchedLength * sizeof(double));
        broadcastPerSiteRates(tr, localTree);
      }

      for(model = 0; model < localTree->NumberOfModels; model++)
      {
        localTree->partitionData[model].numberOfCategories = tr->partitionData[model].numberOfCategories;

        if(localTree->manyPartitions)
        {
          if(isThisMyPartition(localTree, tid, model, n))
            for(localCounter = 0, i = localTree->partitionData[model].lower;  i < localTree->partitionData[model].upper; i++, localCounter++)
            {	     
              localTree->partitionData[model].rateCategory[localCounter] = tr->cdta->rateCategory[i];
              localTree->partitionData[model].wr[localCounter]             = tr->wr[i];
              localTree->partitionData[model].wr2[localCounter]            = tr->wr2[i];		 		 	     
            } 
        }
        else	  
        {
          for(localCounter = 0, i = localTree->partitionData[model].lower;  i < localTree->partitionData[model].upper; i++)
          {
            if(i % n == tid)
            {		 
              localTree->partitionData[model].rateCategory[localCounter] = tr->cdta->rateCategory[i];
              localTree->partitionData[model].wr[localCounter]             = tr->wr[i];
              localTree->partitionData[model].wr2[localCounter]            = tr->wr2[i];		 

              localCounter++;
            }
          }
        }
      }
      break;
    case THREAD_OPT_ALPHA:
      if(tid > 0)
      {
        memcpy(localTree->executeModel, tr->executeModel, localTree->NumberOfModels * sizeof(boolean));
        for(model = 0; model < localTree->NumberOfModels; model++)
          memcpy(localTree->partitionData[model].gammaRates, tr->partitionData[model].gammaRates, sizeof(double) * 4);
      }

      result = evaluateIterative(localTree, FALSE);


      if(localTree->NumberOfModels > 1)
      {
        for(model = 0; model < localTree->NumberOfModels; model++)
          reductionBuffer[tid *  localTree->NumberOfModels + model] = localTree->perPartitionLH[model];
      }
      else
        reductionBuffer[tid] = result;

      if(tid > 0)
      {
        for(model = 0; model < localTree->NumberOfModels; model++)
          localTree->executeModel[model] = TRUE;
      }
      break;
    case THREAD_OPTIMIZE_PER_SITE_AA:
      sendTraversalInfo(localTree, tr);      
      {
        int
          s,
          p;

        double  
          *bestScore = (double *)malloc(localTree->originalCrunchedLength * sizeof(double));	  

        for(s = 0; s < localTree->originalCrunchedLength; s++)	    
          bestScore[s] = unlikely;

        for(p = 0; p < NUM_PROT_MODELS - 2; p++)
        {
          int 
            model;

          for(model = 0; model < localTree->NumberOfModels; model++)
          { 
            boolean 
              execute = ((tr->manyPartitions && isThisMyPartition(tr, tid, model, n)) || (!tr->manyPartitions));

            if(execute)
            {
              double
                lh;

              int
                counter = 0,
                        i,
                        lower = localTree->partitionData[model].lower,
                        upper = localTree->partitionData[model].upper;

              memcpy(localTree->partitionData[model].EIGN,        localTree->siteProtModel[p].EIGN,        sizeof(double) * 19);
              memcpy(localTree->partitionData[model].EV,          localTree->siteProtModel[p].EV,          sizeof(double) * 400);                
              memcpy(localTree->partitionData[model].EI,          localTree->siteProtModel[p].EI,          sizeof(double) * 380);
              memcpy(localTree->partitionData[model].substRates,  localTree->siteProtModel[p].substRates,  sizeof(double) * 190);        
              memcpy(localTree->partitionData[model].frequencies, localTree->siteProtModel[p].frequencies, sizeof(double) * 20);
              memcpy(localTree->partitionData[model].tipVector,   localTree->siteProtModel[p].tipVector,   sizeof(double) * 460);

              for(i = lower, counter = 0; i < upper; i++)
              {
                if(tr->manyPartitions || (i % n == tid))
                {
                  lh = evaluatePartialGeneric(localTree, counter, 0.0, model);

                  if(lh > bestScore[i])
                  {
                    bestScore[i] = lh; 
                    localTree->partitionData[model].perSiteAAModel[counter] = p;			    
                  }
                  counter++;
                }
              }
            }	     	           
          }
        }

        free(bestScore);      					
      }
      break;
    default:
      printf("Job %d\n", currentJob);
      assert(0);
  }
}




void masterBarrier(int jobType, tree *tr)
{
  const int 
    n = NumberOfThreads;

  int 
    i, 
    sum;

  jobCycle = !jobCycle;
  threadJob = (jobType << 16) + jobCycle;

  execFunction(tr, tr, 0, n);


  do
  {
    for(i = 1, sum = 1; i < n; i++)
      sum += barrierBuffer[i];
  }
  while(sum < n);  

  for(i = 1; i < n; i++)
    barrierBuffer[i] = 0;
}

#ifndef _PORTABLE_PTHREADS

static void pinToCore(int tid)
{
  cpu_set_t cpuset;

  CPU_ZERO(&cpuset);    
  CPU_SET(tid, &cpuset);

  if(pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0)
  {
    printBothOpen("\n\nThere was a problem finding a physical core for thread number %d to run on.\n", tid);
    printBothOpen("Probably this happend because you are trying to run more threads than you have cores available,\n");
    printBothOpen("which is a thing you should never ever do again, good bye .... \n\n");
    assert(0);
  }
}

#endif

static void *likelihoodThread(void *tData)
{
  threadData *td = (threadData*)tData;
  tree
    *tr = td->tr,
    *localTree = (tree *)malloc(sizeof(tree));
  int
    myCycle = 0;

  const int 
    n = NumberOfThreads,
      tid = td->threadNumber;

#ifndef _PORTABLE_PTHREADS
  pinToCore(tid);
#endif

  printf("\nThis is RAxML Worker Pthread Number: %d\n", tid);

  while(1)
  {
    while (myCycle == threadJob);
    myCycle = threadJob;

    execFunction(tr, localTree, tid, n);


    barrierBuffer[tid] = 1;     
  }

  return (void*)NULL;
}

static void startPthreads(tree *tr)
{
  pthread_t *threads;
  pthread_attr_t attr;
  int rc, t;
  threadData *tData;

  jobCycle        = 0;
  threadJob       = 0;

  printf("\nThis is the RAxML Master Pthread\n");

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

  pthread_mutex_init(&mutex , (pthread_mutexattr_t *)NULL);

  threads    = (pthread_t *)malloc(NumberOfThreads * sizeof(pthread_t));
  tData      = (threadData *)malloc(NumberOfThreads * sizeof(threadData));
  reductionBuffer          = (volatile double *)malloc(sizeof(volatile double) *  NumberOfThreads * tr->NumberOfModels);
  reductionBufferTwo       = (volatile double *)malloc(sizeof(volatile double) *  NumberOfThreads * tr->NumberOfModels);
  reductionBufferThree     = (volatile double *)malloc(sizeof(volatile double) *  NumberOfThreads * tr->NumberOfModels);
  reductionBufferParsimony = (volatile int *)malloc(sizeof(volatile int) *  NumberOfThreads);


  barrierBuffer            = (volatile char *)malloc(sizeof(volatile char) *  NumberOfThreads);

  for(t = 0; t < NumberOfThreads; t++)
    barrierBuffer[t] = 0;


  branchInfos              = (volatile branchInfo **)malloc(sizeof(volatile branchInfo *) * NumberOfThreads);

  for(t = 1; t < NumberOfThreads; t++)
  {
    tData[t].tr  = tr;
    tData[t].threadNumber = t;
    rc = pthread_create(&threads[t], &attr, likelihoodThread, (void *)(&tData[t]));
    if(rc)
    {
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }
}

#endif


#endif








static int iterated_bitcount(unsigned int n)
{
  int 
    count=0;    

  while(n)
  {
    count += n & 0x1u ;    
    n >>= 1 ;
  }

  return count;
}

static char bits_in_16bits [0x1u << 16];

static void compute_bits_in_16bits(void)
{
  unsigned int i;    

  for (i = 0; i < (0x1u<<16); i++)
    bits_in_16bits[i] = iterated_bitcount(i);

  return ;
}

unsigned int precomputed16_bitcount (unsigned int n)
{
  /* works only for 32-bit int*/

  return bits_in_16bits [n         & 0xffffu]
    +  bits_in_16bits [(n >> 16) & 0xffffu] ;
}



#if (defined(_USE_PTHREADS) || (_FINE_GRAIN_MPI))    

static int partCompare(const void *p1, const void *p2)
{
  partitionType 
    *rc1 = (partitionType *)p1,
    *rc2 = (partitionType *)p2;

  int 
    i = rc1->partitionLength,
      j = rc2->partitionLength;

  if (i > j)
    return (-1);
  if (i < j)
    return (1);
  return (0);
}

static void multiprocessorScheduling(tree *tr)
{
  size_t   
    checkSum = 0,
             sum = 0;

  int    
    i,
#ifndef _FINE_GRAIN_MPI
    n = NumberOfThreads,
#else
    n = processes,
#endif
    p = tr->NumberOfModels,    
    *assignments = (int *)calloc(n, sizeof(int));  

  partitionType 
    *pt = (partitionType *)malloc(sizeof(partitionType) * p);

  tr->partitionAssignment = (int *)malloc(p * sizeof(int));

  for(i = 0; i < p; i++)
  {
    pt[i].partitionNumber = i;
    pt[i].partitionLength = tr->partitionData[i].upper - tr->partitionData[i].lower;
    sum += (size_t)pt[i].partitionLength;
  }

  qsort(pt, p, sizeof(partitionType), partCompare);

  /*for(i = 0; i < p; i++)
    printf("%d %d\n", pt[i].partitionLength, pt[i].partitionNumber);*/

  for(i = 0; i < p; i++)
  {
    int 
      k, 
      min = INT_MAX,
      minIndex = -1;

    for(k = 0; k < n; k++)	
      if(assignments[k] < min)
      {
        min = assignments[k];
        minIndex = k;
      }

    assert(minIndex >= 0);

    assignments[minIndex] +=  pt[i].partitionLength;
    assert(pt[i].partitionNumber >= 0 && pt[i].partitionNumber < p);
    tr->partitionAssignment[pt[i].partitionNumber] = minIndex;
  }

  printBothOpen("\nMulti-processor partition data distribution enabled (-Q option)\n");

  for(i = 0; i < n; i++)
  {      
    printBothOpen("Process %d has %d sites\n", i, assignments[i]);
    checkSum += (size_t)assignments[i];
  }
  printBothOpen("\n");

  /*
     for(i = 0; i < p; i++)
     printf("%d ", tr->partitionAssignment[i]);
     printf("\n");
     */

  assert(sum == checkSum);

  free(assignments);
  free(pt);
}

#endif




int main (int argc, char *argv[])
{
  rawdata      *rdta;
  cruncheddata *cdta;
  tree         *tr;
  analdef      *adef;
  int
    i,
    countGTR = 0,
    countOtherModel = 0;  

#if (defined(_USE_PTHREADS) && !defined(_PORTABLE_PTHREADS))  
  pinToCore(0);
#endif 

#if ! (defined(__ppc) || defined(__powerpc__) || defined(PPC))
  _mm_setcsr( _mm_getcsr() | _MM_FLUSH_ZERO_ON);
#endif 

#ifdef _FINE_GRAIN_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &processID);
  MPI_Comm_size(MPI_COMM_WORLD, &processes);
  printf("\nThis is RAxML FINE-GRAIN MPI Process Number: %d\n", processID);   
  MPI_Barrier(MPI_COMM_WORLD);
#else
  processID = 0;
#endif

  masterTime = gettime();

  adef = (analdef *)malloc(sizeof(analdef));
  rdta = (rawdata *)malloc(sizeof(rawdata));
  cdta = (cruncheddata *)malloc(sizeof(cruncheddata));
  tr   = (tree *)malloc(sizeof(tree));

  /* initialize lookup table for fast bit counter */


#ifdef _FINE_GRAIN_MPI
  if(processID == 0)
  {
#endif

    compute_bits_in_16bits();

    initAdef(adef);
    get_args(argc,argv, adef, tr); 

    getinput(adef, rdta, cdta, tr);

    checkOutgroups(tr, adef);
    makeFileNames();  

    makeweights(adef, rdta, cdta, tr);     

    makevalues(rdta, cdta, tr, adef);      

    tr->innerNodes = tr->mxtips;

    setRateHetAndDataIncrement(tr, adef);

    /*
       if(adef->readBinaryFile)
       fclose(byteFile);
       */

    if(adef->writeBinaryFile)
    {
      char 
        byteFileName[1024] = "";

      int 
        model;

      strcpy(byteFileName, workdir);
      strcat(byteFileName, seq_file);
      strcat(byteFileName, ".binary");

      printBothOpen("\n\nBinary and compressed alignment file written to file %s\n\n", byteFileName);
      printBothOpen("Parsing completed, exiting now ... \n\n");

      for(model = 0; model < tr->NumberOfModels; model++)
      {	  	      
        const 
          partitionLengths *pl = getPartitionLengths(&(tr->partitionData[model]));

        tr->partitionData[model].frequencies       = (double*)malloc(pl->frequenciesLength * sizeof(double));
      }

      baseFrequenciesGTR(tr->rdta, tr->cdta, tr); 

      for(model = 0; model < tr->NumberOfModels; model++)	    	    
        myBinFwrite(tr->partitionData[model].frequencies, sizeof(double), tr->partitionData[model].states);	      	   

      fclose(byteFile);
      return 0;
    }


#if (defined(_USE_PTHREADS) || (_FINE_GRAIN_MPI))
    if(tr->manyPartitions)
      multiprocessorScheduling(tr);
#endif


#ifdef _USE_PTHREADS
    startPthreads(tr);
    masterBarrier(THREAD_INIT_PARTITION, tr);       
#else      
#ifdef _FINE_GRAIN_MPI
    startFineGrainMpi(tr, adef);
#else
    allocNodex(tr);    
#endif
#endif

    /* recom */
#ifdef _DEBUG_RECOMPUTATION
    allocTraversalCounter(tr);
    tr->stlenTime = 0.0;
#endif
    /* E  recom */



    printModelAndProgramInfo(tr, adef, argc, argv);

    printBothOpen("Memory Saving via Subtree Equality Vectors: %s\n", (tr->saveMemory == TRUE)?"ENABLED":"DISABLED");   	             
    /* recom */   
    printBothOpen("Memory Saving via Additional Vector Recomputations: %s\n", (tr->useRecom == TRUE)?"ENABLED":"DISABLED");
    if(tr->useRecom)
      printBothOpen("Using a fraction %f of the total inner vectors that would normally be required\n", tr->vectorRecomFraction);
    /* E recom */

    initModel(tr, rdta, cdta, adef);                      

    if(tr->searchConvergenceCriterion)
    {                     
      tr->bitVectors = initBitVector(tr, &(tr->vLength));
      tr->h = initHashTable(tr->mxtips * 4);        
    }

    if(adef->useCheckpoint)
    {
#ifdef _JOERG
      /* this is for a checkpoint-based restart, we don't need this here 
         so we will just exit gracefully */

      assert(0);
#endif

      restart(tr, adef);

      computeBIGRAPID(tr, adef, TRUE); 
    }
    else
    {
      accumulatedTime = 0.0;   

      getStartingTree(tr, adef);     
#ifdef _JOERG		  
      /* 
         at this point the code has parsed the input alignment 
         and read the tree on which we want to estimate the best 
         model. We now just branch into the function on which you can work.
         This function will never return, hence, you don't need to worry 
         about the rest of the code below modOptJoerg().
         */

      modOptJoerg(tr, adef);
#else
      evaluateGenericInitrav(tr, tr->start);	 
      treeEvaluate(tr, 1); 	 	 	 	 	 
      computeBIGRAPID(tr, adef, TRUE); 	     
#ifdef _DEBUG_RECOMPUTATION
      /* recom */
      printBothOpen("Traversal freq after search \n");
      printTraversalInfo(tr);
      printBothOpen("stlen update (recom specific orientations full trav) time %f \n", tr->stlenTime);
      /* E recom */
#endif
#endif
    }            

    finalizeInfoFile(tr, adef);

#ifdef _FINE_GRAIN_MPI
    masterBarrierMPI(EXIT_GRACEFULLY, tr);
  }
  else
  {
    compute_bits_in_16bits();
    fineGrainWorker(tr);
  }
#endif

#ifdef _FINE_GRAIN_MPI
  MPI_Finalize();    
#endif

  return 0;
}


