/*

Fast mex K-means clustering algorithm with possibility of K-mean++ initialization
(mex-interface modified from the original yael package https://gforge.inria.fr/projects/yael)

- Accept single/double precision input
- Support of BLAS/OpenMP for multi-core computation

Usage
-----

[centroids, dis, assign , nassign , qerr] = yael_kmeans(X , [options]);


Inputs
------

X                                        Input data matrix (d x N) in single/double format 

options
       K                                 Number of centroid  (default K = 10)
       max_ite                           Number of iteration (default max_ite = 50)
       redo                              Number of time to restart K-means (default redo = 1)
       verbose                           Verbose level = {0,1} (default verbose = 0)
       init_random_mode                  0 <=> Kmeans++ initialization, 1<=> random selection ...
       normalize_sophisticated_mode      0/1 (No/Yes)
       BLOCK_N1                          Cache size block (default BLOCK_N1 = 1024)
       BLOCK_N2                          Cache size block (default BLOCK_N2 = 1024)
       seed                              Seed number for internal random generator (default random seed according to time)

If compiled with the "OMP" compilation flag

       num_threads                       Number of threads   (default num_threads = max number of core)

Outputs
-------

centroids                                Centroids matrix (d x K) in single/double format 
dis                                      Distance of each xi to the closest centroid (1 x N) in single/double format
assign                                   Index of closest centroid to xi (1 x N) in UINT32 format
nassign                                  Number of data associated with each cluster (1 x K) in UINT32 format
qerr                                     Quantification error during training process (1 x options.max_ite) in single/double format


Example 1
---------

clear

d                                    = 128;                   % dimensionality of the vectors
N                                    = 100000;                % number of vectors

X                                    = randn(d, N , 'single'); % random set of vectors 

options.K                            = 100;
options.max_ite                      = 30;
options.init_random_mode             = 0;
options.normalize_sophisticated_mode = 0;
options.BLOCK_N1                     = 1024;
options.BLOCK_N2                     = 1024;

options.seed                         = 1234543;
options.num_threads                  = 2;


tic,[centroids, dis, assign , nassign , qerr] = yael_kmeans(X , options);,toc


Example 2
---------

clear

d                                    = 128;                   % dimensionality of the vectors
N                                    = 100000;                % number of vectors

X                                    = randn(d, N); % random set of vectors 

options.K                            = 100;
options.max_ite                      = 30;
options.init_random_mode             = 0;
options.normalize_sophisticated_mode = 0;
options.BLOCK_N1                     = 1024;
options.BLOCK_N2                     = 1024;

options.seed                         = 1234543;
options.num_threads                  = 1;


tic,[centroids, dis, assign , nassign , qerr] = yael_kmeans(X , options);,toc



Example 3
---------

clear

d                                    = 128;                   % dimensionality of the vectors
N                                    = 100000;                % number of vectors

X                                    = randn(d, N , 'single'); % random set of vectors 

options.K                            = 100;
options.max_ite                      = 30;
options.init_random_mode             = 0;
options.normalize_sophisticated_mode = 0;
options.BLOCK_N1                     = 1024;
options.BLOCK_N2                     = 1024;

options.seed                         = 1234543;
options.num_threads                  = -1;
options.verbose                      = 1;


tic,[centroids, dis, assign , nassign , qerr] = yael_kmeans(X , options);,toc




To compile
----------

mex  yael_kmeans.c 

mex  -g  yael_kmeans.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -g -DBLAS -DOMP yael_kmeans.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex  -f mexopts_intel10.bat yael_kmeans.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

If compiled with OMP option, OMP support

mex -v -DOMP  yael_kmeans.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -DOMP -f mexopts_intel10.bat yael_kmeans.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

If compiled with BLAS & OMP options

mex -v -DBLAS -DOMP  yael_kmeans.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -DBLAS -DOMP -f mexopts_intel10.bat yael_kmeans.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"



Reference  [1] Hervé Jégou, Matthijs Douze and Cordelia Schmid, Product quantization for nearest neighbor search,
---------      IEEE Transactions on Pattern Analysis and Machine Intelligence


Author : Sébastien PARIS : sebastien.paris@lsis.org
-------  Date : 10/28/2011

Changelog : 
---------
            v 1.1 Add online help 11/01/2011
            v 1.0 Initial release 10/31/2011

*/

#include <time.h>
#include <math.h>
#include <mex.h>

#ifdef OMP
 #include <omp.h>
#endif

#ifndef max
    #define max(a,b) (a >= b ? a : b)
    #define min(a,b) (a <= b ? a : b)
#endif

#define NMAX_KMEANSPP 8192
#define rationNK_KMEANSPP 8

#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

#if defined(__OS2__)  || defined(__WINDOWS__) || defined(WIN32) || defined(WIN64) || defined(_MSC_VER)
#define BLASCALL(f) f
#else
#define BLASCALL(f) f ##_
#endif

#define SHR3   ( jsr ^= (jsr<<17), jsr ^= (jsr>>13), jsr ^= (jsr<<5) )
#define randint SHR3
#define rand() (0.5 + (signed)randint*2.328306e-10)

#ifdef __x86_64__
    typedef int UL;
#else
    typedef unsigned long UL;
#endif

static UL jsrseed = 31340134 , jsr;

struct opts
{
	int    K;
	int    max_ite;
	int    redo;
	int    verbose;
	int    init_random_mode;
	int    normalize_sophisticated_mode;
	int    BLOCK_N1;
	int    BLOCK_N2;
	UL     seed;
#ifdef OMP 
    int    num_threads;
#endif
};

/* ------------------------------------------------------------------------------------------------------------------------------------ */
extern void   BLASCALL(sgemm)(char *, char *, int*, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
extern void   BLASCALL(sgemv)(char *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
extern void   BLASCALL(saxpy)(int *, float *, float *, int *, float *, int *);
extern void   BLASCALL(sscal)(int *, float *, float  *, int *);
extern float  BLASCALL(snrm2)(int * , float * , int *);

extern void   BLASCALL(dgemm)(char *, char *, int*, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
extern void   BLASCALL(dgemv)(char *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
extern void   BLASCALL(daxpy)(int *, double *, double *, int *, double *, int *);
extern void   BLASCALL(dscal)(int *, double *, double  *, int *);
extern double BLASCALL(dnrm2)(int * , double * , int *);

void skmeans(float *, int , int , struct opts , float *, float *, int *, int *,float *);
void skmeanspp_init(float * , int , int , int , int * , float *, float *);
void scompute_distances_1(float *, float * , int , int , float *);
void scompute_cross_distances_nonpacked(float *, float *, int , int , int , float *, float *);
void snn_single_full(float * , float * , int , int , int , int , int , float * , float * , float *, int *);

void dkmeans(double *, int , int , struct opts , double *, double *, int *, int *,double *);
void dkmeanspp_init(double * , int , int , int , int * , double *, double *);
void dcompute_distances_1(double *, double * , int , int , double *);
void dcompute_cross_distances_nonpacked(double *, double *, int , int , int , double *, double *);
void dnn_single_full(double * , double * , int , int , int , int , int , double * , double * , double *, int *);

void randperm(int * , int );
void randini(UL);

/* ------------------------------------------------------------------------------------------------------------------------------------ */
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
	int d,N,i , issingle = 0;
	float  *sX , *scentroids , *sdis , *sqerr;
	double *dX , *dcentroids , *ddis , *dqerr;
	int *assign , *nassign;

#ifdef OMP 
	struct opts options = {10 , 50 , 1 , 0 , 0 , 0 , 1024 , 1024 ,(UL)NULL , -1};
#else
	struct opts options = {10 , 50 , 1 , 0 , 0 , 0 , 1024 , 1024 , (UL)NULL};
#endif
	mxArray *mxtemp;
	double *tmp;
	int tempint;
	UL templint;

	if ((nrhs < 1) || (nrhs > 2)) 
	{
		mexPrintf(
			"\n"
			"\n"
			"Fast mex K-means clustering algorithm with possibility of K-mean++ initialization\n"
			"(mex-interface modified from the original yael package https://gforge.inria.fr/projects/yael)\n"
			"\n"
			"- Accept single/double precision input\n"
			"- Support of BLAS/OpenMP for multi-core computation\n"
			"\n"
			"Usage\n"
			"-----\n"
			"\n"
			"[centroids, dis, assign , nassign , qerr] = yael_kmeans(X , [options]);\n"
			"\n"
			"\n"
			"Inputs\n"
			"------\n"
			"\n"
			"X                                        Input data matrix (d x N) in single/double format\n"
			"\n"
			"options\n"
			"       K                                 Number of centroid  (default K = 10)\n"
			"       max_ite                           Number of iteration (default max_ite = 50)\n"
			"       redo                              Number of time to restart K-means (default redo = 1)\n"
			"       verbose                           Verbose level = {0,1} (default verbose = 0)\n"
			"       init_random_mode                  0 <=> Kmeans++ initialization, 1<=> random selection ...\n"
			"       normalize_sophisticated_mode      0/1 (No/Yes)\n"
			"       BLOCK_N1                          Cache size block (default BLOCK_N1 = 1024)\n"
			"       BLOCK_N2                          Cache size block (default BLOCK_N2 = 1024)\n"
			"       seed                              Seed number for internal random generator (default random seed according to time)\n"
#ifdef OMP 
			"       num_threads                       Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)\n"
#endif
			"\n"
			"\n"
			"Outputs\n"
			"-------\n"
			"\n"
			"centroids                                Centroids matrix (d x K) in single/double format\n"
			"dis                                      Distance of each xi to the closest centroid (1 x N) in single/double format\n"
			"assign                                   Index of closest centroid to xi (1 x N) in UINT32 format\n"
			"nassign                                  Number of data associated with each cluster (1 x K) in UINT32 format\n"
			"qerr                                     Quantification error during training process (1 x options.max_ite) in single/double format\n"
			"\n"
			"\n"
			);
		return;
	}

	if(mxIsSingle(prhs[0]))
	{
		sX       = (float *)mxGetData(prhs[0]);
		issingle = 1;
	}
	else
	{
		dX       = (double *)mxGetData(prhs[0]);
	}

	d           = mxGetM(prhs[0]);
	N           = mxGetN(prhs[0]);

	if ((nrhs > 1) && !mxIsEmpty(prhs[1]))
	{
		mxtemp                            = mxGetField(prhs[1] , 0 , "K");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 0))
			{
				mexPrintf("K must be > 0, force to 10\n");	
				options.K                 = 0;
			}
			else
			{
				options.K                 = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "max_ite");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 0))
			{
				mexPrintf("max_ite must be > 0, force to 50\n");	
				options.max_ite           = 50;
			}
			else
			{
				options.max_ite             = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "redo");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0))
			{
				mexPrintf("redo must be >= 0, force to 1\n");	
				options.redo              = 1;
			}
			else
			{
				options.redo              = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "verbose");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("verbose must be ={0,1}, force to 0\n");	
				options.verbose              = 0;
			}
			else
			{
				options.verbose              = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "init_random_mode");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("init_random_mode must be ={0,1}, force to 0\n");	
				options.init_random_mode  = 0;
			}
			else
			{
				options.init_random_mode  = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "normalize_sophisticated_mode");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("normalize_sophisticated_mode must be ={0,1}, force to 0\n");	
				options.normalize_sophisticated_mode  = 0;
			}
			else
			{
				options.normalize_sophisticated_mode  = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "BLOCK_N1");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0))
			{
				mexPrintf("BLOCK_N1 must be >0 a power of 2, force to 1024\n");	
				options.BLOCK_N1         = 1024;
			}
			else
			{
				options.BLOCK_N1         = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "BLOCK_N2");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0))
			{
				mexPrintf("BLOCK_N2 must be >0 a power of 2, force to 1024\n");	
				options.BLOCK_N2         = 1024;
			}
			else
			{
				options.BLOCK_N2         = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "seed");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			templint                      = (UL) tmp[0];
			if( (templint < 1) )
			{
				mexPrintf("seed >= 1 , force to NULL (random seed)\n");	
				options.seed             = (UL)NULL;
			}
			else
			{
				options.seed             = templint;
			}
		}

#ifdef OMP
		mxtemp                            = mxGetField(prhs[1] , 0 , "num_threads");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < -1))
			{
				mexPrintf("num_threads must be >= -1, force to -1\n");	
				options.num_threads       = -1;
			}
			else
			{
				options.num_threads       = tempint;
			}
		}
#endif
	}

	if(N < options.K) 
	{
		mexErrMsgTxt("fewer points than centroids");    
	}

	/*------------------------ Main Call ----------------------------*/

	randini(options.seed);

	if(issingle)
	{
		plhs[0]    = mxCreateNumericMatrix (d , options.K , mxSINGLE_CLASS, mxREAL);  
		scentroids = (float*)mxGetPr(plhs[0]);

		plhs[1]    = mxCreateNumericMatrix (1, N , mxSINGLE_CLASS, mxREAL);
		sdis       = (float*) mxGetPr (plhs[1]);

		plhs[2]    = mxCreateNumericMatrix (1, N , mxINT32_CLASS, mxREAL);
		assign     = (int*) mxGetPr (plhs[2]);

		plhs[3]    = mxCreateNumericMatrix (1, options.K , mxINT32_CLASS, mxREAL);
		nassign    = (int*) mxGetPr (plhs[3]);

		plhs[4]    = mxCreateNumericMatrix (1, options.max_ite , mxSINGLE_CLASS, mxREAL);
		sqerr      = (float *) mxGetPr (plhs[4]);

		skmeans(sX , d , N , options , scentroids , sdis , assign , nassign , sqerr);
	}
	else
	{
		plhs[0]    = mxCreateNumericMatrix (d , options.K , mxDOUBLE_CLASS, mxREAL);  
		dcentroids = (double*)mxGetPr(plhs[0]);

		plhs[1]    = mxCreateNumericMatrix (1, N , mxDOUBLE_CLASS, mxREAL);
		ddis       = (double*) mxGetPr (plhs[1]);

		plhs[2]    = mxCreateNumericMatrix (1, N , mxINT32_CLASS, mxREAL);
		assign     = (int*) mxGetPr (plhs[2]);

		plhs[3]    = mxCreateNumericMatrix (1, options.K , mxINT32_CLASS, mxREAL);
		nassign    = (int*) mxGetPr (plhs[3]);

		plhs[4]    = mxCreateNumericMatrix (1, options.max_ite , mxDOUBLE_CLASS, mxREAL);
		dqerr      = (double *) mxGetPr (plhs[4]);

		dkmeans(dX , d , N , options , dcentroids , ddis , assign , nassign , dqerr);
	}

	for (i = 0 ; i < N ; i++)
	{
		assign[i]++;
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void skmeans(float *X , int d , int N , struct opts options  ,  float *centroids_out , float *dis_out, int *assign_out, int *nassign_out , float *qerr_out)
{
	float *dists , *sum_c , *norms , *disbest , *distmp;
	float *centroids, *dis , *quanterr;
	int *assign, *nassign;
	int *selected;
	int K = options.K, Kd = K*d, redo = options.redo , verbose = options.verbose , max_ite = options.max_ite;
	int normalize_sophisticated_mode = options.normalize_sophisticated_mode , init_random_mode = options.init_random_mode;
	int step1 = min(N, options.BLOCK_N1), step2 = min(K, options.BLOCK_N2);
	int run , Nsubset;
	int i , j , id , jd , index , iter , iter_tot = 0;
	float temp , sum;
	double qerr, qerr_old , qerr_best = HUGE_VAL ;

#ifdef BLAS
	float one = 1.0f;
	int inc   = 1;
#endif
#ifdef OMP 
    int num_threads = options.num_threads;
    num_threads     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif


	centroids     = (float *)malloc(Kd*sizeof(float));
	dis           = (float *)malloc(N*sizeof(float));
	quanterr      = (float *)malloc(max_ite*sizeof(float));

	dists         = (float *)malloc(step1*step2*sizeof(float));
	sum_c         = (float *)malloc(step2*sizeof(float));
	selected      = (int *)  malloc(N*sizeof(int)); /* Only the first K<N will be used */

	assign        = (int *)  malloc(N*sizeof(int)); 
	nassign       = (int *)  malloc(K*sizeof(int)); 


	if(normalize_sophisticated_mode)
	{			
		norms     = (float *)malloc(K*sizeof(float));
	}

	if(!init_random_mode) 
	{
		Nsubset = N;
		if((N>(K*rationNK_KMEANSPP)) && (N>NMAX_KMEANSPP)) 
		{
			Nsubset = K*rationNK_KMEANSPP;
			if(verbose)
			{
				mexPrintf("Restricting k-means++ initialization to %d points\n" , Nsubset);
				mexEvalString("drawnow;");

			}
		}
		disbest       = (float *) malloc(Nsubset*sizeof(float));
		distmp        = (float *) malloc(Nsubset*sizeof(float));
	}

	for (run = 0 ; run < redo ; run++) 
	{
do_redo: 

		if(verbose)
		{
			mexPrintf("<><><><> kmeans / run %d <><><><><>\n", run+1);
			mexEvalString("drawnow;");
		}

		if(init_random_mode) 
		{
			randperm(selected , N);
		} 
		else 
		{
			skmeanspp_init(X , d , Nsubset , K , selected , disbest , distmp);
		}

		for(j = 0 ; j < K ; j++)
		{
			index = selected[j]*d;
			jd    = j*d;
#ifdef BLAS
			memcpy(centroids + jd , X + index , d*sizeof(float));
#else
			for(i = 0 ; i < d ; i++)
			{
				centroids[i + jd] = X[i + index];
			}
#endif
		}

		/* the quantization error */
		qerr = HUGE_VAL;

		for (iter = 1 ; iter <= max_ite ; iter++) 
		{

#ifdef BLAS
			memset(nassign , 0 , K*sizeof(int));
#else
			for(i = 0 ; i < K ; i++)
			{
				nassign[i] = 0;
			}
#endif

			iter_tot++;

			/* Assign point to cluster and count the cluster size */

			snn_single_full(X , centroids , d , N , K , step1 , step2 , dis , dists , sum_c , assign );

			for (i = 0 ; i < N ; i++) 
			{
				nassign[assign[i]]++;
			}

			for (i = 0 ; i < K ; i++) 
			{
				if(nassign[i]==0) 
				{
					if(verbose)
					{
						mexPrintf("WARN nassign %d is 0, redoing!\n",(int)i);
						mexEvalString("drawnow;");

					}
					goto do_redo;
				}
			}

			if(normalize_sophisticated_mode)
			{
#ifdef BLAS
				memset(centroids , 0 , Kd*sizeof(float));
				memset(norms , 0 , K*sizeof(float));
#else
				for( i = 0 ; i < Kd ; i++)
				{
					centroids[i] = 0.0f;
				}
				for( i = 0 ; i < K ; i++)
				{
					norms[i]    = 0.0f;
				}
#endif
				for(i = 0 ; i < N ; i++)
				{
					index = assign[i]*d;
					id    = i*d;
#ifdef BLAS
					BLASCALL(saxpy)(&d , &one , X + id , &inc , centroids + index , &inc);
					sum  = BLASCALL(snrm2)(&d , X + id , &inc);
					sum *= sum;
#else
					sum   = 0.0f;
					for(j = 0  ; j < d ; j++)
					{
						temp                  = X[j + id];
						centroids[j + index] += temp;
						sum                  += (temp*temp);
					}
#endif
					norms[assign[i]]         += (float)sqrt(sum);

				}
				for (i = 0 ; i < K ; i++) 
				{
					id  = i*d;
#ifdef BLAS
					sum = BLASCALL(snrm2)(&d , centroids + id , &inc);
					sum *= sum;
#else
					sum = 0.0f;
					for(j = 0  ; j < d ; j++)
					{
						temp  = centroids[j + id];
						sum  += (temp*temp);
					}
#endif
					sum  = (sum != 0.0f) ? (1.0f/sqrt(sum)) : 1.0f;
					temp = sum*(norms[i]/nassign[i]);
#ifdef BLAS
					BLASCALL(sscal)(&d , &temp , centroids + id , &inc);
#else
					for(j = 0  ; j < d ; j++)
					{
						centroids[j + id] *= temp;
					}
#endif
				}
			} 
			else 
			{
#ifdef BLAS		                
				memset(centroids , 0 , Kd*sizeof(float));
#else
				for( i = 0 ; i < Kd ; i++)
				{
					centroids[i] = 0.0f;
				}
#endif

/*
#ifdef OMP
#ifdef BLAS
#pragma omp parallel for default(none) private(i,j,index,id) shared(X,N,d,assign,centroids,inc,one)
#else
#pragma omp parallel for default(none) private(i,j,index,id) shared(X,N,d,assign,centroids)
#endif
#endif
*/
				for(i = 0 ; i < N ; i++)
				{
					index = assign[i]*d;
					id    = i*d;
#ifdef BLAS
					BLASCALL(saxpy)(&d , &one , X + id , &inc , centroids + index , &inc);
#else
					for(j = 0  ; j < d ; j++)
					{
						centroids[j + index] += X[j + id];
					}
#endif
				}

#ifdef OMP
#ifdef BLAS
#pragma omp parallel for default(none) private(i,j,temp,id) shared(K,d,nassign,centroids,inc)
#else
#pragma omp parallel for default(none) private(i,j,temp,id) shared(K,d,nassign,centroids)
#endif
#endif
				for (i = 0 ; i < K ; i++) 
				{
					id   = i*d;
					temp = 1.0f/nassign[i];
#ifdef BLAS
					BLASCALL(sscal)(&d , &temp , centroids + id , &inc);
#else
					for(j = 0  ; j < d ; j++)
					{
						centroids[j + id] *= temp;
					}
#endif
				}
			}
			qerr_old  = qerr;
			qerr      = 0.0f;
			for(i = 0 ; i < N ; i++)
			{
				qerr += dis[i];
			}
			quanterr[iter-1] = qerr;
			if(verbose)
			{
				mexPrintf("Kmean ite = (%d/%d), qerr = %10.10f\n",iter,max_ite,qerr);
				mexEvalString("drawnow;");
			}
			if (qerr_old == qerr)
			{
				break;
			}
		}

		if (qerr < qerr_best) 
		{
			qerr_best = qerr;
			memcpy(centroids_out , centroids , Kd*sizeof(float));
			memcpy(dis_out, dis , N*sizeof(float));
			memcpy(qerr_out, quanterr , max_ite*sizeof(float));
			memcpy(assign_out , assign , N*sizeof(int));
			memcpy(nassign_out , nassign , K*sizeof (int));
		}
	}

	if(verbose)
	{
		mexPrintf ("Total number of iterations: %d\n", (int)iter_tot);
		mexEvalString("drawnow;");
	}

	free(centroids);
	free(dis);
	free(quanterr);
	free(assign);
	free(nassign);
	free(selected);
	free(sum_c);
	if(normalize_sophisticated_mode)
	{			
		free(norms);
	}
	if(!init_random_mode) 
	{
		free(disbest);      
		free(distmp);  
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dkmeans(double *X , int d , int N , struct opts options  ,  double *centroids_out , double *dis_out, int *assign_out, int *nassign_out , double *qerr_out)
{
	double *dists , *sum_c , *norms , *disbest , *distmp;
	double *centroids, *dis , *quanterr;
	int *assign, *nassign;
	int *selected;
	int K = options.K, Kd = K*d, redo = options.redo , verbose = options.verbose , max_ite = options.max_ite;
	int normalize_sophisticated_mode = options.normalize_sophisticated_mode , init_random_mode = options.init_random_mode;
	int step1 = min(N, options.BLOCK_N1), step2 = min(K, options.BLOCK_N2);
	int run , Nsubset;
	int i , j , id , jd , index , iter , iter_tot = 0;
	double temp , sum;
	double qerr, qerr_old , qerr_best = HUGE_VAL ;

#ifdef BLAS
	double one = 1.0;
	int inc = 1;
#endif
#ifdef OMP 
    int num_threads = options.num_threads;

    num_threads     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);

#endif

	centroids     = (double *)malloc(Kd*sizeof(double));
	dis           = (double *)malloc(N*sizeof(double));
	quanterr      = (double *)malloc(max_ite*sizeof(double));

	dists         = (double *)malloc(step1*step2*sizeof(double));
	sum_c         = (double *)malloc(step2*sizeof(double));
	selected      = (int *)  malloc(N*sizeof(int)); /* Only the first K<N will be used */

	assign        = (int *)  malloc(N*sizeof(int)); 
	nassign       = (int *)  malloc(K*sizeof(int)); 

	if(normalize_sophisticated_mode)
	{			
		norms     = (double *)malloc(K*sizeof(double));
	}

	if(!init_random_mode) 
	{
		Nsubset = N;
		if((N>(K*rationNK_KMEANSPP)) && (N>NMAX_KMEANSPP)) 
		{
			Nsubset = K*rationNK_KMEANSPP;
			if(verbose)
			{
				mexPrintf("Restricting k-means++ initialization to %d points\n" , Nsubset);
				mexEvalString("drawnow;");
			}
		}
		disbest       = (double *) malloc(Nsubset*sizeof(double));
		distmp        = (double *) malloc(Nsubset*sizeof(double));
	}

	for (run = 0 ; run < redo ; run++) 
	{
do_redo: 

		if(verbose)
		{
			mexPrintf("<><><><> kmeans / run %d <><><><><>\n", run+1);
			mexEvalString("drawnow;");
		}

		if(init_random_mode) 
		{
			randperm(selected , N);
		} 
		else 
		{
			dkmeanspp_init(X , d , Nsubset , K , selected , disbest , distmp);
		}

		for(j = 0 ; j < K ; j++)
		{
			index = selected[j]*d;
			jd    = j*d;
#ifdef BLAS
			memcpy(centroids + jd , X + index , d*sizeof(double));
#else
			for(i = 0 ; i < d ; i++)
			{
				centroids[i + jd] = X[i + index];
			}
#endif
		}

		/* the quantization error */
		qerr = HUGE_VAL;

		for (iter = 1 ; iter <= max_ite ; iter++) 
		{

#ifdef BLAS
			memset(nassign , 0 , K*sizeof(int));
#else
			for(i = 0 ; i < K ; i++)
			{
				nassign[i] = 0;
			}
#endif

			iter_tot++;

			/* Assign point to cluster and count the cluster size */

			dnn_single_full(X , centroids , d , N , K , step1 , step2 , dis , dists , sum_c , assign );

			for (i = 0 ; i < N ; i++) 
			{
				nassign[assign[i]]++;
			}

			for (i = 0 ; i < K ; i++) 
			{
				if(nassign[i]==0) 
				{
					if(verbose)
					{
						mexPrintf("WARN nassign %d is 0, redoing!\n",(int)i);
						mexEvalString("drawnow;");
					}
					goto do_redo;
				}
			}

			if(normalize_sophisticated_mode)
			{
#ifdef BLAS
				memset(centroids , 0 , Kd*sizeof(double));
				memset(norms , 0 , K*sizeof(double));
#else
				for( i = 0 ; i < Kd ; i++)
				{
					centroids[i] = 0.0;
				}
				for( i = 0 ; i < K ; i++)
				{
					norms[i]     = 0.0;
				}
#endif
				for(i = 0 ; i < N ; i++)
				{
					index = assign[i]*d;
					id    = i*d;
#ifdef BLAS
					BLASCALL(daxpy)(&d , &one , X + id , &inc , centroids + index , &inc);
					sum  = BLASCALL(dnrm2)(&d , X + id , &inc);
					sum *= sum;
#else
					sum   = 0.0;
					for(j = 0  ; j < d ; j++)
					{
						temp                  = X[j + id];
						centroids[j + index] += temp;
						sum                  += (temp*temp);
					}
#endif
					norms[assign[i]] += sqrt(sum);

				}
				for (i = 0 ; i < K ; i++) 
				{
					id  = i*d;
#ifdef BLAS
					sum = BLASCALL(dnrm2)(&d , centroids + id , &inc);
					sum *= sum;
#else
					sum = 0.0;
					for(j = 0  ; j < d ; j++)
					{
						temp  = centroids[j + id];
						sum  += (temp*temp);
					}
#endif
					sum  = (sum != 0.0) ? (1.0/sqrt(sum)) : 1.0;
					temp = sum*(norms[i]/nassign[i]);
#ifdef BLAS
					BLASCALL(dscal)(&d , &temp , centroids + id , &inc);
#else
					for(j = 0  ; j < d ; j++)
					{
						centroids[j + id] *= temp;
					}
#endif
				}
			} 
			else 
			{
#ifdef BLAS		                
				memset(centroids , 0 , Kd*sizeof(double));
#else
				for( i = 0 ; i < Kd ; i++)
				{
					centroids[i] = 0.0;
				}
#endif

/*
#ifdef OMP
#ifdef BLAS
#pragma omp parallel for default(none) private(i,j,index,id) shared(X,N,d,assign,centroids,inc,one)
#else
#pragma omp parallel for default(none) private(i,j,index,id) shared(X,N,d,assign,centroids)
#endif
#endif
*/
				for(i = 0 ; i < N ; i++)
				{
					index = assign[i]*d;
					id    = i*d;
#ifdef BLAS
					BLASCALL(daxpy)(&d , &one , X + id , &inc , centroids + index , &inc);
#else
					for(j = 0  ; j < d ; j++)
					{
						centroids[j + index] += X[j + id];
					}
#endif
				}

#ifdef OMP
#ifdef BLAS
#pragma omp parallel for default(none) private(i,j,temp,id) shared(K,d,nassign,centroids,inc)
#else
#pragma omp parallel for default(none) private(i,j,temp,id) shared(K,d,nassign,centroids)
#endif
#endif
				for (i = 0 ; i < K ; i++) 
				{
					id   = i*d;
					temp = 1.0/nassign[i];
#ifdef BLAS
					BLASCALL(dscal)(&d , &temp , centroids + id , &inc);
#else
					for(j = 0  ; j < d ; j++)
					{
						centroids[j + id] *= temp;
					}
#endif
				}
			}
			qerr_old  = qerr;
			qerr      = 0.0;
			for(i = 0 ; i < N ; i++)
			{
				qerr += dis[i];
			}
			quanterr[iter-1] = qerr;
			if(verbose)
			{
				mexPrintf("Kmean ite = (%d/%d), qerr = %10.10lf\n",iter,max_ite,qerr);
				mexEvalString("drawnow;");
			}
			if (qerr_old == qerr)
			{
				break;
			}
		}

		if (qerr < qerr_best) 
		{
			qerr_best = qerr;
			memcpy(centroids_out , centroids , Kd*sizeof(double));
			memcpy(dis_out, dis , N*sizeof(double));
			memcpy(qerr_out, quanterr , max_ite*sizeof(double));
			memcpy(assign_out , assign , N*sizeof(int));
			memcpy(nassign_out , nassign , K*sizeof (int));
		}
	}

	if(verbose)
	{
		mexPrintf ("Total number of iterations: %d\n", (int)iter_tot);
		mexEvalString("drawnow;");
	}
	
    free(centroids);
	free(dis);
	free(quanterr);
	free(assign);
	free(nassign);
	free(selected);
	free(sum_c);
	if(normalize_sophisticated_mode)
	{			
		free(norms);
	}
	if(!init_random_mode) 
	{
		free(disbest);      
		free(distmp);  
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void snn_single_full(float *X , float *centroids, int d , int N, int K , int step1 , int step2 , float *vwdis , float *dists , float *sum_c , int *vw )
{
	int m1 , m2 , imin, i1,i2,j1,j2 , i1d;
	float dmin;
	float *dline;

	for (i1 = 0; i1 < N ; i1 += step1) 
	{  
		m1  = min(step1 , N - i1);
		i1d = i1*d;

		/* clear mins */

		for (j1 = 0; j1 < m1; j1++) 
		{
			vw[j1 + i1]      = -1;
			vwdis[j1 + i1]   = 1e30;
		}
		for (i2 = 0; i2 < K ; i2 += step2) 
		{     
			m2 = min(step2 , K - i2);

			scompute_cross_distances_nonpacked(centroids + i2*d , X + i1d, d , m2 , m1 , dists , sum_c);

			/* update mins */

#ifdef OMP
#pragma omp parallel for default(none) private(j1,j2) shared(m1,m2,i1,i2,vw,vwdis,dists,dline,dmin,imin)
#endif
			for(j1 = 0 ; j1 < m1 ; j1++) 
			{
				dline   = dists + j1*m2;
				imin    = vw[i1 + j1];
				dmin    = vwdis[i1 + j1];
				for(j2 = 0 ; j2 < m2 ; j2++) 
				{
					if(dline[j2] < dmin) 
					{
						imin = j2 + i2;
						dmin = dline[j2];
					}
				}
				vw[i1 + j1]    = imin;
				vwdis[i1 + j1] = dmin;
			}      
		}  
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dnn_single_full(double *X , double *centroids, int d , int N, int K , int step1 , int step2 , double *vwdis , double *dists , double *sum_c , int *vw )
{
	int m1 , m2 , imin, i1,i2,j1,j2 , i1d;
	double dmin;
	double *dline;

	for (i1 = 0; i1 < N ; i1 += step1) 
	{  
		m1  = min(step1 , N - i1);
		i1d = i1*d;

		/* clear mins */

		for (j1 = 0; j1 < m1; j1++) 
		{
			vw[j1 + i1]      = -1;
			vwdis[j1 + i1]   = 1e200;
		}
		for (i2 = 0; i2 < K ; i2 += step2) 
		{     
			m2 = min(step2 , K - i2);

			dcompute_cross_distances_nonpacked(centroids + i2*d , X + i1d, d , m2 , m1 , dists , sum_c);

			/* update mins */

#ifdef OMP
#pragma omp parallel for default(none) private(j1,j2) shared(m1,m2,i1,i2,vw,vwdis,dists,dline,dmin,imin)
#endif
			for(j1 = 0 ; j1 < m1 ; j1++) 
			{
				dline   = dists + j1*m2;
				imin    = vw[i1 + j1];
				dmin    = vwdis[i1 + j1];
				for(j2 = 0 ; j2 < m2 ; j2++) 
				{
					if(dline[j2] < dmin) 
					{
						imin = j2 + i2;
						dmin = dline[j2];
					}
				}
				vw[i1 + j1]    = imin;
				vwdis[i1 + j1] = dmin;
			}      
		}  
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
/* the kmeans++ initialization */
void skmeanspp_init (float *X , int d , int N , int K , int *selected , float *disbest, float *distmp)
{
	int newsel;
	int i, j;
	double rd;
	float norm;

	for (i = 0 ; i < N ; i++)
	{
		disbest[i] = (float)HUGE_VAL;
	}

	/* select the first centroid and set the others unitialized*/

	selected[0] = (int)floor(rand()* K);

	for (i = 1 ; i < K ; i++) 
	{
		newsel = selected[i - 1];			
		scompute_distances_1(X + newsel*d , X , d , N , distmp);

		for(j = 0 ; j < N ; j++) 
		{
			if(distmp[j] < disbest[j]) 
			{
				disbest[j] = distmp[j];
			}
		}
		norm = 0.0f;
		for(j = 0 ; j < N ; j++)
		{
			distmp[j] = disbest[j];
			norm     += (float)fabs(distmp[j]);
		}
		norm = (norm != 0.0f) ? (1.0f/norm) : 1.0f;
		for(j = 0 ; j < N ; j++)
		{
			distmp[j] *= norm;
		}
		rd = rand();
		for (j = 0 ; j < N - 1 ; j++) 
		{
			rd -= distmp[j];
			if (rd < 0.0)
			{
				break;
			}
		}
		selected[i] = j;
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
/* the kmeans++ initialization */
void dkmeanspp_init (double *X , int d , int N , int K , int *selected , double *disbest, double *distmp)
{
	int newsel;
	int i, j;
	double rd;
	double norm;

	for (i = 0 ; i < N ; i++)
	{
		disbest[i] = HUGE_VAL;
	}

	/* select the first centroid and set the others unitialized*/

	selected[0] = (int)floor(rand()* K);

	for (i = 1 ; i < K ; i++) 
	{
		newsel = selected[i - 1];			
		dcompute_distances_1(X + newsel*d , X , d , N , distmp);

		for(j = 0 ; j < N ; j++) 
		{
			if(distmp[j] < disbest[j]) 
			{
				disbest[j] = distmp[j];
			}
		}
		norm = 0.0;
		for(j = 0 ; j < N ; j++)
		{
			distmp[j] = disbest[j];
			norm     +=  fabs(distmp[j]);
		}
		norm = (norm != 0.0) ? (1.0/norm) : 1.0;
		for(j = 0 ; j < N ; j++)
		{
			distmp[j] *= norm;
		}
		rd = rand();
		for (j = 0 ; j < N - 1 ; j++) 
		{
			rd -= distmp[j];
			if (rd < 0.0)
			{
				break;
			}
		}
		selected[i] = j;
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void randperm(int *selected , int N)
{
	int i,j,tmp;

	for(i = 0 ; i < N ; i++)
	{
		selected[i]      = i;				
	}
	for (i = N - 1 ; i >= 0; i--) 
	{
		j                = (int)floor((i + 1) * rand());  /* j is uniformly distributed on {0, 1, ..., i} */	
		tmp              = selected[j];
		selected[j]      = selected[i];
		selected[i]      = tmp;
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void randini(UL seed)
{
	/* SHR3 Seed initialization */

	if(seed == (UL)NULL)
	{
		jsrseed  = (UL) time( NULL );
		jsr     ^= jsrseed;
	}
	else
	{
		jsr     = (UL)NULL;
		jsrseed = seed;
		jsr    ^= jsrseed;
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void scompute_distances_1(float *a , float *b , int d , int N , float *dist2)
{
	int i, ione = 1;
	float minus_two = -2.0f, one = 1.0f;
	double sum_d2, sum_c2 = 0.0;
	float *dl;
#ifdef BLAS
	sum_c2  = BLASCALL(snrm2)(&d , a , &ione);
	sum_c2 *= sum_c2;
#else
	int j;
	for (j = 0 ; j < d ; j++)
	{
		sum_c2 += (a[j]*a[j]);
	}
#endif
	for (i = 0; i < N ; i++) 
	{
		dl     = b + i*d;
#ifdef BLAS
		sum_d2 = BLASCALL(snrm2)(&d , dl , &ione);
		sum_d2 *= sum_d2;
#else
		sum_d2 = 0.0;
		for (j = 0; j < d ; j++)
		{
			sum_d2 += (dl[j]*dl[j]);
		}
#endif
		dist2[i] = (float)(sum_d2 + sum_c2);
	}
	BLASCALL(sgemv)("Transposed", &d , &N , &minus_two , b , &d , a, &ione, &one , dist2, &ione);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dcompute_distances_1(double *a , double *b , int d , int N , double *dist2)
{
	int i, ione = 1;
	double minus_two = -2.0, one = 1.0;
	double sum_d2, sum_c2 = 0.0;
	double *dl;
#ifdef BLAS
	sum_c2  = BLASCALL(dnrm2)(&d , a , &ione);
	sum_c2 *= sum_c2;
#else
	int j;
	for (j = 0 ; j < d ; j++)
	{
		sum_c2 += (a[j]*a[j]);
	}
#endif
	for (i = 0; i < N ; i++) 
	{
		dl     = b + i*d;
#ifdef BLAS
		sum_d2 = BLASCALL(dnrm2)(&d , dl , &ione);
		sum_d2 *= sum_d2;
#else
		sum_d2 = 0.0;
		for (j = 0; j < d ; j++)
		{
			sum_d2 += (dl[j]*dl[j]);
		}
#endif
		dist2[i] = (sum_d2 + sum_c2);
	}
	BLASCALL(dgemv)("Transposed", &d , &N , &minus_two , b , &d , a, &ione, &one , dist2, &ione);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void scompute_cross_distances_nonpacked (float *a, float *b, int d, int na, int nb, float *dist2, float *sum_c2)
{
	int i, j;
	float *cl , *dl;
	float *d2l;
	float minus_two = -2.0f, one = 1.0f;
	double s_c2, sum_d2;
#ifdef BLAS
	int ione = 1;
#endif

	for (i = 0; i < na; i++) 
	{
		cl = a + d*i;
#ifdef BLAS
		s_c2  = BLASCALL(snrm2)(&d , cl , &ione);
		s_c2 *= s_c2;
#else
		s_c2  = 0.0;
		for (j = 0 ; j < d; j++)
		{
			s_c2 += (cl[j]*cl[j]);
		}
#endif
		sum_c2[i] = (float)s_c2;
	}
	for (i = 0 ; i < nb ; i++) 
	{
		dl     = b + d*i;
#ifdef BLAS
		sum_d2  = BLASCALL(snrm2)(&d , dl , &ione);
		sum_d2 *= sum_d2;
#else
		sum_d2 = 0.0;
		for (j = 0 ; j < d ; j++)
		{
			sum_d2 += (dl[j]*dl[j]);
		}
#endif
		d2l = dist2 + i*na;
		for (j = 0 ; j < na; j++)
		{
			d2l[j] = sum_d2 + sum_c2[j];
		}
	}
	BLASCALL(sgemm)("Transposed", "Not trans", &na , &nb , &d , &minus_two , a , &d , b , &d , &one , dist2 , &na);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dcompute_cross_distances_nonpacked (double *a, double *b, int d, int na, int nb, double *dist2, double *sum_c2)
{
	int i, j;
	double *cl , *dl;
	double *d2l;
	double minus_two = -2.0, one = 1.0;
	double s_c2, sum_d2;
#ifdef BLAS
	int ione = 1;
#endif

	for (i = 0; i < na; i++) 
	{
		cl = a + d*i;
#ifdef BLAS
		s_c2  = BLASCALL(dnrm2)(&d , cl , &ione);
		s_c2 *= s_c2;
#else
		s_c2  = 0.0;
		for (j = 0 ; j < d; j++)
		{
			s_c2 += (cl[j]*cl[j]);
		}
#endif
		sum_c2[i] = (double)s_c2;
	}
	for (i = 0 ; i < nb ; i++) 
	{
		dl     = b + d*i;
#ifdef BLAS
		sum_d2  = BLASCALL(dnrm2)(&d , dl , &ione);
		sum_d2 *= sum_d2;
#else
		sum_d2 = 0.0;
		for (j = 0 ; j < d ; j++)
		{
			sum_d2 += (dl[j]*dl[j]);
		}
#endif
		d2l = dist2 + i*na;
		for (j = 0 ; j < na; j++)
		{
			d2l[j] = sum_d2 + sum_c2[j];
		}
	}
	BLASCALL(dgemm)("Transposed", "Not trans", &na , &nb , &d , &minus_two , a , &d , b , &d , &one , dist2 , &na);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
