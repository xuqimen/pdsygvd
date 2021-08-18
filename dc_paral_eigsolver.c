#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "mpi.h"

#include "mkl.h"
#include <mkl_scalapack.h>
#include "mkl_lapacke.h"
#include <mkl_cblas.h>

#include <mkl_pblas.h>
//#include <mkl_scalapack.h>
#include <mkl_blacs.h>

extern void   pdlawrite_();
extern void   pdelset_();
extern double pdlamch_();
extern int    indxg2p_();
extern int    indxg2l_();
extern int    numroc_();
extern void   descinit_();
extern void   pdlaset_();
extern double pdlange_();
extern void   pdlacpy_();
extern int    indxg2p_();

extern void   pdgemr2d_();
extern void   pdgemm_();
extern void   pdsygvx_();
extern void   pdgesv_();
extern void   pdgesvd_();

extern void   pzgemr2d_();
extern void   pzgemm_();
extern void   pzhegvx_();

extern void   Cblacs_pinfo( int* mypnum, int* nprocs);
extern void   Cblacs_get( int context, int request, int* value);
extern int    Cblacs_gridinit( int* context, char * order, int np_row, int np_col);
extern void   Cblacs_gridinfo( int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);
extern void   Cblacs_gridexit( int context);
extern void   Cblacs_exit( int error_code);
extern void   Cblacs_gridmap (int *ConTxt, int *usermap, int ldup, int nprow0, int npcol0);
extern int    Csys2blacs_handle(MPI_Comm comm);
extern void   Cfree_blacs_system_handle(int handle);
#define max(x,y) (((x) > (y)) ? (x) : (y))
#define min(x,y) (((x) > (y)) ? (y) : (x))


// my own divide & conquer parallel eigensolver
void automem_pdsygvd_( 
	int *ibtype, char *jobz, char *uplo, int *n, double *a, int *ia,
	int *ja, int *desca, double *b, int *ib, int *jb, int *descb, 
	double *w, double *z, int *iz, int *jz, int *descz, int *info);

/**
 * @brief Call the pdsyevd_ routine with an automatic workspace setup.
 *
 *        The original pdsyevd_ routine asks uses to provide the size of the 
 *        workspace. This routine calls a workspace query and automatically sets
 *        up the memory for the workspace, and then call the pdsyevd_ routine to
 *        do the calculation.
 */
void automem_pdsyevd_ ( 
	char *jobz, char *uplo, int *n, double *a, int *ia, int *ja, int *desca, 
	double *w, double *z, int *iz, int *jz, int *descz, int *info)
{
	int ictxt = desca[1], nprow, npcol, myrow, mycol;
	Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

	int ZERO = 0, lwork, *iwork, liwork, *icluster;
	double *work;
	lwork = liwork = -1;
	work  = (double *)malloc(100 * sizeof(double));
	iwork = (int *)malloc(100 * sizeof(int)); 

	//** first do a workspace query **//
	pdsyevd_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, 
		work, &lwork, iwork, &liwork, info);

	int NNP, NN, NP0, MQ0, NB, N = *n;
	lwork = (int) fabs(work[0]);
	NB = desca[4]; // distribution block size
	NN = max(max(N, NB),2);
	NP0 = numroc_( &NN, &NB, &ZERO, &ZERO, &nprow );
	MQ0 = numroc_( &NN, &NB, &ZERO, &ZERO, &npcol );
	NNP = max(max(N,4), nprow * npcol+1);

	lwork = max(lwork, 5 * N + max(5 * NN, NP0 * MQ0 + 2 * NB * NB) 
				+ ((N - 1) / (nprow * npcol) + 1) * NN);
	lwork += max(N*N, min(10*lwork,2000000)); // for safety
	work = realloc(work, lwork * sizeof(double));

	liwork = iwork[0];
	liwork = max(liwork, 6 * NNP);
	liwork += max(N*N, min(20*liwork, 200000)); // for safety
	iwork = realloc(iwork, liwork * sizeof(int));

	// call the routine again to perform the calculation
	pdsyevd_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, 
		work, &lwork, iwork, &liwork, info);

	free(work);
	free(iwork);
}


/**
 * @brief A parallel dsygvd routine with an automatic workspace setup.
 *
 *        The only parallel generalized eigensolver in ScaLAPACK is pdsygvx_ 
 *        which uses the standard eigen-problem routine pdsyev. This routine 
 *        calls a workspace query and automatically sets up the memory for 
 *        the workspace.
 */
void automem_pdsygvd_( 
	int *ibtype, char *jobz, char *uplo, int *n, double *a, int *ia,
	int *ja, int *desca, double *b, int *ib, int *jb, int *descb, 
	double *w, double *z, int *iz, int *jz, int *descz, int *info)
{
	int ictxt = desca[1], nprow, npcol, myrow, mycol;
	Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
	int ZERO = 0, NN, NP0, MQ0, NB, N = *n;
	NB = desca[4]; // distribution block size
	NN = max(max(N, NB),2);
	NP0 = numroc_( &NN, &NB, &ZERO, &ZERO, &nprow );
	MQ0 = numroc_( &NN, &NB, &ZERO, &ZERO, &npcol );

	// Cholesky factorization of b = L*L' (uplo = 'L') = R'*R (uplo = 'U')
	pdpotrf_(uplo, n, b, ib, jb, descb, info);
	assert(*info == 0);

	// Transform problem to standard eigenvalue problem
	double scale = 1.0; // need to scale the final eigenvalue by scale
	int lwork = -1;
	double *work = (double *)malloc(100 * sizeof(double));
	pdsyngst_(ibtype, uplo, n, a, ia, ja, desca, b, ib, jb, 
		descb, &scale, work, &lwork, info); // query workspace
	lwork = (int) fabs(work[0]);
	lwork += max(NB * (NP0+1), 3 * NB);
	lwork += 2 * NP0 * NB + MQ0 * NB + NB * NB;
	lwork += 10 * N + N * N; // for safety
	work = realloc(work, lwork * sizeof(double));
	pdsyngst_(ibtype, uplo, n, a, ia, ja, desca, b, ib, jb, 
		descb, &scale, work, &lwork, info); // perform calculation

	// solve standard eigenproblem using D&C algorithm
	automem_pdsyevd_(jobz, uplo, n, a, ia, ja, desca, 
		w, z, iz, jz, descz, info);
	// automem_pdsyev_(jobz, uplo, n, a, ia, ja, desca, 
	//     w, z, iz, jz, descz, info);

	// back transform eigenvectors to the original problem
	int wantz = lsame(jobz, "V", 1, 1);
	int upper = lsame(uplo, "U", 1, 1);

	//printf("wantz = %d\n", wantz);
	//printf("upper = %d\n", upper);
	if (wantz) {
		int neig = *n;
		double alpha = 1.0;
		if (*ibtype == 1 || *ibtype == 2) {
			// For sub( A )*x=(lambda)*sub( B )*x and
			//     sub( A )*sub( B )*x=(lambda)*x; 
			// backtransform eigenvectors:
			//     x = inv(L)'*y or inv(U)*y
			//printf("using pdtrsm_ for backtransforming eigenvectors\n");
			char trans = upper ? 'N' : 'T';
			// pdtrsm_("L", uplo, &trans, "N", n, &neig, &alpha,
			//     b, ib, jb, descb, z, iz, jz, descz);
			pdtrsm_("L", uplo, "N", "N", n, &neig, &alpha,
				b, ib, jb, descb, z, iz, jz, descz);
		} else if (*ibtype == 3) {
			// For sub( B )*sub( A )*x=(lambda)*x;
			// backtransform eigenvectors: x = L*y or U'*y
			//printf("using pdtrmm_ for backtransforming eigenvectors\n");
			char trans = upper ? 'T' : 'N';
			pdtrmm_("L", uplo, &trans, "N", n, &neig, &alpha,
				b, ib, jb, descb, z, iz, jz, descz);
		}
	}

	if ( fabs(scale-1.0) > 1e-14 )
		cblas_dscal(*n, scale, w, 1);

	free(work);
}

