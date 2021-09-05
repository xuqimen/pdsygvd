#include <time.h> // for clock_gettime()
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

#include "dc_paral_eigsolver.h"


// TO PRINT IN COLOR
#define RED   "\x1B[31m"
#define GRN   "\x1B[32m"
#define YEL   "\x1B[33m"
#define BLU   "\x1B[34m"
#define MAG   "\x1B[35m"
#define CYN   "\x1B[36m"
#define WHT   "\x1B[37m"
#define RESET "\x1B[0m"

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


int main(int argc, char **argv) {
	int i, j, k, n, p;
	double t1, t2, ts, te;	
	
	/************  MPI ***************************/
	int myrank_mpi, nprocs_mpi, rank, nproc;
	MPI_Init( &argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	
	/************  BLACS ***************************/
	int ictxt_0, ictxt_1, ictxt_2, dims[2], nprow, npcol, myrow, mycol, nbr, nbc;
	int info,llda;
	int ZERO=0,ONE=1;
	
	// ** Matrix size ** //
	int M, N;
	// Global matrix size
	// M = 439; // number of rows of the global matrix
	// N = 439; // number of columns of the global matrix
	M = 1000; // number of rows of the global matrix
	N = 1000; // number of columns of the global matrix
	
	// ** Process grid dimensions ** //
	// dims[0] = 40; dims[1] = 25;
	dims[0] = nproc; dims[1] = 1;
	if (argc == 4) {
        M = N = atoi(argv[1]);
		dims[0] = atoi(argv[2]);
		dims[1] = atoi(argv[3]);
	}
	
	if (!rank && dims[0] * dims[1] > nproc) {
		printf("Error: dimensions of process grid is greater than the total number of processors!\n");
		exit(1);
	}
	
	// ** Initialize MPI for ScaLAPACK ** //
	Cblacs_pinfo( &myrank_mpi, &nprocs_mpi );
	
	// ** Create a context with ring topology ** //
	Cblacs_get( -1, 0, &ictxt_0 ); 
	Cblacs_gridinit( &ictxt_0, "Row", 1, nprocs_mpi );
	
	// ** Create a new (grid) context ** //
	Cblacs_get( -1, 0, &ictxt_1 ); 
	Cblacs_gridinit( &ictxt_1, "Row", dims[0], dims[1] );
	
	// ** get coordinate of each processor in the process grid ** //
	Cblacs_gridinfo( ictxt_1, &nprow, &npcol, &myrow, &mycol );
	

	
	/***************************************************
	 *            Set up distributed matrix            *
	 ***************************************************/
	int descA[9], MB, NB, M_loc, N_loc;
	//	MB = max(1, M / dims[0]);
	//	NB = max(1, N / dims[1]);
	MB = NB = 64;
	M_loc = numroc_( &M, &MB, &myrow, &ZERO, &nprow );
	N_loc = numroc_( &N, &NB, &mycol, &ZERO, &npcol );
	llda = max(1, M_loc);
	descinit_(descA, &M, &N, &MB, &NB, &ZERO, &ZERO, &ictxt_1, &llda, &info);

	if (!rank) {
		printf(GRN"\n"
				  "**********************************\n"
				  "*        Process Grid Info       *\n"
				  "**********************************\n"
				  "Number of processes: %4d\n"
				  "2-D Process grid dims: %3d x %3d\n"
				  "Matrix size: %d x %d\n"
				  "Local Matrix size: %d x %d\n"
				  "Start calculation ...\n\n"RESET,
				  nproc, nprow, npcol, M, N, M_loc, N_loc);
	}

	/***************************************************
	 *             Set up local data                   *
	 ***************************************************/
	double *A, *B;
	//A = (double *)malloc( sizeof(double) * M_loc * N_loc);
	A = (double *)mkl_malloc( sizeof(double) * max(1,M_loc * N_loc), 64);
	B = (double *)mkl_malloc( sizeof(double) * max(1,M_loc * N_loc), 64);
	if (A == NULL || B == NULL) {
		printf("\nMemory allocation for A failed!\n");
		exit(1);
	}
	
	srand(1+rank+(int)time(NULL));
	for (j = 0; j < N_loc; j++) {
		for (i = 0; i < M_loc; i++) {
			A[j*M_loc+i] = 0.0 + (1.0 - 0.0) * (double) rand() / RAND_MAX;
			B[j*M_loc+i] = 0.0 + (1.0 - 0.0) * (double) rand() / RAND_MAX;
		}
	}

	// ** Print out matrix A ** //
	// for (k = 0; k < nproc; k++) {
	//     MPI_Barrier(MPI_COMM_WORLD);
	//     if (k == rank) {
	//         printf("\n-------------------------------------\n");
	//         printf(  "rank = %3d = (%d, %d)\n", rank, myrow, mycol);
	//         printf(  "Local part of A:\n");
	//         for (i = 0; i < M_loc; i++) {
	//             for (j = 0; j < N_loc; j++) {
	//                 printf("%10.6f ", A[j*M_loc+i]);
	//             }
	//             printf("\n");
	//         }
	//     }
	// }
	
	
	
	/***************************************************
	 *              Set up result matrix               *
	 ***************************************************/
	double *C, *D, *E;
	int MB_C, NB_C, N_loc_C, M_loc_C, descC[9];
	
	// block size for C
	//	MB_C = max(1, N / dims[0]);
	//	NB_C = max(1, N / dims[1]);
	MB_C = NB_C = 64;
	
	// find local matrix sizes
	M_loc_C = numroc_( &N, &MB_C, &myrow, &ZERO, &nprow );
	N_loc_C = numroc_( &N, &NB_C, &mycol, &ZERO, &npcol );
	
	llda = max(1, M_loc_C);
	descinit_(descC, &N, &N, &MB_C, &NB_C, &ZERO, &ZERO, &ictxt_1, &llda, &info);
	
	// allocate memory for C
	C = (double *)mkl_malloc( sizeof(double) * max(M_loc_C * N_loc_C,1), 64);
	D = (double *)mkl_malloc( sizeof(double) * max(M_loc_C * N_loc_C,1), 64);
	E = (double *)mkl_malloc( sizeof(double) * max(M_loc_C * N_loc_C,1), 64);
	if (C == NULL) {
		printf("\nrank = %d = (%d, %d), M_loc_C = %d, N_loc_C = %d, Memory allocation for C failed!\n",
				rank, myrow, mycol, M_loc_C, N_loc_C);
		exit(1);
	}
	
	// make a copy
	double *C2 = (double *)mkl_malloc( sizeof(double) * max(M_loc_C * N_loc_C,1), 64);
	double *D2 = (double *)mkl_malloc( sizeof(double) * max(M_loc_C * N_loc_C,1), 64);
	double *E2 = (double *)mkl_malloc( sizeof(double) * max(M_loc_C * N_loc_C,1), 64);
	// make another copy
	double *C3 = (double *)mkl_malloc( sizeof(double) * max(M_loc_C * N_loc_C,1), 64);
	double *D3 = (double *)mkl_malloc( sizeof(double) * max(M_loc_C * N_loc_C,1), 64);
	double *E3 = (double *)mkl_malloc( sizeof(double) * max(M_loc_C * N_loc_C,1), 64);

	/***************************************************
	 *          Perform matrix multiplication          *
	 ***************************************************/ 
	double alpha = 1.0, beta = 0.0;
	
	t1 = MPI_Wtime();
	
	if (!rank) printf("Start matrix multiplication ...\n");

	// matrix multiplication
	pdgemm_("T", "N", &N, &N, &M, &alpha, A, &ONE, &ONE, descA,
			A, &ONE, &ONE, descA, &beta, C, &ONE, &ONE, descC);
	
	pdgemm_("T", "N", &N, &N, &M, &alpha, B, &ONE, &ONE, descA,
			B, &ONE, &ONE, descA, &beta, D, &ONE, &ONE, descC);
	
	t2 = MPI_Wtime();

	// create a copy of C and D
	for (int i = 0; i < M_loc_C * N_loc_C; i++) {
		C3[i] = C2[i] = C[i];
	}
	for (int i = 0; i < M_loc_C * N_loc_C; i++) {
		D3[i] = D2[i] = D[i];
	}

	// ** Print out result matrix ** //
	//   for (k = 0; k < nproc; k++) {
	//     MPI_Barrier(MPI_COMM_WORLD);
	//     if (k == rank) {
	//         printf("\n-------------------------------------\n");
	//         printf(  "rank = %3d = (%d, %d)\n", rank, myrow, mycol);
	//        printf(  "Local part of C:\n");
	//        for (i = 0; i < M_loc_C; i++) {
	//            for (j = 0; j < N_loc_C; j++) {
	//                printf("%10.6f ", C[j*M_loc_C+i]);
	//            }
	//            printf("\n");
	//        }
	//        printf(  "Local part of D:\n");
	//        for (i = 0; i < M_loc_C; i++) {
	//            for (j = 0; j < N_loc_C; j++) {
	//                printf("%10.6f ", D[j*M_loc_C+i]);
	//            }
	//            printf("\n");
	//        }
	//     }
	// }

	if (!rank) 
	printf(GRN"\n""Multiplying matrix took %.3f ms\n"RESET, (t2-t1)*1e3);
	
	double *lambda = (double *)malloc(N * sizeof(double));
	
	
	// ** Start solving generalized eigenvalue problem ** //
	int il = 1, iu = 1, lwork, *iwork, liwork, *ifail, *icluster;
	double *work, *gap, vl = 0.0, vu = 0.0, abstol, orfac = 0.001; 
    orfac = 0.0; // do not reorthogonalize eigenvectors
	// orfac = 1e-6; // reorthogonalize the corresponding eigvecs if two eigvals differ by < orfac
    t1 = MPI_Wtime();
	// this setting yields the most orthogonal eigenvectors
	abstol = pdlamch_(&ictxt_1, "U");
	//abstol = 2*pdlamch_(&ictxt_1, "S");
	t2 = MPI_Wtime();
	if(!rank) printf("rank = %d, abstol = %.3e, calculating abstol (for most orthogonal eigenvectors) took %.3f ms\n",rank, abstol, (t2-t1)*1e3);
	
	//** first do a workspace query **//
	lwork = liwork = -1;
	work  = (double *)malloc(100 * sizeof(double));
	gap   = (double *)malloc(nprow * npcol * sizeof(double));
	iwork = (int *)malloc(100 * sizeof(int));
	ifail = (int *)malloc(N * sizeof(int));
	icluster = (int *)malloc(2 * nprow * npcol * sizeof(int));
	

	int NZ;  
			
	NZ = N;
	// if (!rank) printf("N = %d\n",N);
	
	t1 = MPI_Wtime();
	pdsygvx_(&ONE, "V", "A", "U", &N, C, &ONE, &ONE, descC, 
			 D, &ONE, &ONE, descC, &vl, &vu, &il, &iu, &abstol, 
			 &NZ, &NZ, lambda, &orfac, 
			 E, &ONE, &ONE, descC, 
			 work, &lwork, iwork, &liwork, ifail, icluster, gap, &info);
	t2 = MPI_Wtime();
	if (!rank) printf("rank = %d, info = %d, work(1) = %f, iwork(1) = %d, time for workspace inquery: %.3f ms\n", rank, info, work[0], iwork[0], (t2 - t1)*1e3);
	
	
	
	lwork = (int) fabs(work[0]);
	// lwork += max(N*N,2000000);
	work = realloc(work, lwork * sizeof(double));
	
	liwork = iwork[0];
	// liwork += max(N*N,2000000);
	iwork = realloc(iwork, liwork * sizeof(int));
	
	NZ = N;
	// printf("2N = %d\n",N);
	
	t1 = MPI_Wtime();
	pdsygvx_(&ONE, "V", "A", "U", &N, C, &ONE, &ONE, descC, 
			 D, &ONE, &ONE, descC, &vl, &vu, &il, &iu, &abstol, 
			 &NZ, &NZ, lambda, &orfac, 
			 E, &ONE, &ONE, descC, 
			 work, &lwork, iwork, &liwork, ifail, icluster, gap, &info);
	t2 = MPI_Wtime();
	if (!rank) printf("rank = %d, info = %d, time for QR eigensolver: %.3f ms\n", 
			rank, info, (t2 - t1)*1e3);
	
	// if (rank == 0) {
	// 	printf("lambda = (N = %d) \n",N);
	// 	for (i = 0; i < 10; i++) {
	// 		printf("%.12f\n",lambda[i]);
	// 	}
	// }
	

	// ** test pdsygvd ** //
	double *lambda2 = (double *)malloc(N * sizeof(double));
	t1 = MPI_Wtime();
	automem_pdsygvd_( 
		&ONE, "V", "U", &N, C2, &ONE, &ONE, descC, D2, &ONE, &ONE, descC, 
		lambda2, E2, &ONE, &ONE, descC, &info);
	assert(info == 0);
	MPI_Bcast(lambda2, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	t2 = MPI_Wtime();
	if (!rank) printf("rank = %d, info = %d, time for pdsygvd (D&C eigensolver): %.3f ms\n", 
			rank, info, (t2 - t1)*1e3);

	// check results
	double eigvaldiff_gvd = 0.0;
	for (int i = 0; i < N; i++) {
		eigvaldiff_gvd = max(fabs(lambda[i] - lambda2[i]), eigvaldiff_gvd);
	}
	MPI_Allreduce(MPI_IN_PLACE, &eigvaldiff_gvd, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	if (rank == 0) {
		printf("err in eigvals (pdsygvd, D&C): %.3e\n", eigvaldiff_gvd);
	}
	// check eigenvectors
	double eigvecdiff_gvd_local = 0.0;
	for (int i = 0; i < M_loc_C * N_loc_C; i++) {
		eigvecdiff_gvd_local = max(fabs(fabs(E[i]) - fabs(E2[i])), eigvecdiff_gvd_local);
	}
	double eigvecdiff_gvd = 0.0;
	MPI_Allreduce(&eigvecdiff_gvd_local, &eigvecdiff_gvd, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	if (rank == 0) {
		printf("err in eigvecs (pdsygvd, D&C): %.3e\n", eigvecdiff_gvd);
	}


#define TEST_PDSYGV
#ifdef TEST_PDSYGV
	// ** test pdsygv ** //
	double *lambda3 = (double *)malloc(N * sizeof(double));
	t1 = MPI_Wtime();
	automem_pdsygv_( 
		&ONE, "V", "U", &N, C3, &ONE, &ONE, descC, D3, &ONE, &ONE, descC, 
		lambda3, E3, &ONE, &ONE, descC, &info);
	assert(info == 0);
	t2 = MPI_Wtime();
	if (!rank) printf("rank = %d, info = %d, time for pdsygv (non-expert QR eigensolver): %.3f ms\n", 
			rank, info, (t2 - t1)*1e3);

	double eigvaldiff_gv = 0.0;
	for (int i = 0; i < N; i++) {
		eigvaldiff_gv = max(fabs(lambda[i] - lambda3[i]), eigvaldiff_gv);
	}
	MPI_Allreduce(MPI_IN_PLACE, &eigvaldiff_gv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("err in eigvals (pdsygv, non-expert QR): %.3e\n", eigvaldiff_gv);
	}
	// check eigenvectors
	double eigvecdiff_gv_local = 0.0;
	for (int i = 0; i < M_loc_C * N_loc_C; i++) {
		eigvecdiff_gv_local = max(fabs(fabs(E[i]) - fabs(E3[i])), eigvecdiff_gv_local);
	}
	double eigvecdiff_gv = 0.0;
	MPI_Allreduce(&eigvecdiff_gv_local, &eigvecdiff_gv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	if (rank == 0) {
		printf("err in eigvecs (pdsygv, non-expert QR): %.3e\n", eigvecdiff_gv);
	}
	free(lambda3);
#endif // #ifdef TEST_PDSYGV



	// ** print out eigenvalues ** //
	// for (int i = 0; i < N; i++) {
	// 	//if ((!rank) && (fabs(lambda[i] - lambda2[i]) > fabs(lambda[i])*1e-5))
	// 		printf("lambda[%d] = %20.16f, lambda2[%d] = %20.16f, diff = %10.3e\n",
	// 				i, lambda[i], i, lambda2[i], lambda[i]-lambda2[i]);
	// }

	// if (rank == 0) {
	// 	printf("lambda2 = (N = %d) \n",N);
	// 	for (i = 0; i < 10; i++) {
	// 		printf("%.12f\n",lambda2[i]);
	// 	}
	// }
  


	// ** Print out result matrix ** //
	// for (k = 1; k < 2; k++) {
	// 	MPI_Barrier(MPI_COMM_WORLD);
	// 	if (k == rank) {
	// 		printf("\n-------------------------------------\n");
	// 		printf(  "rank = %3d = (%d, %d)\n", rank, myrow, mycol);
	// 		printf(  "Local part of E:\n");
	// 		for (i = 0; i < min(M_loc_C,10); i++) {
	// 			for (j = 0; j < min(N_loc_C,10); j++) {
	// 				printf("%10.6f ", E[j*M_loc_C+i]);
	// 			}
	// 			printf("\n");
	// 		}
	// 		printf(  "Local part of E2:\n");
	// 		for (i = 0; i < min(M_loc_C,10); i++) {
	// 			for (j = 0; j < min(N_loc_C,10); j++) {
	// 				printf("%10.6f ", E2[j*M_loc_C+i]);
	// 			}
	// 			printf("\n");
	// 		}

	// 	}
	// }

	free(lambda);
	free(lambda2);
	free(work);
	free(gap);
	free(iwork);
	free(ifail);
	free(icluster);
	
	mkl_free(A);
	mkl_free(B);
	mkl_free(C);
	mkl_free(D);
	mkl_free(E);
	mkl_free(C2);
	mkl_free(D2);
	mkl_free(E2);
	mkl_free(C3);
	mkl_free(D3);
	mkl_free(E3);

	Cblacs_gridexit( ictxt_1 ); // release grid/context with handle ictxt_1
	Cblacs_gridexit( ictxt_0 ); // release grid/context with handle ictxt_0
	MPI_Finalize();
	return 0;	
}

