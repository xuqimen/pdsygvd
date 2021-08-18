#ifndef DC_PARAL_EIGSOLVER_H
#define DC_PARAL_EIGSOLVER_H


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
	double *w, double *z, int *iz, int *jz, int *descz, int *info);


#endif // DC_PARAL_EIGSOLVER_H

