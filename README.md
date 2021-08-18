#### Description
The only available parallel generalized eigensolver in ScaLAPACK is the pdsygvx routine (which uses the implicit QL/QR method). This repo includes an implementation of the parallel divide-and-conquer generalized eigensolver (pdsygvd). The usage of this routine is similar to [pdsygvx](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/scalapack-routines/scalapack-driver-routines/p-sygvx.html), except it always calculates all the eigenvalues and eigenvectors.

#### Caution
This routine is not fully tested, use with caution!
