#### Description
The only available parallel generalized eigensolver in ScaLAPACK is the pdsygvx routine (which uses the implicit QL/QR method). This repo includes an implementation of the parallel divide-and-conquer generalized eigensolver (pdsygvd). The usage of this routine is similar to [pdsygvx](https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/scalapack-routines/scalapack-driver-routines/p-sygvx.html), except it always calculates all the eigenvalues and eigenvectors.

#### Run a test
Compile the code by 
```
make clean; make
```
Run a test by
```
mpirun -np <np> ./test_eig <matrix size> <np1> <np2>
```
where np is the number of processes, np1 x np2 = np. For example, test a 8 x 8 matrix with 4 processes using a 2 x 2 process grid as follows
```
mpirun -np 4 ./test_eig 8 2 2
```


#### Caution
This routine is not fully tested, use with caution!
