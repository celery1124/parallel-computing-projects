#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;

// matrix multiplication
// C = A * B
void Mmult(double **A, int rA, int cA, double **B, int rB, int cB, double **C, int rC, int cC) {
    assert(cA == rB);
    assert(rC == rA && cC == cB);
    #pragma omp parallel for shared (rA, cA, cB, A, B, C) collapse(2) schedule(static) if (rA*cB > 64)  
    for (int r = 0; r < rA; r++) {
        for (int c = 0; c < cB; c++) {
            double sum = 0;
            for (int i = 0; i < cA; i++) {
                sum += A[r][i] * B[i][c];
            }
            C[r][c] = sum;
        }
    }
}

// forward substitution 
// y = L\f
void ForwardSubstitution(double **L, int nL, double *f, int nf, double *y, int ny) {
    assert(nL == nf && nf == ny);
    for (int i = 0; i < nL; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (f[i] - sum) / L[i][i];
    }
}

// backward substitution 
// y = U\f
void BackSubstitution(double **U, int nU, double *f, int nf, double *y, int ny) {
    assert(nU == nf && nf == ny);
    for (int i = nU - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < nU; j++) {
            sum += U[i][j] * y[j];
        }
        y[i] = (f[i] - sum) / U[i][i];
    }
}

// upper triangular matrix transpose
// B = A^T (A is upper triangular matrix)
// B should be initialized as zeroed matrix
void UTMTranspose(double **A, int rA, int cA, double **B, int rB, int cB) {
    assert(rA == cB && cA == rB);
    for (int i = 0; i < rA; i++) {
        for (int j = i; j < cA; j++) {
            B[j][i] = A[i][j];
        }
    }
}

// U^T*U = K
void CholeskyFactorization(double **K, double **U, int n) 
{ 
    // caller make sure Matrix U is zeroed (through calloc)
  
    // Cholesky-Crout Algorithm
    for (int j = 0; j < n; j++) { 
        double sum = 0; 
        // Diagnols 
        for (int k = 0; k < j; k++) sum += (U[k][j] * U[k][j]);
        U[j][j] = sqrt(K[j][j] - sum); 

        #pragma omp parallel for private(sum) shared(n, j, K, U) schedule(static) if (j < n-64)   
        for (int i = j+1; i < n; i++) { 
            sum = 0;
            for (int k = 0; k < j; k++) sum += (U[k][i] * U[k][j]);
            U[j][i] = (K[i][j] - sum) / U[j][j]; 
        } 
    } 
}

double **MatrixAlloc(int row, int col) {
    // Make the matrix in a continues memory chunk
    double **A = new double*[row];
    double *mem_chunk = (double *) malloc(sizeof(double) * row * col);
    for (int i = 0; i < row; i++) A[i] = mem_chunk + i*col;
    return A;
}

void MatrixFree(double **A) {
    free(A[0]);
    delete [] A;
}

double **ZeroMatrixAlloc(int row, int col) {
    // Make the matrix in a continues memory chunk
    double **A = new double*[row];
    double *zeroed_chunk = (double *) calloc(sizeof(double), row * col);
    for (int i = 0; i < row; i++) A[i] = zeroed_chunk + i*col;
    return A;
}

void PrintErrMatrix(double **A, int rA, int cA) {
    for(int i = 0; i < rA; i++) {
        for (int j = 0; j < cA; j++) {
            fprintf(stderr, "%.5f\t", A[i][j]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

double GPR(int m, double rstar_l, double rstar_r) {
    
    auto wcts = chrono::system_clock::now();
    chrono::duration<double, milli> wctduration;

    // Initialize m x m grid of points
    int n = m*m;
    double h = (double)1/(m+1);
    double *XY_1 = (double *)calloc(sizeof(double), n);
    double *XY_2 = (double *)calloc(sizeof(double), n);

    #pragma omp parallel for shared (m, h, XY_1, XY_2) collapse(2) schedule(static) if (m > 64)  
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            XY_1[j+i*m] = (i+1)*h;
            XY_2[j+i*m] = (j+1)*h;
        }
    }

    wctduration = (chrono::system_clock::now() - wcts);
    printf("Initialize m x m grid %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();
    
    // Initialize observed data vector f
    double **f = MatrixAlloc(n, 1);
    int seed = omp_get_thread_num() ;
    struct drand48_data drand_buf;
    
    // #pragma omp parallel for shared (n, f, XY_1, XY_2) schedule(static) if (n > 8192)  
    for (int i = 0; i < n; i++) {
        // int seed = omp_get_thread_num() ;
        // struct drand48_data drand_buf;
        srand48_r (seed, &drand_buf);
        drand48_r (&drand_buf, &(f[i][0]));
        f[i][0] = (f[i][0] - 0.5) * 0.1 + 1 - ((XY_1[i]-0.5) * (XY_1[i]-0.5) + (XY_2[i]-0.5) * (XY_2[i]-0.5));
    }

    wctduration = (chrono::system_clock::now() - wcts);
    printf("Initialize observed data f %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();

#ifdef DBG
    PrintErrMatrix(f, n, 1);
#endif

    // Initialize k
    double **k = MatrixAlloc(1, n);
    #pragma omp parallel for shared (n, f, XY_1, XY_2) schedule(static) if (n >= 8192) 
    for (int i = 0; i < n; i++) {
        double d1 = rstar_l - XY_1[i];
        double d2 = rstar_r - XY_2[i];
        k[0][i] = exp(0 - d1 * d1 - d2 * d2);
    }

    wctduration = (chrono::system_clock::now() - wcts);
    printf("Initialize k %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();

    // Initialize tI + K
    double **K = MatrixAlloc(n, n);
    #pragma omp parallel for shared (n, K, XY_1, XY_2) collapse(2) schedule(static) if (n > 64)   
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double d1 = XY_1[i] - XY_1[j];
            double d2 = XY_2[i] - XY_2[j];
            K[i][j] = exp(0 - d1 * d1 - d2 * d2);
            if (i==j) {
                K[i][j] += 0.01;
            }
        }
    }

    wctduration = (chrono::system_clock::now() - wcts);
    printf("Initialize K %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();

    // Compute LU factorization of tI + K
    double **L = ZeroMatrixAlloc(n, n);
    double **U = ZeroMatrixAlloc(n, n);
    CholeskyFactorization(K, U, n);

    wctduration = (chrono::system_clock::now() - wcts);
    printf("LU factorization %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();

    double **tmp1 = MatrixAlloc(n, n);
    double **tmp2 = MatrixAlloc(n, 1);
    double **fstar = MatrixAlloc(1, 1);

    UTMTranspose(U, n, n, L, n, n); // inv(U) = inv(L)^T
    ForwardSubstitution(L, n, &f[0][0], n, &tmp1[0][0], n);
    BackSubstitution(U, n, &tmp1[0][0], n, &tmp2[0][0], n);


    Mmult(k, 1, n, tmp2, n, 1, fstar, 1, 1);
    double ret = fstar[0][0];

    double **t1 = MatrixAlloc(n, 1);
    double **t2 = MatrixAlloc(n, 1);

    wctduration = (chrono::system_clock::now() - wcts);
    printf("Compute predicted value fstar %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();

    // Free allocated Matrix
    MatrixFree(f);
    MatrixFree(k);
    MatrixFree(L);
    MatrixFree(U);
    MatrixFree(K);
    MatrixFree(tmp1);
    MatrixFree(tmp2);
    MatrixFree(fstar);

    wctduration = (chrono::system_clock::now() - wcts);
    printf("Free up %.3f ms\n", wctduration.count());

    return ret;
}

int main (int argc, char *argv[]) {
    int m;
    double rstar_l, rstar_r;
    if (argc < 4) { 
	    printf("Usage: %s m rstar_l rstar_r\nSample use: \n\t%s 16 0.5 0.5\n", argv[0], argv[0]);
	    exit(0);
	} else {
	    m = atoi(argv[1]); 
	    rstar_l = atof(argv[2]); 
	    rstar_r = atof(argv[3]); 
	}

    // time GPR
	auto wcts = chrono::system_clock::now();
    
    double fstar = GPR(m, rstar_l, rstar_r);
    
    chrono::duration<double, milli> wctduration = (chrono::system_clock::now() - wcts);
    printf("Finished in %.2f miliseconds [Wall Clock]\n", wctduration.count());

    printf("fstar %.6f\n", fstar);
    return 0;
}
