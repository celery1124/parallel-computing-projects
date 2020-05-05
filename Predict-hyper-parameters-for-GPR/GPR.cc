#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <limits>
#include <cmath>
#include <cstring>
#include <vector>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;

#define PI 3.14159265358979323846

// matrix multiplication
// C = A * B
void Mmult(double **A, int rA, int cA, double **B, int rB, int cB, double **C, int rC, int cC) {
    assert(cA == rB);
    assert(rC == rA && cC == cB);
    //#pragma omp parallel for shared (rA, cA, cB, A, B, C) collapse(2) schedule(static) if (rA*cB > 64)  
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
void ForwardSubstitution(double **U, int nL, double *f, int nf, double *y, int ny) {
    // L[i][j] = U[j][i]
    assert(nL == nf && nf == ny);
    for (int i = 0; i < nL; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += U[j][i] * y[j];
        }
        y[i] = (f[i] - sum) / U[i][i];
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

// U^T*U = K
void CholeskyFactorization(double **K, double **U, int n, double t) 
{ 
    // caller make sure Matrix U is zeroed (through calloc)
  
    // Cholesky-Crout Algorithm
    for (int j = 0; j < n; j++) { 
        double sum = 0; 
        // Diagnols 
        for (int k = 0; k < j; k++) sum += (U[k][j] * U[k][j]);
        U[j][j] = sqrt(K[j][j] + t - sum); 

#ifdef OPT1
        #pragma omp parallel for private(sum) shared(n, j, K, U) schedule(static) if (j < n-64) 
#endif
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

void PrintOutMatrix(double **A, int rA, int cA) {
    for(int i = 0; i < rA; i++) {
        for (int j = 0; j < cA; j++) {
            fprintf(stdout, "%.4f\t", A[i][j]);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}
void PrintErrMatrix(double **A, int rA, int cA) {
    for(int i = 0; i < rA; i++) {
        for (int j = 0; j < cA; j++) {
            fprintf(stderr, "%.6f\t", A[i][j]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}
// reload
void PrintErrVector(int *A, int nA) {
    for (int j = 0; j < nA; j++) {
        fprintf(stderr, "%d\t", A[j]);
    }
    fprintf(stderr, "\n");
}

// Kernel function
void kernel(double *x1, double *x2, int x_size, double *y1, double *y2, int y_size, double l1, double l2, double **res) {
    // caller needs to allocate/free res -> MatrixAlloc(x_size, y_size);
    double coeff = 1/sqrt(2*PI);
    for (int i = 0; i < x_size; i++) {
        for (int j = 0; j < y_size; j++) {
            res[i][j] = coeff * exp(-0.5*(pow(x1[i]-y1[j], 2)/pow(l1,2) + pow(x2[i]-y2[j], 2)/pow(l2,2)));
        }
    }
}

// Extrace submatrix from row column vector
void ExtractFromMatrix(double **X, int *r, int *c, double **Y, int rY, int cY) {
    // nr == rY && nc = cY
    for (int i = 0; i < rY; i++) {
        for (int j = 0; j < cY; j++) {
            Y[i][j] = X[r[i]][c[j]];
        }
    }
}

// Extrace subvector from row vector
void ExtractFromMatrix(double **X, int *r, double **Y, int rY) {
    // nr == rY Extract row only
    for (int i = 0; i < rY; i++) {
        Y[i][0] = X[r[i]][0];
    }
}

// Compute predictions at test points using training points and observed 
// values at those training poionts as input to the model with 
// hyperparameters t and l. 
double **GPR(double *XY1, double *XY2, int n, double **f, int *itest, int ntest, \
        int *itrain, int ntrain, double t, double l1, double l2) {
    // caller needs to free return matrix -> MatrixFree(ret);
#ifdef TIME
    auto wcts = chrono::system_clock::now();
    chrono::duration<double, milli> wctduration;
#endif

    // Initialize K for all points (including test and training points)
    double **K0 = MatrixAlloc(n, n);
    kernel (XY1, XY2, n, XY1, XY2, n, l1, l2, K0);

    // Extract training set K
    double **K = MatrixAlloc(ntrain, ntrain);
    ExtractFromMatrix(K0, itrain, itrain, K, ntrain, ntrain);

#ifdef TIME
    wctduration = (chrono::system_clock::now() - wcts);
    printf("Initialize K %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();
#endif

    // Initialize k
    double **k = MatrixAlloc(ntest, ntrain);
    ExtractFromMatrix(K0, itest, itrain, k, ntest, ntrain);

    // Initialize f
    double **f_train = MatrixAlloc(ntrain, 1);
    ExtractFromMatrix(f, itrain, f_train, ntrain);

    // Compute LU factorization of tI + K
    double **U = ZeroMatrixAlloc(ntrain, ntrain);
    CholeskyFactorization(K, U, ntrain, t);
#ifdef TIME
    wctduration = (chrono::system_clock::now() - wcts);
    printf("LU factorization %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();
#endif
    double **tmp1 = MatrixAlloc(ntrain, ntrain);
    double **tmp2 = MatrixAlloc(ntrain, 1);
    double **ftest = MatrixAlloc(ntest, 1);

    ForwardSubstitution(U, ntrain, &f_train[0][0], ntrain, &tmp1[0][0], ntrain);
    BackSubstitution(U, ntrain, &tmp1[0][0], ntrain, &tmp2[0][0], ntrain);

    Mmult(k, ntest, ntrain, tmp2, ntrain, 1, ftest, ntest, 1);
#ifdef TIME
    wctduration = (chrono::system_clock::now() - wcts);
    printf("Compute predicted value ftest %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();
#endif
    // Free allocated Matrix
    MatrixFree(K0);
    MatrixFree(K);
    MatrixFree(k);
    MatrixFree(f_train);
    MatrixFree(U);
    MatrixFree(tmp1);
    MatrixFree(tmp2);

    return ftest;
}

void randperm(int n, int *rperm) {
    // initialize rperm with step sequence
    for (int i = 0; i < n; i++) rperm[i] = i;
    // do 1 round of shuffle;
    srand (n);
    for (int i = 0; i < n; i++) {
        int swap = rand() % n;
        int tmp = rperm[i];
        rperm[i] = rperm[swap];
        rperm[swap] = tmp;
    }
}

int main (int argc, char *argv[]) {
    int m;
    double Lparam_start, Lparam_step;
    int nLparam;
    if (argc < 5) { 
	    printf("Usage: %s <m> <Lparam_start> <Lparam_step> <nLparam>\nSample use: \n\t%s 32 0.25 0.5 20\n", argv[0], argv[0]);
	    exit(0);
	} else {
	    m = atoi(argv[1]); 
        Lparam_start = atof(argv[2]);
        Lparam_step = atof(argv[3]);
        nLparam = atoi(argv[4]);
	}

    // Initialize m x m grid of points
    int n = m*m;
    double h = (double)1/(m+1);
    double *XY_1 = (double *)calloc(sizeof(double), n);
    double *XY_2 = (double *)calloc(sizeof(double), n);

    //#pragma omp parallel for shared (m, h, XY_1, XY_2) collapse(2) schedule(static) if (m > 64)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            XY_1[j+i*m] = (i+1)*h;
            XY_2[j+i*m] = (j+1)*h;
        }
    }

    double **f = MatrixAlloc(n, 1);
    double **f_kernel = MatrixAlloc(n, 1);
    double *fy1 = (double *)malloc(sizeof(double));
    double *fy2 = (double *)malloc(sizeof(double));
    fy1[0] = 0.25; fy2[0] = 0.25; 
    kernel(XY_1, XY_2, n, fy1, fy2, 1, (double)2/m, (double)2/m, f_kernel);
    int seed = m ;
    struct drand48_data drand_buf;
    // Initialize observed data vector f
    for (int i = 0; i < n; i++) {
        // int seed = omp_get_thread_num() ;
        // struct drand48_data drand_buf;
        srand48_r (seed, &drand_buf);
        drand48_r (&drand_buf, &(f[i][0]));
#ifdef DBG
        double **f_DBG = MatrixAlloc (n, 1);
        f_DBG[i][0] = f[i][0]; 
#endif
        f[i][0] = (f[i][0] - 0.5) * 0.02 + f_kernel[i][0] + XY_1[i]*0.2 + XY_2[i]*0.1;
    }
#ifdef DBG
        PrintErrMatrix(f, n, 1); 
#endif

    // Compute hyperparameters
    // Select 10% points as test point randomly 
    // and mark the remaining 90% as training points
    int ntest = (int)round(0.1 * n);
    int ntrain = n - ntest;
    int *rperm = new int[n];
    randperm(n, rperm);
#ifdef DBG
        PrintErrVector(rperm, n); 
#endif
    int *itest = &rperm[0];
    int *itrain = &rperm[ntest];

    //  Compute mse at a grid of points in the parameter space
    //  Parameters l1 and l2 are assigned values from Lparam
    //  Parameter t assigned values from Tparam 
    double Tparam = 0.5; // Select based on data
    double *Lparam = (double *)malloc(sizeof(double)*nLparam); 
    for (int i = 0; i < nLparam; i++) Lparam[i] = (Lparam_start + (double)i*Lparam_step) / m; // Select based on data

    double **MSE = ZeroMatrixAlloc(nLparam, nLparam);

    // time GPR kernels
	auto wcts = chrono::system_clock::now();

#if defined(OPT2) || !defined(OPT1) 
    #pragma omp parallel for collapse(2) schedule(static) 
#endif
    for (int i = 0; i < nLparam; i++) {
        for (int j = 0; j < nLparam; j++) {
            double **ftest = GPR(XY_1, XY_2, n, f, itest, ntest, itrain, ntrain, Tparam, Lparam[i], Lparam[j]);
            // calculate error
            for (int e = 0; e < ntest; e++) MSE[i][j] += pow(f[itest[e]][0] - ftest[e][0], 2);
            MSE[i][j] /= ntest;
            printf("Finished (l1,l2) = %f, %f, mse = %e\n", Lparam[i], Lparam[j], MSE[i][j]);
        }
    }

    chrono::duration<double, milli> wctduration = (chrono::system_clock::now() - wcts);
    printf("Finished in %.2f miliseconds [Wall Clock]\n", wctduration.count());

    printf("MSE: \n");
    PrintOutMatrix(MSE, nLparam, nLparam); 

    // Search minimum MSE
    double min_mse = numeric_limits<double>::max();
    int min_x, min_y;
    for (int i = 0; i < nLparam; i++) {
        for (int j = 0; j < nLparam; j++) {
            if (MSE[i][j] < min_mse) {
                min_mse = MSE[i][j];
                min_x = i; min_y = j;
            } 
        }
    }
    printf("Minimum MSE %f, Lparam_x %f, Lparam_y %f \n", min_mse, Lparam[min_x], Lparam[min_y]);
    return 0;
}
