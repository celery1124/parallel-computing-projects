#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <ctime>
#include <chrono>
#include <omp.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#define MAX_THREADS 1024

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
        U[j][j] = sqrt(K[j][j] + 0.01 - sum); 

        #pragma omp parallel for private(sum) shared(n, j, K, U) schedule(static) if (j < n-64)   
        for (int i = j+1; i < n; i++) { 
            sum = 0;
            for (int k = 0; k < j; k++) sum += (U[k][i] * U[k][j]);
            U[j][i] = (K[i][j] - sum) / U[j][j]; 
        } 
    } 
}

// cuda kernel version
__global__ void CholeskyFactorization_v1(double *K, double *U, int n, int row) 
{ 
    __shared__ double sum; 
    __shared__ double partial_sum[MAX_THREADS];
    // caller make sure Matrix U is zeroed (through calloc)
    // Cholesky-Crout Algorithm on row_id row
    int partial_row = (int ) ceil( (double)row / (double)blockDim.x);
    int row_start = partial_row * threadIdx.x;
    int row_end = partial_row + row_start < row ? partial_row + row_start : row;

    // Calculate Diagnols 
    partial_sum[threadIdx.x] = 0;
    for (int k = row_start; k < row_end; k++) {
        partial_sum[threadIdx.x] += (U[k*n + row] * U[k*n + row]);
    }
    __syncthreads();
    // Thread-0 calculate and write Diagnols for this row
    if (threadIdx.x == 0) {
        sum = 0;
        for (int i = 0; i < blockDim.x; i++) sum += partial_sum[i];
        U[row*n + row] = sqrt(K[row*n + row] + 0.01 - sum); 
    }
    __syncthreads();

    partial_row = (int) ceil( (double)(n-1-row) / (double)blockDim.x );
    row_start = row+1 + partial_row * threadIdx.x;
    row_end = partial_row + row_start < n ? partial_row + row_start : n;
    for (int i = row_start; i < row_end; i++) { 
        double sum2 = 0;
        for (int k = 0; k < row; k++) sum2 += (U[k*n + i] * U[k*n + row]);
        U[row*n + i] = (K[i*n + row] - sum2) / U[row*n + row]; 
    } 
}

__global__ void CholeskyFactorization_v2(double *K, double *U, int n) 
{ 
    __shared__ double sum; 
    __shared__ double partial_sum[MAX_THREADS];
    // caller make sure Matrix U is zeroed (through calloc)
    // Cholesky-Crout Algorithm 

    for (int j = 0; j < n; j++) { 
        int partial_row = (int ) ceil( (double)j / (double)blockDim.x);
        int row_start = partial_row * threadIdx.x;
        int row_end = partial_row + row_start < j ? partial_row + row_start : j;
        // Calculate Diagnols 
        partial_sum[threadIdx.x] = 0;
        for (int k = row_start; k < row_end; k++) {
            partial_sum[threadIdx.x] += (U[k*n + j] * U[k*n + j]);
        }
        __syncthreads();

        // Thread-0 calculate and write Diagnols for this row
        if (threadIdx.x == 0) {
            sum = 0;
            for (int i = 0; i < blockDim.x; i++) sum += partial_sum[i];
            U[j*n + j] = sqrt(K[j*n + j] + 0.01 - sum); 
        }
        __syncthreads();

        partial_row = (int) ceil( (double)(n-1-j) / (double)blockDim.x );
        row_start = j+1 + partial_row * threadIdx.x;
        row_end = partial_row + row_start < n ? partial_row + row_start : n;
        for (int i = row_start; i < row_end; i++) { 
            double sum2 = 0;
            for (int k = 0; k < j; k++) sum2 += (U[k*n + i] * U[k*n + j]);
            U[j*n + i] = (K[i*n + j] - sum2) / U[j*n + j]; 
        } 
        __syncthreads();
    }
}

__global__ void DeriveZFromLU(double *U, double *L, int n, double *f, double *tmp, double *z) {
    int block_size = (int ) ceil( (double)n / (double)blockDim.x);
    int row_start = block_size * threadIdx.x;
    int col_start = block_size * threadIdx.y;
    int row_end = block_size + row_start < n ? block_size + row_start : n;
    int col_end = block_size + col_start < n ? block_size + col_start : n;
    // Transpose U to L

    for (int i = row_start; i < row_end; i++) {
        for (int j = col_start; j < col_end; j++) {
            L[j*n + i] = U[i*n + j];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // ForwardSubstitution
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < i; j++) {
                sum += L[i*n + j] * tmp[j];
            }
            tmp[i] = (f[i] - sum) / L[i*n + i];
        }

        // BackSubstitution
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0;
            for (int j = i + 1; j < n; j++) {
                sum += U[i*n + j] * z[j];
            }
            z[i] = (tmp[i] - sum) / U[i*n + i];
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

double GPR(int m, double rstar_l, double rstar_r, int cuda_threads) {
    
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
        }
    }
    wctduration = (chrono::system_clock::now() - wcts);
    printf("Initialize K %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();

    // Compute LU factorization of tI + K
    double **L = ZeroMatrixAlloc(n, n);

#ifdef VERIFY
    double **U = ZeroMatrixAlloc(n, n);
    CholeskyFactorization(K, U, n);
    wctduration = (chrono::system_clock::now() - wcts);
    printf("LU factorization %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();
#endif

    // cuda lu factorization
    double **U_device = ZeroMatrixAlloc(n, n);
    double* d_K;
    int device_matrix_size = sizeof(double) * n * n;
    cudaMalloc(&d_K, device_matrix_size);
    cudaMemcpy(d_K, &(K[0][0]), device_matrix_size, cudaMemcpyHostToDevice);
    double* d_U;
    cudaMalloc(&d_U, device_matrix_size);
    cudaMemset(d_U, 0, device_matrix_size); 

    // Invoke kernel
#ifndef OPT
    for (int i = 0; i < n; i++)
        CholeskyFactorization_v1<<<1, cuda_threads>>>(d_K, d_U, n, i);
#else
    CholeskyFactorization_v2<<<1, cuda_threads>>>(d_K, d_U, n);
#endif

    cudaMemcpy(&(U_device[0][0]), d_U, device_matrix_size, cudaMemcpyDeviceToHost); 
    wctduration = (chrono::system_clock::now() - wcts);
    printf("[CUDA] LU factorization %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();

#ifdef VERIFY
    // compare
    bool verify = true;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(U_device[i][j] - U[i][j]) > 1e-6) {
                printf("[CUDA] compare d_U results error: i = %d, j = %d, U_device: %.6f, U %.6f\n",i,j,U_device[i][j],U[i][j]);
                verify = false;
                break;
            }
        }
        if (!verify) break;
    }
    if (verify) printf("[CUDA] compare d_U results correct!] \n");
    MatrixFree(U);
#endif 

    double **tmp1 = MatrixAlloc(n, n);
    double **tmp2 = MatrixAlloc(n, 1);
    double **fstar = MatrixAlloc(1, 1);

    UTMTranspose(U_device, n, n, L, n, n); // inv(U) = inv(L)^T
    ForwardSubstitution(L, n, &f[0][0], n, &tmp1[0][0], n);
    BackSubstitution(U_device, n, &tmp1[0][0], n, &tmp2[0][0], n);

    wctduration = (chrono::system_clock::now() - wcts);
    printf("Compute predicted value fstar %.3f ms\n", wctduration.count());
    wcts = chrono::system_clock::now();

    // // CUDA Derive z from LU %% To slow!
    // double* d_L;
    // cudaMalloc(&d_L, device_matrix_size);
    // cudaMemset(d_L, 0, device_matrix_size); 
    // double* d_f;
    // cudaMalloc(&d_f, device_matrix_size);
    // cudaMemcpy(d_f, &(f[0][0]), sizeof(double) * n, cudaMemcpyHostToDevice);
    // double* d_tmp;
    // cudaMalloc(&d_tmp, sizeof(double) * n);
    // double* d_z;
    // cudaMalloc(&d_z, sizeof(double) * n);
    // // Invoke kernel
    // DeriveZFromLU<<<32,32>>>(d_U, d_L, n, d_f, d_tmp, d_z);
    
    // double **z_device = MatrixAlloc(n, 1);
    // cudaMemcpy(&(z_device[0][0]), d_z, sizeof(double) * n, cudaMemcpyDeviceToHost); 

    // compare
    // verify = true;
    // for (int i = 0; i < n; i++) {
    //     if (fabs(z_device[i][0] - tmp2[i][0]) > 1e-4) {
    //         printf("[CUDA results error] i = %d, z_device: %.6f, tmp2 %.6f\n",i,z_device[i],tmp2[i]);
    //         verify = false;
    //         break;
    //     }
    // }
    // if (verify) printf("[CUDA results correct!] \n");

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
    int cuda_threads;
    if (argc < 4) { 
	    printf("Usage: %s m rstar_l rstar_r cuda_threads\nSample use: \n\t%s 16 0.5 0.5 32\n", argv[0], argv[0]);
	    exit(0);
	} else {
	    m = atoi(argv[1]); 
	    rstar_l = atof(argv[2]); 
	    rstar_r = atof(argv[3]); 
	    cuda_threads = atoi(argv[4]); 
	}

    // time GPR
	auto wcts = chrono::system_clock::now();
    
    double fstar = GPR(m, rstar_l, rstar_r, cuda_threads);
    
    chrono::duration<double, milli> wctduration = (chrono::system_clock::now() - wcts);
    printf("Finished in %.2f miliseconds [Wall Clock]\n", wctduration.count());

    printf("fstar %.6f\n", fstar);
    return 0;
}
