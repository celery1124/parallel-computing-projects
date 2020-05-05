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

#define MAX_THREADS 1024
#define MAX_HEAPSIZE (unsigned long long)4<<30
#define PI 3.14159265358979323846

// upper triangular matrix transpose
// B = A^T (A is upper triangular matrix)
// B should be initialized as zeroed matrix
__device__ void UTMTranspose(double *A, double *B, int n) 
{
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            B[j*n + i] = A[i*n + j];
        }
    }
} 

// forward substitution 
// y = L\f
__device__ void ForwardSubstitution(double *U, int nL, double *f, int nf, double *y, int ny) 
{   // L[i][j] = U[j][i]
    // L[i*nL + j] = U[j*nL + i]
    for (int i = 0; i < nL; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += U[j*nL + i] * y[j];
        }
        y[i] = (f[i] - sum) / U[i*nL + i];
    }
} 

// backward substitution 
// y = U\f
__device__ void BackSubstitution(double *U, int nU, double *f, int nf, double *y, int ny) 
{
    for (int i = nU - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < nU; j++) {
            sum += U[i*nU + j] * y[j];
        }
        y[i] = (f[i] - sum) / U[i*nU + i];
    }
} 

__constant__ double d_PI = 3.14159265358979323846;
// Kernel function
__device__ void kernel(double *x1, double *x2, int x_size, double *y1, double *y2, int y_size, double l1, double l2, double *res) {
    double coeff = 1/sqrt(2*d_PI);
    // only parallel outer loop
    int r_step = (int ) ceil( (double)x_size / (double)blockDim.x);
    int r_start = r_step * threadIdx.x;
    int r_end = r_step + r_start < x_size ? r_step + r_start : x_size;
    for (int i = r_start; i < r_end; i++) {
        for (int j = 0; j < y_size; j++) {
            res[i*x_size + j] = coeff * exp(-0.5*(pow(x1[i]-y1[j], 2)/pow(l1,2) + pow(x2[i]-y2[j], 2)/pow(l2,2)));
        }
    }
}

// Extrace submatrix from row column vector
__device__ void ExtractFromMatrix(double *X, int rX, int *r, int *c, double *Y, int rY, int cY) {
    // nr == rY && nc = cY
    // only parallel outer loop
    int r_step = (int ) ceil( (double)rY / (double)blockDim.x);
    int r_start = r_step * threadIdx.x;
    int r_end = r_step + r_start < rY ? r_step + r_start : rY;
    for (int i = r_start; i < r_end; i++) {
        for (int j = 0; j < cY; j++) {
            Y[i*cY + j] = X[r[i]* rX + c[j]];
        }
    }
}

// Extrace subvector from row vector
__device__ void ExtractFromVector(double *X, int *r, double *Y, int rY) {
    int r_step = (int ) ceil( (double)rY / (double)blockDim.x);
    int r_start = r_step * threadIdx.x;
    int r_end = r_step + r_start < rY ? r_step + r_start : rY;
    for (int i = r_start; i < r_end; i++) {
        Y[i] = X[r[i]];
    }
}

// matrix multiplication
// C = A * B
__device__ void Mmult(double *A, int rA, int cA, double *B, int rB, int cB, double *C) {
    // only parallel outer loop
    int r_step = (int ) ceil( (double)rA / (double)blockDim.x);
    int r_start = r_step * threadIdx.x;
    int r_end = r_step + r_start < rA ? r_step + r_start : rA;
    for (int r = r_start; r < r_end; r++) {
        for (int c = 0; c < cB; c++) {
            double sum = 0;
            for (int i = 0; i < cA; i++) {
                sum += A[r*cA + i] * B[i*cB + c];
            }
            C[r*cB + c] = sum;
        }
    }
}

// Compute predictions at test points using training points and observed 
// values at those training poionts as input to the model with 
// hyperparameters t and l. 
__global__ void GPR(double *XY, int n, double *f, int *itest, int ntest, \
                    int *itrain, int ntrain, double t, double *l, int nl, int offset, double *MSE) 
{ 
    __shared__ double sum; 
    __shared__ double partial_sum[MAX_THREADS];
    __shared__ double mse_sum[MAX_THREADS];
    __shared__ double l1;
    __shared__ double l2;

    __shared__ double *K0;
    __shared__ double *K;
    __shared__ double *k;
    __shared__ double *f_train;
    __shared__ double *U;
    __shared__ double *tmp1;
    __shared__ double *tmp2;
    __shared__ double *ftest;
    // First thread in the block dynamic allocate global memory for GPR calculation
    if (threadIdx.x == 0) {
        K0 = (double*) malloc (sizeof(double) * n * n);
        K = (double*) malloc (sizeof(double) * ntrain * ntrain);
        k = (double*) malloc (sizeof(double) * ntest * ntrain);
        f_train = (double*) malloc(sizeof(double) * ntrain);
        // U = (double*) malloc (sizeof(double) * ntrain * ntrain);
        // tmp1 = (double*) malloc (sizeof(double) * ntrain * ntrain);
        // tmp2 = (double*) malloc (sizeof(double) * ntrain);
        // ftest = (double*) malloc (sizeof(double) * ntest);
    }
    __syncthreads();

    // Initialize K for all points
    l1 = l[(offset+blockIdx.x)/nl];
    l2 = l[(offset+blockIdx.x)%nl];
    kernel(XY, &XY[n], n, XY, &XY[n], n, l1, l2, K0);
    __syncthreads();

    ExtractFromMatrix(K0, n, itrain, itrain, K, ntrain, ntrain);
    ExtractFromMatrix(K0, n, itest, itrain, k, ntest, ntrain);
    ExtractFromVector(f, itrain, f_train, ntrain);
    __syncthreads();

    if (threadIdx.x == 0) {
        free(K0);
        U = (double*) malloc (sizeof(double) * ntrain * ntrain);
    }
    __syncthreads();

    // make sure Matrix U is lower zeroed 
    int rr_step = (int ) ceil( (double)ntrain*ntrain / (double)blockDim.x);
    int rr_start = rr_step * threadIdx.x;
    int rr_end = rr_step + rr_start < ntrain*ntrain ? rr_step + rr_start : ntrain*ntrain;
    for (int i = rr_start; i < rr_end; i++) {
        U[i] = 0;
    }
    __syncthreads();

    // Cholesky-Crout Algorithm 
    for (int j = 0; j < ntrain; j++) { 
        int r_step = (int ) ceil( (double)j / (double)blockDim.x);
        int r_start = r_step * threadIdx.x;
        int r_end = r_step + r_start < j ? r_step + r_start : j;
        // Calculate Diagnols 
        partial_sum[threadIdx.x] = 0;
        for (int k = r_start; k < r_end; k++) {
            partial_sum[threadIdx.x] += (U[k*ntrain + j] * U[k*ntrain + j]);
        }
        __syncthreads();

        // Thread-0 calculate and write Diagnols for this row
        if (threadIdx.x == 0) {
            sum = 0;
            for (int i = 0; i < blockDim.x; i++) sum += partial_sum[i];
            U[j*ntrain + j] = sqrt(K[j*ntrain + j] + t - sum); 
        }
        __syncthreads();

        r_step = (int) ceil( (double)(ntrain-1-j) / (double)blockDim.x );
        r_start = j+1 + r_step * threadIdx.x;
        r_end = r_step + r_start < ntrain ? r_step + r_start : ntrain;
        for (int i = r_start; i < r_end; i++) { 
            double sum2 = 0;
            for (int k = 0; k < j; k++) sum2 += (U[k*ntrain + i] * U[k*ntrain + j]);
            U[j*ntrain + i] = (K[i*ntrain + j] - sum2) / U[j*ntrain + j]; 
        } 
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        free(K);
        tmp1 = (double*) malloc (sizeof(double) * ntrain * ntrain);
        tmp2 = (double*) malloc (sizeof(double) * ntrain);
        ftest = (double*) malloc (sizeof(double) * ntest);
    }

    // Compute predicted values
    if (threadIdx.x == 0) {
        ForwardSubstitution(U, ntrain, f_train, ntrain, tmp1, ntrain);
        BackSubstitution(U, ntrain, tmp1, ntrain, tmp2, ntrain);
    }
    __syncthreads();
    Mmult(k, ntest, ntrain, tmp2, ntrain, 1, ftest);
    __syncthreads();

    // Calculate error
    int r_step = (int ) ceil( (double)ntest / (double)blockDim.x);
    int r_start = r_step * threadIdx.x;
    int r_end = r_step + r_start < ntest ? r_step + r_start : ntest;
    mse_sum[threadIdx.x] = 0;
    for (int i = r_start; i < r_end; i++) mse_sum[threadIdx.x] += pow(f[itest[i]] - ftest[i], 2);
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; i++) MSE[offset+blockIdx.x] += mse_sum[i];
        MSE[offset+blockIdx.x] /= ntest;
        
        // Free up global memory
        // free(K0);
        // free(K);
        free(k);
        free(f_train);
        free(U);
        free(tmp1);
        free(tmp2);
        free(ftest);
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

void PrintOutMatrix(double **A, int rA, int cA) {
    for(int i = 0; i < rA; i++) {
        for (int j = 0; j < cA; j++) {
            fprintf(stdout, "%.4f\t", A[i][j]);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}

// Host side Kernel function
void host_kernel(double *x1, double *x2, int x_size, double *y1, double *y2, int y_size, double l1, double l2, double **res) {
    // caller needs to allocate/free res -> MatrixAlloc(x_size, y_size);
    double coeff = 1/sqrt(2*PI);
    for (int i = 0; i < x_size; i++) {
        for (int j = 0; j < y_size; j++) {
            res[i][j] = coeff * exp(-0.5*(pow(x1[i]-y1[j], 2)/pow(l1,2) + pow(x2[i]-y2[j], 2)/pow(l2,2)));
        }
    }
}

int main (int argc, char *argv[]) {
    int m;
    double Lparam_start, Lparam_step;
    int nLparam;
    int cuda_threads;
    if (argc < 6) { 
	    printf("Usage: %s <m> <Lparam_start> <Lparam_step> <nLparam> <cuda_threads>\nSample use: \n\t%s 32 0.25 0.5 20 1024\n", argv[0], argv[0]);
	    exit(0);
	} else {
	    m = atoi(argv[1]); 
        Lparam_start = atof(argv[2]);
        Lparam_step = atof(argv[3]);
        nLparam = atoi(argv[4]);
        cuda_threads = atoi(argv[5]);
	}

    // Initialize m x m grid of points
    int n = m*m;
    double h = (double)1/(m+1);
    double *XY_1 = (double *)calloc(sizeof(double), n);
    double *XY_2 = (double *)calloc(sizeof(double), n);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            XY_1[j+i*m] = (i+1)*h;
            XY_2[j+i*m] = (j+1)*h;
        }
    }

    double *f = (double *)malloc(sizeof(double) * n);
    double **f_kernel = MatrixAlloc(n, 1);
    double *fy1 = (double *)malloc(sizeof(double));
    double *fy2 = (double *)malloc(sizeof(double));
    fy1[0] = 0.25; fy2[0] = 0.25; 
    host_kernel(XY_1, XY_2, n, fy1, fy2, 1, (double)2/m, (double)2/m, f_kernel);
    int seed = m ;
    struct drand48_data drand_buf;
    // Initialize observed data vector f
    for (int i = 0; i < n; i++) {
        // int seed = omp_get_thread_num() ;
        // struct drand48_data drand_buf;
        srand48_r (seed, &drand_buf);
        drand48_r (&drand_buf, &(f[i]));
        f[i] = (f[i] - 0.5) * 0.02 + f_kernel[i][0] + XY_1[i]*0.2 + XY_2[i]*0.1;
    }


    // Compute hyperparameters
    // Select 10% points as test point randomly 
    // and mark the remaining 90% as training points
    int ntest = (int)round(0.1 * n);
    int ntrain = n - ntest;
    int *rperm = new int[n];
    randperm(n, rperm);
    int *itest = &rperm[0];
    int *itrain = &rperm[ntest];

    //  Compute mse at a grid of points in the parameter space
    //  Parameters l1 and l2 are assigned values from Lparam
    //  Parameter t assigned values from Tparam 
    double Tparam = 0.5; // Select based on data
    double *Lparam = (double *)malloc(sizeof(double)*nLparam); 
    for (int i = 0; i < nLparam; i++) Lparam[i] = (Lparam_start + (double)i*Lparam_step) / m; // Select based on data

    double **MSE = ZeroMatrixAlloc(nLparam, nLparam);

    // set maximum heap-size for K20
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, MAX_HEAPSIZE);
    // calculate how much thread blocks we can issue at the same time (limited by MAX_HEAPSIZE)
    int grid_size = floor((MAX_HEAPSIZE) / (2*sizeof(double) * n * n) * 0.9);
    printf("grid size = %d x 1\n", grid_size);

    // time GPR kernels
    auto wcts = chrono::system_clock::now();
    
    double *d_XY;
    cudaMalloc(&d_XY, sizeof(double) * n * 2);
    cudaMemcpy(&d_XY[0], XY_1, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(&d_XY[n], XY_2, sizeof(double) * n, cudaMemcpyHostToDevice);
    double *d_f;
    cudaMalloc(&d_f, sizeof(double) * n);
    cudaMemcpy(d_f, f, sizeof(double) * n, cudaMemcpyHostToDevice);
    int *d_itest;
    cudaMalloc(&d_itest, sizeof(int) * ntest);
    cudaMemcpy(d_itest, itest, sizeof(int) * ntest, cudaMemcpyHostToDevice);
    int *d_itrain;
    cudaMalloc(&d_itrain, sizeof(int) * ntrain);
    cudaMemcpy(d_itrain, itrain, sizeof(int) * ntrain, cudaMemcpyHostToDevice);
    double *d_l;
    cudaMalloc(&d_l, sizeof(double) * nLparam);
    cudaMemcpy(d_l, Lparam, sizeof(double) * nLparam, cudaMemcpyHostToDevice);
    double *d_MSE;
    cudaMalloc(&d_MSE, sizeof(double) * nLparam * nLparam);
    cudaMemset(d_MSE, 0, sizeof(double) * nLparam * nLparam); 
    // invoke cuda kernel
    // dim3 dimGrid(1, 1);
    for (int ix = 0; ix < nLparam * nLparam; ix+=grid_size) {
        int dimGrid = (nLparam * nLparam-ix < grid_size ? nLparam * nLparam-ix : grid_size); 
        GPR<<<dimGrid, cuda_threads>>>(d_XY, n, d_f, d_itest, ntest, d_itrain, ntrain, Tparam, d_l, nLparam, ix, d_MSE);
    }
    

    cudaMemcpy(&(MSE[0][0]), d_MSE, sizeof(double) * nLparam * nLparam, cudaMemcpyDeviceToHost); 

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
