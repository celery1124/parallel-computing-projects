#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <cstring>
#include <vector>
#include <ctime>
#include <chrono>
#include <sched.h>
#include <omp.h>

using namespace std;

// in-place matrix multiplication
// A = A * B
void MmultL(double **A, int cOA, int rA, int cA, double **B, int cOB, int rB, int cB) {
    assert(cA == rB);
    assert(cA == cB);

    vector<vector<double>> C(rA, vector<double>(cB));
#if defined(OPT4) || defined(OPT5)
    if (rA <= 32) {
#else 
    if (false) {
#endif
        for (int r = 0; r < rA; r++) {
            for (int c = 0; c < cB; c++) {
                double sum = 0;
                for (int i = 0; i < cA; i++) {
                    sum += A[r][cOA+i] * B[i][cOB+c];
                }
                C[r][c] = sum;
            }
        }
    }
    else {
#if defined(OPT3) || defined(OPT4)
        #pragma omp parallel for shared(rA, cB, A, B, C) collapse(2) schedule(dynamic, 16384) 
#elif defined(OPT5)
        #pragma omp parallel for shared(rA, cB, A, B, C) collapse(2) schedule(static, 16384)
#else   
#endif
        for (int r = 0; r < rA; r++) {
            for (int c = 0; c < cB; c++) {
                double sum = 0;
                for (int i = 0; i < cA; i++) {
                    sum += A[r][cOA+i] * B[i][cOB+c];
                }
                C[r][c] = sum;
            }
        }
    }   
    // copy C to A
    // int thread_num = omp_get_thread_num();
    // int cpu_num = sched_getcpu();
    // auto wcts = chrono::system_clock::now();
    if (rA <= 1024) {
        for (int i = 0; i < rA; i++) memcpy((void *)&A[i][cOA], C[i].data(), cA * sizeof(double));
    }
    else {
        #pragma omp parallel for shared(rA, cA, A, C) 
        for (int i = 0; i < rA; i++) memcpy((void *)&A[i][cOA], C[i].data(), cA * sizeof(double));
    }
    // chrono::duration<double, milli> wctduration = (chrono::system_clock::now() - wcts);
    // printf("Thread %d (%d), top right %.3f ms\n", thread_num, cpu_num, wctduration.count());
}

// in-place matrix multiplication
// B = -A * B
void MmultR(double **A, int cOA, int rA, int cA, double **B, int cOB, int rB, int cB) {
    assert(cA == rB);
    assert(rA == rB);

    vector<vector<double>> C(rA, vector<double>(cB));
#if defined(OPT4) || defined(OPT5)
    if (rA <= 32) {
#else 
    if (false) {
#endif
        for (int r = 0; r < rA; r++) {
            for (int c = 0; c < cB; c++) {
                double sum = 0;
                for (int i = 0; i < cA; i++) {
                    sum += -A[r][cOA+i] * B[i][cOB+c];
                }
                C[r][c] = sum;
            }
        }
    }
    else {
#if defined(OPT3) || defined(OPT4)
        #pragma omp parallel for shared(rA, cB, A, B, C) collapse(2) schedule(dynamic, 16384) 
#elif defined(OPT5)
        #pragma omp parallel for shared(rA, cB, A, B, C) collapse(2) schedule(static, 16384)
#else   
#endif
        for (int r = 0; r < rA; r++) {
            for (int c = 0; c < cB; c++) {
                double sum = 0;
                for (int i = 0; i < cA; i++) {
                    sum += -A[r][cOA+i] * B[i][cOB+c];
                }
                C[r][c] = sum;
            }
        }
    }
   
    // copy C to B
    if (rB <= 1024) {
        for (int i = 0; i < rB; i++) memcpy((void *)&B[i][cOB], C[i].data(), cB * sizeof(double));
    }
    else {
        #pragma omp parallel for shared(rB, cB, B, C)
        for (int i = 0; i < rB; i++) memcpy((void *)&B[i][cOB], C[i].data(), cB * sizeof(double));
    }
}

void RecurRinverse(double **A, int cOA, int n) {
    assert(cOA >=0 && n > 0);
    if (n == 1){ // 1x1
        A[0][cOA] = 1/A[0][cOA];
    }
    else if (n == 2){ // 2x2
        A[0][cOA] = 1/A[0][cOA];
        A[1][cOA+1] = 1/A[1][cOA+1];
        A[0][cOA+1] = -A[0][cOA] * A[0][cOA+1] * A[1][cOA+1];
    }
    else if (n == 3){ // 3x3
        double old_A11 = A[1][cOA+1];
        A[0][cOA] = 1/A[0][cOA];
        A[1][cOA+1] = 1/A[1][cOA+1];
        A[2][cOA+2] = 1/A[2][cOA+2];
        A[0][cOA+2] = (A[0][cOA+1] * A[1][cOA+2] - A[0][cOA+2] * old_A11) * A[0][cOA] * A[1][cOA+1] * A[2][cOA+2];
        A[0][cOA+1] = -A[0][cOA+1] * A[0][cOA] * A[1][cOA+1];
        A[1][cOA+2] = -A[1][cOA+2] * A[1][cOA+1] * A[2][cOA+2];
    }
    else {
        int nn = n/2;
        RecurRinverse(&A[0], cOA, nn);
        RecurRinverse(&A[nn], cOA+nn, n - nn);
        MmultR(&A[0], cOA, nn, nn, &A[0], cOA+nn, nn, n-nn);
        MmultL(&A[0], cOA+nn, nn, n-nn, &A[nn], cOA+nn, n-nn, n-nn);
    }
}

void _OMPRecurRinverse(double **A, int cOA, int n) {
    assert(cOA >=0 && n > 0);
    int thread_num = omp_get_thread_num();
    int cpu_num = sched_getcpu();
#if defined(OPT2) || defined(OPT3) || defined(OPT4) || defined(OPT5)
    if (n <= 32) {
#else
    if (n <= 2) {
#endif
        // auto wcts = chrono::system_clock::now();
        RecurRinverse(A, cOA, n);
        // chrono::duration<double, milli> wctduration = (chrono::system_clock::now() - wcts);
        // printf("Thread %d (%d), direct Rinverse %.3f ms\n", thread_num, cpu_num, wctduration.count());
    }
    else {
        int nn = n/2;
#if defined(OPT1) || defined(OPT2) || defined(OPT3) || defined(OPT4) || defined(OPT5)
        #pragma omp task 
#endif
        {
            // auto wcts = chrono::system_clock::now();
            _OMPRecurRinverse(&A[0], cOA, nn);
            // chrono::duration<double, milli> wctduration = (chrono::system_clock::now() - wcts);
            // printf("Thread %d (%d), top left %.3f ms\n", thread_num, cpu_num, wctduration.count());
        }
#if defined(OPT1) || defined(OPT2) || defined(OPT3) || defined(OPT4) || defined(OPT5)
        #pragma omp task 
#endif
        {
            // auto wcts = chrono::system_clock::now();
            _OMPRecurRinverse(&A[nn], cOA+nn, n - nn);
            // chrono::duration<double, milli> wctduration = (chrono::system_clock::now() - wcts);
            // printf("Thread %d (%d), bottom right %.3f ms\n", thread_num, cpu_num, wctduration.count());
        }
#if defined(OPT1) || defined(OPT2) || defined(OPT3) || defined(OPT4) || defined(OPT5)
        #pragma omp taskwait
#endif
        // auto wcts = chrono::system_clock::now();
        MmultR(&A[0], cOA, nn, nn, &A[0], cOA+nn, nn, n-nn);
        MmultL(&A[0], cOA+nn, nn, n-nn, &A[nn], cOA+nn, n-nn, n-nn);
        // chrono::duration<double, milli> wctduration = (chrono::system_clock::now() - wcts);
        // printf("Thread %d (%d), top right [%d, %d][%d] %.3f ms\n", thread_num, cpu_num, cOA, cOA+nn, nn, wctduration.count());
    }
    return;
}

void OMPRinverse(double **A, int n) {
    _OMPRecurRinverse(A, 0, n);
}

void PrintMatrix(double **A, int n) {
    for(int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.4f\t", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

double **MatrixAlloc(int n) {
    double **A = new double*[n];
    for (int i = 0; i < n; i++) A[i] = new double[n];
    return A;
}

void MatrixFree(double **A, int n) {
    for (int i = 0; i < n; i++) delete [] A[i];
    delete [] A;
}

void GenUpperTriangleMatrix(double **R, int n) {
    unsigned int seed = n;
    double *sum_col = new double[n];
    for (int j = 0; j < n; j++) {
        sum_col[j] = 0;
        for(int i = 0; i < n; i++) {
            R[i][j] =(double)(rand_r(&seed))/(double)(RAND_MAX);
            sum_col[j] += R[i][j];
        }
    }
    for(int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if(i==j) R[i][j] += sum_col[j];
            else if(i > j) R[i][j] = 0;
        }
    }
    delete [] sum_col;
}

int main (int argc, char *argv[]) {
    int n;
    if (argc == 1) { 
	    printf("Enter matrix size:");
	    scanf("%d",&n);
	} else {
	    n = atoi(argv[1]); 
	}
	printf("Matrix size: %d\n\n", n);

    // Generate random upper triangle matrix
    double **R = MatrixAlloc(n);
    GenUpperTriangleMatrix(R, n);
#ifdef DBG
    printf("oringal matrix: \n\n");
    PrintMatrix(R, n);
#endif

    auto wcts = chrono::system_clock::now();

    // Inverse matrix
#if defined(OPT1) || defined(OPT2) || defined(OPT3) || defined(OPT4) || defined(OPT5)
    #pragma omp parallel
    {
    #pragma omp single
    {
#endif
        OMPRinverse(R, n);
#if defined(OPT1) || defined(OPT2) || defined(OPT3) || defined(OPT4) || defined(OPT5)
    }
    }
#endif

    chrono::duration<double, milli> wctduration = (chrono::system_clock::now() - wcts);
    printf("Finished in %.2f miliseconds [Wall Clock]\n", wctduration.count());
#ifdef DBG
    printf("inversed matrix: \n\n");
    PrintMatrix(R, n);
#endif

    return 0;
}