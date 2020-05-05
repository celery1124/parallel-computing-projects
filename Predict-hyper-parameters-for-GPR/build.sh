#!/bin/bash

icpc -qopenmp -std=c++11 GPR.cc -O3 -DOPT1 -o GPR_omp_opt1
icpc -qopenmp -std=c++11 GPR.cc -O3 -DOPT2 -o GPR_omp_opt2
nvcc -arch=compute_35 -code=sm_35 -ccbin=icc -Xcompiler="-std=c++11 -O3" -O3 -o GPR_gpu GPR.cu
