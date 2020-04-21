#!/bin/bash

icpc -qopenmp -std=c++11 GPR.cc -O3 -o GPR_omp
nvcc -arch=compute_35 -code=sm_35 -ccbin=icc -Xcompiler="-fopenmp -std=c++11 -O3" -O3 -o GPR_gpu GPR.cu
nvcc -arch=compute_35 -code=sm_35 -ccbin=icc -Xcompiler="-fopenmp -std=c++11 -O3" -O3 -DOPT -o GPR_gpu_opt GPR.cu
