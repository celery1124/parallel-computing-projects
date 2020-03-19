#!/bin/bash

g++ -fopenmp -std=c++11 Rinverse.cc -O3 -o Rinv_nonopt_gcc
g++ -fopenmp -std=c++11 Rinverse.cc -O3 -DOPT1 -o Rinv_opt1_gcc
g++ -fopenmp -std=c++11 Rinverse.cc -O3 -DOPT2 -o Rinv_opt2_gcc
g++ -fopenmp -std=c++11 Rinverse.cc -O3 -DOPT3 -o Rinv_opt3_gcc
g++ -fopenmp -std=c++11 Rinverse.cc -O3 -DOPT4 -o Rinv_opt4_gcc
g++ -fopenmp -std=c++11 Rinverse.cc -O3 -DOPT5 -o Rinv_opt5_gcc
