#!/bin/bash

icpc -qopenmp -std=c++11 Rinverse.cc -O3 -o Rinv_nonopt_icc
icpc -qopenmp -std=c++11 Rinverse.cc -O3 -DOPT1 -o Rinv_opt1_icc
icpc -qopenmp -std=c++11 Rinverse.cc -O3 -DOPT2 -o Rinv_opt2_icc
icpc -qopenmp -std=c++11 Rinverse.cc -O3 -DOPT3 -o Rinv_opt3_icc
icpc -qopenmp -std=c++11 Rinverse.cc -O3 -DOPT4 -o Rinv_opt4_icc
icpc -qopenmp -std=c++11 Rinverse.cc -O3 -DOPT5 -o Rinv_opt5_icc
