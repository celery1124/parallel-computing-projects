#!/bin/bash

icpc -qopenmp -std=c++11 GPR.cc Rinverse.cc -O3 -o GPR_icc
icpc -qopenmp -std=c++11 GPR.cc Rinverse.cc -O3 -DOPT1 -o GPR_opt1_icc
icpc -qopenmp -std=c++11 GPR.cc Rinverse.cc -O3 -DOPT2 -o GPR_opt2_icc
