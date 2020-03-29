#ifndef _Rinverse_h
#define _Rinverse_h

#include <stdlib.h>
#include <stdio.h>
#include <cassert>
#include <cstring>
#include <vector>
#include <ctime>
#include <chrono>
#include <sched.h>
#include <omp.h>

// public function

// OMP version of Matrix inverse
void OMPRinverse(double **A, int n);

// Serial version of Matrix inverse
void SerialRinverse(double **A, int n);

#endif