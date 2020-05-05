#!/bin/bash

OUTPUT=cuda_eval
echo "" > ${OUTPUT}_small.$$
echo "" > ${OUTPUT}_large.$$
echo "" > ${OUTPUT}_extreme.$$

## small test 
./GPR_gpu 32 0.25 0.5 10 16 >> ${OUTPUT}_small.$$
./GPR_gpu 32 0.25 0.5 10 64 >> ${OUTPUT}_small.$$
./GPR_gpu 32 0.25 0.5 10 128 >> ${OUTPUT}_small.$$
./GPR_gpu 32 0.25 0.5 10 256 >> ${OUTPUT}_small.$$
./GPR_gpu 32 0.25 0.5 10 512 >> ${OUTPUT}_small.$$
./GPR_gpu 32 0.25 0.5 10 768 >> ${OUTPUT}_small.$$


## large test 
./GPR_gpu 48 0.25 0.5 20 16 >> ${OUTPUT}_large.$$
./GPR_gpu 48 0.25 0.5 20 64 >> ${OUTPUT}_large.$$
./GPR_gpu 48 0.25 0.5 20 128 >> ${OUTPUT}_large.$$
./GPR_gpu 48 0.25 0.5 20 256 >> ${OUTPUT}_large.$$
./GPR_gpu 48 0.25 0.5 20 512 >> ${OUTPUT}_large.$$
./GPR_gpu 48 0.25 0.5 20 768 >> ${OUTPUT}_large.$$


## extreme test 
./GPR_gpu 64 0.25 0.25 20 16 >> ${OUTPUT}_extreme.$$
./GPR_gpu 64 0.25 0.25 20 64 >> ${OUTPUT}_extreme.$$
./GPR_gpu 64 0.25 0.25 20 128 >> ${OUTPUT}_extreme.$$
./GPR_gpu 64 0.25 0.25 20 256 >> ${OUTPUT}_extreme.$$
./GPR_gpu 64 0.25 0.25 20 512 >> ${OUTPUT}_extreme.$$
./GPR_gpu 64 0.25 0.25 20 768 >> ${OUTPUT}_extreme.$$
