#!/bin/bash
set -exu

/opt/topscc/bin/topscc kernels/transpose_kernel.cpp -o kernels/tmp.out  -ltops -arch gcu200 -std=c++17 -O3 --save-temps