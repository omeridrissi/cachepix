#!/bin/bash
gcc bench_simd.c -o bench_simd -O3 -I../include/ -L../build/ -lcachepix -lm
gcc bench_scalar.c -o bench_scalar -O3 -I../include/ -L../build/ -lcachepix -lm
