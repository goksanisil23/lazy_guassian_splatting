#!/bin/bash

export PATH=/home/s0001734/Downloads/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/home/s0001734/Downloads/cuda-12.6/lib64:$LD_LIBRARY_PATH
mkdir -p build
cmake -DCMAKE_PREFIX_PATH=/home/s0001734/Downloads/libtorch/libtorch -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build