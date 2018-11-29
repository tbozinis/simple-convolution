#!/bin/bash

# firstly cd to the cpu dir
# compail and run the main.cpp file
# you will need g++
echo "--------------- Convolution in cpu ---------------"
cd cpu
g++ -o cpu main.cpp
./cpu

# after the cpu program is finished cd back to main dir and then to the gpu dir
cd ..
cd gpu

export LD_LIBRARY_PATH=/usr/local/cuda/lib
export PATH=$PATH:/usr/local/cuda/bin

# compile and run the gpu program
# you will need the nvidia cuda toolkit (nvcc)
echo "--------------- Convolution in gpu ---------------"
nvcc main.cu -o gpu 
nvprof ./gpu