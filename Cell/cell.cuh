#ifndef CELL_H
#define CELL_H
#include <iostream>
#include "math.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuda.h"
#include <cublas_v2.h>

using namespace std;

//code acquired from website. (see .cuh comment for details)
//https://gist.github.com/jefflarkin/5390993
#define cudaCheckError() {                                          \
  cudaError_t e=cudaGetLastError();                                 \
  if(e!=cudaSuccess) {                                              \
    printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__, cudaGetErrorString(e));           \
    system("pause");												\
    exit(0); \
  }                                                                 \
}


class Cell{
  unsigned int size;
  unsigned int *numerals; //need to track and update numeral list/size.
  unsigned int value; // keeping track of the numeral value of the cell (mainly for fixed numerals)

  template<class t>
  __host__ t *allocateHost(unsigned int size);
  template<class t>
  __host__ t *allocateDevice(unsigned int size);
public:
  __host__ __device__ Cell(unsigned int size);
  __host__ __device__ unsigned int getSize();
};

#endif
