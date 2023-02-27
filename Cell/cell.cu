#include "cell.cuh"
#include "math.h"

__host__ __device__ Cell::Cell(unsigned int size){
  this->size = size;
  this->numerals = new unsigned int[size];
  this->value = 0;
  this->path_size = (int)sqrt(this->size);
  this->path = new unsigned int[path_size];
}

__host__ __device__ unsigned int Cell::getSize(){
  return this->size;
}

__host__ __device__ unsigned int Cell::getNextNumeral(int num_ind){
  for (int i = num_ind + 1; i < this->size; i++){
	if (this->numerals[i] == 1)
		return this->numerals[i];
  }

}

template<class t>
__host__ t *Cell::allocateHost(unsigned int size){
	t *aValue;
	cudaMallocHost((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}

template<class t>
__host__ t *Cell::allocateDevice(unsigned int size){
	t *aValue;
	cudaMallocHost((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}
