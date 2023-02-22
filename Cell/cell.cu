#include "cell.cuh"

__host__ __device__ Cell::Cell(unsigned int size){
  this->size = size;
  this->numerals = new unsigned int[size];
  this->value = 0;
}

__host__ __device__ unsigned int Cell::getSize(){
  return this->size;
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
/*************************************************/
/*Helper Functions. Maybe create a misc file to consolidate */
/*************************************************/
