#include "puzzle.cuh"

const int baseThreadCount = 64;

/***************************************************************************/
/*   											      Cuda Kernels   														 */
/***************************************************************************/

/***************************************************************************/
/*   														Constructors   														 */
/***************************************************************************/
__device__ void Puzzle::allocatePuzzle(unsigned int tid){
	this->grid[tid] = Cell(this->dim);
}

__host__ __device__ Puzzle::Puzzle(unsigned int size){
	this->dim = size;
	this->grid = (Cell *)malloc(size * size * sizeof(Cell));
}

/***************************************************************************/
/*														Getter Functions														 */
/***************************************************************************/
__host__ __device__ unsigned int Puzzle::getSize(){
  return this->dim;
}

__host__ __device__ unsigned int Puzzle::getCellSize(unsigned int number){
	return this->grid[number].getSize();
}
/***************************************************************************/
/*														Setter Functions														 */
/***************************************************************************/

/***************************************************************************/
/*														CUDA Helper Masks														 */
/***************************************************************************/
template<class t>
__host__ t *Puzzle::allocateHost(unsigned int size){
	t *aValue;
	cudaMallocHost((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}

template<class t>
__host__ __device__ t *Puzzle::allocateDevice(unsigned int size){
	t *aValue;
	cudaMalloc((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}

__host__ __device__ unsigned int Puzzle::getThreadCount(int offset){
	return min(((offset / 512) + 1)*baseThreadCount, 512);
}

__host__ __device__ unsigned int Puzzle::getBlockCount(int offset, int numberOfThreads){
	return (offset / numberOfThreads) + 1;
}

__host__ __device__ unsigned int Puzzle::getRow(int index, int dim){
	return index % dim;
}

__host__ __device__ unsigned int Puzzle::getCol(int index, int dim){
	return (index / dim) % dim;
}

__host__ __device__ unsigned int Puzzle::getNum(int index, int dim){
	return index / (dim*dim);
}

__host__ __device__ unsigned int Puzzle::getSquare(int index, int dim){
	return get2DIndex(getRow(index, dim) / sqrtf((double)dim), getCol(index, dim) / sqrtf((double)dim), sqrtf((double)dim));
}

__host__ __device__ unsigned int Puzzle::get3DIndex(int x, int y, int z, int dim) {
	return x + dim*(y + dim*z);
}

__host__ __device__ unsigned int Puzzle::get2DIndex(int x, int y, int dim) {
	return x + dim*y;
}
