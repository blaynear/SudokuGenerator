#ifndef PUZZLE_H
#define PUZZLE_H
#include <inttypes.h>
#include "../Cell/cell.cuh"
using namespace std;

class Puzzle{

  unsigned int dim; //Size N of Sudoku puzzle
  Cell *grid; //The N x N cell Sudoku cell grid

  template<class t>
  __host__ t *allocateHost(unsigned int size);
  template<class t>
  __host__ __device__ t *allocateDevice(unsigned int size);
  __host__ __device__ unsigned int getThreadCount(int offset);
  __host__ __device__ unsigned int getBlockCount(int offset, int numberOfThreads);
  __host__ __device__ unsigned int getRow(int index, int dim);
  __host__ __device__ unsigned int getCol(int index, int dim);
  __host__ __device__ unsigned int getNum(int index, int dim);
  __host__ __device__ unsigned int getSquare(int index, int dim);
  __host__ __device__ unsigned int get2DIndex(int x, int y, int dim);
  __host__ __device__ unsigned int get3DIndex(int x, int y, int z, int dim);
public:
  __host__ __device__ unsigned int getSize();
  __host__ __device__ unsigned int getCellSize(unsigned int number);

  __host__ __device__ Puzzle(unsigned int size); //Create generator which specifies a unique puzzle of size N
  __device__ void allocatePuzzle(unsigned int tid);
};

#endif
