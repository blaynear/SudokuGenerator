#ifndef GENERATOR_H
#define GENERATOR_H
#include <inttypes.h>
#include "../Puzzle/puzzle.cuh"
using namespace std;

class Generator{

  unsigned int gridSize; //Number of blocks to launch
  unsigned int blockSize; //Number of threads per block
  Puzzle *h_puzzle; //For testing
  Puzzle *d_puzzle; //aPuzzle allocated

  template<class t>
  __host__ t *allocateHost(unsigned int size);
  template<class t>
  __host__ t *allocateDevice(unsigned int size);
  __host__ __device__ unsigned int getThreadCount(unsigned int offset);
  __host__ __device__ unsigned int getBlockCount(unsigned int offset, unsigned int numberOfThreads);
public:
  __host__ Generator(unsigned int size);
  __host__ unsigned int initialize();
  __host__ unsigned int readFile();
  __host__ bool validate(int i, int j, int x, Cell **grid)
  __host__ void exterminate(int i, int j, int x, Cell **grid)
  __host__ void clueSelection(int x, Cell **grid)
  __host__ unsigned int getNextCell(int cell_ind);
};

#endif
