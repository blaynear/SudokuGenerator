#include "generator.cuh"
#include "cell.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include "math.h"

using namespace std;

const int baseThreadCount = 64;
/***************************************************************************/
/*   											      Cuda Kernels   														 */
/***************************************************************************/
__global__ void setPuzzle(unsigned int *size, Puzzle *aPuzzle){
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if(tid == 0){
    aPuzzle[0] = Puzzle(size[0]);
  }
  __syncthreads();
  while(tid < pow((double)size[0], 2)){
    aPuzzle[0].allocatePuzzle(tid);
    tid++;
  }
}
/***************************************************************************/
/*   														Constructors   														 */
/***************************************************************************/

__host__ Generator::Generator(unsigned int size){
  unsigned int *d_size = allocateDevice<unsigned int>(1);
  this->d_puzzle = allocateDevice<Puzzle>(1);
  
  cudaMemcpy(d_size, &size, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaCheckError();
  setPuzzle<<<1,this->blockSize>>>(d_size, d_puzzle);

  /***********************************************************/
  /*                    From here, implement:
  /*                       Clue selection
  /*                       Extermination
  /***********************************************************/
  this->blockSize = getThreadCount(size * size);
  this->gridSize = getBlockCount(size * size, this->blockSize);
}

__host__ bool Generator::validate(int i, int j, int x, Cell **grid) {
    int x_sqrt = (int)sqrt(x);
    int target = grid[i][j].value;
    for (int k = 0; k < x; k++) {
        if (k != j && target == grid[i][k].value) {
            return false;
        }
        if (k != i && target == grid[k][j].value) {
            return false;
        }
    }

    int div_row = (i + 1) / x_sqrt;
    if ((i + 1) % x_sqrt == 0)
        div_row -= 1;

    int div_col = (j + 1) / x_sqrt;
    if ((j + 1) % x_sqrt == 0)
        div_col -= 1;

    for (int k = div_row * x_sqrt; k < (div_row * x_sqrt + x_sqrt); k++) {
        for (int w = div_col * x_sqrt; w < (div_col * x_sqrt + x_sqrt); w++) {
            if ((k != i || w != j) && grid[k][w].value == target) {
                return false;
            }
        }
    }
    return true;
}


__host__ void Generator::exterminate(int i, int j, int x, Cell **d) {

    int x_sqrt = sqrt(x);

    for (int k = 0; k < x; k++) {
        if (d[i][k].value != 0) 
            d[i][j].numerals[d[i][k].value - 1] = 1;
        if (d[k][j] != 0)
            d[i][j].numerals[d[k][j].value - 1] = 1;
    }

    int div_row = (i + 1) / x_sqrt;
    if ((i + 1) % x_sqrt == 0)
        div_row -= 1;

    int div_col = (j + 1) / x_sqrt;
    if ((j + 1) % x_sqrt == 0)
        div_col -= 1;

    for (int k = div_row * x_sqrt; k < div_row * x_sqrt + x_sqrt; k++) {
        for (int w = div_col * x_sqrt; w < div_col * x_sqrt + x_sqrt; w++) {
            if (d[k][w].value != 0) {
                d[i][j].numerals[d[k][w].value - 1] = 1;
            }
        }
    }
}

__host__ unsigned int Generator::readFile(){
    string s;
    cout << "Enter Filename: ";
    cin >> s;
    ifstream myfile(s);
    int row, num_clue;
    myfile >> row >> num_clue;
    Cell** d = new Cell*[row];

    for (int i = 0; i < row; i++){
        d[i] = new int[row];
        for (int j = 0; j < row; j++)
            d[i][j] = new Cell(row);
            for (int start = 0; start < row; start++)
              d[i][j].numerals[start] = 0;
    }

    string line1;
    getline(myfile, line1); 

    while (getline(myfile, line1) && line1 != "") {
        int r, c, clue;
        sscanf(line1.c_str(), "%d %d %d", &r, &c, &clue);
        d[r - 1][c - 1].value = clue;
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < row; j++) {
            if ( d[i][j].value != 0 && !validate(i, j, row, d)){
                cout << "Given Sudoku is not correct" << endl;
                return 1;
            }
        }
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < row; j++) {
            if ( d[i][j].value == 0){
                exterminate(i, j, row, d);
            }
        }
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < row; j++) {
            cout << d[i][j].value << "\t";
        }
        cout << endl;
    }
    return 0;
}


__host__ unsigned int Generator::initialize(){

  string user_input;
  cout << "Enter 1 to import a puzzle";
  cin >> user_input;
  if (user_input == "1")
    readFile();
  return 0;
}

/***************************************************************************/
/*														Getter Functions														 */
/***************************************************************************/

/***************************************************************************/
/*														Setter Functions														 */
/***************************************************************************/

/***************************************************************************/
/*														CUDA Helper Masks														 */
/***************************************************************************/
template<class t>
__host__ t *Generator::allocateHost(unsigned int size){
	t *aValue;
	cudaMallocHost((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}

template<class t>
__host__ t *Generator::allocateDevice(unsigned int size){
	t *aValue;
	cudaMalloc((void **)&aValue, size * sizeof(t));
	cudaCheckError();
	return aValue;
}

__host__ __device__ unsigned int Generator::getThreadCount(unsigned int offset){
	return min(((offset / 512) + 1)*baseThreadCount, 512);
}

__host__ __device__ unsigned int Generator::getBlockCount(unsigned int offset, unsigned int numberOfThreads){
	return (offset / numberOfThreads) + 1;
}
