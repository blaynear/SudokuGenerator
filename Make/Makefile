# gnu makefile for CUDA compiles
# using as simple pattern rule for .cu files
#NVCC= /opt/nvidia/cuda-6.5/bin/nvcc
#-gencode=arch=compute_75,code=sm_75 -rdc=true required for CDP
NVCC = nvcc -g -m64 -lineinfo -gencode=arch=compute_86,code=sm_86 -rdc=true -Xptxas -O3
LIBS= -lcublas
MAIN = main
CELL = cell
PUZZLE = puzzle
GENERATOR = generator
SUDOKU = SudokuGen

# the build target executable:
$(SUDOKU): $(MAIN).o $(CELL).o $(PUZZLE).o $(GENERATOR).o
	$(NVCC) $(LIBS) -o $(SUDOKU) $(CELL).o $(PUZZLE).o $(MAIN).o $(GENERATOR).o

$(MAIN).o: ../Main/$(MAIN).cu
	$(NVCC) -c $(FLAGS) ../Main/$(MAIN).cu

$(CELL).o: ../Cell/$(CELL).cu
	$(NVCC) $(LIBS) -c ../Cell/$(CELL).cu

$(PUZZLE).o: ../Puzzle/$(PUZZLE).cu
	$(NVCC) $(LIBS) -c ../Puzzle/$(PUZZLE).cu

$(GENERATOR).o: ../Generator/$(GENERATOR).cu
	$(NVCC) $(LIBS) -c ../Generator/$(GENERATOR).cu

clean:
	rm *.o

ARGS = "default"

.PHONY: run
run:
		compute-sanitizer ./$(SUDOKU)
