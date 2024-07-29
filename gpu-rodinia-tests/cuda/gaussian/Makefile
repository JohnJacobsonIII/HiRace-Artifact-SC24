include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -I/uufs/chpc.utah.edu/common/home/u1201441/repos/hirace/src/hirace -DRACECHECK 

clang: $(SRC)
	clang++ $(SRC) -o $(EXE) -I../util --cuda-gpu-arch=sm_20 \
		-L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -DTIMING

clean:
	rm gaussian
