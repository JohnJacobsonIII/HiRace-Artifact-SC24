include ../../rodinia-common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g  -O2

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -lineinfo -g -I$(CUDA_DIR)/include -I../../../src/hirace

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O2
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif

all: backprop backprop_hr

backprop: backprop.o facetrain.o imagenet.o backprop_cuda.o 
	$(CC) $(CC_FLAGS) backprop.o facetrain.o imagenet.o backprop_cuda.o -o backprop -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm -g

backprop_hr: backprop.o facetrain.o imagenet.o backprop_cuda_hr.o 
	$(CC) $(CC_FLAGS) -DRACECHECK backprop.o facetrain.o imagenet.o backprop_cuda.o -o backprop_hirace -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm -g

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_cuda.o: backprop_cuda.cu backprop.h
	$(NVCC) $(NVCC_FLAGS) -c backprop_cuda.cu

backprop_cuda_hr.o: backprop_cuda.cu backprop.h
	$(NVCC) $(NVCC_FLAGS) -DRACECHECK -c backprop_cuda.cu

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop backprop_hirace backprop_cuda.linkinfo
