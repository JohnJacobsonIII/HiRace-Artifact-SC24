include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
a.out:		main.o
	$(CC)	main.o \
				-I$(INCLUDE) \
				-I/uufs/chpc.utah.edu/common/home/u1201441/repos/hirace/src/hirace -DRACECHECK \
				-L$(CUDA_LIB_DIR)  \
				-lm -lcuda -lcudart -o srad

# compile main function file into object (binary)
main.o: 	main.cu \
				define.c \
				graphics.c \
				extract_kernel.cu \
				prepare_kernel.cu \
				reduce_kernel.cu \
				srad_kernel.cu \
				srad2_kernel.cu \
				compress_kernel.cu
	nvcc	main.cu \
				-I/uufs/chpc.utah.edu/common/home/u1201441/repos/hirace/src/hirace -DRACECHECK \
				-c -O3 -arch sm_35

# delete all object files
clean:
	rm *.o srad
