include ../../../rodinia-common/make.config

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

all: srad hr_srad

# link objects(binaries) together
srad:		main.o
	$(CC)	main.o \
				-I$(INCLUDE) \
				-I../../../../src/hirace \
				-L$(CUDA_LIB_DIR)  \
				-lm -lcuda -lcudart -o srad

hr_srad: main.o
	$(CC)	main.o \
				-I$(INCLUDE) \
				-I../../../../src/hirace -DRACECHECK \
				-L$(CUDA_LIB_DIR)  \
				-lm -lcuda -lcudart -o hr_srad


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
				-I../../../../src/hirace -DRACECHECK \
				-c -O3 -arch sm_35

# delete all object files
clean:
	rm *.o srad hr_srad
