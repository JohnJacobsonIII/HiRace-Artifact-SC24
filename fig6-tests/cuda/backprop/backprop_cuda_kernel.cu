

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"

#include "HiRace.h"


__global__ void
bpnn_layerforward_CUDA(float *__hr_input_cuda,
	                   float *__hr_output_hidden_cuda,
					   float *__hr_input_hidden_cuda,
					   float *__hr_hidden_partial_sum,
					   int in,
					   int hid,
             uint64_cu *__hr_metadata_input_cuda,
	           uint64_cu *__hr_metadata_output_hidden_cuda,
					   uint64_cu *__hr_metadata_input_hidden_cuda,
					   uint64_cu *__hr_metadata_hidden_partial_sum)
{
  /**** HIRACE STUFF ****/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  HIRACE_WRAP_DATA(float,input_cuda)
  HIRACE_WRAP_DATA(float,output_hidden_cuda)
  HIRACE_WRAP_DATA(float,input_hidden_cuda)
  HIRACE_WRAP_DATA(float,hidden_partial_sum)

  HIRACE_SET_DATA_GLOBAL(input_cuda)
  HIRACE_SET_DATA_GLOBAL(output_hidden_cuda)
  HIRACE_SET_DATA_GLOBAL(input_hidden_cuda)
  HIRACE_SET_DATA_GLOBAL(hidden_partial_sum)
   
  #define input_cuda         input_cuda.registerCallsite(__LINE__,__FILE__)
  #define output_hidden_cuda output_hidden_cuda.registerCallsite(__LINE__,__FILE__)
  #define input_hidden_cuda  input_hidden_cuda.registerCallsite(__LINE__,__FILE__)
  #define hidden_partial_sum hidden_partial_sum.registerCallsite(__LINE__,__FILE__)
  /**** HIRACE STUFF ****/
  
   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  

   int index_in = HEIGHT * by + ty + 1;
   
   __shared__ float __hr_input_node[HEIGHT];
   __shared__ float __hr_weight_matrix[HEIGHT][WIDTH];

  /**** HIRACE STUFF ****/
  int __hr_in_block_tid = threadIdx.x
         + threadIdx.y * blockDim.x
         + threadIdx.z * (blockDim.x * blockDim.y);
  int __hr_block_size = blockDim.x * blockDim.y * blockDim.z;
  int __hr_tid = __hr_in_block_tid
            + (blockIdx.x * __hr_block_size) // add a full block for each x step in grid
            + (blockIdx.y * gridDim.x * __hr_block_size) // a row of blocks to step in y
            + (blockIdx.z * gridDim.x * gridDim.y * __hr_block_size); // a square of blocks to step z
  int __hr_bid = blockIdx.x
         + blockIdx.y * gridDim.x
         + blockIdx.z * (gridDim.x * gridDim.y);
  HiRaceDataWrap<float> input_node(__hr_input_node);
  HiRaceDataWrap2D<float, HEIGHT, WIDTH> weight_matrix(__hr_weight_matrix);
  __shared__ uint64_cu *__hr_metadata_input_node;
  __shared__ uint64_cu (*__hr_metadata_weight_matrix)[WIDTH];
  
  if (__hr_bid == 0) { // only check one block
    if (__hr_tid == 0) { // malloc with a single thread
      __hr_metadata_input_node = new hr_shadowt[HEIGHT];
      if(__hr_metadata_input_node == NULL) { printf("HiRace: can't malloc shared metadata\n"); }
      
      __hr_metadata_weight_matrix = new hr_shadowt[HEIGHT][WIDTH];
      if(__hr_metadata_weight_matrix == NULL) { printf("HiRace: can't malloc shared metadata\n"); }
      
    }
  
    // initialize the metadata to 0
    int __hr_size = HEIGHT;
    unsigned __hr_offset = 0;
    for (int i=0;i<__hr_size;i++) {
      unsigned __hr_idx = __hr_in_block_tid + __hr_offset;
      if(__hr_idx < __hr_size)
        __hr_metadata_input_node[__hr_idx] = 0;
      __hr_offset += __hr_block_size;
    }
    __syncthreads();
  
    
    // initialize the metadata to 0
    for (int i=0;i<HEIGHT;i++) {
      unsigned offset = 0;
      do {
        unsigned idx = __hr_in_block_tid + offset;
        if (idx < WIDTH) {
          __hr_metadata_weight_matrix[i][idx] = 0;
        }
        offset += __hr_block_size;
      }
      while ( offset < WIDTH );
    }
    
    __syncthreads();

    input_node.setMembers(__hr_input_node,
                       __hr_metadata_input_node,
                       Scope::Block,
                       &__hr_bcount,
                       &__hr_wcount,
                       &__hr_swidx,
                       1,0,0);
    
    weight_matrix.setMembers(__hr_weight_matrix,
                    __hr_metadata_weight_matrix,
                    Scope::Block,
                    &__hr_bcount,
                    &__hr_wcount,
                    &__hr_swidx,
                    2,0);
  }
  #define input_node input_node.registerCallsite(__LINE__,__FILE__)
  #define weight_matrix weight_matrix.registerCallsite(__LINE__,__FILE__)
  
  /**** HIRACE STUFF ****/

   if ( tx == 0 )
   input_node[ty] = input_cuda[index_in] ;
   
   __syncthreads(); __hr_bcount++;

   weight_matrix[ty][tx] = input_hidden_cuda[index];

   __syncthreads(); __hr_bcount++;
   
   weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

   __syncthreads(); __hr_bcount++;
   
   for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){
 
	   int power_two = __powf(2, i);

	   if( ty % power_two == 0 )
	   weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];

     __syncthreads(); __hr_bcount++;

   }
   
   //__syncthreads();

   input_hidden_cuda[index] = weight_matrix[ty][tx];
   
/*
   for ( unsigned int i = 2 ; i <= HEIGHT ; i *= 2){
 
	   unsigned int power_two = i - 1;

	   if( (ty & power_two) == 0 ) {
		weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
	   }

   }
   */

   __syncthreads(); __hr_bcount++;

   if ( tx == 0 ) {
	   hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
   }

   
  /**** HIRACE STUFF ****/
  #undef input_cuda         
  #undef output_hidden_cuda 
  #undef input_hidden_cuda  
  #undef hidden_partial_sum
  #undef input_node
  #undef weight_matrix
   
  // delete global shadow of shared mem 
  __syncthreads();
  if (__hr_tid == 0) { delete[] __hr_metadata_input_node; } 
  if (__hr_tid == 0) { delete[] __hr_metadata_weight_matrix; } 
  /**** HIRACE STUFF ****/
}


__global__ void bpnn_adjust_weights_cuda(float * __hr_delta,   
										 int hid,         
										 float * __hr_ly,      
										 int in,          
										 float * __hr_w,       
										 float * __hr_oldw,
                     uint64_cu * __hr_metadata_delta,	
										 uint64_cu * __hr_metadata_ly,      
										 uint64_cu * __hr_metadata_w,       
										 uint64_cu * __hr_metadata_oldw)
{
  /**** HIRACE STUFF ****/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  HIRACE_WRAP_DATA(float,delta)
  HIRACE_WRAP_DATA(float,ly)
  HIRACE_WRAP_DATA(float,w)
  HIRACE_WRAP_DATA(float,oldw)

  HIRACE_SET_DATA_GLOBAL(delta)
  HIRACE_SET_DATA_GLOBAL(ly)
  HIRACE_SET_DATA_GLOBAL(w)
  HIRACE_SET_DATA_GLOBAL(oldw)
  
  #define delta delta.registerCallsite(__LINE__,__FILE__)
  #define ly    ly.registerCallsite(__LINE__,__FILE__)
  #define w     w.registerCallsite(__LINE__,__FILE__)
  #define oldw  oldw.registerCallsite(__LINE__,__FILE__)
  /**** HIRACE STUFF ****/
 
   
  
   int by = blockIdx.y;

   int tx = threadIdx.x;
   int ty = threadIdx.y;
	
   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
   int index_y = HEIGHT * by + ty + 1;
   int index_x = tx + 1;
   //eta = 0.3;
   //momentum = 0.3;
   w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
   oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

   __syncthreads(); __hr_bcount++;

   if (ty == 0 && by ==0){
   w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
   }

  /**** HIRACE STUFF ****/
  #undef delta
  #undef ly      
  #undef w       
  #undef oldw    
  /**** HIRACE STUFF ****/

}
#endif 
