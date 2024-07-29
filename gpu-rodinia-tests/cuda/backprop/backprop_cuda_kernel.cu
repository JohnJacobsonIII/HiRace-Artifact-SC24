

#ifndef _BACKPROP_CUDA_KERNEL_H_
#define _BACKPROP_CUDA_KERNEL_H_

#include <stdio.h>
#include "backprop.h"
#include "math.h"
#include "cuda.h"

#include "HiRace.h"


__global__ void
bpnn_layerforward_CUDA(float *_input_cuda,
	                   float *_output_hidden_cuda,
					   float *_input_hidden_cuda,
					   float *_hidden_partial_sum,
					   int in,
					   int hid,
             uint64_cu *__hr_input_cuda,
	           uint64_cu *__hr_output_hidden_cuda,
					   uint64_cu *__hr_input_hidden_cuda,
					   uint64_cu *__hr_hidden_partial_sum)
{
  /**** HIRACE STUFF ****/
  HiRaceDataWrap<float> input_cuda         = _input_cuda;
  HiRaceDataWrap<float> output_hidden_cuda = _output_hidden_cuda;
  HiRaceDataWrap<float> input_hidden_cuda  = _input_hidden_cuda;
  HiRaceDataWrap<float> hidden_partial_sum = _hidden_partial_sum;
   
  input_cuda.setMetadata(__hr_input_cuda);
  output_hidden_cuda.setMetadata(__hr_output_hidden_cuda);
  input_hidden_cuda.setMetadata(__hr_input_hidden_cuda);
  hidden_partial_sum.setMetadata(__hr_hidden_partial_sum);
   
  input_cuda.setScope(Scope::Global);
  output_hidden_cuda.setScope(Scope::Global);
  input_hidden_cuda.setScope(Scope::Global);
  hidden_partial_sum.setScope(Scope::Global);
  
  __syncthreads();
   
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
   
   __shared__ float _input_node[HEIGHT];
   __shared__ float weight_matrix[HEIGHT][WIDTH];

  /**** HIRACE STUFF ****/
  int in_block_tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
  int block_size = blockDim.x * blockDim.y * blockDim.z;
  int tid = in_block_tid
            + (blockIdx.x * block_size) // add a full block for each x step in grid
            + (blockIdx.y * gridDim.x * block_size) // a row of blocks to step in y
            + (blockIdx.z * gridDim.x * gridDim.y * block_size); // a square of blocks to step z
  int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);
  //int lane = tid % 32;
  //int warp = tid / 32;
  
  //printf("tid: %d, bid: %d, tidxyz: %d %d %d, bid xyz: %d %d %d\n", tid, bid, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
  
  HiRaceDataWrap<float> input_node;
  input_node.setData(_input_node); // set data
  input_node.setScope(Scope::Block);
  
  __shared__ uint64_cu *__hr_input_node;
  if (bid == 0) { // only check one block
    if (tid == 0) { // malloc with a single thread
      __hr_input_node = new uint64_cu[HEIGHT];
      if(__hr_input_node == NULL) { printf("HiRace: can't malloc shared metadata\n"); }
    }
  
    __syncthreads();
    
    // initialize the metadata to 0
    unsigned offset = 0;
    do {
      unsigned idx = tid + offset;
      if (idx < HEIGHT) {
        __hr_input_node[tid + offset] = 0;
      }
      offset += block_size;
    }
    while ( HEIGHT < offset );
    
    input_node.setMetadata(__hr_input_node);
    
    __syncthreads();
  }
  #define input_node input_node.registerCallsite(__LINE__,__FILE__)
  
  /**** HIRACE STUFF ****/

   if ( tx == 0 )
   input_node[ty] = input_cuda[index_in] ;
   
   __syncthreads(); input_cuda.incBcount(); output_hidden_cuda.incBcount(); input_hidden_cuda.incBcount(); hidden_partial_sum.incBcount(); input_node.incBcount();

   weight_matrix[ty][tx] = input_hidden_cuda[index];

   __syncthreads(); input_cuda.incBcount(); output_hidden_cuda.incBcount(); input_hidden_cuda.incBcount(); hidden_partial_sum.incBcount(); input_node.incBcount();
   
   weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

   __syncthreads(); input_cuda.incBcount(); output_hidden_cuda.incBcount(); input_hidden_cuda.incBcount(); hidden_partial_sum.incBcount(); input_node.incBcount();
   
   for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){
 
	   int power_two = __powf(2, i);

	   if( ty % power_two == 0 )
	   weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];

	   __syncthreads(); input_cuda.incBcount(); output_hidden_cuda.incBcount(); input_hidden_cuda.incBcount(); hidden_partial_sum.incBcount(); input_node.incBcount();

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

   __syncthreads(); input_cuda.incBcount(); output_hidden_cuda.incBcount(); input_hidden_cuda.incBcount(); hidden_partial_sum.incBcount(); input_node.incBcount();

   if ( tx == 0 ) {
	   hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
   }

   
  /**** HIRACE STUFF ****/
  #undef input_cuda         
  #undef output_hidden_cuda 
  #undef input_hidden_cuda  
  #undef hidden_partial_sum
  #undef input_node
   
  // delete global shadow of shared mem 
  __syncthreads();
  if (tid == 0) { delete(__hr_input_node); } 
  /**** HIRACE STUFF ****/
}


__global__ void bpnn_adjust_weights_cuda(float * _delta,   
										 int hid,         
										 float * _ly,      
										 int in,          
										 float * _w,       
										 float * _oldw,
                     uint64_cu * __hr_delta,	
										 uint64_cu * __hr_ly,      
										 uint64_cu * __hr_w,       
										 uint64_cu * __hr_oldw)
{
  /**** HIRACE STUFF ****/
  HiRaceDataWrap<float> delta = _delta;
  HiRaceDataWrap<float> ly    = _ly;
  HiRaceDataWrap<float> w     = _w;  
  HiRaceDataWrap<float> oldw  = _oldw;
   
  delta.setMetadata(__hr_delta);
  ly.setMetadata(__hr_ly);
  w.setMetadata(__hr_w);
  oldw.setMetadata(__hr_oldw);
   
  delta.setScope(Scope::Global);
  ly.setScope(Scope::Global);
  w.setScope(Scope::Global);
  oldw.setScope(Scope::Global);
  
  __syncthreads();
  
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

   __syncthreads(); delta.incBcount(); ly.incBcount(); w.incBcount(); oldw.incBcount();

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
