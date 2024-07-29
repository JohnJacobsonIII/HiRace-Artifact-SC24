/* This file is part of the Indigo benchmark suite version 1.1.

Copyright 2022, Texas State University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Contributors: Yiqian Liu, Noushin Azami, Corbin Walters, and Martin Burtscher

URL: The latest version of the Indigo benchmark suite is available at
https://cs.txstate.edu/~burtscher/research/IndigoSuite/.
 */

typedef int data_t;
#include "indigo_cuda.h"

__global__ void test_kernel(int* nindex, int* nlist, data_t* __hr_data1, data_t* __hr_data2, int numv, hr_shadowt* __hr_metadata_data1, hr_shadowt* __hr_metadata_data2)
{
  /************************/
  /***** HIRACE START *****/
  /************************/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  HIRACE_WRAP_DATA(data_t,data1)
  HIRACE_WRAP_DATA(data_t,data2)
  HIRACE_SET_DATA_GLOBAL(data1)
  HIRACE_SET_DATA_GLOBAL(data2)
  #define data1 data1.registerCallsite(__LINE__,__FILE__)
  #define data2 data2.registerCallsite(__LINE__,__FILE__)
  /************************/
  /***** HIRACE END *****/
  /************************/
  __shared__ data_t __hr_s_carry[32];
  /************************/
  /***** HIRACE START *****/
  /************************/
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
  HiRaceDataWrap<int> s_carry(__hr_s_carry);
  __shared__ hr_shadowt* __hr_metadata_s_carry;
  if (__hr_bid == 0) { // only check one block
  if (__hr_tid == 0) { // malloc with a single thread
  __hr_metadata_s_carry = new hr_shadowt[32];
  if(__hr_metadata_s_carry == NULL) { printf("HiRace: can't malloc shared metadata\n"); }
}
__syncthreads();
int __hr_size = 32;
// initialize the metadata to 0
unsigned __hr_offset = 0;
for (int i=0;i<__hr_size;i++) {
  unsigned __hr_idx = __hr_in_block_tid + __hr_offset;
  if(__hr_idx < __hr_size) __hr_metadata_s_carry[__hr_idx] = 0;
  __hr_offset += __hr_block_size;
}
__syncthreads();
s_carry.setMembers(__hr_s_carry,
__hr_metadata_s_carry,
Scope::Block,
&__hr_bcount,
&__hr_wcount,
&__hr_swidx,
1,0,0);
}

#define s_carry s_carry.registerCallsite(__LINE__,__FILE__)
/************************/
/***** HIRACE END *****/
/************************/
int lane = threadIdx.x % 32;
int warp = threadIdx.x / 32;
if (warp == 0) s_carry[lane] = 0;
__syncthreads();

for (int i = blockIdx.x; i < numv; i += gridDim.x) {
int beg = nindex[i];
int end = nindex[i + 1];
data_t val = 0;
for (int j = beg + threadIdx.x; j < end; j += blockDim.x) {
  int nei = nlist[j];
  val = max(val, data2[nei]);
}
val = max(val, __shfl_xor_sync(~0, val, 1));
val = max(val, __shfl_xor_sync(~0, val, 2));
val = max(val, __shfl_xor_sync(~0, val, 4));
val = max(val, __shfl_xor_sync(~0, val, 8));
val = max(val, __shfl_xor_sync(~0, val, 16));
if (lane == 0) s_carry[warp] = val;
__syncthreads();

if (warp == 0) {
  val = s_carry[lane];
  val = max(val, __shfl_xor_sync(~0, val, 1));
  val = max(val, __shfl_xor_sync(~0, val, 2));
  val = max(val, __shfl_xor_sync(~0, val, 4));
  val = max(val, __shfl_xor_sync(~0, val, 8));
  val = max(val, __shfl_xor_sync(~0, val, 16));
  if (lane == 0) {
    data1[0] = max(data1[0], val);
  }
}
}
/************************/
/***** HIRACE START *****/
/************************/
#undef data1
#undef data2
#undef s_carry
// delete global shadow of shared mem
__syncthreads();
if (__hr_tid == 0) { delete[] __hr_metadata_s_carry; }
/************************/
/***** HIRACE END *****/
/************************/
}

void serial_code(int* nindex, int* nlist, data_t* data1, data_t* data2, int numv)
{
for (int i = 0; i < numv; i++) {
int beg = nindex[i];
int end = nindex[i + 1];
for (int j = beg; j < end; j++) {
  int nei = nlist[j];
  data1[0] = max(data1[0], data2[nei]);
}
}
}

int verify_result(int* nindex, int* nlist, data_t* h_data1, data_t* h_data2, data_t* d_data1, data_t* d_data2, int numv, int nume, int blocks, int threadsperblock)
{
if (threadsperblock % 32 != 0) {
printf("Error: partial warps not supported\n");
return -1;
}
if (h_data1[0] == d_data1[0]) {
return 1;
} else {
return 0;
}
}
