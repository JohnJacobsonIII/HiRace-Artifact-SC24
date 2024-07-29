/* This file is part of the Indigo benchmark suite version 1.0.

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

__global__ void test_kernel(int* nindex, int* nlist, data_t* data1, data_t* data2, int numv)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int beg = nindex[i];
  int end = nindex[i + 1];
  for (int j = beg; j < end; j++) {
    int nei = nlist[j];
    if (i < nei) {
      data_t pos = atomicAdd(data2, (data_t)1);
      data1[pos] = (data_t)j;
      break;
    }
  }
}

void serial_code(int* nindex, int* nlist, data_t* data1, data_t* data2, int numv)
{
  for (int i = 0; i < numv; i++) {
    int beg = nindex[i];
    int end = nindex[i + 1];
    for (int j = beg; j < end; j++) {
      int nei = nlist[j];
      if (i < nei) {
        data_t pos = data2[0]++;
        data1[pos] = (data_t)j;
        break;
      }
    }
  }
}

int compare(const void* a, const void* b)
{
  if (*(data_t*)a < *(data_t*)b) return -1;
  if (*(data_t*)a == *(data_t*)b) return 0;
  return 1;
}

int verify_result(int* nindex, int* nlist, data_t* h_data1, data_t* h_data2, data_t* d_data1, data_t* d_data2, int numv, int nume, int blocks, int threadsperblock)
{
  if (numv > blocks * threadsperblock) {
    printf("Error: too few threads\n");
    return -1;
  }
  if (h_data2[0] != d_data2[0]) {
    return 0;
  }

  qsort(h_data1, h_data2[0], sizeof(data_t), compare);
  qsort(d_data1, d_data2[0], sizeof(data_t), compare);

  for (int i = 0; i < h_data2[0]; i++) {
    if (h_data1[i] != d_data1[i]) {
      return 0;
    }
  }
  return 1;
}