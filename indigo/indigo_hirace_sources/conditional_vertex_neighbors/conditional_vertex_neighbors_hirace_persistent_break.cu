/* This file is part of the Indigo benchmark suite version 1.3.

BSD 3-Clause License

Copyright (c) 2022-2024, Yiqian Liu, Noushin Azami, Corbin Walters, Avery Vanausdal, and Martin Burtscher.

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

URL: The latest version of the Indigo benchmark suite is available at https://cs.txstate.edu/~burtscher/research/IndigoSuite/ and at https://github.com/burtscher/IndigoSuite/.

Publication: This work is described in detail in the following paper.
Yiqian Liu, Noushin Azami, Corbin Walters, and Martin Burtscher. The Indigo Program-Verification Microbenchmark Suite of Irregular Parallel Code Patterns. Proceedings of the 2022 IEEE International Symposium on Performance Analysis of Systems and Software, pp. 24-34. May 2022.

Sponsor: This benchmark suite is based upon work supported by the U.S. National Science Foundation under Grant No. 1955367 as well as by equipment donations from NVIDIA Corporation.
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
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = idx; i < numv; i += gridDim.x * blockDim.x) {
    int beg = nindex[i];
    int end = nindex[i + 1];
    for (int j = beg; j < end; j++) {
      int nei = nlist[j];
      if (i < nei) {
        atomicAdd(data1, (data_t)1);
        break;
      }
    }
  }
  /************************/
  /***** HIRACE START *****/
  /************************/
  #undef data1
  #undef data2
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
      if (i < nei) {
        data1[0]++;
        break;
      }
    }
  }
}

int verify_result(int* nindex, int* nlist, data_t* h_data1, data_t* h_data2, data_t* d_data1, data_t* d_data2, int numv, int nume, int blocks, int threadsperblock)
{
  if (h_data1[0] == d_data1[0]) {
    return 1;
  } else {
    return 0;
  }
}
