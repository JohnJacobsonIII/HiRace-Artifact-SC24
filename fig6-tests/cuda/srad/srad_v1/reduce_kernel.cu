// statistical kernel
__global__ void reduce(	long d_Ne,											// number of elements in array
										int d_no,											// number of sums to reduce
										int d_mul,											// increment
										fp *__hr_d_sums,										// pointer to partial sums variable (DEVICE GLOBAL MEMORY)
										fp *__hr_d_sums2,
										uint64_cu *__hr_metadata_d_sums,										// pointer to partial sums variable (DEVICE GLOBAL MEMORY)
										uint64_cu *__hr_metadata_d_sums2){
  /**** HIRACE STUFF ****/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  HIRACE_WRAP_DATA(fp, d_sums)
  HIRACE_WRAP_DATA(fp, d_sums2)
  
  HIRACE_SET_DATA_GLOBAL(d_sums)
  HIRACE_SET_DATA_GLOBAL(d_sums2)
  
  #define d_sums d_sums.registerCallsite(__LINE__,__FILE__)
  #define d_sums2 d_sums2.registerCallsite(__LINE__,__FILE__)
  /**** HIRACE STUFF ****/

	// indexes
    int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;										// unique thread id, more threads than actual elements !!!
	int nf = NUMBER_THREADS-(gridDim.x*NUMBER_THREADS-d_no);				// number of elements assigned to last block
	int df = 0;																// divisibility factor for the last block

	// statistical
	__shared__ fp __hr_d_psum[NUMBER_THREADS];								// data for block calculations allocated by every block in its shared memory
	__shared__ fp __hr_d_psum2[NUMBER_THREADS];
  
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

  
  HiRaceDataWrap<fp> d_psum(__hr_d_psum);
  __shared__ hr_shadowt* __hr_metadata_d_psum;
  if (__hr_bid == 0) { // only check one block
  if (__hr_tid == 0) { // malloc with a single thread
  __hr_metadata_d_psum = new hr_shadowt[NUMBER_THREADS];
  if(__hr_metadata_d_psum == NULL) { printf("HiRace: can't malloc shared metadata\n"); }
}
__syncthreads();
int __hr_size = NUMBER_THREADS;
// initialize the metadata to 0
unsigned __hr_offset = 0;
for (int i=0;i<__hr_size;i++) {
  unsigned __hr_idx = __hr_in_block_tid + __hr_offset;
  if(__hr_idx < __hr_size) __hr_metadata_d_psum[__hr_idx] = 0;
  __hr_offset += __hr_block_size;
}
__syncthreads();
d_psum.setMembers(__hr_d_psum,
__hr_metadata_d_psum,
Scope::Block,
&__hr_bcount,
&__hr_wcount,
&__hr_swidx,
1,0,0);
}

  
  HiRaceDataWrap<fp> d_psum2(__hr_d_psum2);
  __shared__ hr_shadowt* __hr_metadata_d_psum2;
  if (__hr_bid == 0) { // only check one block
  if (__hr_tid == 0) { // malloc with a single thread
  __hr_metadata_d_psum2 = new hr_shadowt[NUMBER_THREADS];
  if(__hr_metadata_d_psum2 == NULL) { printf("HiRace: can't malloc shared metadata\n"); }
}
__syncthreads();
int __hr_size = NUMBER_THREADS;
// initialize the metadata to 0
unsigned __hr_offset = 0;
for (int i=0;i<__hr_size;i++) {
  unsigned __hr_idx = __hr_in_block_tid + __hr_offset;
  if(__hr_idx < __hr_size) __hr_metadata_d_psum2[__hr_idx] = 0;
  __hr_offset += __hr_block_size;
}
__syncthreads();
d_psum2.setMembers(__hr_d_psum2,
__hr_metadata_d_psum2,
Scope::Block,
&__hr_bcount,
&__hr_wcount,
&__hr_swidx,
1,0,0);
}

#define d_psum d_psum.registerCallsite(__LINE__,__FILE__)
#define d_psum2 d_psum2.registerCallsite(__LINE__,__FILE__)
  /**** HIRACE STUFF ****/

	// counters
	int i;

	// copy data to shared memory
	if(ei<d_no){															// do only for the number of elements, omit extra threads

		d_psum[tx] = d_sums[ei*d_mul];
		d_psum2[tx] = d_sums2[ei*d_mul];

	}

	// Lingjie Zhang modifited at Nov 1 / 2015
    __syncthreads(); __hr_bcount++;
    // end Lingjie Zhang's modification

	// reduction of sums if all blocks are full (rare case)	
	if(nf == NUMBER_THREADS){
		// sum of every 2, 4, ..., NUMBER_THREADS elements
		for(i=2; i<=NUMBER_THREADS; i=2*i){
			// sum of elements
			if((tx+1) % i == 0){											// every ith
				d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
				d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
			}
			// synchronization
			__syncthreads(); __hr_bcount++;
		}
		// final sumation by last thread in every block
		if(tx==(NUMBER_THREADS-1)){											// block result stored in global memory
			d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
			d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
		}
	}
	// reduction of sums if last block is not full (common case)
	else{ 
		// for full blocks (all except for last block)
		if(bx != (gridDim.x - 1)){											//
			// sum of every 2, 4, ..., NUMBER_THREADS elements
			for(i=2; i<=NUMBER_THREADS; i=2*i){								//
				// sum of elements
				if((tx+1) % i == 0){										// every ith
					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
				}
				// synchronization
				__syncthreads(); __hr_bcount++;//
			}
			// final sumation by last thread in every block
			if(tx==(NUMBER_THREADS-1)){										// block result stored in global memory
				d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
				d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
			}
		}
		// for not full block (last block)
		else{																//
			// figure out divisibility
			for(i=2; i<=NUMBER_THREADS; i=2*i){								//
				if(nf >= i){
					df = i;
				}
			}
			// sum of every 2, 4, ..., NUMBER_THREADS elements
			for(i=2; i<=df; i=2*i){											//
				// sum of elements (only busy threads)
				if((tx+1) % i == 0 && tx<df){								// every ith
					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
				}
				// synchronization (all threads)
				__syncthreads(); __hr_bcount++;	//
			}
			// remainder / final summation by last thread
			if(tx==(df-1)){										//
				// compute the remainder and final summation by last busy thread
				for(i=(bx*NUMBER_THREADS)+df; i<(bx*NUMBER_THREADS)+nf; i++){						//
					d_psum[tx] = d_psum[tx] + d_sums[i];
					d_psum2[tx] = d_psum2[tx] + d_sums2[i];
				}
				// final sumation by last thread in every block
				d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
				d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
			}
		}
	}


  #undef d_sums 
  #undef d_sums2
  #undef d_psum
  #undef d_psum2
// delete global shadow of shared mem
__syncthreads();
if (__hr_tid == 0) { delete[] __hr_metadata_d_psum; }
if (__hr_tid == 0) { delete[] __hr_metadata_d_psum2; }
}
