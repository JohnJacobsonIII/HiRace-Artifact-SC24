// statistical kernel
__global__ void reduce(	long d_Ne,											// number of elements in array
										int d_no,											// number of sums to reduce
										int d_mul,											// increment
										fp *_d_sums,										// pointer to partial sums variable (DEVICE GLOBAL MEMORY)
										fp *_d_sums2,
										uint64_cu *__hr_d_sums,										// pointer to partial sums variable (DEVICE GLOBAL MEMORY)
										uint64_cu *__hr_d_sums2){
  /**** HIRACE STUFF ****/
  HiRaceDataWrap<fp> d_sums      = _d_sums ;
  HiRaceDataWrap<fp> d_sums2     = _d_sums2;
   
  d_sums.setMetadata(__hr_d_sums);
  d_sums2.setMetadata(__hr_d_sums2);
   
  d_sums.setScope(Scope::Global);
  d_sums2.setScope(Scope::Global);
  
  __syncthreads();
   
  #define d_sums          d_sums.registerCallsite(__LINE__,__FILE__)
  #define d_sums2         d_sums2.registerCallsite(__LINE__,__FILE__)
  /**** HIRACE STUFF ****/

	// indexes
    int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;										// unique thread id, more threads than actual elements !!!
	int nf = NUMBER_THREADS-(gridDim.x*NUMBER_THREADS-d_no);				// number of elements assigned to last block
	int df = 0;																// divisibility factor for the last block

	// statistical
	__shared__ fp _d_psum[NUMBER_THREADS];								// data for block calculations allocated by every block in its shared memory
	__shared__ fp _d_psum2[NUMBER_THREADS];
  
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
  
  HiRaceDataWrap<fp> d_psum;
  d_psum.setData(_d_psum); // set data
  d_psum.setScope(Scope::Block);
  
  __shared__ uint64_cu *__hr_d_psum;
  if (bid == 0) { // only check one block
    if (tid == 0) { // malloc with a single thread
      __hr_d_psum = new uint64_cu[NUMBER_THREADS];
      if(__hr_d_psum == NULL) { printf("HiRace: can't malloc shared metadata\n"); }
    }
  
    __syncthreads();
    
    // initialize the metadata to 0
    unsigned offset = 0;
    do {
      unsigned idx = tid + offset;
      if (idx < NUMBER_THREADS) {
        __hr_d_psum[tid + offset] = 0;
      }
      offset += block_size;
    }
    while ( NUMBER_THREADS < offset );
    
    d_psum.setMetadata(__hr_d_psum);
    
    __syncthreads();
  }
  #define d_psum d_psum.registerCallsite(__LINE__,__FILE__)
 
  
  
  HiRaceDataWrap<fp> d_psum2;
  d_psum2.setData(_d_psum2); // set data
  d_psum2.setScope(Scope::Block);
  
  __shared__ uint64_cu *__hr_d_psum2;
  if (bid == 0) { // only check one block
    if (tid == 0) { // malloc with a single thread
      __hr_d_psum2 = new uint64_cu[NUMBER_THREADS];
      if(__hr_d_psum2 == NULL) { printf("HiRace: can't malloc shared metadata\n"); }
    }
  
    __syncthreads();
    
    // initialize the metadata to 0
    unsigned offset = 0;
    do {
      unsigned idx = tid + offset;
      if (idx < NUMBER_THREADS) {
        __hr_d_psum2[tid + offset] = 0;
      }
      offset += block_size;
    }
    while ( NUMBER_THREADS < offset );
    
    d_psum2.setMetadata(__hr_d_psum2);
    
    __syncthreads();
  }
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
    __syncthreads(); d_sums.incBcount(); d_sums2.incBcount(); d_psum.incBcount(); d_psum2.incBcount();
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
			__syncthreads(); d_sums.incBcount(); d_sums2.incBcount(); d_psum.incBcount(); d_psum2.incBcount();
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
				__syncthreads(); d_sums.incBcount(); d_sums2.incBcount(); d_psum.incBcount(); d_psum2.incBcount();											//
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
				__syncthreads(); d_sums.incBcount(); d_sums2.incBcount(); d_psum.incBcount(); d_psum2.incBcount();											//
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
}
