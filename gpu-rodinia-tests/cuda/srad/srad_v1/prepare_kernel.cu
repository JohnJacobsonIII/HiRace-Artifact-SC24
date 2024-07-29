// statistical kernel
__global__ void prepare(	long d_Ne,
											fp *_d_I,										// pointer to output image (DEVICE GLOBAL MEMORY)
											fp *_d_sums,									// pointer to input image (DEVICE GLOBAL MEMORY)
											fp *_d_sums2,
											uint64_cu *__hr_d_I,										// pointer to output image (DEVICE GLOBAL MEMORY)
											uint64_cu *__hr_d_sums,									// pointer to input image (DEVICE GLOBAL MEMORY)
											uint64_cu *__hr_d_sums2){
  /**** HIRACE STUFF ****/
  HiRaceDataWrap<fp> d_I         = _d_I;
  HiRaceDataWrap<fp> d_sums      = _d_sums ;
  HiRaceDataWrap<fp> d_sums2     = _d_sums2;
   
  d_I.setMetadata(__hr_d_I);
  d_sums.setMetadata(__hr_d_sums);
  d_sums2.setMetadata(__hr_d_sums2);
   
  d_I.setScope(Scope::Global);
  d_sums.setScope(Scope::Global);
  d_sums2.setScope(Scope::Global);
  
  __syncthreads();
   
  #define d_I             d_I.registerCallsite(__LINE__,__FILE__)
  #define d_sums          d_sums.registerCallsite(__LINE__,__FILE__)
  #define d_sums2         d_sums2.registerCallsite(__LINE__,__FILE__)
  /**** HIRACE STUFF ****/

	// indexes
	int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;										// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei<d_Ne){															// do only for the number of elements, omit extra threads

		d_sums[ei] = d_I[ei];
		d_sums2[ei] = d_I[ei]*d_I[ei];

	}

  #undef d_I    
  #undef d_sums 
  #undef d_sums2

}
