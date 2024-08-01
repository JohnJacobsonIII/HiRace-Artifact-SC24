// statistical kernel
__global__ void prepare(	long d_Ne,
											fp *__hr_d_I,										// pointer to output image (DEVICE GLOBAL MEMORY)
											fp *__hr_d_sums,									// pointer to input image (DEVICE GLOBAL MEMORY)
											fp *__hr_d_sums2,
											uint64_cu *__hr_metadata_d_I,										// pointer to output image (DEVICE GLOBAL MEMORY)
											uint64_cu *__hr_metadata_d_sums,									// pointer to input image (DEVICE GLOBAL MEMORY)
											uint64_cu *__hr_metadata_d_sums2){
  /**** HIRACE STUFF ****/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  HIRACE_WRAP_DATA(fp, d_I)
  HIRACE_WRAP_DATA(fp, d_sums)
  HIRACE_WRAP_DATA(fp, d_sums2)
  
  HIRACE_SET_DATA_GLOBAL(d_I)
  HIRACE_SET_DATA_GLOBAL(d_sums)
  HIRACE_SET_DATA_GLOBAL(d_sums2)
  
  #define d_I d_I.registerCallsite(__LINE__,__FILE__)
  #define d_sums d_sums.registerCallsite(__LINE__,__FILE__)
  #define d_sums2 d_sums2.registerCallsite(__LINE__,__FILE__)
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

  /**** HIRACE STUFF ****/
  #undef d_I    
  #undef d_sums 
  #undef d_sums2
  /**** HIRACE STUFF ****/

}
