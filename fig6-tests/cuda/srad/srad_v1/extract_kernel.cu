// statistical kernel
__global__ void extract(	long d_Ne,
											fp *__hr_d_I,										// pointer to input image (DEVICE GLOBAL MEMORY)
											uint64_cu *__hr_metadata_d_I){										// pointer to input image (DEVICE GLOBAL MEMORY)
  /**** HIRACE STUFF ****/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  HIRACE_WRAP_DATA(fp, d_I)
  
  HIRACE_SET_DATA_GLOBAL(d_I)
  
  #define d_I d_I.registerCallsite(__LINE__,__FILE__)
  /**** HIRACE STUFF ****/
  

	// indexes
	int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;						// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei<d_Ne){															// do only for the number of elements, omit extra threads

		d_I[ei] = exp(d_I[ei]/255);												// exponentiate input IMAGE and copy to output image

	}


  /**** HIRACE STUFF ****/
  #undef d_I    
  /**** HIRACE STUFF ****/
}
