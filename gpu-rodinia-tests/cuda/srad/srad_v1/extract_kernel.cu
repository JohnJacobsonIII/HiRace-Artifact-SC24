// statistical kernel
__global__ void extract(	long d_Ne,
											fp *_d_I,										// pointer to input image (DEVICE GLOBAL MEMORY)
											uint64_cu *__hr_d_I){										// pointer to input image (DEVICE GLOBAL MEMORY)
  /**** HIRACE STUFF ****/
  HiRaceDataWrap<fp> d_I         = _d_I;
   
  d_I.setMetadata(__hr_d_I);
   
  d_I.setScope(Scope::Global);
  
  __syncthreads();
   
  #define d_I             d_I.registerCallsite(__LINE__,__FILE__)
  /**** HIRACE STUFF ****/

	// indexes
	int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;						// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei<d_Ne){															// do only for the number of elements, omit extra threads

		d_I[ei] = exp(d_I[ei]/255);												// exponentiate input IMAGE and copy to output image

	}


  #undef d_I    
}
