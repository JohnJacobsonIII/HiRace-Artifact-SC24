// BUG IN SRAD APPLICATIONS SEEMS TO BE SOMEWHERE IN THIS CODE, MEMORY CORRUPTION

// srad kernel
__global__ void srad2(	fp d_lambda, 
										int d_Nr, 
										int d_Nc, 
										long d_Ne, 
										int *__hr_d_iN, 
										int *__hr_d_iS, 
										int *__hr_d_jE, 
										int *__hr_d_jW,
										fp  *__hr_d_dN, 
										fp  *__hr_d_dS, 
										fp  *__hr_d_dE, 
										fp  *__hr_d_dW, 
										fp  *__hr_d_c, 
										fp  *__hr_d_I,
										uint64_cu *__hr_metadata_d_iN, 
										uint64_cu *__hr_metadata_d_iS, 
										uint64_cu *__hr_metadata_d_jE, 
										uint64_cu *__hr_metadata_d_jW,
										uint64_cu *__hr_metadata_d_dN, 
										uint64_cu *__hr_metadata_d_dS, 
										uint64_cu *__hr_metadata_d_dE, 
										uint64_cu *__hr_metadata_d_dW, 
										uint64_cu *__hr_metadata_d_c, 
										uint64_cu *__hr_metadata_d_I){
  /**** HIRACE STUFF ****/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  HIRACE_WRAP_DATA(fp, d_I)  
  HIRACE_WRAP_DATA(fp, d_c)  
  HIRACE_WRAP_DATA(int, d_iN)
  HIRACE_WRAP_DATA(int, d_iS)
  HIRACE_WRAP_DATA(int, d_jE)
  HIRACE_WRAP_DATA(int, d_jW)
  HIRACE_WRAP_DATA(fp, d_dN) 
  HIRACE_WRAP_DATA(fp, d_dS) 
  HIRACE_WRAP_DATA(fp, d_dW) 
  HIRACE_WRAP_DATA(fp, d_dE) 
   
  HIRACE_SET_DATA_GLOBAL(d_I)  
  HIRACE_SET_DATA_GLOBAL(d_c)  
  HIRACE_SET_DATA_GLOBAL(d_iN)
  HIRACE_SET_DATA_GLOBAL(d_iS)
  HIRACE_SET_DATA_GLOBAL(d_jE)
  HIRACE_SET_DATA_GLOBAL(d_jW)
  HIRACE_SET_DATA_GLOBAL(d_dN) 
  HIRACE_SET_DATA_GLOBAL(d_dS) 
  HIRACE_SET_DATA_GLOBAL(d_dW) 
  HIRACE_SET_DATA_GLOBAL(d_dE) 
  
  #define d_I             d_I.registerCallsite(__LINE__,__FILE__)
  #define d_c             d_c.registerCallsite(__LINE__,__FILE__)
  #define d_iN            d_iN.registerCallsite(__LINE__,__FILE__)
  #define d_iS            d_iS.registerCallsite(__LINE__,__FILE__)
  #define d_jE            d_jE.registerCallsite(__LINE__,__FILE__)
  #define d_jW            d_jW.registerCallsite(__LINE__,__FILE__)
  #define d_dN            d_dN.registerCallsite(__LINE__,__FILE__)
  #define d_dS            d_dS.registerCallsite(__LINE__,__FILE__)
  #define d_dW            d_dW.registerCallsite(__LINE__,__FILE__)
  #define d_dE            d_dE.registerCallsite(__LINE__,__FILE__)
  /**** HIRACE STUFF ****/

	// indexes
    int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = bx*NUMBER_THREADS+tx;											// more threads than actual elements !!!
	int row;																// column, x position
	int col;																// row, y position

	// variables
	fp d_cN,d_cS,d_cW,d_cE;
	fp d_D;

	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;												// (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;											// (0-n) column
	if((ei+1) % d_Nr == 0){
		row = d_Nr - 1;
		col = col - 1;
	}

	if(ei<d_Ne){															// make sure that only threads matching jobs run

		// diffusion coefficent
		d_cN = d_c[ei];														// north diffusion coefficient
		d_cS = d_c[d_iS[row] + d_Nr*col];										// south diffusion coefficient
		d_cW = d_c[ei];														// west diffusion coefficient
		d_cE = d_c[row + d_Nr * d_jE[col]];									// east diffusion coefficient

		// divergence (equ 58)
		d_D = d_cN*d_dN[ei] + d_cS*d_dS[ei] + d_cW*d_dW[ei] + d_cE*d_dE[ei];// divergence

		// image update (equ 61) (every element of IMAGE)
		d_I[ei] = d_I[ei] + 0.25*d_lambda*d_D;								// updates image (based on input time step and divergence)

	}


  /**** HIRACE STUFF ****/
  #undef d_I    
  #undef d_c    
  #undef d_iN   
  #undef d_iS   
  #undef d_jE   
  #undef d_jW   
  #undef d_dN   
  #undef d_dS   
  #undef d_dW   
  #undef d_dE   
  /**** HIRACE STUFF ****/
}
