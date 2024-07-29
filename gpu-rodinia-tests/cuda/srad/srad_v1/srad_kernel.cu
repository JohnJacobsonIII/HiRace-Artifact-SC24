// BUG IN SRAD APPLICATIONS SEEMS TO BE SOMEWHERE IN THIS CODE, WRONG MEMORY ACCESS

// srad kernel
__global__ void srad(	fp d_lambda, 
									 int d_Nr, 
									 int d_Nc, 
									 long d_Ne, 
									 int *_d_iN, 
									 int *_d_iS, 
									 int *_d_jE, 
									 int *_d_jW, 
									 fp *_d_dN, 
									 fp *_d_dS, 
									 fp *_d_dE, 
									 fp *_d_dW, 
									 fp d_q0sqr, 
									 fp *_d_c, 
									 fp *_d_I,
									 uint64_cu *__hr_d_iN, 
									 uint64_cu *__hr_d_iS, 
									 uint64_cu *__hr_d_jE, 
									 uint64_cu *__hr_d_jW, 
									 uint64_cu *__hr_d_dN, 
									 uint64_cu *__hr_d_dS, 
									 uint64_cu *__hr_d_dE, 
									 uint64_cu *__hr_d_dW, 
									 uint64_cu *__hr_d_c, 
									 uint64_cu *__hr_d_I){
  /**** HIRACE STUFF ****/
  HiRaceDataWrap<fp> d_I         = _d_I;
  HiRaceDataWrap<fp> d_c         = _d_c    ;
  HiRaceDataWrap<int> d_iN        = _d_iN   ;
  HiRaceDataWrap<int> d_iS        = _d_iS   ;
  HiRaceDataWrap<int> d_jE        = _d_jE   ;
  HiRaceDataWrap<int> d_jW        = _d_jW   ;
  HiRaceDataWrap<fp> d_dN        = _d_dN   ;
  HiRaceDataWrap<fp> d_dS        = _d_dS   ;
  HiRaceDataWrap<fp> d_dW        = _d_dW   ;
  HiRaceDataWrap<fp> d_dE        = _d_dE   ;
   
  d_I.setMetadata(__hr_d_I);
  d_c.setMetadata(__hr_d_c);
  d_iN.setMetadata(__hr_d_iN);
  d_iS.setMetadata(__hr_d_iS);
  d_jE.setMetadata(__hr_d_jE);
  d_jW.setMetadata(__hr_d_jW);
  d_dN.setMetadata(__hr_d_dN);
  d_dS.setMetadata(__hr_d_dS);
  d_dW.setMetadata(__hr_d_dW);
  d_dE.setMetadata(__hr_d_dE);
   
  d_I.setScope(Scope::Global);
  d_c.setScope(Scope::Global);
  d_iN.setScope(Scope::Global);
  d_iS.setScope(Scope::Global);
  d_jE.setScope(Scope::Global);
  d_jW.setScope(Scope::Global);
  d_dN.setScope(Scope::Global);
  d_dS.setScope(Scope::Global);
  d_dW.setScope(Scope::Global);
  d_dE.setScope(Scope::Global);
  
  __syncthreads();
   
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
	fp d_Jc;
	fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
	fp d_c_loc;
	fp d_G2,d_L,d_num,d_den,d_qsqr;
	
	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;													// (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;												// (0-n) column
	if((ei+1) % d_Nr == 0){
		row = d_Nr - 1;
		col = col - 1;
	}
	
	if(ei<d_Ne){															// make sure that only threads matching jobs run
		
		// directional derivatives, ICOV, diffusion coefficent
		d_Jc = d_I[ei];														// get value of the current element
		
		// directional derivates (every element of IMAGE)(try to copy to shared memory or temp files)
		d_dN_loc = d_I[d_iN[row] + d_Nr*col] - d_Jc;						// north direction derivative
		d_dS_loc = d_I[d_iS[row] + d_Nr*col] - d_Jc;						// south direction derivative
		d_dW_loc = d_I[row + d_Nr*d_jW[col]] - d_Jc;						// west direction derivative
		d_dE_loc = d_I[row + d_Nr*d_jE[col]] - d_Jc;						// east direction derivative
	         
		// normalized discrete gradient mag squared (equ 52,53)
		d_G2 = (d_dN_loc*d_dN_loc + d_dS_loc*d_dS_loc + d_dW_loc*d_dW_loc + d_dE_loc*d_dE_loc) / (d_Jc*d_Jc);	// gradient (based on derivatives)
		
		// normalized discrete laplacian (equ 54)
		d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;			// laplacian (based on derivatives)

		// ICOV (equ 31/35)
		d_num  = (0.5*d_G2) - ((1.0/16.0)*(d_L*d_L)) ;						// num (based on gradient and laplacian)
		d_den  = 1 + (0.25*d_L);												// den (based on laplacian)
		d_qsqr = d_num/(d_den*d_den);										// qsqr (based on num and den)
	 
		// diffusion coefficent (equ 33) (every element of IMAGE)
		d_den = (d_qsqr-d_q0sqr) / (d_q0sqr * (1+d_q0sqr)) ;				// den (based on qsqr and q0sqr)
		d_c_loc = 1.0 / (1.0+d_den) ;										// diffusion coefficient (based on den)
	    
		// saturate diffusion coefficent to 0-1 range
		if (d_c_loc < 0){													// if diffusion coefficient < 0
			d_c_loc = 0;													// ... set to 0
		}
		else if (d_c_loc > 1){												// if diffusion coefficient > 1
			d_c_loc = 1;													// ... set to 1
		}

		// save data to global memory
		d_dN[ei] = d_dN_loc; 
		d_dS[ei] = d_dS_loc; 
		d_dW[ei] = d_dW_loc; 
		d_dE[ei] = d_dE_loc;
		d_c[ei] = d_c_loc;
			
	}
	

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
}
  /**** HIRACE STUFF ****
  HiRaceDataWrap<fp> d_I         = _d_I;
  HiRaceDataWrap<fp> d_c         = _d_c    ;
  HiRaceDataWrap<int> d_iN        = _d_iN   ;
  HiRaceDataWrap<int> d_iS        = _d_iS   ;
  HiRaceDataWrap<int> d_jE        = _d_jE   ;
  HiRaceDataWrap<int> d_jW        = _d_jW   ;
  HiRaceDataWrap<fp> d_dN        = _d_dN   ;
  HiRaceDataWrap<fp> d_dS        = _d_dS   ;
  HiRaceDataWrap<fp> d_dW        = _d_dW   ;
  HiRaceDataWrap<fp> d_dE        = _d_dE   ;
  HiRaceDataWrap<fp> d_sums      = _d_sums ;
  HiRaceDataWrap<fp> d_sums2     = _d_sums2;
   
  d_I.setMetadata(__hr_d_I);
  d_c.setMetadata(__hr_d_c);
  d_iN.setMetadata(__hr_d_iN);
  d_iS.setMetadata(__hr_d_iS);
  d_jE.setMetadata(__hr_d_jE);
  d_jW.setMetadata(__hr_d_jW);
  d_dN.setMetadata(__hr_d_dN);
  d_dS.setMetadata(__hr_d_dS);
  d_dW.setMetadata(__hr_d_dW);
  d_dE.setMetadata(__hr_d_dE);
  d_sums.setMetadata(__hr_d_sums);
  d_sums2.setMetadata(__hr_d_sums2);
   
  d_I.setScope(Scope::Global);
  d_c.setScope(Scope::Global);
  d_iN.setScope(Scope::Global);
  d_iS.setScope(Scope::Global);
  d_jE.setScope(Scope::Global);
  d_jW.setScope(Scope::Global);
  d_dN.setScope(Scope::Global);
  d_dS.setScope(Scope::Global);
  d_dW.setScope(Scope::Global);
  d_dE.setScope(Scope::Global);
  d_sums.setScope(Scope::Global);
  d_sums2.setScope(Scope::Global);
  
  __syncthreads();
   
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
  #define d_sums          d_sums.registerCallsite(__LINE__,__FILE__)
  #define d_sums2         d_sums2.registerCallsite(__LINE__,__FILE__)

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
  #undef d_sums 
  #undef d_sums2
  **** HIRACE STUFF ****/
