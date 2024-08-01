/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define USE_TEXTURE 1
#define POWER_OF_TWO 1

#if (USE_TEXTURE)
#define LOAD_FLOAT(i) tex1Dfetch<float>(texFloat, i)
#define SET_FLOAT_BASE
#else
#define LOAD_FLOAT(i) d_Src[i]
#define SET_FLOAT_BASE
#endif

////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
__global__ void padKernel_kernel(float *__hr_d_Dst, float *__hr_d_Src, int fftH, int fftW,
                                 int kernelH, int kernelW, int kernelY,
                                 int kernelX, hr_shadowt *__hr_metadata_d_Dst, hr_shadowt *__hr_metadata_d_Src
#if (USE_TEXTURE)
                                 ,
                                 cudaTextureObject_t texFloat
#endif
                                 ) {
  /************************/
  /***** HIRACE START *****/
  /************************/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  
  HIRACE_WRAP_DATA(float, d_Dst)
  HIRACE_WRAP_DATA(float, d_Src)
  
  HIRACE_SET_DATA_GLOBAL(d_Dst)
  HIRACE_SET_DATA_GLOBAL(d_Src)
 
#define d_Dst d_Dst.registerCallsite(__LINE__,__FILE__)
#define d_Src d_Src.registerCallsite(__LINE__,__FILE__)
  /************************/
  /***** HIRACE END *****/
  /************************/
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (y < kernelH && x < kernelW) {
    int ky = y - kernelY;

    if (ky < 0) {
      ky += fftH;
    }

    int kx = x - kernelX;

    if (kx < 0) {
      kx += fftW;
    }

    d_Dst[ky * fftW + kx] = LOAD_FLOAT(y * kernelW + x);
  }
  /************************/
  /***** HIRACE START *****/
  /************************/

#undef d_Dst
#undef d_Src

  /************************/
  /***** HIRACE END *****/
  /************************/
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
__global__ void padDataClampToBorder_kernel(float *__hr_d_Dst, float *__hr_d_Src,
                                            int fftH, int fftW, int dataH,
                                            int dataW, int kernelH, int kernelW,
                                            int kernelY, int kernelX, hr_shadowt *__hr_metadata_d_Dst, hr_shadowt *__hr_metadata_d_Src
#if (USE_TEXTURE)
                                            ,
                                            cudaTextureObject_t texFloat
#endif
                                            ) {
  /************************/
  /***** HIRACE START *****/
  /************************/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  
  HIRACE_WRAP_DATA(float, d_Dst)
  HIRACE_WRAP_DATA(float, d_Src)
  
  HIRACE_SET_DATA_GLOBAL(d_Dst)
  HIRACE_SET_DATA_GLOBAL(d_Src)
 
#define d_Dst d_Dst.registerCallsite(__LINE__,__FILE__)
#define d_Src d_Src.registerCallsite(__LINE__,__FILE__)
  /************************/
  /***** HIRACE END *****/
  /************************/
  
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int borderH = dataH + kernelY;
  const int borderW = dataW + kernelX;

  if (y < fftH && x < fftW) {
    int dy, dx;

    if (y < dataH) {
      dy = y;
    }

    if (x < dataW) {
      dx = x;
    }

    if (y >= dataH && y < borderH) {
      dy = dataH - 1;
    }

    if (x >= dataW && x < borderW) {
      dx = dataW - 1;
    }

    if (y >= borderH) {
      dy = 0;
    }

    if (x >= borderW) {
      dx = 0;
    }

    d_Dst[y * fftW + x] = LOAD_FLOAT(dy * dataW + dx);
  }
  /************************/
  /***** HIRACE START *****/
  /************************/

#undef d_Dst
#undef d_Src

  /************************/
  /***** HIRACE END *****/
  /************************/
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
inline __device__ void mulAndScale(fComplex &a, const fComplex &b,
                                   const float &c) {
  fComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
  a = t;
}

__global__ void modulateAndNormalize_kernel(fComplex *__hr_tmp_d_Dst, fComplex *__hr_tmp_d_Src,
                                            int dataSize, float c, hr_shadowt *__hr_metadata_d_Dst, hr_shadowt *__hr_metadata_d_Src) {
  /************************/
  /***** HIRACE START *****/
  /************************/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  
  float *__hr_d_Dst = (float *)__hr_tmp_d_Dst;
  float *__hr_d_Src = (float *)__hr_tmp_d_Src;
  
  HIRACE_WRAP_DATA(float, d_Dst)
  HIRACE_WRAP_DATA(float, d_Src)
  
  HIRACE_SET_DATA_GLOBAL(d_Dst)
  HIRACE_SET_DATA_GLOBAL(d_Src)
 
#define d_Dst d_Dst.registerCallsite(__LINE__,__FILE__)
#define d_Src d_Src.registerCallsite(__LINE__,__FILE__)
  /************************/
  /***** HIRACE END *****/
  /************************/
  
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= dataSize) {
    return;
  }
  
  //fComplex a = d_Src[i];
  //fComplex b = d_Dst[i];
  fComplex a;
  fComplex b;

  //float __hr_temp_a[2];
  //float __hr_temp_b[2];
  
  a.x = d_Src[2*i];
  a.y = d_Src[(2*i)+1];
  b.x = d_Dst[2*i];
  b.y = d_Dst[(2*i)+1];
  
  //__hr_temp_a[0] = d_Src[2*i];
  //__hr_temp_a[1] = d_Src[(2*i)+1];
  //__hr_temp_b[0] = d_Dst[2*i];
  //__hr_temp_b[1] = d_Dst[(2*i)+1];
  
  //a.x = __hr_temp_a[0];
  //a.y = __hr_temp_a[1];
  //b.x = __hr_temp_b[0];
  //b.y = __hr_temp_b[1];

  mulAndScale(a, b, c);

  d_Dst[2*i] = a.x;
  d_Dst[(2*i)+1] = a.y;
  
  /************************/
  /***** HIRACE START *****/
  /************************/

#undef d_Dst
#undef d_Src

  /************************/
  /***** HIRACE END *****/
  /************************/
}

////////////////////////////////////////////////////////////////////////////////
// 2D R2C / C2R post/preprocessing kernels
////////////////////////////////////////////////////////////////////////////////
#if (USE_TEXTURE)
#define LOAD_FCOMPLEX(i) tex1Dfetch<fComplex>(texComplex, i)
#define LOAD_FCOMPLEX_A(i) tex1Dfetch<fComplex>(texComplexA, i)
#define LOAD_FCOMPLEX_B(i) tex1Dfetch<fComplex>(texComplexB, i)

#define SET_FCOMPLEX_BASE
#define SET_FCOMPLEX_BASE_A
#define SET_FCOMPLEX_BASE_B
#else
#define LOAD_FCOMPLEX(i) d_Src[i]
#define LOAD_FCOMPLEX_A(i) d_SrcA[i]
#define LOAD_FCOMPLEX_B(i) d_SrcB[i]

#define SET_FCOMPLEX_BASE
#define SET_FCOMPLEX_BASE_A
#define SET_FCOMPLEX_BASE_B
#endif

inline __device__ void spPostprocessC2C(fComplex &D1, fComplex &D2,
                                        const fComplex &twiddle) {
  float A1 = 0.5f * (D1.x + D2.x);
  float B1 = 0.5f * (D1.y - D2.y);
  float A2 = 0.5f * (D1.y + D2.y);
  float B2 = 0.5f * (D1.x - D2.x);

  D1.x = A1 + (A2 * twiddle.x + B2 * twiddle.y);
  D1.y = (A2 * twiddle.y - B2 * twiddle.x) + B1;
  D2.x = A1 - (A2 * twiddle.x + B2 * twiddle.y);
  D2.y = (A2 * twiddle.y - B2 * twiddle.x) - B1;
}

// Premultiply by 2 to account for 1.0 / (DZ * DY * DX) normalization
inline __device__ void spPreprocessC2C(fComplex &D1, fComplex &D2,
                                       const fComplex &twiddle) {
  float A1 = /* 0.5f * */ (D1.x + D2.x);
  float B1 = /* 0.5f * */ (D1.y - D2.y);
  float A2 = /* 0.5f * */ (D1.y + D2.y);
  float B2 = /* 0.5f * */ (D1.x - D2.x);

  D1.x = A1 - (A2 * twiddle.x - B2 * twiddle.y);
  D1.y = (B2 * twiddle.x + A2 * twiddle.y) + B1;
  D2.x = A1 + (A2 * twiddle.x - B2 * twiddle.y);
  D2.y = (B2 * twiddle.x + A2 * twiddle.y) - B1;
}

inline __device__ void getTwiddle(fComplex &twiddle, float phase) {
  __sincosf(phase, &twiddle.y, &twiddle.x);
}

inline __device__ uint mod(uint a, uint DA) {
  //(DA - a) % DA, assuming a <= DA
  return a ? (DA - a) : a;
}

static inline uint factorRadix2(uint &log2N, uint n) {
  if (!n) {
    log2N = 0;
    return 0;
  } else {
    for (log2N = 0; n % 2 == 0; n /= 2, log2N++)
      ;

    return n;
  }
}

inline __device__ void udivmod(uint &dividend, uint divisor, uint &rem) {
#if (!POWER_OF_TWO)
  rem = dividend % divisor;
  dividend /= divisor;
#else
  rem = dividend & (divisor - 1);
  dividend >>= (__ffs(divisor) - 1);
#endif
}

__global__ void spPostprocess2D_kernel(fComplex *__hr_tmp_d_Dst, fComplex *__hr_tmp_d_Src,
                                       uint DY, uint DX, uint threadCount,
                                       uint padding, float phaseBase, hr_shadowt *__hr_metadata_d_Dst, hr_shadowt *__hr_metadata_d_Src
#if (USE_TEXTURE)
                                       ,
                                       cudaTextureObject_t texComplex
#endif
                                       ) {
  /************************/
  /***** HIRACE START *****/
  /************************/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  
  float *__hr_d_Dst = (float *)__hr_tmp_d_Dst;
  float *__hr_d_Src = (float *)__hr_tmp_d_Src;
  
  HIRACE_WRAP_DATA(float, d_Dst)
  HIRACE_WRAP_DATA(float, d_Src)
  
  HIRACE_SET_DATA_GLOBAL(d_Dst)
  HIRACE_SET_DATA_GLOBAL(d_Src)
 
#define d_Dst d_Dst.registerCallsite(__LINE__,__FILE__)
#define d_Src d_Src.registerCallsite(__LINE__,__FILE__)
  /************************/
  /***** HIRACE END *****/
  /************************/
  
  const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= threadCount) {
    return;
  }

  uint x, y, i = threadId;
  udivmod(i, DX / 2, x);
  udivmod(i, DY, y);

  // Avoid overwrites in columns DX / 2 by different threads
  if ((x == 0) && (y > DY / 2)) {
    return;
  }

  const uint srcOffset = i * DY * DX;
  const uint dstOffset = i * DY * (DX + padding);

  // Process x = [0 .. DX / 2 - 1] U [DX / 2 + 1 .. DX]
  {
    const uint loadPos1 = srcOffset + y * DX + x;
    const uint loadPos2 = srcOffset + mod(y, DY) * DX + mod(x, DX);
    const uint storePos1 = dstOffset + y * (DX + padding) + x;
    const uint storePos2 = dstOffset + mod(y, DY) * (DX + padding) + (DX - x);

    fComplex D1 = LOAD_FCOMPLEX(loadPos1);
    fComplex D2 = LOAD_FCOMPLEX(loadPos2);

    fComplex twiddle;
    getTwiddle(twiddle, phaseBase * (float)x);
    spPostprocessC2C(D1, D2, twiddle);

    d_Dst[2*storePos1] = D1.x;
    d_Dst[(2*storePos1)+1] = D1.y;
    d_Dst[2*storePos2] = D2.x;
    d_Dst[(2*storePos2)+1] = D2.y;
    //d_Dst[storePos1] = D1;
    //d_Dst[storePos2] = D2;
  }

  // Process x = DX / 2
  if (x == 0) {
    const uint loadPos1 = srcOffset + y * DX + DX / 2;
    const uint loadPos2 = srcOffset + mod(y, DY) * DX + DX / 2;
    const uint storePos1 = dstOffset + y * (DX + padding) + DX / 2;
    const uint storePos2 = dstOffset + mod(y, DY) * (DX + padding) + DX / 2;

    fComplex D1 = LOAD_FCOMPLEX(loadPos1);
    fComplex D2 = LOAD_FCOMPLEX(loadPos2);

    // twiddle = getTwiddle(phaseBase * (DX / 2)) = exp(dir * j * PI / 2)
    fComplex twiddle = {0, (phaseBase > 0) ? 1.0f : -1.0f};
    spPostprocessC2C(D1, D2, twiddle);

    d_Dst[2*storePos1] = D1.x;
    d_Dst[(2*storePos1)+1] = D1.y;
    d_Dst[2*storePos2] = D2.x;
    d_Dst[(2*storePos2)+1] = D2.y;
    //d_Dst[storePos1] = D1;
    //d_Dst[storePos2] = D2;
  }
  
  /************************/
  /***** HIRACE START *****/
  /************************/

#undef d_Dst
#undef d_Src

  /************************/
  /***** HIRACE END *****/
  /************************/
}

__global__ void spPreprocess2D_kernel(fComplex *__hr_tmp_d_Dst, fComplex *__hr_tmp_d_Src, uint DY,
                                      uint DX, uint threadCount, uint padding,
                                      float phaseBase, hr_shadowt *__hr_metadata_d_Dst, hr_shadowt *__hr_metadata_d_Src
#if (USE_TEXTURE)
                                      ,
                                      cudaTextureObject_t texComplex
#endif
                                      ) {
  /************************/
  /***** HIRACE START *****/
  /************************/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  
  float *__hr_d_Dst = (float *)__hr_tmp_d_Dst;
  float *__hr_d_Src = (float *)__hr_tmp_d_Src;
  
  HIRACE_WRAP_DATA(float, d_Dst)
  HIRACE_WRAP_DATA(float, d_Src)
  
  HIRACE_SET_DATA_GLOBAL(d_Dst)
  HIRACE_SET_DATA_GLOBAL(d_Src)
 
#define d_Dst d_Dst.registerCallsite(__LINE__,__FILE__)
#define d_Src d_Src.registerCallsite(__LINE__,__FILE__)
  /************************/
  /***** HIRACE END *****/
  /************************/
  
  const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= threadCount) {
    return;
  }

  uint x, y, i = threadId;
  udivmod(i, DX / 2, x);
  udivmod(i, DY, y);

  // Avoid overwrites in columns 0 and DX / 2 by different threads (lower and
  // upper halves)
  if ((x == 0) && (y > DY / 2)) {
    return;
  }

  const uint srcOffset = i * DY * (DX + padding);
  const uint dstOffset = i * DY * DX;

  // Process x = [0 .. DX / 2 - 1] U [DX / 2 + 1 .. DX]
  {
    const uint loadPos1 = srcOffset + y * (DX + padding) + x;
    const uint loadPos2 = srcOffset + mod(y, DY) * (DX + padding) + (DX - x);
    const uint storePos1 = dstOffset + y * DX + x;
    const uint storePos2 = dstOffset + mod(y, DY) * DX + mod(x, DX);

    fComplex D1 = LOAD_FCOMPLEX(loadPos1);
    fComplex D2 = LOAD_FCOMPLEX(loadPos2);

    fComplex twiddle;
    getTwiddle(twiddle, phaseBase * (float)x);
    spPreprocessC2C(D1, D2, twiddle);

    d_Dst[2*storePos1] = D1.x;
    d_Dst[(2*storePos1)+1] = D1.y;
    d_Dst[2*storePos2] = D2.x;
    d_Dst[(2*storePos2)+1] = D2.y;
    //d_Dst[storePos1] = D1;
    //d_Dst[storePos2] = D2;
  }

  // Process x = DX / 2
  if (x == 0) {
    const uint loadPos1 = srcOffset + y * (DX + padding) + DX / 2;
    const uint loadPos2 = srcOffset + mod(y, DY) * (DX + padding) + DX / 2;
    const uint storePos1 = dstOffset + y * DX + DX / 2;
    const uint storePos2 = dstOffset + mod(y, DY) * DX + DX / 2;

    fComplex D1 = LOAD_FCOMPLEX(loadPos1);
    fComplex D2 = LOAD_FCOMPLEX(loadPos2);

    // twiddle = getTwiddle(phaseBase * (DX / 2)) = exp(-dir * j * PI / 2)
    fComplex twiddle = {0, (phaseBase > 0) ? 1.0f : -1.0f};
    spPreprocessC2C(D1, D2, twiddle);

    d_Dst[2*storePos1] = D1.x;
    d_Dst[(2*storePos1)+1] = D1.y;
    d_Dst[2*storePos2] = D2.x;
    d_Dst[(2*storePos2)+1] = D2.y;
    //d_Dst[storePos1] = D1;
    //d_Dst[storePos2] = D2;
  }
  
  /************************/
  /***** HIRACE START *****/
  /************************/

#undef d_Dst
#undef d_Src

  /************************/
  /***** HIRACE END *****/
  /************************/
}

////////////////////////////////////////////////////////////////////////////////
// Combined spPostprocess2D + modulateAndNormalize + spPreprocess2D
////////////////////////////////////////////////////////////////////////////////
__global__ void spProcess2D_kernel(fComplex *__hr_tmp_d_Dst, fComplex *__hr_tmp_d_SrcA,
                                   fComplex *__hr_tmp_d_SrcB, uint DY, uint DX,
                                   uint threadCount, float phaseBase, float c, hr_shadowt *__hr_metadata_d_Dst, hr_shadowt *__hr_metadata_d_SrcA, hr_shadowt *__hr_metadata_d_SrcB
#if (USE_TEXTURE)
                                   ,
                                   cudaTextureObject_t texComplexA,
                                   cudaTextureObject_t texComplexB
#endif
                                   ) {
  /************************/
  /***** HIRACE START *****/
  /************************/
  unsigned __hr_bcount = 0, __hr_wcount = 0, __hr_swidx = 0;
  
  float *__hr_d_Dst  = (float *)__hr_tmp_d_Dst;
  float *__hr_d_SrcA = (float *)__hr_tmp_d_SrcA;
  float *__hr_d_SrcB = (float *)__hr_tmp_d_SrcB;
  
  HIRACE_WRAP_DATA(float, d_Dst)
  HIRACE_WRAP_DATA(float, d_SrcA)
  HIRACE_WRAP_DATA(float, d_SrcB)
  
  HIRACE_SET_DATA_GLOBAL(d_Dst)
  HIRACE_SET_DATA_GLOBAL(d_SrcA)
  HIRACE_SET_DATA_GLOBAL(d_SrcB)
 
#define d_Dst d_Dst.registerCallsite(__LINE__,__FILE__)
#define d_SrcA d_SrcA.registerCallsite(__LINE__,__FILE__)
#define d_SrcB d_SrcB.registerCallsite(__LINE__,__FILE__)
  /************************/
  /***** HIRACE END *****/
  /************************/
  
  const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= threadCount) {
    return;
  }

  uint x, y, i = threadId;
  udivmod(i, DX, x);
  udivmod(i, DY / 2, y);

  const uint offset = i * DY * DX;

  // Avoid overwrites in rows 0 and DY / 2 by different threads (left and right
  // halves) Otherwise correctness for in-place transformations is affected
  if ((y == 0) && (x > DX / 2)) {
    return;
  }

  fComplex twiddle;

  // Process y = [0 .. DY / 2 - 1] U [DY - (DY / 2) + 1 .. DY - 1]
  {
    const uint pos1 = offset + y * DX + x;
    const uint pos2 = offset + mod(y, DY) * DX + mod(x, DX);

    fComplex D1 = LOAD_FCOMPLEX_A(pos1);
    fComplex D2 = LOAD_FCOMPLEX_A(pos2);
    fComplex K1 = LOAD_FCOMPLEX_B(pos1);
    fComplex K2 = LOAD_FCOMPLEX_B(pos2);
    getTwiddle(twiddle, phaseBase * (float)x);

    spPostprocessC2C(D1, D2, twiddle);
    spPostprocessC2C(K1, K2, twiddle);
    mulAndScale(D1, K1, c);
    mulAndScale(D2, K2, c);
    spPreprocessC2C(D1, D2, twiddle);

    d_Dst[2*pos1] = D1.x;
    d_Dst[(2*pos1)+1] = D1.y;
    d_Dst[2*pos2] = D2.x;
    d_Dst[(2*pos2)+1] = D2.y;
    //d_Dst[pos1] = D1;
    //d_Dst[pos2] = D2;
  }

  if (y == 0) {
    const uint pos1 = offset + (DY / 2) * DX + x;
    const uint pos2 = offset + (DY / 2) * DX + mod(x, DX);

    fComplex D1 = LOAD_FCOMPLEX_A(pos1);
    fComplex D2 = LOAD_FCOMPLEX_A(pos2);
    fComplex K1 = LOAD_FCOMPLEX_B(pos1);
    fComplex K2 = LOAD_FCOMPLEX_B(pos2);

    spPostprocessC2C(D1, D2, twiddle);
    spPostprocessC2C(K1, K2, twiddle);
    mulAndScale(D1, K1, c);
    mulAndScale(D2, K2, c);
    spPreprocessC2C(D1, D2, twiddle);

    d_Dst[2*pos1] = D1.x;
    d_Dst[(2*pos1)+1] = D1.y;
    d_Dst[2*pos2] = D2.x;
    d_Dst[(2*pos2)+1] = D2.y;
    //d_Dst[pos1] = D1;
    //d_Dst[pos2] = D2;
  }
  
  /************************/
  /***** HIRACE START *****/
  /************************/

#undef d_Dst
#undef d_SrcA
#undef d_SrcB

  /************************/
  /***** HIRACE END *****/
  /************************/
}
