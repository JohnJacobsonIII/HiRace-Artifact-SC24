#ifndef HIRACE_WRAPPERS_H
#define HIRACE_WRAPPERS_H

#include "HiRaceDataWrap.h"


/*******************/
/***** Atomics *****/
/*******************/

#define WRAP_ATOMIC(name) \
  template <typename T, typename U> \
  CUDA_CALLABLE_MEMBER T name(HiRaceDataWrap<T> &data, U val) { \
    return data.name(val); \
  }

WRAP_ATOMIC(atomicAdd)
WRAP_ATOMIC(atomicSub)
WRAP_ATOMIC(atomicExch)
WRAP_ATOMIC(atomicMin)
WRAP_ATOMIC(atomicMax)
WRAP_ATOMIC(atomicInc)
WRAP_ATOMIC(atomicDec)
WRAP_ATOMIC(atomicAnd)
WRAP_ATOMIC(atomicOr)
WRAP_ATOMIC(atomicXor)

#undef WRAP_ATOMIC

#endif // HIRACE_WRAPPERS_H
