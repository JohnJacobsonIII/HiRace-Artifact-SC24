#ifndef HIRACE_H
#define HIRACE_H

#include <assert.h>

/* Bit masks
 * 
 * 64-bit state:
 * [--tid--|--bid--|--state--|--bsync count--|--wsync count--]
 * |  10   |  29   |    5    |      15       |       5       ]
 * 
 *   10: thread id    (1024 tid per blk)
 *   31: block id     (2^32-1 blocks max)
 *   05: system state (25 states)
 *   15: block barriers (__syncthreads) observed
 *     - scalar clock for block hierarchy
 *   05: warp barriers (__syncwarp and others) observed
 *     - scalar clock for warp hierarchy
 *     - sub-warp masked synchronization is 
 *       stored and monitored with all
 *       warp level synchronization activity
 */

using hr_shadowt = unsigned long long int;
using uint64_cu  = unsigned long long int;

// Some temporary macros for manually instrumenting code
#define HIRACE_SHADOW_DECL(NAME) hr_shadowt *__hr_metadata_ ## NAME ;
#define HIRACE_MALLOC(NAME, SIZE) cudaMalloc((void **)&__hr_metadata_ ## NAME, SIZE * sizeof(hr_shadowt));
#define HIRACE_MEMSET(NAME, SIZE) cudaMemset(__hr_metadata_ ## NAME, 0, SIZE * sizeof(hr_shadowt));
#define HIRACE_CUDA_FREE(NAME) cudaFree(__hr_metadata_ ## NAME);
#define HIRACE_WRAP_DATA(TYPE, NAME) HiRaceDataWrap<TYPE> NAME(__hr_ ## NAME);
#define HIRACE_SET_DATA_GLOBAL(NAME) NAME.setMembers(__hr_ ## NAME, __hr_metadata_ ## NAME, Scope::Global, &__hr_bcount, &__hr_wcount, &__hr_swidx, 1, 0, 0);

#define WARP_SIZE (32U)

//** Internal State Machine (ISM) **//
// Distinct ISM edges
// 25 states * 4 memory actions * 3 sync levels * 4 thread relations
#define TRANSITION_COUNT (1200U)

// ISM states
#define INIT            (0U)
#define READ            (1U)
#define WRITE           (2U)
#define WREAD           (3U)
#define BREAD           (4U)
#define GREAD           (5U)
#define WSYNC           (6U)
#define BSYNC           (7U)
#define MR_WSYNC        (8U)
#define BWSYNC          (9U)
#define B_MR_BSYNC      (10U)  
#define W_MR_BWSYNC     (11U)
#define BATOM_B         (12U)
#define WSYNC_ATOM      (13U)
#define BATOM_W         (14U)
#define WSYNC_ATOM_M    (15U)
#define GATOM_G         (16U)
#define BSYNC_ATOM      (17U)
#define BATOM           (18U)
#define GATOM_B         (19U)
#define BSYNC_ATOM_B    (20U)
#define GATOM           (21U)
#define GATOM_W         (22U)
#define BSYNC_ATOM_W    (23U)
#define RACE            (24U)

// memory operations
#define R  (0U)
#define W  (1U)
#define BA (2U)
#define GA (3U)

// sync scopes
#define UNSYNC    (0U)
#define WARPSYNC  (1U)
#define BLOCKSYNC (2U)

// thread relations
#define SAMETHREAD (0U)
#define SAMEWARP   (1U)
#define SAMEBLOCK  (2U)
#define SAMEGRID   (3U)

// shadow word layout
#define SHADOW_SIZE (64U)

#define TID_BITS    (10U)
#define BLOCK_BITS  (15U)
#define STATE_BITS  (5U)
#define BCOUNT_BITS (16U)
#define WCOUNT_BITS (10U)
#define SWIDX_BITS  (8U)

#define TID_OFFSET    (SHADOW_SIZE - TID_BITS)
#define BLOCK_OFFSET  (TID_OFFSET - BLOCK_BITS)
#define STATE_OFFSET  (BLOCK_OFFSET - STATE_BITS)
#define BCOUNT_OFFSET (STATE_OFFSET - BCOUNT_BITS)
#define WCOUNT_OFFSET (BCOUNT_OFFSET - WCOUNT_BITS)
#define SWIDX_OFFSET  (WCOUNT_OFFSET - SWIDX_BITS)

#define TID_MASK    (((1ULL << TID_BITS) - 1) << TID_OFFSET)
#define BLOCK_MASK  (((1ULL << BLOCK_BITS) - 1) << BLOCK_OFFSET)
#define STATE_MASK  (((1ULL << STATE_BITS) - 1) << STATE_OFFSET)
#define BCOUNT_MASK (((1ULL << BCOUNT_BITS) - 1) << BCOUNT_OFFSET)
#define WCOUNT_MASK (((1ULL << WCOUNT_BITS) - 1) << WCOUNT_OFFSET)
#define SWIDX_MASK  (((1ULL << SWIDX_BITS) - 1) << SWIDX_OFFSET)

// Decomposing one shadow value
#define GET_TID(X)    ((unsigned)((X & TID_MASK)    >> TID_OFFSET))
#define GET_BLOCK(X)  ((unsigned)((X & BLOCK_MASK)  >> BLOCK_OFFSET))
#define GET_STATE(X)  ((unsigned)((X & STATE_MASK)  >> STATE_OFFSET))
#define GET_BCOUNT(X) ((unsigned)((X & BCOUNT_MASK) >> BCOUNT_OFFSET))
#define GET_WCOUNT(X) ((unsigned)((X & WCOUNT_MASK) >> WCOUNT_OFFSET))
#define GET_SWIDX(X)  ((unsigned)((X & SWIDX_MASK)  >> SWIDX_OFFSET))

__device__ unsigned get_tid(hr_shadowt val) {
  return (unsigned)((val & TID_MASK) >> TID_OFFSET);
}
__device__ unsigned get_block(hr_shadowt val) {
  return (unsigned)((val & BLOCK_MASK) >> BLOCK_OFFSET);
}
__device__ unsigned get_state(hr_shadowt val) {
  return (unsigned)((val & STATE_MASK) >> STATE_OFFSET);
}
__device__ unsigned get_bcount(hr_shadowt val) {
  return (unsigned)((val & BCOUNT_MASK) >> BCOUNT_OFFSET);
}
__device__ unsigned get_wcount(hr_shadowt val) {
  return (unsigned)((val & WCOUNT_MASK) >> WCOUNT_OFFSET);
}
__device__ unsigned get_swidx(hr_shadowt val) {
  return (unsigned)((val & SWIDX_MASK) >> SWIDX_OFFSET);
}

// create one shadow value
__device__ hr_shadowt create_shadow(unsigned tid,
                                    unsigned bid,
                                    unsigned state,
                                    unsigned bcount,
                                    unsigned wcount,
                                    unsigned swidx) {
  return (((hr_shadowt)tid    << TID_OFFSET)
        | ((hr_shadowt)bid    << BLOCK_OFFSET)
        | ((hr_shadowt)state  << STATE_OFFSET)
        | ((hr_shadowt)bcount << BCOUNT_OFFSET)
        | ((hr_shadowt)wcount << WCOUNT_OFFSET)
        | ((hr_shadowt)swidx  << SWIDX_OFFSET));
}

// check for values that overflow set bounds
// for example, number of barriers (bCount) 
//   hit at runtime is unbounded and this
//   algorithm may fail once the finite
//   counter overflows
__device__ void
overflow_test(unsigned tid,
              unsigned bid,
              unsigned state,
              unsigned bcount,
              unsigned wcount,
              unsigned swidx) {
  hr_shadowt t  = (hr_shadowt)tid    << TID_OFFSET;
  hr_shadowt b  = (hr_shadowt)bid    << BLOCK_OFFSET;
  hr_shadowt s  = (hr_shadowt)state  << STATE_OFFSET;
  hr_shadowt bc = (hr_shadowt)bcount << BCOUNT_OFFSET;
  hr_shadowt wc = (hr_shadowt)wcount << WCOUNT_OFFSET;
  hr_shadowt sw = (hr_shadowt)swidx  << SWIDX_OFFSET;
  if (!(t  == (t  & TID_MASK))) { printf("overflow tid %d\n", tid); }
  if (!(b  == (b  & BLOCK_MASK))) { printf("overflow bid %d\n", bid); }
  if (!(s  == (s  & STATE_MASK))) { printf("overflow state %d\n", state); }
  if (!(bc == (bc & BCOUNT_MASK))) { printf("overflow bcount %d\n", bcount); }
  if (!(wc == (wc & WCOUNT_MASK))) { printf("overflow wcount %d\n", wcount); }
  if (!(sw == (sw & SWIDX_MASK))) { printf("overflow swidx %d\n", swidx); }
}

// Scan subwarp index array to determine synchronization status
__device__ bool swsync_scan(unsigned  tid, unsigned  bid, unsigned  swidx, unsigned  prev_swidx) {
  return 0; // TODO
}

#ifndef RACECHECK

__device__ void log_access(
    unsigned   tid, 
    unsigned   bid, 
    unsigned   rw_indicator, 
    unsigned   *bcount, 
    unsigned   *wcount, 
    unsigned   *swidx, 
    hr_shadowt *prev_access,
    unsigned   offset,
    int        lineNo,
    const char *fileName) {}


#else // RACECHECK

/*
 * Log access to memory.
 * 
 * For every variable access in user source the accessing thread should
 * call this function to (1) check whether the current access constitutes
 * a data race with some prior access, and (2) record its access for
 * future comparisons.
 * 
 * The accessing threads local metadata will be compared with the shadow
 * value metadata recorded by the prior accessor to determine whether 
 * the current and prior access are concurrent. This is done by determining
 * three properties of the current access:
 * 
 *   1. Current Access type: Current access
 *        (R)ead, 
 *        (W)rite,
 *        (B]lock [A)tomic,
 *        (G]lobal [A)tomic
 *   2. Synchronization Status: Largest scope of synchnorization witnessed
 *      by current accesor since prior access
 *        (U)nsynchronized,
 *        (W)arp of current accessor synchronized,
 *        (B)lock of current accessor synchronized
 *   3. Thread relation: smallest shared scope by current and prior accessor
 *        (S)ame thread as prior accessor,
 *        (W)arp scope,
 *        (B)lock scope,
 *        (G)lobal scope
 * 
 * These 3 access properties, along with the prior ISM state, form a unique
 * transition from the current ISM state to another ISM state. 
 * 
 */
__device__ void log_access(
    unsigned   tid, 
    unsigned   bid, 
    unsigned   rw_indicator, 
    unsigned   *bcount, 
    unsigned   *wcount, 
    unsigned   *swidx, 
    hr_shadowt *prev_access,
    unsigned   offset,
    int        lineNo,
    const char *fileName)
{
  //******** INTERNAL STATE MACHINE ********//
  // ISM Transition Map
  // 
  // INDEX: transition value, formed by prev state and access metadata
  // VALUE: corresponds to next state
  // 
  // index format is a bit array in the following order:
  //  -- prior state | access | sync | thread relation --
  //  prior state:     ID of prior state from shadow balue, out of 25 states
  //  access:          memory action, i.e. Read, Write, Block Atomic, Global Atomic
  //  sync:            Unsynced, Warp synced, Block synced
  //  thread relation: Same thread, Warp scope, Block scope, Global scope
  //  
  //  TODO: Move to header
  static const unsigned char ism_transitions[TRANSITION_COUNT] = {
  //new_st, // old_st|access|sync|thread-rel
    READ, // INIT|R|U|S
    READ, // INIT|R|U|W
    READ, // INIT|R|U|B
    READ, // INIT|R|U|G
    READ, // INIT|R|W|S
    READ, // INIT|R|W|W
    READ, // INIT|R|W|B
    READ, // INIT|R|W|G
    READ, // INIT|R|B|S
    READ, // INIT|R|B|W
    READ, // INIT|R|B|B
    READ, // INIT|R|B|G
    WRITE, // INIT|W|U|S
    WRITE, // INIT|W|U|W
    WRITE, // INIT|W|U|B
    WRITE, // INIT|W|U|G
    WRITE, // INIT|W|W|S
    WRITE, // INIT|W|W|W
    WRITE, // INIT|W|W|B
    WRITE, // INIT|W|W|G
    WRITE, // INIT|W|B|S
    WRITE, // INIT|W|B|W
    WRITE, // INIT|W|B|B
    WRITE, // INIT|W|B|G
    BATOM, // INIT|BA|U|S
    BATOM, // INIT|BA|U|W
    BATOM, // INIT|BA|U|B
    BATOM, // INIT|BA|U|G
    BATOM, // INIT|BA|W|S
    BATOM, // INIT|BA|W|W
    BATOM, // INIT|BA|W|B
    BATOM, // INIT|BA|W|G
    BATOM, // INIT|BA|B|S
    BATOM, // INIT|BA|B|W
    BATOM, // INIT|BA|B|B
    BATOM, // INIT|BA|B|G
    GATOM, // INIT|GA|U|S
    GATOM, // INIT|GA|U|W
    GATOM, // INIT|GA|U|B
    GATOM, // INIT|GA|U|G
    GATOM, // INIT|GA|W|S
    GATOM, // INIT|GA|W|W
    GATOM, // INIT|GA|W|B
    GATOM, // INIT|GA|W|G
    GATOM, // INIT|GA|B|S
    GATOM, // INIT|GA|B|W
    GATOM, // INIT|GA|B|B
    GATOM, // INIT|GA|B|G
    READ, // READ|R|U|S
    WREAD, // READ|R|U|W
    BREAD, // READ|R|U|B
    GREAD, // READ|R|U|G
    READ, // READ|R|W|S
    READ, // READ|R|W|W
    BREAD, // READ|R|W|B
    GREAD, // READ|R|W|G
    READ, // READ|R|B|S
    READ, // READ|R|B|W
    READ, // READ|R|B|B
    GREAD, // READ|R|B|G
    WRITE, // READ|W|U|S
    RACE, // READ|W|U|W
    RACE, // READ|W|U|B
    RACE, // READ|W|U|G
    WRITE, // READ|W|W|S
    WRITE, // READ|W|W|W
    RACE, // READ|W|W|B
    RACE, // READ|W|W|G
    WRITE, // READ|W|B|S
    WRITE, // READ|W|B|W
    WRITE, // READ|W|B|B
    RACE, // READ|W|B|G
    WRITE, // READ|BA|U|S
    RACE, // READ|BA|U|W
    RACE, // READ|BA|U|B
    RACE, // READ|BA|U|G
    WRITE, // READ|BA|W|S
    WRITE, // READ|BA|W|W
    RACE, // READ|BA|W|B
    RACE, // READ|BA|W|G
    WRITE, // READ|BA|B|S
    WRITE, // READ|BA|B|W
    WRITE, // READ|BA|B|B
    RACE, // READ|BA|B|G
    WRITE, // READ|GA|U|S
    RACE, // READ|GA|U|W
    RACE, // READ|GA|U|B
    RACE, // READ|GA|U|G
    WRITE, // READ|GA|W|S
    WRITE, // READ|GA|W|W
    RACE, // READ|GA|W|B
    RACE, // READ|GA|W|G
    WRITE, // READ|GA|B|S
    WRITE, // READ|GA|B|W
    WRITE, // READ|GA|B|B
    RACE, // READ|GA|B|G
    WRITE, // WRITE|R|U|S
    RACE, // WRITE|R|U|W
    RACE, // WRITE|R|U|B
    RACE, // WRITE|R|U|G
    WSYNC, // WRITE|R|W|S
    WSYNC, // WRITE|R|W|W
    RACE, // WRITE|R|W|B
    RACE, // WRITE|R|W|G
    BSYNC, // WRITE|R|B|S
    BSYNC, // WRITE|R|B|W
    BSYNC, // WRITE|R|B|B
    RACE, // WRITE|R|B|G
    WRITE, // WRITE|W|U|S
    RACE, // WRITE|W|U|W
    RACE, // WRITE|W|U|B
    RACE, // WRITE|W|U|G
    WRITE, // WRITE|W|W|S
    WRITE, // WRITE|W|W|W
    RACE, // WRITE|W|W|B
    RACE, // WRITE|W|W|G
    WRITE, // WRITE|W|B|S
    WRITE, // WRITE|W|B|W
    WRITE, // WRITE|W|B|B
    RACE, // WRITE|W|B|G
    WRITE, // WRITE|BA|U|S
    RACE, // WRITE|BA|U|W
    RACE, // WRITE|BA|U|B
    RACE, // WRITE|BA|U|G
    WSYNC_ATOM, // WRITE|BA|W|S
    WSYNC_ATOM, // WRITE|BA|W|W
    RACE, // WRITE|BA|W|B
    RACE, // WRITE|BA|W|G
    BSYNC_ATOM, // WRITE|BA|B|S
    BSYNC_ATOM, // WRITE|BA|B|W
    BSYNC_ATOM, // WRITE|BA|B|B
    RACE, // WRITE|BA|B|G
    WRITE, // WRITE|GA|U|S
    RACE, // WRITE|GA|U|W
    RACE, // WRITE|GA|U|B
    RACE, // WRITE|GA|U|G
    WSYNC_ATOM, // WRITE|GA|W|S
    WSYNC_ATOM, // WRITE|GA|W|W
    RACE, // WRITE|GA|W|B
    RACE, // WRITE|GA|W|G
    BSYNC_ATOM, // WRITE|GA|B|S
    BSYNC_ATOM, // WRITE|GA|B|W
    BSYNC_ATOM, // WRITE|GA|B|B
    RACE, // WRITE|GA|B|G
    WREAD, // WREAD|R|U|S
    WREAD, // WREAD|R|U|W
    BREAD, // WREAD|R|U|B
    GREAD, // WREAD|R|U|G
    READ, // WREAD|R|W|S
    READ, // WREAD|R|W|W
    BREAD, // WREAD|R|W|B
    GREAD, // WREAD|R|W|G
    READ, // WREAD|R|B|S
    READ, // WREAD|R|B|W
    READ, // WREAD|R|B|B
    GREAD, // WREAD|R|B|G
    RACE, // WREAD|W|U|S
    RACE, // WREAD|W|U|W
    RACE, // WREAD|W|U|B
    RACE, // WREAD|W|U|G
    WRITE, // WREAD|W|W|S
    WRITE, // WREAD|W|W|W
    RACE, // WREAD|W|W|B
    RACE, // WREAD|W|W|G
    WRITE, // WREAD|W|B|S
    WRITE, // WREAD|W|B|W
    WRITE, // WREAD|W|B|B
    RACE, // WREAD|W|B|G
    RACE, // WREAD|BA|U|S
    RACE, // WREAD|BA|U|W
    RACE, // WREAD|BA|U|B
    RACE, // WREAD|BA|U|G
    WSYNC_ATOM, // WREAD|BA|W|S
    WSYNC_ATOM, // WREAD|BA|W|W
    RACE, // WREAD|BA|W|B
    RACE, // WREAD|BA|W|G
    BSYNC_ATOM, // WREAD|BA|B|S
    BSYNC_ATOM, // WREAD|BA|B|W
    BSYNC_ATOM, // WREAD|BA|B|B
    RACE, // WREAD|BA|B|G
    RACE, // WREAD|GA|U|S
    RACE, // WREAD|GA|U|W
    RACE, // WREAD|GA|U|B
    RACE, // WREAD|GA|U|G
    WSYNC_ATOM, // WREAD|GA|W|S
    WSYNC_ATOM, // WREAD|GA|W|W
    RACE, // WREAD|GA|W|B
    RACE, // WREAD|GA|W|G
    BSYNC_ATOM, // WREAD|GA|B|S
    BSYNC_ATOM, // WREAD|GA|B|W
    BSYNC_ATOM, // WREAD|GA|B|B
    RACE, // WREAD|GA|B|G
    BREAD, // BREAD|R|U|S
    BREAD, // BREAD|R|U|W
    BREAD, // BREAD|R|U|B
    GREAD, // BREAD|R|U|G
    BREAD, // BREAD|R|W|S
    BREAD, // BREAD|R|W|W
    BREAD, // BREAD|R|W|B
    GREAD, // BREAD|R|W|G
    READ, // BREAD|R|B|S
    READ, // BREAD|R|B|W
    READ, // BREAD|R|B|B
    GREAD, // BREAD|R|B|G
    RACE, // BREAD|W|U|S
    RACE, // BREAD|W|U|W
    RACE, // BREAD|W|U|B
    RACE, // BREAD|W|U|G
    RACE, // BREAD|W|W|S
    RACE, // BREAD|W|W|W
    RACE, // BREAD|W|W|B
    RACE, // BREAD|W|W|G
    WRITE, // BREAD|W|B|S
    WRITE, // BREAD|W|B|W
    WRITE, // BREAD|W|B|B
    RACE, // BREAD|W|B|G
    RACE, // BREAD|BA|U|S
    RACE, // BREAD|BA|U|W
    RACE, // BREAD|BA|U|B
    RACE, // BREAD|BA|U|G
    RACE, // BREAD|BA|W|S
    RACE, // BREAD|BA|W|W
    RACE, // BREAD|BA|W|B
    RACE, // BREAD|BA|W|G
    BSYNC_ATOM, // BREAD|BA|B|S
    BSYNC_ATOM, // BREAD|BA|B|W
    BSYNC_ATOM, // BREAD|BA|B|B
    RACE, // BREAD|BA|B|G
    RACE, // BREAD|GA|U|S
    RACE, // BREAD|GA|U|W
    RACE, // BREAD|GA|U|B
    RACE, // BREAD|GA|U|G
    RACE, // BREAD|GA|W|S
    RACE, // BREAD|GA|W|W
    RACE, // BREAD|GA|W|B
    RACE, // BREAD|GA|W|G
    BSYNC_ATOM, // BREAD|GA|B|S
    BSYNC_ATOM, // BREAD|GA|B|W
    BSYNC_ATOM, // BREAD|GA|B|B
    RACE, // BREAD|GA|B|G
    GREAD, // GREAD|R|U|S
    GREAD, // GREAD|R|U|W
    GREAD, // GREAD|R|U|B
    GREAD, // GREAD|R|U|G
    GREAD, // GREAD|R|W|S
    GREAD, // GREAD|R|W|W
    GREAD, // GREAD|R|W|B
    GREAD, // GREAD|R|W|G
    GREAD, // GREAD|R|B|S
    GREAD, // GREAD|R|B|W
    GREAD, // GREAD|R|B|B
    GREAD, // GREAD|R|B|G
    RACE, // GREAD|W|U|S
    RACE, // GREAD|W|U|W
    RACE, // GREAD|W|U|B
    RACE, // GREAD|W|U|G
    RACE, // GREAD|W|W|S
    RACE, // GREAD|W|W|W
    RACE, // GREAD|W|W|B
    RACE, // GREAD|W|W|G
    RACE, // GREAD|W|B|S
    RACE, // GREAD|W|B|W
    RACE, // GREAD|W|B|B
    RACE, // GREAD|W|B|G
    RACE, // GREAD|BA|U|S
    RACE, // GREAD|BA|U|W
    RACE, // GREAD|BA|U|B
    RACE, // GREAD|BA|U|G
    RACE, // GREAD|BA|W|S
    RACE, // GREAD|BA|W|W
    RACE, // GREAD|BA|W|B
    RACE, // GREAD|BA|W|G
    RACE, // GREAD|BA|B|S
    RACE, // GREAD|BA|B|W
    RACE, // GREAD|BA|B|B
    RACE, // GREAD|BA|B|G
    RACE, // GREAD|GA|U|S
    RACE, // GREAD|GA|U|W
    RACE, // GREAD|GA|U|B
    RACE, // GREAD|GA|U|G
    RACE, // GREAD|GA|W|S
    RACE, // GREAD|GA|W|W
    RACE, // GREAD|GA|W|B
    RACE, // GREAD|GA|W|G
    RACE, // GREAD|GA|B|S
    RACE, // GREAD|GA|B|W
    RACE, // GREAD|GA|B|B
    RACE, // GREAD|GA|B|G
    WSYNC, // WSYNC|R|U|S
    MR_WSYNC, // WSYNC|R|U|W
    RACE, // WSYNC|R|U|B
    RACE, // WSYNC|R|U|G
    WSYNC, // WSYNC|R|W|S
    WSYNC, // WSYNC|R|W|W
    RACE, // WSYNC|R|W|B
    RACE, // WSYNC|R|W|G
    BSYNC, // WSYNC|R|B|S
    BSYNC, // WSYNC|R|B|W
    BSYNC, // WSYNC|R|B|B
    RACE, // WSYNC|R|B|G
    WRITE, // WSYNC|W|U|S
    RACE, // WSYNC|W|U|W
    RACE, // WSYNC|W|U|B
    RACE, // WSYNC|W|U|G
    WRITE, // WSYNC|W|W|S
    WRITE, // WSYNC|W|W|W
    RACE, // WSYNC|W|W|B
    RACE, // WSYNC|W|W|G
    WRITE, // WSYNC|W|B|S
    WRITE, // WSYNC|W|B|W
    WRITE, // WSYNC|W|B|B
    RACE, // WSYNC|W|B|G
    WRITE, // WSYNC|BA|U|S
    RACE, // WSYNC|BA|U|W
    RACE, // WSYNC|BA|U|B
    RACE, // WSYNC|BA|U|G
    WSYNC_ATOM, // WSYNC|BA|W|S
    WSYNC_ATOM, // WSYNC|BA|W|W
    RACE, // WSYNC|BA|W|B
    RACE, // WSYNC|BA|W|G
    BSYNC_ATOM, // WSYNC|BA|B|S
    BSYNC_ATOM, // WSYNC|BA|B|W
    BSYNC_ATOM, // WSYNC|BA|B|B
    RACE, // WSYNC|BA|B|G
    WRITE, // WSYNC|GA|U|S
    RACE, // WSYNC|GA|U|W
    RACE, // WSYNC|GA|U|B
    RACE, // WSYNC|GA|U|G
    WSYNC_ATOM, // WSYNC|GA|W|S
    WSYNC_ATOM, // WSYNC|GA|W|W
    RACE, // WSYNC|GA|W|B
    RACE, // WSYNC|GA|W|G
    BSYNC_ATOM, // WSYNC|GA|B|S
    BSYNC_ATOM, // WSYNC|GA|B|W
    BSYNC_ATOM, // WSYNC|GA|B|B
    RACE, // WSYNC|GA|B|G
    BSYNC, // BSYNC|R|U|S
    W_MR_BWSYNC, // BSYNC|R|U|W
    B_MR_BSYNC, // BSYNC|R|U|B
    RACE, // BSYNC|R|U|G
    BWSYNC, // BSYNC|R|W|S
    BWSYNC, // BSYNC|R|W|W
    B_MR_BSYNC, // BSYNC|R|W|B
    RACE, // BSYNC|R|W|G
    BSYNC, // BSYNC|R|B|S
    BSYNC, // BSYNC|R|B|W
    BSYNC, // BSYNC|R|B|B
    RACE, // BSYNC|R|B|G
    WRITE, // BSYNC|W|U|S
    RACE, // BSYNC|W|U|W
    RACE, // BSYNC|W|U|B
    RACE, // BSYNC|W|U|G
    WRITE, // BSYNC|W|W|S
    WRITE, // BSYNC|W|W|W
    RACE, // BSYNC|W|W|B
    RACE, // BSYNC|W|W|G
    WRITE, // BSYNC|W|B|S
    WRITE, // BSYNC|W|B|W
    WRITE, // BSYNC|W|B|B
    RACE, // BSYNC|W|B|G
    WRITE, // BSYNC|BA|U|S
    RACE, // BSYNC|BA|U|W
    RACE, // BSYNC|BA|U|B
    RACE, // BSYNC|BA|U|G
    WSYNC_ATOM, // BSYNC|BA|W|S
    WSYNC_ATOM, // BSYNC|BA|W|W
    RACE, // BSYNC|BA|W|B
    RACE, // BSYNC|BA|W|G
    BSYNC_ATOM, // BSYNC|BA|B|S
    BSYNC_ATOM, // BSYNC|BA|B|W
    BSYNC_ATOM, // BSYNC|BA|B|B
    RACE, // BSYNC|BA|B|G
    WRITE, // BSYNC|GA|U|S
    RACE, // BSYNC|GA|U|W
    RACE, // BSYNC|GA|U|B
    RACE, // BSYNC|GA|U|G
    WSYNC_ATOM, // BSYNC|GA|W|S
    WSYNC_ATOM, // BSYNC|GA|W|W
    RACE, // BSYNC|GA|W|B
    RACE, // BSYNC|GA|W|G
    BSYNC_ATOM, // BSYNC|GA|B|S
    BSYNC_ATOM, // BSYNC|GA|B|W
    BSYNC_ATOM, // BSYNC|GA|B|B
    RACE, // BSYNC|GA|B|G
    MR_WSYNC, // MR_WSYNC|R|U|S
    MR_WSYNC, // MR_WSYNC|R|U|W
    RACE, // MR_WSYNC|R|U|B
    RACE, // MR_WSYNC|R|U|G
    WSYNC, // MR_WSYNC|R|W|S
    WSYNC, // MR_WSYNC|R|W|W
    RACE, // MR_WSYNC|R|W|B
    RACE, // MR_WSYNC|R|W|G
    BSYNC, // MR_WSYNC|R|B|S
    BSYNC, // MR_WSYNC|R|B|W
    BSYNC, // MR_WSYNC|R|B|B
    RACE, // MR_WSYNC|R|B|G
    RACE, // MR_WSYNC|W|U|S
    RACE, // MR_WSYNC|W|U|W
    RACE, // MR_WSYNC|W|U|B
    RACE, // MR_WSYNC|W|U|G
    WRITE, // MR_WSYNC|W|W|S
    WRITE, // MR_WSYNC|W|W|W
    RACE, // MR_WSYNC|W|W|B
    RACE, // MR_WSYNC|W|W|G
    WRITE, // MR_WSYNC|W|B|S
    WRITE, // MR_WSYNC|W|B|W
    WRITE, // MR_WSYNC|W|B|B
    RACE, // MR_WSYNC|W|B|G
    RACE, // MR_WSYNC|BA|U|S
    RACE, // MR_WSYNC|BA|U|W
    RACE, // MR_WSYNC|BA|U|B
    RACE, // MR_WSYNC|BA|U|G
    WSYNC_ATOM, // MR_WSYNC|BA|W|S
    WSYNC_ATOM, // MR_WSYNC|BA|W|W
    RACE, // MR_WSYNC|BA|W|B
    RACE, // MR_WSYNC|BA|W|G
    BSYNC_ATOM, // MR_WSYNC|BA|B|S
    BSYNC_ATOM, // MR_WSYNC|BA|B|W
    BSYNC_ATOM, // MR_WSYNC|BA|B|B
    RACE, // MR_WSYNC|BA|B|G
    RACE, // MR_WSYNC|GA|U|S
    RACE, // MR_WSYNC|GA|U|W
    RACE, // MR_WSYNC|GA|U|B
    RACE, // MR_WSYNC|GA|U|G
    WSYNC_ATOM, // MR_WSYNC|GA|W|S
    WSYNC_ATOM, // MR_WSYNC|GA|W|W
    RACE, // MR_WSYNC|GA|W|B
    RACE, // MR_WSYNC|GA|W|G
    BSYNC_ATOM, // MR_WSYNC|GA|B|S
    BSYNC_ATOM, // MR_WSYNC|GA|B|W
    BSYNC_ATOM, // MR_WSYNC|GA|B|B
    RACE, // MR_WSYNC|GA|B|G
    BWSYNC, // BWSYNC|R|U|S
    W_MR_BWSYNC, // BWSYNC|R|U|W
    B_MR_BSYNC, // BWSYNC|R|U|B
    RACE, // BWSYNC|R|U|G
    BWSYNC, // BWSYNC|R|W|S
    BWSYNC, // BWSYNC|R|W|W
    B_MR_BSYNC, // BWSYNC|R|W|B
    RACE, // BWSYNC|R|W|G
    BSYNC, // BWSYNC|R|B|S
    BSYNC, // BWSYNC|R|B|W
    BSYNC, // BWSYNC|R|B|B
    RACE, // BWSYNC|R|B|G
    WRITE, // BWSYNC|W|U|S
    RACE, // BWSYNC|W|U|W
    RACE, // BWSYNC|W|U|B
    RACE, // BWSYNC|W|U|G
    WRITE, // BWSYNC|W|W|S
    WRITE, // BWSYNC|W|W|W
    RACE, // BWSYNC|W|W|B
    RACE, // BWSYNC|W|W|G
    WRITE, // BWSYNC|W|B|S
    WRITE, // BWSYNC|W|B|W
    WRITE, // BWSYNC|W|B|B
    RACE, // BWSYNC|W|B|G
    WRITE, // BWSYNC|BA|U|S
    RACE, // BWSYNC|BA|U|W
    RACE, // BWSYNC|BA|U|B
    RACE, // BWSYNC|BA|U|G
    WSYNC_ATOM, // BWSYNC|BA|W|S
    WSYNC_ATOM, // BWSYNC|BA|W|W
    RACE, // BWSYNC|BA|W|B
    RACE, // BWSYNC|BA|W|G
    BSYNC_ATOM, // BWSYNC|BA|B|S
    BSYNC_ATOM, // BWSYNC|BA|B|W
    BSYNC_ATOM, // BWSYNC|BA|B|B
    RACE, // BWSYNC|BA|B|G
    WRITE, // BWSYNC|GA|U|S
    RACE, // BWSYNC|GA|U|W
    RACE, // BWSYNC|GA|U|B
    RACE, // BWSYNC|GA|U|G
    WSYNC_ATOM, // BWSYNC|GA|W|S
    WSYNC_ATOM, // BWSYNC|GA|W|W
    RACE, // BWSYNC|GA|W|B
    RACE, // BWSYNC|GA|W|G
    BSYNC_ATOM, // BWSYNC|GA|B|S
    BSYNC_ATOM, // BWSYNC|GA|B|W
    BSYNC_ATOM, // BWSYNC|GA|B|B
    RACE, // BWSYNC|GA|B|G
    B_MR_BSYNC, // B_MR_BSYNC|R|U|S
    B_MR_BSYNC, // B_MR_BSYNC|R|U|W
    B_MR_BSYNC, // B_MR_BSYNC|R|U|B
    RACE, // B_MR_BSYNC|R|U|G
    B_MR_BSYNC, // B_MR_BSYNC|R|W|S
    B_MR_BSYNC, // B_MR_BSYNC|R|W|W
    B_MR_BSYNC, // B_MR_BSYNC|R|W|B
    RACE, // B_MR_BSYNC|R|W|G
    BSYNC, // B_MR_BSYNC|R|B|S
    BSYNC, // B_MR_BSYNC|R|B|W
    BSYNC, // B_MR_BSYNC|R|B|B
    RACE, // B_MR_BSYNC|R|B|G
    RACE, // B_MR_BSYNC|W|U|S
    RACE, // B_MR_BSYNC|W|U|W
    RACE, // B_MR_BSYNC|W|U|B
    RACE, // B_MR_BSYNC|W|U|G
    RACE, // B_MR_BSYNC|W|W|S
    RACE, // B_MR_BSYNC|W|W|W
    RACE, // B_MR_BSYNC|W|W|B
    RACE, // B_MR_BSYNC|W|W|G
    WRITE, // B_MR_BSYNC|W|B|S
    WRITE, // B_MR_BSYNC|W|B|W
    WRITE, // B_MR_BSYNC|W|B|B
    RACE, // B_MR_BSYNC|W|B|G
    RACE, // B_MR_BSYNC|BA|U|S
    RACE, // B_MR_BSYNC|BA|U|W
    RACE, // B_MR_BSYNC|BA|U|B
    RACE, // B_MR_BSYNC|BA|U|G
    RACE, // B_MR_BSYNC|BA|W|S
    RACE, // B_MR_BSYNC|BA|W|W
    RACE, // B_MR_BSYNC|BA|W|B
    RACE, // B_MR_BSYNC|BA|W|G
    BSYNC_ATOM, // B_MR_BSYNC|BA|B|S
    BSYNC_ATOM, // B_MR_BSYNC|BA|B|W
    BSYNC_ATOM, // B_MR_BSYNC|BA|B|B
    RACE, // B_MR_BSYNC|BA|B|G
    RACE, // B_MR_BSYNC|GA|U|S
    RACE, // B_MR_BSYNC|GA|U|W
    RACE, // B_MR_BSYNC|GA|U|B
    RACE, // B_MR_BSYNC|GA|U|G
    RACE, // B_MR_BSYNC|GA|W|S
    RACE, // B_MR_BSYNC|GA|W|W
    RACE, // B_MR_BSYNC|GA|W|B
    RACE, // B_MR_BSYNC|GA|W|G
    BSYNC_ATOM, // B_MR_BSYNC|GA|B|S
    BSYNC_ATOM, // B_MR_BSYNC|GA|B|W
    BSYNC_ATOM, // B_MR_BSYNC|GA|B|B
    RACE, // B_MR_BSYNC|GA|B|G
    W_MR_BWSYNC, // W_MR_BWSYNC|R|U|S
    W_MR_BWSYNC, // W_MR_BWSYNC|R|U|W
    B_MR_BSYNC, // W_MR_BWSYNC|R|U|B
    RACE, // W_MR_BWSYNC|R|U|G
    BWSYNC, // W_MR_BWSYNC|R|W|S
    BWSYNC, // W_MR_BWSYNC|R|W|W
    B_MR_BSYNC, // W_MR_BWSYNC|R|W|B
    RACE, // W_MR_BWSYNC|R|W|G
    BSYNC, // W_MR_BWSYNC|R|B|S
    BSYNC, // W_MR_BWSYNC|R|B|W
    BSYNC, // W_MR_BWSYNC|R|B|B
    RACE, // W_MR_BWSYNC|R|B|G
    RACE, // W_MR_BWSYNC|W|U|S
    RACE, // W_MR_BWSYNC|W|U|W
    RACE, // W_MR_BWSYNC|W|U|B
    RACE, // W_MR_BWSYNC|W|U|G
    WRITE, // W_MR_BWSYNC|W|W|S
    WRITE, // W_MR_BWSYNC|W|W|W
    RACE, // W_MR_BWSYNC|W|W|B
    RACE, // W_MR_BWSYNC|W|W|G
    WRITE, // W_MR_BWSYNC|W|B|S
    WRITE, // W_MR_BWSYNC|W|B|W
    WRITE, // W_MR_BWSYNC|W|B|B
    RACE, // W_MR_BWSYNC|W|B|G
    RACE, // W_MR_BWSYNC|BA|U|S
    RACE, // W_MR_BWSYNC|BA|U|W
    RACE, // W_MR_BWSYNC|BA|U|B
    RACE, // W_MR_BWSYNC|BA|U|G
    WSYNC_ATOM, // W_MR_BWSYNC|BA|W|S
    WSYNC_ATOM, // W_MR_BWSYNC|BA|W|W
    RACE, // W_MR_BWSYNC|BA|W|B
    RACE, // W_MR_BWSYNC|BA|W|G
    BSYNC_ATOM, // W_MR_BWSYNC|BA|B|S
    BSYNC_ATOM, // W_MR_BWSYNC|BA|B|W
    BSYNC_ATOM, // W_MR_BWSYNC|BA|B|B
    RACE, // W_MR_BWSYNC|BA|B|G
    RACE, // W_MR_BWSYNC|GA|U|S
    RACE, // W_MR_BWSYNC|GA|U|W
    RACE, // W_MR_BWSYNC|GA|U|B
    RACE, // W_MR_BWSYNC|GA|U|G
    WSYNC_ATOM, // W_MR_BWSYNC|GA|W|S
    WSYNC_ATOM, // W_MR_BWSYNC|GA|W|W
    RACE, // W_MR_BWSYNC|GA|W|B
    RACE, // W_MR_BWSYNC|GA|W|G
    BSYNC_ATOM, // W_MR_BWSYNC|GA|B|S
    BSYNC_ATOM, // W_MR_BWSYNC|GA|B|W
    BSYNC_ATOM, // W_MR_BWSYNC|GA|B|B
    RACE, // W_MR_BWSYNC|GA|B|G
    RACE, // BATOM_B|R|U|S
    RACE, // BATOM_B|R|U|W
    RACE, // BATOM_B|R|U|B
    RACE, // BATOM_B|R|U|G
    RACE, // BATOM_B|R|W|S
    RACE, // BATOM_B|R|W|W
    RACE, // BATOM_B|R|W|B
    RACE, // BATOM_B|R|W|G
    BSYNC, // BATOM_B|R|B|S
    BSYNC, // BATOM_B|R|B|W
    BSYNC, // BATOM_B|R|B|B
    RACE, // BATOM_B|R|B|G
    RACE, // BATOM_B|W|U|S
    RACE, // BATOM_B|W|U|W
    RACE, // BATOM_B|W|U|B
    RACE, // BATOM_B|W|U|G
    RACE, // BATOM_B|W|W|S
    RACE, // BATOM_B|W|W|W
    RACE, // BATOM_B|W|W|B
    RACE, // BATOM_B|W|W|G
    WRITE, // BATOM_B|W|B|S
    WRITE, // BATOM_B|W|B|W
    WRITE, // BATOM_B|W|B|B
    RACE, // BATOM_B|W|B|G
    BATOM_B, // BATOM_B|BA|U|S
    BATOM_B, // BATOM_B|BA|U|W
    BATOM_B, // BATOM_B|BA|U|B
    RACE, // BATOM_B|BA|U|G
    BATOM_B, // BATOM_B|BA|W|S
    BATOM_B, // BATOM_B|BA|W|W
    BATOM_B, // BATOM_B|BA|W|B
    RACE, // BATOM_B|BA|W|G
    BATOM_B, // BATOM_B|BA|B|S
    BATOM_B, // BATOM_B|BA|B|W
    BATOM, // BATOM_B|BA|B|B
    RACE, // BATOM_B|BA|B|G
    BATOM_B, // BATOM_B|GA|U|S
    BATOM_B, // BATOM_B|GA|U|W
    BATOM_B, // BATOM_B|GA|U|B
    RACE, // BATOM_B|GA|U|G
    BATOM_B, // BATOM_B|GA|W|S
    BATOM_B, // BATOM_B|GA|W|W
    BATOM_B, // BATOM_B|GA|W|B
    RACE, // BATOM_B|GA|W|G
    BATOM_B, // BATOM_B|GA|B|S
    BATOM_B, // BATOM_B|GA|B|W
    BATOM, // BATOM_B|GA|B|B
    RACE, // BATOM_B|GA|B|G
    WRITE, // WSYNC_ATOM|R|U|S
    RACE, // WSYNC_ATOM|R|U|W
    RACE, // WSYNC_ATOM|R|U|B
    RACE, // WSYNC_ATOM|R|U|G
    WSYNC, // WSYNC_ATOM|R|W|S
    WSYNC, // WSYNC_ATOM|R|W|W
    RACE, // WSYNC_ATOM|R|W|B
    RACE, // WSYNC_ATOM|R|W|G
    BSYNC, // WSYNC_ATOM|R|B|S
    BSYNC, // WSYNC_ATOM|R|B|W
    BSYNC, // WSYNC_ATOM|R|B|B
    RACE, // WSYNC_ATOM|R|B|G
    WRITE, // WSYNC_ATOM|W|U|S
    RACE, // WSYNC_ATOM|W|U|W
    RACE, // WSYNC_ATOM|W|U|B
    RACE, // WSYNC_ATOM|W|U|G
    WRITE, // WSYNC_ATOM|W|W|S
    WRITE, // WSYNC_ATOM|W|W|W
    RACE, // WSYNC_ATOM|W|W|B
    RACE, // WSYNC_ATOM|W|W|G
    WRITE, // WSYNC_ATOM|W|B|S
    WRITE, // WSYNC_ATOM|W|B|W
    WRITE, // WSYNC_ATOM|W|B|B
    RACE, // WSYNC_ATOM|W|B|G
    WSYNC_ATOM, // WSYNC_ATOM|BA|U|S
    WSYNC_ATOM_M, // WSYNC_ATOM|BA|U|W
    RACE, // WSYNC_ATOM|BA|U|B
    RACE, // WSYNC_ATOM|BA|U|G
    WSYNC_ATOM, // WSYNC_ATOM|BA|W|S
    WSYNC_ATOM, // WSYNC_ATOM|BA|W|W
    RACE, // WSYNC_ATOM|BA|W|B
    RACE, // WSYNC_ATOM|BA|W|G
    BSYNC_ATOM, // WSYNC_ATOM|BA|B|S
    BSYNC_ATOM, // WSYNC_ATOM|BA|B|W
    BSYNC_ATOM, // WSYNC_ATOM|BA|B|B
    RACE, // WSYNC_ATOM|BA|B|G
    WSYNC_ATOM, // WSYNC_ATOM|GA|U|S
    WSYNC_ATOM, // WSYNC_ATOM|GA|U|W
    RACE, // WSYNC_ATOM|GA|U|B
    RACE, // WSYNC_ATOM|GA|U|G
    WSYNC_ATOM, // WSYNC_ATOM|GA|W|S
    WSYNC_ATOM, // WSYNC_ATOM|GA|W|W
    RACE, // WSYNC_ATOM|GA|W|B
    RACE, // WSYNC_ATOM|GA|W|G
    BSYNC_ATOM, // WSYNC_ATOM|GA|B|S
    BSYNC_ATOM, // WSYNC_ATOM|GA|B|W
    BSYNC_ATOM, // WSYNC_ATOM|GA|B|B
    RACE, // WSYNC_ATOM|GA|B|G
    RACE, // BATOM_W|R|U|S
    RACE, // BATOM_W|R|U|W
    RACE, // BATOM_W|R|U|B
    RACE, // BATOM_W|R|U|G
    WSYNC, // BATOM_W|R|W|S
    WSYNC, // BATOM_W|R|W|W
    RACE, // BATOM_W|R|W|B
    RACE, // BATOM_W|R|W|G
    BSYNC, // BATOM_W|R|B|S
    BSYNC, // BATOM_W|R|B|W
    BSYNC, // BATOM_W|R|B|B
    RACE, // BATOM_W|R|B|G
    RACE, // BATOM_W|W|U|S
    RACE, // BATOM_W|W|U|W
    RACE, // BATOM_W|W|U|B
    RACE, // BATOM_W|W|U|G
    WRITE, // BATOM_W|W|W|S
    WRITE, // BATOM_W|W|W|W
    RACE, // BATOM_W|W|W|B
    RACE, // BATOM_W|W|W|G
    WRITE, // BATOM_W|W|B|S
    WRITE, // BATOM_W|W|B|W
    WRITE, // BATOM_W|W|B|B
    RACE, // BATOM_W|W|B|G
    BATOM_W, // BATOM_W|BA|U|S
    BATOM_W, // BATOM_W|BA|U|W
    BATOM_B, // BATOM_W|BA|U|B
    RACE, // BATOM_W|BA|U|G
    BATOM, // BATOM_W|BA|W|S
    BATOM, // BATOM_W|BA|W|W
    BATOM_B, // BATOM_W|BA|W|B
    RACE, // BATOM_W|BA|W|G
    BATOM, // BATOM_W|BA|B|S
    BATOM, // BATOM_W|BA|B|W
    BATOM, // BATOM_W|BA|B|B
    RACE, // BATOM_W|BA|B|G
    BATOM_W, // BATOM_W|GA|U|S
    BATOM_W, // BATOM_W|GA|U|W
    BATOM_B, // BATOM_W|GA|U|B
    RACE, // BATOM_W|GA|U|G
    BATOM, // BATOM_W|GA|W|S
    BATOM, // BATOM_W|GA|W|W
    BATOM_B, // BATOM_W|GA|W|B
    RACE, // BATOM_W|GA|W|G
    BATOM, // BATOM_W|GA|B|S
    BATOM, // BATOM_W|GA|B|W
    BATOM, // BATOM_W|GA|B|B
    RACE, // BATOM_W|GA|B|G
    RACE, // WSYNC_ATOM_M|R|U|S
    RACE, // WSYNC_ATOM_M|R|U|W
    RACE, // WSYNC_ATOM_M|R|U|B
    RACE, // WSYNC_ATOM_M|R|U|G
    WSYNC, // WSYNC_ATOM_M|R|W|S
    WSYNC, // WSYNC_ATOM_M|R|W|W
    RACE, // WSYNC_ATOM_M|R|W|B
    RACE, // WSYNC_ATOM_M|R|W|G
    BSYNC, // WSYNC_ATOM_M|R|B|S
    BSYNC, // WSYNC_ATOM_M|R|B|W
    BSYNC, // WSYNC_ATOM_M|R|B|B
    RACE, // WSYNC_ATOM_M|R|B|G
    RACE, // WSYNC_ATOM_M|W|U|S
    RACE, // WSYNC_ATOM_M|W|U|W
    RACE, // WSYNC_ATOM_M|W|U|B
    RACE, // WSYNC_ATOM_M|W|U|G
    WRITE, // WSYNC_ATOM_M|W|W|S
    WRITE, // WSYNC_ATOM_M|W|W|W
    RACE, // WSYNC_ATOM_M|W|W|B
    RACE, // WSYNC_ATOM_M|W|W|G
    WRITE, // WSYNC_ATOM_M|W|B|S
    WRITE, // WSYNC_ATOM_M|W|B|W
    WRITE, // WSYNC_ATOM_M|W|B|B
    RACE, // WSYNC_ATOM_M|W|B|G
    WSYNC_ATOM_M, // WSYNC_ATOM_M|BA|U|S
    WSYNC_ATOM_M, // WSYNC_ATOM_M|BA|U|W
    RACE, // WSYNC_ATOM_M|BA|U|B
    RACE, // WSYNC_ATOM_M|BA|U|G
    WSYNC_ATOM, // WSYNC_ATOM_M|BA|W|S
    WSYNC_ATOM, // WSYNC_ATOM_M|BA|W|W
    RACE, // WSYNC_ATOM_M|BA|W|B
    RACE, // WSYNC_ATOM_M|BA|W|G
    BSYNC_ATOM, // WSYNC_ATOM_M|BA|B|S
    BSYNC_ATOM, // WSYNC_ATOM_M|BA|B|W
    BSYNC_ATOM, // WSYNC_ATOM_M|BA|B|B
    RACE, // WSYNC_ATOM_M|BA|B|G
    WSYNC_ATOM_M, // WSYNC_ATOM_M|GA|U|S
    WSYNC_ATOM_M, // WSYNC_ATOM_M|GA|U|W
    RACE, // WSYNC_ATOM_M|GA|U|B
    RACE, // WSYNC_ATOM_M|GA|U|G
    WSYNC_ATOM, // WSYNC_ATOM_M|GA|W|S
    WSYNC_ATOM, // WSYNC_ATOM_M|GA|W|W
    RACE, // WSYNC_ATOM_M|GA|W|B
    RACE, // WSYNC_ATOM_M|GA|W|G
    BSYNC_ATOM, // WSYNC_ATOM_M|GA|B|S
    BSYNC_ATOM, // WSYNC_ATOM_M|GA|B|W
    BSYNC_ATOM, // WSYNC_ATOM_M|GA|B|B
    RACE, // WSYNC_ATOM_M|GA|B|G
    RACE, // GATOM_G|R|U|S
    RACE, // GATOM_G|R|U|W
    RACE, // GATOM_G|R|U|B
    RACE, // GATOM_G|R|U|G
    RACE, // GATOM_G|R|W|S
    RACE, // GATOM_G|R|W|W
    RACE, // GATOM_G|R|W|B
    RACE, // GATOM_G|R|W|G
    RACE, // GATOM_G|R|B|S
    RACE, // GATOM_G|R|B|W
    RACE, // GATOM_G|R|B|B
    RACE, // GATOM_G|R|B|G
    RACE, // GATOM_G|W|U|S
    RACE, // GATOM_G|W|U|W
    RACE, // GATOM_G|W|U|B
    RACE, // GATOM_G|W|U|G
    RACE, // GATOM_G|W|W|S
    RACE, // GATOM_G|W|W|W
    RACE, // GATOM_G|W|W|B
    RACE, // GATOM_G|W|W|G
    RACE, // GATOM_G|W|B|S
    RACE, // GATOM_G|W|B|W
    RACE, // GATOM_G|W|B|B
    RACE, // GATOM_G|W|B|G
    RACE, // GATOM_G|BA|U|S
    RACE, // GATOM_G|BA|U|W
    RACE, // GATOM_G|BA|U|B
    RACE, // GATOM_G|BA|U|G
    RACE, // GATOM_G|BA|W|S
    RACE, // GATOM_G|BA|W|W
    RACE, // GATOM_G|BA|W|B
    RACE, // GATOM_G|BA|W|G
    RACE, // GATOM_G|BA|B|S
    RACE, // GATOM_G|BA|B|W
    RACE, // GATOM_G|BA|B|B
    RACE, // GATOM_G|BA|B|G
    GATOM_G, // GATOM_G|GA|U|S
    GATOM_G, // GATOM_G|GA|U|W
    GATOM_G, // GATOM_G|GA|U|B
    GATOM_G, // GATOM_G|GA|U|G
    GATOM_G, // GATOM_G|GA|W|S
    GATOM_G, // GATOM_G|GA|W|W
    GATOM_G, // GATOM_G|GA|W|B
    GATOM_G, // GATOM_G|GA|W|G
    GATOM_G, // GATOM_G|GA|B|S
    GATOM_G, // GATOM_G|GA|B|W
    GATOM_G, // GATOM_G|GA|B|B
    GATOM_G, // GATOM_G|GA|B|G
    WRITE, // BSYNC_ATOM|R|U|S
    RACE, // BSYNC_ATOM|R|U|W
    RACE, // BSYNC_ATOM|R|U|B
    RACE, // BSYNC_ATOM|R|U|G
    WSYNC, // BSYNC_ATOM|R|W|S
    WSYNC, // BSYNC_ATOM|R|W|W
    RACE, // BSYNC_ATOM|R|W|B
    RACE, // BSYNC_ATOM|R|W|G
    BSYNC, // BSYNC_ATOM|R|B|S
    BSYNC, // BSYNC_ATOM|R|B|W
    BSYNC, // BSYNC_ATOM|R|B|B
    RACE, // BSYNC_ATOM|R|B|G
    WRITE, // BSYNC_ATOM|W|U|S
    RACE, // BSYNC_ATOM|W|U|W
    RACE, // BSYNC_ATOM|W|U|B
    RACE, // BSYNC_ATOM|W|U|G
    WRITE, // BSYNC_ATOM|W|W|S
    WRITE, // BSYNC_ATOM|W|W|W
    RACE, // BSYNC_ATOM|W|W|B
    RACE, // BSYNC_ATOM|W|W|G
    WRITE, // BSYNC_ATOM|W|B|S
    WRITE, // BSYNC_ATOM|W|B|W
    WRITE, // BSYNC_ATOM|W|B|B
    RACE, // BSYNC_ATOM|W|B|G
    BSYNC_ATOM, // BSYNC_ATOM|BA|U|S
    BSYNC_ATOM_W, // BSYNC_ATOM|BA|U|W
    BSYNC_ATOM_B, // BSYNC_ATOM|BA|U|B
    RACE, // BSYNC_ATOM|BA|U|G
    BSYNC_ATOM, // BSYNC_ATOM|BA|W|S
    BSYNC_ATOM, // BSYNC_ATOM|BA|W|W
    BSYNC_ATOM_B, // BSYNC_ATOM|BA|W|B
    RACE, // BSYNC_ATOM|BA|W|G
    BSYNC_ATOM, // BSYNC_ATOM|BA|B|S
    BSYNC_ATOM, // BSYNC_ATOM|BA|B|W
    BSYNC_ATOM, // BSYNC_ATOM|BA|B|B
    RACE, // BSYNC_ATOM|BA|B|G
    BSYNC_ATOM, // BSYNC_ATOM|GA|U|S
    BSYNC_ATOM_W, // BSYNC_ATOM|GA|U|W
    BSYNC_ATOM_B, // BSYNC_ATOM|GA|U|B
    RACE, // BSYNC_ATOM|GA|U|G
    BSYNC_ATOM, // BSYNC_ATOM|GA|W|S
    BSYNC_ATOM, // BSYNC_ATOM|GA|W|W
    BSYNC_ATOM_B, // BSYNC_ATOM|GA|W|B
    RACE, // BSYNC_ATOM|GA|W|G
    BSYNC_ATOM, // BSYNC_ATOM|GA|B|S
    BSYNC_ATOM, // BSYNC_ATOM|GA|B|W
    BSYNC_ATOM, // BSYNC_ATOM|GA|B|B
    RACE, // BSYNC_ATOM|GA|B|G
    WRITE, // BATOM|R|U|S
    RACE, // BATOM|R|U|W
    RACE, // BATOM|R|U|B
    RACE, // BATOM|R|U|G
    WSYNC, // BATOM|R|W|S
    WSYNC, // BATOM|R|W|W
    RACE, // BATOM|R|W|B
    RACE, // BATOM|R|W|G
    BSYNC, // BATOM|R|B|S
    BSYNC, // BATOM|R|B|W
    BSYNC, // BATOM|R|B|B
    RACE, // BATOM|R|B|G
    WRITE, // BATOM|W|U|S
    RACE, // BATOM|W|U|W
    RACE, // BATOM|W|U|B
    RACE, // BATOM|W|U|G
    WRITE, // BATOM|W|W|S
    WRITE, // BATOM|W|W|W
    RACE, // BATOM|W|W|B
    RACE, // BATOM|W|W|G
    WRITE, // BATOM|W|B|S
    WRITE, // BATOM|W|B|W
    WRITE, // BATOM|W|B|B
    RACE, // BATOM|W|B|G
    BATOM, // BATOM|BA|U|S
    BATOM_W, // BATOM|BA|U|W
    BATOM_B, // BATOM|BA|U|B
    RACE, // BATOM|BA|U|G
    BATOM, // BATOM|BA|W|S
    BATOM, // BATOM|BA|W|W
    BATOM_B, // BATOM|BA|W|B
    RACE, // BATOM|BA|W|G
    BATOM, // BATOM|BA|B|S
    BATOM, // BATOM|BA|B|W
    BATOM, // BATOM|BA|B|B
    RACE, // BATOM|BA|B|G
    BATOM, // BATOM|GA|U|S
    BATOM_W, // BATOM|GA|U|W
    BATOM_B, // BATOM|GA|U|B
    RACE, // BATOM|GA|U|G
    BATOM, // BATOM|GA|W|S
    BATOM, // BATOM|GA|W|W
    BATOM_B, // BATOM|GA|W|B
    RACE, // BATOM|GA|W|G
    BATOM, // BATOM|GA|B|S
    BATOM, // BATOM|GA|B|W
    BATOM, // BATOM|GA|B|B
    RACE, // BATOM|GA|B|G
    RACE, // GATOM_B|R|U|S
    RACE, // GATOM_B|R|U|W
    RACE, // GATOM_B|R|U|B
    RACE, // GATOM_B|R|U|G
    RACE, // GATOM_B|R|W|S
    RACE, // GATOM_B|R|W|W
    RACE, // GATOM_B|R|W|B
    RACE, // GATOM_B|R|W|G
    BSYNC, // GATOM_B|R|B|S
    BSYNC, // GATOM_B|R|B|W
    BSYNC, // GATOM_B|R|B|B
    RACE, // GATOM_B|R|B|G
    RACE, // GATOM_B|W|U|S
    RACE, // GATOM_B|W|U|W
    RACE, // GATOM_B|W|U|B
    RACE, // GATOM_B|W|U|G
    RACE, // GATOM_B|W|W|S
    RACE, // GATOM_B|W|W|W
    RACE, // GATOM_B|W|W|B
    RACE, // GATOM_B|W|W|G
    WRITE, // GATOM_B|W|B|S
    WRITE, // GATOM_B|W|B|W
    WRITE, // GATOM_B|W|B|B
    RACE, // GATOM_B|W|B|G
    BATOM_B, // GATOM_B|BA|U|S
    BATOM_B, // GATOM_B|BA|U|W
    BATOM_B, // GATOM_B|BA|U|B
    RACE, // GATOM_B|BA|U|G
    BATOM_B, // GATOM_B|BA|W|S
    BATOM_B, // GATOM_B|BA|W|W
    BATOM_B, // GATOM_B|BA|W|B
    RACE, // GATOM_B|BA|W|G
    BATOM, // GATOM_B|BA|B|S
    BATOM, // GATOM_B|BA|B|W
    BATOM, // GATOM_B|BA|B|B
    RACE, // GATOM_B|BA|B|G
    GATOM_B, // GATOM_B|GA|U|S
    GATOM_B, // GATOM_B|GA|U|W
    GATOM_B, // GATOM_B|GA|U|B
    GATOM_G, // GATOM_B|GA|U|G
    GATOM_B, // GATOM_B|GA|W|S
    GATOM_B, // GATOM_B|GA|W|W
    GATOM_B, // GATOM_B|GA|W|B
    GATOM_G, // GATOM_B|GA|W|G
    GATOM, // GATOM_B|GA|B|S
    GATOM, // GATOM_B|GA|B|W
    GATOM, // GATOM_B|GA|B|B
    GATOM_G, // GATOM_B|GA|B|G
    RACE, // BSYNC_ATOM_B|R|U|S
    RACE, // BSYNC_ATOM_B|R|U|W
    RACE, // BSYNC_ATOM_B|R|U|B
    RACE, // BSYNC_ATOM_B|R|U|G
    RACE, // BSYNC_ATOM_B|R|W|S
    RACE, // BSYNC_ATOM_B|R|W|W
    RACE, // BSYNC_ATOM_B|R|W|B
    RACE, // BSYNC_ATOM_B|R|W|G
    BSYNC, // BSYNC_ATOM_B|R|B|S
    BSYNC, // BSYNC_ATOM_B|R|B|W
    BSYNC, // BSYNC_ATOM_B|R|B|B
    RACE, // BSYNC_ATOM_B|R|B|G
    RACE, // BSYNC_ATOM_B|W|U|S
    RACE, // BSYNC_ATOM_B|W|U|W
    RACE, // BSYNC_ATOM_B|W|U|B
    RACE, // BSYNC_ATOM_B|W|U|G
    RACE, // BSYNC_ATOM_B|W|W|S
    RACE, // BSYNC_ATOM_B|W|W|W
    RACE, // BSYNC_ATOM_B|W|W|B
    RACE, // BSYNC_ATOM_B|W|W|G
    WRITE, // BSYNC_ATOM_B|W|B|S
    WRITE, // BSYNC_ATOM_B|W|B|W
    WRITE, // BSYNC_ATOM_B|W|B|B
    RACE, // BSYNC_ATOM_B|W|B|G
    BSYNC_ATOM_B, // BSYNC_ATOM_B|BA|U|S
    BSYNC_ATOM_B, // BSYNC_ATOM_B|BA|U|W
    BSYNC_ATOM_B, // BSYNC_ATOM_B|BA|U|B
    RACE, // BSYNC_ATOM_B|BA|U|G
    BSYNC_ATOM_B, // BSYNC_ATOM_B|BA|W|S
    BSYNC_ATOM_B, // BSYNC_ATOM_B|BA|W|W
    BSYNC_ATOM_B, // BSYNC_ATOM_B|BA|W|B
    RACE, // BSYNC_ATOM_B|BA|W|G
    BSYNC_ATOM, // BSYNC_ATOM_B|BA|B|S
    BSYNC_ATOM, // BSYNC_ATOM_B|BA|B|W
    BSYNC_ATOM, // BSYNC_ATOM_B|BA|B|B
    RACE, // BSYNC_ATOM_B|BA|B|G
    BSYNC_ATOM_B, // BSYNC_ATOM_B|GA|U|S
    BSYNC_ATOM_B, // BSYNC_ATOM_B|GA|U|W
    BSYNC_ATOM_B, // BSYNC_ATOM_B|GA|U|B
    RACE, // BSYNC_ATOM_B|GA|U|G
    BSYNC_ATOM_B, // BSYNC_ATOM_B|GA|W|S
    BSYNC_ATOM_B, // BSYNC_ATOM_B|GA|W|W
    BSYNC_ATOM_B, // BSYNC_ATOM_B|GA|W|B
    RACE, // BSYNC_ATOM_B|GA|W|G
    BSYNC_ATOM, // BSYNC_ATOM_B|GA|B|S
    BSYNC_ATOM, // BSYNC_ATOM_B|GA|B|W
    BSYNC_ATOM, // BSYNC_ATOM_B|GA|B|B
    RACE, // BSYNC_ATOM_B|GA|B|G
    WRITE, // GATOM|R|U|S
    RACE, // GATOM|R|U|W
    RACE, // GATOM|R|U|B
    RACE, // GATOM|R|U|G
    WSYNC, // GATOM|R|W|S
    WSYNC, // GATOM|R|W|W
    RACE, // GATOM|R|W|B
    RACE, // GATOM|R|W|G
    BSYNC, // GATOM|R|B|S
    BSYNC, // GATOM|R|B|W
    BSYNC, // GATOM|R|B|B
    RACE, // GATOM|R|B|G
    WRITE, // GATOM|W|U|S
    RACE, // GATOM|W|U|W
    RACE, // GATOM|W|U|B
    RACE, // GATOM|W|U|G
    WRITE, // GATOM|W|W|S
    WRITE, // GATOM|W|W|W
    RACE, // GATOM|W|W|B
    RACE, // GATOM|W|W|G
    WRITE, // GATOM|W|B|S
    WRITE, // GATOM|W|B|W
    WRITE, // GATOM|W|B|B
    RACE, // GATOM|W|B|G
    BATOM, // GATOM|BA|U|S
    BATOM_W, // GATOM|BA|U|W
    BATOM_B, // GATOM|BA|U|B
    RACE, // GATOM|BA|U|G
    BATOM, // GATOM|BA|W|S
    BATOM, // GATOM|BA|W|W
    BATOM_B, // GATOM|BA|W|B
    RACE, // GATOM|BA|W|G
    BATOM, // GATOM|BA|B|S
    BATOM, // GATOM|BA|B|W
    BATOM, // GATOM|BA|B|B
    RACE, // GATOM|BA|B|G
    GATOM, // GATOM|GA|U|S
    GATOM_W, // GATOM|GA|U|W
    GATOM_B, // GATOM|GA|U|B
    GATOM_G, // GATOM|GA|U|G
    GATOM, // GATOM|GA|W|S
    GATOM, // GATOM|GA|W|W
    GATOM_B, // GATOM|GA|W|B
    GATOM_G, // GATOM|GA|W|G
    GATOM, // GATOM|GA|B|S
    GATOM, // GATOM|GA|B|W
    GATOM, // GATOM|GA|B|B
    GATOM_G, // GATOM|GA|B|G
    RACE, // GATOM_W|R|U|S
    RACE, // GATOM_W|R|U|W
    RACE, // GATOM_W|R|U|B
    RACE, // GATOM_W|R|U|G
    WSYNC, // GATOM_W|R|W|S
    WSYNC, // GATOM_W|R|W|W
    RACE, // GATOM_W|R|W|B
    RACE, // GATOM_W|R|W|G
    BSYNC, // GATOM_W|R|B|S
    BSYNC, // GATOM_W|R|B|W
    BSYNC, // GATOM_W|R|B|B
    RACE, // GATOM_W|R|B|G
    RACE, // GATOM_W|W|U|S
    RACE, // GATOM_W|W|U|W
    RACE, // GATOM_W|W|U|B
    RACE, // GATOM_W|W|U|G
    WRITE, // GATOM_W|W|W|S
    WRITE, // GATOM_W|W|W|W
    RACE, // GATOM_W|W|W|B
    RACE, // GATOM_W|W|W|G
    WRITE, // GATOM_W|W|B|S
    WRITE, // GATOM_W|W|B|W
    WRITE, // GATOM_W|W|B|B
    RACE, // GATOM_W|W|B|G
    BATOM_W, // GATOM_W|BA|U|S
    BATOM_W, // GATOM_W|BA|U|W
    BATOM_B, // GATOM_W|BA|U|B
    RACE, // GATOM_W|BA|U|G
    BATOM, // GATOM_W|BA|W|S
    BATOM, // GATOM_W|BA|W|W
    BATOM_B, // GATOM_W|BA|W|B
    RACE, // GATOM_W|BA|W|G
    BATOM, // GATOM_W|BA|B|S
    BATOM, // GATOM_W|BA|B|W
    BATOM, // GATOM_W|BA|B|B
    RACE, // GATOM_W|BA|B|G
    GATOM_W, // GATOM_W|GA|U|S
    GATOM_W, // GATOM_W|GA|U|W
    GATOM_B, // GATOM_W|GA|U|B
    GATOM_G, // GATOM_W|GA|U|G
    GATOM, // GATOM_W|GA|W|S
    GATOM, // GATOM_W|GA|W|W
    GATOM_B, // GATOM_W|GA|W|B
    GATOM_G, // GATOM_W|GA|W|G
    GATOM, // GATOM_W|GA|B|S
    GATOM, // GATOM_W|GA|B|W
    GATOM, // GATOM_W|GA|B|B
    GATOM_G, // GATOM_W|GA|B|G
    RACE, // BSYNC_ATOM_W|R|U|S
    RACE, // BSYNC_ATOM_W|R|U|W
    RACE, // BSYNC_ATOM_W|R|U|B
    RACE, // BSYNC_ATOM_W|R|U|G
    WSYNC, // BSYNC_ATOM_W|R|W|S
    WSYNC, // BSYNC_ATOM_W|R|W|W
    RACE, // BSYNC_ATOM_W|R|W|B
    RACE, // BSYNC_ATOM_W|R|W|G
    BSYNC, // BSYNC_ATOM_W|R|B|S
    BSYNC, // BSYNC_ATOM_W|R|B|W
    BSYNC, // BSYNC_ATOM_W|R|B|B
    RACE, // BSYNC_ATOM_W|R|B|G
    RACE, // BSYNC_ATOM_W|W|U|S
    RACE, // BSYNC_ATOM_W|W|U|W
    RACE, // BSYNC_ATOM_W|W|U|B
    RACE, // BSYNC_ATOM_W|W|U|G
    WRITE, // BSYNC_ATOM_W|W|W|S
    WRITE, // BSYNC_ATOM_W|W|W|W
    RACE, // BSYNC_ATOM_W|W|W|B
    RACE, // BSYNC_ATOM_W|W|W|G
    WRITE, // BSYNC_ATOM_W|W|B|S
    WRITE, // BSYNC_ATOM_W|W|B|W
    WRITE, // BSYNC_ATOM_W|W|B|B
    RACE, // BSYNC_ATOM_W|W|B|G
    BSYNC_ATOM_W, // BSYNC_ATOM_W|BA|U|S
    BSYNC_ATOM_W, // BSYNC_ATOM_W|BA|U|W
    BSYNC_ATOM_B, // BSYNC_ATOM_W|BA|U|B
    RACE, // BSYNC_ATOM_W|BA|U|G
    BSYNC_ATOM, // BSYNC_ATOM_W|BA|W|S
    BSYNC_ATOM, // BSYNC_ATOM_W|BA|W|W
    BSYNC_ATOM_B, // BSYNC_ATOM_W|BA|W|B
    RACE, // BSYNC_ATOM_W|BA|W|G
    BSYNC_ATOM, // BSYNC_ATOM_W|BA|B|S
    BSYNC_ATOM, // BSYNC_ATOM_W|BA|B|W
    BSYNC_ATOM, // BSYNC_ATOM_W|BA|B|B
    RACE, // BSYNC_ATOM_W|BA|B|G
    BSYNC_ATOM_W, // BSYNC_ATOM_W|GA|U|S
    BSYNC_ATOM_W, // BSYNC_ATOM_W|GA|U|W
    BSYNC_ATOM_B, // BSYNC_ATOM_W|GA|U|B
    RACE, // BSYNC_ATOM_W|GA|U|G
    BSYNC_ATOM, // BSYNC_ATOM_W|GA|W|S
    BSYNC_ATOM, // BSYNC_ATOM_W|GA|W|W
    BSYNC_ATOM_B, // BSYNC_ATOM_W|GA|W|B
    RACE, // BSYNC_ATOM_W|GA|W|G
    BSYNC_ATOM, // BSYNC_ATOM_W|GA|B|S
    BSYNC_ATOM, // BSYNC_ATOM_W|GA|B|W
    BSYNC_ATOM, // BSYNC_ATOM_W|GA|B|B
    RACE, // BSYNC_ATOM_W|GA|B|G
    RACE, // RACE|R|U|S
    RACE, // RACE|R|U|W
    RACE, // RACE|R|U|B
    RACE, // RACE|R|U|G
    RACE, // RACE|R|W|S
    RACE, // RACE|R|W|W
    RACE, // RACE|R|W|B
    RACE, // RACE|R|W|G
    RACE, // RACE|R|B|S
    RACE, // RACE|R|B|W
    RACE, // RACE|R|B|B
    RACE, // RACE|R|B|G
    RACE, // RACE|W|U|S
    RACE, // RACE|W|U|W
    RACE, // RACE|W|U|B
    RACE, // RACE|W|U|G
    RACE, // RACE|W|W|S
    RACE, // RACE|W|W|W
    RACE, // RACE|W|W|B
    RACE, // RACE|W|W|G
    RACE, // RACE|W|B|S
    RACE, // RACE|W|B|W
    RACE, // RACE|W|B|B
    RACE, // RACE|W|B|G
    RACE, // RACE|BA|U|S
    RACE, // RACE|BA|U|W
    RACE, // RACE|BA|U|B
    RACE, // RACE|BA|U|G
    RACE, // RACE|BA|W|S
    RACE, // RACE|BA|W|W
    RACE, // RACE|BA|W|B
    RACE, // RACE|BA|W|G
    RACE, // RACE|BA|B|S
    RACE, // RACE|BA|B|W
    RACE, // RACE|BA|B|B
    RACE, // RACE|BA|B|G
    RACE, // RACE|GA|U|S
    RACE, // RACE|GA|U|W
    RACE, // RACE|GA|U|B
    RACE, // RACE|GA|U|G
    RACE, // RACE|GA|W|S
    RACE, // RACE|GA|W|W
    RACE, // RACE|GA|W|B
    RACE, // RACE|GA|W|G
    RACE, // RACE|GA|B|S
    RACE, // RACE|GA|B|W
    RACE, // RACE|GA|B|B
    RACE  // RACE|GA|B|G
  };
  hr_shadowt prev_state;
  hr_shadowt tmp_val = atomicAdd(prev_access, 0); // atomicRead
 
  unsigned wid,
           prev_tid,
           prev_wid,
           prev_bid,
           prev_internal_state,
           prev_bcount,
           prev_wcount,
           prev_swidx,
           next_internal_state;
  
  unsigned new_bcount = *bcount;
  unsigned new_wcount = *wcount;
  unsigned new_swidx = *swidx;
  
  // Shadow value update and ISM transition
  // *Lock free implementation provided by Martin Burtscher
  // 
  // Intuition:
  //   1. Atomically read previous shadow value (tmp_val above)
  //   2. Decompose previous shadow value
  //   3. Compare current and previous metadata
  //      to generate transition values
  //   4. Create new shadow value with resulting
  //      state and current access metadata
  //   5. Atomically compare to metadata address;
  //      if shadow value is the same as last read
  //      then swap, else use the shadow value
  //      found as the new "previous" from step 1
  //      and repeat from step 2.
  do {
    prev_internal_state = GET_STATE(tmp_val);
    
    // if race already flagged, eject
    if (prev_internal_state == RACE) { return; }
    
    prev_state = tmp_val;
    
    wid = tid / WARP_SIZE;
    prev_tid = GET_TID(tmp_val);
    prev_wid = prev_tid / WARP_SIZE;
    prev_bid = GET_BLOCK(tmp_val);
    prev_bcount = GET_BCOUNT(tmp_val);
    prev_wcount = GET_WCOUNT(tmp_val);
    prev_swidx = GET_SWIDX(tmp_val);
    
    int sync_scope;
    if (new_bcount > prev_bcount)      sync_scope = BLOCKSYNC;
    else if (new_wcount > prev_wcount) sync_scope = WARPSYNC;
    else if (new_swidx > prev_swidx)   sync_scope = swsync_scan(tid, bid, new_swidx, prev_swidx); //TODO, fix this
    else                               sync_scope = UNSYNC;
                             
    int thread_rel;
    if (bid != prev_bid)      thread_rel = SAMEGRID;
    else if (wid != prev_wid) thread_rel = SAMEBLOCK;
    else if (tid != prev_tid) thread_rel = SAMEWARP;
    else                      thread_rel = SAMETHREAD;
    
    // recall rw_indicator is 0: read, 1: write, 2: block atomic, 3: global atomic
    // sync_scope only has 3 values, so offset the missing value by subtraction
    const unsigned transition = ( (prev_internal_state << 6) | (rw_indicator << 4) | (sync_scope << 2) | thread_rel ) - (( (prev_internal_state << 2) | rw_indicator ) << 2);
    
    next_internal_state = ism_transitions[transition];
    
    tmp_val = create_shadow(tid, bid, next_internal_state, new_bcount, new_wcount, new_swidx);
  } while ( (tmp_val = atomicCAS(prev_access, prev_state, tmp_val)) != prev_state );
  
  if (next_internal_state == RACE) {
    const char *rw_map[4] = { "READ", "WRITE", "BLOCK ATOMIC", "DEVICE ATOMIC" };
    const char *state_map[25] = { "INIT", "READ", "WRITE", "WREAD",
                                  "BREAD", "GREAD", "WSYNC", "BSYNC",
                                  "MR_WSYNC", "BWSYNC", "B_MR_BSYNC",
                                  "W_MR_BWSYNC", "BATOM_B", "WSYNC_ATOM",
                                  "BATOM_W", "WSYNC_ATOM_M", "GATOM_G",
                                  "BSYNC_ATOM", "BATOM", "GATOM_B",
                                  "BSYNC_ATOM_B", "GATOM", "GATOM_W",
                                  "BSYNC_ATOM_W", "RACE" };
    printf("HiRace:\n\tRace @ line %d in file %s on a %s:\n\t\tprior access  - tid %u, bid %u, state %s, bcount %u, wcount %u,\n\t\tlatest access - tid %u, bid %u, state %s, bcount %u, wcount %u\n",
      lineNo, fileName, rw_map[rw_indicator], prev_tid, prev_bid, state_map[prev_internal_state], 
      prev_bcount, prev_wcount, tid, bid, state_map[next_internal_state], new_bcount, new_wcount);
  }
}

#endif // RACECHECK

#include "HiRaceWrappers.h"

#undef WARP_SIZE

#undef TRANSITION_COUNT

#undef INIT        
#undef READ        
#undef WRITE       
#undef WREAD       
#undef BREAD       
#undef GREAD       
#undef WSYNC       
#undef BSYNC       
#undef MR_WSYNC    
#undef BWSYNC      
#undef B_MR_BSYNC    
#undef W_MR_BWSYNC 
#undef BATOM_B     
#undef WSYNC_ATOM  
#undef BATOM_W     
#undef WSYNC_ATOM_M
#undef GATOM_G     
#undef BSYNC_ATOM  
#undef BATOM       
#undef GATOM_B     
#undef BSYNC_ATOM_B
#undef GATOM       
#undef GATOM_W     
#undef BSYNC_ATOM_W
#undef RACE        

// memory operations
#undef R 
#undef W 
#undef BA
#undef GA

// sync scopes
#undef UNSYNC
#undef WARPSYNC
#undef BLOCKSYNC

// thread relations
#undef SAMETHREAD
#undef SAMEWARP
#undef SAMEBLOCK
#undef SAMEGRID

// shadow word layout
#undef SHADOW_SIZE

#undef TID_BITS   
#undef BLOCK_BITS 
#undef STATE_BITS 
#undef BCOUNT_BITS
#undef WCOUNT_BITS
#undef SWIDX_BITS 

#undef TID_OFFSET    
#undef BLOCK_OFFSET  
#undef STATE_OFFSET  
#undef BCOUNT_OFFSET 
#undef WCOUNT_OFFSET 
#undef SWIDX_OFFSET  

#undef TID_MASK   
#undef BLOCK_MASK 
#undef STATE_MASK 
#undef BCOUNT_MASK
#undef WCOUNT_MASK
#undef SWIDX_MASK 

// Decomposing one shadow value
#undef GET_TID
#undef GET_BLOCK
#undef GET_STATE
#undef GET_BCOUNT
#undef GET_WCOUNT
#undef GET_SWIDX

#endif // HIRACE_H



