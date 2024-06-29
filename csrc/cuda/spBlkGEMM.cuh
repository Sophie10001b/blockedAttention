#pragma once
#include <mma.h>

#define THREADX 16
#define THREADY 16
#define GEMMTHREADNUM 256
#define HGEMMTHREADNUM 128
#define TILENUM 3

constexpr int BMList[TILENUM] = {32, 64, 128};
constexpr int BNList[TILENUM] = {32, 64, 128};
constexpr int BKList[TILENUM] = {8, 16, 32};

#define COALESCE(type, pointer) (reinterpret_cast<type>(&(pointer))[0])

#define KERNEL_IDX(BMIdx, BNIdx, BKIdx)\
    ((BMIdx) * (sizeof(BNList)/sizeof(BNList[0])) * (sizeof(BKList)/sizeof(BKList[0])) + (BNIdx) * (sizeof(BKList)/sizeof(BKList[0])) + (BKIdx))

#define BLKGEMM_KERNEL_PARAM(macroT)\
    macroT *mat1, macroT *mat2, macroT *out,\
    const int64_t *accum1, const int64_t *accum2, const int64_t *accumOut,\
    const int64_t *graphTiling, const int64_t Dm1, const int64_t Dm2, const int64_t Dout

#define BLKGEMM_KERNEL_INPUT\
    mat1, mat2, out, accum1, accum2, accumOut, MList, NList, KList, graphTiling, Dm1, Dm2, Dout, transpose1, transpose2

#define BLKGEMM_KERNEL_PARAMTYPE(macroT)\
    const macroT*, const macroT*, macroT*,\
    const int64_t*, const int64_t*, const int64_t*, const int64_t*,\
    const int64_t, const int64_t, const int64_t

#define CALL_GEMM_KERNEL(T, M, N, K, PAD, BLOCK, ...) sparseBlockedGEMM<T, M, N, K, PAD><<<BLOCK, dim3(GEMMTHREADNUM, 1, 1)>>>(__VA_ARGS__)

#define CALL_HGEMM_KERNEL(Tn, M, N, K, PAD, BLOCK, ...) sparseBlockedHGEMM_##Tn<M, N, K, PAD><<<BLOCK, dim3(HGEMMTHREADNUM, 1, 1)>>>(__VA_ARGS__)