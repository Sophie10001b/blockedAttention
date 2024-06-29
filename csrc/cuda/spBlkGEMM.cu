#include "spBlkGEMM.cuh"
#include "sparseBlockedAttention.h"

// blocked gemm with pre tiling
// mat1 -> (H, (m1*k1 + m2*k2 + ...)), mat2 -> (H, (n1*k1 + n2*k2 + ...)), out -> (H, (m1*n1 + m2*n2 + ...))
// accum1 -> (0, m1*k1, (m1*k1) + m2*k2...), accum2 -> (0, n1*k1, (n1*k1) + n2*k2...), accumOut -> (0, m1*n1, (m1*n1) + m2*n2...)
// graphTiling -> ((graph ID for this block, i-th block for this graph))
// default transpose1 = false: mat1(M, K) -> true: mat1(K, M)
// default transpose2 = false: mat2(N, K) -> true: mat1(K, N)
// <<<(B, H, 1), (256，1， 1)>>>

template<typename T>
__device__ inline void __mat1LDG_GEMM(
    T *mat1, T *mat1LDG,
    const int M, const int K, const int BK, const int tLoad1,
    const int mat1StartIdx, int &mat1LDGRow, int &mat1LDGCol,
    const bool transpose1
){
    // 1.load from global
    // 2.fill 0 for out of range
    // 3.go to next tiling
    if (!transpose1){
        #pragma unroll
        for (int i=0; i < tLoad1; i++){
            mat1LDG[i] = mat1[mat1StartIdx + mat1LDGRow * K + mat1LDGCol + i];
        }
        #pragma unroll
        for (int i=0; i < tLoad1; i++){
            if (mat1LDGRow >= M || mat1LDGCol + i >= K) mat1LDG[i] = 0;
        }
        mat1LDGCol += BK;
    }
    else{
        #pragma unroll
        for (int i=0; i < tLoad1; i++){
            mat1LDG[i] = mat1[mat1StartIdx + mat1LDGRow * M + mat1LDGCol + i];
        }
        #pragma unroll
        for (int i=0; i < tLoad1; i++){
            if (mat1LDGRow >= K || mat1LDGCol + i >= M) mat1LDG[i] = 0;
        }
        mat1LDGRow += BK;
    }
}
template<typename T>
__device__ inline void __mat1STS_GEMM(
    T *mat1Cache, T *mat1LDG,
    const int BK, const int BM, const int PAD, const int tLoad1,
    const uint loadFlag, const int mat1STSRow, const int mat1STSCol,
    const bool transpose1
){
    if (!transpose1){
        #pragma unroll
        for (int i=0; i < tLoad1; i++){
            *(mat1Cache + loadFlag * BK * (BM+PAD) + (mat1STSRow + i) * (BM+PAD) + mat1STSCol) = mat1LDG[i];
        }
    }
    else{
        #pragma unroll
        for (int i=0; i < tLoad1; i++){
            *(mat1Cache + loadFlag * BK * (BM+PAD) + mat1STSRow * (BM+PAD) + mat1STSCol + i) = mat1LDG[i];
        }
    }
}

template<typename T>
__device__ inline void __mat2LDG_GEMM(
    T *mat2, T *mat2LDG,
    const int N, const int K, const int BK, const int tLoad2,
    const int mat2StartIdx, int &mat2LDGRow, int &mat2LDGCol,
    const bool transpose2
){
    // 1.load from global
    // 2.fill 0 for out of range
    // 3.go to next tiling
    if (!transpose2){
        #pragma unroll
        for (int i=0; i < tLoad2; i++){
            mat2LDG[i] = mat2[mat2StartIdx + mat2LDGRow * K + mat2LDGCol + i];
        }
        #pragma unroll
        for (int i=0; i < tLoad2; i++){
            if (mat2LDGRow >= N || mat2LDGCol + i >= K) mat2LDG[i] = 0;
        }
        mat2LDGCol += BK;
    }
    else{
        #pragma unroll
        for (int i=0; i < tLoad2; i++){
            mat2LDG[i] = mat2[mat2StartIdx + mat2LDGRow * N + mat2LDGCol + i];
        }
        #pragma unroll
        for (int i=0; i < tLoad2; i++){
            if (mat2LDGRow >= K || mat2LDGCol + i >= N) mat2LDG[i] = 0;
        }
        mat2LDGRow += BK;
    }
}
template<typename T>
__device__ inline void __mat2STS_GEMM(
    T *mat2Cache, T *mat2LDG,
    const int BK, const int BN, const int PAD, const int tLoad2,
    const uint loadFlag, const int mat2STSRow, const int mat2STSCol,
    const bool transpose2
){
    if (!transpose2){
        #pragma unroll
        for (int i=0; i < tLoad2; i++){
            *(mat2Cache + loadFlag * BK * (BN+PAD) + (mat2STSRow + i) * (BN+PAD) + mat2STSCol) = mat2LDG[i];
        }
    }
    else{
        #pragma unroll
        for (int i=0; i < tLoad2; i++){
            *(mat2Cache + loadFlag * BK * (BN+PAD) + mat2STSRow * (BN+PAD) + mat2STSCol + i) = mat2LDG[i];
        }
    }
}

template<typename T, int BM, int BN, int BK, int PAD=0>
__global__ void sparseBlockedGEMM(
    T *mat1, T *mat2, T *out,
    const int64_t *accum1, const int64_t *accum2, const int64_t *accumOut,
    const int64_t *MList, const int64_t *NList, const int64_t *KList, const int64_t *graphTiling,
    const int64_t Dm1, const int64_t Dm2, const int64_t Dout,
    const bool transpose1, const bool transpose2
){
    constexpr int TM = BM / THREADY;
    constexpr int TN = BN / THREADX;
    constexpr int tLoad1 = (BM * BK) / (GEMMTHREADNUM);
    constexpr int tLoad2 = (BN * BK) / (GEMMTHREADNUM);

    const int graphId = graphTiling[blockIdx.x * 2];
    const int tileId = graphTiling[blockIdx.x * 2 + 1];

    const int M = MList[graphId];
    const int N = NList[graphId];
    const int K = KList[graphId];

    // calculate start index
    int64_t mat1StartIdx = blockIdx.y * Dm1 + accum1[graphId];
    int64_t mat2StartIdx = blockIdx.y * Dm2 + accum2[graphId];
    int64_t outStartIdx = blockIdx.y * Dout + accumOut[graphId];
    int mat1StartM = 0;
    int mat2StartN = 0;
    if (tileId > 0){
        const int NTile = RANGE_COUNT(BN, N);
        mat1StartM += (tileId / NTile) * BM;
        mat2StartN += (tileId % NTile) * BN;
    }

    __shared__ T mat1Cache[2][BK][BM+PAD];
    __shared__ T mat2Cache[2][BK][BN+PAD];

    T mat1LDG[tLoad1];
    T mat2LDG[tLoad2];
    T mat1LDS[TM];
    T mat2LDS[TN];
    T fmaRes[TM][TN] = {0};
    uint loadFlag = 0;

    // first load global to shared

    // STS: index for buffer register to shared
    // LDG: index for global to buffer register
    // LDS: index for shared to compute register
    // STG: index for compute register to global

    int mat1STSRow = (threadIdx.x * tLoad1) % BK;
    int mat1STSCol = (threadIdx.x * tLoad1) / BK;
    int mat2STSRow = (threadIdx.x * tLoad2) % BK;
    int mat2STSCol = (threadIdx.x * tLoad2) / BK;

    int mat1LDGRow = mat1StartM + mat1STSCol;
    int mat1LDGCol = mat1STSRow;
    int mat2LDGRow = mat2StartN + mat2STSCol;
    int mat2LDGCol = mat2STSRow;

    if (transpose1){
        mat1STSRow = (threadIdx.x * tLoad1) / BM;
        mat1STSCol = (threadIdx.x * tLoad1) % BM;

        mat1LDGRow = mat1STSRow;
        mat1LDGCol = mat1StartM + mat1STSCol;
    }
    if (transpose2){
        mat2STSRow = (threadIdx.x * tLoad2) / BN;
        mat2STSCol = (threadIdx.x * tLoad2) % BN;

        mat2LDGRow = mat2STSRow;
        mat2LDGCol = mat2StartN + mat2STSCol;
    }

    int mat1LDSCol = (threadIdx.x / THREADX) * TM;
    int mat2LDSCol = (threadIdx.x % THREADX) * TN;

    #define outSTGRow (mat1StartM + mat1LDSCol)
    #define outSTGCol (mat2StartN + mat2LDSCol)

    __mat1LDG_GEMM(mat1, mat1LDG, M, K, BK, tLoad1, mat1StartIdx, mat1LDGRow, mat1LDGCol, transpose1);
    __mat2LDG_GEMM(mat2, mat2LDG, N, K, BK, tLoad2, mat2StartIdx, mat2LDGRow, mat2LDGCol, transpose2);
    __mat1STS_GEMM(&mat1Cache[0][0][0], mat1LDG, BK, BM, PAD, tLoad1, loadFlag, mat1STSRow, mat1STSCol, transpose1);
    __mat2STS_GEMM(&mat2Cache[0][0][0], mat2LDG, BK, BN, PAD, tLoad2, loadFlag, mat2STSRow, mat2STSCol, transpose2);
    __syncthreads();

    // load next global to register and compute last shared
    for (int KIter=1, KLoop=RANGE_COUNT(BK, K); KIter < KLoop; KIter++){

        __mat1LDG_GEMM(mat1, mat1LDG, M, K, BK, tLoad1, mat1StartIdx, mat1LDGRow, mat1LDGCol, transpose1);
        __mat2LDG_GEMM(mat2, mat2LDG, N, K, BK, tLoad2, mat2StartIdx, mat2LDGRow, mat2LDGCol, transpose2);

        #pragma unroll
        for (int bkIter=0; bkIter < BK; bkIter++){
            // load shared memory to register
            #pragma unroll
            for (int i=0; i < TM; i++){
                mat1LDS[i] = mat1Cache[loadFlag][bkIter][mat1LDSCol + i];
            }
            #pragma unroll
            for (int i=0; i < TN; i++){
                mat2LDS[i] = mat2Cache[loadFlag][bkIter][mat2LDSCol + i];
            }

            // compute mma
            #pragma unroll
            for (int tmIter=0; tmIter < TM; tmIter++){
                #pragma unroll
                for(int tnIter=0; tnIter < TN; tnIter++){
                    fmaRes[tmIter][tnIter] += mat1LDS[tmIter] * mat2LDS[tnIter];
                }
            }
        }
        loadFlag = !loadFlag;

        __mat1STS_GEMM(&mat1Cache[0][0][0], mat1LDG, BK, BM, PAD, tLoad1, loadFlag, mat1STSRow, mat1STSCol, transpose1);
        __mat2STS_GEMM(&mat2Cache[0][0][0], mat2LDG, BK, BN, PAD, tLoad2, loadFlag, mat2STSRow, mat2STSCol, transpose2);
        __syncthreads();
    }

    // compute last mma
    #pragma unroll
    for (int bkIter=0; bkIter < BK; bkIter++){
        #pragma unroll
        for (int i=0; i < TM; i++){
            mat1LDS[i] = mat1Cache[loadFlag][bkIter][mat1LDSCol + i];
        }
        #pragma unroll
        for (int i=0; i < TN; i++){
            mat2LDS[i] = mat2Cache[loadFlag][bkIter][mat2LDSCol + i];
        }

        #pragma unroll
        for (int tmIter=0; tmIter < TM; tmIter++){
            #pragma unroll
            for(int tnIter=0; tnIter < TN; tnIter++){
                fmaRes[tmIter][tnIter] += mat1LDS[tmIter] * mat2LDS[tnIter];
            }
        }
    }

    // store result to global
    #pragma unroll
    for (int tmIter=0; tmIter < TM; tmIter++){
        #pragma unroll
        for (int tnIter=0; tnIter < TN; tnIter++){
            if (outSTGRow + tmIter < M && outSTGCol + tnIter < N){
                out[outStartIdx + (outSTGRow + tmIter) * N + (outSTGCol + tnIter)] = fmaRes[tmIter][tnIter];
            }
        }
    }
}
#undef mat1LDGRow
#undef mat2LDGRow
#undef outSTGRow
#undef outSTGCol

// kernel specialized for Half and BFloat16
// default transpose1 = false: mat1(M, K) -> true: mat1(K, M)
// default transpose2 = false: mat2(N, K) -> true: mat1(K, N)
// <<<(B, H, 1), (128，1， 1)>>>

template<typename T, typename vT>
__device__ inline void __mat1LDG_HGEMM(
    T *mat1, T *mat1LDG, T __zero,
    const int M, const int K, const int BK, const int tLoad1,
    const int mat1StartIdx, int &mat1LDGRow, int &mat1LDGCol,
    const bool transpose1
){
    // 1.load from global
    // 2.fill 0 for out of range
    // 3.go to next tiling
    if (!transpose1){
        if (!(K & 1)){
            #pragma unroll
            for (int i=0; i < tLoad1; i+=2){
                COALESCE(vT*, mat1LDG[i]) = COALESCE(vT*, mat1[mat1StartIdx + mat1LDGRow * K + mat1LDGCol + i]);
            }
        }
        else{
            #pragma unroll
            for (int i=0; i < tLoad1; i++){
                mat1LDG[i] = mat1[mat1StartIdx + mat1LDGRow * K + mat1LDGCol + i];
            }
        }
        #pragma unroll
        for (int i=0; i < tLoad1; i++){
            if (mat1LDGRow >= M || mat1LDGCol + i >= K) mat1LDG[i] = __zero;
        }
        mat1LDGCol += BK;
    }
    else{
        if (!(K & 1)){
            #pragma unroll
            for (int i=0; i < tLoad1; i+=2){
                COALESCE(vT*, mat1LDG[i]) = COALESCE(vT*, mat1[mat1StartIdx + mat1LDGRow * M + mat1LDGCol + i]);
            }
        }
        else{
            #pragma unroll
            for (int i=0; i < tLoad1; i++){
                mat1LDG[i] = mat1[mat1StartIdx + mat1LDGRow * M + mat1LDGCol + i];
            }
        }
        #pragma unroll
        for (int i=0; i < tLoad1; i++){
            if (mat1LDGRow >= K || mat1LDGCol + i >= M) mat1LDG[i] = __zero;
        }
        mat1LDGRow += BK;
    }
}
template<typename T, typename vT>
__device__ inline void __mat1STS_HGEMM(
    T *mat1Cache, T *mat1LDG,
    const int BK, const int BM, const int tLoad1,
    const uint loadFlag, const int mat1STSRow, const int mat1STSCol,
    const bool transpose1
){
    if (!transpose1){
        #pragma unroll
        for (int i=0; i < tLoad1; i+=2){
            COALESCE(vT*, mat1Cache[loadFlag * BM * BK + mat1STSRow * BK + mat1STSCol + i]) = COALESCE(vT*, mat1LDG[i]);
            // *(mat1Cache + loadFlag * BM * BK + mat1STSRow * BK + mat1STSCol + i) = mat1LDG[i];
        }
    }
    else{
        #pragma unroll
        for (int i=0; i < tLoad1; i++){
            *(mat1Cache + loadFlag * BM * BK + (mat1STSRow + i) * BK + mat1STSCol) = mat1LDG[i];
        }
    }
}

template<typename T, typename vT>
__device__ inline void __mat2LDG_HGEMM(
    T *mat2, T *mat2LDG, T __zero,
    const int N, const int K, const int BK, const int tLoad2,
    const int mat2StartIdx, int &mat2LDGRow, int &mat2LDGCol,
    const bool transpose2
){
    // 1.load from global
    // 2.fill 0 for out of range
    // 3.go to next tiling
    if (!transpose2){
        if (!(K & 1)){
            #pragma unroll
            for (int i=0; i < tLoad2; i+=2){
                COALESCE(vT*, mat2LDG[i]) = COALESCE(vT*, mat2[mat2StartIdx + mat2LDGRow * K + mat2LDGCol + i]);
            }
        }
        else{
            #pragma unroll
            for (int i=0; i < tLoad2; i++){
                mat2LDG[i] = mat2[mat2StartIdx + mat2LDGRow * K + mat2LDGCol + i];
            }
        }
        #pragma unroll
        for (int i=0; i < tLoad2; i++){
            if (mat2LDGRow >= N || mat2LDGCol + i >= K) mat2LDG[i] = __zero;
        }
        mat2LDGCol += BK;
    }
    else{
        if (!(K & 1)){
            #pragma unroll
            for (int i=0; i < tLoad2; i+=2){
                COALESCE(vT*, mat2LDG[i]) = COALESCE(vT*, mat2[mat2StartIdx + mat2LDGRow * N + mat2LDGCol + i]);
            }
        }
        else{
            #pragma unroll
            for (int i=0; i < tLoad2; i++){
                mat2LDG[i] = mat2[mat2StartIdx + mat2LDGRow * N + mat2LDGCol + i];
            }
        }
        #pragma unroll
        for (int i=0; i < tLoad2; i++){
            if (mat2LDGRow >= K || mat2LDGCol + i >= N) mat2LDG[i] = __zero;
        }
        mat2LDGRow += BK;
    }
}
template<typename T, typename vT>
__device__ inline void __mat2STS_HGEMM(
    T *mat2Cache, T *mat2LDG,
    const int BK, const int BN, const int tLoad2,
    const uint loadFlag, const int mat2STSRow, const int mat2STSCol,
    const bool transpose2
){
    if (!transpose2){
        #pragma unroll
        for (int i=0; i < tLoad2; i++){
            *(mat2Cache + loadFlag * BK * BN + (mat2STSRow + i) * BN + mat2STSCol) = mat2LDG[i];
        }
    }
    else{
        #pragma unroll
        for (int i=0; i < tLoad2; i+=2){
            COALESCE(vT*, *(mat2Cache + loadFlag * BK * BN + mat2STSRow * BN + mat2STSCol + i)) = COALESCE(vT*, mat2LDG[i]);
            // *(mat2Cache + loadFlag * BK * BN + mat2STSRow * BN + mat2STSCol + i) = mat2LDG[i];
        }
    }
}

template<int BM, int BN, int BK, int PAD=0>
__global__ void sparseBlockedHGEMM_half(
    half *mat1, half *mat2, half *out,
    const int64_t *accum1, const int64_t *accum2, const int64_t *accumOut,
    const int64_t *MList, const int64_t *NList, const int64_t *KList, const int64_t *graphTiling,
    const int64_t Dm1, const int64_t Dm2, const int64_t Dout,
    const bool transpose1, const bool transpose2
){
    constexpr int wmmaM = 16;
    constexpr int wmmaN = 16;
    constexpr int wmmaK = 16;
    constexpr int tLoad1 = (BM * BK) / (HGEMMTHREADNUM);
    constexpr int tLoad2 = (BN * BK) / (HGEMMTHREADNUM);
    constexpr int WM = BM / 32; // 4 warp for 32*32
    constexpr int WN = BN / 32;

    const int warpRow = (threadIdx.x / 32) / 2;
    const int warpCol = (threadIdx.x / 32) % 2;
    const int threadRow = (threadIdx.x % 32) / 4;
    const int threadCol = ((threadIdx.x % 32) % 4) * 2;

    const int graphId = graphTiling[blockIdx.x * 2];
    const int tileId = graphTiling[blockIdx.x * 2 + 1];

    const int M = MList[graphId];
    const int N = NList[graphId];
    const int K = KList[graphId];

    // calculate start index
    int64_t mat1StartIdx = blockIdx.y * Dm1 + accum1[graphId];
    int64_t mat2StartIdx = blockIdx.y * Dm2 + accum2[graphId];
    int64_t outStartIdx = blockIdx.y * Dout + accumOut[graphId];
    int mat1StartM = 0;
    int mat2StartN = 0;
    if (tileId > 0){
        const int NTile = RANGE_COUNT(BN, N);
        mat1StartM += (tileId / NTile) * BM;
        mat2StartN += (tileId % NTile) * BN;
    }

    __shared__ half mat1Cache[2][BM][BK];
    __shared__ half mat2Cache[2][BK][BN];

    half mat1LDG[tLoad1];
    half mat2LDG[tLoad2];
    half __zero = __float2half(0.0f);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> mat1LDS[WM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> mat2LDS[WN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> wmmaRes[WM][WN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, half> wmmaResHalf;
    uint loadFlag = 0;

    #pragma unroll
    for (int i=0; i < WM; i++){
        #pragma unroll
        for (int j=0; j < WN; j++){
            nvcuda::wmma::fill_fragment(wmmaRes[i][j], 0.0);
        }
    }

    // first load global to shared

    // STS: index for buffer register to shared
    // LDG: index for global to buffer register
    // LDS: index for shared to compute register
    // STG: index for compute register to global

    int mat1STSRow = (threadIdx.x * tLoad1) / BK;
    int mat1STSCol = (threadIdx.x * tLoad1) % BK;
    int mat2STSRow = (threadIdx.x * tLoad2) % BK;
    int mat2STSCol = (threadIdx.x * tLoad2) / BK;

    int mat1LDGRow = mat1StartM + mat1STSRow;
    int mat1LDGCol = mat1STSCol;
    int mat2LDGRow = mat2StartN + mat2STSCol;
    int mat2LDGCol = mat2STSRow;

    if (transpose1){
        mat1STSRow = (threadIdx.x * tLoad1) % BM;
        mat1STSCol = (threadIdx.x * tLoad1) / BM;

        mat1LDGRow = mat1STSCol;
        mat1LDGCol = mat1StartM + mat1STSRow;
    }
    if (transpose2){
        mat2STSRow = (threadIdx.x * tLoad2) / BN;
        mat2STSCol = (threadIdx.x * tLoad2) % BN;

        mat2LDGRow = mat2STSRow;
        mat2LDGCol = mat2StartN + mat2STSCol;
    }

    // index for load shared
    int mat1LDSRow = warpRow * WM * wmmaM;
    int mat2LDSCol = warpCol * WN * wmmaN;
    // index for store result to global (thread level)
    #define outSTGRow (mat1StartM + mat1LDSRow)
    #define outSTGCol (mat2StartN + mat2LDSCol)

    __mat1LDG_HGEMM<half, half2>(mat1, mat1LDG, __zero, M, K, BK, tLoad1, mat1StartIdx, mat1LDGRow, mat1LDGCol, transpose1);
    __mat2LDG_HGEMM<half, half2>(mat2, mat2LDG, __zero, N, K, BK, tLoad2, mat2StartIdx, mat2LDGRow, mat2LDGCol, transpose2);
    __mat1STS_HGEMM<half, half2>(&mat1Cache[0][0][0], mat1LDG, BK, BM, tLoad1, loadFlag, mat1STSRow, mat1STSCol, transpose1);
    __mat2STS_HGEMM<half, half2>(&mat2Cache[0][0][0], mat2LDG, BK, BN, tLoad2, loadFlag, mat2STSRow, mat2STSCol, transpose2);
    __syncthreads();

    // load next global to register and compute last shared
    for (int KIter=1, KLoop=RANGE_COUNT(BK, K); KIter < KLoop; KIter++){

    __mat1LDG_HGEMM<half, half2>(mat1, mat1LDG, __zero, M, K, BK, tLoad1, mat1StartIdx, mat1LDGRow, mat1LDGCol, transpose1);
    __mat2LDG_HGEMM<half, half2>(mat2, mat2LDG, __zero, N, K, BK, tLoad2, mat2StartIdx, mat2LDGRow, mat2LDGCol, transpose2);

        #pragma unroll
        for (int bkIter=0, bkLoop=RANGE_COUNT(wmmaK, BK); bkIter < bkLoop; bkIter++){
            // load shared memory to register
            #pragma unroll
            for (int i=0; i < WM; i++){
                nvcuda::wmma::load_matrix_sync(mat1LDS[i], &mat1Cache[loadFlag][mat1LDSRow + wmmaM * i][bkIter * 16], BK);
            }
            #pragma unroll
            for (int i=0; i < WN; i++){
                nvcuda::wmma::load_matrix_sync(mat2LDS[i], &mat2Cache[loadFlag][bkIter * 16][mat2LDSCol + wmmaN * i], BN);
            }

            // compute mma
            #pragma unroll
            for (int tmIter=0; tmIter < WM; tmIter++){
                #pragma unroll
                for(int tnIter=0; tnIter < WN; tnIter++){
                    nvcuda::wmma::mma_sync(wmmaRes[tmIter][tnIter], mat1LDS[tmIter], mat2LDS[tnIter], wmmaRes[tmIter][tnIter]);
                }
            }
        }
        loadFlag = !loadFlag;

        __mat1STS_HGEMM<half, half2>(&mat1Cache[0][0][0], mat1LDG, BK, BM, tLoad1, loadFlag, mat1STSRow, mat1STSCol, transpose1);
        __mat2STS_HGEMM<half, half2>(&mat2Cache[0][0][0], mat2LDG, BK, BN, tLoad2, loadFlag, mat2STSRow, mat2STSCol, transpose2);
        __syncthreads();
    }

    // compute last mma
    #pragma unroll
    for (int bkIter=0, bkLoop=RANGE_COUNT(wmmaK, BK); bkIter < bkLoop; bkIter++){
        #pragma unroll
        for (int i=0; i < WM; i++){
            nvcuda::wmma::load_matrix_sync(mat1LDS[i], &mat1Cache[loadFlag][mat1LDSRow + wmmaM * i][bkIter * 16], BK);
        }
        #pragma unroll
        for (int i=0; i < WN; i++){
            nvcuda::wmma::load_matrix_sync(mat2LDS[i], &mat2Cache[loadFlag][bkIter * 16][mat2LDSCol + wmmaN * i], BN);
        }

        #pragma unroll
        for (int tmIter=0; tmIter < WM; tmIter++){
            #pragma unroll
            for(int tnIter=0; tnIter < WN; tnIter++){
                nvcuda::wmma::mma_sync(wmmaRes[tmIter][tnIter], mat1LDS[tmIter], mat2LDS[tnIter], wmmaRes[tmIter][tnIter]);
            }
        }
    }
    
    // store result to global
    #pragma unroll
    for (int tmIter=0, baseRow=outSTGRow; tmIter < WM; tmIter++, baseRow+=16){
        #pragma unroll
        for (int tnIter=0, baseCol=outSTGCol; tnIter < WN; tnIter++, baseCol+=16){
            wmmaResHalf.x[0] = __float2half(wmmaRes[tmIter][tnIter].x[0]);
            wmmaResHalf.x[1] = __float2half(wmmaRes[tmIter][tnIter].x[1]);
            wmmaResHalf.x[2] = __float2half(wmmaRes[tmIter][tnIter].x[2]);
            wmmaResHalf.x[3] = __float2half(wmmaRes[tmIter][tnIter].x[3]);
            wmmaResHalf.x[4] = __float2half(wmmaRes[tmIter][tnIter].x[4]);
            wmmaResHalf.x[5] = __float2half(wmmaRes[tmIter][tnIter].x[5]);
            wmmaResHalf.x[6] = __float2half(wmmaRes[tmIter][tnIter].x[6]);
            wmmaResHalf.x[7] = __float2half(wmmaRes[tmIter][tnIter].x[7]);

            if (!(N & 1) && baseRow + 15 < M && baseCol + 15 < N){
                nvcuda::wmma::store_matrix_sync(&(out[outStartIdx + baseRow * N + baseCol]), wmmaResHalf, N, nvcuda::wmma::mem_row_major);
            }
            else{
                if (baseRow + threadRow < M){
                    if (baseCol + threadCol < N) out[outStartIdx + (baseRow + threadRow) * N + baseCol + threadCol] = wmmaResHalf.x[0];
                    if (baseCol + threadCol + 1 < N) out[outStartIdx + (baseRow + threadRow) * N + baseCol + threadCol + 1] = wmmaResHalf.x[1];
                    if (baseCol + threadCol + 8 < N) out[outStartIdx + (baseRow + threadRow) * N + baseCol + threadCol + 8] = wmmaResHalf.x[4];
                    if (baseCol + threadCol + 9 < N) out[outStartIdx + (baseRow + threadRow) * N + baseCol + threadCol + 9] = wmmaResHalf.x[5];
                }
                if (baseRow + threadRow + 8 < M){
                    if (baseCol + threadCol < N) out[outStartIdx + (baseRow + threadRow + 8) * N + baseCol + threadCol] = wmmaResHalf.x[2];
                    if (baseCol + threadCol + 1 < N) out[outStartIdx + (baseRow + threadRow + 8) * N + baseCol + threadCol + 1] = wmmaResHalf.x[3];
                    if (baseCol + threadCol + 8 < N) out[outStartIdx + (baseRow + threadRow + 8) * N + baseCol + threadCol + 8] = wmmaResHalf.x[6];
                    if (baseCol + threadCol + 9 < N) out[outStartIdx + (baseRow + threadRow + 8) * N + baseCol + threadCol + 9] = wmmaResHalf.x[7];
                }
            }
        }
    }
}

template<int BM, int BN, int BK, int PAD=0>
__global__ void sparseBlockedHGEMM_bf16(
    nv_bfloat16 *mat1, nv_bfloat16 *mat2, nv_bfloat16 *out,
    const int64_t *accum1, const int64_t *accum2, const int64_t *accumOut,
    const int64_t *MList, const int64_t *NList, const int64_t *KList, const int64_t *graphTiling,
    const int64_t Dm1, const int64_t Dm2, const int64_t Dout,
    const bool transpose1, const bool transpose2
){
    constexpr int wmmaM = 16;
    constexpr int wmmaN = 16;
    constexpr int wmmaK = 16;
    constexpr int tLoad1 = (BM * BK) / (HGEMMTHREADNUM);
    constexpr int tLoad2 = (BN * BK) / (HGEMMTHREADNUM);
    constexpr int WM = BM / 32; // 4 warp for 32*32
    constexpr int WN = BN / 32;

    const int warpRow = (threadIdx.x / 32) / 2;
    const int warpCol = (threadIdx.x / 32) % 2;
    const int threadRow = (threadIdx.x % 32) / 4;
    const int threadCol = ((threadIdx.x % 32) % 4) * 2;

    const int graphId = graphTiling[blockIdx.x * 2];
    const int tileId = graphTiling[blockIdx.x * 2 + 1];

    const int M = MList[graphId];
    const int N = NList[graphId];
    const int K = KList[graphId];

    // calculate start index
    int64_t mat1StartIdx = blockIdx.y * Dm1 + accum1[graphId];
    int64_t mat2StartIdx = blockIdx.y * Dm2 + accum2[graphId];
    int64_t outStartIdx = blockIdx.y * Dout + accumOut[graphId];
    int mat1StartM = 0;
    int mat2StartN = 0;
    if (tileId > 0){
        const int NTile = RANGE_COUNT(BN, N);
        mat1StartM += (tileId / NTile) * BM;
        mat2StartN += (tileId % NTile) * BN;
    }

    __shared__ nv_bfloat16 mat1Cache[2][BM][BK];
    __shared__ nv_bfloat16 mat2Cache[2][BK][BN];

    nv_bfloat16 mat1LDG[tLoad1];
    nv_bfloat16 mat2LDG[tLoad2];
    nv_bfloat16 __zero = __float2bfloat16(0.0f);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, nv_bfloat16, nvcuda::wmma::row_major> mat1LDS[WM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, nv_bfloat16, nvcuda::wmma::row_major> mat2LDS[WN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> wmmaRes[WM][WN];
    nv_bfloat16 wmmaResHalf[8];
    uint loadFlag = 0;

    #pragma unroll
    for (int i=0; i < WM; i++){
        #pragma unroll
        for (int j=0; j < WN; j++){
            nvcuda::wmma::fill_fragment(wmmaRes[i][j], 0.0);
        }
    }

    // first load global to shared

    // STS: index for buffer register to shared
    // LDG: index for global to buffer register
    // LDS: index for shared to compute register
    // STG: index for compute register to global

    int mat1STSRow = (threadIdx.x * tLoad1) / BK;
    int mat1STSCol = (threadIdx.x * tLoad1) % BK;
    int mat2STSRow = (threadIdx.x * tLoad2) % BK;
    int mat2STSCol = (threadIdx.x * tLoad2) / BK;

    int mat1LDGRow = mat1StartM + mat1STSRow;
    int mat1LDGCol = mat1STSCol;
    int mat2LDGRow = mat2StartN + mat2STSCol;
    int mat2LDGCol = mat2STSRow;

    if (transpose1){
        mat1STSRow = (threadIdx.x * tLoad1) % BM;
        mat1STSCol = (threadIdx.x * tLoad1) / BM;

        mat1LDGRow = mat1STSCol;
        mat1LDGCol = mat1StartM + mat1STSRow;
    }
    if (transpose2){
        mat2STSRow = (threadIdx.x * tLoad2) / BN;
        mat2STSCol = (threadIdx.x * tLoad2) % BN;

        mat2LDGRow = mat2STSRow;
        mat2LDGCol = mat2StartN + mat2STSCol;
    }

    // index for load shared
    int mat1LDSRow = warpRow * WM * wmmaM;
    int mat2LDSCol = warpCol * WN * wmmaN;
    // index for store result to global (thread level)
    #define outSTGRow (mat1StartM + mat1LDSRow)
    #define outSTGCol (mat2StartN + mat2LDSCol)

    __mat1LDG_HGEMM<nv_bfloat16, nv_bfloat162>(mat1, mat1LDG, __zero, M, K, BK, tLoad1, mat1StartIdx, mat1LDGRow, mat1LDGCol, transpose1);
    __mat2LDG_HGEMM<nv_bfloat16, nv_bfloat162>(mat2, mat2LDG, __zero, N, K, BK, tLoad2, mat2StartIdx, mat2LDGRow, mat2LDGCol, transpose2);
    __mat1STS_HGEMM<nv_bfloat16, nv_bfloat162>(&mat1Cache[0][0][0], mat1LDG, BK, BM, tLoad1, loadFlag, mat1STSRow, mat1STSCol, transpose1);
    __mat2STS_HGEMM<nv_bfloat16, nv_bfloat162>(&mat2Cache[0][0][0], mat2LDG, BK, BN, tLoad2, loadFlag, mat2STSRow, mat2STSCol, transpose2);
    __syncthreads();

    // load next global to register and compute last shared
    for (int KIter=1, KLoop=RANGE_COUNT(BK, K); KIter < KLoop; KIter++){

        __mat1LDG_HGEMM<nv_bfloat16, nv_bfloat162>(mat1, mat1LDG, __zero, M, K, BK, tLoad1, mat1StartIdx, mat1LDGRow, mat1LDGCol, transpose1);
        __mat2LDG_HGEMM<nv_bfloat16, nv_bfloat162>(mat2, mat2LDG, __zero, N, K, BK, tLoad2, mat2StartIdx, mat2LDGRow, mat2LDGCol, transpose2);

        #pragma unroll
        for (int bkIter=0, bkLoop=RANGE_COUNT(wmmaK, BK); bkIter < bkLoop; bkIter++){
            // load shared memory to register
            #pragma unroll
            for (int i=0; i < WM; i++){
                nvcuda::wmma::load_matrix_sync(mat1LDS[i], &mat1Cache[loadFlag][mat1LDSRow + wmmaM * i][bkIter * 16], BK);
            }
            #pragma unroll
            for (int i=0; i < WN; i++){
                nvcuda::wmma::load_matrix_sync(mat2LDS[i], &mat2Cache[loadFlag][bkIter * 16][mat2LDSCol + wmmaN * i], BN);
            }

            // compute mma
            #pragma unroll
            for (int tmIter=0; tmIter < WM; tmIter++){
                #pragma unroll
                for(int tnIter=0; tnIter < WN; tnIter++){
                    nvcuda::wmma::mma_sync(wmmaRes[tmIter][tnIter], mat1LDS[tmIter], mat2LDS[tnIter], wmmaRes[tmIter][tnIter]);
                }
            }
        }
        loadFlag = !loadFlag;

        __mat1STS_HGEMM<nv_bfloat16, nv_bfloat162>(&mat1Cache[0][0][0], mat1LDG, BK, BM, tLoad1, loadFlag, mat1STSRow, mat1STSCol, transpose1);
        __mat2STS_HGEMM<nv_bfloat16, nv_bfloat162>(&mat2Cache[0][0][0], mat2LDG, BK, BN, tLoad2, loadFlag, mat2STSRow, mat2STSCol, transpose2);
        __syncthreads();
    }

    // compute last mma
    #pragma unroll
    for (int bkIter=0, bkLoop=RANGE_COUNT(wmmaK, BK); bkIter < bkLoop; bkIter++){
        #pragma unroll
        for (int i=0; i < WM; i++){
            nvcuda::wmma::load_matrix_sync(mat1LDS[i], &mat1Cache[loadFlag][mat1LDSRow + wmmaM * i][bkIter * 16], BK);
        }
        #pragma unroll
        for (int i=0; i < WN; i++){
            nvcuda::wmma::load_matrix_sync(mat2LDS[i], &mat2Cache[loadFlag][bkIter * 16][mat2LDSCol + wmmaN * i], BN);
        }

        #pragma unroll
        for (int tmIter=0; tmIter < WM; tmIter++){
            #pragma unroll
            for(int tnIter=0; tnIter < WN; tnIter++){
                nvcuda::wmma::mma_sync(wmmaRes[tmIter][tnIter], mat1LDS[tmIter], mat2LDS[tnIter], wmmaRes[tmIter][tnIter]);
            }
        }
    }
    
    // store result to global
    #pragma unroll
    for (int tmIter=0, baseRow=outSTGRow; tmIter < WM; tmIter++, baseRow+=16){
        #pragma unroll
        for (int tnIter=0, baseCol=outSTGCol; tnIter < WN; tnIter++, baseCol+=16){
            wmmaResHalf[0] = __float2bfloat16(wmmaRes[tmIter][tnIter].x[0]);
            wmmaResHalf[1] = __float2bfloat16(wmmaRes[tmIter][tnIter].x[1]);
            wmmaResHalf[2] = __float2bfloat16(wmmaRes[tmIter][tnIter].x[2]);
            wmmaResHalf[3] = __float2bfloat16(wmmaRes[tmIter][tnIter].x[3]);
            wmmaResHalf[4] = __float2bfloat16(wmmaRes[tmIter][tnIter].x[4]);
            wmmaResHalf[5] = __float2bfloat16(wmmaRes[tmIter][tnIter].x[5]);
            wmmaResHalf[6] = __float2bfloat16(wmmaRes[tmIter][tnIter].x[6]);
            wmmaResHalf[7] = __float2bfloat16(wmmaRes[tmIter][tnIter].x[7]);

            if (!(N & 1) && baseRow + 15 < M && baseCol + 15 < N){
                COALESCE(nv_bfloat162*, out[outStartIdx + (baseRow + threadRow) * N + baseCol + threadCol]) = COALESCE(nv_bfloat162*, wmmaResHalf[0]);
                COALESCE(nv_bfloat162*, out[outStartIdx + (baseRow + threadRow) * N + baseCol + threadCol + 8]) = COALESCE(nv_bfloat162*, wmmaResHalf[4]);
                COALESCE(nv_bfloat162*, out[outStartIdx + (baseRow + threadRow + 8) * N + baseCol + threadCol]) = COALESCE(nv_bfloat162*, wmmaResHalf[2]);
                COALESCE(nv_bfloat162*, out[outStartIdx + (baseRow + threadRow + 8) * N + baseCol + threadCol + 8]) = COALESCE(nv_bfloat162*, wmmaResHalf[6]);
            }
            else{
                if (baseRow + threadRow < M){
                    if (baseCol + threadCol < N) out[outStartIdx + (baseRow + threadRow) * N + baseCol + threadCol] = wmmaResHalf[0];
                    if (baseCol + threadCol + 1 < N) out[outStartIdx + (baseRow + threadRow) * N + baseCol + threadCol + 1] = wmmaResHalf[1];
                    if (baseCol + threadCol + 8 < N) out[outStartIdx + (baseRow + threadRow) * N + baseCol + threadCol + 8] = wmmaResHalf[4];
                    if (baseCol + threadCol + 9 < N) out[outStartIdx + (baseRow + threadRow) * N + baseCol + threadCol + 9] = wmmaResHalf[5];
                }
                if (baseRow + threadRow + 8 < M){
                    if (baseCol + threadCol < N) out[outStartIdx + (baseRow + threadRow + 8) * N + baseCol + threadCol] = wmmaResHalf[2];
                    if (baseCol + threadCol + 1 < N) out[outStartIdx + (baseRow + threadRow + 8) * N + baseCol + threadCol + 1] = wmmaResHalf[3];
                    if (baseCol + threadCol + 8 < N) out[outStartIdx + (baseRow + threadRow + 8) * N + baseCol + threadCol + 8] = wmmaResHalf[6];
                    if (baseCol + threadCol + 9 < N) out[outStartIdx + (baseRow + threadRow + 8) * N + baseCol + threadCol + 9] = wmmaResHalf[7];
                }
            }
        }
    }
}
#undef mat1LDGRow
#undef mat2LDGRow
#undef outSTGRow
#undef outSTGCol


// kernel launch
// for half and bfloat16, the minimize k for wmma is 16, so BK is as large as possible
template<typename T>
void cudaKernelLaunch(
    const int BMIdx, const int BNIdx, const int BKIdx, const dim3 &blockSize,
    T *src1, T *src2, T *tgt,
    const int64_t *accum1, const int64_t *accum2, const int64_t *accumOut,
    const int64_t *MList, const int64_t *NList, const int64_t *KList, const int64_t *graphTiling,
    const int64_t Dm1, const int64_t Dm2, const int64_t Dout,
    const bool transpose1, const bool transpose2
){
    const int kernelIdx = KERNEL_IDX(BMIdx, BNIdx, BKIdx);
    if (std::is_same<T, c10::Half>::value){
        auto mat1 = reinterpret_cast<half*>(src1);
        auto mat2 = reinterpret_cast<half*>(src2);
        auto out = reinterpret_cast<half*>(tgt);

        switch (kernelIdx){
            case 0: CALL_HGEMM_KERNEL(half, 32, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 1: CALL_HGEMM_KERNEL(half, 32, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 2: CALL_HGEMM_KERNEL(half, 32, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 3: CALL_HGEMM_KERNEL(half, 32, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 4: CALL_HGEMM_KERNEL(half, 32, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 5: CALL_HGEMM_KERNEL(half, 32, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 6: CALL_HGEMM_KERNEL(half, 32, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 7: CALL_HGEMM_KERNEL(half, 32, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 8: CALL_HGEMM_KERNEL(half, 32, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 9: CALL_HGEMM_KERNEL(half, 64, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 10: CALL_HGEMM_KERNEL(half, 64, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 11: CALL_HGEMM_KERNEL(half, 64, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 12: CALL_HGEMM_KERNEL(half, 64, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 13: CALL_HGEMM_KERNEL(half, 64, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 14: CALL_HGEMM_KERNEL(half, 64, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 15: CALL_HGEMM_KERNEL(half, 64, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 16: CALL_HGEMM_KERNEL(half, 64, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 17: CALL_HGEMM_KERNEL(half, 64, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 18: CALL_HGEMM_KERNEL(half, 128, 32, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 19: CALL_HGEMM_KERNEL(half, 128, 32, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 20: CALL_HGEMM_KERNEL(half, 128, 32, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 21: CALL_HGEMM_KERNEL(half, 128, 64, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 22: CALL_HGEMM_KERNEL(half, 128, 64, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 23: CALL_HGEMM_KERNEL(half, 128, 64, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 24: CALL_HGEMM_KERNEL(half, 128, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 25: CALL_HGEMM_KERNEL(half, 128, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 26: CALL_HGEMM_KERNEL(half, 128, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
        }
    }
    else if (std::is_same<T, c10::BFloat16>::value){
        auto mat1 = reinterpret_cast<nv_bfloat16*>(src1);
        auto mat2 = reinterpret_cast<nv_bfloat16*>(src2);
        auto out = reinterpret_cast<nv_bfloat16*>(tgt);

        switch (kernelIdx){
            case 0: CALL_HGEMM_KERNEL(bf16, 32, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 1: CALL_HGEMM_KERNEL(bf16, 32, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 2: CALL_HGEMM_KERNEL(bf16, 32, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 3: CALL_HGEMM_KERNEL(bf16, 32, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 4: CALL_HGEMM_KERNEL(bf16, 32, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 5: CALL_HGEMM_KERNEL(bf16, 32, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 6: CALL_HGEMM_KERNEL(bf16, 32, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 7: CALL_HGEMM_KERNEL(bf16, 32, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 8: CALL_HGEMM_KERNEL(bf16, 32, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 9: CALL_HGEMM_KERNEL(bf16, 64, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 10: CALL_HGEMM_KERNEL(bf16, 64, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 11: CALL_HGEMM_KERNEL(bf16, 64, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 12: CALL_HGEMM_KERNEL(bf16, 64, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 13: CALL_HGEMM_KERNEL(bf16, 64, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 14: CALL_HGEMM_KERNEL(bf16, 64, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 15: CALL_HGEMM_KERNEL(bf16, 64, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 16: CALL_HGEMM_KERNEL(bf16, 64, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 17: CALL_HGEMM_KERNEL(bf16, 64, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 18: CALL_HGEMM_KERNEL(bf16, 128, 32, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 19: CALL_HGEMM_KERNEL(bf16, 128, 32, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 20: CALL_HGEMM_KERNEL(bf16, 128, 32, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 21: CALL_HGEMM_KERNEL(bf16, 128, 64, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 22: CALL_HGEMM_KERNEL(bf16, 128, 64, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 23: CALL_HGEMM_KERNEL(bf16, 128, 64, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 24: CALL_HGEMM_KERNEL(bf16, 128, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 25: CALL_HGEMM_KERNEL(bf16, 128, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 26: CALL_HGEMM_KERNEL(bf16, 128, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
        }
    }
    else{
        auto mat1 = reinterpret_cast<T*>(src1);
        auto mat2 = reinterpret_cast<T*>(src2);
        auto out = reinterpret_cast<T*>(tgt);

        switch (kernelIdx){
            case 0: CALL_GEMM_KERNEL(T, 32, 32, 8, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 1: CALL_GEMM_KERNEL(T, 32, 32, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 2: CALL_GEMM_KERNEL(T, 32, 32, 32, 1, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 3: CALL_GEMM_KERNEL(T, 32, 64, 8, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 4: CALL_GEMM_KERNEL(T, 32, 64, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 5: CALL_GEMM_KERNEL(T, 32, 64, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 6: CALL_GEMM_KERNEL(T, 32, 128, 8, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 7: CALL_GEMM_KERNEL(T, 32, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 8: CALL_GEMM_KERNEL(T, 32, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 9: CALL_GEMM_KERNEL(T, 64, 32, 8, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 10: CALL_GEMM_KERNEL(T, 64, 32, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 11: CALL_GEMM_KERNEL(T, 64, 32, 32, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 12: CALL_GEMM_KERNEL(T, 64, 64, 8, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 13: CALL_GEMM_KERNEL(T, 64, 64, 16, 2, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 14: CALL_GEMM_KERNEL(T, 64, 64, 16, 2, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 15: CALL_GEMM_KERNEL(T, 64, 128, 8, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 16: CALL_GEMM_KERNEL(T, 64, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 17: CALL_GEMM_KERNEL(T, 64, 128, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 18: CALL_GEMM_KERNEL(T, 128, 32, 8, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 19: CALL_GEMM_KERNEL(T, 128, 32, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 20: CALL_GEMM_KERNEL(T, 128, 32, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 21: CALL_GEMM_KERNEL(T, 128, 64, 8, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 22: CALL_GEMM_KERNEL(T, 128, 64, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 23: CALL_GEMM_KERNEL(T, 128, 64, 16, 0, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 24: CALL_GEMM_KERNEL(T, 128, 128, 8, 4, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 25: CALL_GEMM_KERNEL(T, 128, 128, 8, 4, blockSize, BLKGEMM_KERNEL_INPUT); break;
            case 26: CALL_GEMM_KERNEL(T, 128, 128, 8, 4, blockSize, BLKGEMM_KERNEL_INPUT); break;
        }
    }
}

tilingData __getTiling(
    const variable &MList, const variable &NList, const variable &KList,
    const c10::ScalarType &outType
){
    const int avgM = MList.mean(torch::kFloat).item().toInt();
    const int avgN = NList.mean(torch::kFloat).item().toInt();
    const int maxM = MList.max().item().toInt();
    const int maxN = NList.max().item().toInt();

    int BMIdx = avgM == maxM ? std::lower_bound(BMList, BMList+TILENUM, avgM) - BMList : std::upper_bound(BMList, BMList+TILENUM, avgM) - BMList;
    BMIdx = BMIdx == TILENUM ? TILENUM - 1 : BMIdx;
    if (BMIdx > 0 && BMList[BMIdx] > maxM && avgM != maxM) BMIdx -= 1;

    int BNIdx = avgN == maxN ? std::lower_bound(BNList, BNList+TILENUM, avgN) - BNList : std::upper_bound(BNList, BNList+TILENUM, avgN) - BNList;
    BNIdx = BNIdx == TILENUM ? TILENUM - 1 : BNIdx;
    if (BNIdx > 0 && BNList[BNIdx] > maxN && avgN != maxN) BNIdx -= 1;

    int BKIdx = TILENUM - min(BMIdx, BNIdx) - 1;

    // for half, use less BM, BN and larger BK
    if (outType == c10::ScalarType::Half || outType == c10::ScalarType::BFloat16){
        // BMIdx = BMIdx > 0 ? BMIdx-1 : BMIdx;
        // BNIdx = BNIdx > 0 ? BNIdx-1 : BNIdx;
        BKIdx = BKIdx < TILENUM-1 ? BKIdx+1 : BKIdx;
    }

    const int bm = BMList[BMIdx];
    const int bn = BNList[BNIdx];

    auto graphSize = ((RANGE_COUNT(bm, MList)).to(torch::kInt64) * (RANGE_COUNT(bn, NList)).to(torch::kInt64)).cpu();
    auto graphSizePtr = graphSize.data_ptr<int64_t>();

    std::vector<int64_t> graphTiling;
    for (int64_t i=0, graphNum=graphSize.size(0); i < graphNum; i++){
        for (int64_t j=0, tileSize=graphSizePtr[i]; j < tileSize; j++){
            graphTiling.push_back(i);
            graphTiling.push_back(j);
        }
    }
    return tilingData(torch::tensor(graphTiling, MList.options()), BMIdx, BNIdx, BKIdx);
}

torch::Tensor sparseBlockedGEMMLaunch(
    const variable &mat1, const variable &mat2,
    const variable &mat1Accum, const variable &mat2Accum, const variable &outAccum,
    const variable &MList, const variable &NList, const variable &KList,
    const bool transpose1, const bool transpose2
){
    INPUT_CHECKING(mat1); INPUT_CHECKING(mat2);
    INPUT_CHECKING(mat1Accum); INPUT_CHECKING(mat2Accum); INPUT_CHECKING(outAccum);
    INPUT_CHECKING(MList); INPUT_CHECKING(NList); INPUT_CHECKING(KList);
    assert(mat1Accum.sizes() == mat2Accum.sizes() && mat1Accum.sizes() == outAccum.sizes());

    const int64_t Dm1 = mat1Accum.index({-1}).item().toInt();
    const int64_t Dm2 = mat2Accum.index({-1}).item().toInt();
    const int64_t Dout = outAccum.index({-1}).item().toInt();
    const int64_t H = mat1.numel() / Dm1;

    assert(mat1.size(-1) == Dm1);
    assert(mat2.size(-1) == Dm2);

    auto tilingData = __getTiling(MList, NList, KList, mat1.scalar_type());

    const dim3 blockSize(tilingData.graphTiling.size(0) / 2, H, 1);

    auto out = torch::empty({H, Dout}, mat1.options());
    
    AT_DISPATCH_ALL_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::BFloat16, out.scalar_type(), "__sparseBlockedGEMM", [&]{
        auto mat1Ptr = mat1.data_ptr<scalar_t>();
        auto mat2Ptr = mat2.data_ptr<scalar_t>();
        auto outPtr = out.data_ptr<scalar_t>();

        auto accum1 = mat1Accum.data_ptr<int64_t>();
        auto accum2 = mat2Accum.data_ptr<int64_t>();
        auto accumOut = outAccum.data_ptr<int64_t>();

        auto MListPtr = MList.data_ptr<int64_t>();
        auto NListPtr = NList.data_ptr<int64_t>();
        auto KListPtr = KList.data_ptr<int64_t>();
        auto graphTilingPtr = tilingData.graphTiling.data_ptr<int64_t>();

        cudaKernelLaunch<scalar_t>(
            tilingData.BMIdx, tilingData.BNIdx, tilingData.BKIdx, blockSize,
            mat1Ptr, mat2Ptr, outPtr, accum1, accum2, accumOut,
            MListPtr, NListPtr, KListPtr, graphTilingPtr,
            Dm1, Dm2, Dout, transpose1, transpose2
        );
    });
    
    return out;
}