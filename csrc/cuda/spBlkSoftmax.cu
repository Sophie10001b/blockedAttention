#include "spBlkSoftmax.cuh"
#include "sparseBlockedAttention.h"

template<typename T>
__device__ __forceinline__ T __warpReduceSum(T res){
    res += __shfl_xor_sync(0xffffffff, res, 16);
    res += __shfl_xor_sync(0xffffffff, res, 8);
    res += __shfl_xor_sync(0xffffffff, res, 4);
    res += __shfl_xor_sync(0xffffffff, res, 2);
    res += __shfl_xor_sync(0xffffffff, res, 1);
    return res;
}
template<>
__device__ __forceinline__ c10::Half __warpReduceSum<c10::Half>(c10::Half res){
    res += __shfl_xor_sync(0xffffffff, (reinterpret_cast<half*>(&res))[0], 16);
    res += __shfl_xor_sync(0xffffffff, (reinterpret_cast<half*>(&res))[0], 8);
    res += __shfl_xor_sync(0xffffffff, (reinterpret_cast<half*>(&res))[0], 4);
    res += __shfl_xor_sync(0xffffffff, (reinterpret_cast<half*>(&res))[0], 2);
    res += __shfl_xor_sync(0xffffffff, (reinterpret_cast<half*>(&res))[0], 1);
    return res;
}
template<>
__device__ __forceinline__ c10::BFloat16 __warpReduceSum<c10::BFloat16>(c10::BFloat16 res){
    res += __shfl_xor_sync(0xffffffff, (reinterpret_cast<nv_bfloat16*>(&res))[0], 16);
    res += __shfl_xor_sync(0xffffffff, (reinterpret_cast<nv_bfloat16*>(&res))[0], 8);
    res += __shfl_xor_sync(0xffffffff, (reinterpret_cast<nv_bfloat16*>(&res))[0], 4);
    res += __shfl_xor_sync(0xffffffff, (reinterpret_cast<nv_bfloat16*>(&res))[0], 2);
    res += __shfl_xor_sync(0xffffffff, (reinterpret_cast<nv_bfloat16*>(&res))[0], 1);
    return res;
}

template<typename T>
__device__ __forceinline__ T __warpReduceMax(T res){
    res = max(res, __shfl_xor_sync(0xffffffff, res, 16));
    res = max(res, __shfl_xor_sync(0xffffffff, res, 8));
    res = max(res, __shfl_xor_sync(0xffffffff, res, 4));
    res = max(res, __shfl_xor_sync(0xffffffff, res, 2));
    res = max(res, __shfl_xor_sync(0xffffffff, res, 1));
    return res;
}
template<>
__device__ __forceinline__ c10::Half __warpReduceMax<c10::Half>(c10::Half __res){
    auto res = (reinterpret_cast<half*>(&__res))[0];
    res = __hmax(res, __shfl_xor_sync(0xffffffff, res, 16));
    res = __hmax(res, __shfl_xor_sync(0xffffffff, res, 8));
    res = __hmax(res, __shfl_xor_sync(0xffffffff, res, 4));
    res = __hmax(res, __shfl_xor_sync(0xffffffff, res, 2));
    res = __hmax(res, __shfl_xor_sync(0xffffffff, res, 1));
    return res;
}
template<>
__device__ __forceinline__ c10::BFloat16 __warpReduceMax<c10::BFloat16>(c10::BFloat16 __res){
    auto res = (reinterpret_cast<nv_bfloat16*>(&__res))[0];
    res = __hmax(res, __shfl_xor_sync(0xffffffff, res, 16));
    res = __hmax(res, __shfl_xor_sync(0xffffffff, res, 8));
    res = __hmax(res, __shfl_xor_sync(0xffffffff, res, 4));
    res = __hmax(res, __shfl_xor_sync(0xffffffff, res, 2));
    res = __hmax(res, __shfl_xor_sync(0xffffffff, res, 1));
    return res;
}

template<typename T>
__device__ __forceinline__ T __maxCmp(T a, T b){
    return max(a, b);
}
template<>
__device__ __forceinline__ c10::Half __maxCmp<c10::Half>(c10::Half a, c10::Half b){
    return __hmax((reinterpret_cast<half*>(&a))[0], (reinterpret_cast<half*>(&b))[0]);
}
template<>
__device__ __forceinline__ c10::BFloat16 __maxCmp<c10::BFloat16>(c10::BFloat16 a, c10::BFloat16 b){
    return __hmax((reinterpret_cast<nv_bfloat16*>(&a))[0], (reinterpret_cast<nv_bfloat16*>(&b))[0]);
}

template<typename T>
__device__ __forceinline__ T __expImpl(T a){
    return exp(a);
}
template<>
__device__ __forceinline__ c10::Half __expImpl<c10::Half>(c10::Half a){
    return hexp((reinterpret_cast<half*>(&a))[0]);
}
template<>
__device__ __forceinline__ c10::BFloat16 __expImpl<c10::BFloat16>(c10::BFloat16 a){
    return hexp((reinterpret_cast<nv_bfloat16*>(&a))[0]);
}

// <<<(Total Rows / 8, 1, 1), (32, 8, 1)>>>
template<typename T>
__global__ void sparseBlockedSoftmax_warp(
    T *src, T *tgt,
    const int64_t *accumSrc, const int64_t *rowList, const int64_t *colList, const int64_t *graphTiling,
    const int64_t Dsrc, const int64_t H, const int64_t N
){
    const int curRow = blockIdx.x * COLPERBLK(WARPTHREADNUM) + threadIdx.y;
    const int curNode = curRow % N;
    const int curHead = curRow / N;

    constexpr int tReduceNum = WARPMAXCOL / 32;
    T tmpReg[tReduceNum];

    if (curRow < H * N){
        const int graphId = graphTiling[curNode * 2];
        const int tileId = graphTiling[curNode * 2 + 1];

        const int colSize = colList[graphId];
        const int64_t startIdx = curHead * Dsrc + accumSrc[graphId] + tileId * colSize;

        // 1. thread load & thread max
        T __max = -INFINITY;
        int regNum = 0;
        #pragma unroll
        for (int i=threadIdx.x; i < colSize; i+=32, regNum++){
            tmpReg[regNum] = src[startIdx + i];
            __max = __maxCmp<T>(__max, tmpReg[regNum]);
        }

        // 2. wrap reduce for global max
        T globalMax = __warpReduceMax<T>(__max);

        // 3. compute exp and sum
        float __sum = 0.0f;
        #pragma unroll
        for (int i=0; i < regNum; i++){
            tmpReg[i] = __expImpl<T>(tmpReg[i] - globalMax);
            __sum += tmpReg[i];
        }

        // 4. warp reduce for global sum
        float globalSum = __warpReduceSum<float>(__sum);

        // 5. store softmax result to global
        #pragma unroll
        for (int regId=0, i=threadIdx.x; i < colSize; i+=32, regId++){
            tgt[startIdx + i] = tmpReg[regId] / globalSum;
        }
    }
}

template<typename T>
__global__ void sparseBlockedSoftmaxBackward_warp(
    T *src, T *dsrc, T *tgt,
    const int64_t *accumSrc, const int64_t *rowList, const int64_t *colList, const int64_t *graphTiling,
    const int64_t Dsrc, const int64_t H, const int64_t N
){
    const int curRow = blockIdx.x * COLPERBLK(WARPTHREADNUM) + threadIdx.y;
    const int curNode = curRow % N;
    const int curHead = curRow / N;

    constexpr int tReduceNum = WARPMAXCOL / 32;
    T srcReg[tReduceNum];
    T dsrcReg[tReduceNum];

    if (curRow < H * N){
        const int graphId = graphTiling[curNode * 2];
        const int tileId = graphTiling[curNode * 2 + 1];

        const int colSize = colList[graphId];
        const int64_t startIdx = curHead * Dsrc + accumSrc[graphId] + tileId * colSize;

        // 1. thread load & thread sum
        float __sum = 0.0f;
        int regNum = 0;
        #pragma unroll
        for (int i=threadIdx.x; i < colSize; i+=32, regNum++){
            srcReg[regNum] = src[startIdx + i];
            dsrcReg[regNum] = dsrc[startIdx + i];
            __sum += srcReg[regNum] * dsrcReg[regNum];
        }

        // 2. wrap reduce for global sum
        float globalSum = __warpReduceSum<float>(__sum);

        // 3. store softmax grad result to global
        #pragma unroll
        for (int regId=0, i=threadIdx.x; i < colSize; i+=32, regId++){
            tgt[startIdx + i] = (dsrcReg[regId] - globalSum) * srcReg[regId];
        }
    }
}

// <<<(Total Rows, 1, 1), (256, 1, 1)>>>
template<typename T>
__global__ void sparseBlockedSoftmax_block(
    T *src, T *tgt,
    const int64_t *accumSrc, const int64_t *rowList, const int64_t *colList, const int64_t *graphTiling,
    const int64_t Dsrc, const int64_t H, const int64_t N
){
    const int curRow = blockIdx.x;
    const int curNode = curRow % N;
    const int curHead = curRow / N;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    constexpr int tReduceNum = BLKMAXCOL / BLKTHREADNUM;
    constexpr int warpNum = BLKTHREADNUM / 32;
    T tmpReg[tReduceNum];
    __shared__ float tmpShared[warpNum];

    if (curRow < H * N){
        const int graphId = graphTiling[curNode * 2];
        const int tileId = graphTiling[curNode * 2 + 1];

        const int colSize = colList[graphId];
        const int64_t startIdx = curHead * Dsrc + accumSrc[graphId] + tileId * colSize;

        // 1. thread load & thread max
        T __max = -INFINITY;
        int regNum = 0;
        #pragma unroll
        for (int i=threadIdx.x; i < colSize; i+=BLKTHREADNUM, regNum++){
            tmpReg[regNum] = src[startIdx + i];
            __max = max(__max, tmpReg[regNum]);
        }

        // 2. wrap reduce + shared reduce for global max
        __max = __warpReduceMax<T>(__max);
        if (laneId == 0){
            tmpShared[warpId] = __max;
        }
        __syncthreads();

        if (threadIdx.x == 0){
            __max = tmpShared[0];
            #pragma unroll
            for (int i=1; i < warpNum; i++){
                __max = __maxCmp<T>(__max, tmpShared[i]);
            }
            tmpShared[0] = __max;
        }
        __syncthreads();

        T globalMax = tmpShared[0];

        // 3. compute exp and sum
        float __sum = 0.0f;
        #pragma unroll
        for (int i=0; i < regNum; i++){
            tmpReg[i] = __expImpl<T>(tmpReg[i] - globalMax);
            __sum += tmpReg[i];
        }

        // 4. wrap reduce + shared reduce for global sum
        __sum = __warpReduceSum<float>(__sum);
        if (laneId == 0){
            tmpShared[warpId] = __sum;
        }
        __syncthreads();

        if (threadIdx.x == 0){
            __sum = tmpShared[0];
            #pragma unroll
            for (int i=1; i < warpNum; i++){
                __sum += tmpShared[i];
            }
            tmpShared[0] = __sum;
        }
        __syncthreads();

        float globalSum = tmpShared[0];

        // 5. store softmax result to global
        #pragma unroll
        for (int regId=0, i=threadIdx.x; i < colSize; i+=BLKTHREADNUM, regId++){
            tgt[startIdx + i] = tmpReg[regId] / globalSum;
        }
    }
}

template<typename T>
__global__ void sparseBlockedSoftmaxBackward_block(
    T *src, T *dsrc, T *tgt,
    const int64_t *accumSrc, const int64_t *rowList, const int64_t *colList, const int64_t *graphTiling,
    const int64_t Dsrc, const int64_t H, const int64_t N
){
    const int curRow = blockIdx.x;
    const int curNode = curRow % N;
    const int curHead = curRow / N;

    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    constexpr int tReduceNum = BLKMAXCOL / BLKTHREADNUM;
    constexpr int warpNum = BLKTHREADNUM / 32;
    T srcReg[tReduceNum];
    T dsrcReg[tReduceNum];
    __shared__ float tmpShared[warpNum];

    if (curRow < H * N){
        const int graphId = graphTiling[curNode * 2];
        const int tileId = graphTiling[curNode * 2 + 1];

        const int colSize = colList[graphId];
        const int64_t startIdx = curHead * Dsrc + accumSrc[graphId] + tileId * colSize;

        // 1. thread load & thread sum
        float __sum = 0.0f;
        int regNum = 0;
        #pragma unroll
        for (int i=threadIdx.x; i < colSize; i+=BLKTHREADNUM, regNum++){
            srcReg[regNum] = src[startIdx + i];
            dsrcReg[regNum] = dsrc[startIdx + i];
            __sum += srcReg[regNum] * dsrcReg[regNum];
        }

        // 2. wrap reduce + shared reduce for global sum
        __sum = __warpReduceSum<T>(__sum);
        if (laneId == 0){
            tmpShared[warpId] = __sum;
        }
        __syncthreads();

        if (threadIdx.x == 0){
            __sum = tmpShared[0];
            #pragma unroll
            for (int i=1; i < warpNum; i++){
                __sum += tmpShared[i];
            }
            tmpShared[0] = __sum;
        }
        __syncthreads();

        float globalSum = tmpShared[0];

        // 3. store softmax grad result to global
        #pragma unroll
        for (int regId=0, i=threadIdx.x; i < colSize; i+=BLKTHREADNUM, regId++){
            tgt[startIdx + i] = (dsrcReg[regId] - globalSum) * srcReg[regId];
        }
    }
}

template<typename T>
void cudaKernelLaunch(
    const dim3 &blockSize, const dim3 &threadSize, T *src, T *tgt,
    const int64_t *accumSrc, const int64_t *rowList, const int64_t *colList, const int64_t *graphTiling, const int64_t Dsrc, const int64_t H, const int64_t N,
    const int kernelDispatch=0
){
    switch (kernelDispatch){
        case 0: CALL_SOFTMAX_KERNEL(warp, T, blockSize, threadSize, BLKSOFTMAX_KERNEL_INPUT); break;
        case 1: CALL_SOFTMAX_KERNEL(block, T, blockSize, threadSize, BLKSOFTMAX_KERNEL_INPUT); break;
    }
}

template<typename T>
void cudaBackwardKernelLaunch(
    const dim3 &blockSize, const dim3 &threadSize, T *src, T *dsrc, T *tgt,
    const int64_t *accumSrc, const int64_t *rowList, const int64_t *colList, const int64_t *graphTiling, const int64_t Dsrc, const int64_t H, const int64_t N,
    const int kernelDispatch=0
){
    switch (kernelDispatch){
        case 0: CALL_SOFTMAX_BACKWARD_KERNEL(warp, T, blockSize, threadSize, BLKSOFTMAXBWD_KERNEL_INPUT); break;
        case 1: CALL_SOFTMAX_BACKWARD_KERNEL(block, T, blockSize, threadSize, BLKSOFTMAXBWD_KERNEL_INPUT); break;
    }
}

torch::Tensor __getTiling(const variable &rowList, const int64_t rowNum){
    const int graphNum = rowList.size(0);
    std::vector<int> graphTiling(rowNum * 2);

    auto rowListHost = rowList.cpu();
    auto rowListPtr = rowListHost.data_ptr<int64_t>();

    int count = 0;
    for (int i=0; i < graphNum; i++){
        for (int j=0, tileSize=rowListPtr[i]; j < tileSize; j++){
            graphTiling[count++] = i;
            graphTiling[count++] = j;
        }
    }
    return torch::tensor(graphTiling, rowList.options());
}

torch::Tensor sparseBlockedSoftmaxLaunch(
    const variable &src, const variable &accumSrc, const variable &rowList,
    const variable &colList
){
    INPUT_CHECKING(src); INPUT_CHECKING(accumSrc); INPUT_CHECKING(rowList); INPUT_CHECKING(colList);

    const int64_t Dsrc = accumSrc.index({-1}).item().toInt();
    const int64_t H = src.numel() / Dsrc;
    const int64_t N = rowList.sum().item().toInt();

    assert(src.size(-1) == Dsrc);

    auto graphTiling = __getTiling(rowList, N);

    const int maxCol = colList.max().item().toInt();
    const int blockNum = maxCol > WARPMAXCOL ? (H * N) : RANGE_COUNT(COLPERBLK(WARPTHREADNUM), (H * N));

    assert(maxCol < BLKMAXCOL);
    const int kernelDispatch = maxCol > WARPMAXCOL ? 1 : 0;

    const dim3 blockSize(blockNum, 1, 1);
    const dim3 threadSize = maxCol > WARPMAXCOL ? dim3(BLKTHREADNUM, 1, 1) : dim3(32, WARPTHREADNUM / 32, 1);

    auto out = torch::empty({H, Dsrc}, src.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::BFloat16, out.scalar_type(), "__sparseBlockedSoftmax", [&]{
        auto srcPtr = src.data_ptr<scalar_t>();
        auto tgtPtr = out.data_ptr<scalar_t>();

        auto accumSrcPtr = accumSrc.data_ptr<int64_t>();
        auto rowListPtr = rowList.data_ptr<int64_t>();
        auto colListPtr = colList.data_ptr<int64_t>();
        auto graphTilingPtr = graphTiling.data_ptr<int64_t>();

        cudaKernelLaunch<scalar_t>(
            blockSize, threadSize, srcPtr, tgtPtr, accumSrcPtr, rowListPtr, colListPtr, graphTilingPtr,
            Dsrc, H, N, kernelDispatch
        );
    });

    return out;
}

torch::Tensor sparseBlockedSoftmaxBackwardLaunch(
    const variable &src, const variable &dsrc, const variable &accumSrc, const variable &rowList, const variable &colList
){
    INPUT_CHECKING(src); INPUT_CHECKING(dsrc); INPUT_CHECKING(accumSrc); INPUT_CHECKING(rowList); INPUT_CHECKING(colList);

    const int64_t Dsrc = accumSrc.index({-1}).item().toInt();
    const int64_t H = src.numel() / Dsrc;
    const int64_t N = rowList.sum().item().toInt();

    assert(src.size(-1) == Dsrc);

    auto graphTiling = __getTiling(rowList, N);

    const int maxCol = colList.max().item().toInt();
    const int blockNum = maxCol > WARPMAXCOL ? (H * N) : RANGE_COUNT(COLPERBLK(WARPTHREADNUM), (H * N));

    assert(maxCol < BLKMAXCOL);
    const int kernelDispatch = maxCol > WARPMAXCOL ? 1 : 0;

    const dim3 blockSize(blockNum, 1, 1);
    const dim3 threadSize = maxCol > WARPMAXCOL ? dim3(BLKTHREADNUM, 1, 1) : dim3(32, WARPTHREADNUM / 32, 1);

    auto out = torch::empty({H, Dsrc}, src.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::BFloat16, out.scalar_type(), "__sparseBlockedSoftmaxBackward", [&]{
        auto srcPtr = src.data_ptr<scalar_t>();
        auto dsrcPtr = dsrc.data_ptr<scalar_t>();
        auto tgtPtr = out.data_ptr<scalar_t>();

        auto accumSrcPtr = accumSrc.data_ptr<int64_t>();
        auto rowListPtr = rowList.data_ptr<int64_t>();
        auto colListPtr = colList.data_ptr<int64_t>();
        auto graphTilingPtr = graphTiling.data_ptr<int64_t>();

        cudaBackwardKernelLaunch<scalar_t>(
            blockSize, threadSize, srcPtr, dsrcPtr, tgtPtr, accumSrcPtr, rowListPtr, colListPtr, graphTilingPtr,
            Dsrc, H, N, kernelDispatch
        );
    });

    return out;
}