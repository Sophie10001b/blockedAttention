#pragma once

#define WARPTHREADNUM 256
#define BLKTHREADNUM 256
#define WARPMAXCOL 1024
#define BLKMAXCOL 8192
#define COLPERBLK(thread) (thread / 32)

#define CALL_SOFTMAX_KERNEL(Dispatch, T, BLOCK, THREAD, ...) sparseBlockedSoftmax_##Dispatch<T><<<BLOCK, THREAD>>>(__VA_ARGS__)

#define CALL_SOFTMAX_BACKWARD_KERNEL(Dispatch, T, BLOCK, THREAD, ...) sparseBlockedSoftmaxBackward_##Dispatch<T><<<BLOCK, THREAD>>>(__VA_ARGS__)

#define BLKSOFTMAX_KERNEL_INPUT src, tgt, accumSrc, rowList, colList, graphTiling, Dsrc, H, N

#define BLKSOFTMAXBWD_KERNEL_INPUT src, dsrc, tgt, accumSrc, rowList, colList, graphTiling, Dsrc, H, N