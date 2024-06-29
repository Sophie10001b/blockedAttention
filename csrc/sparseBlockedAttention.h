#pragma once
#include <iostream>
#include <torch/extension.h>
#include <ATen/Dispatch.h>

#define CUDA_CHECKING(x) TORCH_CHECK(x.type().is_cuda(), #x, "must be CUDA Tensor")
#define CONTIGUOUS_CHECKING(x) TORCH_CHECK(x.is_contiguous(), #x, "must be contiguous Tensor")
#define INPUT_CHECKING(x) CUDA_CHECKING(x); CONTIGUOUS_CHECKING(x)

#define BUILD_PYBIND11 0

#define RANGE_COUNT(m, n) ((m + n - 1) / m)

using autogradContext = torch::autograd::AutogradContext;
using variable = torch::autograd::Variable;
using variableList = torch::autograd::variable_list;

torch::Tensor sparseBlockedGEMMLaunch(const variable &mat1, const variable &mat2, const variable &mat1Accum, const variable &mat2Accum, const variable &outAccum, const variable &MList, const variable &NList, const variable &KList, const bool transpose1=false, const bool transpose2=false);

torch::Tensor sparseBlockedSoftmaxLaunch(const variable &src, const variable &accumSrc, const variable &rowList, const variable &colList);
torch::Tensor sparseBlockedSoftmaxBackwardLaunch(const variable &out, const variable &dout, const variable &accumSrc, const variable &rowList, const variable &colList);

struct tilingData{
    torch::Tensor graphTiling;
    const int BMIdx;
    const int BNIdx;
    const int BKIdx;

    tilingData(torch::Tensor tiling, int midx, int nidx, int kidx): graphTiling(tiling), BMIdx(midx), BNIdx(nidx), BKIdx(kidx){}
};