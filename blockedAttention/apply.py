import torch
import blockAttention

from typing import Optional

def blockGEMM(
    mat1: torch.Tensor, mat2: torch.Tensor,
    mat1Accum: Optional[torch.Tensor]=None, mat2Accum: Optional[torch.Tensor]=None, outAccum: Optional[torch.Tensor]=None,
    MList: Optional[torch.Tensor]=None, NList: Optional[torch.Tensor]=None, KList: Optional[torch.Tensor]=None,
    transpose1: Optional[bool]=False, transpose2: Optional[bool]=False
) -> torch.Tensor:
    '''
    sparseBlockedGEMM wrapper, need to provide either accumulate results(for matrix start index) or MNK List(for matrix shape)

    mat1 -> (H, (m1*k1 + m2*k2 + ...)), mat2 -> (H, (n1*k1 + n2*k2 + ...)), out -> (H, (m1*n1 + m2*n2 + ...))

    mat1Accum -> (0, m1*k1, (m1*k1) + m2*k2...), mat2Accum -> (0, n1*k1, (n1*k1) + n2*k2...), outAccum -> (0, m1*n1, (m1*n1) + m2*n2...)

    MList -> (m1, m2, ...), NList -> (n1, n2, ...), KList -> (k1, k2, ...)

    default transpose1 = false: mat1(M, K) -> true: mat1(K, M)
    default transpose2 = false: mat2(N, K) -> true: mat1(K, N)
    '''

    return blockAttention.sparseBlockedGEMM(mat1, mat2, mat1Accum, mat2Accum, outAccum, MList, NList, KList, transpose1, transpose2)

def blockSoftmax(
    src: torch.Tensor,
    rowList: torch.Tensor, colList: torch.Tensor, srcAccum: Optional[torch.Tensor]=None
) -> torch.Tensor:
    '''
    sparseBlockedSoftmax wrapper (only for last dim), need to provide rowList and colList, the accumulate results is optional

    src -> (H, (m1*k1 + m2*k2 + ...)), out -> (H, (m1*k1 + m2*k2 + ...))

    accumSrc -> (0, m1*k1, (m1*k1) + m2*k2...)

    rowList -> (m1, m2, ...), colList -> (k1, k2, ...)
    '''

    return blockAttention.sparseBlockedSoftmax(src, srcAccum, rowList, colList)