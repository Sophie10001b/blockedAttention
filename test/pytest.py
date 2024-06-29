import torch
import blockedAttention

from torch.profiler import profile

if __name__ == "__main__":
    h = 128
    m = 33
    n = 99
    k = 64

    __type = torch.float
    __device = "cuda:0"

    dummyA = torch.randn((h, m*k + n*k + k*k), dtype=__type, device=__device)
    dummyB = torch.randn((h, m*k + n*k + k*k), dtype=__type, device=__device)
    MList = torch.tensor([m, n, k], dtype=torch.long, device=__device)
    NList = torch.tensor([m, n, k], dtype=torch.long, device=__device)
    KList = torch.tensor([k, k, k], dtype=torch.long, device=__device)

    dummyC = torch.randn((h, 3, max(m, n, k), k), dtype=__type, device=__device)
    dummyD = torch.randn((h, 3, max(m, n, k), k), dtype=__type, device=__device)

    dummyA.requires_grad_()
    dummyB.requires_grad_()
    dummyC.requires_grad_()
    dummyD.requires_grad_()

    dummyA.retain_grad()
    dummyB.retain_grad()
    dummyC.retain_grad()
    dummyD.retain_grad()

    with profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.profiler.record_function("blockedGEMM"):
            res = blockedAttention.blockGEMM(dummyA, dummyB, None, None, None, MList, NList, KList)
        with torch.profiler.record_function("blockedSoftmax"):
            res = blockedAttention.blockSoftmax(res, MList, NList)
        with torch.profiler.record_function("naiveGEMM"):
            checkRes = torch.matmul(dummyC, dummyD.transpose(-1, -2))
        with torch.profiler.record_function("naiveSoftmax"):
            checkRes = checkRes.softmax(-1)
        with torch.profiler.record_function("blockedGEMM_bw"):
            res.mean().backward()
        with torch.profiler.record_function("naiveGEMM_bw"):
            checkRes.mean().backward()

    # gap = (res - checkRes.reshape(h, -1)).abs().mean()
    # gapGrad1 = (dummyA.grad - dummyC.grad).abs().mean()
    # gapGrad2 = (dummyB.grad - dummyD.grad).abs().mean()

    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))

    softmax_raw = blockedAttention.blockSoftmax(dummyA, MList, KList)
    softmax_check = torch.cat([
        dummyA[:, :m*k].reshape((h, m, k)).softmax(-1).reshape((h, -1)),
        dummyA[:, m*k:(m*k + n*k)].reshape((h, n, k)).softmax(-1).reshape((h, -1)),
        dummyA[:, (m*k + n*k):].reshape((h, k, k)).softmax(-1).reshape((h, -1)),
    ], -1)

    gap = (softmax_raw - softmax_check).abs().mean()
    print(gap)
