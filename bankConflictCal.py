import numpy as np

if __name__ == "__main__":
    bid = []
    properPad = 0
    maxConflict = 32

    bk = 32
    bm = 32

    for pad in range(0, 32):
        curBid = []
        for i in range(32):
            r = (i * 4) // bk
            c = (i * 4) % bk
            curBid.append((c * (bm+pad) + r) % 32)
        curBid = np.bincount(np.array(curBid, int))
        curBid[curBid < 2] = 0
        curBid[curBid > 1] -= 1
        if curBid.sum() < maxConflict:
            maxConflict = curBid.sum()
            properPad = pad
    pass
    