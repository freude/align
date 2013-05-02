import numpy as np

def lerp(a, b, t):
    return (b - a) * t + a

def remap(src, warpcol, warprow, dest, repeat=False):
    if repeat:
        warpcol[warpcol < 0] = 0
        warpcol[warpcol >= src.shape[1]] = (src.shape[1] - 1)
        warprow[warprow < 0] = 0
        warprow[warprow >= src.shape[0]] = (src.shape[0] - 1)
        mask = np.ones_like(warpcol, dtype=np.bool)
    else:
        mask = (warpcol >= 0) & (warprow >= 0) & (warpcol < src.shape[1] - 1) & (warprow < src.shape[1] - 1)
        warpcol = warpcol[mask]
        warprow = warprow[mask]

    intcol = warpcol.astype(int)
    introw = warprow.astype(int)
    intcol[intcol == src.shape[1] - 1] = src.shape[1] - 2
    introw[introw == src.shape[0] - 1] = src.shape[0] - 2
    tc = (warpcol - intcol)
    tr = (warprow - introw)
    if repeat:
        dest[...] = lerp(lerp(src[introw, intcol], src[introw + 1, intcol], tr),
                         lerp(src[introw, intcol + 1], src[introw + 1, intcol + 1], tr),
                         tc)
    else:
        dest[mask] = lerp(lerp(src[introw, intcol], src[introw + 1, intcol], tr),
                          lerp(src[introw, intcol + 1], src[introw + 1, intcol + 1], tr),
                          tc)
