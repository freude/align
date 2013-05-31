import sys
import os
import threading
import numpy as np
import cv2
import gc
from warp import NonlinearWarp


def display_warp(w, im1, im2):
    warpedim = w.warp([im2], im1.shape)[0]
    while True:
        cv2.imshow("view", warpedim.astype(im1.dtype))
        k = cv2.waitKey()
        if k == 27:
            break
        cv2.imshow("view", im1)
        k = cv2.waitKey()
        if k == 27:
            break

def interpolate_warps(images, warpdir, outdir, offset, prevR, prevT):
    def load_warp(idx1, idx2):
        fname = os.path.join(warpdir, "warp.%03d.to.%03d.hdf5" % (offset + idx1 + 1, offset + idx2 + 1))
        print "   loading", fname
        return NonlinearWarp.load(fname)

    forward_warps = dict(((idx, idx + 1), load_warp(idx, idx + 1))
                         for idx in range(len(images) - 1))
    backward_warps = dict(((idx + 1, idx), load_warp(idx + 1, idx))
                          for idx in range(len(images) - 1))

    lastim = len(images) - 1
    for idx in range(2, len(images)):
        fw = forward_warps[0, idx] = forward_warps[0, idx - 1].chain(forward_warps[idx - 1, idx])

    for idx in range(lastim - 2, -1, -1):
        bw = backward_warps[lastim, idx] = backward_warps[lastim, idx + 1].chain(backward_warps[idx + 1, idx])

    print "done chaining"

    forward_warps[0, 0] = backward_warps[lastim, lastim] = NonlinearWarp.identity(forward_warps[0, 1].row_warp.shape)
    for idx in range(len(images)):
        outf = os.path.join(outdir, "out%04d.tif" % (offset + idx + 1))
        t = idx / float(len(images) - 1)
        w = NonlinearWarp.lerp(forward_warps[0, idx], backward_warps[lastim, idx], t)
        # Correct so that the rigid transformation from 0 to lastim still occurs.
        # Otherwise, any rigid shift between keypoints causes drift in the
        # intermediate interpolations.
        w.correct(t, forward_warps[0, lastim], prevR, prevT)
        if os.path.exists(outf):
           continue
        im = cv2.resize(cv2.imread(images[idx], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE), (1024, 1024))
        print "warping"
        im = w.warp([im], im.shape)[0]
        print "   writing", offset + idx + 1
        cv2.imwrite(outf, im)
    gc.collect()
    return w.R, w.T

if __name__ == '__main__':
    step = 30
    images = [f.strip() for f in open(sys.argv[1])]
    warpdir = sys.argv[2]
    outdir = sys.argv[3]

    prevR = prevT = None
    for base in range(0, len(images), step - 1):
        end = min(base + step, len(images))
        print "GLOBALIZING", base + 1, end
        prevR, prevT = interpolate_warps(images[base:end], warpdir, outdir, base, prevR, prevT)
