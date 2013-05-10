import sys
import os
import threading
import numpy as np
import cv2
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


images = [f.strip() for f in open(sys.argv[1])]
warpdir = sys.argv[2]

def load_warp(idx1, idx2):
    fname = os.path.join(warpdir, "warp.%d.to.%d.hdf5" % (idx1 + 1, idx2 + 1))
    return NonlinearWarp.load(fname)

forward_warps = dict(((idx, idx + 1), load_warp(idx, idx + 1)) for idx in range(len(images) - 1))
backward_warps = dict(((idx + 1, idx), load_warp(idx + 1, idx)) for idx in range(len(images) - 1))

lastim = len(images) - 1
for idx in range(2, len(images)):
    fw = forward_warps[0, idx] = forward_warps[0, idx - 1].chain(forward_warps[idx - 1, idx])
    print idx, fw.R[0,0], fw.T.ravel()

for idx in range(lastim - 2, -1, -1):
    bw = backward_warps[lastim, idx] = backward_warps[lastim, idx + 1].chain(backward_warps[idx + 1, idx])
    print idx, bw.R[0,0], bw.T.ravel()

forward_warps[0, 0] = backward_warps[lastim, lastim] = NonlinearWarp.identity(forward_warps[0, 1].row_warp.shape)
for idx in range(len(images)):
    t = idx / float(len(images) - 1)
    w = NonlinearWarp.lerp(forward_warps[0, idx], backward_warps[lastim, idx], t)
    # Correct so that the rigid transformation from 0 to lastim still occurs.
    # Otherwise, any rigid shift between keypoints causes drift in the
    # intermediate interpolations.
    w.correct(t, forward_warps[0, lastim])
    im = cv2.resize(cv2.imread(images[idx], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE), (1024, 1024))
    im = w.warp([im], im.shape)[0]
    cv2.imwrite(os.path.join("OUT", "out%02d.tif" % idx), im)
