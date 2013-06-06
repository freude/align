import sys
import re
import gc

import numpy as np
import cv2

import template_matching
import ransac
import weakref

# setup for multithreading
import threading
from multiprocessing.pool import ThreadPool
if not hasattr(threading.current_thread(), "_children"):
    threading.current_thread()._children = weakref.WeakKeyDictionary()

STEP = 128
TEMPLATE = 128
WINDOW = 512

def coords(filename):
    print filename
    m = re.match('.*Tile_r(?P<Row>[0-9]+)-c(?P<Col>[0-9]+)_.*', filename)
    assert m is not None
    return int(m.group('Row')) - 1, int(m.group('Col')) - 1

def angle_from_rotation_matrix(R):
    return np.arcsin(R[1, 0])

def rigid_overlap(imf1, imf2, overlap, dr=0, dc=0, pool=None):
    assert dr or dc
    im1 = cv2.imread(imf1, flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # extract subimage and make a new copy to free up memory
    orig_im1_shape = im1.shape
    if dr:
        im1 = im1[-overlap:, :].copy()
    else:
        im1 = im1[:, -overlap:].copy()
    im2 = cv2.imread(imf2, flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)

    gc.collect()

    # grid of template locations (upper left corners)
    trows, tcols = np.mgrid[0:im1.shape[0]:STEP,
                            0:im1.shape[1]:STEP]
    trows = trows.astype(np.int32)
    tcols = tcols.astype(np.int32)
    # window locations are centered on template locations
    wrows = trows - WINDOW // 2 + TEMPLATE // 2
    wcols = tcols - WINDOW // 2 + TEMPLATE // 2
    wrows = np.clip(wrows, 0, im2.shape[0] - WINDOW)
    wcols = np.clip(wcols, 0, im2.shape[1] - WINDOW)

    new_rows = np.zeros(trows.shape, np.int32)
    new_cols = np.zeros(trows.shape, np.int32)
    scores = np.zeros(trows.shape, np.float32)

    def match_row(rowidx):
        template_matching.best_matches(trows[rowidx, :], tcols[rowidx, :],
                                       wrows[rowidx, :], wcols[rowidx, :],
                                       TEMPLATE, WINDOW,
                                       im1, im2,
                                       new_rows[rowidx, :], new_cols[rowidx, :], scores[rowidx, :])

    nothreads = False
    if pool is None:
        for rr in range(trows.shape[0]):
            match_row(rr)
    else:
        newpts = pool.map_async(match_row, range(trows.shape[0]))
        newpts.wait()

    # find ransac estimate of rotation and translation using strongest matches
    medscore = np.median(scores[scores > 0])
    mask = scores >= medscore
    X = np.row_stack((new_rows[mask], new_cols[mask]))
    Y = np.row_stack((trows[mask], tcols[mask]))
    R, T = ransac.estimate_rigid_transformation(X, Y)

    # adjust for clipping of im1
    if dr:
        T[0] += orig_im1_shape[0] - overlap
    if dc:
        T[1] += orig_im1_shape[1] - overlap
    print imf1, imf2, R.tolist(), T.tolist()
    return R, T

if __name__ == '__main__':
    overlap = 2048
    sys.argv.pop(0)
    if sys.argv[0].startswith('--overlap'):
        overlap = int(sys.argv[1])
        sys.argv = sys.argv[2:]

    outfile = sys.argv.pop(0)

    pool = ThreadPool(8)


    imcoords = dict((coords(imf), imf) for imf in sys.argv)
    maxrow = max(c[0] for c in imcoords.keys())
    maxcol = max(c[1] for c in imcoords.keys())
    R = {(0, 0) : np.matrix([[1, 0], [0, 1]])}
    T = {(0, 0) : np.matrix([[0], [0]])}
    for r in range(maxrow + 1):
        for c in range(maxcol + 1):
            if r == c == 0:
                continue
            print ""
            print r, c
            if r > 0:
                Rr, Tr = rigid_overlap(imcoords[r - 1, c], imcoords[r, c],
                                       overlap,
                                       dr=1, dc=0,
                                       pool=pool)
                if c == 0:
                    R[r, c] = R[r - 1, c] * Rr
                    T[r, c] = T[r - 1, c] + Tr
                    continue
            if c > 0:
                Rc, Tc = rigid_overlap(imcoords[r, c - 1], imcoords[r, c],
                                       overlap,
                                       dr=0, dc=1,
                                       pool=pool)
                if r == 0:
                    R[r, c] = R[r, c - 1] * Rc
                    T[r, c] = T[r, c - 1] + Tc
                    continue
            angle = (angle_from_rotation_matrix(R[r - 1, c] * Rr) +
                     angle_from_rotation_matrix(R[r, c - 1] * Rc)) / 2.0
            R[r, c] = np.matrix([[np.cos(angle), - np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            T[r, c] = (T[r - 1, c] + Tr + T[r, c - 1] + Tc) / 2.0

    outf = open(outfile, "w")
    for r in range(maxrow + 1):
        for c in range(maxcol + 1):
            outf.write("%s R: %s T: %s\n" % (imcoords[r, c], R[r, c].tolist(), T[r, c].tolist()))
    outf.close()
