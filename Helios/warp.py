import numpy as np
import remap
import cv2
import pylab

import template_matching

class Warp(object):
    def __init__(self):
        pass

class RigidWarp(Warp):
    def __init__(self, R, T):
        '''R, T should take normalized coordinates to normalized coordinates, in [[j, i]].T layout.'''
        Warp.__init__(self)
        self.R = R
        self.T = T

    def warp(self, sources, dest_shape, dests=None, repeat=False):
        src_i, src_j = self.resize(dest_shape)
        src_i *= (sources[0].shape[0] - 1)
        src_j *= (sources[0].shape[1] - 1)
        if dests is None:
            dests = [np.zeros(dest_shape) for s in sources]
        for s, d in zip(sources, dests):
            remap.remap(s, src_j, src_i, d, repeat=repeat)
        return dests

    def resize(self, sz):
        dst_i, dst_j = np.mgrid[:sz[0], :sz[1]]
        dst_i = dst_i.astype(np.float32) / (sz[0] - 1)
        dst_j = dst_j.astype(np.float32) / (sz[1] - 1)
        coords = np.vstack((dst_j.ravel(), dst_i.ravel()))
        coords = (self.R * coords + self.T).A
        src_j = coords[0, :].reshape(sz)
        src_i = coords[1, :].reshape(sz)
        return src_i, src_j


class NonlinearWarp(Warp):
    def __init__(self, row_warp, column_warp):
        Warp.__init__(self)
        self.row_warp = row_warp
        self.column_warp = column_warp
        assert row_warp.shape == column_warp.shape

    def warp(self, sources, dest_shape, dests=None, repeat=False):
        row_warp, column_warp = self.resize(dest_shape)
        src_i = row_warp * (sources[0].shape[0] - 1)
        src_j = column_warp * (sources[0].shape[1] - 1)
        if dests is None:
            dests = [np.zeros(dest_shape) for s in sources]
        for s, d in zip(sources, dests):
            remap.remap(s, src_j, src_i, d, repeat=repeat)
        return dests

    def resize(self, sz):
        row_warp = self.row_warp
        column_warp = self.column_warp
        # resize the warps to be the size of the output (using remap)
        if row_warp.shape != tuple(sz):
            temp_i = np.linspace(0, row_warp.shape[0] - 1, sz[0]).reshape((-1, 1))
            temp_j = np.linspace(0, row_warp.shape[1] - 1, sz[1]).reshape((1, -1))
            temp_i, temp_j = np.broadcast_arrays(temp_i, temp_j)
            new_row = np.zeros(sz, dtype=np.float32)
            new_column = new_row.copy()
            remap.remap(row_warp, temp_j, temp_i, new_row)
            remap.remap(column_warp, temp_j, temp_i, new_column)
            row_warp = new_row
            column_warp = new_column
            return row_warp, column_warp
        return row_warp.copy(), column_warp.copy()

def refine_warp(prev_warp, im1, im2, template_size, window_size, step_size):
    # warp im2's coordinates to im1's space
    dest_shape = (np.array(im1.shape) // step_size) + 1
    normalized_i, normalized_j = np.ogrid[:dest_shape[0], :dest_shape[1]]
    normalized_i = normalized_i.astype(float) / (dest_shape[0] - 1)
    normalized_j = normalized_j.astype(float) / (dest_shape[1] - 1)
    row_warp, column_warp = prev_warp.resize(dest_shape)
    orig_row_warp = row_warp.copy()
    orig_column_warp = column_warp.copy()
    weights = np.zeros_like(row_warp)
    for i in range(dest_shape[0]):
        print i, '/', dest_shape[0]
        for j in range(dest_shape[1]):
            i1 = (im1.shape[0] - 1) * normalized_i[i, 0]
            j1 = (im1.shape[1] - 1) * normalized_j[0, j]
            i2 = (im2.shape[0] - 1) * row_warp[i, j]
            j2 = (im2.shape[1] - 1) * column_warp[i, j]
            newr, newc, w = best_match((i1, j1), (i2, j2), im1, im2, template_size, window_size)
            # Threshold matches at 4 stdev above the mean
            if w > 4.0:
                # convert to deltas for better interpolation
                row_warp[i, j] = float(newr) / (im2.shape[0] - 1) - normalized_i[i, 0]
                column_warp[i, j] = float(newc) / (im2.shape[1] - 1) - normalized_j[0, j]
                weights[i, j] = w
    # smooth and normalize
    # we use weights squared 
    weights = weights ** 2
    weighted_r = row_warp * weights
    weighted_c = column_warp * weights
    # Loop enough that there should be weight everywhere.
    # Filter radius for sigma=3 is approximately 10
    for iter in range(max(weights.shape) / 10 + 1):
        weights = cv2.GaussianBlur(weights, (0, 0), 3)
        weighted_r = cv2.GaussianBlur(weighted_r, (0, 0), 3)
        weighted_c = cv2.GaussianBlur(weighted_c, (0, 0), 3)
    # convert back from deltas
    new_r = (weighted_r / weights) + normalized_i
    new_c = (weighted_c / weights) + normalized_j
    # Keep the old warp values anywhere we don't have new data
    zeromask = weights == 0
    weights[zeromask] = 1
    weighted_r[zeromask] = orig_row_warp[zeromask]
    weighted_c[zeromask] = orig_column_warp[zeromask]
    return NonlinearWarp(new_r, new_c)

def best_match(pt1, pt2, im1, im2, template_size, window_size):
    # cut out template
    r1, c1 = pt1
    r1 -= template_size // 2
    c1 -= template_size // 2
    if r1 < 0: r1 = 0
    if r1 > im1.shape[0] - template_size: r1 = im1.shape[0] - template_size
    if c1 < 0: c1 = 0
    if c1 > im1.shape[1] - template_size: c1 = im1.shape[1] - template_size
    template = im1[r1:(r1 + template_size), c1:(c1 + template_size)]
    # cut out window
    r2, c2 = pt2
    r2 -= window_size // 2
    c2 -= window_size // 2
    if r2 < 0: r2 = 0
    if r2 > im2.shape[0] - window_size: r2 = im2.shape[0] - window_size
    if c2 < 0: c2 = 0
    if c2 > im2.shape[1] - window_size: c2 = im2.shape[1] - window_size
    window = im2[r2:(r2 + window_size), c2:(c2 + window_size)]
    # Run normalized cross-correlation
    match = cv2.matchTemplate(window, template, cv2.TM_CCORR_NORMED)
    # template_matching.best_match(r1, c1, r2, c2, template_size, window_size,
    # im1, im2)
    # find highest point
    bestr, bestc = np.unravel_index(match.argmax(), match.shape)
    score = (match[bestr, bestc] - match.mean()) / match.std()
    if np.isnan(score):
        score = 0
    bestr = r2 + bestr - (r1 - pt1[0])
    bestc = c2 + bestc - (c1 - pt1[1])
    return bestr, bestc, score
