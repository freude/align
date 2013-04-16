import sys
import re
import numpy as np
import h5py
import pylab
import mahotas

import warp
import sparse_6d
from scipy.ndimage import gaussian_filter

step = 64
k_structural = 1.0
k_shear = 1.0
k_flex = 1.0
k_link = 0.25

sqrt_2 = np.sqrt(2.0)

def hypot(a, b):
    return np.sqrt(a**2 + b**2)

def link_local(imnum, shape, links):
    print "Local links", imnum
    num_i_steps = int(shape[0] / step) + 1
    num_j_steps = int(shape[1] / step) + 1
    for i in range(num_i_steps):
        for j in range(num_j_steps):
            links.set_x(imnum, i, j, i)
            links.set_y(imnum, i, j, j)
            # structural
            links[imnum, i, j, imnum, i + 1, j] = (k_structural, 1)
            links[imnum, i, j, imnum, i, j + 1] = (k_structural, 1)
            # shear
            links[imnum, i, j, imnum, i + 1, j + 1] = (k_shear, sqrt_2)
            links[imnum, i + 1, j, imnum, i, j + 1] = (k_shear, sqrt_2)
            # flex
            links[imnum, i, j, imnum, i + 2, j] = (k_shear, 2)
            links[imnum, i, j, imnum, i, j + 2] = (k_shear, 2)

def link_nonlocal(imnum1, warp, imnum2, shape, warpshape, links):
    print "Cross links", imnum, imnum2
    num_i_steps = int(shape[0] / step) + 1
    num_j_steps = int(shape[1] / step) + 1
    i_vals = np.linspace(0, 1.0, num_i_steps)
    j_vals = np.linspace(0, 1.0, num_j_steps)
    delta_i = i_vals[1]
    delta_j = j_vals[1]
    for i in range(num_i_steps):
        for j in range(num_j_steps):
            dest_j, dest_i = warp(i_vals[i], j_vals[j])
            if i == j:
                print i_vals[i] * (shape[0] - 1), j_vals[j] * (shape[1] - 1), "warped to", \
                dest_i * (shape[0] - 1), dest_j * (shape[1] - 1)
            if (dest_i > 0) and (dest_j > 0) and \
                    (dest_i < 1.0) and (dest_j < 1.0):
                base_i = np.searchsorted(i_vals, dest_i) - 1
                base_j = np.searchsorted(j_vals, dest_j) - 1
                d_i = (dest_i - i_vals[base_i]) / delta_i
                d_j = (dest_j - j_vals[base_j]) / delta_j
                assert d_i >= 0 and d_i < 1.0
                assert d_j >= 0 and d_j < 1.0
                links[imnum1, i, j, imnum2, base_i, base_j] = (k_link, hypot(d_i, d_j))
                links[imnum1, i, j, imnum2, base_i + 1, base_j] = (k_link, hypot((1.0 - d_i), d_j))
                links[imnum1, i, j, imnum2, base_i, base_j + 1] = (k_link, hypot(d_i, 1.0 - d_j))
                links[imnum1, i, j, imnum2, base_i + 1, base_j + 1] = (k_link, hypot(1.0 - d_i, 1.0 - d_j))


if __name__ == '__main__':
    warpfile = open(sys.argv[1])
    lines = warpfile.readlines()[:3]
    r = re.compile('^.*: (.*) to (.*) : (.*)$')
    matches = [r.match(l) for l in lines]
    first_images = [m.group(1) for m in matches]
    second_images = [m.group(2) for m in matches]
    warps = [warp.Warp(m.group(3)) for m in matches]
    shape = mahotas.imread(first_images[0]).shape
    warpshape = warps[0].shape

    nodes_per_image = (int(shape[0] / step) + 1) * (int(shape[1] / step + 1))
    springs = sparse_6d.Sparse6DSpring(len(warps) + 1, int(shape[0] / step) + 1, int(shape[1] / step) + 1)
    for imnum in range(len(warps) + 1):
        link_local(imnum, shape, springs)
    for imnum, w in enumerate(warps):
        link_nonlocal(imnum, w, imnum + 1, shape, warpshape, springs)

    while springs.step() > 10:
        pass

    # generate new maps
    pad = 128
    num_i_steps = int(shape[0] / step) + 1
    num_j_steps = int(shape[1] / step) + 1
    for imnum in range(len(warps) + 1):
        warpim_i = np.zeros((shape[0] + 2 * pad, shape[1] + 2 * pad))
        warpim_j = np.zeros((shape[0] + 2 * pad, shape[1] + 2 * pad))
        weightim = np.zeros((shape[0] + 2 * pad, shape[1] + 2 * pad))
        for i in range(num_i_steps):
            for j in range(num_j_steps):
                non_warped_i = i / float(num_i_steps) * shape[0]
                non_warped_j = j / float(num_j_steps) * shape[1]
                warped_i = springs.get_x(imnum, i, j) / float(num_i_steps) * shape[0]
                warped_j = springs.get_y(imnum, i, j) / float(num_i_steps) * shape[1]
                warpim_i[pad + int(warped_i), pad + int(warped_j)] = non_warped_i - warped_i
                warpim_j[pad + int(warped_i), pad + int(warped_j)] = non_warped_j - warped_j
                weightim[pad + int(warped_i), pad + int(warped_j)] = 1
        gaussian_filter(warpim_i, step, output=warpim_i)
        gaussian_filter(warpim_j, step, output=warpim_j)
        gaussian_filter(weightim, step, output=weightim)
        warpim_i /= weightim;
        warpim_j /= weightim;
        warpim_i += (np.arange(-pad, shape[0] + pad)).reshape((-1, 1))
        warpim_j += (np.arange(-pad, shape[1] + pad)).reshape((1, -1))
        im = mahotas.imread(first_images[imnum])
        ii = warpim_i.astype(int)
        jj = warpim_j.astype(int)
        bad = (ii < 0) | (ii >= im.shape[0]) | (jj < 0) | (jj >= im.shape[1])
        ii[bad] = 0
        jj[bad] = 0
        outim = im[ii, jj]
        pylab.imshow(outim, cmap=pylab.cm.gray)
        pylab.figure()
        pylab.imshow(im,  cmap=pylab.cm.gray)
        pylab.show()
