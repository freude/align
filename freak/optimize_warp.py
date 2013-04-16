import sys
import re
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg, eye
from pyamg import smoothed_aggregation_solver
import h5py
import pylab
import allow_quit

import warp
import sparse_6d

step = 64
intra_weight = 0.5

def embed_local(imnum, shape, weights):
    num_i_steps = int(shape[0] / step) + 1
    num_j_steps = int(shape[1] / step) + 1
    def n(k, c):
        return (c if (k == 0) else -(c + 1))
    # corners
    for i, j in zip([0, 0, -1, -1], [0, -1, -1, 0]):
        weights[imnum, i, j, imnum, i, n(j, 1)] = 2
        weights[imnum, i, j, imnum, i, n(j, 2)] = -1
        weights[imnum, i, j, imnum, n(i, 1), j] = 2
        weights[imnum, i, j, imnum, n(i, 2), j] = -1
    # edges 
    for i in [0, -1]:
        for j in range(1, num_j_steps - 1):
            weights[imnum, i, j, imnum, i, j + 1] = 1
            weights[imnum, i, j, imnum, i, j - 1] = 1
            weights[imnum, i, j, imnum, n(i, 1), j] = 2
            weights[imnum, i, j, imnum, n(i, 2), j] = -1
    for j in [0, -1]:
        for i in range(1, num_i_steps - 1):
            weights[imnum, i, j, imnum, i + 1, j] = 1
            weights[imnum, i, j, imnum, i - 1, j] = 1
            weights[imnum, i, j, imnum, i, n(j, 1)] = 2
            weights[imnum, i, j, imnum, i, n(j, 2)] = -1
    # interior
    for i in range(1, num_i_steps - 1):
        for j in range(1, num_j_steps - 1):
            weights[imnum, i, j, imnum, i + 1, j] = 1
            weights[imnum, i, j, imnum, i - 1, j] = 1
            weights[imnum, i, j, imnum, i, j + 1] = 1
            weights[imnum, i, j, imnum, i, j - 1] = 1

def embed_nonlocal(imnum1, warp, imnum2, shape, weights):
    num_i_steps = int(shape[0] / step) + 1
    num_j_steps = int(shape[1] / step) + 1
    i_vals = np.linspace(0, shape[0] - 1, num_i_steps)
    j_vals = np.linspace(0, shape[1] - 1, num_j_steps)
    delta_i = i_vals[1]
    delta_j = j_vals[1]
    for i in range(num_i_steps):
        for j in range(num_j_steps):
            dest_j, dest_i = warp(i_vals[i], j_vals[j])
            print i_vals[i], j_vals[j], "warpted to", dest_i, dest_j
            if (dest_i > 0) and (dest_j > 0) and \
                    (dest_i < shape[0] - 1) and (dest_j < shape[1] - 1):
                base_i = np.searchsorted(i_vals, dest_i) - 1
                base_j = np.searchsorted(j_vals, dest_j) - 1
                w_i = 1.0 - (dest_i - i_vals[base_i]) / delta_i
                w_j = 1.0 - (dest_j - j_vals[base_j]) / delta_j
                weights[imnum1, i, j, imnum2, base_i, base_j] = intra_weight * w_i * w_j
                weights[imnum1, i, j, imnum2, base_i, base_j + 1] = intra_weight * w_i * (1.0 - w_j)
                weights[imnum1, i, j, imnum2, base_i + 1, base_j] = intra_weight * (1.0 - w_i) * w_j
                weights[imnum1, i, j, imnum2, base_i + 1, base_j + 1] = intra_weight * (1.0 - w_i) * (1.0 - w_j)

# from https://gist.github.com/fabianp/934363
# Author: Fabian Pedregosa
def locally_linear_embedding(W, out_dim, i_dim, j_dim, tol=1e-5, max_iter=2000):
    # M = (I-W)' (I-W)
    A = eye(*W.shape, format=W.format) - W
    A = (A.T).dot(A).tocsr()

    # initial approximation to the eigenvectors - use coords
    X = np.random.rand(W.shape[0], out_dim)
    X[:, 1] = np.arange(W.shape[0]) % j_dim
    X[:, 0] = (np.arange(W.shape[0], dtype=int) / j_dim) % i_dim
    ml = smoothed_aggregation_solver(A, symmetry='symmetric')
    prec = ml.aspreconditioner()

    # compute eigenvalues and eigenvectors with LOBPCG
    eigen_values, eigen_vectors = linalg.lobpcg(
        A, X, M=prec, largest=False, tol=tol, maxiter=max_iter)

    index = np.argsort(eigen_values)
    return eigen_vectors[:, index], np.sum(eigen_values)


if __name__ == '__main__':
    warpfile = open(sys.argv[1])
    lines = warpfile.readlines()[:1]
    r = re.compile('^.*: (.*) to (.*) : (.*)$')
    matches = [r.match(l) for l in lines]
    first_images = [m.group(1) for m in matches]
    second_images = [m.group(2) for m in matches]
    warps = [warp.Warp(m.group(3)) for m in matches]
    shape = warps[0].shape

    nodes_per_image = (int(shape[0] / step) + 1) * (int(shape[1] / step + 1))
    weights = sparse_6d.Sparse6D(len(warps) + 1, int(shape[0] / step) + 1, int(shape[1] / step) + 1)
    for imnum in range(len(warps) + 1):
        embed_local(imnum, shape, weights)
    for imnum, w in enumerate(warps):
        if 0:
            embed_nonlocal(imnum, w, imnum + 1, shape, weights)
    weights.normalize()
    print "solving"
    X_r, cost = locally_linear_embedding(weights.sp, 2, shape[0], shape[1])
    print cost
    import pylab
    pylab.scatter(X_r[:, 0], X_r[:, 1])
    pylab.show()
