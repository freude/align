import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import numpy as np
import sys


def crank_nicholson(im, diffusitivity, beta):
    r, c = im.shape
    N = r * c
    i_indices, j_indices = np.mgrid[0:r, 0:c]
    left_indices = i_indices - 1
    right_indices = i_indices + 1
    up_indices = j_indices - 1
    down_indices = j_indices + 1

    # reflect boundary
    left_indices[left_indices < 0] = 1
    right_indices[right_indices == r] = r - 2
    up_indices[up_indices < 0] = 1
    down_indices[down_indices == c] = c - 2

    C = (i_indices * c + j_indices).ravel()
    L = (left_indices * c + j_indices).ravel()
    R = (right_indices * c + j_indices).ravel()
    U = (i_indices * c + up_indices).ravel()
    D = (i_indices * c + down_indices).ravel()

    v_beta = beta * np.ones_like(C)
    # B is the [0  1  0]
    #          [1 -4  1]
    #          [0  1  0] stencil matrix
    B = sparse.coo_matrix((v_beta, (C, L)), (N, N)).tocsr() + \
        sparse.coo_matrix((v_beta, (C, R)), (N, N)).tocsr() + \
        sparse.coo_matrix((v_beta, (C, U)), (N, N)).tocsr() + \
        sparse.coo_matrix((v_beta, (C, D)), (N, N)).tocsr() + \
        sparse.coo_matrix((-4 * v_beta, (C, C)), (N, N)).tocsr()

    # bake in diffusitivity
    B = sparse.diags([diffusitivity.ravel()], [0]) / 2 * B
# 
#     print sparse.diags([diffusitivity.ravel()], [0]).tocsc()[5 * c + 5]
#     print "X"
#     print B[5 * c + 5, :]
#     return (im.ravel() + B * im.ravel()).reshape(im.shape)
# 
    # solve
    def foo(x):
        print ".",
        sys.stdout.flush()

    val, status  = linalg.cg(sparse.eye(N, N) - B,
                             (sparse.eye(N, N) + B) * im.ravel(),
                             im.ravel(),
                             callback=foo)
    assert status == 0
    return val.reshape(im.shape)
