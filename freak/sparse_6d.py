import numpy as np
import scipy.sparse as sparse
import numpy.lib.stride_tricks as tricks

class Sparse6D(object):
    def __init__(self, n0, n1, n2):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.m0 = n1 * n2
        self.m1 = n2
        print "SZ", n1, n2
        self.sp = sparse.dok_matrix((n0 * n1 * n2, n0 * n1 * n2))

    def __setitem__(self, indices, value):
        def wrap(x, mx):
            return (mx + x) if x < 0 else x
        c1 = indices[0] * self.m0 + wrap(indices[1], self.n1) * self.m1 + wrap(indices[2], self.n2)
        c2 = indices[3] * self.m0 + wrap(indices[4], self.n1) * self.m1 + wrap(indices[5], self.n2)
        print "setting", indices, value, c1, c2
        self.sp[c1, c2] = value

    def normalize(self):
        self.sp = self.sp.tocsr()
        print "nonzeros", self.sp.nnz
        print self.sp.sum(axis=1).shape
        rownormalizers = sparse.diags(1.0 / self.sp.sum(axis=1).ravel().A, [0])
        self.sp = rownormalizers * self.sp
        print self.sp.sum(axis=1).ravel()
        self.sp = self.sp.todok();
        print "nonzeros", self.sp.nnz

class Sparse6DSpring(object):
    def __init__(self, n0, n1, n2):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.m0 = n1 * n2
        self.m1 = n2
        self.x_pos = np.zeros(n0 * n1 * n2)
        self.y_pos = np.zeros(n0 * n1 * n2)
        self.k = sparse.dok_matrix((n0 * n1 * n2, n0 * n1 * n2))
        self.rest_lengths = sparse.dok_matrix((n0 * n1 * n2, n0 * n1 * n2))
        

    def __setitem__(self, indices, value):
        # ignore out of bounds springs
        if (indices[1] < 0) or (indices[1] >= self.n1) or (indices[2]) < 0 or (indices[2] >= self.n2):
            return
        if (indices[4] < 0) or (indices[4] >= self.n1) or (indices[5]) < 0 or (indices[5] >= self.n2):
            return
        c1 = indices[0] * self.m0 + indices[1] * self.m1 + indices[2]
        c2 = indices[3] * self.m0 + indices[4] * self.m1 + indices[5]
        k, rest_length = value
        self.k[c1, c2] = k
        self.k[c2, c1] = k
        self.rest_lengths[c1, c2] = rest_length
        self.rest_lengths[c2, c1] = rest_length

    def set_x(self, imnum, i, j, x):
        self.x_pos[imnum * self.m0 + i * self.m1 + j] = x

    def set_y(self, imnum, i, j, y):
        self.y_pos[imnum * self.m0 + i * self.m1 + j] = y

    def get_x(self, imnum, i, j):
        return self.x_pos[imnum * self.m0 + i * self.m1 + j]

    def get_y(self, imnum, i, j):
        return self.y_pos[imnum * self.m0 + i * self.m1 + j]



    def step(self):
        self.k = self.k.tocoo()
        self.rest_lengths = self.rest_lengths.tocoo()
        idx_0 = self.k.row
        idx_1 = self.k.col
        k = self.k.data
        assert np.all(self.rest_lengths.row == self.k.row)
        assert np.all(self.rest_lengths.col == self.k.col)
        rl = self.rest_lengths.data
        dx = self.x_pos[idx_1] - self.x_pos[idx_0]
        dy = self.y_pos[idx_1] - self.y_pos[idx_0]
        dists = np.sqrt(dx ** 2 + dy ** 2)
        dists[dists == 0] = 1.0
        fscale = - k * (rl / dists - 1)
        fx = fscale * dx
        fy = fscale * dy
        # reorder
        fx = np.bincount(idx_0, fx, self.k.shape[0])
        fy = np.bincount(idx_0, fy, self.k.shape[0])
        maxstep = max(np.abs(fx).max(), np.abs(fy).max())
        maxstep *= 5
        if maxstep > 1:
            fx /= maxstep
            fy /= maxstep
        self.x_pos += fx * 0.05
        self.y_pos += fy * 0.05
        print "STEP", maxstep, self.x_pos[0], self.y_pos[0]
        return maxstep
