import sys
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import time
import cg

class NonlinearWarper(object):
    def __init__(self):
        self.base_row_idx = 0
        self.rows = []
        self.cols = []
        self.lvals = []
        self.rvals = []

    def add_rigidity(self, ipos, jpos):
        # For now, we assume the positions are on a regular grid, after a rigid
        # alignment to neighbors, so we just use deltas from the corner.
        di_di = ipos[1, 0] - ipos[0, 0]
        di_dj = ipos[0, 1] - ipos[0, 0]
        dj_di = jpos[1, 0] - jpos[0, 0]
        dj_dj = jpos[0, 1] - jpos[0, 0]

        # i coordinates are first, then j coordinates
        numpts = ipos.size
        j_offset = numpts
        self.i_indices = np.arange(numpts, dtype=np.int32).reshape(ipos.shape)
        self.j_indices = j_offset + self.i_indices

        # rigidity terms
        for t0, t1, rval in \
                [[self.i_indices[:-1, :], self.i_indices[1:, :], di_di],
                 [self.j_indices[:-1, :], self.j_indices[1:, :], dj_di],
                 [self.i_indices[:, :-1], self.i_indices[:, 1:], di_dj],
                 [self.j_indices[:, :-1], self.j_indices[:, 1:], dj_dj]]:
            r = self.base_row_idx + np.arange(t0.size)
            self.rows.append(r)
            self.cols.append(t0)
            self.lvals.append(-np.ones(t0.size))
            self.rows.append(r)
            self.cols.append(t1)
            self.lvals.append(np.ones(t0.size))
            self.rvals.append(rval * np.ones(t0.size))
            self.base_row_idx += t0.size

    def add_neighbor(self, d_ipos, d_jpos, weight=0.5):
        mask = ~ np.isnan(d_ipos)
        for s, d in zip([self.i_indices, self.j_indices],
                        [d_ipos, d_jpos]):
            t0 = s[mask]
            rv = d[mask]
            r = self.base_row_idx + np.arange(t0.size)
            self.rows.append(r)
            self.cols.append(t0)
            self.lvals.append(weight * np.ones(t0.size))
            self.rvals.append(weight * rv)
            self.base_row_idx += t0.size

    def solve(self, prev_positions=None):
        Csize = 2 * self.i_indices.size
        # add damping
        self.rows.append(self.base_row_idx + np.arange(Csize))
        self.cols.append(np.arange(Csize))
        self.lvals.append(np.ones(Csize) * 0.001)
        self.rvals.append(np.zeros(Csize))

        x0 = np.hstack((v.ravel() for v in prev_positions))

        # build matrix
        rows = np.hstack([r.ravel() for r in self.rows])
        cols = np.hstack([c.ravel() for c in self.cols])
        lvals = np.hstack([l.ravel() for l in self.lvals])
        rvals = np.hstack([r.ravel() for r in self.rvals])
        Rsize = rows.max() + 1
        M = sparse.coo_matrix((lvals, (rows, cols)),
                              (Rsize, Csize)).tocsr()
        t0 = time.time()
        # V = linalg.lsqr(M.T * M, M.T * rvals.reshape((-1, 1)), show=False, damp=0.001)
        V = cg.cg(M.T * M, M.T * rvals.reshape((-1, 1)), x0=None)
        # print "   solved in", time.time() - t0, "NNZ", M.nnz, (M.T * M).nnz
        new_ivals = V[0][:self.i_indices.size].reshape(self.i_indices.shape)
        new_jvals = V[0][self.i_indices.size:].reshape(self.i_indices.shape)
        return new_ivals, new_jvals
