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

    def add_rigidity_delta(self, ipos, jpos):
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

    def add_rigidity(self, ipos, jpos):
        # i coordinates are first, then j coordinates
        numpts = ipos.size
        j_offset = numpts
        self.i_indices = np.arange(numpts, dtype=np.int32).reshape(ipos.shape)
        self.j_indices = j_offset + self.i_indices

        # rigidity terms
        t0 = self.i_indices
        r = self.base_row_idx + np.arange(t0.size)
        self.rows.append(r)
        self.cols.append(t0)
        self.lvals.append(np.ones(t0.size))
        self.rvals.append(ipos.ravel())
        self.base_row_idx += t0.size
        t0 = self.j_indices
        r = self.base_row_idx + np.arange(t0.size)
        self.rows.append(r)
        self.cols.append(t0)
        self.lvals.append(np.ones(t0.size))
        self.rvals.append(jpos.ravel())
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
            print "neigh", t0[0], rv[0], weight

    def solve(self, prev_positions=None, tol=1e-5, damp=False):
        Csize = 2 * self.i_indices.size

        if damp:
            print "DAMP"
            # add damping
            rows = self.rows + [self.base_row_idx + np.arange(Csize)]
            cols = self.cols + [np.arange(Csize)]
            lvals = self.lvals + [np.ones(Csize) * damp]
            rvals = self.rvals + [np.zeros(Csize)]
        else:
            rows = self.rows
            cols = self.cols
            lvals = self.lvals
            rvals = self.rvals

        # build matrix
        rows = np.hstack([r.ravel() for r in rows])
        cols = np.hstack([c.ravel() for c in cols])
        lvals = np.hstack([l.ravel() for l in lvals])
        rvals = np.hstack([r.ravel() for r in rvals])
        x0 = np.hstack((v.ravel() for v in prev_positions))

        Rsize = rows.max() + 1
        M = sparse.coo_matrix((lvals, (rows, cols)),
                              (Rsize, Csize)).tocsr()
        t0 = time.time()
        r0 = rvals - M * x0
        V = linalg.lsqr(M, r0, show=False, damp=0)
        x0 = x0 + V[0]
        for i in range(30):
            r0 = rvals - M * x0
            print r0[:10], type(r0)
            w = sparse.diags([np.abs(r0)**0.5], [0])
            V = linalg.lsqr(w * M, w * r0, show=False, damp=0)
            print i, np.max(np.abs(r0)), V[2]
            x0 = x0 + V[0]
        new_pos = x0
        # V = cg.cg(M.T * M, M.T * rvals.reshape((-1, 1)), x0=x0, tol=tol)
        # print "   solved in", time.time() - t0, "NNZ", M.nnz, (M.T * M).nnz
        new_ivals = new_pos[:self.i_indices.size].reshape(self.i_indices.shape)
        new_jvals = new_pos[self.i_indices.size:].reshape(self.i_indices.shape)
        return new_ivals, new_jvals, V[2]
