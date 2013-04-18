import sys
import h5py
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import pylab
import invdisttree

RIGIDITY_WEIGHT = .1

if __name__ == '__main__':
    warps = sys.argv[1:]
    num_warps = len(warps)

    f = h5py.File(warps[0], 'r')
    warp_shape = f['column_map'].shape
    f.close()
    base_pts = np.prod(warp_shape)
    print "WARP SHAPE", warp_shape
    print "BASE PTS", base_pts
    print "TOTAL", num_warps * base_pts

    i_indices_base = 2 * np.arange(base_pts, dtype=np.int32).reshape(warp_shape)
    j_indices_base = i_indices_base + 1
    offset_w = 2 * base_pts

    base_row_idx = 0
    col_indices = []
    row_indices = []
    lvals = []
    rvals = []


    # rigidity
    for w_idx, w in enumerate(warps):
        # R(w, i + 1, j, w) - R(w, i, j) = 1
        term0 = i_indices_base[1:, :] + offset_w * w_idx
        term1 = i_indices_base[:-1, :] + offset_w * w_idx
        cur_rows = np.arange(base_row_idx, base_row_idx + term0.size)
        col_indices.append(term0)
        row_indices.append(cur_rows)
        lvals.append(RIGIDITY_WEIGHT * np.ones(term0.size))
        col_indices.append(term1)
        row_indices.append(cur_rows)
        lvals.append(- RIGIDITY_WEIGHT * np.ones(term1.size))
        rvals.append(RIGIDITY_WEIGHT * np.ones(term0.size))
        base_row_idx += term0.size

        # C(w, i, j + 1) - C(w, i, j) = 1
        term0 = j_indices_base[:, 1:] + offset_w * w_idx
        term1 = j_indices_base[:, :-1] + offset_w * w_idx
        cur_rows = np.arange(base_row_idx, base_row_idx + term0.size)
        col_indices.append(term0)
        row_indices.append(cur_rows)
        lvals.append(RIGIDITY_WEIGHT * np.ones(term0.size))
        col_indices.append(term1)
        row_indices.append(cur_rows)
        lvals.append(- RIGIDITY_WEIGHT * np.ones(term1.size))
        rvals.append(RIGIDITY_WEIGHT * np.ones(term0.size))
        base_row_idx += term0.size

    # mapping
    for w_idx, w in enumerate(warps[:-1]):
        f = h5py.File(w, 'r')
        rowmap = f['row_map']
        colmap = f['column_map']
        desti = np.around(rowmap[...] * i_indices_base.shape[0]).astype(np.int32)
        destj = np.around(colmap[...] * i_indices_base.shape[1]).astype(np.int32)
        valid = (desti >= 0) & (desti < i_indices_base.shape[0]) & (destj >= 0) & (destj < i_indices_base.shape[1])
        desti = desti[valid]
        destj = destj[valid]

        # R(w, i, j) - R(w + 1, W(w, i, j)) = 0
        term0 = (i_indices_base + offset_w * w_idx)[valid]
        term1 = (i_indices_base + offset_w * (w_idx + 1))[desti, destj]
        assert term0.size == term1.size
        cur_rows = np.arange(base_row_idx, base_row_idx + term0.size)
        col_indices.append(term0)
        row_indices.append(cur_rows)
        lvals.append(np.ones(term0.size))
        col_indices.append(term1)
        row_indices.append(cur_rows)
        lvals.append(-np.ones(term0.size))
        rvals.append(np.zeros(term0.size))
        base_row_idx += term0.size

        # C(w, i, j) - C(w + 1, W(w, i, j)) = 0
        term0 = (j_indices_base + offset_w * w_idx)[valid]
        term1 = (j_indices_base + offset_w * (w_idx + 1))[desti, destj]
        assert term0.size == term1.size
        cur_rows = np.arange(base_row_idx, base_row_idx + term0.size)
        col_indices.append(term0)
        row_indices.append(cur_rows)
        lvals.append(np.ones(term0.size))
        col_indices.append(term1)
        row_indices.append(cur_rows)
        lvals.append(-np.ones(term0.size))
        rvals.append(np.zeros(term0.size))
        base_row_idx += term0.size

for c, r, l in zip(col_indices, row_indices, lvals):
    assert c.ravel().shape == r.ravel().shape
    assert c.ravel().shape == l.ravel().shape

assert sum(r.size for r in rvals) - 1 == max(r.max() for r in row_indices)

row_indices = np.hstack([r.ravel() for r in row_indices])
col_indices = np.hstack([c.ravel() for c in col_indices])
lvals = np.hstack([l.ravel() for l in lvals])
rvals = np.hstack([r.ravel() for r in rvals])

Rsize = row_indices.max() + 1
Csize = 2 * base_pts * num_warps
M = sparse.coo_matrix((lvals, (row_indices, col_indices)), (Rsize, Csize)).tocsr()

V = linalg.lsqr(M, rvals.reshape((-1, 1)), show=True, damp=0.001)
new_ivals = V[0][::2]
new_jvals = V[0][1::2]

mini = new_ivals.min()
minj = new_jvals.min()
new_ivals -= mini
new_jvals -= minj

maxi = int(new_ivals.max() + 1)
maxj = int(new_jvals.max() + 1)

origi, origj = np.mgrid[:warp_shape[0], :warp_shape[1]]
origcoords = np.column_stack((origi.ravel() / float(warp_shape[0]),
                              origj.ravel() / float(warp_shape[1])))

for widx, w in enumerate(warps):
    sl = slice(widx * base_pts, (widx + 1) * base_pts)
    wi = new_ivals[sl] / maxi
    wj = new_jvals[sl] / maxj

    print widx, "0, 0 mapped to ", wi[0], wj[0]

    # interpolate as deltas
    idt = invdisttree.Invdisttree(np.column_stack((wi, wj)),
                                  origcoords - np.column_stack((wi, wj)))

    qi, qj = np.mgrid[:maxi, :maxj]
    query = np.column_stack((qi.ravel() / float(maxi), qj.ravel() / float(maxj)))
    interped_ij = idt(query)

    print "    q:", idt((wi[0], wj[0]))
    print "    q:", idt((0, 0))

    interped_i = interped_ij[:, 0].reshape(qi.shape) + (qi / float(maxi))
    interped_j = interped_ij[:, 1].reshape(qi.shape) + (qj / float(maxj))
    print "     ", interped_i[0, 0], interped_j[0, 0]
    f = h5py.File(w + 'global', 'w')
    c1 = f.create_dataset('row_map', interped_i.shape, dtype=np.float)
    c1[...] = interped_i
    c2 = f.create_dataset('column_map', interped_i.shape, dtype=np.float)
    c2[...] = interped_j
    f.close()
    print w + 'global'
