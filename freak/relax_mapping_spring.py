import sys
import re
import numpy as np
import h5py
import pylab
import mahotas

import warp
import sparse_6d
from scipy.ndimage import gaussian_filter

from spring import newton_step

step = 64
k_structural = .1
k_shear = .1
k_flex = .1
k_link = 5.0

sqrt_2 = np.sqrt(2.0)

def hypot(a, b):
    return np.sqrt(a**2 + b**2)

if __name__ == '__main__':
    warpfile = open(sys.argv[1])
    lines = warpfile.readlines()[:3]
    r = re.compile('^.*: (.*) to (.*) : (.*)$')
    matches = [r.match(l) for l in lines]
    first_images = [m.group(1) for m in matches]
    second_images = [m.group(2) for m in matches]
    warpfiles = [m.group(3) for m in matches]
    shape = mahotas.imread(first_images[0]).shape
    num_images = len(warpfiles) + 1

    r = (shape[0] / step) + 1
    c = (shape[1] / step) + 1
    i_indices, j_indices = np.mgrid[0:r, 0:c]
    nodes_per_image = i_indices.size
    i_indices = i_indices.ravel()
    j_indices = j_indices.ravel()
    _ones = np.ones_like(i_indices)
    # structural springs
    structural_i = np.vstack((i_indices, j_indices,
                              i_indices + 1, j_indices,
                              k_structural * _ones,
                              _ones))
    structural_j =  np.vstack((i_indices, j_indices,
                               i_indices, j_indices + 1,
                               k_structural * _ones,
                               _ones))
    # shear
    shear_d1 = np.vstack((i_indices, j_indices,
                          i_indices + 1, j_indices + 1,
                          k_shear * _ones,
                          sqrt_2 * _ones))
    shear_d2 = np.vstack((i_indices + 1, j_indices,
                          i_indices, j_indices + 1,
                          k_shear * _ones,
                          sqrt_2 *_ones))
    # flex
    flex_i = np.vstack((i_indices, j_indices,
                        i_indices + 2, j_indices,
                        k_flex * _ones,
                        2 * _ones))
    flex_j =  np.vstack((i_indices, j_indices,
                         i_indices, j_indices + 2,
                         k_flex * _ones,
                         2 * _ones))

    all_springs = np.hstack((structural_i, structural_j,
                             shear_d1, shear_d2,
                             flex_i, flex_j))
    bad_indices = (all_springs[0, :] >= r) | (all_springs[1, :] >= c) | \
        (all_springs[2, :] >= r) | (all_springs[3, :] >= c)
    all_springs = all_springs[:, ~bad_indices] 
    def idx(*args):
        if len(args) == 2:
            i, j = args
            l = 0
        else:
            l, i, j = args
        assert np.all(i * c + j < i_indices.size)
        assert l < num_images
        return l * r * c + i * c + j
    basic_springs = np.vstack((idx(all_springs[0, :], all_springs[1, :]),
                             idx(all_springs[2, :], all_springs[3, :]),
                             all_springs[4, :],
                             all_springs[5, :])).astype(np.float32)
    print basic_springs.shape[1], "springs per layer"
    num_pts_per_layer = r * c
    layer_idx_offset = np.array([[num_pts_per_layer],
                                 [num_pts_per_layer],
                                 [0],
                                 [0]])

    # Build a layer for each image to be aligned
    layer_springs = np.hstack([basic_springs + imnum * layer_idx_offset for imnum in range(num_images)])
    i_positions = np.tile(i_indices, (1, num_images)).ravel()
    j_positions = np.tile(j_indices, (1, num_images)).ravel()

    print layer_springs.shape[1], "basic springs"
    print i_positions.size, "points"

    new_pt_idx = num_images * r * c
    print "nsi", new_pt_idx
    # Go through each warp and add inter-layer springs
    to_add = []
    new_i_positions = []
    new_j_positions = []
    for layeridx, w in enumerate(warpfiles):
        h = h5py.File(w, 'r')
        correspondences = h['match_points'][...]
        coords = correspondences / step
        baseidx = coords.astype(int)
        deltas = coords - baseidx
        new_pt_indices = new_pt_idx + np.arange(correspondences.shape[0])
        new_i_positions.append((coords[:, 0] + coords[:, 2]).ravel() / 2.0)
        new_j_positions.append((coords[:, 1] + coords[:, 3]).ravel() / 2.0)

        print "adding", new_i_positions[-1].size
        print "should be", new_pt_indices.max()
        for dl in range(2):
            for di in range(2):
                for dj in range(2):
                    rest_lengths = np.sqrt((deltas[:, dl * 2] - di) ** 2 +
                                           (deltas[:, dl * 2 + 1] - dj) ** 2)
                    to_add.append(np.vstack((idx(layeridx + dl, di + baseidx[:, dl * 2].T, dj + baseidx[:, dl * 2 + 1].T),
                                             new_pt_indices,
                                             k_link * np.ones_like(new_pt_indices),
                                             rest_lengths.T)))
        new_pt_idx += new_pt_indices.size

    all_springs = np.hstack([layer_springs] + to_add)
    print all_springs.shape[1], "total springs"
    i_positions = np.hstack([i_positions] + new_i_positions)
    j_positions = np.hstack([j_positions] + new_j_positions)

    while True:
        i_positions, j_positions, F_max = newton_step(all_springs, i_positions, j_positions)
        if F_max < 0.001:
            break

    # write out new positions
    outwarpfile = sys.argv[2]
    of = h5py.File(outwarpfile, 'w')
    out_warp = of.create_dataset('relaxed', (num_images * r * c, 5))
    rc = r * c
    for imnum in range(num_images):
        out_warp[imnum * rc:(imnum+1) * rc, 0] = imnum
        out_warp[imnum * rc:(imnum+1) * rc, 1] = i_indices * step
        out_warp[imnum * rc:(imnum+1) * rc, 2] = j_indices * step
    out_warp[:, 3] = i_positions[:num_images * rc] * step
    out_warp[:, 4] = j_positions[:num_images * rc] * step
    of.close()

def test_Jacobian(i_indices, j_indices, all_springs):
    i_positions = j_indices / 2.0
    j_positions = i_indices / 2.0

    (Fi, Fj), (Jii, Jij, Jji, Jjj) = newton_step(all_springs, i_positions, j_positions)
    h = 1e-6

    for iidx in range(r - 6, r):
        for jidx in range(6):
            test_idx = idx(iidx, jidx)

            print "perturb i", test_idx
            orig = i_positions[test_idx]
            i_positions[test_idx] += h
            (Fi_h, Fj_h),  _= newton_step(all_springs, i_positions, j_positions)
            print "check i",
            v = abs((Fi_h - Fi).ravel() / h - Jii[:, test_idx].todense().A.ravel()).max()
            assert v < 1e-5
            print v

            print "check j",
            v = abs((Fj_h - Fj).ravel() / h - Jji[:, test_idx].todense().A.ravel()).max()
            assert v < 1e-5
            print v
            i_positions[test_idx] = orig

            print ""
            print "perturb j", test_idx
            orig = j_positions[test_idx]
            j_positions[test_idx] += h
            (Fi_h, Fj_h),  _= newton_step(all_springs, i_positions, j_positions)
            print "check i",
            v= abs((Fi_h - Fi).ravel() / h - Jij[:, test_idx].todense().A.ravel()).max()
            assert v < 1e-5
            print v
            print "check j",
            v= abs((Fj_h - Fj).ravel() / h - Jjj[:, test_idx].todense().A.ravel()).max()
            assert v < 1e-5
            print v
            j_positions[test_idx] = orig
            print ""
            print ""

            print "perturb i", test_idx
            orig = i_positions[test_idx]
            i_positions[test_idx] -= h
            (Fi_h, Fj_h),  _= newton_step(all_springs, i_positions, j_positions)
            print "check i",
            v = abs((Fi_h - Fi).ravel() / (-h) - Jii[:, test_idx].todense().A.ravel()).max()
            assert v < 1e-5
            print v

            print "check j",
            v = abs((Fj_h - Fj).ravel() / (-h) - Jji[:, test_idx].todense().A.ravel()).max()
            assert v < 1e-5
            print v
            i_positions[test_idx] = orig

            print ""
            print "perturb j", test_idx
            orig = j_positions[test_idx]
            j_positions[test_idx] -= h
            (Fi_h, Fj_h),  _= newton_step(all_springs, i_positions, j_positions)
            print "check i",
            v= abs((Fi_h - Fi).ravel() / (-h) - Jij[:, test_idx].todense().A.ravel()).max()
            assert v < 1e-5
            print v
            print "check j",
            v= abs((Fj_h - Fj).ravel() / (-h) - Jjj[:, test_idx].todense().A.ravel()).max()
            assert v < 1e-5
            print v
            j_positions[test_idx] = orig
            print ""
            print ""
