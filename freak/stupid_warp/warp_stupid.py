import sys
import os
from collections import defaultdict

import cv2
import h5py
import numpy as np
import pylab
import threading
import remap

class Warpinfo(object):
    def __init__(self, warpfile, falloff=0.5, positions_dir="POSITIONS"):
        self.falloff = falloff
        self.positions_dir = positions_dir
        self.parse_warpfile(warpfile)
        self.hdf5_lock = threading.RLock()
        self.corrections = {(0, 0) : (np.matrix([[1,0],[0,1]]), np.array([[0], [0]]))}

    def parse_warpfile(self, fname):
        f = open(fname)
        self.warps = {}
        self.warp_dests = defaultdict(list)
        images = set()
        for l in f.readlines():
            src, dest, wf = l.strip().split(' ')
            src = int(src)
            dest = int(dest)
            self.warps[src, dest] = wf
            self.warp_dests[src].append(dest)
            images.add(src)
            images.add(dest)
        self.images = list(set(images))
        self.num_images = len(self.images)
        self.create_warps_hdf5()

    def create_warps_hdf5(self):
        numwarps = len(self.warps)
        print "CREATING WARP FILE"

        # initialize the full warp file
        self.warps_hdf5 = h5py.File(os.path.join(self.positions_dir, "WARPS.hdf5"))
        f = h5py.File(self.warps[0, self.warp_dests[0][0]], 'r')
        w = f['row_map']
        rows = self.warps_hdf5.require_dataset('row_map', tuple([numwarps] + list(w.shape)),
                                               maxshape = tuple([None] + list(w.shape)),
                                               chunks=(1, 32, 32), dtype=w.dtype, compression='gzip')
        cols = self.warps_hdf5.require_dataset('column_map', tuple([numwarps] + list(w.shape)),
                                               maxshape = tuple([None] + list(w.shape)),
                                               chunks=(1, 32, 32), dtype=w.dtype, compression='gzip')
        f.close()

        # load each warp
        self.warp_idx = {}
        for (s, d), wf in self.warps.iteritems():
            print len(self.warp_idx), '/', len(self.warps)
            idx = self.warp_idx[s, d] = len(self.warp_idx)
            f = h5py.File(wf, 'r')
            rows[idx, ...] = f['row_map'][...]
            cols[idx, ...] = f['column_map'][...]
            f.close()

    def chain_warps(self, src, intermediate, dest):
        if (src, dest) in self.warp_idx:
            return
        with self.hdf5_lock:
            sh = list(self.warps_hdf5['row_map'].shape)
            idx = sh[0]
            sh[0] += 1
            print "resize"
            self.warps_hdf5['row_map'].resize(sh)
            self.warps_hdf5['column_map'].resize(sh)
            print "done"

            r1 = row_warp = self.row_warp(src, intermediate)
            c1 = col_warp = self.column_warp(src, intermediate)
            shape = row_warp.shape
            row_warp = row_warp * (shape[0] - 1)
            col_warp = col_warp * (shape[1] - 1)

            orig_i = self.row_warp(intermediate, dest)
            orig_j = self.column_warp(intermediate, dest)

            # convert to deltas
            base_i, base_j = np.mgrid[:orig_i.shape[0], :orig_i.shape[1]]
            base_i = base_i.astype(np.float32) / (base_i.shape[0] - 1)
            base_j = base_j.astype(np.float32) / (base_j.shape[1] - 1)

            orig_i -= base_i
            orig_j -= base_j

            dest_i = np.zeros_like(orig_i)
            dest_j = np.zeros_like(orig_j)
            # interpolate

            # We shouldn't really use repeat=True, here.  Instead, we should find the rigid transformation between the warps and 

            remap.remap(orig_i,
                        col_warp.astype(np.float32),
                        row_warp.astype(np.float32),
                        dest_i,
                        repeat=True)
            remap.remap(orig_j,
                        col_warp.astype(np.float32),
                        row_warp.astype(np.float32),
                        dest_j,
                        repeat=True)

            dest_i += r1
            dest_j += c1

            self.warps_hdf5['row_map'][idx, :, :] = dest_i
            self.warps_hdf5['column_map'][idx, :, :] = dest_j
            self.warp_idx[src, dest] = idx
            self.set_chained_warp(src, dest, dest_i, dest_j)

    def compute_correction(self, lo, hi):
        fw_i = self.row_warp(lo, hi)
        fw_j = self.column_warp(lo, hi)

        base_i, base_j = np.mgrid[:fw_i.shape[0], :fw_i.shape[1]]
        base_i = base_i.astype(np.float32) / (base_i.shape[0] - 1)
        base_j = base_j.astype(np.float32) / (base_j.shape[1] - 1)

        X = np.row_stack((base_i.ravel(), base_j.ravel()))
        Y = np.row_stack((fw_i.ravel(), fw_j.ravel()))
        Xmean = X.mean(axis=1).reshape((2, 1))
        Ymean = Y.mean(axis=1).reshape((2, 1))
        X -= Xmean
        Y -= Ymean
        XY = np.dot(X, Y.T)
        u, s, vt = np.linalg.svd(XY)
        R = np.dot(vt.T, u.T)
        T = Ymean - np.dot(R, Xmean)
        if lo > 0:
            prevR, prevT = self.corrections[0, lo]
            self.corrections[0, hi] = (np.dot(R, prevR), prevT + T)
        self.corrections[lo, hi] = (R, T)

    def average_warps(self, lo, hi, dst):
        rlo = self.row_warp(lo, dst)
        clo = self.column_warp(lo, dst)
        if dst == hi:
            rhi, chi = np.mgrid[:rlo.shape[0], :rlo.shape[1]]
            rhi = rhi.astype(np.float32) / (rhi.shape[0] - 1)
            chi = chi.astype(np.float32) / (chi.shape[1] - 1)
        else:
            rhi = self.row_warp(hi, dst)
            chi = self.column_warp(hi, dst)

        wlo = float(hi - dst) / (hi - lo)
        whi = 1.0 - wlo
        rav = wlo * rlo + whi * rhi
        cav = wlo * clo + whi * chi

        R, T = self.corrections[lo, hi]
        baseR, baseT = self.corrections[0, lo]
        print "T", dst, T, whi * T
        T =  whi * T + baseT
        angle =  whi * np.arccos(R[0,0])
        R = np.matrix([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
        R = np.dot(R, baseR)
        stack = np.row_stack((rav.ravel(), cav.ravel()))
        stack = R * stack + T
        rav = stack[0, :].reshape(rav.shape)
        cav = stack[1, :].reshape(rav.shape)

        self.set_average_warp(dst, rav, cav)

    def row_warp(self, src, dest):
        idx = self.warp_idx[src, dest]
        with self.hdf5_lock:
            return self.warps_hdf5['row_map'][idx, ...]

    def column_warp(self, src, dest):
        idx = self.warp_idx[src, dest]
        with self.hdf5_lock:
            return self.warps_hdf5['column_map'][idx, ...]

    def set_chained_warp(self, src, dest, newipos, newjpos):
        fn = os.path.join(self.positions_dir,
                          "ch.%d.%d.hdf5" % (src, dest))
        f = h5py.File(fn)
        if 'ipos' not in f.keys():
            f.create_dataset('row_map', newipos.shape, dtype=np.float32)
            f.create_dataset('column_map', newipos.shape, dtype=np.float32)
        f['row_map'][...] = newipos
        f['column_map'][...] = newjpos
        f.close()

    def set_average_warp(self, dest, rowwarp, colwarp):
        fn = os.path.join(self.positions_dir,
                          "av.%d.hdf5" % (dest))
        f = h5py.File(fn)
        if 'ipos' not in f.keys():
            f.create_dataset('row_map', rowwarp.shape, dtype=np.float32)
            f.create_dataset('column_map', rowwarp.shape, dtype=np.float32)
        f['row_map'][...] = rowwarp
        f['column_map'][...] = colwarp
        f.close()

if __name__ == '__main__':
    warpinfo = Warpinfo(sys.argv[1], positions_dir=sys.argv[2])

    STEP = 30

    for firstim in range(0, warpinfo.num_images, 30):
        lastim = min(firstim + STEP, warpinfo.num_images - 1)
        print "WARPS BETWEEN", firstim, lastim

        # We compute the warps for both endpoint images, in order to compute
        # the rigid translation between them for later correction.
        for idx in range(firstim + 1, lastim + 1):
            if (idx - firstim) % 2 == 0:
                warpinfo.chain_warps(firstim, idx - 2, idx)
            else:
                warpinfo.chain_warps(firstim, idx - 1, idx)
            print "  FORWARD", idx, "to", firstim

        for idx in range(lastim - 1, firstim - 1, -1):
            if (idx - lastim) % 2 == 0:
                warpinfo.chain_warps(lastim, idx + 2, idx)
            else:
                warpinfo.chain_warps(lastim, idx + 1, idx)
            print "  BACKWARD", idx, "to", lastim

        warpinfo.compute_correction(firstim, lastim)

        for idx in range(firstim + 1, lastim + 1):
            warpinfo.average_warps(firstim, lastim, idx)
            print "  AVERAGE", idx
