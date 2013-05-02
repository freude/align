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

    def get_warped_positions(self, src, dest):
        row_warp = self.row_warp(src, dest)
        shape = row_warp.shape
        col_warp = self.column_warp(src, dest)
        row_warp = row_warp * (shape[0] - 1)
        col_warp = col_warp * (shape[1] - 1)
        pos_i, pos_j = self.get_positions(dest)
        dest_i = (np.nan * pos_i).astype(np.float32)
        dest_j = dest_i.copy()
        # interpolate
        # TODO: convert to deltas and use BORDER_REPLICATE?
        remap.remap(pos_i,
                    col_warp.astype(np.float32),
                    row_warp.astype(np.float32),
                    dest_i)
        remap.remap(pos_j,
                    col_warp.astype(np.float32),
                    row_warp.astype(np.float32),
                    dest_j)
        return dest_i, dest_j

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

            row_warp = self.row_warp(src, intermediate)
            col_warp = self.column_warp(src, intermediate)
            shape = row_warp.shape
            row_warp = row_warp * (shape[0] - 1)
            col_warp = col_warp * (shape[1] - 1)

            # convert to deltas
            orig_i = self.row_warp(intermediate, dest)
            orig_j = self.column_warp(intermediate, dest)

            dest_i = np.zeros_like(orig_i)
            dest_j = np.zeros_like(orig_j)
            # interpolate
            # TODO: convert to deltas and use BORDER_REPLICATE?
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
            self.warps_hdf5['row_map'][idx, :, :] = dest_i
            self.warps_hdf5['column_map'][idx, :, :] = dest_j
            self.warp_idx[src, dest] = idx
            self.set_chained_warp(src, dest, dest_i, dest_j)

    def average_warps(self, lo, hi, dst):
        rlo = self.row_warp(lo, dst)
        clo = self.column_warp(lo, dst)
        rhi = self.row_warp(hi, dst)
        chi = self.column_warp(hi, dst)
        wlo = float(hi - dst) / (hi - lo)
        whi = 1.0 - wlo
        self.set_average_warp(dst, wlo * rlo + whi * rhi, wlo * clo + whi * chi)

    def row_warp(self, src, dest):
        idx = self.warp_idx[src, dest]
        with self.hdf5_lock:
            return self.warps_hdf5['row_map'][idx, ...]

    def column_warp(self, src, dest):
        idx = self.warp_idx[src, dest]
        with self.hdf5_lock:
            return self.warps_hdf5['column_map'][idx, ...]

    def get_positions(self, idx):
        with self.hdf5_lock:
            return self.positions_hdf5['ipos'][idx, ...], self.positions_hdf5['jpos'][idx, ...]

    def set_positions(self, idx, newipos, newjpos):
        with self.hdf5_lock:
            self.positions_hdf5.require_dataset('ipos', tuple([self.num_images] + list(newipos.shape)), dtype=np.float32)
            self.positions_hdf5.require_dataset('jpos', tuple([self.num_images] + list(newipos.shape)), dtype=np.float32)
            self.positions_hdf5['ipos'][idx, ...] = newipos
            self.positions_hdf5['jpos'][idx, ...] = newjpos

    def set_global_warp(self, idx, newipos, newjpos):
        f = h5py.File(self.global_warp_file(idx))
        if 'ipos' not in f.keys():
            f.create_dataset('row_map', newipos.shape, dtype=np.float32)
            f.create_dataset('column_map', newipos.shape, dtype=np.float32)
        f['row_map'][...] = newipos
        f['column_map'][...] = newjpos
        f.close()

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


    def global_warp_file(self, idx):
        return os.path.join(self.positions_dir,
                            "global_warp.%d.hdf5" % idx)

    def create_global_warps(self):
        orig_shape = self.get_positions(0)[0].shape
        def get_extrema(idx):
            ipos, jpos = self.get_positions(idx)
            return ipos.min(), jpos.min(), ipos.max(), jpos.max()
        minmax = [get_extrema(idx) for idx in range(self.num_images)]
        imin = min(m[0] for m in minmax)
        jmin = min(m[1] for m in minmax)
        imax = max(m[2] for m in minmax)
        jmax = max(m[3] for m in minmax)
        # shift everything to min 0,0
        for idx in range(self.num_images):
            ipos, jpos = self.get_positions(idx)
            ipos -= imin
            jpos -= jmin
            self.set_positions(idx, ipos, jpos)
        height = int(imax - imin) + 1
        width = int(jmax - jmin) + 1

        origi, origj = np.mgrid[:orig_shape[0], :orig_shape[1]]
        origcoords = np.column_stack((origi.ravel() / float(orig_shape[0]),
                                      origj.ravel() / float(orig_shape[1])))

        for idx in range(self.num_images):
            dest_i, dest_j = self.get_positions(idx)
            dest_i = (dest_i / height).ravel()
            dest_j = (dest_j / width).ravel()

            print idx, "0, 0 mapped to ", dest_i[0], dest_j[0]

            # interpolate as deltas
            idt = invdisttree.Invdisttree(np.column_stack((dest_i, dest_j)),
                                          origcoords - np.column_stack((dest_i, dest_j)))

            qi, qj = np.mgrid[:height, :width]
            query = np.column_stack((qi.ravel() / float(height), qj.ravel() / float(width)))
            interped_ij = idt(query)

            print "    q:", idt((dest_i[0], dest_j[0]))
            print "    q:", idt((0, 0))

            interped_i = interped_ij[:, 0].reshape(qi.shape) + (qi / float(height))
            interped_j = interped_ij[:, 1].reshape(qi.shape) + (qj / float(width))
            self.set_global_warp(idx, interped_i, interped_j)

if __name__ == '__main__':
    warpinfo = Warpinfo(sys.argv[1], positions_dir=sys.argv[2])

    for idx in range(1, warpinfo.num_images):
        warpinfo.chain_warps(0, idx - 1, idx)
        print idx

    for idx in range(warpinfo.num_images - 2, 0, -1):
        warpinfo.chain_warps(warpinfo.num_images - 1, idx + 1, idx)
        print idx

    for idx in range(1, warpinfo.num_images - 1):
        warpinfo.average_warps(0, warpinfo.num_images - 1, idx)
        print idx
