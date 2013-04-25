import sys
import os
from collections import defaultdict

from multiprocessing.pool import ThreadPool
import threading

import cv2
import h5py
import numpy as np
import pylab

from nonlinear_warper import NonlinearWarper
import invdisttree

link_weight = 1.0

class Warpinfo(object):
    def __init__(self, warpfile, falloff=0.5, positions_dir="POSITIONS"):
        self.falloff = falloff
        self.positions_dir = positions_dir
        self.parse_warpfile(warpfile)

        # build independent sets
        self.independent_sets = defaultdict(set)
        for src in range(self.num_images):
            # add src to the first set that doesn't have any of its destinations
            for s in range(self.num_images):
                if all(d not in self.independent_sets[s] for d in self.warp_dests[src]):
                    self.independent_sets[s].add(src)
                    break

        # create the positions file
        self.positions_hdf5 = h5py.File(os.path.join(self.positions_dir, "POSITIONS.hdf5"))

        self.hdf5_lock = threading.Lock()

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
                                               chunks=(1, 32, 32), dtype=w.dtype, compression='gzip')
        cols = self.warps_hdf5.require_dataset('column_map', tuple([numwarps] + list(w.shape)),
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

    def init_positions(self, src):
        # get a warp from this src
        dest = self.warp_dests[src][0]
        shape = self.row_warp(src, dest).shape
        ipos, jpos = np.mgrid[:shape[0], :shape[1]]
        ipos -= ipos.mean()
        jpos -= jpos.mean()
        self.set_positions(src, ipos, jpos)

    def init_positions_from_existing(self, oldi, oldj, idx):
        dest = self.warp_dests[idx][0]
        shape = self.row_warp(idx, dest).shape
        # scale to new size
        self.set_positions(idx,
                           cv2.resize((oldi / (oldi.shape[0] - 1)) * (shape[0] - 1), shape[::-1]),
                           cv2.resize((oldj / (oldj.shape[0] - 1)) * (shape[1] - 1), shape[::-1]))

    def align_rigid(self, idx, oneway=False):
        dests = self.warp_dests[idx]
        shape = self.row_warp(idx, dests[0]).shape
        ipos, jpos = np.mgrid[:shape[0], :shape[1]]
        X = np.row_stack((ipos.ravel(), jpos.ravel()))
        Xmean = X.mean(axis=1).reshape((2, 1))
        X -= Xmean
        XWY = 0
        mY = 0
        totalW = 0
        for dest in dests:
            if oneway and (dest > idx):
                continue
            d_ipos, d_jpos = self.get_warped_positions(idx, dest)
            mask = ~ np.isnan(d_ipos)
            Y = np.row_stack((d_ipos[mask], d_jpos[mask]))
            Ymean = Y.mean(axis=1).reshape((2, 1))
            Y -= Ymean
            frac_kept = mask.sum() / float(mask.size)
            w = frac_kept * (self.falloff ** (np.abs(idx - dest) - 1))
            Xmasked = np.row_stack((ipos[mask], jpos[mask]))
            XWY += w * np.dot(Xmasked, Y.T)
            mY += w * Ymean
            totalW += w
        XWY /= totalW
        mY /= totalW
        u, s, vt = np.linalg.svd(XWY)
        R = np.dot(vt.T, u.T)
        T = mY - np.dot(R, Xmean)
        newX = np.dot(R, np.row_stack((ipos.ravel(), jpos.ravel()))) + T
        oldi, oldj = self.get_positions(idx)
        newi = newX[0, :].reshape(ipos.shape)
        newj = newX[1, :].reshape(ipos.shape)
        change = max(abs(newi - oldi).max(), abs(newj - oldj).max())
        self.set_positions(idx, newi, newj)
        return change

    def align_rigid_single(self, src, dest):
        dests = self.warp_dests[src]
        shape = self.row_warp(src, dests[0]).shape
        ipos, jpos = np.mgrid[:shape[0], :shape[1]]
        X = np.row_stack((ipos.ravel(), jpos.ravel()))
        Xmean = X.mean(axis=1).reshape((2, 1))
        X -= Xmean
        d_ipos, d_jpos = self.get_warped_positions(src, dest)
        mask = ~ np.isnan(d_ipos)
        Y = np.row_stack((d_ipos[mask], d_jpos[mask]))
        Ymean = Y.mean(axis=1).reshape((2, 1))
        Y -= Ymean
        Xmasked = np.row_stack((ipos[mask], jpos[mask])) - Xmean
        XY = np.dot(Xmasked, Y.T)
        u, s, vt = np.linalg.svd(XY)
        R = np.dot(vt.T, u.T)
        T = Ymean - np.dot(R, Xmean)
        newX = np.dot(R, np.row_stack((ipos.ravel(), jpos.ravel()))) + T
        oldi, oldj = self.get_positions(src)
        newi = newX[0, :].reshape(ipos.shape)
        newj = newX[1, :].reshape(ipos.shape)
        change = max(abs(newi - oldi).max(), abs(newj - oldj).max())
        self.set_positions(src, newi, newj)
        return change

    def align_nonlinear(self, src):
        # grab current positions
        old_i, old_j = self.get_positions(src)
        # do a rigid alignment to neighbors
        self.align_rigid(src)
        # Set up least-squares problem
        nl = NonlinearWarper()
        nl.add_rigidity(*self.get_positions(src))
        for dest in self.warp_dests[src]:
            w = link_weight * (self.falloff ** abs(src - dest))
            nl.add_neighbor(*self.get_warped_positions(src, dest),
                             weight=w)
        new_i, new_j = nl.solve()
        change = max(abs(new_i - old_i).max(), abs(new_j - old_j).max())
        with self.hdf5_lock:
            print "    NONLINEAR", src, change
        self.set_positions(src, new_i, new_j)
        return change

    def get_warped_positions(self, src, dest):
        row_warp = self.row_warp(src, dest)
        shape = row_warp.shape
        col_warp = self.column_warp(src, dest)
        row_warp = row_warp * shape[0]
        col_warp = col_warp * shape[1]
        pos_i, pos_j = self.get_positions(dest)
        dest_i = (np.nan * pos_i).astype(np.float32)
        dest_j = dest_i.copy()
        # interpolate
        # TODO: convert to deltas and use BORDER_REPLICATE?
        cv2.remap(pos_i,
                  col_warp.astype(np.float32),
                  row_warp.astype(np.float32),
                  cv2.INTER_LINEAR,
                  dest_i,
                  cv2.BORDER_TRANSPARENT)
        cv2.remap(pos_j,
                  col_warp.astype(np.float32),
                  row_warp.astype(np.float32),
                  cv2.INTER_LINEAR,
                  dest_j,
                  cv2.BORDER_TRANSPARENT)
        return dest_i, dest_j


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
    if not hasattr(threading.current_thread(), "_children"):
        threading.current_thread()._children = weakref.WeakKeyDictionary()

    warpinfo = Warpinfo(sys.argv[1], positions_dir=sys.argv[2])

    if len(sys.argv) == 3:
        print "INIT"
        for idx in range(warpinfo.num_images):
            print "   ", idx, '/', warpinfo.num_images
            warpinfo.init_positions(idx)
    else:
        print "INITEXIST"
        oldf = h5py.File(sys.argv[3], 'r')
        for idx in range(warpinfo.num_images):
            print "   ", idx, '/', warpinfo.num_images
            warpinfo.init_positions_from_existing(oldf['ipos'][idx, ...],
                                                  oldf['jpos'][idx, ...],
                                                  idx)
        oldf.close()

    print "INDEPENDENT SETS", warpinfo.independent_sets.keys()

    if False:
        print "RIGID"
        # Quick alignment from N+1 to N
        for idx in range(1, warpinfo.num_images):
            warpinfo.align_rigid_single(idx, idx - 1)

        print "ONEWAY"
        # looping Rigid alignment with all neighbors, but holding 0 still
        change = 0
        for idx in range(1, warpinfo.num_images):
            change = max(warpinfo.align_rigid(idx, oneway=True), change)
        print "ONEWAY prewarp, max delta:", change

        print "LOOP RIGID"
        # looping Rigid alignment with all neighbors, but holding 0 still
        change = 1
        while change > 0.25:
            change = 0
            for idx in range(1, warpinfo.num_images):
                change = max(warpinfo.align_rigid(idx), change)
            print "RIGID prewarp, max delta:", change

    pool = ThreadPool(1)
    # looping nonlinear warping
    change = 10
    while change > .5:
        change = 0
        for s, idxs in warpinfo.independent_sets.iteritems():
            changes = pool.map_async(warpinfo.align_nonlinear, idxs).get()
            change = max(change, max(changes))
        print "Nonlinear adjustment, max delta:", change

    sys.exit(0)
    warpinfo.create_global_warps()
if True:
    for idx in range(warpinfo.num_images):
        color = 'bgrmcykbgrmcyk'[idx]
        pi, pj = warpinfo.get_positions(idx)
        for c in range(0, pj.shape[1], 10) + [-1]:
            xpos = pj[:, c]
            ypos = pi[:, c]
            pylab.plot(xpos, ypos, '-' + color)
        for r in range(0, pj.shape[0], 10) + [-1]:
            xpos = pj[r, :]
            ypos = pi[r, :]
            pylab.plot(xpos, ypos, '-' + color)

pylab.show()
