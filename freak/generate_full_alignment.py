import sys
import os
import subprocess
from multiprocessing.pool import ThreadPool
import threading

def doit(*args):
    print "START", args
    subprocess.check_call(*args)
    print "DONE", args

class Warp(object):
    image_list = None
    pool = ThreadPool()

    def __init__(self, im1_idx, im2_idx, output_path, command, args):
        self.im1_idx = im1_idx
        self.im2_idx = im2_idx
        self.output_path = output_path
        if not os.path.exists(os.path.dirname(output_path)):
            os.mkdir(os.path.dirname(output_path))
        args = [a.get() if isinstance(a, Warp) else a for a in args]
        self.promise = Warp.pool.apply_async(doit, [[command] + args])

    @classmethod
    def img(cls, idx):
        return cls.image_list[idx]

    def get(self):
        self.promise.get()
        return self.output_path

def find_rough_warp(idx1, idx2):
    out_path = os.path.join("ROUGH", "rough.%d.to.%d.hdf5" % (idx1, idx2))
    return Warp(idx1, idx2, out_path,
                "compute_mapping",
                [Warp.img(idx1), Warp.img(idx2), out_path])

def refine_warp(warp_in, octaves):
    idx1 = warp_in.im1_idx
    idx2 = warp_in.im2_idx
    out_path = os.path.join("REFINED_%d" % octaves,
                            "refined.%d.to.%d.oct.%d.hdf5" % (idx1, idx2, octaves))

    return Warp(idx1, idx2, out_path,
                "refine_mapping",
                [Warp.img(idx1), Warp.img(idx2),
                 str(octaves),
                 warp_in.get(), out_path])

def chain_warp(warp1, warp2):
    idx1 = warp1.im1_idx
    idx2 = warp2.im2_idx
    out_path = os.path.join("CHAINED",
                            "chained.%d.to.%d.hdf5" % (idx1, idx2))
    return Warp(idx1, idx2, out_path,
                "chain_warps",
                [warp1.get(), warp2.get(), out_path])

if __name__ == '__main__':
    if not hasattr(threading.current_thread(), "_children"):
        threading.current_thread()._children = weakref.WeakKeyDictionary()


    images = [l.strip() for l in open(sys.argv[1]).readlines()]
    Warp.image_list = images
    imgindices = range(len(images))
    forward_rough = [find_rough_warp(i1, i1 + 1) for i1 in imgindices[:-1]]
    backward_rough = [find_rough_warp(i1, i1 - 1) for i1 in imgindices[1:]]
    forward_refined_3 = [refine_warp(w, 3) for w in forward_rough]
    backward_refined_3 = [refine_warp(w, 3) for w in backward_rough]
    forward_chained_3 = [chain_warp(w1, w2) for w1, w2 in zip(forward_refined_3[:-1], forward_refined_3[1:])]
    backward_chained_3 = [chain_warp(w1, w2) for w1, w2 in zip(backward_refined_3[1:], backward_refined_3[:-1])]
    all_3 = forward_refined_3 + backward_refined_3 + forward_chained_3 + backward_chained_3
    refined_2 = [refine_warp(w, 2) for w in all_3]
    refined_1 = [refine_warp(w, 1) for w in refined_2]
    refined_0 = [refine_warp(w, 0) for w in refined_1]
    for w in refined_0:
        w.get()
