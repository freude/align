import sys
import os
import threading
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import cv2

from ransac import ransac
from warp import RigidWarp, refine_warp

def scalespace(im, octaves):
    sp = {}
    for o in range(octaves + 1):
        sp[o] = im
        if o < octaves:
            im = cv2.medianBlur(im, 3)
            im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    return sp

if __name__ == '__main__':
    # default values
    # number of times to halve the image initially
    downsample_octaves = 4

    # number of keypoint matches to keep, minimum to require
    maximum_matches = 1000
    minimum_matches = 20

    # template and window sizes for normalized cross-correlation
    template_size = 96
    window_size = 160

    # step size for refined warps
    step_size = 16

    # TODO: parse args


    # setup for multithreading
    if not hasattr(threading.current_thread(), "_children"):
        threading.current_thread()._children = weakref.WeakKeyDictionary()

    pool = ThreadPool(8)

    out_forward = sys.argv[3]
    out_backward = sys.argv[4]

    if os.path.exists(out_forward) and os.path.exists(out_backward):
        sys.exit(0)

    st = time.time()
    im1 = pool.apply_async(cv2.imread, [sys.argv[1]], {'flags':cv2.CV_LOAD_IMAGE_GRAYSCALE})
    im2 = pool.apply_async(cv2.imread, [sys.argv[2]], {'flags':cv2.CV_LOAD_IMAGE_GRAYSCALE})
    im1 = im1.get()
    im2 = im2.get()
    loadtime = time.time() - st

    print "ALIGNING", sys.argv[1], sys.argv[2]

    # compute scaled versions
    st = time.time()
    im1_scales = pool.apply_async(scalespace, [im1, downsample_octaves])
    im2_scales = pool.apply_async(scalespace, [im2, downsample_octaves])
    im1_scales = im1_scales.get()
    im2_scales = im2_scales.get()
    smallest1 = im1_scales[downsample_octaves]
    smallest2 = im2_scales[downsample_octaves]
    scaletime = time.time() - st

    # detect keypoints, extract descriptors
    def detect(im, threshold=10, step=10, max_keypoints=(10 * maximum_matches)):
        while True:
            detector = cv2.BRISK(threshold, 0)
            keypoints = detector.detect(im)
            if len(keypoints) < max_keypoints:
                return keypoints
            threshold += step

    st = time.time()
    extractor = cv2.DescriptorExtractor_create('FREAK')
    kp1 = pool.apply_async(detect, [smallest1])
    kp2 = pool.apply_async(detect, [smallest2])
    # Freak doesn't seem to be threadsafe
    keypoints1, descriptors1 = extractor.compute(smallest1, kp1.get())
    keypoints2, descriptors2 = extractor.compute(smallest2, kp2.get())
    detecttime = time.time() - st

    print len(keypoints1), len(keypoints2), "keypoints"

    st = time.time()
    # match keypoints, keep only distinct matches
    matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
    def good_match(m1, m2, thresh=0.7, eps=1e-5):
        return (m1.distance + eps) / (m2.distance + eps) < thresh
    matches = [m1 for m1, m2 in matcher.knnMatch(descriptors1, descriptors2, 2)
               if good_match(m1, m2)]
    matchtime = time.time() - st
    # make sure we have enough distinct matches to proceed
    if len(matches) < minimum_matches:
        sys.stderr.write("Found too few distinct matches between {0} and {1}.\n" +
                         "Needed {2} and found {3}\n".format(sys.argv[1], sys.argv[2],
                                                             minimum_matches, len(matches)))
        sys.exit(1)

    print len(matches), "matches"
    print "Times: load: %0.2f, scale %0.2f, detect: %0.2f, match: %0.2f" % (loadtime, scaletime, detecttime, matchtime)

    # compute R, T that transform kpts in 1 to 2
    MAD, R, T, dists = ransac(keypoints1, keypoints2, matches)
    print "MAD of keypoint distsances after rigid correction:", MAD

    # filter matches with too large a separation
    thresh = 2.0 * 1.48 * min(MAD, 500 >> downsample_octaves)
    matches = [m for idx, m in enumerate(matches) if dists[idx] < thresh]
    print len(matches), "after filtering"

    # Create the initial warps
    # change from normalized coords, to image coords (for R, T), then back.
    # Keypoints are in image coords.
    def to_normalized(shape):
        return np.matrix([[1.0 / (shape[0] - 1), 0], [0, 1.0 / (shape[1] - 1)]])
    def from_normalized(shape):
        return np.matrix([[shape[0] - 1, 0], [0, shape[1] - 1]])

    # Forward warp
    pre = from_normalized(smallest1.shape)
    post = to_normalized(smallest2.shape)
    warp = RigidWarp(post * R * pre, post * T)

    # Backward warp
    pre = from_normalized(smallest2.shape)
    post = to_normalized(smallest1.shape)
    revwarp = RigidWarp(post * R.T * pre, - post * T)

    def display_warp(w, im1, im2):
        warpedim = w.warp([im2], im1.shape)[0]
        while True:
            cv2.imshow("view", warpedim.astype(im1.dtype))
            k = cv2.waitKey()
            if k == 27:
                break
            cv2.imshow("view", im1)
            k = cv2.waitKey()
            if k == 27:
                break
    # coarse-to-fine warp refinement using normalized cross correlation in subimages
    for cur_octave in range(downsample_octaves, -1, -1):
        print "FORWARD", cur_octave, 
        warp = refine_warp(warp, 
                           im1_scales[cur_octave], im2_scales[cur_octave],
                           template_size, window_size, step_size, pool)
        # display_warp(warp, im1_scales[downsample_octaves - 1], im2_scales[downsample_octaves - 1])

    for cur_octave in range(downsample_octaves, -1, -1):
        print "BACKWARD", cur_octave, 
        revwarp = refine_warp(revwarp, 
                              im2_scales[cur_octave], im1_scales[cur_octave],
                              template_size, window_size, step_size, pool)
        # display_warp(revwarp, im2_scales[downsample_octaves], im1_scales[downsample_octaves])


    warp.save(out_forward)
    revwarp.save(out_backward)
