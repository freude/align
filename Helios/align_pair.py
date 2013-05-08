import sys
import threading
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
    template_size = 64
    window_size = 128

    # step size for refined warps
    step_size = 8

    # TODO: parse args

    im1 = cv2.imread(sys.argv[1], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    im2 = cv2.imread(sys.argv[2], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    outfile = sys.argv[3]

    # compute scaled versions
    im1_scales = scalespace(im1, downsample_octaves)
    im2_scales = scalespace(im2, downsample_octaves)
    smallest1 = im1_scales[downsample_octaves]
    smallest2 = im2_scales[downsample_octaves]

    # detect keypoints, extract descriptors
    def detect(im, threshold=10, step=10, max_keypoints=(10 * maximum_matches)):
        while True:
            detector = cv2.BRISK(threshold, 0, 1)
            keypoints = detector.detect(im)
            if len(keypoints) < max_keypoints:
                return keypoints
            threshold += step

    extractor = cv2.DescriptorExtractor_create('FREAK')
    keypoints1, descriptors1 = extractor.compute(smallest1, detect(smallest1))
    keypoints2, descriptors2 = extractor.compute(smallest2, detect(smallest2))

    print len(keypoints1), len(keypoints2), "keypoints"

    # match keypoints, keep only distinct matches
    matcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
    def good_match(m1, m2, thresh=0.7, eps=1e-5):
        return (m1.distance + eps) / (m2.distance + eps) < thresh
    matches = [m1 for m1, m2 in matcher.knnMatch(descriptors1, descriptors2, 2)
               if good_match(m1, m2)]

    # make sure we have enough distinct matches to proceed
    if len(matches) < minimum_matches:
        sys.stderr.write("Found too few distinct matches between {0} and {1}.\n" +
                         "Needed {2} and found {3}\n".format(sys.argv[1], sys.argv[2],
                                                             minimum_matches, len(matches)))
        sys.exit(1)

    print len(matches), "matches"

    # compute R, T that transform kpts in 1 to 2
    MAD, R, T, dists = ransac(keypoints1, keypoints2, matches)
    print "MAD of keypoint distsances after rigid correction:", MAD

    # filter matches with too large a separation
    thresh = 2.0 * 1.48 * min(MAD, 500 >> downsample_octaves)
    matches = [m for idx, m in enumerate(matches) if dists[idx] < thresh]
    print len(matches), "after filtering"

    # Create the initial warp
    # change from normalized coords, to image coords (for R, T), then back
    def scale(sj, si):
        return np.matrix([[sj, 0], [0, si]])
    pre = scale(smallest1.shape[1] - 1, smallest1.shape[0] - 1)
    post = scale(1.0 / (smallest2.shape[1] - 1), 1.0 / (smallest2.shape[0] - 1))
    R = post * R * pre
    T = post * T
    warp = RigidWarp(R, T)

    def display_warp(w, im1, im2):
        warpedim = w.warp([im2_scales[0]], im1.shape)[0]
        while True:
            cv2.imshow("view", warpedim.astype(smallest1.dtype))
            k = cv2.waitKey()
            if k == 27:
                break
            cv2.imshow("view", im1)
            k = cv2.waitKey()
            if k == 27:
                break

    if not hasattr(threading.current_thread(), "_children"):
        threading.current_thread()._children = weakref.WeakKeyDictionary()
    pool = ThreadPool(8)

    # coarse-to-fine warp refinement using normalized cross correlation in subimages
    for cur_octave in range(downsample_octaves, -1, -1):
        warp = refine_warp(warp, 
                           im1_scales[cur_octave], im2_scales[cur_octave],
                           template_size, window_size, step_size, pool)
    display_warp(warp, im1_scales[downsample_octaves - 1], im2_scales[downsample_octaves - 1])


