import sys
import threading
import numpy as np
import cv2
from warp import NonlinearWarp

w1 = NonlinearWarp.load(sys.argv[1])
w2 = NonlinearWarp.load(sys.argv[2])
w3 = w1.chain(w2)

def display_warp(w, im1, im2, w1, w2):
    warpedim = w.warp([im2], im1.shape)[0]
    warpedim2 = w1.warp([w2.warp([im2], im1.shape)[0]], im1.shape)[0]
    while True:
        cv2.imshow("view", warpedim.astype(im1.dtype))
        print "chained"
        k = cv2.waitKey()
        if k == 27:
            break
        cv2.imshow("view", warpedim2.astype(im1.dtype))
        print "composed"
        k = cv2.waitKey()
        if k == 27:
            break
        cv2.imshow("view", im1)
        k = cv2.waitKey()
        if k == 27:
            break

print "ehre"

im1 = cv2.resize(cv2.imread(sys.argv[3], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE), (2048, 2048))
im2 = cv2.resize(cv2.imread(sys.argv[4], flags=cv2.CV_LOAD_IMAGE_GRAYSCALE), (2048, 2048))
print "foo"
display_warp(w3, im1, im2, w1, w2)
