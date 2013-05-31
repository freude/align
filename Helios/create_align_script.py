import sys

images = sys.argv[1:]
for idx1, (im1, im2) in enumerate(zip(images[:-1], images[1:])):
    idx2 = idx1 + 1
    out12 = "OUT/warp.%03d.to.%03d.hdf5" % (idx1 + 1, idx2 + 1)
    out21 = "OUT/warp.%03d.to.%03d.hdf5" % (idx2 + 1, idx1 + 1)
    print 'python align_pair.py "%s" "%s" %s %s' % (im1, im2, out12, out21)
