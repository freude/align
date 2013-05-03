import sys
import os
import re
import subprocess

import numpy as np
import h5py

mapping_dir = sys.argv[1]
image_dir = sys.argv[2]
out_dir = sys.argv[3]

mapping_files = sorted([os.path.join(mapping_dir, fn) for fn in
                        os.listdir(mapping_dir)])
image_files = sorted([os.path.join(image_dir, fn) for fn in 
                      sorted(os.listdir(image_dir))])

p = re.compile('.*av[.]([0-9]+)[.]hdf5.*')

for mf in mapping_files:
    match = p.match(mf)
    if not match:
        continue
    print mf
    idx = int(match.group(1))
    outfile = "out.%0.3d.tif" % (idx + 1)
    print "MAP", image_files[idx], outfile
    subprocess.check_call(['show_mapping',
                           image_files[idx],
                           image_files[idx],
                           '2',
                           mf,
                           os.path.join(out_dir, outfile)])
    subprocess.check_call(['exiftool',
                           '-ImageDescription=Warped from %s using %s' % (image_files[idx], mf),
                           os.path.join(out_dir, outfile)])

