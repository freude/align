import sys
import subprocess

images = sys.argv[1:]
num_images = len(images)
max_delta = 3

def find_rough_warp(im1_path, im2_path, out_path):
    subprocess.check_call(["./compute_mapping", im1_path, im2_path, out_path])

for idx, f1 in enumerate(images):
    for offset in range(-max_delta, max_delta + 1):
        if (idx + offset >= 0) and (idx + offset < num_images) and (offset != 0):
            print "IDX", idx, idx + offset
            find_rough_warp(images[idx], images[idx + offset], ".tmp")
