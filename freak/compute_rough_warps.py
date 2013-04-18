import sys
import os
import subprocess

def find_rough_warp(im1_path, im2_path, out_path):
    subprocess.check_call(["./compute_mapping", im1_path, im2_path, out_path])

def refine_warp(im1_path, im2_path, octaves, in_warp, out_warp):
    subprocess.check_call(["./refine_mapping", im1_path, im2_path, str(octaves), in_warp, out_warp])

if __name__ == '__main__':
    outdir = sys.argv[1]
    warpfile = open(os.path.join(outdir, "warpfile.txt"), "w")
    filenames = sys.argv[2:]
    for idx in range(len(filenames) - 1):
        outfile = os.path.join(outdir, "rough_warp.{0:04}.to.{1:04}.hdf5".format(idx, idx + 1))
        find_rough_warp(filenames[idx], filenames[idx + 1], outfile)
        print "Done", filenames[idx], filenames[idx + 1], outfile, '/', idx, '/',  len(filenames) - 1
        warpfile.write("ROUGH_WARP: {} to {} : {}\n".format(filenames[idx],
                                                            filenames[idx + 1],
                                                            outfile))

        outfile_2 = os.path.join(outdir, "refined_warp.{0:04}.to.{1:04}.hdf5".format(idx, idx + 1))
        refine_warp(filenames[idx], filenames[idx + 1], 3, outfile, outfile_2)
        warpfile.write("REFINED_WARP: {} to {} : {}\n".format(filenames[idx],
                                                              filenames[idx + 1],
                                                              outfile_2))

        outfile = os.path.join(outdir, "rough_warp.{0:04}.to.{1:04}.hdf5".format(idx + 1, idx))
        find_rough_warp(filenames[idx + 1], filenames[idx], outfile)
        print "Done", filenames[idx + 1], filenames[idx], outfile, '/', idx, '/',  len(filenames) - 1
        warpfile.write("ROUGH_WARP: {} to {} : {}\n".format(filenames[idx + 1],
                                                            filenames[idx],
                                                            outfile))
        outfile_2 = os.path.join(outdir, "refined_warp.{0:04}.to.{1:04}.hdf5".format(idx + 1, idx))
        refine_warp(filenames[idx + 1], filenames[idx], 3, outfile, outfile_2)
        warpfile.write("REFINED_WARP: {} to {} : {}\n".format(filenames[idx + 1],
                                                              filenames[idx],
                                                              outfile_2))
        
