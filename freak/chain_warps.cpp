// chain_warps.cpp - 

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <H5Cpp.h>

using namespace cv;
using namespace H5;
using namespace std;

H5File create_hdf5_file(char *filename);
H5File open_hdf5_file(char *filename);
void write_hdf5_image(H5File h5f, const char *name, const Mat &im);
void read_hdf5_image(H5File h5f, Mat &image_out, const char *name, const Rect &roi=Rect(0,0,0,0));

static inline double lerp(float a, float b, float t)
{
    return (b - a) * t + a;
}

double binterp(const Mat im, float fi, float fj)
{
    float v00 = im.at<float>((int) fi, (int) fj);
    float v10 = im.at<float>(MIN((int) fi + 1, im.rows - 1), (int) fj);
    float v01 = im.at<float>((int) fi, MIN((int) fj + 1, im.cols - 1));
    float v11 = im.at<float>(MIN((int) (fi + 1), im.rows - 1), MIN((int) fj + 1, im.cols - 1));
    return lerp(lerp(v00, v10, fi - (int) fi),
                lerp(v01, v11, fi - (int) fi),
                fj - (int) fj);
}

int main( int argc, char** argv ) {
    // check http://opencv.itseez.com/doc/tutorials/features2d/table_of_content_features2d/table_of_content_features2d.html
    // for OpenCV general detection/matching framework details

    char *warp1path = argv[1];
    char *warp2path = argv[2];
    char *warp3path = argv[3];

    H5File in_hdf5 = open_hdf5_file(warp1path);
    Mat row_warp_1, col_warp_1;
    read_hdf5_image(in_hdf5, row_warp_1, "row_map");
    read_hdf5_image(in_hdf5, col_warp_1, "column_map");
    in_hdf5.close();

    in_hdf5 = open_hdf5_file(warp2path);
    Mat row_warp_2, col_warp_2;
    read_hdf5_image(in_hdf5, row_warp_2, "row_map");
    read_hdf5_image(in_hdf5, col_warp_2, "column_map");
    in_hdf5.close();
    
    Mat new_row_warp(row_warp_1.rows, row_warp_1.cols, CV_32F);
    Mat new_col_warp(row_warp_1.rows, row_warp_1.cols, CV_32F);
    for (int row_warp_1_row = 0; row_warp_1_row < row_warp_1.rows; row_warp_1_row++) {
        for (int row_warp_1_col = 0; row_warp_1_col < row_warp_1.cols; row_warp_1_col++) {
            float r2 = row_warp_1.at<float>(row_warp_1_row, row_warp_1_col) * row_warp_2.rows;
            float c2 = col_warp_1.at<float>(row_warp_1_row, row_warp_1_col) * row_warp_2.cols;
            r2 = MIN(MAX(0, r2), row_warp_2.rows - 1);
            c2 = MIN(MAX(0, c2), row_warp_2.cols - 1);

            new_row_warp.at<float>(row_warp_1_row, row_warp_1_col) = binterp(row_warp_2, r2, c2);
            new_col_warp.at<float>(row_warp_1_row, row_warp_1_col) = binterp(col_warp_2, r2, c2);
        }
    }
    H5File out_hdf5 = create_hdf5_file(warp3path);
    write_hdf5_image(out_hdf5, "row_map", new_row_warp);
    write_hdf5_image(out_hdf5, "column_map", new_col_warp);
    out_hdf5.close();
}
