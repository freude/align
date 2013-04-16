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

#define TEMPLATE_SIZE 64
#define WINDOW_SIZE 128
#define STEP 32

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

    char *im1path = argv[1];
    char *im2path = argv[2];
    int octaves = atoi(argv[3]);
    char *prev_warp = argv[4];

    // Load images
    double t = (double)getTickCount();
    Mat img1 = imread(im1path, CV_LOAD_IMAGE_GRAYSCALE );
    if( !img1.data ) {
        cout<< " --(!) Error reading image " << im1path << endl;
        return -1;
    }
    Mat img2 = imread(im2path, CV_LOAD_IMAGE_GRAYSCALE );
    if( !img2.data ) {
        cout << " --(!) Error reading image " << im2path << endl;
        return -1;
    }
    for (int i = 0; i < octaves; i++) {
        medianBlur(img1, img1, 3);
        resize(img1, img1, Size(0, 0), 1.0 / 2, 1.0 / 2, INTER_CUBIC);
    }

    for (int i = 0; i < octaves; i++) {
        medianBlur(img2, img2, 3);
        resize(img2, img2, Size(0, 0), 1.0 / 2, 1.0 / 2, INTER_CUBIC);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "load time [s]: " << t << endl;

    H5File in_hdf5 = open_hdf5_file(prev_warp);
    Mat row_warp, col_warp;
    read_hdf5_image(in_hdf5, row_warp, "row_map");
    read_hdf5_image(in_hdf5, col_warp, "column_map");
    
    Mat new_row_warp(img1.rows, img1.cols, CV_32F);
    Mat new_col_warp(img1.rows, img1.cols, CV_32F);
    for (int img1_row = 0; img1_row < img1.rows; img1_row++) {
        float warp_i = row_warp.rows * float(img1_row) / img1.rows;
        for (int img1_col = 0; img1_col < img1.cols; img1_col++) {
            float warp_j = row_warp.cols * float(img1_col) / img1.cols;
            
            int img2_row = binterp(row_warp, warp_i, warp_j) * img2.rows;
            int img2_col = binterp(col_warp, warp_i, warp_j) * img2.cols;
            new_row_warp.at<float>(img1_row, img1_col) = img2_row;
            new_col_warp.at<float>(img1_row, img1_col) = img2_col;
        }
    }
    Mat warped_im2;
    remap(img2, warped_im2, new_col_warp, new_row_warp, INTER_LINEAR);
    while (1) {
        imshow("result", warped_im2);
        cout << "warped" << endl;
        waitKey(0);
        imshow("result", img1);
        cout << "img1" << endl;
        waitKey(0);
    }
}
