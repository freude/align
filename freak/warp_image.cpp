#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <H5Cpp.h>

using namespace cv;
using namespace H5;
using namespace std;

H5File create_hdf5_file(char *filename);
H5File open_hdf5_file(char *filename);
void write_hdf5_image(H5File h5f, const char *name, const Mat &im);
void read_hdf5_image(H5File h5f, Mat &image_out, const char *name, const Rect &roi=Rect(0,0,0,0));

int main( int argc, char** argv ) {
    int imnum = atoi(argv[1]);

    // Load images
    Mat imgA = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if( !imgA.data ) {
        cout<< " --(!) Error reading image " << argv[1] << endl;
        return -1;
    }

    H5File h5f = open_hdf5_file(argv[3]);
    Mat warpinfo;
    read_hdf5_image(h5f, warpinfo, "relaxed");

    Mat imidx = warpinfo.col(0);
    Mat orig_i = warpinfo.col(1);
    Mat orig_j = warpinfo.col(2);
    Mat dest_i = warpinfo.col(3);
    Mat dest_j = warpinfo.col(4);
    double mini, maxi, minj, maxj;
    minMaxLoc(dest_i, &mini, &maxi);
    minMaxLoc(dest_j, &minj, &maxj);
    int base_i = mini - 1;
    int base_j = minj - 1;
    int new_height =  maxi + 1 - base_i;
    int new_width =  maxj + 1 - base_j;

    cout << "Will write " << new_height << " " << new_width << endl;

    // create map by diffusion
    Mat xmap = Mat::zeros(Size(new_width, new_height), CV_32F);
    Mat ymap = Mat::zeros(Size(new_width, new_height), CV_32F);
    Mat weight = Mat::zeros(Size(new_width, new_height), CV_32F);

    int ct = 0;
    for (int p = 0; p < imidx.rows; p++) {
        if (imidx.at<float>(p) == imnum) {
            ct++;
            int di = dest_i.at<float>(p) - base_i;
            int dj = dest_j.at<float>(p) - base_j;
            float delta_i = orig_i.at<float>(p) - di;
            float delta_j = orig_j.at<float>(p) - dj;
            xmap.at<float>(di, dj) = delta_i;
            ymap.at<float>(di, dj) = delta_j;
            weight.at<float>(di, dj) = 1.0;
        }
    }
    cout << "mapped " << ct << " points" << endl;
    for (int iter = 0; iter < 10; iter++) {
        cout << "iter " << iter << endl;
        GaussianBlur(xmap, xmap, Size(0, 0), 5.0);
        GaussianBlur(ymap, ymap, Size(0, 0), 5.0);
        GaussianBlur(weight, weight, Size(0, 0), 5.0);
    }
    add(weight, weight == 0, weight, noArray(), CV_32F);
    xmap = xmap / weight;
    ymap = ymap / weight;
    
    for (int xbase = 0; xbase < xmap.cols; xbase++) {
        for (int ybase = 0; ybase < xmap.rows; ybase++) {
            xmap.at<float>(ybase, xbase) += xbase;
            ymap.at<float>(ybase, xbase) += ybase;
        }
    }

    Mat warped;
    remap(imgA, warped, xmap, ymap, INTER_LINEAR);
    imwrite(argv[4], warped);

    resize(warped, warped, Size(1024, 1024));
    imwrite(argv[5], warped);
}
