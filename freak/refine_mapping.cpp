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
#define APPROXSTEP 16

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
    char *out_warp = argv[5];

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
    
    int rowsteps = (img1.rows / APPROXSTEP) + 1;
    int colsteps = (img1.cols / APPROXSTEP) + 1;
    float rowstepsize = float(img1.rows - 1) / (rowsteps - 1);
    float colstepsize = float(img1.cols - 1) / (colsteps - 1);

    Mat new_row_warp = Mat::zeros(rowsteps, colsteps, CV_32F);
    Mat new_col_warp = Mat::zeros(rowsteps, colsteps, CV_32F);
    Mat match_weight = Mat::zeros(rowsteps, colsteps, CV_32F);
    for (int rowstep = 0; rowstep < rowsteps; rowstep++) {
        int img1_row = cvRound(rowstep * rowstepsize);
        // cout << img1_row << " / " << img1.rows << endl;
        float warp_i = (row_warp.rows - 1) * float(img1_row) / (img1.rows - 1);
        for (int colstep = 0; colstep < colsteps; colstep++) {
            int img1_col = cvRound(colstep * colstepsize);
            float warp_j = (row_warp.cols - 1) * float(img1_col) / (img1.cols - 1);
            
            int img2_row = binterp(row_warp, warp_i, warp_j) * (img2.rows - 1);
            int img2_col = binterp(col_warp, warp_i, warp_j) * (img2.cols - 1);
            
            // cout << "(" << img1_row << ", " << img1_col << ") maps to (" << img2_row << ", " << img2_col << ")" << endl;

            Mat subimg1 = img1(Range(img1_row, MIN(img1_row + TEMPLATE_SIZE, img1.rows)),
                               Range(img1_col, MIN(img1_col + TEMPLATE_SIZE, img1.cols)));
            
            int rlo2 = MAX(img2_row - (WINDOW_SIZE - TEMPLATE_SIZE) / 2, 0);
            int clo2 = MAX(img2_col - (WINDOW_SIZE - TEMPLATE_SIZE) / 2, 0);
            if ((rlo2 > img2.rows) || (clo2 > img2.cols))
                continue;
            Mat subimg2 = img2(Range(rlo2, MIN(rlo2 + WINDOW_SIZE, img2.rows )),
                               Range(clo2, MIN(clo2 + WINDOW_SIZE, img2.cols)));

            if ((subimg1.rows < subimg2.rows) && (subimg1.cols < subimg2.cols)) {
                Mat result;
                double minVal; double maxVal; Point minLoc; Point maxLoc;
                Scalar mean, stddev;
                matchTemplate(subimg2, subimg1, result, CV_TM_CCORR_NORMED);
                minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
                meanStdDev(result, mean, stddev);

                
                float new_row = (rlo2 + maxLoc.y) / ((float) img2.rows - 1) - ((float) img1_row) / (img1.rows - 1);
                float new_col = (clo2 + maxLoc.x) / ((float) img2.cols - 1) - ((float) img1_col) / (img1.cols - 1);
                //cout << img1_row << " " << img1_col << " should map to " <<  (rlo2 + maxLoc.y) << " " << (clo2 + maxLoc.x) << endl;
                //                cout << "    stored at " << rowstep << " " << colstep << endl;
                float w;
                if (stddev[0] > 0) {
                    w = (maxVal - mean[0]) / stddev[0];
                } else {
                    w = 0;
                }

                // Keep only strongest matches
                if (w < 3) {
                    w = 0;
                }
                new_row_warp.at<float>(rowstep, colstep) = new_row;
                new_col_warp.at<float>(rowstep, colstep) = new_col;
                match_weight.at<float>(rowstep, colstep) = w;
                
                // cout << "   new (" << (int) (new_row * img2.rows) << ", " << (int) (new_col * img2.cols) << ") weight " << w << endl;

            }
        }
    }
    // smooth with weights**2
    match_weight = match_weight.mul(match_weight);
    new_row_warp = new_row_warp.mul(match_weight);
    new_col_warp = new_col_warp.mul(match_weight);
    // Loop enough that there should be weight everywhere.
    // Filter radius for sigma=3 is approximately 10
    for (int l = 0; l < MAX(match_weight.rows, match_weight.cols) / 10 + 1; l++) {
        GaussianBlur(new_row_warp, new_row_warp, Size(0, 0), 3);
        GaussianBlur(new_col_warp, new_col_warp, Size(0, 0), 3);
        GaussianBlur(match_weight, match_weight, Size(0, 0), 3);
    }
    new_row_warp = new_row_warp / match_weight;
    new_col_warp = new_col_warp / match_weight;
    
//     Mat disp_row_warp, disp_col_warp;
//      normalize( new_row_warp, disp_row_warp, 0, 1, NORM_MINMAX, -1, Mat() );
//      imshow("row", disp_row_warp);
//      normalize( new_col_warp, disp_col_warp, 0, 1, NORM_MINMAX, -1, Mat() );
//      imshow("col", disp_col_warp);
//      normalize( match_weight, match_weight, 0, 1, NORM_MINMAX, -1, Mat() );
//      imshow("w", match_weight);
//      waitKey(0);
//  
    // convert from deltas to raw coords
    for (int rowstep = 0; rowstep < rowsteps; rowstep++) {
        int img1_row = cvRound(rowstep * rowstepsize);
        for (int colstep = 0; colstep < colsteps; colstep++) {
            int img1_col = cvRound(colstep * colstepsize);
            new_row_warp.at<float>(rowstep, colstep) += ((float) img1_row) / (img1.rows - 1);
            new_col_warp.at<float>(rowstep, colstep) += ((float) img1_col) / (img1.cols - 1);
        }
    }

    H5File out_hdf5 = create_hdf5_file(out_warp);
    write_hdf5_image(out_hdf5, "row_map", new_row_warp);
    write_hdf5_image(out_hdf5, "column_map", new_col_warp);
    out_hdf5.close();

//     normalize( new_row_warp, new_row_warp, 0, 1, NORM_MINMAX, -1, Mat() );
//     imshow("row", new_row_warp);
//     normalize( new_col_warp, new_col_warp, 0, 1, NORM_MINMAX, -1, Mat() );
//     imshow("col", new_col_warp);
//     normalize( match_weight, match_weight, 0, 1, NORM_MINMAX, -1, Mat() );
//     imshow("w", match_weight);
//     waitKey(0);
//     waitKey(0);
// 
}
