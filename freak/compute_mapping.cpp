#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "CThinPlateSpline.h"
#include <H5Cpp.h>

using namespace cv;
using namespace H5;
using namespace std;

H5File create_hdf5_file(char *filename);
H5File open_hdf5_file(char *filename);
void write_hdf5_image(H5File h5f, const char *name, const Mat &im);
void read_hdf5_image(H5File h5f, Mat &image_out, const char *name, const Rect &roi=Rect(0,0,0,0));

static vector<int>
nearby_indices(int x, int y, int cur_blocksize, int end_block_size,
               map<pair<int, int>, vector<int> > bins)
{
    // round to nearest end_block_size
    x = end_block_size * (int) (((float) x) / end_block_size + 0.5);
    y = end_block_size * (int) (((float) y) / end_block_size + 0.5);
    vector<int> out;
    for (int dx = -cur_blocksize; dx <= cur_blocksize; dx += end_block_size)
        for (int dy = -cur_blocksize; dy <= cur_blocksize; dy += end_block_size) {
            pair<int, int> key(x + dx, y + dy);
            if (bins.count(key) > 0) {
                out.insert(out.end(), bins[key].begin(), bins[key].end());
            } 
        }
    return out;
}
            
#define OCTAVE 3

int main( int argc, char** argv ) {
    // check http://opencv.itseez.com/doc/tutorials/features2d/table_of_content_features2d/table_of_content_features2d.html
    // for OpenCV general detection/matching framework details

    // Load images
    double t = (double)getTickCount();
    Mat imgA = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    if( !imgA.data ) {
        cout<< " --(!) Error reading image " << argv[1] << endl;
        return -1;
    }
    Mat imgB = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE );
    if( !imgB.data ) {
        cout << " --(!) Error reading image " << argv[2] << endl;
        return -1;
    }
    for (int i = 0; i < OCTAVE; i++) {
        medianBlur(imgA, imgA, 3);
        resize(imgA, imgA, Size(0, 0), 1.0 / 2, 1.0 / 2, INTER_CUBIC);
    }

    for (int i = 0; i < OCTAVE; i++) {
        medianBlur(imgB, imgB, 3);
        resize(imgB, imgB, Size(0, 0), 1.0 / 2, 1.0 / 2, INTER_CUBIC);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "load time [s]: " << t << endl;

    // DETECTION
    // Any openCV detector such as
    // BRISK detector(20, 0, 1);
    // SurfFeatureDetector detector(1000,4);
    // MSER detector;
    // FastFeatureDetector detector(80, true);
    // StarDetector detector;
    SurfFeatureDetector detector( 1000, 4 );

    // DESCRIPTOR
    // Our proposed FREAK descriptor
    // (roation invariance, scale invariance, pattern radius corresponding to SMALLEST_KP_SIZE,
    // number of octaves, optional vector containing the selected pairs)
    // FREAK extractor(true, true, 22, 4, vector<int>());
    FREAK extractor(false, false, 25, 1);

    // MATCHER
    // The standard Hamming distance can be used such as
    // BruteForceMatcher<Hamming> matcher;
    // or the proposed cascade of hamming distance using SSSE3
    BruteForceMatcher<Hamming> matcher;

    // detect
    vector<KeyPoint> keypointsA, keypointsB;
    t = (double)getTickCount();
    detector.detect(imgA, keypointsA);
    detector.detect(imgB, keypointsB);
    cout << "Detected " << keypointsA.size() << ", " << keypointsB.size() << endl;
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "detection time [s]: " << t/1.0 << endl;

    // extract features
    Mat descriptorsA, descriptorsB;
    t = (double)getTickCount();
    extractor.compute(imgA, keypointsA, descriptorsA);
    extractor.compute(imgB, keypointsB, descriptorsB);
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "extraction (" << descriptorsA.rows << ", " << descriptorsB.rows << ") time [s]: " << t << endl;

    // match
    vector<DMatch> matches;
    t = (double)getTickCount();
    matcher.match(descriptorsA, descriptorsB, matches);
    cout << "Found matches " << matches.size() << endl;
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "matching time [s]: " << t << endl;

    // compute median and MAD of matches
    vector<float> x_shifts, y_shifts;
    vector<float> match_distances;
    for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++) {
        Point2f delta = keypointsA[it->queryIdx].pt - keypointsB[it->trainIdx].pt;
        x_shifts.push_back(delta.x);
        y_shifts.push_back(delta.y);
    }
    sort(x_shifts.begin(), x_shifts.end());
    sort(y_shifts.begin(), y_shifts.end());
    cout << "X shifts " << x_shifts[x_shifts.size() / 4] << " " << x_shifts[x_shifts.size() / 2] << " " << x_shifts[(x_shifts.size() * 3) / 4] << endl;
    cout << "Y shifts " << y_shifts[y_shifts.size() / 4] << " " << y_shifts[y_shifts.size() / 2] << " " << y_shifts[(y_shifts.size() * 3) / 4] << endl;
    float median_X = x_shifts[x_shifts.size() / 2];
    float median_Y = y_shifts[y_shifts.size() / 2];
    vector<float> abs_deviations_x, abs_deviations_y;
    for (int idx = 0; idx < matches.size(); idx++) {
        abs_deviations_x.push_back(abs(x_shifts[idx] - median_X));
        abs_deviations_y.push_back(abs(y_shifts[idx] - median_Y));
    }
    sort(abs_deviations_x.begin(), abs_deviations_x.end());
    sort(abs_deviations_y.begin(), abs_deviations_y.end());
    float MAD_X = abs_deviations_x[abs_deviations_x.size() / 2];
    float MAD_Y = abs_deviations_y[abs_deviations_y.size() / 2];
    cout << "MAD " << MAD_X << " " << MAD_Y << endl;

    // filter for sane matches, those with <= 3 sigma_mad = 3 * 1.48 * MAD
    vector<DMatch> good_matches;
    for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++) {
        Point2f delta = keypointsA[it->queryIdx].pt - keypointsB[it->trainIdx].pt;
        if (abs(delta.x - median_X) < 2.0 * 1.48 * MAD_X &&
            abs(delta.y - median_Y) < 2.0 * 1.48 * MAD_Y)
            good_matches.push_back(*it);
    }

    cout << "Good matches " << good_matches.size() <<endl;

//     // Draw matches
//     Mat imgMatch;
//     random_shuffle(good_matches.begin(), good_matches.end());
//     vector<DMatch> curgood_matches(good_matches.begin(), good_matches.begin() + 100);
//     drawMatches(imgA, keypointsA, imgB, keypointsB, curgood_matches, imgMatch);
//     namedWindow("good_matches", CV_WINDOW_KEEPRATIO);
//     imshow("good_matches", imgMatch);
//     waitKey(0);
// 
    // create map by diffusion
    Mat xmap = Mat::zeros(imgA.size(), CV_32F);
    Mat ymap = Mat::zeros(imgA.size(), CV_32F);
    Mat weight = Mat::zeros(imgA.size(), CV_32F);
    for (vector<DMatch>::iterator it = good_matches.begin(); it != good_matches.end(); it++) {
        // map points in A onto B
        Point2f delta = keypointsB[it->trainIdx].pt - keypointsA[it->queryIdx].pt;
        xmap.at<float>(keypointsA[it->queryIdx].pt) = delta.x;
        ymap.at<float>(keypointsA[it->queryIdx].pt) = delta.y;
        weight.at<float>(keypointsA[it->queryIdx].pt) = 1.0;
    }
    for (int iter = 0; iter < 4; iter++) {
        GaussianBlur(xmap, xmap, Size(0, 0), 10.0);
        GaussianBlur(ymap, ymap, Size(0, 0), 10.0);
        GaussianBlur(weight, weight, Size(0, 0), 10.0);
    }
    add(weight, weight == 0, weight, noArray(), CV_32F);
    xmap = xmap / weight;
    ymap = ymap / weight;
    
    for (int xbase = 0; xbase < imgA.cols; xbase++) {
        for (int ybase = 0; ybase < imgA.rows; ybase++) {
            xmap.at<float>(ybase, xbase) += xbase;
            ymap.at<float>(ybase, xbase) += ybase;
        }
    }

    H5File out_hdf5 = create_hdf5_file(argv[3]);
    write_hdf5_image(out_hdf5, "warp_rough_x", xmap);
    write_hdf5_image(out_hdf5, "warp_rough_y", ymap);

#if 0
    Mat warpedB;
    remap(imgB, warpedB, xmap, ymap, INTER_LINEAR);
    while (1) {
        cout << "wB" << endl;
        imshow("warped", warpedB);
        waitKey(0);
        
        cout << "A" << endl;
        imshow("warped", imgA);
        waitKey(0);
    }
#endif
}
