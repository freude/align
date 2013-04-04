#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "CThinPlateSpline.h"

using namespace cv;
using namespace std;

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
    for (int i = 0; i < 3; i++) {
        medianBlur(imgA, imgA, 3);
        resize(imgA, imgA, Size(0, 0), 1.0 / 2, 1.0 / 2, INTER_CUBIC);
    }

    for (int i = 0; i < 3; i++) {
        medianBlur(imgB, imgB, 3);
        resize(imgB, imgB, Size(0, 0), 1.0 / 2, 1.0 / 2, INTER_CUBIC);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "load time [s]: " << t << endl;

    // DETECTION
    // Any openCV detector such as
    BRISK detector(10, 3, 1);
    // SurfFeatureDetector detector(1000,4);

    // DESCRIPTOR
    // Our proposed FREAK descriptor
    // (roation invariance, scale invariance, pattern radius corresponding to SMALLEST_KP_SIZE,
    // number of octaves, optional vector containing the selected pairs)
    // FREAK extractor(true, true, 22, 4, vector<int>());
    FREAK extractor(false, false, 50, 4);

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
    cout << "extraction time [s]: " << t << endl;

    int initial_blocksize = 1024;
    int end_block_size = 64;
    
    // Shifts from A to B
    Mat shift_x = Mat::zeros(imgA.rows / end_block_size + 1, imgA.cols / end_block_size + 1, CV_32F);
    Mat shift_y = shift_x.clone();
    
    t = (double)getTickCount();
    // Bin keypoints by position, mapping to index
    map<pair<int, int>, vector<int> > binsA, binsB;

    for (int idx = 0; idx < keypointsA.size(); idx++) {
        pair<int, int> key(end_block_size * (int) (keypointsA[idx].pt.x / end_block_size + 0.5),
                           end_block_size * (int) (keypointsA[idx].pt.y / end_block_size + 0.5));
        binsA[key].push_back(idx);
    }
    for (int idx = 0; idx < keypointsB.size(); idx++) {
        pair<int, int> key(end_block_size * (int) (keypointsB[idx].pt.x / end_block_size + 0.5),
                           end_block_size * (int) (keypointsB[idx].pt.y / end_block_size + 0.5));
        
        binsB[key].push_back(idx);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "binning time [s]: " << t << endl;

    int cur_blocksize = initial_blocksize;
    for (int cur_blocksize = initial_blocksize; cur_blocksize >= end_block_size; cur_blocksize /= 2) {
        for (int x = 0; x < imgA.cols; x += cur_blocksize) {
            for (int y = 0; y < imgA.rows; y += cur_blocksize) {
                cout << "Block " << x << " " << y << " " << cur_blocksize << endl;
                // filter keypoints to only use those near the current block
                t = (double)getTickCount();
                vector<KeyPoint> local_keypointsA, local_keypointsB;
                vector<int> local_indicesA = nearby_indices(x, y, cur_blocksize, end_block_size, binsA);
                cout << "SHIFT " << shift_x.at<float>(x / end_block_size, y / end_block_size) << " " << shift_y.at<float>(x / end_block_size, y / end_block_size) << endl;
                vector<int> local_indicesB = nearby_indices(x + shift_x.at<float>(x / end_block_size, y / end_block_size),
                                                            y + shift_y.at<float>(x / end_block_size, y / end_block_size),
                                                            cur_blocksize, end_block_size, binsB);
                t = ((double)getTickCount() - t)/getTickFrequency();
                cout << "local find [s]: " << t << endl;
                cout << "     " << local_indicesA.size() << " " << local_indicesB.size() << endl;

                // Create a copy of the local descriptors
                Mat local_descriptorsA(local_indicesA.size(), descriptorsA.cols, descriptorsA.depth());
                Mat local_descriptorsB(local_indicesB.size(), descriptorsB.cols, descriptorsB.depth());
                vector<int>::iterator it;
                for (int idx = 0; idx < local_indicesA.size(); idx++)
                    descriptorsA.row(local_indicesA[idx]).copyTo(local_descriptorsA.row(idx));
                for (int idx = 0; idx < local_indicesB.size(); idx++)
                    descriptorsB.row(local_indicesB[idx]).copyTo(local_descriptorsB.row(idx));

                // match
                vector<DMatch> matches;
                t = (double)getTickCount();
                matcher.match(local_descriptorsA, local_descriptorsB, matches);
                cout << "Found matches " << matches.size() << endl;
                t = ((double)getTickCount() - t)/getTickFrequency();
                cout << "matching time [s]: " << t << endl;

                int num_matches_to_use = min(local_indicesA.size(), local_indicesB.size()) / 2;
                if (num_matches_to_use > 0) {
                    // find weighted average
                    double weight = 0.0;
                    double new_shift_x = 0.0, new_shift_y = 0.0;
                    sort(matches.begin(), matches.end());
                    double mindist = matches[0].distance;
                    for (int idx = 0; idx < num_matches_to_use; idx++) {
                        Point2f delta = keypointsB[local_indicesB[matches[idx].trainIdx]].pt - \
                            keypointsA[local_indicesA[matches[idx].queryIdx]].pt;
                        double curweight = exp(- (matches[idx].distance - mindist));
                        weight += curweight;
                        new_shift_x += delta.x * curweight;
                        new_shift_y += delta.y * curweight;
                    }
                    shift_x.at<float>(x / end_block_size, y / end_block_size) = new_shift_x / weight;
                    shift_y.at<float>(x / end_block_size, y / end_block_size) = new_shift_y / weight;
                    cout << "    " << new_shift_x / weight << " " << new_shift_y / weight << endl << endl;
                }
            }
        }

        // propagate shifts to neighbors
        if (cur_blocksize > end_block_size) {
            map<pair<int, int>, int > neighbor_count;
            for (int x = 0; x < imgA.cols; x += cur_blocksize) {
                for (int y = 0; y < imgA.rows; y += cur_blocksize) {
                    for (int dx = -cur_blocksize / 2; dx <= cur_blocksize / 2; dx += cur_blocksize / 2)
                        for(int dy = -cur_blocksize / 2; dy <= cur_blocksize / 2; dy += cur_blocksize / 2) {
                            int neighborx = x + dx;
                            int neighbory = y + dy;
                            if ((neighborx >= 0) && (neighborx < imgA.cols) &&
                                (neighbory >= 0) && (neighbory < imgA.rows)) {
                                pair<int, int> key(neighborx, neighbory);
                                if (neighbor_count.count(key))
                                    neighbor_count[key]++;
                                else
                                    neighbor_count[key] = 1;

                                // running mean
                                float oldval = shift_x.at<float>(neighborx / end_block_size, neighbory / end_block_size);
                                shift_x.at<float>(neighborx / end_block_size, neighbory / end_block_size) += \
                                    (shift_x.at<float>(x / end_block_size, y / end_block_size) - oldval) / neighbor_count[key];
                                oldval = shift_y.at<float>(neighborx / end_block_size, neighbory / end_block_size);
                                shift_y.at<float>(neighborx / end_block_size, neighbory / end_block_size) += \
                                    (shift_y.at<float>(x / end_block_size, y / end_block_size) - oldval) / neighbor_count[key];
                            }
                        }
                }
            }
        }
    }

    std::vector<Point> srcP, dstP;
    for (int xidx = 0; xidx < shift_x.cols; xidx++) {
        for (int yidx = 0; yidx < shift_y.rows; yidx++) {
            srcP.push_back(Point(xidx * end_block_size, yidx * end_block_size));
            dstP.push_back(Point(xidx * end_block_size + shift_x.at<float>(xidx, yidx),
                                 yidx * end_block_size + shift_y.at<float>(xidx, yidx)));
        }
    }
    CThinPlateSpline tps(srcP, dstP);
    
    // warp the image to dst
    Mat dst;
    tps.warpImage(imgA,dst, 0.1, INTER_LINEAR, BACK_WARP);
    imshow("orig", imgA);
    imshow("warp", dst);
    imshow("second", imgB);
    waitKey(0);
}
