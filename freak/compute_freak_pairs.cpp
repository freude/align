// compute_freak_pairs.cpp - 

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/video/tracking.hpp>
#include <H5Cpp.h>

using namespace cv;
using namespace std;

#define OCTAVES 3

int main( int argc, char** argv ) {
    // DETECTION
    // Any openCV detector such as
    // BRISK detector(20, 0, 1);
    // SurfFeatureDetector detector(1000,4);
    // MSER detector;
    // FastFeatureDetector detector(80, true);
    // StarDetector detector;
    SurfFeatureDetector detector( 1000, 4 );
    FREAK extractor(false, false, 25, 1);

    vector<Mat> ims;
    vector<vector<KeyPoint> > kpts;

    for (int i = 1; i < argc; i++) {
        // Load images
        double t = (double)getTickCount();
        Mat imgA = imread(argv[i], CV_LOAD_IMAGE_GRAYSCALE );
        if( !imgA.data ) {
            cout<< " --(!) Error reading image " << argv[i] << endl;
            return -1;
        }
        for (int o = 0; o < OCTAVES; o++) {
            medianBlur(imgA, imgA, 3);
            resize(imgA, imgA, Size(0, 0), 1.0 / 2, 1.0 / 2, INTER_CUBIC);
        }
        ims.push_back(imgA);
        cout << "load time [s]: " << t << endl;

        // detect
        vector<KeyPoint> keypointsA;
        t = (double)getTickCount();
        detector.detect(imgA, keypointsA);
        cout << "Detected " << keypointsA.size() << endl;
        t = ((double)getTickCount() - t)/getTickFrequency();
        cout << "detection time [s]: " << t/1.0 << endl;
        kpts.push_back(keypointsA);
    }
    
    vector<int> pairs = extractor.selectPairs(ims, kpts, 0.7);
    for (int i = 0; i < pairs.size(); i++) {
        cout << pairs[i] << endl;
    }
}
