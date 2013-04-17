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

Point2f operator*(Mat M, const Point2f& p) {
    Mat src(3/*rows*/, 1 /* cols */, CV_64F); 
    src.at<double>(0, 0) = p.x; 
    src.at<double>(1, 0) = p.y; 
    src.at<double>(2, 0) = 1.0; 
    Mat dst = M*src;
    return Point2f(dst.at<double>(0,0),dst.at<double>(1,0)); 
}

Point2f operator*(Mat M, const KeyPoint& p) {
    return M * p.pt; 
}

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

#define OCTAVES 3

void 
compute_MAD(vector <KeyPoint>  &keypointsA,
            vector <KeyPoint> &keypointsB,
            Mat descriptorsA,
            Mat descriptorsB,
            const char *direction) {
    // match
    BruteForceMatcher<Hamming> matcher;
    vector <DMatch> matches;
    matcher.match(descriptorsA, descriptorsB, matches);
    
    // compute median and MAD of matches
    vector<float> x_shifts, y_shifts;
    vector<float> match_distances;
    for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++) {
        Point2f delta = keypointsA[it->queryIdx].pt - keypointsB[it->trainIdx].pt;
        x_shifts.push_back(delta.x);
        y_shifts.push_back(delta.y);
        match_distances.push_back(norm(delta));
    }
    sort(x_shifts.begin(), x_shifts.end());
    sort(y_shifts.begin(), y_shifts.end());
    sort(match_distances.begin(), match_distances.end());
    cout << "X shifts " << x_shifts[x_shifts.size() / 4] << " " << x_shifts[x_shifts.size() / 2] << " " << x_shifts[(x_shifts.size() * 3) / 4] << endl;
    cout << "Y shifts " << y_shifts[y_shifts.size() / 4] << " " << y_shifts[y_shifts.size() / 2] << " " << y_shifts[(y_shifts.size() * 3) / 4] << endl;
    cout << "L2 " << match_distances[match_distances.size() / 4] << " " << match_distances[match_distances.size() / 2] << " " << match_distances[(match_distances.size() * 3) / 4] << endl;
    float median_X = x_shifts[x_shifts.size() / 2];
    float median_Y = y_shifts[y_shifts.size() / 2];
    float median_L2 = match_distances[match_distances.size() / 2];
    vector<float> abs_deviations_x, abs_deviations_y;
    for (int idx = 0; idx < matches.size(); idx++) {
        abs_deviations_x.push_back(abs(x_shifts[idx] - median_X));
        abs_deviations_y.push_back(abs(y_shifts[idx] - median_Y));
    }
    sort(abs_deviations_x.begin(), abs_deviations_x.end());
    sort(abs_deviations_y.begin(), abs_deviations_y.end());
    float MAD_X = abs_deviations_x[abs_deviations_x.size() / 2];
    float MAD_Y = abs_deviations_y[abs_deviations_y.size() / 2];
    cout << "MAD " << direction << " " << MAD_X << " " << MAD_Y << " " << sqrt(MAD_X * MAD_X + MAD_Y * MAD_Y) << endl;
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
    for (int i = 0; i < OCTAVES; i++) {
        medianBlur(imgA, imgA, 3);
        resize(imgA, imgA, Size(0, 0), 1.0 / 2, 1.0 / 2, INTER_CUBIC);
    }

    for (int i = 0; i < OCTAVES; i++) {
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

    compute_MAD(keypointsA, keypointsB, descriptorsA, descriptorsB, "FORWARD");
    compute_MAD(keypointsB, keypointsA, descriptorsB, descriptorsA, "REVERSE");
}
