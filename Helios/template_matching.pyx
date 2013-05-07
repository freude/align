cimport cython

cdef extern from "math.h":
    double sqrt(double) nogil

cdef extern from "opencv2/core/core.hpp":
    int CV_32F
    int CV_8U

cdef extern from "opencv2/core/core.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat(int, int, int, void *, int) nogil
        Mat() nogil
        unsigned char *ptr(int) nogil
        int rows, cols
        int depth() nogil
    cdef cppclass Point:
        int x, y
    cdef cppclass Scalar:
        double operator[](int)
    void minMaxLoc(Mat &src, double* minVal, double* maxVal, Point* minLoc, Point* maxLoc) nogil
    void meanStdDev(const Mat& mtx, Scalar& mean, Scalar& stddev) nogil

cdef extern from "opencv2/highgui/highgui.hpp" namespace "cv":
   void imshow(const char *name, Mat &im) nogil


cdef extern from "opencv2/imgproc/imgproc.hpp" namespace "cv":
    void matchTemplate(Mat &, Mat &, Mat &, int) nogil
    int TM_CCORR_NORMED

# Use Welford's method to compute the mean, stddev, and also locate the maximum
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void meanstdmax(float *data, int count, float *out_mean, float *out_std, float *max, int *maxidx) nogil:
    cdef float M, S, oldM, oldS, x
    cdef int n, i
    oldM = M = data[0]
    oldS = S = 0
    max[0] = data[0]
    maxidx[0] = 0
    n = 1
    for i in range(1, count):
        x = data[i]
        if x > max[0]:
            max[0] = x
            maxidx[0] = i
        n += 1
        M = oldM + (x - oldM) / n
        S = oldS + (x - oldM) * (x - M)
        oldM = M
        oldS = S
    out_mean[0] = M
    out_std[0] = sqrt(S / (n - 1))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _best_match(int template_row, int template_col,
                      int window_row, int window_col,
                      int template_size, int window_size,
                      unsigned char [:, :] template_im, unsigned char [:, :] window_im,
                      int *out_row, int *out_col, float *out_score) nogil:
    cdef Mat template, window, match
    cdef int maxidx
    cdef float mean, stddev, maxval
    template = Mat(template_size, template_size, CV_8U,
                   &(template_im[template_row, template_col]),
                   template_im.strides[0])
    window = Mat(window_size, window_size, CV_8U,
                 &(window_im[window_row, window_col]),
                 window_im.strides[0])

    matchTemplate(window, template, match, TM_CCORR_NORMED)
    meanstdmax(<float *> match.ptr(0), match.rows * match.cols, &mean, &stddev, &maxval, &maxidx)
    out_row[0] = maxidx / match.cols
    out_col[0] = maxidx % match.cols
    if stddev > 0:
        out_score[0] = (maxval - mean) / stddev
    else:
        out_score[0] = 0.0

cpdef best_match(template_row, template_col,
               window_row, window_col,
               template_size, window_size,
               tim, wim):
    cdef int a, b
    cdef float c
    _best_match(template_row, template_col,
                window_row, window_col,
                template_size, window_size,
                tim, wim, &a, &b, &c)
    return (a, b, c)
