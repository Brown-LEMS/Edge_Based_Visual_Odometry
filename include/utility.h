#ifndef UTILITY_H
#define UTILITY_H

#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <random>
#include <vector>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "definitions.h"
#include "Frame.h"

// =====================================================================================================================
// UTILITY_TOOLS: useful functions for debugging, writing data to files, displaying images, etc.
//
// ChangeLogs
//    Chien  23-01-18    Initially created.
//    Chien  23-01-19    Add bilinear interpolation    
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ======================================================================================================================

class Utility {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Utility> Ptr;

    Utility();
    double get_Interpolated_Depth( Frame::Ptr Frame, cv::Point2d P );
    double get_Interpolated_Gradient_Depth( Frame::Ptr Frame, cv::Point2d P, std::string grad_Direction );
    void get_dG_2D(cv::Mat &Gx_2d, cv::Mat &Gy_2d, int w, double sigma);
    void Display_Feature_Correspondences(cv::Mat Img1, cv::Mat Img2, \
                                         std::vector<cv::KeyPoint> KeyPoint1, std::vector<cv::KeyPoint> KeyPoint2, \
                                         std::vector<cv::DMatch> Good_Matches );
    std::string cvMat_Type(int type);
};

template <typename T>
double Bilinear_Interpolation(cv::Mat meshGrid, cv::Point2d P)
{

    //> y2 Q12--------Q22
    //      |          |
    //      |    P     |
    //      |          |
    //  y1 Q11--------Q21
    //      x1         x2
    cv::Point2d Q12(floor(P.x), floor(P.y));
    cv::Point2d Q22(ceil(P.x), floor(P.y));
    cv::Point2d Q11(floor(P.x), ceil(P.y));
    cv::Point2d Q21(ceil(P.x), ceil(P.y));

    double f_x_y1 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * meshGrid.at<T>(Q11.y, Q11.x) + ((P.x - Q11.x) / (Q21.x - Q11.x)) * meshGrid.at<T>(Q21.y, Q21.x);
    double f_x_y2 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * meshGrid.at<T>(Q12.y, Q12.x) + ((P.x - Q11.x) / (Q21.x - Q11.x)) * meshGrid.at<T>(Q22.y, Q22.x);
    return ((Q12.y - P.y) / (Q12.y - Q11.y)) * f_x_y1 + ((P.y - Q11.y) / (Q12.y - Q11.y)) * f_x_y2;
}

inline double Bilinear_Interpolation(const cv::Mat &meshGrid, cv::Point2d P)
{
    cv::Point2d Q12(floor(P.x), floor(P.y));
    cv::Point2d Q22(ceil(P.x), floor(P.y));
    cv::Point2d Q11(floor(P.x), ceil(P.y));
    cv::Point2d Q21(ceil(P.x), ceil(P.y));

    if (Q11.x < 0 || Q11.y < 0 || Q21.x >= meshGrid.cols || Q21.y >= meshGrid.rows ||
        Q12.x < 0 || Q12.y < 0 || Q22.x >= meshGrid.cols || Q22.y >= meshGrid.rows)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double fQ11 = meshGrid.at<float>(Q11.y, Q11.x);
    double fQ21 = meshGrid.at<float>(Q21.y, Q21.x);
    double fQ12 = meshGrid.at<float>(Q12.y, Q12.x);
    double fQ22 = meshGrid.at<float>(Q22.y, Q22.x);

    double f_x_y1 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * fQ11 + ((P.x - Q11.x) / (Q21.x - Q11.x)) * fQ21;
    double f_x_y2 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * fQ12 + ((P.x - Q11.x) / (Q21.x - Q11.x)) * fQ22;
    return ((Q12.y - P.y) / (Q12.y - Q11.y)) * f_x_y1 + ((P.y - Q11.y) / (Q12.y - Q11.y)) * f_x_y2;
}

#endif