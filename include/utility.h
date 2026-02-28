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
#include "toed/cpu_toed.hpp"

// =====================================================================================================================
// UTILITY_TOOLS: useful functions for debugging, writing data to files, displaying images, etc.
//
// ChangeLogs
//    Chien  23-01-18    Initially created.
//    Chien  23-01-19    Add bilinear interpolation
//    Jue    25-06-11    Added other utility functions
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ======================================================================================================================

class Utility
{

public:
    //> Make the class shareable as a pointer
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Utility> Ptr;

    Utility();
    double getNormalDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &epiline_x, double &epiline_y);
    double getNormalDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::VectorXd edges, int index, double &epiline_x, double &epiline_y);
    double getTangentialDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &x_intersection, double &y_intersection);
    double getTangentialDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::VectorXd edges, int index, double &x_intersection, double &y_intersection);

    Eigen::Vector3d backproject_2D_point_to_3D_point_using_rays( const Eigen::Matrix3d rel_R, const Eigen::Vector3d rel_T, const Eigen::Vector3d ray1, const Eigen::Vector3d ray2 );
    Eigen::Vector3d reconstruct_3D_Tangent( const Eigen::Matrix3d rel_R, Eigen::Vector3d gamma1, Eigen::Vector3d gamma2, Eigen::Vector3d tangent1, Eigen::Vector3d tangent2 );
    
    std::pair<cv::Mat, cv::Mat> get_edge_patches(const Edge edge, const cv::Mat img, bool b_debug = false);
    std::pair<cv::Point2d, cv::Point2d> get_Orthogonal_Shifted_Points(const Edge edgel);
    std::pair<cv::Point2d, cv::Point2d> get_Orthogonal_Shifted_Points(const Edge edgel, double shift_magnitude);
    void get_patch_on_one_edge_side(cv::Point2d shifted_point, double theta, cv::Mat &patch_coord_x, cv::Mat &patch_coord_y, cv::Mat &patch_val, const cv::Mat img);
    double get_patch_similarity(const cv::Mat patch_one, const cv::Mat patch_two);

    void Display_Feature_Correspondences(cv::Mat Img1, cv::Mat Img2,
                                         std::vector<cv::KeyPoint> KeyPoint1, std::vector<cv::KeyPoint> KeyPoint2,
                                         std::vector<cv::DMatch> Good_Matches);

    Eigen::Vector3d two_view_linear_triangulation(
        const Eigen::Vector3d gamma1, const Eigen::Vector3d gamma2,
        const Eigen::Matrix3d K1, const Eigen::Matrix3d K2,
        const Eigen::Matrix3d Rel_R, const Eigen::Vector3d Rel_T);
    Eigen::Vector3d multiview_linear_triangulation(
        const int N, const std::vector<Eigen::Vector2d> pts,
        const std::vector<Eigen::Matrix3d> &Rs, const std::vector<Eigen::Vector3d> &Ts, const Eigen::Matrix3d K);

    std::string cvMat_Type(int type);
private:
    //> basis vectors in 3D space
    static Eigen::Vector3d e1;
    static Eigen::Vector3d e2;
    static Eigen::Vector3d e3;
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

    if (Q11.x < 0 || Q11.y < 0 || Q21.x >= meshGrid.cols || Q21.y >= meshGrid.rows ||
        Q12.x < 0 || Q12.y < 0 || Q22.x >= meshGrid.cols || Q22.y >= meshGrid.rows)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double f_x_y1 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * meshGrid.at<T>(Q11.y, Q11.x) + ((P.x - Q11.x) / (Q21.x - Q11.x)) * meshGrid.at<T>(Q21.y, Q21.x);
    double f_x_y2 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * meshGrid.at<T>(Q12.y, Q12.x) + ((P.x - Q11.x) / (Q21.x - Q11.x)) * meshGrid.at<T>(Q22.y, Q22.x);
    return ((Q12.y - P.y) / (Q12.y - Q11.y)) * f_x_y1 + ((P.y - Q11.y) / (Q12.y - Q11.y)) * f_x_y2;
}
/*
    Other version of Bilinear interpolation for a point P in a mesh grid.
    Returns NaN if the point is out of bounds.
*/
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
inline void util_compute_Img_Gradients(const cv::Mat I, cv::Mat &gx, cv::Mat &gy)
{
    // Utility util = Utility();
    // std::cout << "Image type: " << util.cvMat_Type(I.type()) << std::endl;

    // convert the image to 32F
    cv::Mat I_32F;
    I.convertTo(I_32F, CV_32F);
    cv::Sobel(I_32F, gx, CV_32F, 1, 0, 3, 1.0 / 8.0);
    cv::Sobel(I_32F, gy, CV_32F, 0, 1, 3, 1.0 / 8.0);
}

inline void util_make_rotated_patch_coords(const cv::Point2d &c, double theta, std::vector<cv::Point2d> &coords)
{
    coords.clear();
    coords.reserve(PATCH_SIZE * PATCH_SIZE);
    int half = PATCH_SIZE / 2;
    double ct = std::cos(theta);
    double st = std::sin(theta);
    for (int i = -half; i <= half; ++i)
    {
        for (int j = -half; j <= half; ++j)
        {
            coords.emplace_back(c.x + ct * i - st * j, c.y + st * i + ct * j);
        }
    }
}

inline float util_bilinear_Sample_F(const cv::Mat &I, double x, double y)
{
    int w = I.cols, h = I.rows;
    x = std::clamp(x, 0.0, (double)w - 1.0);
    y = std::clamp(y, 0.0, (double)h - 1.0);
    int x0 = (int)floor(x), y0 = (int)floor(y);
    int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    double a = x - x0, b = y - y0;
    float v00 = I.at<float>(y0, x0);
    float v10 = I.at<float>(y0, x1);
    float v01 = I.at<float>(y1, x0);
    float v11 = I.at<float>(y1, x1);
    return (1 - a) * (1 - b) * v00 + a * (1 - b) * v10 + (1 - a) * b * v01 + a * b * v11;
}

inline void util_sample_patch_at_coords(const cv::Mat &I, const std::vector<cv::Point2d> &coords, std::vector<double> &vals)
{
    vals.resize(coords.size());
    for (size_t k = 0; k < coords.size(); ++k)
    {
        vals[k] = static_cast<double>(util_bilinear_Sample_F(I, coords[k].x, coords[k].y));
    }
}

template <typename T>
inline T util_vector_mean(const std::vector<T> &v)
{
    T sum = 0;
    for (T x : v)
    {
        sum += x;
    }
    return sum / v.size();
}
/*
    Average computation for a vector of integers.
*/
inline double ComputeAverage(const std::vector<int> &values)
{
    if (values.empty())
        return 0.0;

    double sum = 0.0;
    for (int val : values)
    {
        sum += static_cast<double>(val);
    }

    return sum / values.size();
}

/*
    Pick a unique color based on the index and total number of colors.
    This function generates a color in the HSV color space and converts it to BGR.
*/
inline cv::Scalar PickUniqueColor(int index, int total)
{
    int hue = (index * 180) / total;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(color[0], color[1], color[2]);
}

/*
    NCC value computation between two patches.
*/
inline double ComputeNCC(const cv::Mat &patch_one, const cv::Mat &patch_two)
{
    double mean_one = (cv::mean(patch_one))[0];
    double mean_two = (cv::mean(patch_two))[0];
    double sum_of_squared_one = (cv::sum((patch_one - mean_one).mul(patch_one - mean_one))).val[0];
    double sum_of_squared_two = (cv::sum((patch_two - mean_two).mul(patch_two - mean_two))).val[0];

    cv::Mat norm_one = (patch_one - mean_one) / sqrt(sum_of_squared_one);
    cv::Mat norm_two = (patch_two - mean_two) / sqrt(sum_of_squared_two);
    return norm_one.dot(norm_two);
}

/*
    Build image pyramids for the current and next left/right images.
    The pyramids are built using OpenCV's buildPyramid function.
*/
inline void BuildImagePyramids(
    const cv::Mat &curr_left_image,
    const cv::Mat &curr_right_image,
    const cv::Mat &next_left_image,
    const cv::Mat &next_right_image,
    int num_levels,
    std::vector<cv::Mat> &curr_left_pyramid,
    std::vector<cv::Mat> &curr_right_pyramid,
    std::vector<cv::Mat> &next_left_pyramid,
    std::vector<cv::Mat> &next_right_pyramid)
{
    curr_left_pyramid.clear();
    curr_right_pyramid.clear();
    next_left_pyramid.clear();
    next_right_pyramid.clear();

    curr_left_pyramid.reserve(num_levels);
    curr_right_pyramid.reserve(num_levels);
    next_left_pyramid.reserve(num_levels);
    next_right_pyramid.reserve(num_levels);

    cv::buildPyramid(curr_left_image, curr_left_pyramid, num_levels - 1);
    cv::buildPyramid(curr_right_image, curr_right_pyramid, num_levels - 1);
    cv::buildPyramid(next_left_image, next_left_pyramid, num_levels - 1);
    cv::buildPyramid(next_right_image, next_right_pyramid, num_levels - 1);
}

/*
    Convert a 3x3 matrix represented as a vector of vectors to an Eigen::Matrix3d.
*/
inline Eigen::Matrix3d ConvertToEigenMatrix(const std::vector<std::vector<double>> &matrix)
{
    Eigen::Matrix3d eigen_matrix;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            eigen_matrix(i, j) = matrix[i][j];
        }
    }
    return eigen_matrix;
}

template <typename T>
T rad_to_deg(T theta)
{
    return theta * (180.0 / M_PI);
}

template <typename T>
T deg_to_rad(T theta)
{
    return theta * (M_PI / 180.0);
}

inline std::vector<int> find_Unique_Sorted_Numbers(std::vector<int> vec)
{
    std::vector<int> unique_sorted_vec = vec;
    std::sort(unique_sorted_vec.begin(), unique_sorted_vec.end());

    //> Move unique elements to the front
    auto last = std::unique(unique_sorted_vec.begin(), unique_sorted_vec.end());

    //> Erase the duplicate elements at the end
    unique_sorted_vec.erase(last, unique_sorted_vec.end());

    return unique_sorted_vec;
}

// wasn't used in the original code, but kept for reference

// /*
//     Linear triangulation of 3D points from confirmed matches, returning a vector of 3D points.
// */
// std::vector<cv::Point3d> LinearTriangulatePoints(
//     const std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>>& confirmed_matches
// ) {

//     std::vector<cv::Point3d> triangulated_points;

//     if (confirmed_matches.empty()) {
//         std::cerr << "WARNING: No confirmed matches to triangulate using linear method.\n";
//         return triangulated_points;
//     }

//     Eigen::Matrix3d K;
//     K << left_intr[0], 0, left_intr[2],
//          0, left_intr[1], left_intr[3],
//          0, 0, 1;

//     Eigen::Matrix3d R;
//     for (int i = 0; i < 3; ++i)
//         for (int j = 0; j < 3; ++j)
//             R(i, j) = rot_mat_21[i][j];

//     Eigen::Vector3d T(trans_vec_21[0], trans_vec_21[1], trans_vec_21[2]);

//     for (const auto& [left_edge, right_edge] : confirmed_matches) {
//         std::vector<Eigen::Vector2d> pts;
//         pts.emplace_back(left_edge.position.x, left_edge.position.y);
//         pts.emplace_back(right_edge.position.x, right_edge.position.y);

//         std::vector<Eigen::Vector2d> pts_meters;
//         for (const auto& pt : pts) {
//             Eigen::Vector3d homo_pt(pt.x(), pt.y(), 1.0);
//             Eigen::Vector3d pt_cam = K.inverse() * homo_pt;
//             pts_meters.emplace_back(pt_cam.x(), pt_cam.y());
//         }

//         Eigen::MatrixXd A(4, 4);

//         A.row(0) << 0.0, -1.0, pts_meters[0].y(), 0.0;
//         A.row(1) << 1.0,  0.0, -pts_meters[0].x(), 0.0;

//         Eigen::Matrix3d Rp = R;
//         Eigen::Vector3d Tp = T;
//         Eigen::Vector2d mp = pts_meters[1];

//         A.row(2) << mp.y() * Rp(2, 0) - Rp(1, 0),
//                     mp.y() * Rp(2, 1) - Rp(1, 1),
//                     mp.y() * Rp(2, 2) - Rp(1, 2),
//                     mp.y() * Tp.z()   - Tp.y();

//         A.row(3) << Rp(0, 0) - mp.x() * Rp(2, 0),
//                     Rp(0, 1) - mp.x() * Rp(2, 1),
//                     Rp(0, 2) - mp.x() * Rp(2, 2),
//                     Tp.x()   - mp.x() * Tp.z();

//         Eigen::Matrix4d ATA = A.transpose() * A;
//         Eigen::Vector4d gamma = ATA.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3);

//         if (std::abs(gamma(3)) > 1e-5) {
//             gamma /= gamma(3);
//             triangulated_points.emplace_back(gamma(0), gamma(1), gamma(2));
//         }
//     }

//     return triangulated_points;
// }

// /*
//     Calculate 3D points from confirmed matches using triangulation.
//     Returns a vector of 3D points.
// */
// std::vector<cv::Point3d> Calculate3DPoints(
//     const std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>>& confirmed_matches
// ) {
//     std::vector<cv::Point3d> points_3d;

//     if (confirmed_matches.empty()) {
//         std::cerr << "WARNING: No confirmed matches to triangulate.\n";
//         return points_3d;
//     }

//     cv::Mat K_left = (cv::Mat_<double>(3, 3) <<
//         left_intr[0], 0,            left_intr[2],
//         0,           left_intr[1], left_intr[3],
//         0,           0,            1);

//     cv::Mat K_right = (cv::Mat_<double>(3, 3) <<
//         right_intr[0], 0,             right_intr[2],
//         0,            right_intr[1], right_intr[3],
//         0,            0,             1);

//     cv::Mat R_left = cv::Mat::eye(3, 3, CV_64F);
//     cv::Mat T_left = cv::Mat::zeros(3, 1, CV_64F);

//     cv::Mat R_right(3, 3, CV_64F);
//     for (int i = 0; i < 3; ++i)
//         for (int j = 0; j < 3; ++j)
//             R_right.at<double>(i, j) = rot_mat_21[i][j];

//     cv::Mat T_right = (cv::Mat_<double>(3, 1) <<
//         trans_vec_21[0],
//         trans_vec_21[1],
//         trans_vec_21[2]);

//     cv::Mat extrinsic_left, extrinsic_right;
//     cv::hconcat(R_left, T_left, extrinsic_left);
//     cv::hconcat(R_right, T_right, extrinsic_right);

//     cv::Mat P_left = K_left * extrinsic_left;
//     cv::Mat P_right = K_right * extrinsic_right;

//     std::vector<cv::Point2f> points_left, points_right;
//     for (const auto& [left, right] : confirmed_matches) {
//         points_left.emplace_back(static_cast<cv::Point2f>(left.position));
//         points_right.emplace_back(static_cast<cv::Point2f>(right.position));
//     }

//     cv::Mat points_4d_homogeneous;
//     cv::triangulatePoints(P_left, P_right, points_left, points_right, points_4d_homogeneous);

//     int skipped = 0;
//     for (int i = 0; i < points_4d_homogeneous.cols; ++i) {
//         float w = points_4d_homogeneous.at<float>(3, i);
//         if (std::abs(w) > 1e-5) {
//             points_3d.emplace_back(
//                 points_4d_homogeneous.at<float>(0, i) / w,
//                 points_4d_homogeneous.at<float>(1, i) / w,
//                 points_4d_homogeneous.at<float>(2, i) / w
//             );
//         } else {
//             ++skipped;
//         }
//     }

//     int total = static_cast<int>(points_4d_homogeneous.cols);
//     if (skipped > 0.1 * total) {
//         std::cerr << "WARNING: " << skipped << " out of " << total
//                 << " triangulated points had near-zero depth (w ≈ 0) and were skipped.\n";
//     }

//     return points_3d;
// }

// /*
//     Calculate 3D orientations from confirmed matches.
//     Returns a vector of 3D tangent vectors.
// */
// std::vector<Eigen::Vector3d> Calculate3DOrientations(
//     const std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>>& confirmed_matches
// ) {
//     std::vector<Eigen::Vector3d> tangent_vectors;

//     if (confirmed_matches.empty()) {
//         std::cerr << "WARNING: No confirmed matches to compute 3D orientations.\n";
//         return tangent_vectors;
//     }

//     Eigen::Matrix3d K;
//     K << left_intr[0], 0, left_intr[2],
//          0, left_intr[1], left_intr[3],
//          0, 0, 1;

//     Eigen::Matrix3d R21;
//     for (int i = 0; i < 3; ++i)
//         for (int j = 0; j < 3; ++j)
//             R21(i, j) = rot_mat_21[i][j];

//     for (const auto& [left_edge, right_edge] : confirmed_matches) {
//         Eigen::Vector3d gamma1 = K.inverse() * Eigen::Vector3d(left_edge.position.x, left_edge.position.y, 1.0);
//         Eigen::Vector3d gamma2 = K.inverse() * Eigen::Vector3d(right_edge.position.x, right_edge.position.y, 1.0);

//         double theta1 = left_edge.orientation;
//         double theta2 = right_edge.orientation;

//         Eigen::Vector3d t1(std::cos(theta1), std::sin(theta1), 0.0);
//         Eigen::Vector3d t2(std::cos(theta2), std::sin(theta2), 0.0);

//         Eigen::Vector3d n1 = gamma1.cross(t1);
//         Eigen::Vector3d n2 = R21.transpose() * (t2.cross(gamma2));

//         Eigen::Vector3d T = n1.cross(n2);
//         T.normalize();

//         tangent_vectors.push_back(T);
//     }

//     return tangent_vectors;
// }

#endif