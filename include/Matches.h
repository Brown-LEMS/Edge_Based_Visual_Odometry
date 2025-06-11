#ifndef MATCHES_H
#define MATCHES_H

#include "Dataset.h"
#include "definitions.h"
#include "utility.h"
StereoMatchResult DisplayMatches(const cv::Mat &left_image, const cv::Mat &right_image, std::vector<cv::Point2d> right_edge_coords, std::vector<double> right_edge_orientations, Dataset &dataset);

EdgeMatchResult CalculateMatches(const std::vector<cv::Point2d> &selected_primary_edges, const std::vector<double> &selected_primary_orientations, const std::vector<cv::Point2d> &secondary_edge_coords,
                                 const std::vector<double> &secondary_edge_orientations, const std::vector<cv::Mat> &primary_patch_set_one, const std::vector<cv::Mat> &primary_patch_set_two, const std::vector<Eigen::Vector3d> &epipolar_lines_secondary,
                                 const cv::Mat &secondary_image, const std::vector<cv::Point2d> &selected_ground_truth_edges = std::vector<cv::Point2d>(),
                                 Dataset &dataset);

void ExtractPatches(
    int patch_size,
    const cv::Mat &image,
    const std::vector<cv::Point2d> &edges,
    const std::vector<double> &orientations,
    const std::vector<cv::Point2d> &shifted_one,
    const std::vector<cv::Point2d> &shifted_two,
    std::vector<cv::Point2d> &filtered_edges_out,
    std::vector<double> &filtered_orientations_out,
    std::vector<cv::Mat> &patch_set_one_out,
    std::vector<cv::Mat> &patch_set_two_out,
    const std::vector<cv::Point2d> *ground_truth_edges,
    std::vector<cv::Point2d> *filtered_gt_edges_out);

std::vector<Eigen::Vector3d> Dataset::CalculateEpipolarLine(const Eigen::Matrix3d &fund_mat, const std::vector<cv::Point2d> &edges);

std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> CalculateOrthogonalShifts(const std::vector<cv::Point2d> &edge_points, const std::vector<double> &orientations, double shift_magnitude);

#endif // MATCHES_H
