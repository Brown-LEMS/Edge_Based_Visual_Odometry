#ifndef MATCHES_H
#define MATCHES_H

class Dataset;
struct StereoMatchResult;
struct EdgeMatchResult;

#include "definitions.h"
#include "utility.h"
#include "Dataset.h"

StereoMatchResult DisplayMatches(const cv::Mat &left_image, const cv::Mat &right_image,
                                 std::vector<cv::Point2d> right_edge_coords,
                                 std::vector<double> right_edge_orientations,
                                 Dataset &dataset);
EdgeMatchResult CalculateMatches(const std::vector<cv::Point2d> &selected_primary_edges, const std::vector<double> &selected_primary_orientations, const std::vector<cv::Point2d> &secondary_edge_coords,
                                 const std::vector<double> &secondary_edge_orientations, const std::vector<cv::Mat> &primary_patch_set_one, const std::vector<cv::Mat> &primary_patch_set_two, const std::vector<Eigen::Vector3d> &epipolar_lines_secondary,
                                 const cv::Mat &secondary_image, Dataset &dataset, const std::vector<cv::Point2d> &selected_ground_truth_edges = std::vector<cv::Point2d>());
void ExtractClusterPatches(
    int patch_size,
    const cv::Mat &image,
    const std::vector<EdgeCluster> &cluster_centers,
    const std::vector<cv::Point2d> *right_edges,
    const std::vector<cv::Point2d> &shifted_one,
    const std::vector<cv::Point2d> &shifted_two,
    std::vector<EdgeCluster> &cluster_centers_out,
    std::vector<cv::Point2d> *filtered_right_edges_out,
    std::vector<cv::Mat> &patch_set_one_out,
    std::vector<cv::Mat> &patch_set_two_out);

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

cv::Point2d PerformEpipolarShift(
    cv::Point2d original_edge_location,
    double edge_orientation,
    std::vector<double> epipolar_line_coeffs,
    bool &b_pass_epipolar_tengency_check);

std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> ClusterEpipolarShiftedEdges(std::vector<cv::Point2d> &valid_shifted_edges, std::vector<double> &valid_shifted_orientations);

std::pair<std::vector<cv::Point2d>, std::vector<double>> ExtractEpipolarEdges(const Eigen::Vector3d &epipolar_line, const std::vector<cv::Point2d> &edge_locations, const std::vector<double> &edge_orientations, double distance_threshold);

std::vector<Eigen::Vector3d> CalculateEpipolarLine(const Eigen::Matrix3d &fund_mat, const std::vector<cv::Point2d> &edges);

std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> CalculateOrthogonalShifts(const std::vector<cv::Point2d> &edge_points, const std::vector<double> &orientations, double shift_magnitude, Dataset &dataset);

#endif // MATCHES_H
