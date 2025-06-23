#ifndef MATCHES_H
#define MATCHES_H

class Dataset;
struct StereoMatchResult;
struct EdgeMatchResult;

#include "definitions.h"
#include "utility.h"
#include "Dataset.h"

StereoMatchResult DisplayMatches(const cv::Mat &left_image, const cv::Mat &right_image,
                                 Dataset &dataset);
EdgeMatchResult CalculateMatches(const std::vector<Edge> &selected_primary_edges, const std::vector<Edge> &secondary_edges,
                                 const std::vector<cv::Mat> &primary_patch_set_one, const std::vector<cv::Mat> &primary_patch_set_two, const std::vector<Eigen::Vector3d> &epipolar_lines_secondary,
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
// clear
void ExtractPatches(
    int patch_size,
    const cv::Mat &image,
    const std::vector<Edge> &edges,
    const std::vector<cv::Point2d> &shifted_one,
    const std::vector<cv::Point2d> &shifted_two,
    std::vector<Edge> &filtered_edges_out,
    std::vector<cv::Mat> &patch_set_one_out,
    std::vector<cv::Mat> &patch_set_two_out,
    const std::vector<cv::Point2d> *ground_truth_edges,
    std::vector<cv::Point2d> *filtered_gt_edges_out);

// checked
Edge PerformEpipolarShift(
    Edge original_edge,
    std::vector<double> epipolar_line_coeffs, bool &b_pass_epipolar_tengency_check);
// checked
std::vector<std::vector<Edge>> ClusterEpipolarShiftedEdges(std::vector<Edge> &valid_shifted_edges);

std::vector<Edge> ExtractEpipolarEdges(const Eigen::Vector3d &epipolar_line, const std::vector<Edge> &edges, double distance_threshold);

std::vector<Eigen::Vector3d> CalculateEpipolarLine(const Eigen::Matrix3d &fund_mat, const std::vector<Edge> &edges);

// clear
std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> CalculateOrthogonalShifts(const std::vector<Edge> &edge_points, double shift_magnitude, Dataset &dataset);

#endif // MATCHES_H
