#ifndef MATCHES_H
#define MATCHES_H

class Dataset;
struct StereoMatchResult;
struct EdgeMatchResult;

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "definitions.h"
#include "utility.h"
#include "Dataset.h"
#include "EdgeClusterer.h"

std::pair<cv::Point2d, cv::Point2d> get_Orthogonal_Shifted_Points(const Eigen::Vector3d edgel);
void get_patch_on_one_edge_side(cv::Point2d shifted_point, double theta,
                                cv::Mat &patch_coord_x, cv::Mat &patch_coord_y,
                                cv::Mat &patch_val, const cv::Mat img);
<<<<<<< HEAD
double get_similarity(const cv::Mat &patch_one, const cv::Mat &patch_two);
double edge_patch_similarity(const Edge &edge1, const Edge &edge2, cv::Mat gray_img_H1, cv::Mat gray_img_H2);
StereoMatchResult get_Stereo_Edge_Pairs(const cv::Mat &left_image, const cv::Mat &right_image, Dataset &dataset, int idx);
=======
double get_similarity(const cv::Mat patch_one, const cv::Mat patch_two);
double edge_patch_similarity(const Edge target_edge_H1, const Edge target_edge_H2, const cv::Mat gray_img_H1, const cv::Mat gray_img_H2);
StereoMatchResult get_Stereo_Edge_Pairs(const cv::Mat &left_image, const cv::Mat &right_image, StereoEdgeCorrespondencesGT prev_stereo_frame, Dataset &dataset, int idx);
>>>>>>> copilot/vscode1760022595288
EdgeMatchResult CalculateMatches(const std::vector<Edge> &selected_primary_edges, const std::vector<Edge> &secondary_edges,
                                 const std::vector<cv::Mat> &primary_patch_set_one, const std::vector<cv::Mat> &primary_patch_set_two, const std::vector<Eigen::Vector3d> &epipolar_lines_secondary,
                                 const cv::Mat &secondary_image, Dataset &dataset, const std::vector<cv::Point2d> &selected_ground_truth_edges = std::vector<cv::Point2d>());

//> The following functions are newly added
void get_Stereo_Edge_GT_Pairs(Dataset &dataset, StereoEdgeCorrespondencesGT &stereo_frame, const std::vector<Edge> &right_edges, bool is_left);
std::pair<std::vector<int>, int> get_right_edges_close_to_GT_location(Edge target_left_edge, const cv::Point2d GT_location, const std::vector<Edge> constrained_right_edges, const std::vector<Edge> right_edges, const double dist_tol = 1.0);
void augment_Edge_Data(StereoEdgeCorrespondencesGT &stereo_frame, const cv::Mat image);

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
std::vector<EdgeCluster> ClusterEpipolarShiftedEdges(std::vector<Edge> &valid_shifted_edges);

std::vector<int> ExtractEpipolarEdges(const Eigen::Vector3d &epipolar_line, const std::vector<Edge> &edges, const double dist_tol = 0.5);

std::vector<Eigen::Vector3d> CalculateEpipolarLine(const Eigen::Matrix3d &fund_mat, const std::vector<Edge> &edges, const std::vector<int> &edge_indices);

std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> CalculateOrthogonalShifts(const std::vector<Edge> &edge_points, double shift_magnitude, Dataset &dataset);

bool CheckEpipolarTangency(const Edge &primary_edge, const Eigen::Vector3d &epipolar_line);

bool FilterByEpipolarDistance(
    int &epi_true_positive,
    int &epi_false_negative,
    int &epi_true_negative,
    double &per_edge_epi_precision,
    int &epi_edges_evaluated,
    const std::vector<Edge> &secondary_edges,
    const std::vector<Edge> &test_secondary_edges,
    cv::Point2d &ground_truth_edge,
    double threshold);

void FilterByDisparity(
    std::vector<Edge> &filtered_secondary_edges,
    const std::vector<Edge> &edge_candidates,
    bool gt,
    const Edge &primary_edge);

template <typename Container>
void RecallUpdate(int &true_positive,
                  int &false_negative,
                  int &edges_evaluated,
                  double &per_edge_precision,
                  const Container &output_candidates,
                  cv::Point2d &ground_truth_edge,
                  double threshold);
void FormClusterCenters(
    std::vector<EdgeCluster> &cluster_centers,
    std::vector<std::vector<Edge>> &clusters);

void EpipolarShiftFilter(
    const std::vector<Edge> &filtered_edges,
    std::vector<Edge> &shifted_edges,
    const Eigen::Vector3d &epipolar_line,
    Utility util = Utility());

void FilterByNCC(
    const cv::Mat &primary_patch_one,
    const cv::Mat &primary_patch_two,
    const std::vector<cv::Mat> &secondary_patch_set_one,
    const std::vector<cv::Mat> &secondary_patch_set_two,
    const cv::Point2d &ground_truth_edge,
    std::vector<EdgeMatch> &passed_ncc_matches,
    std::vector<EdgeCluster> &filtered_cluster_centers,
    bool gt,
    int &ncc_true_positive,
    int &ncc_false_negative,
    double &per_edge_ncc_precision,
    int &ncc_edges_evaluated,
    double threshold

);

void FilterByLowe(
    std::vector<std::vector<std::pair<Edge, EdgeMatch>>> &local_final_matches,
    std::vector<std::vector<int>> &local_lowe_input_counts,
    std::vector<std::vector<int>> &local_lowe_output_counts,
    std::vector<std::vector<cv::Point2d>> &local_GT_right_edges_after_lowe,
    std::vector<std::unordered_map<Edge, std::vector<Edge>>> &local_final_lowe_matches,
    std::vector<std::unordered_map<Edge, std::vector<Edge>>> &local_final_reverse_lowe_matches,
    int thread_id,
    const std::vector<EdgeMatch> &passed_ncc_matches,
    bool gt,
    const Edge &primary_edge,
    cv::Point2d &ground_truth_edge,
    int &lowe_true_positive,
    int &lowe_false_negative,
    double &per_edge_lowe_precision,
    int &lowe_edges_evaluated,
    double threshold);
#endif // MATCHES_H
