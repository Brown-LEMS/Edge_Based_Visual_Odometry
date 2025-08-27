#ifndef MATCHES_H
#define MATCHES_H

#include "definitions.h"
#include "utility.h"
#include "Dataset.h"

class Dataset;

struct StereoMatchResult;

struct EdgeMatchResult;

StereoMatchResult DisplayMatches(const cv::Mat &left_image, const cv::Mat &right_image,
                                 Dataset &dataset);

EdgeMatchResult CalculateMatches(const std::vector<Edge> &selected_primary_edges, const std::vector<Edge> &secondary_edges,
                                 const std::vector<cv::Mat> &primary_patch_set_one, const std::vector<cv::Mat> &primary_patch_set_two, const std::vector<Eigen::Vector3d> &epipolar_lines_secondary,
                                 const cv::Mat &secondary_image, Dataset &dataset, const std::vector<cv::Point2d> &selected_ground_truth_edges = std::vector<cv::Point2d>(), int image_pair_index = -1, bool forward_direction = true);

std::pair<cv::Point2d, cv::Point2d> get_Orthogonal_Shifted_Points(const Edge edgel);

void get_patch_on_one_edge_side(cv::Point2d shifted_point, double theta, cv::Mat &patch_coord_x, cv::Mat &patch_coord_y, cv::Mat &patch_val, const cv::Mat image);

double getNormalDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &epiline_x, double &epiline_y);

double getTangentialDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &x_intersection, double &y_intersection);

std::vector<Eigen::Vector3d> PerformEpipolarShift(
                                                const Eigen::Vector3d& edge1,
                                                const Eigen::MatrixXd& edge2,
                                                const Eigen::Vector3d& epip_coeffs);

std::vector<std::vector<Edge>> ClusterEpipolarShiftedEdges(std::vector<Edge> &valid_shifted_edges);

std::vector<std::pair<Edge, double>> ExtractEpipolarEdges(const Eigen::Vector3d &epipolar_line, const std::vector<Edge> &edges, double distance_threshold);

std::vector<Eigen::Vector3d> CalculateEpipolarLine(const Eigen::Matrix3d &fund_mat, const std::vector<Edge> &edges);

bool CheckEpipolarTangency(const Edge &primary_edge, const Eigen::Vector3d &epipolar_line);

bool FilterByEpipolarDistance(
    int &epi_true_positive,
    int &epi_false_negative,
    int &epi_true_negative,
    double &per_edge_epi_precision,
    int &epi_edges_evaluated,
    const std::vector<std::pair<Edge, double>> &secondary_edges,
    const std::vector<std::pair<Edge, double>> &test_secondary_edges,
    cv::Point2d &ground_truth_edge,
    double threshold,
    std::vector<double> &passing_distances);

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
    double threshold,
    bool forward_direction,
    int image_pair_index,
    std::ofstream &veridical_csv,
    std::ofstream &nonveridical_csv,
    const Edge& primary_edge,
    const Eigen::Vector3d& epipolar_line
);

void FilterByLowe(
    std::vector<std::vector<std::pair<Edge, EdgeMatch>>> &local_final_matches,
    std::vector<std::vector<int>> &local_lowe_input_counts,
    std::vector<std::vector<int>> &local_lowe_output_counts,
    std::vector<std::vector<cv::Point2d>> &local_GT_right_edges_after_lowe,
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

#endif
