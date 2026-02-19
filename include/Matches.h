#ifndef MATCHES_H
#define MATCHES_H

class Dataset;
struct StereoMatchResult;
struct EdgeMatchResult;

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <random>

#include "definitions.h"
#include "utility.h"
#include "Dataset.h"
#include "EdgeClusterer.h"
#include "io.h"

std::pair<cv::Point2d, cv::Point2d> get_Orthogonal_Shifted_Points(const Edge edgel);
void get_patch_on_one_edge_side(cv::Point2d shifted_point, double theta,
                                cv::Mat &patch_coord_x, cv::Mat &patch_coord_y,
                                cv::Mat &patch_val, const cv::Mat img);
std::pair<cv::Mat, cv::Mat> get_edge_patches(const Edge edge, const cv::Mat img, bool b_debug = false);
double get_patch_similarity(const cv::Mat patch_one, const cv::Mat patch_two);
double get_similarity(const cv::Mat &patch_one, const cv::Mat &patch_two);
double edge_patch_similarity(const Edge &edge1, const Edge &edge2, const cv::Mat &gray_img_H1, const cv::Mat &gray_img_H2);

//> ========== START OF CH'S EDITIONS ==========
//> The following functions are newly added
void Find_Stereo_GT_Locations(Dataset &dataset, const cv::Mat left_disparity_map, const StereoFrame &stereo_frame, Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left);
void Find_Stereo_GT_Locations(Dataset &dataset, const std::vector<double> &edge_disparities, const StereoFrame &stereo_frame, Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left);

void get_Stereo_Edge_GT_Pairs(Dataset &dataset, const StereoFrame &stereo_frame, Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left);
std::vector<int> get_right_edge_indices_close_to_GT_location(const StereoFrame &stereo_frame, const cv::Point2d GT_location, double GT_orientation, const std::vector<int> right_candidate_edge_indices, const double dist_tol, const double orient_tol, bool is_left = true);
void record_Ambiguity_Distribution(const std::string &stage_name, const Stereo_Edge_Pairs &stereo_frame_edge_pairs, const std::string &output_dir, size_t frame_idx);

std::vector<int> extract_Epipolar_Edge_Indices(const Eigen::Vector3d &epipolar_line, const std::vector<Edge> &edges, const double dist_tol);

struct Stage_Metrics
{
    double recall;
    double precision;
    double precision_pair;
    double ambiguity;
};

struct Frame_Evaluation_Metrics
{
    std::map<std::string, Stage_Metrics> stages;
};

Frame_Evaluation_Metrics get_Stereo_Edge_Pairs(Dataset &dataset, Stereo_Edge_Pairs &stereo_frame_edge_pairs, size_t frame_idx);
void augment_Edge_Data(Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left);
void augment_all_Edge_Data(Stereo_Edge_Pairs &stereo_frame_edge_pairs, std::vector<std::pair<cv::Mat, cv::Mat>> &edge_descriptors, bool is_left);
void construct_candidate_set(Stereo_Edge_Pairs &stereo_frame_edge_pairs, std::vector<Edge> &candidate_edge_set);

void record_Filter_Distribution(const std::string &filter_name, const std::vector<double> &filter_values, const std::vector<int> &is_veridical, const std::string &output_dir, size_t frame_idx = 0);
void apply_NCC_Filtering(Stereo_Edge_Pairs &stereo_frame_edge_pairs, std::vector<std::vector<EdgeCluster>> &test_false_negative_edge_clusters,
                         std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> &test_false_negative_matching_edge_patches,
                         std::vector<int> &left_edge_indices_to_false_negatives,
                         const std::string &output_dir = "", size_t frame_idx = 0, bool is_left = true);

void write_Stereo_Edge_Pairs_to_file(Dataset &dataset, Stereo_Edge_Pairs &stereo_frame_edge_pairs, int frame_idx);

void record_correspondences_for_visualization(const Stereo_Edge_Pairs &stereo_frame_edge_pairs,
                                              const std::string &output_dir,
                                              size_t frame_idx,
                                              int num_samples = 10);

//> ========== END OF CH'S EDITIONS ==========

std::vector<Eigen::Vector3d> CalculateEpipolarLine(const Eigen::Matrix3d &fund_mat, const std::vector<Edge> &edges);

#endif // MATCHES_H
