#pragma once
#ifndef DATASET_H
#define DATASET_H
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "definitions.h"
#include "Frame.h"
#include "utility.h"
#include "./toed/cpu_toed.hpp"

// =======================================================================================================
// class Dataset: Fetch data from dataset specified in the configuration file
//
// ChangeLogs
//    Lopez  25-01-26    Modified for euroc dataset support.
//    Chien  23-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu), Saul Lopez Lucas (saul_lopez_lucas@brown.edu)
// =======================================================================================================

struct EdgeCluster
{
    cv::Point2d center_coord;
    double center_orientation;

    std::vector<cv::Point2d> contributing_edges;
    std::vector<double> contributing_orientations;
};

struct EdgeMatch
{
    cv::Point2d coord;
    double orientation;
    double final_score;

    std::vector<cv::Point2d> contributing_edges;
    std::vector<double> contributing_orientations;
};

struct RecallMetrics
{
    double epi_distance_recall;
    double max_disparity_recall;
    double epi_shift_recall;
    double epi_cluster_recall;
    double ncc_recall;
    double lowe_recall;

    std::vector<int> epi_input_counts;
    std::vector<int> epi_output_counts;
    std::vector<int> disp_input_counts;
    std::vector<int> disp_output_counts;
    std::vector<int> shift_input_counts;
    std::vector<int> shift_output_counts;
    std::vector<int> clust_input_counts;
    std::vector<int> clust_output_counts;
    std::vector<int> patch_input_counts;
    std::vector<int> patch_output_counts;
    std::vector<int> ncc_input_counts;
    std::vector<int> ncc_output_counts;
    std::vector<int> lowe_input_counts;
    std::vector<int> lowe_output_counts;

    double per_image_epi_precision;
    double per_image_disp_precision;
    double per_image_shift_precision;
    double per_image_clust_precision;
    double per_image_ncc_precision;
    double per_image_lowe_precision;

    int lowe_true_positive;
    int lowe_false_negative;

    double per_image_epi_time;
    double per_image_disp_time;
    double per_image_shift_time;
    double per_image_clust_time;
    double per_image_patch_time;
    double per_image_ncc_time;
    double per_image_lowe_time;
    double per_image_total_time;
};

struct SourceEdge
{
    cv::Point2d position;
    double orientation;
};

struct EdgeMatchResult
{
    RecallMetrics recall_metrics;
    std::vector<std::pair<SourceEdge, EdgeMatch>> edge_to_cluster_matches;
};

struct BidirectionalMetrics
{
    int matches_before_bct;
    int matches_after_bct;
    double per_image_bct_recall;
    double per_image_bct_precision;
    double per_image_bct_time;
};

struct ConfirmedMatchEdge
{
    cv::Point2d position;
    double orientation;
};

struct StereoMatchResult
{
    EdgeMatchResult forward_match;
    EdgeMatchResult reverse_match;
    std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>> confirmed_matches;
    BidirectionalMetrics bidirectional_metrics;
};

struct OrientedPoint3D
{
    cv::Point3d position;
    Eigen::Vector3d orientation;
};

extern cv::Mat merged_visualization_global;
class Dataset
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(YAML::Node, bool);

    static void onMouse(int event, int x, int y, int, void *);

    unsigned Total_Num_Of_Imgs;
    int img_height, img_width;

    std::vector<cv::Point2d> left_third_order_edges_locations;
    std::vector<double> left_third_order_edges_orientation;
    std::vector<cv::Point2d> right_third_order_edges_locations;
    std::vector<double> right_third_order_edges_orientation;

    std::vector<Eigen::Matrix3d> unaligned_GT_Rot;
    std::vector<Eigen::Vector3d> unaligned_GT_Transl;
    std::vector<Eigen::Matrix3d> aligned_GT_Rot;
    std::vector<Eigen::Vector3d> aligned_GT_Transl;
    std::vector<std::tuple<cv::Point2d, cv::Point2d, double>> forward_gt_data;
    std::vector<std::tuple<cv::Point2d, cv::Point2d, double>> reverse_gt_data;

    std::vector<std::pair<double, double>> ncc_one_vs_err;
    std::vector<std::pair<double, double>> ncc_two_vs_err;

    std::vector<cv::Point2d> ground_truth_right_edges_after_lowe;

private:
    YAML::Node config_file;
    int omp_threads;

    std::string dataset_type;
    std::string dataset_path;
    std::string output_path;
    std::string sequence_name;

    std::string GT_file_name;

    Eigen::Matrix3d rot_frame2body_left;
    Eigen::Vector3d transl_frame2body_left;

    std::vector<int> left_res;
    int left_rate;
    std::string left_model;
    std::vector<double> left_intr;
    std::string left_dist_model;
    std::vector<double> left_dist_coeffs;

    std::vector<int> right_res;
    int right_rate;
    std::string right_model;
    std::vector<double> right_intr;
    std::string right_dist_model;
    std::vector<double> right_dist_coeffs;

    std::vector<std::vector<double>> rot_mat_21;
    std::vector<double> trans_vec_21;
    std::vector<std::vector<double>> fund_mat_21;

    std::vector<std::vector<double>> rot_mat_12;
    std::vector<double> trans_vec_12;
    std::vector<std::vector<double>> fund_mat_12;
    double focal_length;
    double baseline;
    double max_disparity;

    void PrintDatasetInfo();

    std::vector<std::pair<cv::Mat, cv::Mat>> LoadEuRoCImages(const std::string &csv_path, const std::string &left_path, const std::string &right_path, int num_images);

    std::vector<std::pair<cv::Mat, cv::Mat>> LoadETH3DImages(const std::string &stereo_pairs_path, int num_images);

    //    std::vector<double> LoadMaximumDisparityValues(const std::string& stereo_pairs_path, int num_images);

    std::vector<cv::Mat> LoadETH3DLeftReferenceMaps(const std::string &stereo_pairs_path, int num_maps);

    //    std::vector<cv::Mat> LoadETH3DRightReferenceMaps(const std::string &stereo_pairs_path, int num_maps);

    void WriteDisparityToBinary(const std::string &filepath, const cv::Mat &disparity_map);

    cv::Mat ReadDisparityFromBinary(const std::string &filepath);

    cv::Mat LoadDisparityFromCSV(const std::string &path);

    void WriteEdgesToBinary(const std::string &filepath,
                            const std::vector<cv::Point2d> &locations,
                            const std::vector<double> &orientations);

    void ReadEdgesFromBinary(const std::string &filepath,
                             std::vector<cv::Point2d> &locations,
                             std::vector<double> &orientations);

    void ProcessEdges(const cv::Mat &image,
                      const std::string &filepath,
                      std::shared_ptr<ThirdOrderEdgeDetectionCPU> &toed,
                      std::vector<cv::Point2d> &locations,
                      std::vector<double> &orientations);

    void VisualizeGTRightEdge(const cv::Mat &left_image, const cv::Mat &right_image, const std::vector<std::pair<cv::Point2d, cv::Point2d>> &edge_pairs);

    void CalculateGTRightEdge(const std::vector<cv::Point2d> &left_third_order_edges_locations, const std::vector<double> &left_third_order_edges_orientation, const cv::Mat &disparity_map, const cv::Mat &left_image, const cv::Mat &right_image);

    void CalculateGTLeftEdge(const std::vector<cv::Point2d> &right_third_order_edges_locations, const std::vector<double> &right_third_order_edges_orientation, const cv::Mat &disparity_map_right_reference, const cv::Mat &left_image, const cv::Mat &right_image);

    cv::Point2d PerformEpipolarShift(cv::Point2d original_edge_location, double edge_orientation, std::vector<double> epipolar_line_coeffs, bool &b_pass_epipolar_tengency_check);

    void Load_GT_Poses(std::string GT_Poses_File_Path);

    std::vector<double> GT_time_stamps;

    std::vector<double> Img_time_stamps;

    void Align_Images_and_GT_Poses();

    bool compute_grad_depth = false;
    cv::Mat Gx_2d, Gy_2d;
    cv::Mat Small_Patch_Radius_Map;
    Utility::Ptr utility_tool = nullptr;
};

#endif