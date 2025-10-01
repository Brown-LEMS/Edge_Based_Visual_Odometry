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
#include "Stereo_Iterator.h"

// =======================================================================================================
// class Dataset: Fetch data from dataset specified in the configuration file
//
// ChangeLogs
//    Jue    25-06-17    Modified data structure for readability.
//    Lopez  25-01-26    Modified for euroc dataset support.
//    Chien  23-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu), Saul Lopez Lucas (saul_lopez_lucas@brown.edu), Jue Han (jhan192@brown.edu)
// =======================================================================================================

struct FileInfo
{
    std::string dataset_type;
    std::string dataset_path;
    std::string output_path;
    std::string sequence_name;
    std::string GT_file_name;

    bool has_gt = false; //> whether the dataset has ground truth or not

    std::vector<double> GT_time_stamps;
    std::vector<double> Img_time_stamps;
};

struct Camera
{
    std::vector<int> resolution;    // [width, height]
    std::vector<double> intrinsics; // fx, fy, cx, cy
    std::vector<double> distortion; // distortion coefficients
    Eigen::Matrix3d R;              // rotation (to stereo)
    Eigen::Vector3d T;              // translation (to stereo)
    Eigen::Matrix3d F;              // fundamental matrix
    Eigen::Matrix3d K;              //> Calibration matrix
};

struct CameraInfo
{
    Camera left;
    Camera right;
    Eigen::Matrix3d rot_frame2body_left; // From cam to body
    Eigen::Vector3d transl_frame2body_left;
    double focal_length;
    double baseline;
};

struct EdgeCluster
{
    Edge center_edge;
    std::vector<Edge> contributing_edges;
};

struct EdgeMatch
{
    Edge edge;
    double final_score;

    std::vector<Edge> contributing_edges;
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

struct EdgeMatchResult
{
    RecallMetrics recall_metrics;
    std::vector<std::pair<Edge, EdgeMatch>> edge_to_cluster_matches;
};

struct BidirectionalMetrics
{
    int matches_before_bct;
    int matches_after_bct;
    double per_image_bct_recall;
    double per_image_bct_precision;
    double per_image_bct_time;
};

struct StereoMatchResult
{
    EdgeMatchResult forward_match;
    EdgeMatchResult reverse_match;
    std::vector<std::pair<Edge, Edge>> confirmed_matches;
    BidirectionalMetrics bidirectional_metrics;
};

//> This struct defines a pool of GT edge correspondences in a stereo image pair
struct StereoEdgeCorrespondencesGT
{
    std::vector<Edge> left_edges;                           //> left edges
    std::vector<cv::Point2d> GT_locations_from_disparity;   //> GT location on the right image from the left edge and the GT disparity
    std::vector<std::vector<Edge>> GT_right_edges;          //> A set of right edges that are "very close" to the GT location from disparity
    std::vector<Eigen::Vector3d> Gamma_in_left_cam_coord;   //> 3D points under the left camera coordinate
    // std::unordered_map<int, cv::Mat> edge_SIFT_descriptors; //> SIFT descriptors of all left edges
    std::vector<cv::Mat> left_edge_descriptors;
    std::vector<int> grid_indices;

    bool b_is_size_consistent() 
    { 
        return left_edges.size() == GT_locations_from_disparity.size() && left_edges.size() == GT_right_edges.size() && left_edges.size() == Gamma_in_left_cam_coord.size() && left_edges.size() == left_edge_descriptors.size(); 
    }

    void print_size_consistency()
    {
        std::cout << "The sizes of the StereoEdgeCorrespondencesGT are not consistent!" << std::endl;
        std::cout << "- Size of the left_edges = " << left_edges.size() << std::endl;
        std::cout << "- Size of the GT_locations_from_disparity = " << GT_locations_from_disparity.size() << std::endl;
        std::cout << "- Size of the GT_right_edges = " << GT_right_edges.size() << std::endl;
        std::cout << "- Size of the Gamma_in_left_cam_coord = " << Gamma_in_left_cam_coord.size() << std::endl;
        std::cout << "- Size of the left_edge_descriptors = " << left_edge_descriptors.size() << std::endl;
    }
};

struct Keyframe_CurrentFrame_EdgeCorrespondencesGT
{
    StereoEdgeCorrespondencesGT last_keyframe;
    StereoEdgeCorrespondencesGT current_frame;

    //>>>>>>>>>> This block is a pool of GT data >>>>>>>>>
    std::vector<cv::Point2d> GT_locations_on_current_image;
    std::vector<std::vector<Edge>> GT_current_edges;
    // std::vector<std::vector<cv::Mat>> GT_current_edge_descriptors; //> currently not in use
    std::vector<int> GT_pair_indices_for_last_keyframe;
    //>>>>>>>>>> This block is a pool of GT data >>>>>>>>>

    //>>>>>>>>>> This block is a pool of edge matching data >>>>>>>>>
    std::vector<std::vector<Edge>> matching_current_edges;
    //>>>>>>>>>> This block is a pool of edge matching data >>>>>>>>>
};

struct KF_CF_Edge_Correspondences
{
    //> Maybe use index instead of the edge itself?
    int kf_edge_index;

    //>>>>>>>>>> This block is a pool of GT data >>>>>>>>>
    cv::Point2d gt_location_on_cf;              // the ground truth correspondence edge in the current frame
    std::vector<int> veridical_cf_edges_indices;       // corresponding vertical edges in the current frame

    //> FIXME: This should be obtained from the stereo edge matching result
    // std::vector<Edge> veridical_stereo_right_edges_for_kf;                  // corresponding edges in the stereo frame for the keyframe
    // std::vector<std::vector<Edge>> veridical_stereo_right_edges_for_cf;     // corresponding edges in the stereo frame for the current frame
    //>>>>>>>>>> This block is a pool of GT data >>>>>>>>>

    std::vector<int> matching_cf_edges_indices;       // corresponding edge indices in the current frame
};

extern cv::Mat merged_visualization_global;
class Dataset
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(YAML::Node);
    std::unique_ptr<StereoIterator> stereo_iterator;

    void load_dataset(const std::string &dataset_type, std::vector<cv::Mat> &left_ref_disparity_maps, int num_pairs);

    std::vector<Edge> left_edges;
    std::vector<Edge> right_edges;

    // should we make it edge pairs?
    std::vector<std::tuple<cv::Point2d, cv::Point2d, double>> forward_gt_data;
    std::vector<std::tuple<cv::Point2d, cv::Point2d, double>> reverse_gt_data;

    std::vector<std::pair<double, double>> ncc_one_vs_err;
    std::vector<std::pair<double, double>> ncc_two_vs_err;

    std::vector<cv::Point2d> ground_truth_right_edges_after_lowe;

    // getters
    bool has_gt() { return file_info.has_gt; };

    Eigen::Matrix3d get_fund_mat_21() { return camera_info.left.F; };
    Eigen::Matrix3d get_fund_mat_12() { return camera_info.right.F; };

    std::string get_dataset_type() { return file_info.dataset_type; }
    std::string get_output_path() { return file_info.output_path; }
    int get_omp_threads() const { return omp_threads; }

    unsigned get_num_imgs() { return Total_Num_Of_Imgs; };
    int get_height() { return img_height; };
    int get_width() { return img_width; };

    double get_focal_length() { return camera_info.focal_length; };
    double get_left_focal_length() { return camera_info.left.intrinsics[0]; };
    double get_right_focal_length() { return camera_info.right.intrinsics[0]; };
    Eigen::Matrix3d get_left_calib_matrix() { return camera_info.left.K; }
    Eigen::Matrix3d get_right_calib_matrix() { return camera_info.right.K; }
    double get_left_baseline() { return camera_info.left.T[0]; };
    double get_right_baseline() { return camera_info.right.T[0]; };
    double get_baseline() { return camera_info.baseline; };
    Eigen::Matrix3d get_relative_rot_left_to_right() { return camera_info.left.R; }
    Eigen::Vector3d get_relative_transl_left_to_right() { return camera_info.left.T; }
    Eigen::Matrix3d get_relative_rot_right_to_left() { return camera_info.right.R; }
    Eigen::Vector3d get_relative_transl_right_to_left() { return camera_info.right.T; }

    std::vector<double> left_intr() { return camera_info.left.intrinsics; };
    std::vector<double> right_intr() { return camera_info.right.intrinsics; };
    std::vector<double> left_dist_coeffs() { return camera_info.left.distortion; };
    std::vector<double> right_dist_coeffs() { return camera_info.right.distortion; };

    // setters
    void increment_num_imgs() { Total_Num_Of_Imgs++; };
    void set_height(int height) { img_height = height; };
    void set_width(int width) { img_width = width; };

private:
    YAML::Node config_file;
    int omp_threads;
    // file info
    FileInfo file_info;
    // camera info
    CameraInfo camera_info;
    // Images info
    unsigned Total_Num_Of_Imgs;
    int img_height, img_width;

    // didn't find this:
    double max_disparity;

    // functions
    void PrintDatasetInfo();

    std::vector<std::pair<cv::Mat, cv::Mat>> LoadEuRoCImages(const std::string &csv_path, const std::string &left_path, const std::string &right_path, int num_images);

    std::vector<std::pair<cv::Mat, cv::Mat>> LoadETH3DImages(const std::string &stereo_pairs_path, int num_images);

    //    std::vector<double> LoadMaximumDisparityValues(const std::string& stereo_pairs_path, int num_images);

    std::vector<cv::Mat> LoadETH3DLeftReferenceMaps(const std::string &stereo_pairs_path, int num_maps);

    //    std::vector<cv::Mat> LoadETH3DRightReferenceMaps(const std::string &stereo_pairs_path, int num_maps);

    void WriteDisparityToBinary(const std::string &filepath, const cv::Mat &disparity_map);

    cv::Mat ReadDisparityFromBinary(const std::string &filepath);

    cv::Mat LoadDisparityFromCSV(const std::string &path);

    void CalculateGTLeftEdge(const std::vector<cv::Point2d> &right_third_order_edges_locations, const std::vector<double> &right_third_order_edges_orientation, const cv::Mat &disparity_map_right_reference, const cv::Mat &left_image, const cv::Mat &right_image);

    void Load_GT_Poses(std::string GT_Poses_File_Path);

    void Align_Images_and_GT_Poses();

    cv::Mat Small_Patch_Radius_Map;
    Utility::Ptr utility_tool = nullptr;
};
// struct CameraInfo{
//     Eigen::Matrix3d rot_frame2body_left;
//     Eigen::Vector3d transl_frame2body_left;
//     //left:
//     std::vector<int> left_res;
//     std::vector<double> left_intr;
//     std::vector<double> left_dist_coeffs;

//     //Left to right image, will become R,T,F
//     std::vector<std::vector<double>> rot_mat_21;
//     std::vector<double> trans_vec_21;
//     std::vector<std::vector<double>> fund_mat_21;
//     //right:
//     std::vector<int> right_res;
//     std::vector<double> right_intr;
//     std::vector<double> right_dist_coeffs;
//     //Right to left, will be come R, T, F too.
//     std::vector<std::vector<double>> rot_mat_12;
//     std::vector<double> trans_vec_12;
//     std::vector<std::vector<double>> fund_mat_12;
//     //some other info.
//     double focal_length;
//     double baseline;
// };
#endif