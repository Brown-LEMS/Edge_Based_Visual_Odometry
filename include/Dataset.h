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
struct Edge3D
{
    Eigen::Vector3d location;
    Eigen::Vector3d orientation; // unit 3D vector T

    bool b_isEmpty;   //> check if this struct is value-assigned
    int frame_source; //> which frame this edge is constructed from
    int index;        //> index of the edge in the original edge list
    Edge3D() : location(Eigen::Vector3d(-1.0, -1.0, -1.0)), orientation(Eigen::Vector3d(-100, -100, -100)), b_isEmpty(true), frame_source(-1), index(-1) {}
    Edge3D(Eigen::Vector3d location, Eigen::Vector3d orientation, bool b_isEmpty, int frame_source) : location(location), orientation(orientation), b_isEmpty(b_isEmpty), frame_source(frame_source) {}

    bool operator==(const Edge3D &other) const
    {

        return location.x() == other.location.x() &&
               location.y() == other.location.y() &&
               location.z() == other.location.z() &&
               orientation.x() == other.orientation.x() &&
               orientation.y() == other.orientation.y() &&
               orientation.z() == other.orientation.z() &&
               b_isEmpty == other.b_isEmpty;
    }
};

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
    std::vector<int> focused_edges;                       //> the edges index we consider for matching
    std::vector<cv::Point2d> GT_locations_from_disparity; //> GT location on the right/left image from the left edge and the GT disparity
    std::vector<std::vector<int>> GT_corresponding_edges; //> veridical right/left edges: A set of right/left edges that are "very close" to the GT location from disparity
    std::vector<int> Closest_GT_veridical_edges;          //> GT locations on the current image
    std::vector<Eigen::Vector3d> Gamma_in_cam_coord;      //> 3D points under the left/right camera coordinate
    // std::unordered_map<int, cv::Mat> edge_SIFT_descriptors; //> SIFT descriptors of all left edges
    std::vector<cv::Mat> edge_descriptors;
    std::vector<int> grid_indices;
    std::unordered_map<int, int> edge_idx_to_stereo_frame_idx; //> Fast lookup: edge_index -> index in focused_edges vector (O(1) instead of O(n) search)

    bool b_is_size_consistent()
    {
        return focused_edges.size() == GT_locations_from_disparity.size() && focused_edges.size() == GT_corresponding_edges.size() && focused_edges.size() == Gamma_in_cam_coord.size() && focused_edges.size() == edge_descriptors.size() && focused_edges.size() == Closest_GT_veridical_edges.size();
    }

    void print_size_consistency()
    {
        std::cout << "The sizes of the StereoEdgeCorrespondencesGT are not consistent!" << std::endl;
        std::cout << "- Size of the focused_edges = " << focused_edges.size() << std::endl;
        std::cout << "- Size of the GT_locations_from_disparity = " << GT_locations_from_disparity.size() << std::endl;
        std::cout << "- Size of the GT_corresponding_edges = " << GT_corresponding_edges.size() << std::endl;
        std::cout << "- Size of the Closest_GT_veridical_edges = " << Closest_GT_veridical_edges.size() << std::endl;
        std::cout << "- Size of the Gamma_in_cam_coord = " << Gamma_in_cam_coord.size() << std::endl;
        std::cout << "- Size of the edge_descriptors = " << edge_descriptors.size() << std::endl;
    }

    void clear_all()
    {
        focused_edges.clear();
        GT_locations_from_disparity.clear();
        GT_corresponding_edges.clear();
        Closest_GT_veridical_edges.clear();
        Gamma_in_cam_coord.clear();
        edge_descriptors.clear();
        grid_indices.clear();
        edge_idx_to_stereo_frame_idx.clear();
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

struct EdgeCorrespondenceData
{
    int stereo_frame_idx;
    cv::Point2d gt_location_on_cf; // the ground truth correspondence edge in the current frame
    double gt_orientation_on_cf;
    std::vector<int> veridical_cf_edges_indices;          // corresponding vertical edges in the current frame
    std::vector<int> matching_cf_edges_indices;           // corresponding edge indices in the current frame after filtering
    std::vector<int> matching_cf_edges_indices_in_GTindx; // SIFT descriptor distances for the matching edges
};
using KF_CF_EdgeCorrespondenceMap = std::unordered_map<int, EdgeCorrespondenceData>;

extern cv::Mat merged_visualization_global;
class Dataset
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(YAML::Node);
    std::unique_ptr<StereoIterator> stereo_iterator;

    void load_dataset(const std::string &dataset_type, std::vector<cv::Mat> &left_ref_disparity_maps,
                      std::vector<cv::Mat> &left_occlusion_masks, int num_pairs);

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

    std::vector<cv::Mat> LoadETH3DOcclusionMasks(const std::string &stereo_pairs_path, int num_maps, bool left = true);

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

#endif