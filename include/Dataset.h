#pragma once
#ifndef DATASET_H
#define DATASET_H
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>
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
struct SpatialGrid
{
    int cell_size;
    int grid_width, grid_height;
    std::vector<std::vector<int>> grid;

    // Default constructor
    SpatialGrid() : cell_size(35), grid_width(0), grid_height(0) {}

    SpatialGrid(int img_width, int img_height, int cell_sz = 35)
        : cell_size(cell_sz)
    {
        grid_width = (img_width + cell_size - 1) / cell_size;
        grid_height = (img_height + cell_size - 1) / cell_size;
        grid.resize(grid_width * grid_height);
    }

    void addEdge(int edge_idx, cv::Point2f location)
    {
        int grid_x = static_cast<int>(location.x) / cell_size;
        int grid_y = static_cast<int>(location.y) / cell_size;
        if (grid_x >= 0 && grid_x < grid_width && grid_y >= 0 && grid_y < grid_height)
        {
            grid[grid_y * grid_width + grid_x].push_back(edge_idx); // push back the 3rd order edge index
        }
    }

    //> Add edge to grids and return the grid index
    int add_edge_to_grids(int edge_idx, cv::Point2f location)
    {
        int grid_x = static_cast<int>(location.x) / cell_size;
        int grid_y = static_cast<int>(location.y) / cell_size;
        if (grid_x >= 0 && grid_x < grid_width && grid_y >= 0 && grid_y < grid_height)
        {
            grid[grid_y * grid_width + grid_x].push_back(edge_idx);
        }
        return grid_y * grid_width + grid_x;
    }

    void reset()
    {
        for (auto &cell : grid)
        {
            cell.clear();
        }
    }

    std::vector<int> getCandidatesWithinRadius(Edge Edge, double radius)
    {
        std::vector<int> candidates;
        int grid_x = static_cast<int>(Edge.location.x) / cell_size;
        int grid_y = static_cast<int>(Edge.location.y) / cell_size;
        int search_radius = static_cast<int>(std::ceil(radius / cell_size));
        for (int dy = -search_radius; dy <= search_radius; ++dy)
        {
            for (int dx = -search_radius; dx <= search_radius; ++dx)
            {
                int neighbor_x = grid_x + dx;
                int neighbor_y = grid_y + dy;
                if (neighbor_x >= 0 && neighbor_x < grid_width &&
                    neighbor_y >= 0 && neighbor_y < grid_height)
                {
                    const auto &cell = grid[neighbor_y * grid_width + neighbor_x];
                    candidates.insert(candidates.end(), cell.begin(), cell.end());
                }
            }
        }
        return candidates;
    }

    std::vector<int> getCandidatesWithinRadius(cv::Point2d e_location, double radius)
    {
        std::vector<int> candidates;
        int grid_x = static_cast<int>(e_location.x) / cell_size;
        int grid_y = static_cast<int>(e_location.y) / cell_size;
        int search_radius = static_cast<int>(std::ceil(radius / cell_size));
        for (int dy = -search_radius; dy <= search_radius; ++dy)
        {
            for (int dx = -search_radius; dx <= search_radius; ++dx)
            {
                int neighbor_x = grid_x + dx;
                int neighbor_y = grid_y + dy;
                if (neighbor_x >= 0 && neighbor_x < grid_width &&
                    neighbor_y >= 0 && neighbor_y < grid_height)
                {
                    const auto &cell = grid[neighbor_y * grid_width + neighbor_x];
                    candidates.insert(candidates.end(), cell.begin(), cell.end());
                }
            }
        }
        return candidates;
    }
};

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
    std::vector<int> contributing_edges_toed_indices;

    int paired_left_edge_index;
    bool b_is_TP = false;
};

struct Evaluation_Statistics
{
    std::map<std::string, std::vector<EdgeCluster>> edge_clusters_in_each_step;
    std::vector<double> refine_final_scores;
    std::vector<double> refine_confidences;
    std::vector<double> refine_validities;
    std::vector<double> FN_dist_error_to_GT;
    std::vector<EdgeCluster> false_negative_edge_clusters;
};

struct Stereo_Matching_Edge_Clusters
{
    std::vector<EdgeCluster> edge_clusters;
    std::vector<std::pair<cv::Mat, cv::Mat>> matching_edge_patches;

    //> This block is a pool of refinement results
    std::vector<double> refine_final_scores;
    std::vector<double> refine_confidences;
    std::vector<bool> refine_validities;
};

struct Stereo_Edge_Pairs
{
    //> This struct is designed to hold all the relevant information for matching edges between a pair of stereo frames

    //> dataset related
    const StereoFrame *stereo_frame; //> pointer to StereoFrame without owning the data of StereoFrame
    cv::Mat left_disparity_map;
    cv::Mat right_disparity_map;
    //> Edge related --> Populated at the start and would be changed when finished
    std::vector<int> focused_edge_indices;                          //> indices into source edges
    std::vector<int> candidate_edge_indices;                        //> indices into candidate edges
    std::vector<cv::Point2d> GT_locations_from_left_edges;          //> GT locations of the source edges on the candidate image
    std::vector<std::vector<int>> veridical_right_edges_indices;    //> indices into candidate edges that are veridical to the source edges
    std::vector<Eigen::Vector3d> Gamma_in_left_cam_coord;           //> 3D points under the left camera coordinate
    std::vector<Eigen::Vector3d> Gamma_in_right_cam_coord;          //> 3D points under the right camera coordinate
    std::vector<std::pair<cv::Mat, cv::Mat>> left_edge_descriptors; //> SIFT descriptors of source edges on both sides
    std::vector<int> grid_indices;                                  //> grid indices of source edges
    std::vector<Eigen::Vector3d> epip_line_coeffs_of_left_edges;    //> epipolar line coefficients of source edges
    std::vector<std::pair<cv::Mat, cv::Mat>> left_edge_patches;     //> patches on the two sides of the source edges
    std::unordered_map<int, size_t> toed_left_id_to_Stereo_Edge_Pairs_left_id_map;
    std::unordered_map<Edge, int> final_candidate_set; //> find the corresponeding left edge when given right edge, would be populated after Best filter

    //> Matching edge clusters for each left edge
    std::vector<Stereo_Matching_Edge_Clusters> matching_edge_clusters;

    //> Constructor: defining which StereoFrame it points to
    Stereo_Edge_Pairs(const StereoFrame *stereo_frame_ptr) : stereo_frame(stereo_frame_ptr) {}
    Stereo_Edge_Pairs() : stereo_frame(nullptr) {}

    void clean_up_vector_data_structures()
    {
        focused_edge_indices.clear();
        candidate_edge_indices.clear();
        GT_locations_from_left_edges.clear();
        veridical_right_edges_indices.clear();
        Gamma_in_left_cam_coord.clear();
        left_edge_descriptors.clear();
        grid_indices.clear();
        epip_line_coeffs_of_left_edges.clear();
        left_edge_patches.clear();
        matching_edge_clusters.clear();
    }

    void construct_toed_left_id_to_Stereo_Edge_Pairs_left_id_map()
    {
        for (int i = 0; i < focused_edge_indices.size(); ++i)
            toed_left_id_to_Stereo_Edge_Pairs_left_id_map[focused_edge_indices[i]] = i;
    }

    //> Access left and right edges by logical index (through mapping)
    Edge get_left_edge_by_StereoFrame_index(size_t i) const
    {
        if (i >= stereo_frame->left_edges.size())
        {
            std::cerr << "ERROR: left edge index " << i << " out of bounds!" << std::endl;
            return Edge();
        }
        return stereo_frame->left_edges[i];
    }
    Edge get_focused_edge_by_Stereo_Edge_Pairs_index(size_t i) const { return stereo_frame->left_edges[focused_edge_indices[i]]; }
    Edge get_focused_edge_by_toed_index(size_t i) const { return get_focused_edge_by_Stereo_Edge_Pairs_index(get_Stereo_Edge_Pairs_left_id_index(i)); }
    //> Return the number of left and right edge pairs
    size_t size() const { return focused_edge_indices.size(); }

    //> Return a full vector of left edges (in the form of Edge objects)
    std::vector<Edge> get_focused_edges() const
    {
        std::vector<Edge> subset;
        subset.reserve(focused_edge_indices.size());
        for (int idx : focused_edge_indices)
            subset.push_back(stereo_frame->left_edges[idx]);
        return subset;
    }

    //> Return a full vector of right edges (in the form of Edge objects)
    std::vector<Edge> get_candidate_edges() const
    {
        std::vector<Edge> subset;
        subset.reserve(candidate_edge_indices.size());
        for (int idx : candidate_edge_indices)
            subset.push_back(stereo_frame->right_edges[idx]);
        return subset;
    }

    //> getters
    int get_focused_toed_edge_index(size_t i) const { return focused_edge_indices[i]; }
    int get_Stereo_Edge_Pairs_left_id_index(int toed_left_id) const
    {
        auto it = toed_left_id_to_Stereo_Edge_Pairs_left_id_map.find(toed_left_id);
        return (it != toed_left_id_to_Stereo_Edge_Pairs_left_id_map.end()) ? it->second : -1;
    }
    int get_focused_edge_indices_size() const { return focused_edge_indices.size(); }

    bool b_is_size_consistent()
    {
        return focused_edge_indices.size() == GT_locations_from_left_edges.size() && focused_edge_indices.size() == veridical_right_edges_indices.size() && focused_edge_indices.size() == Gamma_in_left_cam_coord.size() && focused_edge_indices.size() == left_edge_descriptors.size();
    }

    void print_size_consistency()
    {
        std::cout << "The sizes of the Stereo_Edge_Pairs are not consistent!" << std::endl;
        std::cout << "- Size of the focused_edge_indices = " << focused_edge_indices.size() << std::endl;
        std::cout << "- Size of the GT_locations_from_left_edges = " << GT_locations_from_left_edges.size() << std::endl;
        std::cout << "- Size of the veridical_right_edges_indices = " << veridical_right_edges_indices.size() << std::endl;
        std::cout << "- Size of the Gamma_in_left_cam_coord = " << Gamma_in_left_cam_coord.size() << std::endl;
        std::cout << "- Size of the left_edge_descriptors = " << left_edge_descriptors.size() << std::endl;
    }
};

struct final_stereo_edge_pair
{
    Edge left_edge;
    Edge right_edge;

    std::pair<cv::Mat, cv::Mat> left_edge_patches;
    std::pair<cv::Mat, cv::Mat> right_edge_patches;

    std::pair<cv::Mat, cv::Mat> left_edge_descriptors;
    std::pair<cv::Mat, cv::Mat> right_edge_descriptors;

    Eigen::Vector3d Gamma_in_left_cam_coord;
    Eigen::Vector3d Gamma_in_right_cam_coord;

    bool b_is_TP;
};

struct scores
{
    double ncc_score;
    double sift_score;
};

struct temporal_edge_pair
{
    //> pointers to the stereo edge pairs
    const final_stereo_edge_pair *KF_stereo_edge_mate;
    // const std::vector<final_stereo_edge_pair> *CF_stereo_edge_mates;

    //> On the current frame
    Eigen::Vector3d projected_point;
    double projected_orientation;

    //> pointer to a set of final_stereo_edge_pair on the current frame
    std::vector<int> veridical_CF_stereo_edge_mate_indices;
    std::vector<int> candidate_CF_stereo_edge_mate_indices;
    std::vector<scores> matching_scores;
};

struct KF_CF_EdgeCorrespondence
{
    bool is_left = true;       //> whether the keyframe is left or right
    std::vector<int> kf_edges; //> stores the 3rd order edge indices in the keyframe-left / stores reprsentative edge indices in the keyframe-right
    std::vector<double> gt_orientation_on_cf;
    std::vector<cv::Point2d> gt_location_on_cf;

    const Stereo_Edge_Pairs *key_frame_pairs;
    const Stereo_Edge_Pairs *current_frame_pairs;

    std::vector<std::vector<int>> veridical_cf_edges_indices; //> corresponding veridical edge indices in the current frame
    std::vector<std::vector<int>> matching_cf_edges_indices;  // corresponding edge indices in the current frame after filtering
    std::vector<std::vector<scores>> matching_scores;
    // getters:
    // Edge get_kf_edge_by_index(size_t i) const
    // {
    //     // returns the edge in keyframe given index in keyframe edge list
    //     if (i >= kf_edges.size())
    //     {
    //         std::cerr << "ERROR: keyframe edge index " << i << " out of bounds!" << std::endl;
    //         return Edge();
    //     }
    //     int idx = kf_edges[i];
    //     Edge e = is_left ? key_frame_pairs->get_focused_edge_by_toed_index(idx) : key_frame_pairs->matching_edge_clusters[key_frame_pairs->get_Stereo_Edge_Pairs_left_id_index(idx)].edge_clusters[0].center_edge;
    //     return e;
    // }
    // Edge get_cf_edge_by_indexij(size_t i, size_t j) const
    // {
    //     // returns the edge in current frame given index in keyframe edge list and index in corresponding cf edge list
    //     if (i >= kf_edges.size() || j >= matching_cf_edges_indices[i].size())
    //     {
    //         std::cerr << "ERROR: current frame edge index " << i << " out of bounds!" << std::endl;
    //         return Edge();
    //     }
    //     int idx = matching_cf_edges_indices[i][j];
    //     Edge e = is_left ? current_frame_pairs->get_focused_edge_by_toed_index(idx) : current_frame_pairs->matching_edge_clusters[current_frame_pairs->get_Stereo_Edge_Pairs_left_id_index(idx)].edge_clusters[0].center_edge;
    //     return e;
    // }
    // Edge get_cf_edge_by_index(size_t i)
    // {
    //     // returns the edge in current frame given its toed index
    //     const std::vector<Edge> &cf_edges = is_left
    //                                             ? current_frame_pairs->stereo_frame->left_edges
    //                                             : current_frame_pairs->stereo_frame->right_edges;
    //     if (i >= cf_edges.size())
    //     {
    //         std::cerr << "ERROR: current frame edge index " << i << " out of bounds!" << std::endl;
    //         return Edge();
    //     }
    //     return cf_edges[i];
    // }
};

extern cv::Mat merged_visualization_global;
class Dataset
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(YAML::Node);
    std::unique_ptr<StereoIterator> stereo_iterator;

    void load_dataset(const std::string &dataset_type, std::vector<cv::Mat> &left_ref_disparity_maps, std::vector<cv::Mat> &right_ref_disparity_maps,
                      std::vector<cv::Mat> &left_occlusion_masks, std::vector<cv::Mat> &right_occlusion_masks, int num_pairs);

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

    /**
     * Read PFM (Portable Float Map) file into cv::Mat
     *
     * @param file_path Path to the .pfm file
     * @return cv::Mat containing the float data (CV_32F, single channel or 3 channels)
     * @throws std::runtime_error if file cannot be read or is malformed
     */
    cv::Mat readPFM(const std::string &file_path);

    /**
     * Read Middlebury disparity file (disp0GT.pfm) with non-occlusion mask
     *
     * @param disp_file_path Path to disp0GT.pfm file
     * @param disparity Output disparity map (CV_32F, single channel)
     * @param valid_mask Output validity mask (CV_8U, 0 or 255)
     * @return true if successful, false otherwise
     */
    bool readDispMiddlebury(const std::string &disp_file_path, cv::Mat &disparity, cv::Mat &valid_mask);

    /**
     * Read ETH3D disparity file (disp0GT.pfm)
     * Uses readPFM and reads mask0nocc.png separately
     *
     * @param disp_file_path Path to disp0GT.pfm file
     * @param disparity Output disparity map (CV_32F, single channel)
     * @param valid_mask Output validity mask (CV_8U, 0 or 255)
     * @return true if successful, false otherwise
     */
    bool readDispETH3D(const std::string &disp_file_path, cv::Mat &disparity, cv::Mat &valid_mask);

    /**
     * Read third-order edge data from a text file.
     * File format: x y orientation disparity (space or tab separated)
     *
     * @param file_path Path to the text file containing edge data
     * @param edges Output vector of Edge objects (left_edges)
     * @param edge_disparity_map Output map from edge index to ground-truth disparity
     * @return true if successful, false otherwise
     */
    bool read_edges_and_disparities_from_file(const std::string &file_path, std::vector<Edge> &edges, std::vector<double> &edge_disparities);

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

    std::vector<cv::Mat> LoadETH3DLeftReferenceMaps(const std::string &stereo_pairs_path, int num_maps);

    std::vector<cv::Mat> LoadETH3DOcclusionMasks(const std::string &stereo_pairs_path, int num_maps, bool left = true);

    void LoadETH3DDisparityMaps(const std::string &stereo_pairs_path, int num_maps, std::vector<cv::Mat> &left_disparity_maps, std::vector<cv::Mat> &right_disparity_maps);

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