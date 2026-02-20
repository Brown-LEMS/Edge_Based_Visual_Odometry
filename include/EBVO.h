#ifndef EBVO_H
#define EBVO_H

#include "Dataset.h"
#include "utility.h"
#include "Stereo_Matches.h"

#include <yaml-cpp/yaml.h>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif
// =======================================================================================================
// EBVO: Main structure of LEMS Edge-Based Visual Odometry
//
// ChangeLogs
//    Jue  25-06-10    Created to reorganize VO.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
//> Jue Han (jhan192@brown.edu)
// =======================================================================================================

struct EdgeGTMatchInfo
{
    Edge edge;                        // the edge we want to find the correspondence
    Edge gt_edge;                     // the ground truth correspondence edge in the next frame
    std::vector<Edge> vertical_edges; // corresponding vertical edges in the next frame
};

class EBVO
{
public:
    EBVO(YAML::Node config_map);

    // Main function to perform edge-based visual odometry
    void PerformEdgeBasedVO();
    void ProcessEdges(const cv::Mat &image,
                      std::shared_ptr<ThirdOrderEdgeDetectionCPU> &toed,
                      std::vector<Edge> &edges);
    void Find_Stereo_GT_Locations(const cv::Mat left_disparity_map, const cv::Mat occlusion_mask, bool is_left, Stereo_Edge_Pairs &stereo_frame_edge_pairs);
    void add_edges_to_spatial_grid(Stereo_Edge_Pairs &stereo_frame, SpatialGrid &spatial_grid, bool is_left);

    //> filtering methods
    void apply_spatial_grid_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, SpatialGrid &spatial_grid, double grid_radius = 1.0);
    void apply_orientation_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, double orientation_threshold, bool is_left);

    void apply_SIFT_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, double sift_dist_threshold, bool is_left);
    void apply_NCC_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, const Stereo_Edge_Pairs &keyframe_stereo, const Stereo_Edge_Pairs &current_stereo, double ncc_val_threshold,
                             const cv::Mat &keyframe_image, const cv::Mat &current_image, bool is_left);
    void apply_best_nearly_best_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, double threshold, bool is_NCC);

    void min_Edge_Photometric_Residual_by_Gauss_Newton(
        /* inputs */
        Edge left_edge, Eigen::Vector2d init_disp, const cv::Mat &left_image_undistorted,
        const cv::Mat &right_image_undistorted, const cv::Mat &right_image_gradients_x, const cv::Mat &right_image_gradients_y,
        /* outputs */
        Eigen::Vector2d &refined_disparity, double &refined_final_score, double &refined_confidence, bool &refined_validity, std::vector<double> &residual_log,
        /* optional inputs */
        int max_iter = 20, double tol = 1e-3, double huber_delta = 3.0, bool b_verbose = false);

    void apply_stereo_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs_left, KF_CF_EdgeCorrespondence &KF_CF_edge_pairs_right,
                                const Stereo_Edge_Pairs &last_keyframe_stereo_left, const Stereo_Edge_Pairs &current_frame_stereo_left,
                                const Stereo_Edge_Pairs &last_keyframe_stereo_right, const Stereo_Edge_Pairs &current_frame_stereo_right,
                                size_t frame_idx);

    void Find_Veridical_Edge_Correspondences_on_CF(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, Stereo_Edge_Pairs &last_keyframe_stereo, Stereo_Edge_Pairs &current_frame_stereo, SpatialGrid &spatial_grid, bool is_left, double gt_dist_threshold = 1.0);
    //> Evaluations
    void Evaluate_KF_CF_Edge_Correspondences(const KF_CF_EdgeCorrespondence &KF_CF_edge_pairs,
                                             Stereo_Edge_Pairs &keyframe_stereo, Stereo_Edge_Pairs &current_stereo,
                                             size_t frame_idx, const std::string &stage_name);

    std::tuple<std::vector<cv::Point2d>, std::vector<double>, std::vector<cv::Point2d>> PickRandomEdges(int patch_size, const std::vector<cv::Point2d> &edges, const std::vector<cv::Point2d> &ground_truth_right_edges, const std::vector<double> &orientations, size_t num_points, int img_width, int img_height);

    void augment_all_Edge_Data(Stereo_Edge_Pairs &stereo_frame_edge_pairs, std::vector<std::pair<cv::Mat, cv::Mat>> &edge_descriptors, bool is_left);

    void EvaluateEdgeMatchPerformance(const std::unordered_map<Edge, std::vector<Edge>> &Edge_match,
                                      const std::unordered_map<Edge, EdgeGTMatchInfo> &gt_correspondences,
                                      size_t frame_idx,
                                      const std::string &stage_name,
                                      double distance_threshold = 3.0);
    void debug_veridical(int edge_idx, const KF_CF_EdgeCorrespondence &KF_CF_edge_pairs_left, const KF_CF_EdgeCorrespondence &KF_CF_edge_pairs_right, const Stereo_Edge_Pairs &keyframe_stereo, const Stereo_Edge_Pairs &current_stereo_left, const Stereo_Edge_Pairs &current_stereo_right, bool is_left);

private:
    //> CH: shared pointer to the class of third-order edge detector
    std::shared_ptr<ThirdOrderEdgeDetectionCPU> TOED = nullptr;
    //> JH: dataset we are working on and its corresponding spatial grid
    Dataset dataset;
    SpatialGrid left_grid;
    SpatialGrid right_grid;
    //> third order edges
    std::vector<Edge> kf_edges_left;  //> 3rd order edges in the keyframe-left
    std::vector<Edge> kf_edges_right; //> Representative edges in the keyframe-right
    std::vector<bool> kf_right_eval;  //> whether the representative edges in the keyframe-right are veridical
    std::vector<Edge> cf_edges_left;  //> 3rd order edges in the current frame-left
    std::vector<Edge> cf_edges_right; //> Representative edges in the current frame-right
    std::vector<bool> cf_right_eval;  //> whether the representative edges in the current frame-right are veridical
    // SIFT descriptor cache for efficient temporal matching

    std::vector<std::pair<cv::Mat, cv::Mat>> current_frame_descriptors_left; // Maps previous frame edge index to its descriptor
    std::vector<std::pair<cv::Mat, cv::Mat>> current_frame_descriptors_right;

    std::vector<int> previous_edge_indices; // Track which edges have descriptors
};

#endif // EBVO_H