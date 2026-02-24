#ifndef TEMPORAL_MATCHES_H
#define TEMPORAL_MATCHES_H

#include "Dataset.h"
#include "utility.h"
#include "Stereo_Matches.h"

#include <yaml-cpp/yaml.h>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif
// =======================================================================================================
// Temporal_Matches: Main structure of LEMS Edge-Based Visual Odometry
//
// ChangeLogs
//    Jue  25-06-10    Created to reorganize VO.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
//> Jue Han (jhan192@brown.edu)
// =======================================================================================================

//> One KF stereo correspondence -> all veridical CF stereo correspondences (multiple quads per KF)
struct KF_Veridical_Quads
{
    const final_stereo_edge_pair *KF_stereo_mate;
    std::vector<const final_stereo_edge_pair *> veridical_CF_stereo_mates;
};

class Temporal_Matches
{
public:
    typedef std::shared_ptr<Temporal_Matches> Ptr;

    Temporal_Matches(Dataset::Ptr dataset);

    void get_Temporal_Edge_Pairs( \
        std::vector<final_stereo_edge_pair> &current_frame_stereo_edge_mates, \
        std::vector<temporal_edge_pair> &left_temporal_edge_mates, std::vector<temporal_edge_pair> &right_temporal_edge_mates, \
        const SpatialGrid &left_spatial_grids, const SpatialGrid &right_spatial_grids, \
        const StereoFrame &keyframe, const StereoFrame &current_frame, \
        size_t frame_idx);

    void add_edges_to_spatial_grid(const std::vector<final_stereo_edge_pair> &stereo_edge_mates, SpatialGrid &left_spatial_grids, SpatialGrid &right_spatial_grids);

    //> filtering methods
    void apply_spatial_grid_filtering(std::vector<temporal_edge_pair> &temporal_edge_mates, const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, const SpatialGrid &spatial_grid, double grid_radius = 1.0, bool b_is_left = true);
    void apply_orientation_filtering(std::vector<temporal_edge_pair> &temporal_edge_mates,
                                     const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
                                     double orientation_threshold, bool b_is_left);

    void apply_SIFT_filtering(std::vector<temporal_edge_pair> &temporal_edge_mates,
                              const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
                              double sift_dist_threshold, bool b_is_left);
    void apply_NCC_filtering(std::vector<temporal_edge_pair> &temporal_edge_mates,
                             const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
                             double ncc_val_threshold,
                             const cv::Mat &keyframe_image, const cv::Mat &current_image, bool b_is_left);
    void apply_best_nearly_best_filtering(std::vector<temporal_edge_pair> &temporal_edge_mates, double threshold, const std::string scoring_type);

    void apply_mate_consistency_filtering(std::vector<temporal_edge_pair> &left_temporal_edge_mates,
                                          std::vector<temporal_edge_pair> &right_temporal_edge_mates);

    void apply_photometric_refinement(std::vector<temporal_edge_pair> &temporal_edge_mates,
                                      const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
                                      const StereoFrame &keyframe, const StereoFrame &current_frame,
                                      bool b_is_left);

    void apply_temporal_edge_clustering(std::vector<temporal_edge_pair> &temporal_edge_mates, bool b_cluster_by_orientation = true);

    void min_Edge_Photometric_Residual_by_Gauss_Newton(
        /* inputs */
        Edge kf_edge, Edge cf_edge, Eigen::Vector2d init_disp, const cv::Mat &kf_image_undistorted,
        const cv::Mat &cf_image_undistorted, const cv::Mat &cf_image_gradients_x, const cv::Mat &cf_image_gradients_y,
        /* outputs */
        Eigen::Vector2d &refined_disparity, double &refined_final_score, bool &refined_validity, std::vector<double> &residual_log,
        /* optional inputs */
        int max_iter = 20, double tol = 1e-3, double huber_delta = 3.0, bool b_verbose = false);

    void Find_Veridical_Edge_Correspondences_on_CF(std::vector<temporal_edge_pair> &temporal_edge_mates,
                                                   const std::vector<final_stereo_edge_pair> &KF_stereo_edge_mates,
                                                   const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
                                                   Stereo_Edge_Pairs &last_keyframe_stereo, Stereo_Edge_Pairs &current_frame_stereo,
                                                   SpatialGrid &spatial_grid, bool b_is_left, double gt_dist_threshold = 1.0);
    
    //> Construct veridical quads grouped by KF: each KF stereo correspondence maps to all veridical CF stereo
    //> correspondences (left/right agree), so total quads = sum over KF of veridical_CF_stereo_mates.size().
    std::vector<KF_Veridical_Quads> find_Veridical_Quads(
        const std::vector<temporal_edge_pair> &left_temporal_edge_mates,
        const std::vector<temporal_edge_pair> &right_temporal_edge_mates,
        const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates);
    
    double orientation_mapping(const Edge &e_left, const Edge &e_right, const Eigen::Vector3d projected_point, bool is_left_cam, const StereoFrame &last_keyframe, const StereoFrame &current_frame, Dataset &dataset);

private:
    //> Evaluations
    void Evaluate_Temporal_Edge_Pairs(const std::vector<temporal_edge_pair> &temporal_edge_mates,
        size_t frame_idx, const std::string &stage_name, const std::string which_side_of_temporal_edge_mates);

    Dataset::Ptr dataset;
};

#endif // EBVO_H