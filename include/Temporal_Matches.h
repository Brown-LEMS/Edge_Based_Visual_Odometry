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

//> One KF stereo correspondence -> veridical quads + candidate quads (from spatial grid).
struct KF_Veridical_Quads
{
    const final_stereo_edge_pair *KF_stereo_mate;
    std::vector<Veridical_Quad_Entry> veridical_quads;
    std::vector<Candidate_Quad_Entry> candidate_quads;

    Eigen::Vector3d projected_point_left;
    Eigen::Vector3d projected_point_right;
    double projected_orientation_left;
    double projected_orientation_right;
};

class Temporal_Matches
{
public:
    typedef std::shared_ptr<Temporal_Matches> Ptr;

    Temporal_Matches(Dataset::Ptr dataset);

    //> Quad-centric pipeline: build veridical quads and apply all filters in one flow.
    //> Output: filtered_quads. Optionally populates left/right temporal_edge_mates for backward compatibility.
    void get_Temporal_Edge_Pairs_from_Quads(
        std::vector<KF_Veridical_Quads> &filtered_quads,
        const std::vector<final_stereo_edge_pair> &KF_stereo_edge_mates,
        const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
        const SpatialGrid &left_spatial_grids, const SpatialGrid &right_spatial_grids,
        Stereo_Edge_Pairs &last_keyframe_stereo, Stereo_Edge_Pairs &current_frame_stereo,
        const StereoFrame &keyframe, const StereoFrame &current_frame,
        size_t keyframe_idx, size_t current_frame_idx);

    void add_edges_to_spatial_grid(const std::vector<final_stereo_edge_pair> &stereo_edge_mates, SpatialGrid &left_spatial_grids, SpatialGrid &right_spatial_grids);

    //> Build veridical quads in one step (combines Find_Veridical + find_Veridical_Quads).
    void build_Veridical_Quads(
        std::vector<KF_Veridical_Quads> &out,
        const std::vector<final_stereo_edge_pair> &KF_stereo_edge_mates,
        const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
        Stereo_Edge_Pairs &last_keyframe_stereo, Stereo_Edge_Pairs &current_frame_stereo,
        const SpatialGrid &left_spatial_grids, const SpatialGrid &right_spatial_grids);

    //> filtering methods (temporal_edge_pair - legacy)
    void apply_spatial_grid_filtering(std::vector<temporal_edge_pair> &temporal_edge_mates, const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, const SpatialGrid &spatial_grid, double grid_radius = 1.0, bool b_is_left = true);

    //> Quad-centric filter methods (operate on std::vector<KF_Veridical_Quads>)
    void apply_spatial_grid_filtering_quads(std::vector<KF_Veridical_Quads> &quads_by_kf, const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, const SpatialGrid &left_spatial_grids, const SpatialGrid &right_spatial_grids, double grid_radius = 1.0);
    void apply_orientation_filtering_quads(std::vector<KF_Veridical_Quads> &quads_by_kf, const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, double orientation_threshold);
    void apply_NCC_filtering_quads(std::vector<KF_Veridical_Quads> &quads_by_kf, const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, double ncc_val_threshold, const cv::Mat &keyframe_left_image, const cv::Mat &keyframe_right_image, const cv::Mat &cf_left_image, const cv::Mat &cf_right_image);
    void apply_SIFT_filtering_quads(std::vector<KF_Veridical_Quads> &quads_by_kf, const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, double sift_dist_threshold);
    void apply_best_nearly_best_filtering_quads(std::vector<KF_Veridical_Quads> &quads_by_kf, double threshold, const std::string scoring_type);
    void apply_photometric_refinement_quads(std::vector<KF_Veridical_Quads> &quads_by_kf, const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, const StereoFrame &keyframe, const StereoFrame &current_frame);
    void apply_temporal_edge_clustering_quads(std::vector<KF_Veridical_Quads> &quads_by_kf, bool b_cluster_by_orientation = true);
    void apply_orientation_filtering(std::vector<temporal_edge_pair> &temporal_edge_mates,
                                     const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
                                     double orientation_threshold, bool b_is_left);

    void min_Edge_Photometric_Residual_by_Gauss_Newton(
        /* inputs */
        Edge kf_edge, Edge cf_edge, Eigen::Vector2d init_disp, const cv::Mat &kf_image_undistorted,
        const cv::Mat &cf_image_undistorted, const cv::Mat &cf_image_gradients_x, const cv::Mat &cf_image_gradients_y,
        /* outputs */
        Eigen::Vector2d &refined_disparity, double &refined_final_score, bool &refined_validity, std::vector<double> &residual_log,
        /* optional inputs */
        int max_iter = 20, double tol = 1e-3, double huber_delta = 3.0, bool b_verbose = false);
   
    double orientation_mapping(const Edge &e_left, const Edge &e_right, const Eigen::Vector3d projected_point, bool is_left_cam, const StereoFrame &last_keyframe, const StereoFrame &current_frame, Dataset &dataset);

private:
    //> Evaluate precision/recall/ambiguity on candidate quads (from left/right temporal mates).
    //> TP = candidate quads whose left and right cluster centers are near GT.
    void Evaluate_Temporal_Edge_Pairs_on_Quads(
        const std::vector<KF_Veridical_Quads> &temporal_quads_by_kf,
        const size_t keyframe_idx, const size_t current_frame_idx, const std::string &stage_name);

    //> Cluster storage backing Candidate_Quad_Entry pointers; [kf_idx][candidate_idx] -> (left, right) from same cf_stereo_edge_mate_index
    std::vector<std::vector<std::pair<Temporal_CF_Edge_Cluster, Temporal_CF_Edge_Cluster>>> candidate_cluster_pairs_;

    void print_eval_results_with_no_quads(const size_t keyframe_idx, const size_t current_frame_idx, const std::string &stage_name)
    {
        std::cout << "Quads Evaluation: Key Frame (" << keyframe_idx << ") <-> Current Frame (" << current_frame_idx << ") | Stage: " << stage_name << std::endl;
        std::cout << " (NO CANDIDATE QUADS)" << std::endl;
        std::cout << "========================================================\n" << std::endl;
    }

    Dataset::Ptr dataset;
};

#endif // EBVO_H