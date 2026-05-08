#ifndef STEREO_MATCHES_H
#define STEREO_MATCHES_H

class Dataset;

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <random>
#include <unordered_set>
#include <vector>

#include "definitions.h"
#include "utility.h"
#include "Dataset.h"
#include "EdgeClusterer.h"
#include "io.h"

struct Stage_Metrics
{
    double recall;
    double precision;
    double precision_pair;
    double ambiguity;
};

struct Frame_Evaluation_Metrics
{
    std::vector<std::pair<std::string, Stage_Metrics>> stages;  // preserves pipeline order
};

struct Timing_Statistics
{
    double time_EP = 0.0;
    double time_DP = 0.0;
    double time_OR = 0.0;
    double time_NCC = 0.0;
    double time_SIFT = 0.0;
    double time_BNB_NCC = 0.0;
    double time_BNB_SIFT = 0.0;
    double time_Refinement = 0.0;
    double time_Clustering = 0.0;
    double time_Post_NCC = 0.0;
    double time_Best = 0.0;
    double time_Finalize = 0.0;
    double total_time = 0.0;
};

class Stereo_Matches
{
public:
    Stereo_Matches() { utility_tool = std::make_shared<Utility>(); };
    ~Stereo_Matches() {};

    typedef std::shared_ptr<Stereo_Matches> Ptr;

    //> main function to get stereo edge pairs
    Frame_Evaluation_Metrics get_Stereo_Edge_Pairs(Dataset::Ptr dataset, Stereo_Edge_Pairs &stereo_frame_edge_pairs, size_t frame_idx, Timing_Statistics &timing_statistics);

    //> filtering methods
    void apply_Epipolar_Line_Distance_Filtering(Stereo_Edge_Pairs &stereo_frame_edge_pairs, Dataset::Ptr dataset, const std::vector<Edge> right_edges, const std::string &output_dir = "", bool is_left = true, size_t frame_idx = 0, int num_random_edges_for_distribution = 0);
    void apply_Disparity_Filtering(Stereo_Edge_Pairs &stereo_frame_edge_pairs, const std::string &output_dir = "", size_t frame_idx = 0);
    void apply_SIFT_filtering(Stereo_Edge_Pairs &stereo_frame_edge_pairs, double sift_dist_threshold, const std::string &output_dir = "", size_t frame_idx = 0, bool is_left = true);
    void apply_NCC_Filtering(Stereo_Edge_Pairs &stereo_frame_edge_pairs, const std::string &output_dir, size_t frame_idx, bool is_left = true);
    void apply_Best_Nearly_Best_Test(Stereo_Edge_Pairs &stereo_frame_edge_pairs, double lowe_ratio_threshold = LOWES_RATIO, const std::string &output_dir = "", size_t frame_idx = 0, bool is_NCC = true);
    void apply_Lowe_Ratio_Test(Stereo_Edge_Pairs &stereo_frame_edge_pairs, double lowe_ratio_threshold, const std::string &output_dir, size_t frame_idx);
    void apply_bidirectional_test(Stereo_Edge_Pairs &left_frame, Stereo_Edge_Pairs &right_frame, const std::string &output_dir, int frame_idx);
    void apply_orientation_filter(Stereo_Edge_Pairs &stereo_frame_edge_pairs, double orientation_threshold, const std::string &output_dir, size_t frame_idx);

    void refine_edge_disparity(Stereo_Edge_Pairs &stereo_frame_edge_pairs, size_t frame_idx, bool is_left);
    void consolidate_redundant_edge_hypothesis(Stereo_Edge_Pairs &stereo_frame_edge_pairs, size_t frame_idx, bool b_do_epipolar_shift = true, bool b_do_clustering = true);
    void min_Edge_Photometric_Residual_by_Gauss_Newton(
        Edge left_edge, double init_disp, const cv::Mat &left_image_undistorted, const cv::Mat &right_image_undistorted, const cv::Mat &right_image_gradients_x, /* outputs */
        double &refined_disparity, double &refined_final_score, double &refined_confidence, bool &refined_validity, std::vector<double> &residual_log,           /* optional inputs */
        int max_iter = 20, double tol = 1e-3, double huber_delta = 3.0, bool b_verbose = false);

    //> Same as above but optimizes displacement along an arbitrary epipolar line (1D along (dx,dy)).
    //> epipolar_direction: unit vector (dx,dy) - corresponding point is at left - alpha * (dx,dy). For rectified stereo use (1,0).
    void min_Edge_Photometric_Residual_by_Gauss_Newton_along_EpipolarLine(
        Edge left_edge, Edge right_candidate_edge, cv::Point2d epipolar_direction,
        double init_alpha, const cv::Mat &left_image_undistorted, const cv::Mat &right_image_undistorted,
        const cv::Mat &right_image_gradients_x, const cv::Mat &right_image_gradients_y,
        double &refined_alpha, double &refined_final_score, double &refined_confidence, bool &refined_validity, std::vector<double> &residual_log,
        int max_iter = 20, double tol = 1e-3, double huber_delta = 3.0, bool b_verbose = false);

    void finalize_stereo_edge_mates(Stereo_Edge_Pairs &stereo_frame_edge_pairs, std::vector<final_stereo_edge_pair> &final_stereo_edge_pairs);

    //> utility functions
    void augment_Edge_Data(Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left);
    void add_edges_to_spatial_grid(Stereo_Edge_Pairs &stereo_frame_edge_pairs, SpatialGrid &spatial_grid, bool is_left);

    //>
    void Find_Stereo_GT_Locations(Dataset::Ptr dataset, const cv::Mat left_disparity_map, const StereoFrame &stereo_frame, Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left);
    void get_Stereo_Edge_GT_Pairs(Dataset::Ptr dataset, const StereoFrame &stereo_frame, Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left);
    void record_Filter_Distribution(const std::string &filter_name, const std::vector<double> &filter_values, const std::vector<int> &is_veridical, const std::string &output_dir, size_t frame_idx = 0);
    void record_Ambiguity_Distribution(const std::string &stage_name, const Stereo_Edge_Pairs &stereo_frame_edge_pairs, const std::string &output_dir, size_t frame_idx);

    //> disparity recording
    void record_disparities(const std::vector<final_stereo_edge_pair> &final_stereo_edge_pairs,
                            const std::string &output_dir, size_t frame_idx);

    void Stereo_Matches_Metrics_Statistics(const std::vector<Frame_Evaluation_Metrics> &all_stereo_matches_metrics);

    void write_finalized_stereo_edge_pairs_to_file(Dataset::Ptr dataset, const std::vector<final_stereo_edge_pair> &final_stereo_edge_pairs, size_t frame_idx);

    //> write timings to a file
    void write_timings_to_file(std::ofstream &out_file_stream, Timing_Statistics &timing_statistics)
    {
        out_file_stream << timing_statistics.time_EP << " " << timing_statistics.time_DP << " " \
                        << timing_statistics.time_OR << " " << timing_statistics.time_NCC << " " \
                        << timing_statistics.time_SIFT << " " << timing_statistics.time_BNB_NCC << " " \
                        << timing_statistics.time_BNB_SIFT << " " << timing_statistics.time_Refinement << " " \
                        << timing_statistics.time_Clustering << " " << timing_statistics.time_Post_NCC << " " \
                        << timing_statistics.time_Best << " " << timing_statistics.time_Finalize << " " \
                        << timing_statistics.total_time << std::endl;
    }

private:
    //> visualization methods
    void record_correspondences_for_visualization(const Stereo_Edge_Pairs &stereo_frame_edge_pairs, const std::string &output_dir, size_t frame_idx, int num_samples = 10);
    std::vector<Eigen::Vector3d> CalculateEpipolarLine(const Eigen::Matrix3d &fund_mat, const std::vector<Edge> &edges);
    void write_Stereo_Edge_Pairs_to_file(Dataset::Ptr dataset, Stereo_Edge_Pairs &stereo_frame_edge_pairs, int frame_idx);

    //> evaluation methods
    std::vector<int> get_right_edge_indices_close_to_GT_location(const StereoFrame &stereo_frame, const cv::Point2d GT_location, double GT_orientation, const std::vector<int> right_candidate_edge_indices, const double dist_tol, const double orient_tol, bool is_left = true);
    void Evaluate_Stereo_Edge_Correspondences(
        Stereo_Edge_Pairs &stereo_frame_edge_pairs, size_t frame_idx, const std::string &stage_name,                                              /* Inputs */
        double &recall_per_image, double &precision_per_image, double &precision_pair_per_image, double &num_of_target_edges_per_source_edge_avg, /* Outputs */
        Evaluation_Statistics &evaluation_statistics, bool b_store_FN = false, bool b_store_photo_refine_statistics = false);

    Edge shift_Edge_to_Epipolar_Line(Edge original_edge, const Eigen::Vector3d epipolar_line_coeffs);
    std::vector<int> extract_Epipolar_Edge_Indices(const Eigen::Vector3d &epipolar_line, const std::vector<Edge> &edges, const double dist_tol);
    void remove_empty_clusters(Stereo_Edge_Pairs &stereo_frame_edge_pairs);

    //> Pick random edges(return the indices) from the valid edges that can find their GT correspondences.
    std::vector<int> PickRandomEdges(int size, const int num_points)
    {

        int num_points_clamped = std::min(num_points, size);
        std::vector<int> selected_indices;
        std::unordered_set<int> used_indices;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, size - 1);

        while (selected_indices.size() < num_points_clamped)
        {
            int index = dis(gen);
            if (used_indices.find(index) == used_indices.end())
            {
                used_indices.insert(index);
                selected_indices.push_back(index);
            }
        }

        return selected_indices;
    }

    //> Pointers to other classes
    Utility::Ptr utility_tool = nullptr;
};

#endif // STEREO_MATCHES_H
