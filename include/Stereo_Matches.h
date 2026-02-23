#ifndef STEREO_MATCHES_H
#define STEREO_MATCHES_H

class Dataset;

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <random>
#include <unordered_set>

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
    std::map<std::string, Stage_Metrics> stages;
};

class Stereo_Matches
{
public:
    Stereo_Matches() { utility_tool = std::make_shared<Utility>(); };
    ~Stereo_Matches() {};

    //> main function to get stereo edge pairs
    Frame_Evaluation_Metrics get_Stereo_Edge_Pairs(Dataset &dataset, Stereo_Edge_Pairs &stereo_frame_edge_pairs, size_t frame_idx);

    //> filtering methods
    void apply_Epipolar_Line_Distance_Filtering(Stereo_Edge_Pairs &stereo_frame_edge_pairs, Dataset &dataset, const std::vector<Edge> right_edges, const std::string &output_dir = "", bool is_left = true, size_t frame_idx = 0, int num_random_edges_for_distribution = 0);
    void apply_Disparity_Filtering(Stereo_Edge_Pairs &stereo_frame_edge_pairs, const std::string &output_dir = "", size_t frame_idx = 0, bool is_left = true);
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

    void finalize_stereo_edge_mates(Stereo_Edge_Pairs &stereo_frame_edge_pairs, std::vector<final_stereo_edge_pair> &final_stereo_edge_pairs);

    //> utility functions
    void augment_Edge_Data(Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left);
    void add_edges_to_spatial_grid(Stereo_Edge_Pairs &stereo_frame_edge_pairs, SpatialGrid &spatial_grid, bool is_left);

    //>
    void Find_Stereo_GT_Locations(Dataset &dataset, const cv::Mat left_disparity_map, const StereoFrame &stereo_frame, Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left);
    void get_Stereo_Edge_GT_Pairs(Dataset &dataset, const StereoFrame &stereo_frame, Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left);
    void record_Filter_Distribution(const std::string &filter_name, const std::vector<double> &filter_values, const std::vector<int> &is_veridical, const std::string &output_dir, size_t frame_idx = 0);
    void record_Ambiguity_Distribution(const std::string &stage_name, const Stereo_Edge_Pairs &stereo_frame_edge_pairs, const std::string &output_dir, size_t frame_idx);

private:
    //> visualization methods
    void record_correspondences_for_visualization(const Stereo_Edge_Pairs &stereo_frame_edge_pairs, const std::string &output_dir, size_t frame_idx, int num_samples = 10);
    std::vector<Eigen::Vector3d> CalculateEpipolarLine(const Eigen::Matrix3d &fund_mat, const std::vector<Edge> &edges);
    void write_Stereo_Edge_Pairs_to_file(Dataset &dataset, Stereo_Edge_Pairs &stereo_frame_edge_pairs, int frame_idx);

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
