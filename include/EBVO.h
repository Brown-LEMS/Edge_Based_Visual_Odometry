#ifndef EBVO_H
#define EBVO_H

#include "Dataset.h"
#include "utility.h"
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
struct SpatialGrid
{
    int cell_size;
    int grid_width, grid_height;
    std::vector<std::vector<int>> grid;

    // Default constructor
    SpatialGrid() : cell_size(30), grid_width(0), grid_height(0) {}

    SpatialGrid(int img_width, int img_height, int cell_sz = 30)
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
                      const std::string &filepath,
                      std::shared_ptr<ThirdOrderEdgeDetectionCPU> &toed,
                      std::vector<Edge> &edges);
    void augment_Edge_Data(StereoEdgeCorrespondencesGT &stereo_frame, const cv::Mat image, bool is_left);
    // void Find_Stereo_GT_Locations(const std::vector<Edge> left_edges, const cv::Mat left_disparity_map, StereoEdgeCorrespondencesGT& prev_stereo_frame);
    void Find_Stereo_GT_Locations(const cv::Mat left_disparity_map, const cv::Mat occlusion_mask, bool left, StereoEdgeCorrespondencesGT &prev_stereo_frame, const std::vector<Edge> &left_edges);

    void add_edges_to_spatial_grid(StereoEdgeCorrespondencesGT &stereo_frame, SpatialGrid &spatial_grid, const std::vector<Edge> &edges);
    void Right_Edges_Stereo_Reconstruction(const StereoEdgeCorrespondencesGT &stereo_left, StereoEdgeCorrespondencesGT &stereo_right, StereoFrame &current_frame);

    //> filtering methods
    void apply_spatial_grid_filtering(KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs, const StereoEdgeCorrespondencesGT &keyframe_stereo, const std::vector<Edge> &edges, SpatialGrid &spatial_grid, double grid_radius = 1.0);
    void apply_orientation_filtering(KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs, const StereoEdgeCorrespondencesGT &keyframe_stereo, const StereoEdgeCorrespondencesGT &current_stereo, double orientation_threshold, bool is_left);

    void apply_SIFT_filtering(KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs, const StereoEdgeCorrespondencesGT &keyframe_stereo, const StereoEdgeCorrespondencesGT &current_stereo, double sift_dist_threshold = 600.0, bool is_left = true);
    void apply_NCC_filtering(KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs, const StereoEdgeCorrespondencesGT &keyframe_stereo, const StereoEdgeCorrespondencesGT &current_stereo, double ncc_val_threshold,
                             const cv::Mat &keyframe_image, const cv::Mat &current_image, bool is_left);
    void apply_stereo_filtering(KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs_left, KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs_right,
                                const StereoEdgeCorrespondencesGT &last_keyframe_stereo_left, const StereoEdgeCorrespondencesGT &current_frame_stereo_left,
                                const StereoEdgeCorrespondencesGT &last_keyframe_stereo_right, const StereoEdgeCorrespondencesGT &current_frame_stereo_right,
                                size_t frame_idx);

    void Find_Veridical_Edge_Correspondences_on_CF(KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs, StereoEdgeCorrespondencesGT &last_keyframe_stereo, StereoEdgeCorrespondencesGT &current_frame_stereo, StereoFrame &last_keyframe, StereoFrame &current_frame, SpatialGrid &spatial_grid, bool is_left, double gt_dist_threshold = 1.0);
    //> Evaluations
    void Evaluate_KF_CF_Edge_Correspondences(const KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs,
                                             StereoEdgeCorrespondencesGT &keyframe_stereo, StereoEdgeCorrespondencesGT &current_stereo,
                                             size_t frame_idx, const std::string &stage_name);

    std::tuple<std::vector<cv::Point2d>, std::vector<double>, std::vector<cv::Point2d>> PickRandomEdges(int patch_size, const std::vector<cv::Point2d> &edges, const std::vector<cv::Point2d> &ground_truth_right_edges, const std::vector<double> &orientations, size_t num_points, int img_width, int img_height);
    std::vector<Eigen::Vector2f> LucasKanadeOpticalFlow(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const std::vector<Edge> &edges,
        int patch_size);

    void EvaluateEdgeMatchPerformance(const std::unordered_map<Edge, std::vector<Edge>> &Edge_match,
                                      const std::unordered_map<Edge, EdgeGTMatchInfo> &gt_correspondences,
                                      size_t frame_idx,
                                      const std::string &stage_name,
                                      double distance_threshold = 3.0);
    void debug_veridical(int edge_idx, const KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs_left, const KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs_right, const StereoEdgeCorrespondencesGT &keyframe_stereo, const StereoEdgeCorrespondencesGT &current_stereo_left, const StereoEdgeCorrespondencesGT &current_stereo_right, bool is_left);

private:
    //> CH: shared pointer to the class of third-order edge detector
    std::shared_ptr<ThirdOrderEdgeDetectionCPU> TOED = nullptr;
    //> JH: dataset we are working on and its corresponding spatial grid
    Dataset dataset;
    SpatialGrid left_grid;
    SpatialGrid right_grid;
    //> third order edges
    std::vector<Edge> kf_edges_left;
    std::vector<Edge> kf_edges_right;
    std::vector<Edge> cf_edges_left;
    std::vector<Edge> cf_edges_right;
    // SIFT descriptor cache for efficient temporal matching
    std::unordered_map<int, cv::Mat> previous_frame_descriptors_cache; // Maps previous frame edge index to its descriptor
    std::vector<int> previous_edge_indices;                            // Track which edges have descriptors
};

#endif // EBVO_H