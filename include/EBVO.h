#ifndef EBVO_H
#define EBVO_H

#include "Dataset.h"
#include "utility.h"
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
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
            grid[grid_y * grid_width + grid_x].push_back(edge_idx);
        }
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
    void Find_GT_Locations(const std::vector<Edge> left_edges, const cv::Mat left_disparity_map, StereoEdgeCorrespondencesGT& prev_stereo_frame);
    void ReadEdgesFromBinary(const std::string &filepath,
                             std::vector<Edge> &edges);
    void WriteEdgesToBinary(const std::string &filepath,
                            const std::vector<Edge> &edges);
    void WriteEdgeMatchResult(StereoMatchResult &match_result,
                              std::vector<double> &max_disparity_values,
                              std::vector<double> &per_image_avg_before_epi,
                              std::vector<double> &per_image_avg_after_epi,
                              std::vector<double> &per_image_avg_before_disp,
                              std::vector<double> &per_image_avg_after_disp,
                              std::vector<double> &per_image_avg_before_shift,
                              std::vector<double> &per_image_avg_after_shift,
                              std::vector<double> &per_image_avg_before_clust,
                              std::vector<double> &per_image_avg_after_clust,
                              std::vector<double> &per_image_avg_before_patch,
                              std::vector<double> &per_image_avg_after_patch,
                              std::vector<double> &per_image_avg_before_ncc,
                              std::vector<double> &per_image_avg_after_ncc,
                              std::vector<double> &per_image_avg_before_lowe,
                              std::vector<double> &per_image_avg_after_lowe,
                              std::vector<double> &per_image_avg_before_bct,
                              std::vector<double> &per_image_avg_after_bct,
                              std::vector<RecallMetrics> &all_forward_recall_metrics,
                              std::vector<BidirectionalMetrics> &all_bct_metrics);

    std::tuple<std::vector<cv::Point2d>, std::vector<double>, std::vector<cv::Point2d>> PickRandomEdges(int patch_size, const std::vector<cv::Point2d> &edges, const std::vector<cv::Point2d> &ground_truth_right_edges, const std::vector<double> &orientations, size_t num_points, int img_width, int img_height);
    std::vector<Eigen::Vector2f> LucasKanadeOpticalFlow(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const std::vector<Edge> &edges,
        int patch_size);
    void VisualizeTracks_OpenCVStyle(
        const std::vector<std::vector<cv::Point2d>> &all_tracks,
        const std::vector<cv::Mat> &left_images,
        int n_tracks = 5);

    std::pair<Eigen::Matrix3d, std::vector<int>> Ransac4EdgeEssential(
        const std::vector<Edge> &edge1,
        const std::vector<Edge> &edge2,
        const Eigen::Matrix3d &K,
        int num_iterations = 1000,
        double threshold = 1.0);

    std::pair<Eigen::Matrix3d, Eigen::Vector3d> RelativePoseFromEssential(
        const Eigen::Matrix3d &E,
        int inlierCount,
        const std::vector<Edge> &edge1,
        const std::vector<Edge> &edge2,
        double threshold = 1.0);

    Eigen::Vector3d Point3DFromEdge(
        bool left,

        Edge &edge);
    Edge GetGTEdge(bool left, StereoFrame &current_frame, StereoFrame &next_frame,
                   const cv::Mat &disparity_map, const cv::Mat &K_inverse, const cv::Mat &K,
                   const Edge &edge);

    void GetGTEdges(size_t &frame_idx, StereoFrame &previous_frame, StereoFrame &current_frame,
                    const std::vector<Edge> &previous_frame_edges,
                    const cv::Mat &left_ref_map, const cv::Mat &left_calib_inv,
                    const cv::Mat &left_calib, std::vector<Edge> &gt_edges,
                    std::unordered_map<Edge, EdgeGTMatchInfo> &left_edges_GT_Info);

    void EvaluateSpatialGridPerformance(const std::unordered_map<Edge, Edge> &gt_correspondences, size_t frame_idx, const std::vector<Edge> &previous_frame_edges);
    void EvaluateSIFTMatches(const std::vector<cv::DMatch> &matches,
                             const std::vector<cv::KeyPoint> &previous_keypoints,
                             const std::vector<cv::KeyPoint> &current_keypoints,
                             const std::unordered_map<Edge, Edge> &gt_correspondences,
                             const std::vector<Edge> &previous_frame_edges,
                             const std::vector<Edge> &current_frame_edges,
                             const std::unordered_map<int, cv::Mat> &previous_descriptors_cache,
                             const std::unordered_map<int, cv::Mat> &current_descriptors_cache,
                             size_t frame_idx,
                             double distance_threshold = 5.0);
    void EvaluateEdgeMatchPerformance(const std::unordered_map<Edge, std::vector<Edge>> &Edge_match,
                                      const std::unordered_map<Edge, EdgeGTMatchInfo> &gt_correspondences,
                                      size_t frame_idx,
                                      const std::string &stage_name,
                                      double distance_threshold = 3.0);
    void DebugNCCScoresWithGT(const std::unordered_map<Edge, EdgeGTMatchInfo> &gt_correspondences,
                              size_t frame_idx, const StereoFrame &previous_frame,
                              const StereoFrame &current_frame);

private:
    //> CH: shared pointer to the class of third-order edge detector
    std::shared_ptr<ThirdOrderEdgeDetectionCPU> TOED = nullptr;
    //> JH: dataset we are working on and its corresponding spatial grid
    Dataset dataset;
    SpatialGrid spatial_grid;

    // SIFT descriptor cache for efficient temporal matching
    std::unordered_map<int, cv::Mat> previous_frame_descriptors_cache; // Maps previous frame edge index to its descriptor
    std::vector<int> previous_edge_indices;                            // Track which edges have descriptors
};

#endif // EBVO_H