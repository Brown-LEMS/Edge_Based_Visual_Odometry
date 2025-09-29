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
            grid[grid_y * grid_width + grid_x].push_back(edge_idx);
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

    //> Parallel bulk insertion using per-thread cell buffers and merge
    void addEdgesParallel(const std::vector<int> &edge_indices, const std::vector<cv::Point2f> &locations)
    {
        const int num_cells = static_cast<int>(grid.size());
        if (edge_indices.size() != locations.size() || num_cells == 0)
        {
            return;
        }

        // Allocate per-thread buffers: [num_threads][num_cells]
        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#endif
        std::vector<std::vector<std::vector<int>>> thread_local_cells(
            static_cast<size_t>(num_threads), std::vector<std::vector<int>>(static_cast<size_t>(num_cells)));

        // Fill thread-local buffers in parallel
        #pragma omp parallel if(locations.size() > 1024)
        {
            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif

            #pragma omp for schedule(static)
            for (int i = 0; i < static_cast<int>(locations.size()); ++i)
            {
                const cv::Point2f &location = locations[static_cast<size_t>(i)];
                int grid_x = static_cast<int>(location.x) / cell_size;
                int grid_y = static_cast<int>(location.y) / cell_size;
                if (grid_x >= 0 && grid_x < grid_width && grid_y >= 0 && grid_y < grid_height)
                {
                    const int cell_index = grid_y * grid_width + grid_x;
                    thread_local_cells[static_cast<size_t>(thread_id)][static_cast<size_t>(cell_index)].push_back(edge_indices[static_cast<size_t>(i)]);
                }
            }
        }

        //> Merge per-thread buffers into the shared grid (single-threaded merge per cell)
        for (int cell_index = 0; cell_index < num_cells; ++cell_index)
        {
            auto &dst = grid[static_cast<size_t>(cell_index)];
            // Reserve to reduce reallocations
            size_t additional = 0;
            for (int t = 0; t < num_threads; ++t)
            {
                additional += thread_local_cells[static_cast<size_t>(t)][static_cast<size_t>(cell_index)].size();
            }
            if (additional > 0)
            {
                dst.reserve(dst.size() + additional);
                for (int t = 0; t < num_threads; ++t)
                {
                    auto &src = thread_local_cells[static_cast<size_t>(t)][static_cast<size_t>(cell_index)];
                    dst.insert(dst.end(), src.begin(), src.end());
                }
            }
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
    void augment_Edge_Data(StereoEdgeCorrespondencesGT& stereo_frame, const cv::Mat image);
    // void Find_Stereo_GT_Locations(const std::vector<Edge> left_edges, const cv::Mat left_disparity_map, StereoEdgeCorrespondencesGT& prev_stereo_frame);
    void Find_Stereo_GT_Locations(const cv::Mat left_disparity_map, StereoEdgeCorrespondencesGT& prev_stereo_frame);

    void add_edges_to_spatial_grid(StereoEdgeCorrespondencesGT& stereo_frame);
    void get_KF_CF_Edge_GT_Pairs(Keyframe_CurrentFrame_EdgeCorrespondencesGT& KC_edge_correspondences, double gt_dist_threshold = 1.0);

    //> last_keyframe and current frame
    void Find_GT_Locations_on_Current_Image(Keyframe_CurrentFrame_EdgeCorrespondencesGT& KC_edge_correspondences, StereoFrame& last_keyframe, StereoFrame& current_frame);
    
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