#ifndef EBVO_H
#define EBVO_H

#include "Dataset.h"
#include "utility.h"
#include <yaml-cpp/yaml.h>
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

class EBVO
{
public:
    EBVO(YAML::Node config_map, bool use_GCC_filter = false);

    // Main function to perform edge-based visual odometry
    void PerformEdgeBasedVO();
    void ProcessEdges(const cv::Mat &image,
                      const std::string &filepath,
                      std::shared_ptr<ThirdOrderEdgeDetectionCPU> &toed,
                      std::vector<Edge> &edges);
    void CalculateGTRightEdge(const std::vector<Edge> &edges, const cv::Mat &disparity_map, const cv::Mat &left_image, const cv::Mat &right_image);
    void ReadEdgesFromBinary(const std::string &filepath,
                             std::vector<Edge> &edges);
    void WriteEdgesToBinary(const std::string &filepath,
                            const std::vector<Edge> &edges);

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
        const double &disparity,
        const Eigen::Matrix3d &K_inverse,
        Edge &edge);

private:
    //> CH: shared pointer to the class of third-order edge detector
    std::shared_ptr<ThirdOrderEdgeDetectionCPU> TOED = nullptr;
    Dataset dataset;
};

#endif // EBVO_H