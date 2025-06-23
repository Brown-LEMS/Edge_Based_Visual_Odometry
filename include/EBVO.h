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

private:
    //> CH: shared pointer to the class of third-order edge detector
    std::shared_ptr<ThirdOrderEdgeDetectionCPU> TOED = nullptr;
    Dataset dataset;
};

#endif // EBVO_H