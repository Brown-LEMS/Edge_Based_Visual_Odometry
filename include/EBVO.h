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

    void WriteEpiDistancesToCSV(const std::vector<std::vector<double>> &all_edge_to_epi_distances);

    void ProcessEdges(const cv::Mat &image,
                      const std::string &filepath,
                      std::shared_ptr<ThirdOrderEdgeDetectionCPU> &toed,
                      std::vector<Edge> &edges);

    void CalculateGTRightEdge(const std::vector<Edge> &edges, const cv::Mat &disparity_map, const cv::Mat &left_image, const cv::Mat &right_image);

    void StoreValidDisparitiesToCSV(const std::vector<double>& disparities);

    void ReadEdgesFromBinary(const std::string &filepath,
                             std::vector<Edge> &edges);

    void WriteEdgesToBinary(const std::string &filepath,
                            const std::vector<Edge> &edges);

    void WriteEdgeMatchResultToCSV(StereoMatchResult &match_result,
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

    std::tuple<std::vector<cv::Point2d>, std::vector<double>, std::vector<cv::Point2d>> PickRandomEdges(const std::vector<cv::Point2d> &edges, const std::vector<cv::Point2d> &ground_truth_right_edges, const std::vector<double> &orientations, size_t num_points, int img_width, int img_height);

    Edge GetGTEdge(bool left, StereoFrame &current_frame, StereoFrame &next_frame,
                   const cv::Mat &disparity_map, const cv::Mat &K_inverse, const cv::Mat &K,
                   const Edge &edge);

private:
    //> CH: shared pointer to the class of third-order edge detector
    std::shared_ptr<ThirdOrderEdgeDetectionCPU> TOED = nullptr;
    Dataset dataset;
};

#endif // EBVO_H