#ifndef EBVO_H
#define EBVO_H
#include "Dataset.h"
#include "utility.h"
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
    void ExtractClusterPatches();
    std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> CalculateOrthogonalShifts(
        const std::vector<cv::Point2d> &left_edges,
        const std::vector<double> &left_orientations,
        const std::vector<cv::Point2d> &right_edges,
        const std::vector<double> &right_orientations);
    void ExtractPatches(
        int patch_size,
        const cv::Mat &image,
        const std::vector<cv::Point2d> &edges,
        const std::vector<double> &orientations,
        const std::vector<cv::Point2d> &shifted_one,
        const std::vector<cv::Point2d> &shifted_two,
        std::vector<cv::Point2d> &filtered_edges_out,
        std::vector<double> &filtered_orientations_out,
        std::vector<cv::Mat> &patch_set_one_out,
        std::vector<cv::Mat> &patch_set_two_out,
        const std::vector<cv::Point2d> *ground_truth_edges,
        std::vector<cv::Point2d> *filtered_gt_edges_out);
    std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> ClusterEpipolarShiftedEdges(std::vector<cv::Point2d> &valid_shifted_edges, std::vector<double> &valid_shifted_orientations);
    std::pair<std::vector<cv::Point2d>, std::vector<double>> ExtractEpipolarEdges(const Eigen::Vector3d &epipolar_line, const std::vector<cv::Point2d> &edge_locations, const std::vector<double> &edge_orientations, double distance_threshold);
    std::vector<Eigen::Vector3d> Dataset::CalculateEpipolarLine(const Eigen::Matrix3d &fund_mat, const std::vector<cv::Point2d> &edges);
    std::tuple<std::vector<cv::Point2d>, std::vector<double>, std::vector<cv::Point2d>> Dataset::PickRandomEdges(int patch_size, const std::vector<cv::Point2d> &edges, const std::vector<cv::Point2d> &ground_truth_right_edges, const std::vector<double> &orientations, size_t num_points, int img_width, int img_height);
    cv::Point2d Dataset::PerformEpipolarShift(
        cv::Point2d original_edge_location,
        double edge_orientation,
        std::vector<double> epipolar_line_coeffs,
        bool &b_pass_epipolar_tengency_check);

private:
    //> CH: shared pointer to the class of third-order edge detector
    std::shared_ptr<ThirdOrderEdgeDetectionCPU> TOED = nullptr;
    Dataset dataset;
}

#endif // EBVO_H