#ifndef DATASET_H
#define DATASET_H
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

#include "definitions.h"
#include "Frame.h"
#include "utility.h"
#include "./toed/cpu_toed.hpp"


// =======================================================================================================
// class Dataset: Fetch data from dataset specified in the configuration file
//
// ChangeLogs
//    Lopez  25-01-26    Modified for euroc dataset support.
//    Chien  23-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu), Saul Lopez Lucas (saul_lopez_lucas@brown.edu)
// =======================================================================================================

struct EdgeCluster {
    cv::Point2d center_coord;                  
    double center_orientation;      

    std::vector<cv::Point2d> contributing_edges;
    std::vector<double> contributing_orientations;
};

struct EdgeMatch {
    cv::Point2d coord;
    double orientation;
    double final_score;

    std::vector<cv::Point2d> contributing_edges;
    std::vector<double> contributing_orientations;
};

struct RecallMetrics {
  double epi_distance_recall;
  double max_disparity_recall;
  double epi_shift_recall;
  double epi_cluster_recall;
  double ncc_recall;
  double lowe_recall;


  std::vector<int> epi_input_counts;
  std::vector<int> epi_output_counts;
  std::vector<int> disp_input_counts;
  std::vector<int> disp_output_counts;
  std::vector<int> shift_input_counts;
  std::vector<int> shift_output_counts;
  std::vector<int> clust_input_counts;
  std::vector<int> clust_output_counts;
  std::vector<int> patch_input_counts;
  std::vector<int> patch_output_counts;
  std::vector<int> ncc_input_counts;
  std::vector<int> ncc_output_counts;
  std::vector<int> lowe_input_counts;
  std::vector<int> lowe_output_counts;

   double per_image_epi_precision;
   double per_image_disp_precision;
   double per_image_shift_precision;
   double per_image_clust_precision;
   double per_image_ncc_precision;
   double per_image_lowe_precision;

   int lowe_true_positive;
   int lowe_false_negative;

    double per_image_epi_time;
    double per_image_disp_time;
    double per_image_shift_time;
    double per_image_clust_time;
    double per_image_patch_time;
    double per_image_ncc_time;
    double per_image_lowe_time;
    double per_image_total_time;
};

struct SourceEdge {
    cv::Point2d position;
    double orientation;
};

struct EdgeMatchResult {
    RecallMetrics recall_metrics;
    std::vector<std::pair<SourceEdge, EdgeMatch>> edge_to_cluster_matches; 
};

struct BidirectionalMetrics{
    int matches_before_bct;
    int matches_after_bct;
    double per_image_bct_recall;
    double per_image_bct_precision;
    double per_image_bct_time;
};

struct ConfirmedMatchEdge {
    cv::Point2d position;
    double orientation;
};

struct StereoMatchResult {
    EdgeMatchResult forward_match;
    EdgeMatchResult reverse_match;
    std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>> confirmed_matches;
    BidirectionalMetrics bidirectional_metrics;
};

struct OrientedPoint3D {
    cv::Point3d position;          
    Eigen::Vector3d orientation;  
};

extern cv::Mat merged_visualization_global;
class Dataset {
public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
   typedef std::shared_ptr<Dataset> Ptr;
   Dataset(YAML::Node, bool);

   void PerformEdgeBasedVO();
   static void onMouse(int event, int x, int y, int, void*);

   unsigned Total_Num_Of_Imgs;
   int img_height, img_width;

   std::vector<cv::Point2d> left_third_order_edges_locations;
   std::vector<double> left_third_order_edges_orientation;
   std::vector<cv::Point2d> right_third_order_edges_locations;
   std::vector<double> right_third_order_edges_orientation;

   std::vector<Eigen::Matrix3d> unaligned_GT_Rot;
   std::vector<Eigen::Vector3d> unaligned_GT_Transl;
   std::vector<Eigen::Matrix3d> aligned_GT_Rot;
   std::vector<Eigen::Vector3d> aligned_GT_Transl;
   std::vector<std::tuple<cv::Point2d, cv::Point2d, double>> forward_gt_data;
   std::vector<std::tuple<cv::Point2d, cv::Point2d, double>> reverse_gt_data;

   std::vector<std::pair<double, double>> ncc_one_vs_err;
   std::vector<std::pair<double, double>> ncc_two_vs_err;

private:
   YAML::Node config_file;
   int omp_threads;

   std::string dataset_type;
   std::string dataset_path;
   std::string output_path;
   std::string sequence_name;

   std::string GT_file_name;

   Eigen::Matrix3d rot_frame2body_left;
   Eigen::Vector3d transl_frame2body_left;

   std::vector<int> left_res;
   int left_rate;
   std::string left_model;
   std::vector<double> left_intr;
   std::string left_dist_model;
   std::vector<double> left_dist_coeffs;

   std::vector<int> right_res;
   int right_rate;
   std::string right_model;
   std::vector<double> right_intr;
   std::string right_dist_model;
   std::vector<double> right_dist_coeffs;

   std::vector<std::vector<double>> rot_mat_21;
   std::vector<double> trans_vec_21;
   std::vector<std::vector<double>> fund_mat_21;

   std::vector<std::vector<double>> rot_mat_12;
   std::vector<double> trans_vec_12;
   std::vector<std::vector<double>> fund_mat_12;
   double focal_length;
   double baseline;
   double max_disparity;

   void write_ncc_vals_to_files( int img_index );

   void PrintDatasetInfo();
   StereoMatchResult DisplayMatches(const cv::Mat& left_image, const cv::Mat& right_image, std::vector<cv::Point2d> right_edge_coords, std::vector<double> right_edge_orientations);

    EdgeMatchResult CalculateMatches(const std::vector<cv::Point2d>& selected_primary_edges, const std::vector<double>& selected_primary_orientations, const std::vector<cv::Point2d>& secondary_edge_coords, 
    const std::vector<double>& secondary_edge_orientations, const std::vector<cv::Mat>& primary_patch_set_one, const std::vector<cv::Mat>& primary_patch_set_two, const std::vector<Eigen::Vector3d>& epipolar_lines_secondary, 
    const cv::Mat& secondary_image, const std::vector<cv::Point2d>& selected_ground_truth_edges = std::vector<cv::Point2d>());

    std::vector<cv::Point3d> Calculate3DPoints(
        const std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>>& confirmed_matches
    );

    std::vector<Eigen::Vector3d> Calculate3DOrientations(
    const std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>>& confirmed_matches); 

    void BuildImagePyramids(
    const cv::Mat& curr_left_image,
    const cv::Mat& curr_right_image,
    const cv::Mat& next_left_image,
    const cv::Mat& next_right_image,
    int num_levels,
    std::vector<cv::Mat>& curr_left_pyramid,
    std::vector<cv::Mat>& curr_right_pyramid,
    std::vector<cv::Mat>& next_left_pyramid,
    std::vector<cv::Mat>& next_right_pyramid);

    std::vector<cv::Point3d> LinearTriangulatePoints(
    const std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>>& confirmed_matches
    );

   double ComputeNCC(const cv::Mat& patch_one, const cv::Mat& patch_two);

   std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> CalculateOrthogonalShifts(const std::vector<cv::Point2d>& edge_points, const std::vector<double>& orientations, double shift_magnitude);

   std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> ClusterEpipolarShiftedEdges(std::vector<cv::Point2d>& valid_corrected_edges, std::vector<double>& valid_corrected_orientations); 

   void ExtractClusterPatches(
      int patch_size,
      const cv::Mat& image,
      const std::vector<EdgeCluster>& cluster_centers,
      const std::vector<cv::Point2d>* right_edges, 
      const std::vector<cv::Point2d>& shifted_one,
      const std::vector<cv::Point2d>& shifted_two,
      std::vector<EdgeCluster>& cluster_centers_out,
      std::vector<cv::Point2d>* filtered_right_edges_out,
      std::vector<cv::Mat>& patch_set_one_out,
      std::vector<cv::Mat>& patch_set_two_out
   );

   void ExtractPatches(
    int patch_size,
    const cv::Mat& image,
    const std::vector<cv::Point2d>& edges,
    const std::vector<double>& orientations,
    const std::vector<cv::Point2d>& shifted_one,
    const std::vector<cv::Point2d>& shifted_two,
    std::vector<cv::Point2d>& filtered_edges_out,
    std::vector<double>& filtered_orientations_out,
    std::vector<cv::Mat>& patch_set_one_out,
    std::vector<cv::Mat>& patch_set_two_out,
    const std::vector<cv::Point2d>* ground_truth_edges = nullptr, 
    std::vector<cv::Point2d>* filtered_gt_edges_out = nullptr
   );

   std::pair<std::vector<cv::Point2d>, std::vector<double>> ExtractEpipolarEdges(const Eigen::Vector3d& epipolar_line, const std::vector<cv::Point2d>& edge_locations, const std::vector<double>& edge_orientations, double distance_threshold); 
   
   std::vector<Eigen::Vector3d> CalculateEpipolarLine(const Eigen::Matrix3d& fund_mat, const std::vector<cv::Point2d>& edges);
   
   std::tuple<std::vector<cv::Point2d>, std::vector<double>, std::vector<cv::Point2d>> PickRandomEdges(int patch_size, const std::vector<cv::Point2d>& edges, const std::vector<cv::Point2d>& ground_truth_right_edges, const std::vector<double>& orientations, size_t num_points, int img_width, int img_height);
   
   Eigen::Matrix3d ConvertToEigenMatrix(const std::vector<std::vector<double>>& matrix);
   
   std::vector<std::pair<cv::Mat, cv::Mat>> LoadEuRoCImages(const std::string& csv_path, const std::string& left_path, const std::string& right_path, int num_images);
   
   std::vector<std::pair<cv::Mat, cv::Mat>> LoadETH3DImages(const std::string &stereo_pairs_path, int num_images);
   
   std::vector<double> LoadMaximumDisparityValues(const std::string& stereo_pairs_path, int num_images);
   
   std::vector<cv::Mat> LoadETH3DLeftReferenceMaps(const std::string &stereo_pairs_path, int num_maps);
   
   std::vector<cv::Mat> LoadETH3DRightReferenceMaps(const std::string &stereo_pairs_path, int num_maps);
   
   void WriteDisparityToBinary(const std::string& filepath, const cv::Mat& disparity_map);
   
   cv::Mat ReadDisparityFromBinary(const std::string& filepath);
   
   cv::Mat LoadDisparityFromCSV(const std::string& path);
   
   void WriteEdgesToBinary(const std::string& filepath,
                           const std::vector<cv::Point2d>& locations,
                           const std::vector<double>& orientations); 

   void ReadEdgesFromBinary(const std::string& filepath,
                                   std::vector<cv::Point2d>& locations,
                                   std::vector<double>& orientations);

   void ProcessEdges(const cv::Mat& image,
                     const std::string& filepath,
                     std::shared_ptr<ThirdOrderEdgeDetectionCPU>& toed,
                     std::vector<cv::Point2d>& locations,
                     std::vector<double>& orientations);

   void VisualizeGTRightEdge(const cv::Mat &left_image, const cv::Mat &right_image, const std::vector<std::pair<cv::Point2d, cv::Point2d>> &edge_pairs);
   
   void CalculateGTRightEdge(const std::vector<cv::Point2d> &left_third_order_edges_locations, const std::vector<double> &left_third_order_edges_orientation, const cv::Mat &disparity_map, const cv::Mat &left_image, const cv::Mat &right_image);
   
   void CalculateGTLeftEdge(const std::vector<cv::Point2d>& right_third_order_edges_locations,const std::vector<double>& right_third_order_edges_orientation,const cv::Mat& disparity_map_right_reference,const cv::Mat& left_image,const cv::Mat& right_image);
   
   cv::Point2d PerformEpipolarShift( cv::Point2d original_edge_location, double edge_orientation, std::vector<double> epipolar_line_coeffs, bool& b_pass_epipolar_tengency_check);
   
   void Load_GT_Poses( std::string GT_Poses_File_Path );
   
   std::vector<double> GT_time_stamps;
   
   std::vector<double> Img_time_stamps;
   
   void Align_Images_and_GT_Poses();

   bool compute_grad_depth = false;
   cv::Mat Gx_2d, Gy_2d;
   cv::Mat Small_Patch_Radius_Map;
   Utility::Ptr utility_tool = nullptr;

   //> CH: shared pointer to the class of third-order edge detector
   std::shared_ptr< ThirdOrderEdgeDetectionCPU > TOED = nullptr;

   std::vector< cv::Point2d > ground_truth_right_edges_after_lowe;
};


#endif