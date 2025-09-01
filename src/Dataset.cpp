#ifndef DATASET_CPP
#define DATASET_CPP
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <time.h>
#include <filesystem>
#include <sys/time.h>
#include <random>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <utility> 
#include <cmath>
#include <iomanip>
#include "Dataset.h"
#include "definitions.h"

#if USE_GLOGS
#include <glog/logging.h>
#endif

#if USE_CPP17
#include <filesystem>
#else
#include <boost/filesystem.hpp>
#endif

// =======================================================================================================
// Class Dataset: Fetch data from dataset specified in the configuration file
//
// ChangeLogs
//    Lopez  25-01-26    Modified for euroc dataset support.
//    Chien  23-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu), Saul Lopez Lucas (saul_lopez_lucas@brown.edu)
// =======================================================================================================

cv::Mat merged_visualization_global;

double ComputeAverage(const std::vector<int>& values) {
    if (values.empty()) return 0.0;

    double sum = 0.0;
    for (int val : values) {
        sum += static_cast<double>(val);
    }

    return sum / values.size();
}

cv::Scalar PickUniqueColor(int index, int total) {
    int hue = (index * 180) / total;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(color[0], color[1], color[2]); 
}

Dataset::Dataset(YAML::Node config_map, bool use_GCC_filter) : config_file(config_map), compute_grad_depth(use_GCC_filter) {

#if USE_DEFINED_NUM_OF_CORES
    omp_threads = USE_NUM_CORES_FOR_OMP;
#else
    omp_threads = omp_get_num_procs();
#endif

   dataset_path = config_file["dataset_dir"].as<std::string>();
   output_path = config_file["output_dir"].as<std::string>();
   sequence_name = config_file["sequence_name"].as<std::string>();
   dataset_type = config_file["dataset_type"].as<std::string>();

   if (dataset_type == "EuRoC") {
       try {
           GT_file_name = config_file["state_GT_estimate_file_name"].as<std::string>();

           YAML::Node left_cam = config_file["left_camera"];
           YAML::Node right_cam = config_file["right_camera"];
           YAML::Node stereo = config_file["stereo"];
           YAML::Node frame_to_body = config_file["frame_to_body"];

           left_res = left_cam["resolution"].as<std::vector<int>>();
           left_intr = left_cam["intrinsics"].as<std::vector<double>>();
           left_dist_coeffs = left_cam["distortion_coefficients"].as<std::vector<double>>();

           right_res = right_cam["resolution"].as<std::vector<int>>();
           right_intr = right_cam["intrinsics"].as<std::vector<double>>();
           right_dist_coeffs = right_cam["distortion_coefficients"].as<std::vector<double>>();

           if (stereo["R21"] && stereo["T21"] && stereo["F21"]) {
               for (const auto& row : stereo["R21"]) {
                   rot_mat_21.push_back(row.as<std::vector<double>>());
               }

               trans_vec_21 = stereo["T21"].as<std::vector<double>>();

               for (const auto& row : stereo["F21"]) {
                   fund_mat_21.push_back(row.as<std::vector<double>>());
               }
           } else {
               std::cerr << "ERROR: Missing left-to-right stereo parameters (R21, T21, F21) in YAML file!" << std::endl;
           }

           if (stereo["R12"] && stereo["T12"] && stereo["F12"]) {
               for (const auto& row : stereo["R12"]) {
                   rot_mat_12.push_back(row.as<std::vector<double>>());
               }
               trans_vec_12 = stereo["T12"].as<std::vector<double>>();

               for (const auto& row : stereo["F12"]) {
                   fund_mat_12.push_back(row.as<std::vector<double>>());
               }
           } else {
               std::cerr << "ERROR: Missing right-to-left stereo parameters (R12, T12, F12) in YAML file!" << std::endl;
           }

            if (frame_to_body["rotation"] && frame_to_body["translation"]) {
                rot_frame2body_left = Eigen::Map<Eigen::Matrix3d>(frame_to_body["rotation"].as<std::vector<double>>().data()).transpose();
                transl_frame2body_left = Eigen::Map<Eigen::Vector3d>(frame_to_body["translation"].as<std::vector<double>>().data());
            } else {
                LOG_ERROR("Missing relative rotation and translation from the left camera to the body coordinate (should be given by cam0/sensor.yaml)");
            }

        } catch (const YAML::Exception &e) {
            std::cerr << "ERROR: Could not parse YAML file! " << e.what() << std::endl;
        }
    }
    else if (dataset_type == "ETH3D")
       try {
           YAML::Node left_cam = config_file["left_camera"];
           YAML::Node right_cam = config_file["right_camera"];
           YAML::Node stereo = config_file["stereo"];

           left_res = left_cam["resolution"].as<std::vector<int>>();
           left_intr = left_cam["intrinsics"].as<std::vector<double>>();
           left_dist_coeffs = left_cam["distortion_coefficients"].as<std::vector<double>>();

           right_res = right_cam["resolution"].as<std::vector<int>>();
           right_intr = right_cam["intrinsics"].as<std::vector<double>>();
           right_dist_coeffs = right_cam["distortion_coefficients"].as<std::vector<double>>();

           if (stereo["R21"] && stereo["T21"] && stereo["F21"]) {
               for (const auto& row : stereo["R21"]) {
                   rot_mat_21.push_back(row.as<std::vector<double>>());
               }

               trans_vec_21 = stereo["T21"].as<std::vector<double>>();

               for (const auto& row : stereo["F21"]) {
                   fund_mat_21.push_back(row.as<std::vector<double>>());
               }
           } else {
               std::cerr << "ERROR: Missing left-to-right stereo parameters (R21, T21, F21) in YAML file!" << std::endl;
           }

           if (stereo["R12"] && stereo["T12"] && stereo["F12"]) {
               for (const auto& row : stereo["R12"]) {
                   rot_mat_12.push_back(row.as<std::vector<double>>());
               }
               trans_vec_12 = stereo["T12"].as<std::vector<double>>();

               for (const auto& row : stereo["F12"]) {
                   fund_mat_12.push_back(row.as<std::vector<double>>());
               }
           } else {
               std::cerr << "ERROR: Missing right-to-left stereo parameters (R12, T12, F12) in YAML file!" << std::endl;
           }
           if (stereo["focal_length"] && stereo["baseline"]) {
            focal_length = stereo["focal_length"].as<double>();
            baseline = stereo["baseline"].as<double>();
            } else {
                std::cerr << "ERROR: Missing stereo parameters (focal_length, baseline) in YAML file!" << std::endl;
            }
        } catch (const YAML::Exception &e) {
            std::cerr << "ERROR: Could not parse YAML file! " << e.what() << std::endl;
        }
    
   Total_Num_Of_Imgs = 0;
}

void Dataset::write_ncc_vals_to_files( int img_index ) {
    std::string file_path = OUTPUT_WRITE_PATH + "ncc_vs_err/img_" + std::to_string(img_index) + ".txt";
    std::ofstream ncc_vs_err_file_out(file_path);
    for (unsigned i = 0; i < ncc_one_vs_err.size(); i++) {
        ncc_vs_err_file_out << ncc_one_vs_err[i].first << "\t" << ncc_one_vs_err[i].second << "\t" \
                            << ncc_two_vs_err[i].first << "\t" << ncc_two_vs_err[i].second << "\n";
    }
    ncc_vs_err_file_out.close();
}

void Dataset::PerformEdgeBasedVO() {
    int num_pairs = 2;
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs; 
    std::vector<cv::Mat> left_ref_disparity_maps;
    std::vector<double> max_disparity_values;

    std::vector<double> per_image_avg_before_epi;
    std::vector<double> per_image_avg_after_epi;

    std::vector<double> per_image_avg_before_disp;
    std::vector<double> per_image_avg_after_disp;

    std::vector<double> per_image_avg_before_shift;
    std::vector<double> per_image_avg_after_shift;

    std::vector<double> per_image_avg_before_clust;
    std::vector<double> per_image_avg_after_clust;

    std::vector<double> per_image_avg_before_patch;
    std::vector<double> per_image_avg_after_patch;

    std::vector<double> per_image_avg_before_ncc;
    std::vector<double> per_image_avg_after_ncc;

    std::vector<double> per_image_avg_before_lowe;
    std::vector<double> per_image_avg_after_lowe;
    
    std::vector<double> per_image_avg_before_bct;
    std::vector<double> per_image_avg_after_bct;

    std::vector<RecallMetrics> all_forward_recall_metrics;
    std::vector<BidirectionalMetrics> all_bct_metrics;

    if (dataset_type == "EuRoC"){
        std::string left_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data/";
        std::string right_path = dataset_path + "/" + sequence_name + "/mav0/cam1/data/";
        std::string image_csv_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data.csv";
        std::string ground_truth_path = dataset_path + "/" + sequence_name + "/mav0/state_groundtruth_estimate0/data.csv";

        image_pairs = LoadEuRoCImages(image_csv_path, left_path, right_path, num_pairs);

        Load_GT_Poses(ground_truth_path);
        Align_Images_and_GT_Poses();
    }
    else if (dataset_type == "ETH3D"){
        std::string stereo_pairs_path = dataset_path + "/" + sequence_name + "/stereo_pairs";
        image_pairs = LoadETH3DImages(stereo_pairs_path, num_pairs);
        left_ref_disparity_maps = LoadETH3DLeftReferenceMaps(stereo_pairs_path, num_pairs);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    LOG_INFO("Start looping over all image pairs");

    //> TODO: Determine how to process the last two image pairs
    for (size_t i = 0; i < image_pairs.size()-1; ++i) {
        const cv::Mat& curr_left_img = image_pairs[i].first;
        const cv::Mat& curr_right_img = image_pairs[i].second;

        const cv::Mat& left_ref_map = left_ref_disparity_maps[i]; 

        const cv::Mat& next_left_img = image_pairs[i + 1].first;
        const cv::Mat& next_right_img = image_pairs[i + 1].second;

        ncc_one_vs_err.clear();
        ncc_two_vs_err.clear();
        ground_truth_right_edges_after_lowe.clear();

        std::cout << "\n================== Image Pair #" << i << " ==================\n";
        std::cout << "Folder Path"      << ": " << stereo_image_data[i].folder_path << "\n";
        std::cout << "Left Image Path"  << ": " << stereo_image_data[i].left_image_path << "\n";
        std::cout << "Right Image Path" << ": " << stereo_image_data[i].right_image_path << "\n\n";

        cv::Mat left_calib = (cv::Mat_<double>(3, 3) << left_intr[0], 0, left_intr[2], 0, left_intr[1], left_intr[3], 0, 0, 1);
        cv::Mat right_calib = (cv::Mat_<double>(3, 3) << right_intr[0], 0, right_intr[2], 0, right_intr[1], right_intr[3], 0, 0, 1);
        cv::Mat left_dist_coeff_mat(left_dist_coeffs);
        cv::Mat right_dist_coeff_mat(right_dist_coeffs);

        cv::Mat left_undistorted, right_undistorted;
        cv::undistort(curr_left_img, left_undistorted, left_calib, left_dist_coeff_mat);
        cv::undistort(curr_right_img, right_undistorted, right_calib, right_dist_coeff_mat);

        if (Total_Num_Of_Imgs == 0) {
            left_img_height = left_undistorted.rows;
            left_img_width  = left_undistorted.cols;
            TOED = std::shared_ptr<ThirdOrderEdgeDetectionCPU>(new ThirdOrderEdgeDetectionCPU( left_img_height, left_img_width ));
        }

        std::string edge_dir = output_path + "/edge_bins";
        std::filesystem::create_directories(edge_dir);

        std::string left_edge_path = edge_dir + "/left_edges_" + std::to_string(i);
        std::string right_edge_path = edge_dir + "/right_edges_" + std::to_string(i);

        ProcessEdges(left_undistorted, left_edge_path, TOED, left_third_order_edges_locations, left_third_order_edges_orientation);
        ProcessEdges(right_undistorted, right_edge_path, TOED, right_third_order_edges_locations, right_third_order_edges_orientation);
        
        std::cout << "Left  Image Edges"      << ": " << left_third_order_edges_locations.size() << "\n";
        std::cout << "Right Image Edges"      << ": " << right_third_order_edges_locations.size() << "\n\n";

        Total_Num_Of_Imgs++;


        cv::Mat left_edge_map = cv::Mat::zeros(left_undistorted.size(), CV_8UC1);
        cv::Mat right_edge_map = cv::Mat::zeros(right_undistorted.size(), CV_8UC1);

        for (const auto& edge : left_third_order_edges_locations) {
            if (edge.x >= 0 && edge.x < left_edge_map.cols && edge.y >= 0 && edge.y < left_edge_map.rows) {
                left_edge_map.at<uchar>(cv::Point(edge.x, edge.y)) = 255;
            }
        }

        for (const auto& edge : right_third_order_edges_locations) {
            if (edge.x >= 0 && edge.x < right_edge_map.cols && edge.y >= 0 && edge.y < right_edge_map.rows) {
                right_edge_map.at<uchar>(cv::Point(edge.x, edge.y)) = 255;
            }
        }

        CalculateGTRightEdge(left_third_order_edges_locations, left_third_order_edges_orientation, left_ref_map, left_edge_map, right_edge_map);

        StereoMatchResult match_result = DisplayMatches(
            left_undistorted,
            right_undistorted,
            right_third_order_edges_locations,
            right_third_order_edges_orientation,
            i
        );

#if DISPLAY_STERO_EDGE_MATCHES
        cv::Mat left_visualization, right_visualization;
        cv::cvtColor(left_edge_map, left_visualization, cv::COLOR_GRAY2BGR);
        cv::cvtColor(right_edge_map, right_visualization, cv::COLOR_GRAY2BGR);

        cv::Mat merged_visualization;
        cv::hconcat(left_visualization, right_visualization, merged_visualization);

        int total_matches = static_cast<int>(match_result.confirmed_matches.size());
        int index = 0;

        for (const auto& [left_edge, right_edge] : match_result.confirmed_matches) {
            cv::Scalar color = PickUniqueColor(index, total_matches);

            cv::Point2d left_position = left_edge.position;
            cv::Point2d right_position = right_edge.position;

            cv::circle(merged_visualization, left_position, 4, color, cv::FILLED);
            cv::Point2d right_shifted(right_position.x + left_visualization.cols, right_position.y);
            cv::circle(merged_visualization, right_shifted, 4, color, cv::FILLED);
            cv::line(merged_visualization, left_position, right_shifted, color, 1);

            ++index;
        }

        std::string save_path = output_path + "/edge_matches_image" + std::to_string(i) + ".png";
        cv::imwrite(save_path, merged_visualization);
#endif
        const RecallMetrics& forward_metrics = match_result.forward_match.recall_metrics;
        all_forward_recall_metrics.push_back(forward_metrics);

        const BidirectionalMetrics& bidirectional_metrics = match_result.bidirectional_metrics;
        all_bct_metrics.push_back(bidirectional_metrics);

        double avg_before_epi = ComputeAverage(forward_metrics.epi_input_counts);
        double avg_after_epi = ComputeAverage(forward_metrics.epi_output_counts);

        double avg_before_disp = ComputeAverage(forward_metrics.disp_input_counts);
        double avg_after_disp = ComputeAverage(forward_metrics.disp_output_counts);

        double avg_before_shift = ComputeAverage(forward_metrics.shift_input_counts);
        double avg_after_shift = ComputeAverage(forward_metrics.shift_output_counts);

        double avg_before_clust = ComputeAverage(forward_metrics.clust_input_counts);
        double avg_after_clust = ComputeAverage(forward_metrics.clust_output_counts);

        double avg_before_patch = ComputeAverage(forward_metrics.patch_input_counts);
        double avg_after_patch = ComputeAverage(forward_metrics.patch_output_counts);

        double avg_before_ncc = ComputeAverage(forward_metrics.ncc_input_counts);
        double avg_after_ncc = ComputeAverage(forward_metrics.ncc_output_counts);

        double avg_before_lowe = ComputeAverage(forward_metrics.lowe_input_counts);
        double avg_after_lowe = ComputeAverage(forward_metrics.lowe_output_counts);

        per_image_avg_before_epi.push_back(avg_before_epi);
        per_image_avg_after_epi.push_back(avg_after_epi);

        per_image_avg_before_disp.push_back(avg_before_disp);
        per_image_avg_after_disp.push_back(avg_after_disp);

        per_image_avg_before_shift.push_back(avg_before_shift);
        per_image_avg_after_shift.push_back(avg_after_shift);

        per_image_avg_before_clust.push_back(avg_before_clust);
        per_image_avg_after_clust.push_back(avg_after_clust);

        per_image_avg_before_patch.push_back(avg_before_patch);
        per_image_avg_after_patch.push_back(avg_after_patch);

        per_image_avg_before_ncc.push_back(avg_before_ncc);
        per_image_avg_after_ncc.push_back(avg_after_ncc);

        per_image_avg_before_lowe.push_back(avg_before_lowe);
        per_image_avg_after_lowe.push_back(avg_after_lowe);

        per_image_avg_before_bct.push_back(match_result.bidirectional_metrics.matches_before_bct);
        per_image_avg_after_bct.push_back(match_result.bidirectional_metrics.matches_after_bct);                              
    }

    double total_epi_recall = 0.0;
    double total_disp_recall = 0.0;
    double total_shift_recall = 0.0;
    double total_cluster_recall = 0.0;
    double total_ncc_recall = 0.0;
    double total_lowe_recall = 0.0;
    double total_bct_recall = 0.0;

    double total_epi_precision = 0.0;
    double total_disp_precision = 0.0;
    double total_shift_precision = 0.0;
    double total_cluster_precision = 0.0;
    double total_ncc_precision = 0.0;
    double total_lowe_precision = 0.0;
    double total_bct_precision = 0.0;

    double total_epi_time = 0.0;
    double total_disp_time = 0.0;
    double total_shift_time = 0.0;
    double total_clust_time = 0.0;
    double total_patch_time = 0.0;
    double total_ncc_time = 0.0;
    double total_lowe_time = 0.0;
    double total_image_time = 0.0;
    double total_bct_time = 0.0;

    for (const RecallMetrics& m : all_forward_recall_metrics) {
        total_epi_recall += m.epi_distance_recall;
        total_disp_recall += m.max_disparity_recall;
        total_shift_recall += m.epi_shift_recall;
        total_cluster_recall += m.epi_cluster_recall;
        total_ncc_recall += m.ncc_recall;
        total_lowe_recall += m.lowe_recall;

        total_epi_precision += m.per_image_epi_precision;
        total_disp_precision += m.per_image_disp_precision;
        total_shift_precision += m.per_image_shift_precision;
        total_cluster_precision += m.per_image_clust_precision;
        total_ncc_precision += m.per_image_ncc_precision;
        total_lowe_precision += m.per_image_lowe_precision;

        total_epi_time += m.per_image_epi_time;
        total_disp_time += m.per_image_disp_time;
        total_shift_time += m.per_image_shift_time;
        total_clust_time += m.per_image_clust_time;
        total_patch_time += m.per_image_patch_time;
        total_ncc_time += m.per_image_ncc_time;
        total_lowe_time += m.per_image_lowe_time;
        total_image_time += m.per_image_total_time;
    }

    for (const BidirectionalMetrics& m : all_bct_metrics) {
        total_bct_recall += m.per_image_bct_recall;
        total_bct_precision += m.per_image_bct_precision;
        total_bct_time += m.per_image_bct_time;
    }

    int total_images = static_cast<int>(all_forward_recall_metrics.size());

    double avg_epi_recall   = (total_images > 0) ? total_epi_recall / total_images : 0.0;
    double avg_disp_recall  = (total_images > 0) ? total_disp_recall / total_images : 0.0;
    double avg_shift_recall = (total_images > 0) ? total_shift_recall / total_images : 0.0;
    double avg_cluster_recall = (total_images > 0) ? total_cluster_recall / total_images : 0.0;
    double avg_ncc_recall = (total_images > 0) ? total_ncc_recall / total_images : 0.0;
    double avg_lowe_recall = (total_images > 0) ? total_lowe_recall / total_images : 0.0;
    double avg_bct_recall = (total_images > 0) ? total_bct_recall / total_images : 0.0;

    double avg_epi_precision   = (total_images > 0) ? total_epi_precision / total_images : 0.0;
    double avg_disp_precision  = (total_images > 0) ? total_disp_precision / total_images : 0.0;
    double avg_shift_precision = (total_images > 0) ? total_shift_precision / total_images : 0.0;
    double avg_cluster_precision = (total_images > 0) ? total_cluster_precision / total_images : 0.0;
    double avg_ncc_precision = (total_images > 0) ? total_ncc_precision / total_images : 0.0;
    double avg_lowe_precision = (total_images > 0) ? total_lowe_precision / total_images: 0.0;
    double avg_bct_precision = (total_images > 0) ? total_bct_precision / total_images: 0.0;

    double avg_epi_time = (total_images > 0) ? total_epi_time / total_images : 0.0;
    double avg_disp_time = (total_images > 0) ? total_disp_time / total_images : 0.0;
    double avg_shift_time = (total_images > 0) ? total_shift_time / total_images : 0.0;
    double avg_clust_time = (total_images > 0) ? total_clust_time / total_images : 0.0;
    double avg_patch_time = (total_images > 0) ? total_patch_time / total_images : 0.0;
    double avg_ncc_time = (total_images > 0) ? total_ncc_time / total_images : 0.0;
    double avg_lowe_time = (total_images > 0) ? total_lowe_time / total_images : 0.0;
    double avg_total_time = (total_images > 0) ? total_image_time / total_images : 0.0;
    double avg_bct_time = (total_images > 0) ? total_bct_time / total_images : 0.0;

    std::string edge_stat_dir = output_path + "/edge_stats";
    std::filesystem::create_directories(edge_stat_dir);

    std::ofstream recall_csv(edge_stat_dir + "/recall_metrics.csv");
    recall_csv << "ImageIndex,EpiDistanceRecall,MaxDisparityRecall,EpiShiftRecall,EpiClusterRecall,NCCRecall,LoweRecall,BidirectionalRecall\n";

    std::ofstream time_elapsed_csv(edge_stat_dir + "/time_elapsed_metrics.csv");
    time_elapsed_csv << "ImageIndex,EpiDistanceTime,MaxDisparityTime,EpiShiftTime,EpiClusterTime,PatchTime,NCCTime,LoweTime,TotalLoopTime,BidirectionalTime\n";

    for (size_t i = 0; i < all_forward_recall_metrics.size(); i++) {
        const auto& m = all_forward_recall_metrics[i];
        const auto& bct = all_bct_metrics[i];
        recall_csv << i << ","
                << std::fixed << std::setprecision(4) << m.epi_distance_recall * 100 << ","
                << std::fixed << std::setprecision(4) << m.max_disparity_recall * 100 << ","
                << std::fixed << std::setprecision(4) << m.epi_shift_recall * 100 << ","
                << std::fixed << std::setprecision(4) << m.epi_cluster_recall * 100 << ","
                << std::fixed << std::setprecision(4) << m.ncc_recall * 100 << ","
                << std::fixed << std::setprecision(4) << m.lowe_recall * 100 << ","
                << std::fixed << std::setprecision(4) << bct.per_image_bct_recall * 100 << "\n";
    }

    recall_csv << "Average,"
            << std::fixed << std::setprecision(4) << avg_epi_recall * 100 << ","
            << std::fixed << std::setprecision(4) << avg_disp_recall * 100 << ","
            << std::fixed << std::setprecision(4) << avg_shift_recall * 100 << ","
            << std::fixed << std::setprecision(4) << avg_cluster_recall * 100 << ","
            << std::fixed << std::setprecision(4) << avg_ncc_recall * 100 << ","
            << std::fixed << std::setprecision(4) << avg_lowe_recall * 100 << ","
            << std::fixed << std::setprecision(4) << avg_bct_recall * 100 << "\n";

    for (size_t i = 0; i < all_forward_recall_metrics.size(); i++) {
        const auto& m = all_forward_recall_metrics[i];
        const auto& bct = all_bct_metrics[i];
        time_elapsed_csv << i << ","
                << std::fixed << std::setprecision(4) << m.per_image_epi_time << ","
                << std::fixed << std::setprecision(4) << m.per_image_disp_time << ","
                << std::fixed << std::setprecision(4) << m.per_image_shift_time << ","
                << std::fixed << std::setprecision(4) << m.per_image_clust_time << ","
                << std::fixed << std::setprecision(4) << m.per_image_patch_time << ","
                << std::fixed << std::setprecision(4) << m.per_image_ncc_time << ","
                << std::fixed << std::setprecision(4) << m.per_image_lowe_time << ","
                << std::fixed << std::setprecision(4) << m.per_image_total_time << ","
                << std::fixed << std::setprecision(4) << bct.per_image_bct_time << "\n";
    }

    time_elapsed_csv << "Average,"
            << std::fixed << std::setprecision(4) << avg_epi_time << ","
            << std::fixed << std::setprecision(4) << avg_disp_time << ","
            << std::fixed << std::setprecision(4) << avg_shift_time << ","
            << std::fixed << std::setprecision(4) << avg_clust_time << ","
            << std::fixed << std::setprecision(4) << avg_patch_time << ","
            << std::fixed << std::setprecision(4) << avg_ncc_time << ","
            << std::fixed << std::setprecision(4) << avg_lowe_time << ","
            << std::fixed << std::setprecision(4) << avg_total_time << ","
            << std::fixed << std::setprecision(4) << avg_bct_time << "\n";
    
    std::ofstream count_csv(edge_stat_dir + "/count_metrics.csv");
    count_csv 
        << "before_epi_distance,after_epi_distance,average_before_epi_distance,average_after_epi_distance,"
        << "before_max_disp,after_max_disp,average_before_max_disp,average_after_max_disp,"
        << "before_epi_shift,after_epi_shift,average_before_epi_shift,average_after_epi_shift,"
        << "before_epi_cluster,after_epi_cluster,average_before_epi_cluster,average_after_epi_cluster,"
        << "before_patch, after_patch, average_before_patch, average_after_patch,"
        << "before_ncc,after_ncc,average_before_ncc,average_after_ncc,"
        << "before_lowe,after_lowe,average_before_lowe,after_after_lowe,"
        << "before_bct (PER IMAGE),after_bct (PER IMAGE),average_before_bct (PER IMAGE),after_after_bct (PER IMAGE)\n";

    double total_avg_before_epi = 0.0;
    double total_avg_after_epi = 0.0;

    double total_avg_before_disp = 0.0;
    double total_avg_after_disp = 0.0;

    double total_avg_before_shift = 0.0;
    double total_avg_after_shift = 0.0;

    double total_avg_before_clust = 0.0;
    double total_avg_after_clust = 0.0;

    double total_avg_before_patch = 0.0;
    double total_avg_after_patch = 0.0;

    double total_avg_before_ncc = 0.0;
    double total_avg_after_ncc = 0.0;

    double total_avg_before_lowe = 0.0;
    double total_avg_after_lowe = 0.0;

    double total_avg_before_bct = 0.0;
    double total_avg_after_bct = 0.0;

    size_t num_rows = per_image_avg_before_epi.size();
    
    for (size_t i = 0; i < num_rows; ++i) {
        total_avg_before_epi += per_image_avg_before_epi[i];
        total_avg_after_epi += per_image_avg_after_epi[i];

        total_avg_before_disp += per_image_avg_before_disp[i];
        total_avg_after_disp += per_image_avg_after_disp[i];

        total_avg_before_shift += per_image_avg_before_shift[i];
        total_avg_after_shift += per_image_avg_after_shift[i];

        total_avg_before_clust += per_image_avg_before_clust[i];
        total_avg_after_clust += per_image_avg_after_clust[i];

        total_avg_before_patch += per_image_avg_before_patch[i];
        total_avg_after_patch += per_image_avg_after_patch[i];

        total_avg_before_ncc += per_image_avg_before_ncc[i];
        total_avg_after_ncc += per_image_avg_after_ncc[i];

        total_avg_before_lowe += per_image_avg_before_lowe[i];
        total_avg_after_lowe += per_image_avg_after_lowe[i];

        total_avg_before_bct += per_image_avg_before_bct[i];
        total_avg_after_bct += per_image_avg_after_bct[i];

        count_csv
            << static_cast<int>(std::ceil(per_image_avg_before_epi[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_epi[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_disp[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_disp[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_shift[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_shift[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_clust[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_clust[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_patch[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_patch[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_ncc[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_ncc[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_lowe[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_lowe[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_bct[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_bct[i])) << ","
            <<"\n";
    }

    int avg_of_avgs_before_epi = 0;
    int avg_of_avgs_after_epi = 0;

    int avg_of_avgs_before_disp = 0;
    int avg_of_avgs_after_disp = 0;

    int avg_of_avgs_before_shift = 0;
    int avg_of_avgs_after_shift = 0;

    int avg_of_avgs_before_clust = 0;
    int avg_of_avgs_after_clust = 0;

    int avg_of_avgs_before_patch = 0;
    int avg_of_avgs_after_patch = 0;
    
    int avg_of_avgs_before_ncc = 0;
    int avg_of_avgs_after_ncc = 0;

    int avg_of_avgs_before_lowe = 0;
    int avg_of_avgs_after_lowe = 0;

    int avg_of_avgs_before_bct = 0;
    int avg_of_avgs_after_bct = 0;

    if (num_rows > 0) {
        avg_of_avgs_before_epi = std::ceil(total_avg_before_epi / num_rows);
        avg_of_avgs_after_epi = std::ceil(total_avg_after_epi / num_rows);

        avg_of_avgs_before_disp = std::ceil(total_avg_before_disp / num_rows);
        avg_of_avgs_after_disp = std::ceil(total_avg_after_disp / num_rows);

        avg_of_avgs_before_shift = std::ceil(total_avg_before_shift / num_rows);
        avg_of_avgs_after_shift = std::ceil(total_avg_after_shift / num_rows);

        avg_of_avgs_before_clust = std::ceil(total_avg_before_clust / num_rows);
        avg_of_avgs_after_clust = std::ceil(total_avg_after_clust / num_rows);

        avg_of_avgs_before_patch = std::ceil(total_avg_before_patch / num_rows);
        avg_of_avgs_after_patch = std::ceil(total_avg_after_patch / num_rows);

        avg_of_avgs_before_ncc = std::ceil(total_avg_before_ncc / num_rows);
        avg_of_avgs_after_ncc = std::ceil(total_avg_after_ncc / num_rows);

        avg_of_avgs_before_lowe = std::ceil(total_avg_before_lowe / num_rows);
        avg_of_avgs_after_lowe = std::ceil(total_avg_after_lowe / num_rows);

        avg_of_avgs_before_bct = std::ceil(total_avg_before_bct / num_rows);
        avg_of_avgs_after_bct = std::ceil(total_avg_after_bct / num_rows);
    }

    count_csv 
        << ","                                   
        << ","                                                                     
        << avg_of_avgs_before_epi << ","       
        << avg_of_avgs_after_epi << ","   
        << "," 
        << ","     
        << avg_of_avgs_before_disp << ","   
        << avg_of_avgs_after_disp << ","
        << ","      
        << ","                              
        << avg_of_avgs_before_shift << ","              
        << avg_of_avgs_after_shift << ","    
        << ","   
        << ","                                            
        << avg_of_avgs_before_clust << ","            
        << avg_of_avgs_after_clust << ","
        << ","
        << ","
        << avg_of_avgs_before_patch << ","            
        << avg_of_avgs_after_patch << ","
        << ","
        << ","
        << avg_of_avgs_before_ncc << ","            
        << avg_of_avgs_after_ncc << ","
        << ","
        << ","
        << avg_of_avgs_before_lowe << ","            
        << avg_of_avgs_after_lowe << ","
        << ","
        << ","
        << avg_of_avgs_before_bct << ","            
        << avg_of_avgs_after_bct << "\n";  
    
    std::ofstream precision_csv(edge_stat_dir + "/precision_metrics.csv");
    precision_csv << "ImageIndex,EpiDistancePrecision,MaxDisparityPrecision,EpiShiftPrecision,EpiClusterPrecision,NCCPrecision,LowePrecision,BidirectionalPrecision\n";

    for (size_t i = 0; i < all_forward_recall_metrics.size(); i++) {
        const auto& m = all_forward_recall_metrics[i];
        const auto& bct = all_bct_metrics[i];
        precision_csv << i << ","
                << std::fixed << std::setprecision(4) << m.per_image_epi_precision * 100 << ","
                << std::fixed << std::setprecision(4) << m.per_image_disp_precision * 100 << ","
                << std::fixed << std::setprecision(4) << m.per_image_shift_precision * 100 << ","
                << std::fixed << std::setprecision(4) << m.per_image_clust_precision * 100 << ","
                << std::fixed << std::setprecision(4) << m.per_image_ncc_precision * 100 << ","
                << std::fixed << std::setprecision(4) << m.per_image_lowe_precision * 100 << ","
                << std::fixed << std::setprecision(4) << bct.per_image_bct_precision * 100 << "\n";
    }

    precision_csv << "Average,"
        << std::fixed << std::setprecision(4) << avg_epi_precision * 100 << ","
        << std::fixed << std::setprecision(4) << avg_disp_precision * 100 << ","
        << std::fixed << std::setprecision(4) << avg_shift_precision * 100 << ","
        << std::fixed << std::setprecision(4) << avg_cluster_precision * 100 << ","
        << std::fixed << std::setprecision(4) << avg_ncc_precision * 100 << ","
        << std::fixed << std::setprecision(4) << avg_lowe_precision * 100 << ","
        << std::fixed << std::setprecision(4) << avg_bct_precision * 100 << "\n";
}

StereoMatchResult Dataset::DisplayMatches(const cv::Mat& left_image, const cv::Mat& right_image, std::vector<cv::Point2d> right_edge_coords, std::vector<double> right_edge_orientations, int image_pair_index) {
    cv::Mat left_image_64f, right_image_64f;
    left_image.convertTo(left_image_64f, CV_64F);
    right_image.convertTo(right_image_64f, CV_64F);

    ///////////////////////////////FORWARD DIRECTION///////////////////////////////
    std::vector<cv::Point2d> left_edge_coords;
    std::vector<cv::Point2d> ground_truth_right_edges;
    std::vector<double> left_edge_orientations;
    
    for (const auto& data : forward_gt_data) {
        left_edge_coords.push_back(std::get<0>(data)); 
        ground_truth_right_edges.push_back(std::get<1>(data)); 
        left_edge_orientations.push_back(std::get<2>(data)); 
    }
    
    std::vector<cv::Point2d> filtered_left_edges;
    std::vector<double> filtered_left_orientations;
    std::vector<cv::Point2d> filtered_ground_truth_right_edges;
    
    std::vector<cv::Mat> left_patch_set_one;
    std::vector<cv::Mat> left_patch_set_two;
    
    int left_img_width  = left_image_64f.cols;
    int left_img_height = left_image_64f.rows;
    
    for (size_t i = 0; i < left_edge_coords.size(); ++i) {
        Edge edge;
        edge.location = left_edge_coords[i];
        edge.orientation = left_edge_orientations[i];
    
        auto [shifted_plus, shifted_minus] = get_Orthogonal_Shifted_Points(edge);
    
        if (!is_patch_in_bounds(shifted_plus, PATCH_HALF_SIZE, left_img_width, left_img_height) ||
            !is_patch_in_bounds(shifted_minus, PATCH_HALF_SIZE, left_img_width, left_img_height)) {
            continue;
        }
    
        cv::Mat patch_coord_x_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_coord_y_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_plus          = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    
        cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_minus         = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    
        get_patch_on_one_edge_side(
            shifted_plus,
            edge.orientation,
            patch_coord_x_plus,
            patch_coord_y_plus,
            patch_plus,
            left_image_64f
        );
    
        get_patch_on_one_edge_side(
            shifted_minus,
            edge.orientation,
            patch_coord_x_minus,
            patch_coord_y_minus,
            patch_minus,
            left_image_64f
        );
    
        cv::Mat patch_plus_32f, patch_minus_32f;
        patch_plus.convertTo(patch_plus_32f, CV_32F);
        patch_minus.convertTo(patch_minus_32f, CV_32F);
    
        filtered_left_edges.push_back(edge.location);
        filtered_left_orientations.push_back(edge.orientation);
        if (!ground_truth_right_edges.empty()) {
            filtered_ground_truth_right_edges.push_back(ground_truth_right_edges[i]);
        }
    
        left_patch_set_one.push_back(patch_plus_32f);
        left_patch_set_two.push_back(patch_minus_32f);
    }    

    Eigen::Matrix3d fundamental_matrix_21 = ConvertToEigenMatrix(fund_mat_21);
    Eigen::Matrix3d fundamental_matrix_12 = ConvertToEigenMatrix(fund_mat_12);

    std::vector<Eigen::Vector3d> epipolar_lines_right = CalculateEpipolarLine(fundamental_matrix_21, filtered_left_edges);

    EdgeMatchResult forward_match = CalculateMatches(
        filtered_left_edges,
        filtered_left_orientations,
        right_edge_coords,
        right_edge_orientations,
        left_patch_set_one,
        left_patch_set_two,
        epipolar_lines_right,
        right_image_64f,
        filtered_ground_truth_right_edges,
        image_pair_index,
        true
    );

    ///////////////////////////////REVERSE DIRECTION///////////////////////////////
    std::vector<cv::Point2d> reverse_primary_edges;
    std::vector<double> reverse_primary_orientations;
    
    for (const auto& match_pair : forward_match.edge_to_cluster_matches) {
        const EdgeMatch& match_info = match_pair.second;
    
        for (const auto& edge : match_info.contributing_edges) {
            reverse_primary_edges.push_back(edge);
        }
        for (const auto& orientation : match_info.contributing_orientations) {
            reverse_primary_orientations.push_back(orientation);
        }
    }
    
    std::vector<cv::Point2d> filtered_right_edges;
    std::vector<double> filtered_right_orientations;
    
    std::vector<cv::Mat> right_patch_set_one;
    std::vector<cv::Mat> right_patch_set_two;
    
    int right_img_width  = right_image_64f.cols;
    int right_img_height = right_image_64f.rows;
    
    for (size_t i = 0; i < reverse_primary_edges.size(); ++i) {
        Edge edge;
        edge.location = reverse_primary_edges[i];
        edge.orientation = reverse_primary_orientations[i];
    
        auto [shifted_plus, shifted_minus] = get_Orthogonal_Shifted_Points(edge);
    
        if (!is_patch_in_bounds(shifted_plus, PATCH_HALF_SIZE, right_img_width, right_img_height) ||
            !is_patch_in_bounds(shifted_minus, PATCH_HALF_SIZE, right_img_width, right_img_height)) {
            continue;
        }
    
        cv::Mat patch_coord_x_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_coord_y_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_plus          = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    
        cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_minus         = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    
        get_patch_on_one_edge_side(
            shifted_plus,
            edge.orientation,
            patch_coord_x_plus,
            patch_coord_y_plus,
            patch_plus,
            right_image_64f
        );
    
        get_patch_on_one_edge_side(
            shifted_minus,
            edge.orientation,
            patch_coord_x_minus,
            patch_coord_y_minus,
            patch_minus,
            right_image_64f
        );
    
        cv::Mat patch_plus_32f, patch_minus_32f;
        patch_plus.convertTo(patch_plus_32f, CV_32F);
        patch_minus.convertTo(patch_minus_32f, CV_32F);
    
        filtered_right_edges.push_back(edge.location);
        filtered_right_orientations.push_back(edge.orientation);
    
        right_patch_set_one.push_back(patch_plus_32f);
        right_patch_set_two.push_back(patch_minus_32f);
    }    

    std::vector<Eigen::Vector3d> epipolar_lines_left = CalculateEpipolarLine(fundamental_matrix_12, filtered_right_edges);

    EdgeMatchResult reverse_match = CalculateMatches(
        filtered_right_edges,
        filtered_right_orientations,
        left_edge_coords,
        left_edge_orientations,
        right_patch_set_one,
        right_patch_set_two,
        epipolar_lines_left,
        left_image_64f,
        std::vector<cv::Point2d>(),
        image_pair_index,
        false
    );

    std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>> confirmed_matches;

    int matches_before_bct = static_cast<int>(forward_match.edge_to_cluster_matches.size());
    auto bct_start = std::chrono::high_resolution_clock::now();

    ///////////////////////////////BCT///////////////////////////////
    int forward_left_index = 0;
    int bct_true_positive = 0;
    for (const auto& [left_oriented_edge, patch_match_forward] : forward_match.edge_to_cluster_matches) {
        const cv::Point2d& left_position = left_oriented_edge.position;
        const double left_orientation = left_oriented_edge.orientation;

        const auto& right_contributing_edges = patch_match_forward.contributing_edges;
        const auto& right_contributing_orientations = patch_match_forward.contributing_orientations;
        bool break_flag = false;
        for (size_t i = 0; i < right_contributing_edges.size(); ++i) {
            break_flag = false;
            const cv::Point2d& right_position = right_contributing_edges[i];
            const double right_orientation = right_contributing_orientations[i];

            for (const auto& [rev_right_edge, patch_match_rev] : reverse_match.edge_to_cluster_matches) {
                if (cv::norm(rev_right_edge.position - right_position) <= MATCH_TOL) {

                    for (const auto& rev_contributing_left : patch_match_rev.contributing_edges) {
                        if (cv::norm(rev_contributing_left - left_position) <= MATCH_TOL) { 
                            ConfirmedMatchEdge left_confirmed{left_position, left_orientation};
                            ConfirmedMatchEdge right_confirmed{right_position, right_orientation};
                            confirmed_matches.emplace_back(left_confirmed, right_confirmed);

                            cv::Point2d GT_right_edge_location = ground_truth_right_edges_after_lowe[forward_left_index];
                            if (cv::norm(right_position - GT_right_edge_location) <= MATCH_TOL) {
                                bct_true_positive++;
                            }
                            break_flag = true;
                            break;
                        }
                    }
                }
                if (break_flag) break;
            }
            if (break_flag) break;
        }
        forward_left_index++;
    }

    auto bct_end = std::chrono::high_resolution_clock::now();
    double total_time_bct = std::chrono::duration<double, std::milli>(bct_end - bct_start).count();

    double per_image_bct_time = (matches_before_bct > 0) ? total_time_bct / matches_before_bct : 0.0;

    int matches_after_bct = static_cast<int>(confirmed_matches.size());

    double per_image_bct_precision = (matches_before_bct > 0) ? bct_true_positive / (double)(matches_after_bct) : 0.0;

    int bct_denonimator = forward_match.recall_metrics.lowe_true_positive + forward_match.recall_metrics.lowe_false_negative;

    double bct_recall = (bct_denonimator > 0) ? bct_true_positive / (double)(bct_denonimator) : 0.0;

    std::cout << "Matches Before BCT"     << ": " << matches_before_bct << "\n";
    std::cout << "Matches After BCT"      << ": " << matches_after_bct << "\n";
    std::cout << "BCT True Positives"     << ": " << bct_true_positive << "\n";
    std::cout << "Stacked GT Right Edges" << ": " << ground_truth_right_edges_after_lowe.size() << "\n";

    std::cout << "BCT Precision"           << ": " << per_image_bct_precision << "\n";
    std::cout << "BCT Recall"              << ": " << bct_recall << "\n";

    BidirectionalMetrics bidirectional_metrics;
    bidirectional_metrics.matches_before_bct = matches_before_bct;
    bidirectional_metrics.matches_after_bct = matches_after_bct;
    bidirectional_metrics.per_image_bct_recall = bct_recall;
    bidirectional_metrics.per_image_bct_precision = per_image_bct_precision;
    bidirectional_metrics.per_image_bct_time = per_image_bct_time;

    return StereoMatchResult{forward_match, reverse_match, confirmed_matches, bidirectional_metrics};
}

//> MARK: Main Edge Pairing
EdgeMatchResult Dataset::CalculateMatches(const std::vector<cv::Point2d>& selected_primary_edges, const std::vector<double>& selected_primary_orientations, const std::vector<cv::Point2d>& secondary_edge_coords, 
    const std::vector<double>& secondary_edge_orientations, const std::vector<cv::Mat>& primary_patch_set_one, const std::vector<cv::Mat>& primary_patch_set_two, const std::vector<Eigen::Vector3d>& epipolar_lines_secondary, 
    const cv::Mat& secondary_image, const std::vector<cv::Point2d>& selected_ground_truth_edges, int image_pair_index, bool forward_direction) {
    auto total_start = std::chrono::high_resolution_clock::now();

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

    double total_time;

    //> CH: this is a global structure of final_matches
    std::vector<std::pair<SourceEdge, EdgeMatch>> final_matches;

    //> CH: this is local structure of final matches
    std::vector< std::vector<std::pair<SourceEdge, EdgeMatch>> > local_final_matches(omp_get_max_threads());

    //> CH: Local structures of all counts
    std::vector< std::vector<int> > local_epi_input_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_epi_output_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_disp_input_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_disp_output_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_shift_input_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_shift_output_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_clust_input_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_clust_output_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_patch_input_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_patch_output_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_ncc_input_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_ncc_output_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_lowe_input_counts(omp_get_max_threads());
    std::vector< std::vector<int> > local_lowe_output_counts(omp_get_max_threads());

    //> CH: Local structures for GT right edge after Lowe's ratio test
    std::vector< std::vector< cv::Point2d > > local_GT_right_edges_after_lowe(omp_get_max_threads());

    int time_epi_edges_evaluated = 0;
    int time_disp_edges_evaluated = 0;
    int time_shift_edges_evaluated = 0;
    int time_clust_edges_evaluated = 0;
    int time_patch_edges_evaluated = 0;
    int time_ncc_edges_evaluated = 0;
    int time_lowe_edges_evaluated = 0;
    double time_epi = 0.0;
    double time_disp = 0.0;
    double time_shift = 0.0;
    double time_patch = 0.0;
    double time_cluster = 0.0;
    double time_ncc = 0.0;
    double time_lowe = 0.0;

    //> These are global variables for reduction sum
    double per_edge_epi_precision = 0.0;
    double per_edge_disp_precision = 0.0;
    double per_edge_shift_precision = 0.0;
    double per_edge_clust_precision = 0.0;
    double per_edge_ncc_precision = 0.0;
    double per_edge_lowe_precision = 0.0;
    int epi_true_positive = 0;
    int epi_false_negative = 0;
    int epi_true_negative = 0;
    int disp_true_positive = 0;
    int disp_false_negative = 0;
    int shift_true_positive = 0;
    int shift_false_negative = 0;
    int cluster_true_positive = 0;
    int cluster_false_negative = 0;
    int cluster_true_negative = 0;
    int ncc_true_positive = 0;
    int ncc_false_negative = 0;
    int lowe_true_positive = 0;
    int lowe_false_negative = 0;
    int epi_edges_evaluated = 0;
    int disp_edges_evaluated = 0;
    int shift_edges_evaluated = 0;
    int clust_edges_evaluated = 0;
    int ncc_edges_evaluated = 0;
    int lowe_edges_evaluated = 0;

    std::ofstream veridical_csv;
    std::ofstream nonveridical_csv;
    std::ofstream epi_distance_csv;
    
    if (forward_direction) {
        std::filesystem::path ncc_dir = output_path;
        ncc_dir /= "ncc_stats";

        std::string ncc_header = ",left_x,left_y,left_theta,"
                                 "right_x,right_y,right_theta,"
                                 "gt_right_x,gt_right_y,"
                                 "epipolar_a,epipolar_b,epipolar_c,"
                                 "ncc1,ncc2,ncc3,ncc4,score1,score2,final_score\n";
    
        veridical_csv = OpenCsvFile(ncc_dir,
            "image_pair_" + std::to_string(image_pair_index) + "_veridical_edges.csv",
            ncc_header);
    
        nonveridical_csv = OpenCsvFile(ncc_dir,
            "image_pair_" + std::to_string(image_pair_index) + "_nonveridical_edges.csv",
            ncc_header);

        std::filesystem::path epi_dir = output_path;
        epi_dir /= "epi_distance_stats";

        std::string epi_header = "left_edge_index,"
        "left_x,left_y,left_theta,"
        "right_x,right_y,right_theta,"
        "epipolar_a,epipolar_b,epipolar_c,"
        "epipolar_distance\n";
    
        epi_distance_csv = OpenCsvFile(epi_dir,
            "image_pair_" + std::to_string(image_pair_index) + "_epipolar_distances.csv",
            epi_header);
    }         

#pragma omp parallel
{
    int thread_id = omp_get_thread_num();    
    cv::Point2d ground_truth_edge;

    const int skip = 1;

    //> MARK: Start looping over left edges
    #pragma omp for schedule(static, omp_threads) reduction(+: epi_true_positive, epi_false_negative, epi_true_negative, disp_true_positive, disp_false_negative, shift_true_positive, shift_false_negative, cluster_true_positive, cluster_false_negative, cluster_true_negative, ncc_true_positive, ncc_false_negative, lowe_true_positive, lowe_false_negative, per_edge_epi_precision, per_edge_disp_precision, per_edge_shift_precision, per_edge_clust_precision, per_edge_ncc_precision, per_edge_lowe_precision, epi_edges_evaluated, disp_edges_evaluated, shift_edges_evaluated, clust_edges_evaluated, ncc_edges_evaluated, lowe_edges_evaluated)
    for (size_t i = 0; i < selected_primary_edges.size(); i += skip) {
        const auto& primary_edge = selected_primary_edges[i];
        const auto& primary_orientation = selected_primary_orientations[i];

        if (!selected_ground_truth_edges.empty()) {
            ground_truth_edge = selected_ground_truth_edges[i];
        }

        const auto& epipolar_line = epipolar_lines_secondary[i];
        const auto& primary_patch_one = primary_patch_set_one[i];
        const auto& primary_patch_two = primary_patch_set_two[i];

        double a = epipolar_line(0);
        double b = epipolar_line(1);
        double c = epipolar_line(2);

        if (std::abs(b) < 1e-6) continue;

        double a1_line = -a / b;
        double b1_line = -1;

        double m_epipolar = -a1_line / b1_line; 
        double angle_diff_rad = abs(primary_orientation - atan(m_epipolar));
        double angle_diff_deg = angle_diff_rad * (180.0 / M_PI);
        if (angle_diff_deg > 180) {
            angle_diff_deg -= 180;
        }

        //> TODO: Check if this needs to removed/changed
        bool primary_passes_tangency = (abs(angle_diff_deg - 0) > EPIP_TENGENCY_ORIENT_THRESH && abs(angle_diff_deg - 180) > EPIP_TENGENCY_ORIENT_THRESH) ? (true) : (false);
        if (!primary_passes_tangency) {
            continue;
        }

        ///////////////////////////////EPIPOLAR DISTANCE THRESHOLD///////////////////////////////
#if MEASURE_TIMINGS
        auto start_epi = std::chrono::high_resolution_clock::now();
#endif

        auto [secondary_candidate_edges, secondary_candidate_orientations, secondary_candidate_distances] =
            ExtractEpipolarEdges(epipolar_line, secondary_edge_coords, secondary_edge_orientations, 0.5);

        auto [test_secondary_candidate_edges, test_secondary_candidate_orientations, test_secondary_candidate_distances] =
            ExtractEpipolarEdges(epipolar_line, secondary_edge_coords, secondary_edge_orientations, 3.0);

        local_epi_input_counts[thread_id].push_back(secondary_edge_coords.size());

#if MEASURE_TIMINGS
        time_epi_edges_evaluated++;
        auto end_epi = std::chrono::high_resolution_clock::now();
        time_epi += std::chrono::duration<double, std::milli>(end_epi - start_epi).count();
#endif
        //> MARK: Epipolar Distance
        ///////////////////////////////EPIPOLAR DISTANCE THRESHOLD RECALL//////////////////////////
        if(!selected_ground_truth_edges.empty()){
            int epi_precision_numerator = 0;
            bool match_found = false;

            for (const auto& candidate : secondary_candidate_edges) {
                if (cv::norm(candidate - ground_truth_edge) <= 0.5) {
                    epi_precision_numerator++;
                    match_found = true;
                }
            }

            if (match_found) {
                epi_true_positive++;
            } 
            else {
                bool gt_match_found = false;
                for (const auto& test_candidate : test_secondary_candidate_edges) {
                    if (cv::norm(test_candidate - ground_truth_edge) <= 0.5) {
                        gt_match_found = true;
                        break;
                    }
                }

                if (!gt_match_found) {
                    epi_true_negative++;
                    continue;
                } else {
                    epi_false_negative++;
                }
            }
            if (!secondary_candidate_edges.empty()) {
                per_edge_epi_precision += static_cast<double>(epi_precision_numerator) / secondary_candidate_edges.size();
                epi_edges_evaluated++; 
            }
        }
        ///////////////////////////////MAXIMUM DISPARITY THRESHOLD//////////////////////////
#if MEASURE_TIMINGS
        auto start_disp = std::chrono::high_resolution_clock::now();
#endif
        
        local_epi_output_counts[thread_id].push_back(secondary_candidate_edges.size());

        std::vector<cv::Point2d> filtered_secondary_edge_coords;
        std::vector<double> filtered_secondary_edge_orientations;

        for (size_t j = 0; j < secondary_candidate_edges.size(); j++) {
            const cv::Point2d& secondary_edge = secondary_candidate_edges[j];

            double disparity = (!selected_ground_truth_edges.empty()) ? (primary_edge.x - secondary_edge.x) : (secondary_edge.x - primary_edge.x);

            bool within_horizontal = (disparity >= 0) && (disparity <= MAX_DISPARITY);
            bool within_vertical = std::abs(secondary_edge.y - primary_edge.y) <= MAX_DISPARITY;

            if (within_horizontal && within_vertical) {
                filtered_secondary_edge_coords.push_back(secondary_edge);
                filtered_secondary_edge_orientations.push_back(secondary_candidate_orientations[j]);
            }
        }

        local_disp_input_counts[thread_id].push_back(secondary_candidate_edges.size());

#if MEASURE_TIMINGS
        time_disp_edges_evaluated++;
        auto end_disp = std::chrono::high_resolution_clock::now();
        time_disp += std::chrono::duration<double, std::milli>(end_disp - start_disp).count();
#endif
        //> MARK: Maximum Disparity
        ///////////////////////////////MAXIMUM DISPARITY THRESHOLD RECALL//////////////////////////
        if (!selected_ground_truth_edges.empty()) {
            int disp_precision_numerator = 0;
            bool disp_match_found = false;

            for(const auto& filtered_candidate : filtered_secondary_edge_coords){
                if (cv::norm(filtered_candidate - ground_truth_edge) <= 0.5){
                    disp_precision_numerator++;
                    disp_match_found = true;
                }
            }

            if (disp_match_found){
                disp_true_positive++;
            } 
            else {
                disp_false_negative++;
            }
            if (!filtered_secondary_edge_coords.empty()) {
                per_edge_disp_precision += static_cast<double>(disp_precision_numerator) / filtered_secondary_edge_coords.size();
                disp_edges_evaluated++;
            }
        }
        ///////////////////////////////EPIPOLAR SHIFT THRESHOLD//////////////////////////
#if MEASURE_TIMINGS
        auto start_shift = std::chrono::high_resolution_clock::now();
#endif

        local_disp_output_counts[thread_id].push_back(filtered_secondary_edge_coords.size());

        Eigen::Vector3d eigen_primary_edge(primary_edge.x, primary_edge.y, primary_orientation);

        Eigen::MatrixXd eigen_secondary_edges(filtered_secondary_edge_coords.size(), 3);
        for (size_t i = 0; i < filtered_secondary_edge_coords.size(); ++i) {
            eigen_secondary_edges(i, 0) = filtered_secondary_edge_coords[i].x;
            eigen_secondary_edges(i, 1) = filtered_secondary_edge_coords[i].y;
            eigen_secondary_edges(i, 2) = filtered_secondary_edge_orientations[i];
        }

        Eigen::Vector3d epip_coeffs(a, b, c);

        std::vector<Eigen::Vector3d> corrected_edges = PerformEpipolarShift(
            eigen_primary_edge,
            eigen_secondary_edges,
            epip_coeffs);

        std::vector<cv::Point2d> shifted_secondary_edge_coords;
        std::vector<double> shifted_secondary_edge_orientations;

        for (const auto& edge : corrected_edges) {
            shifted_secondary_edge_coords.emplace_back(edge(0), edge(1));   
            shifted_secondary_edge_orientations.emplace_back(edge(2));
        }

        local_shift_input_counts[thread_id].push_back(filtered_secondary_edge_coords.size());

#if MEASURE_TIMINGS
        time_shift_edges_evaluated++;
        auto end_shift = std::chrono::high_resolution_clock::now();
        time_shift += std::chrono::duration<double, std::milli>(end_shift - start_shift).count();
#endif
        //> MARK: Epipolar Shift
        ///////////////////////////////EPIPOLAR SHIFT THRESHOLD RECALL//////////////////////////
        if (!selected_ground_truth_edges.empty()) {
            int shift_precision_numerator = 0;
            bool shift_match_found = false;

            for(const auto& shifted_candidate : shifted_secondary_edge_coords){
                if (cv::norm(shifted_candidate - ground_truth_edge) <= GT_SPATIAL_TOLERANCE){
                    shift_precision_numerator++;
                    shift_match_found = true;
                }
            }

            if (shift_match_found){
                shift_true_positive++;
            } 
            else {
                shift_false_negative++;
            }
            if (!shifted_secondary_edge_coords.empty()) {
                per_edge_shift_precision += static_cast<double>(shift_precision_numerator) / shifted_secondary_edge_coords.size();
                shift_edges_evaluated++;
            }
        }
        ///////////////////////////////EPIPOLAR CLUSTER THRESHOLD//////////////////////////
#if MEASURE_TIMINGS
        auto start_cluster = std::chrono::high_resolution_clock::now();
#endif

        local_shift_output_counts[thread_id].push_back(shifted_secondary_edge_coords.size());

        std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> clusters = ClusterEpipolarShiftedEdges(shifted_secondary_edge_coords, shifted_secondary_edge_orientations);
        std::vector<EdgeCluster> cluster_centers;

        for (size_t j = 0; j < clusters.size(); j++) {
            const auto& cluster_edges = clusters[j].first;
            const auto& cluster_orientations = clusters[j].second;

            if (cluster_edges.empty()) continue;

            cv::Point2d sum_point(0.0, 0.0);
            double sum_orientation = 0.0;

            for (size_t j = 0; j < cluster_edges.size(); ++j) {
                sum_point += cluster_edges[j];
            }

            for (size_t j = 0; j < cluster_orientations.size(); ++j) {
                sum_orientation += cluster_orientations[j];
            }

            cv::Point2d avg_point = sum_point * (1.0 / cluster_edges.size());
            double avg_orientation = sum_orientation * (1.0 / cluster_orientations.size());

            EdgeCluster cluster;
            cluster.center_coord = avg_point;
            cluster.center_orientation = avg_orientation;
            cluster.contributing_edges = cluster_edges;
            cluster.contributing_orientations = cluster_orientations;

            cluster_centers.push_back(cluster);
        }

        local_clust_input_counts[thread_id].push_back(shifted_secondary_edge_coords.size());

#if MEASURE_TIMINGS
        time_clust_edges_evaluated++;
        auto end_cluster = std::chrono::high_resolution_clock::now();
        time_cluster += std::chrono::duration<double, std::milli>(end_cluster - start_cluster).count();
#endif
        //> MARK: Clustering
        ///////////////////////////////EPIPOLAR CLUSTER THRESHOLD RECALL//////////////////////////
        if (!selected_ground_truth_edges.empty()) {
            int clust_precision_numerator = 0;
            bool cluster_match_found = false;

            for (const auto& cluster : cluster_centers) {
                if (cv::norm(cluster.center_coord - ground_truth_edge) <= GT_SPATIAL_TOLERANCE) {
                    clust_precision_numerator++;
                    cluster_match_found = true;
                }
            }

            if (cluster_match_found) {
                cluster_true_positive++;
            }
            else {
                cluster_false_negative++;
            }
            if (!cluster_centers.empty()) {
                per_edge_clust_precision += static_cast<double>(clust_precision_numerator) / cluster_centers.size();
                clust_edges_evaluated++;
            }
        }
        ///////////////////////////////EXTRACT PATCHES THRESHOLD////////////////////////////////////////////
#if MEASURE_TIMINGS
        auto start_patch = std::chrono::high_resolution_clock::now();
#endif

        local_clust_output_counts[thread_id].push_back(cluster_centers.size());

        std::vector<EdgeCluster> filtered_cluster_centers;
        std::vector<cv::Mat> secondary_patch_set_one;
        std::vector<cv::Mat> secondary_patch_set_two;

        int secondary_img_width  = secondary_image.cols;
        int secondary_img_height = secondary_image.rows;

        for (const auto& cluster : cluster_centers) {
            Edge edge;
            edge.location = cluster.center_coord;
            edge.orientation = cluster.center_orientation;

            auto [shifted_plus, shifted_minus] = get_Orthogonal_Shifted_Points(edge);

            if (!is_patch_in_bounds(shifted_plus, PATCH_HALF_SIZE, secondary_img_width, secondary_img_height) ||
                !is_patch_in_bounds(shifted_minus, PATCH_HALF_SIZE, secondary_img_width, secondary_img_height)) {
                continue;
            }

            cv::Mat patch_coord_x_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
            cv::Mat patch_coord_y_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
            cv::Mat patch_plus          = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

            cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
            cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
            cv::Mat patch_minus         = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

            get_patch_on_one_edge_side(
                shifted_plus,
                edge.orientation,
                patch_coord_x_plus,
                patch_coord_y_plus,
                patch_plus,
                secondary_image
            );

            get_patch_on_one_edge_side(
                shifted_minus,
                edge.orientation,
                patch_coord_x_minus,
                patch_coord_y_minus,
                patch_minus,
                secondary_image
            );

            cv::Mat patch_plus_32f, patch_minus_32f;
            patch_plus.convertTo(patch_plus_32f, CV_32F);
            patch_minus.convertTo(patch_minus_32f, CV_32F);
        
            filtered_cluster_centers.push_back(cluster);
            secondary_patch_set_one.push_back(patch_plus_32f);
            secondary_patch_set_two.push_back(patch_minus_32f);
        }

        local_patch_input_counts[thread_id].push_back(cluster_centers.size());

#if MEASURE_TIMINGS
        time_patch_edges_evaluated++;
        auto end_patch = std::chrono::high_resolution_clock::now();
        time_patch += std::chrono::duration<double, std::milli>(end_patch - start_patch).count();
        //> MARK: NCC
       ///////////////////////////////NCC THRESHOLD/////////////////////////////////////////////////////
       auto start_ncc = std::chrono::high_resolution_clock::now();
#endif

       local_patch_output_counts[thread_id].push_back(filtered_cluster_centers.size());

       int ncc_precision_numerator = 0;

       bool ncc_match_found = false;
       std::vector<EdgeMatch> passed_ncc_matches;

       if (!primary_patch_one.empty() && !primary_patch_two.empty() &&
           !secondary_patch_set_one.empty() && !secondary_patch_set_two.empty()) {

            for (size_t j = 0; j < filtered_cluster_centers.size(); ++j) {
                double ncc_one = ComputeNCC(primary_patch_one, secondary_patch_set_one[j]);
                double ncc_two = ComputeNCC(primary_patch_two, secondary_patch_set_two[j]);
                double ncc_three = ComputeNCC(primary_patch_one, secondary_patch_set_two[j]);
                double ncc_four = ComputeNCC(primary_patch_two, secondary_patch_set_one[j]);
 
                double score_one = std::min(ncc_one, ncc_two);
                double score_two = std::min(ncc_three, ncc_four);
                double final_score = std::max(score_one, score_two);

               if (final_score >= NCC_THRESH_FINAL_SCORE) {
                    EdgeMatch info;
                    info.coord = filtered_cluster_centers[j].center_coord;
                    info.orientation = filtered_cluster_centers[j].center_orientation;
                    info.final_score = final_score;
                    info.contributing_edges = filtered_cluster_centers[j].contributing_edges;
                    info.contributing_orientations = filtered_cluster_centers[j].contributing_orientations;
                    passed_ncc_matches.push_back(info);

                    if (!selected_ground_truth_edges.empty()) {
                        if (cv::norm(filtered_cluster_centers[j].center_coord - ground_truth_edge) <= GT_SPATIAL_TOLERANCE) {
                            ncc_match_found = true;
                            ncc_precision_numerator++;
                        }
                    }

                    if (forward_direction) {
                        std::ostream& target_stream = 
                            (!selected_ground_truth_edges.empty() &&
                             cv::norm(filtered_cluster_centers[j].center_coord - ground_truth_edge) <= GT_SPATIAL_TOLERANCE)
                            ? veridical_csv : nonveridical_csv;
                        
                        #pragma omp critical(csv_write)
                        {
                            target_stream << std::fixed << std::setprecision(8)
                                            << "," << primary_edge.x << "," << primary_edge.y << "," << primary_orientation << ","
                                            << info.coord.x << "," << info.coord.y << "," << info.orientation << ","
                                            << ground_truth_edge.x << "," << ground_truth_edge.y << ","
                                            << a << "," << b << "," << c << ","
                                            << ncc_one << "," << ncc_two << "," << ncc_three << "," << ncc_four << ","
                                            << score_one << "," << score_two << "," << final_score << "\n";
                        }                                            
                    }
               }
           }

           if (ncc_match_found) {
               ncc_true_positive++;
           } else {
               ncc_false_negative++;
           }
       }

       if (!passed_ncc_matches.empty()) {
           per_edge_ncc_precision += static_cast<double>(ncc_precision_numerator) / passed_ncc_matches.size();
           ncc_edges_evaluated++;
       }

       local_ncc_input_counts[thread_id].push_back(filtered_cluster_centers.size());
       local_ncc_output_counts[thread_id].push_back(passed_ncc_matches.size());

#if MEASURE_TIMINGS
        time_ncc_edges_evaluated++;
        auto end_ncc = std::chrono::high_resolution_clock::now();
        time_ncc += std::chrono::duration<double, std::milli>(end_ncc - start_ncc).count();
        //> MARK: Lowe's Ratio Test
        ///////////////////////////////LOWES RATIO TEST//////////////////////////////////////////////
        auto start_lowe = std::chrono::high_resolution_clock::now();
#endif
        local_lowe_input_counts[thread_id].push_back(passed_ncc_matches.size());

        int lowe_precision_numerator = 0;

        EdgeMatch best_match;
        double best_score = -1;

        if(passed_ncc_matches.size() >= 2){
            EdgeMatch second_best_match;
            double second_best_score = -1;

            for(const auto& match : passed_ncc_matches){
                if(match.final_score > best_score){
                    second_best_score = best_score;
                    second_best_match = best_match;

                    best_score = match.final_score;
                    best_match = match;
                }
                else if (match.final_score > second_best_score){
                    second_best_score = match.final_score;
                    second_best_match = match;
                }
            }
            double lowe_ratio = second_best_score / best_score;

            if (lowe_ratio < 1) {
                if (!selected_ground_truth_edges.empty()) {
                    local_GT_right_edges_after_lowe[thread_id].push_back(ground_truth_edge);
                    if (cv::norm(best_match.coord - ground_truth_edge) <= GT_SPATIAL_TOLERANCE) {
                        lowe_precision_numerator++;
                        lowe_true_positive++;
                    }
                    else {
                        lowe_false_negative++;
                    }
                }
                SourceEdge source_edge {primary_edge, primary_orientation};
                local_final_matches[thread_id].emplace_back(source_edge, best_match);
                local_lowe_output_counts[thread_id].push_back(1);
            }
            else {
                lowe_false_negative++;
                local_lowe_output_counts[thread_id].push_back(0);
            }
        }   
        else if (passed_ncc_matches.size() == 1){
            best_match = passed_ncc_matches[0];

            if (!selected_ground_truth_edges.empty()) {
                local_GT_right_edges_after_lowe[thread_id].push_back(ground_truth_edge);
                if (cv::norm(best_match.coord - ground_truth_edge) <= GT_SPATIAL_TOLERANCE) {
                    lowe_precision_numerator++;
                    lowe_true_positive++;
                } else {
                    lowe_false_negative++;
                }
            }
            
            SourceEdge source_edge {primary_edge, primary_orientation};
            local_final_matches[thread_id].emplace_back(source_edge, best_match);
            local_lowe_output_counts[thread_id].push_back(1);
        }
        else {
            lowe_false_negative++;
            local_lowe_output_counts[thread_id].push_back(0);
        }
        per_edge_lowe_precision += (static_cast<double>(lowe_precision_numerator) > 0) ? 1.0: 0.0;
        
        if (!passed_ncc_matches.empty()) {
            lowe_edges_evaluated++;
        }
#if MEASURE_TIMINGS
        time_lowe_edges_evaluated++;
        auto end_lowe = std::chrono::high_resolution_clock::now();
        time_lowe += std::chrono::duration<double, std::milli>(end_lowe - start_lowe).count();
#endif
    }   //> MARK: end of looping over left edges   
}
if (forward_direction) {
    veridical_csv.close();
    nonveridical_csv.close();
}

#if MEASURE_TIMINGS
    auto total_end = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
#endif
    
    double epi_distance_recall = 0.0;
    if ((epi_true_positive + epi_false_negative) > 0) {
        epi_distance_recall = static_cast<double>(epi_true_positive) / (epi_true_positive + epi_false_negative);
    }

    double max_disparity_recall = 0.0;
    if ((disp_true_positive + disp_false_negative) > 0) {
        max_disparity_recall = static_cast<double>(disp_true_positive) / (disp_true_positive + disp_false_negative);
    }

    double epi_shift_recall = 0.0;
    if ((shift_true_positive + shift_false_negative) > 0) {
        epi_shift_recall = static_cast<double>(shift_true_positive) / (shift_true_positive + shift_false_negative);
    }

    double epi_cluster_recall = 0.0;
    if ((cluster_true_positive + cluster_false_negative) > 0) {
        epi_cluster_recall = static_cast<double>(cluster_true_positive) / (cluster_true_positive + cluster_false_negative);
    }

    double ncc_recall = 0.0;
    if ((ncc_true_positive + ncc_false_negative) > 0) {
        ncc_recall = static_cast<double>(ncc_true_positive) / (ncc_true_positive + ncc_false_negative);
    }

    double lowe_recall = 0.0;
    if ((lowe_true_positive + lowe_false_negative) > 0) {
        lowe_recall = static_cast<double>(lowe_true_positive) / (lowe_true_positive + lowe_false_negative);
    }

    double per_image_epi_precision   = (epi_edges_evaluated > 0)   ? (per_edge_epi_precision / epi_edges_evaluated)     : (0.0);
    double per_image_disp_precision  = (disp_edges_evaluated > 0)  ? (per_edge_disp_precision / disp_edges_evaluated)   : (0.0);
    double per_image_shift_precision = (shift_edges_evaluated > 0) ? (per_edge_shift_precision / shift_edges_evaluated) : (0.0);
    double per_image_clust_precision = (clust_edges_evaluated > 0) ? (per_edge_clust_precision / clust_edges_evaluated) : (0.0);
    double per_image_ncc_precision   = (ncc_edges_evaluated > 0)   ? (per_edge_ncc_precision / ncc_edges_evaluated)     : (0.0);
    double per_image_lowe_precision  = (lowe_edges_evaluated > 0)  ? (per_edge_lowe_precision / lowe_edges_evaluated)   : (0.0);

    double per_image_epi_time = (time_epi_edges_evaluated > 0) ? (time_epi / time_epi_edges_evaluated) : (0.0);
    double per_image_disp_time = (time_disp_edges_evaluated > 0) ? (time_disp / time_disp_edges_evaluated) : 0.0;
    double per_image_shift_time = (time_shift_edges_evaluated > 0) ? (time_shift / time_shift_edges_evaluated) : 0.0;
    double per_image_clust_time = (time_clust_edges_evaluated > 0) ? (time_cluster / time_clust_edges_evaluated) : 0.0;
    double per_image_patch_time = (time_patch_edges_evaluated > 0) ? (time_patch / time_patch_edges_evaluated) : 0.0;
    double per_image_ncc_time = (time_ncc_edges_evaluated > 0) ? (time_ncc / time_ncc_edges_evaluated) : 0.0;
    double per_image_lowe_time = (time_lowe_edges_evaluated> 0) ? (time_lowe / time_lowe_edges_evaluated) : 0.0;
    double per_image_total_time = (selected_primary_edges.size() > 0) ? (total_time / selected_primary_edges.size()) : 0.0;

    //> CH: stack all local_final_matches to a global final_matches
    for (const auto& local_matches: local_final_matches) {
        final_matches.insert(final_matches.end(), local_matches.begin(), local_matches.end());
    }

    for (const auto& local_counts: local_epi_input_counts)  epi_input_counts.insert(epi_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_epi_output_counts) epi_output_counts.insert(epi_output_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_disp_input_counts) disp_input_counts.insert(disp_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_disp_output_counts) disp_output_counts.insert(disp_output_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_shift_input_counts) shift_input_counts.insert(shift_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_shift_output_counts) shift_output_counts.insert(shift_output_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_clust_input_counts) clust_input_counts.insert(clust_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_clust_output_counts) clust_output_counts.insert(clust_output_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_patch_input_counts) patch_input_counts.insert(patch_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_patch_output_counts) patch_output_counts.insert(patch_output_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_ncc_input_counts) ncc_input_counts.insert(ncc_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_ncc_output_counts) ncc_output_counts.insert(ncc_output_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_lowe_input_counts) lowe_input_counts.insert(lowe_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto& local_counts: local_lowe_output_counts) lowe_output_counts.insert(lowe_output_counts.end(), local_counts.begin(), local_counts.end());

    for (const auto& local_GT_right_edges_stack : local_GT_right_edges_after_lowe) {
        ground_truth_right_edges_after_lowe.insert(ground_truth_right_edges_after_lowe.end(), local_GT_right_edges_stack.begin(), local_GT_right_edges_stack.end());
    }

    return EdgeMatchResult {
        RecallMetrics {
            epi_distance_recall,
            max_disparity_recall,
            epi_shift_recall,
            epi_cluster_recall,
            ncc_recall,
            lowe_recall,
            epi_input_counts,
            epi_output_counts,
            disp_input_counts,
            disp_output_counts,
            shift_input_counts,
            shift_output_counts,
            clust_input_counts,
            clust_output_counts,
            patch_input_counts,
            patch_output_counts,
            ncc_input_counts,
            ncc_output_counts,
            lowe_input_counts,
            lowe_output_counts,
            per_image_epi_precision,
            per_image_disp_precision,
            per_image_shift_precision,
            per_image_clust_precision,
            per_image_ncc_precision,
            per_image_lowe_precision,
            lowe_true_positive,
            lowe_false_negative,
            per_image_epi_time,
            per_image_disp_time,
            per_image_shift_time,
            per_image_clust_time,
            per_image_patch_time,
            per_image_ncc_time,
            per_image_lowe_time,
            per_image_total_time
        },
        final_matches
    };
}  

bool Dataset::is_patch_in_bounds(const cv::Point2d& pt, int half_patch, int width, int height) {
    return pt.x - half_patch >= 0 && pt.x + half_patch < width &&
           pt.y - half_patch >= 0 && pt.y + half_patch < height;
}

std::pair<cv::Point2d, cv::Point2d> Dataset::get_Orthogonal_Shifted_Points(const Edge edgel)
{
    double shifted_x1 = edgel.location.x + ORTHOGONAL_SHIFT_MAG * (std::sin(edgel.orientation));
    double shifted_y1 = edgel.location.y + ORTHOGONAL_SHIFT_MAG * (-std::cos(edgel.orientation));
    double shifted_x2 = edgel.location.x + ORTHOGONAL_SHIFT_MAG * (-std::sin(edgel.orientation));
    double shifted_y2 = edgel.location.y + ORTHOGONAL_SHIFT_MAG * (std::cos(edgel.orientation));

    cv::Point2d shifted_point_plus(shifted_x1, shifted_y1);
    cv::Point2d shifted_point_minus(shifted_x2, shifted_y2);

    return {shifted_point_plus, shifted_point_minus};
}

void Dataset::get_patch_on_one_edge_side(cv::Point2d shifted_point, double theta, cv::Mat &patch_coord_x, cv::Mat &patch_coord_y, cv::Mat &patch_val, const cv::Mat img) 
{
    CV_Assert(img.type() == CV_64F);
    
    int half_patch_size = floor(PATCH_SIZE / 2);
    
    for (int i = -half_patch_size; i <= half_patch_size; i++) {
        for (int j = -half_patch_size; j <= half_patch_size; j++) {

            cv::Point2d rotated_point(cos(theta)*(i) - sin(theta)*(j) + shifted_point.x, sin(theta)*(i) + cos(theta)*(j) + shifted_point.y);
            patch_coord_x.at<double>(i + half_patch_size, j + half_patch_size) = rotated_point.x;
            patch_coord_y.at<double>(i + half_patch_size, j + half_patch_size) = rotated_point.y;

            double interp_val = Bilinear_Interpolation<double>(img, rotated_point);
            patch_val.at<double>(i + half_patch_size, j + half_patch_size) = interp_val;
        }
    }
}

double Dataset::getNormalDistance2EpipolarLine( Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &epiline_x, double &epiline_y ) {
	double a1_line = Epip_Line_Coeffs(0);
	double b1_line = Epip_Line_Coeffs(1);
	double c1_line = Epip_Line_Coeffs(2);
	epiline_x = edge(0) - a1_line * (a1_line * edge(0) + b1_line *edge(1) + c1_line)/(pow(a1_line,2) + pow(b1_line,2));
	epiline_y = edge(1) - b1_line* (a1_line * edge(0) + b1_line * edge(1) + c1_line)/(pow(a1_line,2) + pow(b1_line,2));
	return sqrt(pow(edge(0) - epiline_x, 2) + pow(edge(1) - epiline_y, 2));
}

double Dataset::getTangentialDistance2EpipolarLine( Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &x_intersection, double &y_intersection ) {
	double a_edgeH2 = tan(edge(2)); //tan(theta2)
	double b_edgeH2 = -1;
	double c_edgeH2 = -(a_edgeH2 * edge(0) - edge(1)); //−(a⋅x2−y2)
	double a1_line = Epip_Line_Coeffs(0);
	double b1_line = Epip_Line_Coeffs(1);
	double c1_line = Epip_Line_Coeffs(2);
	x_intersection = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
	y_intersection = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
	return sqrt((x_intersection - edge(0))*(x_intersection - edge(0))+(y_intersection - edge(1))*(y_intersection - edge(1)));
}

//> MARK: Perform Epipolar Shift
std::vector<Eigen::Vector3d> Dataset::PerformEpipolarShift(
    const Eigen::Vector3d& edge1,
    const Eigen::MatrixXd& edge2,
    const Eigen::Vector3d& epip_coeffs)
{
std::vector<Eigen::Vector3d> corrected_edges;

for (int i = 0; i < edge2.rows(); i++)
{
    Eigen::Vector3d xy1_H2(edge2(i, 0), edge2(i, 1), 1.0);

    double corrected_x, corrected_y, corrected_theta;
    double epiline_x, epiline_y;

    double normal_distance_epiline = getNormalDistance2EpipolarLine(epip_coeffs, xy1_H2, epiline_x, epiline_y);

    if (normal_distance_epiline < LOCATION_PERTURBATION)
    {
        corrected_x = epiline_x;
        corrected_y = epiline_y;
        corrected_theta = edge2(i, 2);
    }
    else
    {
        double x_intersection, y_intersection;
        Eigen::Vector3d isolated_H2(edge2(i, 0), edge2(i, 1), edge2(i, 2));
        double dist_diff_edg2 = getTangentialDistance2EpipolarLine(epip_coeffs, isolated_H2, x_intersection, y_intersection);

        double theta = edge2(i, 2);

        if (dist_diff_edg2 < EPIP_TANGENCY_DISPL_THRESH)
        {
            corrected_x = x_intersection;
            corrected_y = y_intersection;
            corrected_theta = theta;
        }
        else
        {
            double p_theta = epip_coeffs(0) * cos(theta) + epip_coeffs(1) * sin(theta);
            double derivative_p_theta = -epip_coeffs(0) * sin(theta) + epip_coeffs(1) * cos(theta);

            if (p_theta > 0 && derivative_p_theta < 0) theta -= ORIENT_PERTURBATION;
		    else if (p_theta < 0 && derivative_p_theta < 0) theta -= ORIENT_PERTURBATION;
		    else if (p_theta > 0 && derivative_p_theta > 0) theta += ORIENT_PERTURBATION;
		    else if (p_theta < 0 && derivative_p_theta > 0) theta += ORIENT_PERTURBATION;

            Eigen::Vector3d isolated_H2_(edge2(i, 0), edge2(i, 1), theta);
            dist_diff_edg2 = getTangentialDistance2EpipolarLine(epip_coeffs, isolated_H2_, x_intersection, y_intersection);

            if (dist_diff_edg2 < EPIP_TANGENCY_DISPL_THRESH)
            {
                corrected_x = x_intersection;
                corrected_y = y_intersection;
                corrected_theta = theta;
            }
            else
            {
                continue;
            }
        }
    }

    corrected_edges.emplace_back(corrected_x, corrected_y, corrected_theta);
}

return corrected_edges;
}


double Dataset::ComputeNCC(const cv::Mat& patch_one, const cv::Mat& patch_two){
    double mean_one = (cv::mean(patch_one))[0];
    double mean_two = (cv::mean(patch_two))[0];
    double sum_of_squared_one  = (cv::sum((patch_one - mean_one).mul(patch_one - mean_one))).val[0];
    double sum_of_squared_two  = (cv::sum((patch_two - mean_two).mul(patch_two - mean_two))).val[0];

    cv::Mat norm_one = (patch_one - mean_one) / sqrt(sum_of_squared_one);
    cv::Mat norm_two = (patch_two - mean_two) / sqrt(sum_of_squared_two);
    return norm_one.dot(norm_two);
}

std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> Dataset::ClusterEpipolarShiftedEdges(std::vector<cv::Point2d>& valid_shifted_edges, std::vector<double>& valid_shifted_orientations) {
    std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> clusters;
    
    if (valid_shifted_edges.empty() || valid_shifted_orientations.empty()) {
        return clusters;
    }

    std::vector<std::pair<cv::Point2d, double>> edge_orientation_pairs;
    for (size_t i = 0; i < valid_shifted_edges.size(); ++i) {
        edge_orientation_pairs.emplace_back(valid_shifted_edges[i], valid_shifted_orientations[i]);
    }

    std::sort(edge_orientation_pairs.begin(), edge_orientation_pairs.end(),
              [](const std::pair<cv::Point2d, double>& a, const std::pair<cv::Point2d, double>& b) {
                  return a.first.x < b.first.x;
              });

    valid_shifted_edges.clear();
    valid_shifted_orientations.clear();
    for (const auto& pair : edge_orientation_pairs) {
        valid_shifted_edges.push_back(pair.first);
        valid_shifted_orientations.push_back(pair.second);
    }

    std::vector<cv::Point2d> current_cluster_edges;
    std::vector<double> current_cluster_orientations;
    current_cluster_edges.push_back(valid_shifted_edges[0]);
    current_cluster_orientations.push_back(valid_shifted_orientations[0]);

    for (size_t i = 1; i < valid_shifted_edges.size(); ++i) {
        double distance = cv::norm(valid_shifted_edges[i] - valid_shifted_edges[i - 1]); 
        double orientation_difference = std::abs(valid_shifted_orientations[i] - valid_shifted_orientations[i - 1]);

        if (distance <= EDGE_CLUSTER_THRESH && orientation_difference < 5.0) {
            current_cluster_edges.push_back(valid_shifted_edges[i]);
            current_cluster_orientations.push_back(valid_shifted_orientations[i]);
        } else {
            clusters.emplace_back(current_cluster_edges, current_cluster_orientations);
            current_cluster_edges.clear();
            current_cluster_orientations.clear();
            current_cluster_edges.push_back(valid_shifted_edges[i]);
            current_cluster_orientations.push_back(valid_shifted_orientations[i]);
        }
    }

    if (!current_cluster_edges.empty()) {
        clusters.emplace_back(current_cluster_edges, current_cluster_orientations);
    }

    return clusters;
}

std::tuple<std::vector<cv::Point2d>, std::vector<double>, std::vector<double>> Dataset::ExtractEpipolarEdges(const Eigen::Vector3d& epipolar_line, const std::vector<cv::Point2d>& edge_locations, const std::vector<double>& edge_orientations, double distance_threshold) {
   std::vector<cv::Point2d> extracted_edges;
   std::vector<double> extracted_orientations;
   std::vector<double> extracted_distances;

   if (edge_locations.size() != edge_orientations.size()) {
       throw std::runtime_error("Edge locations and orientations size mismatch.");
   }

    for (size_t i = 0; i < edge_locations.size(); ++i) {
       const auto& edge = edge_locations[i];
       double x = edge.x;
       double y = edge.y;

       double distance = std::abs(epipolar_line(0) * x + epipolar_line(1) * y + epipolar_line(2))
                         / std::sqrt((epipolar_line(0) * epipolar_line(0)) + (epipolar_line(1) * epipolar_line(1)));

       if (distance < distance_threshold) {
           extracted_edges.push_back(edge);
           extracted_orientations.push_back(edge_orientations[i]);
           extracted_distances.push_back(distance);
       }
   }

   return {extracted_edges, extracted_orientations, extracted_distances};
}

std::vector<Eigen::Vector3d> Dataset::CalculateEpipolarLine(const Eigen::Matrix3d& fund_mat, const std::vector<cv::Point2d>& edges) {
   std::vector<Eigen::Vector3d> epipolar_lines;

   for (const auto& point : edges) {
       Eigen::Vector3d homo_point(point.x, point.y, 1.0); 

       Eigen::Vector3d epipolar_line = fund_mat * homo_point;

       epipolar_lines.push_back(epipolar_line);
   }

   return epipolar_lines;
}

std::tuple<std::vector<cv::Point2d>, std::vector<double>, std::vector<cv::Point2d>> Dataset::PickRandomEdges(int patch_size, const std::vector<cv::Point2d>& edges, const std::vector<cv::Point2d>& ground_truth_right_edges, const std::vector<double>& orientations, size_t num_points, int img_width, int img_height) {
    std::vector<cv::Point2d> valid_edges;
    std::vector<double> valid_orientations;
    std::vector<cv::Point2d> valid_ground_truth_edges;

    if (edges.size() != orientations.size() || edges.size() != ground_truth_right_edges.size()) {
        throw std::runtime_error("Edge locations, orientations, and ground truth edges size mismatch.");
    }

    for (size_t i = 0; i < edges.size(); ++i) {
        const auto& edge = edges[i];
        if (edge.x >= PATCH_HALF_SIZE && edge.x < img_width - PATCH_HALF_SIZE &&
            edge.y >= PATCH_HALF_SIZE && edge.y < img_height - PATCH_HALF_SIZE) {
            valid_edges.push_back(edge);
            valid_orientations.push_back(orientations[i]);
            valid_ground_truth_edges.push_back(ground_truth_right_edges[i]);
        }
    }

    num_points = std::min(num_points, valid_edges.size());

    std::vector<cv::Point2d> selected_points;
    std::vector<double> selected_orientations;
    std::vector<cv::Point2d> selected_ground_truth_points;
    std::unordered_set<int> used_indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, valid_edges.size() - 1);

    while (selected_points.size() < num_points) {
        int index = dis(gen);
        if (used_indices.find(index) == used_indices.end()) {
            selected_points.push_back(valid_edges[index]);
            selected_orientations.push_back(valid_orientations[index]);
            selected_ground_truth_points.push_back(valid_ground_truth_edges[index]);
            used_indices.insert(index);
        }
    }

    return {selected_points, selected_orientations, selected_ground_truth_points};
}

void Dataset::VisualizeGTRightEdge(const cv::Mat &left_image, const cv::Mat &right_image, const std::vector<std::pair<cv::Point2d, cv::Point2d>> &left_right_edges) {
    cv::Mat left_visualization, right_visualization;
    cv::cvtColor(left_image, left_visualization, cv::COLOR_GRAY2BGR);
    cv::cvtColor(right_image, right_visualization, cv::COLOR_GRAY2BGR);

    std::vector<cv::Scalar> vibrant_colors = {
        cv::Scalar(255, 0, 0),    
        cv::Scalar(0, 255, 0),   
        cv::Scalar(0, 0, 255),   
        cv::Scalar(255, 255, 0), 
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 165, 0), 
        cv::Scalar(128, 0, 128),
        cv::Scalar(0, 128, 255),
        cv::Scalar(255, 20, 147)
    };

    std::vector<std::pair<cv::Point2d, cv::Point2d>> sampled_pairs;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, left_right_edges.size() - 1);

    int num_samples = std::min(10, static_cast<int>(left_right_edges.size()));
    for (int i = 0; i < num_samples; ++i) {
        sampled_pairs.push_back(left_right_edges[distr(gen)]);
    }

    for (size_t i = 0; i < sampled_pairs.size(); ++i) {
        const auto &[left_edge, right_edge] = sampled_pairs[i];

        cv::Scalar color = vibrant_colors[i % vibrant_colors.size()];

        cv::circle(left_visualization, left_edge, 5, color, cv::FILLED); 
        cv::circle(right_visualization, right_edge, 5, color, cv::FILLED); 
    }

    cv::Mat merged_visualization;
    cv::hconcat(left_visualization, right_visualization, merged_visualization);

    cv::imshow("Ground Truth Disparity Edges Visualization", merged_visualization);
    cv::waitKey(0);
}

void Dataset::CalculateGTRightEdge(const std::vector<cv::Point2d> &left_third_order_edges_locations, const std::vector<double> &left_third_order_edges_orientation, const cv::Mat &disparity_map, const cv::Mat &left_image, const cv::Mat &right_image) {
    forward_gt_data.clear();

    static size_t total_rows_written = 0;
    static int file_index = 1;
    static std::ofstream csv_file;
    static const size_t max_rows_per_file = 1'000'000;

    if (!csv_file.is_open()) {
        std::string filename = "valid_disparities_part_" + std::to_string(file_index) + ".csv";
        csv_file.open(filename, std::ios::out);
    }

    for (size_t i = 0; i < left_third_order_edges_locations.size(); i++) {
        const cv::Point2d &left_edge = left_third_order_edges_locations[i];
        double orientation = left_third_order_edges_orientation[i];

        double disparity = Bilinear_Interpolation(disparity_map, left_edge);

        if (std::isnan(disparity) || std::isinf(disparity) || disparity < 0) {
            continue;
        }

        cv::Point2d right_edge(left_edge.x - disparity, left_edge.y);
        forward_gt_data.emplace_back(left_edge, right_edge, orientation);

        if (total_rows_written >= max_rows_per_file) {
            csv_file.close();
            ++file_index;
            total_rows_written = 0;
            std::string next_filename = "valid_disparities_part_" + std::to_string(file_index) + ".csv";
            csv_file.open(next_filename, std::ios::out);
        }

        csv_file << disparity << "\n";
        ++total_rows_written;
    }

    csv_file.flush();
}

void Dataset::CalculateGTLeftEdge(const std::vector<cv::Point2d>& right_third_order_edges_locations,const std::vector<double>& right_third_order_edges_orientation,const cv::Mat& disparity_map_right_reference,const cv::Mat& left_image,const cv::Mat& right_image) {
    reverse_gt_data.clear();

    static size_t total_rows_written = 0;
    static int file_index = 1;
    static std::ofstream csv_file;
    static const size_t max_rows_per_file = 1'000'000;

    if (!csv_file.is_open()) {
        std::string filename = "valid_reverse_disparities_part_" + std::to_string(file_index) + ".csv";
        csv_file.open(filename, std::ios::out);
    }

    for (size_t i = 0; i < right_third_order_edges_locations.size(); i++) {
        const cv::Point2d& right_edge = right_third_order_edges_locations[i];
        double orientation = right_third_order_edges_orientation[i];

        double disparity = Bilinear_Interpolation(disparity_map_right_reference, right_edge);

        if (std::isnan(disparity) || std::isinf(disparity) || disparity < 0) {
            continue;
        }

        cv::Point2d left_edge(right_edge.x + disparity, right_edge.y);

        reverse_gt_data.emplace_back(right_edge, left_edge, orientation);

        if (total_rows_written >= max_rows_per_file) {
            csv_file.close();
            ++file_index;
            total_rows_written = 0;
            std::string next_filename = "valid_reverse_disparities_part_" + std::to_string(file_index) + ".csv";
            csv_file.open(next_filename, std::ios::out);
        }

        csv_file << disparity << "\n";
        ++total_rows_written;
    }

    csv_file.flush();
}

std::vector<std::pair<cv::Mat, cv::Mat>> Dataset::LoadEuRoCImages(const std::string& csv_path, const std::string& left_path, const std::string& right_path,
   int num_pairs) {
   std::ifstream csv_file(csv_path);
   if (!csv_file.is_open()) {
       std::cerr << "ERROR: Could not open the CSV file located at " << csv_path << "!" << std::endl;
       return {};
   }

   std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;
   std::string line;
   bool first_line = true;

   while (std::getline(csv_file, line) && image_pairs.size() < num_pairs) {
       if (first_line) {
           first_line = false;
           continue;
       }

       std::istringstream line_stream(line);
       std::string timestamp;
       std::getline(line_stream, timestamp, ',');

       Img_time_stamps.push_back( std::stod(timestamp) );
      
       std::string left_img_path = left_path + timestamp + ".png";
       std::string right_img_path = right_path + timestamp + ".png";
      
       cv::Mat curr_left_img = cv::imread(left_img_path, cv::IMREAD_GRAYSCALE);
       cv::Mat curr_right_img = cv::imread(right_img_path, cv::IMREAD_GRAYSCALE);
      
       if (curr_left_img.empty() || curr_right_img.empty()) {
           std::cerr << "ERROR: Could not load the images: " << left_img_path << " or " << right_img_path << "!" << std::endl;
           continue;
       }
      
       image_pairs.emplace_back(curr_left_img, curr_right_img);
   }
  
   csv_file.close();
   return image_pairs;
}

std::vector<std::pair<cv::Mat, cv::Mat>> Dataset::LoadETH3DImages(const std::string &stereo_pairs_path, int num_pairs) {
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;

    std::vector<std::string> stereo_folders;
    for (const auto &entry : std::filesystem::directory_iterator(stereo_pairs_path)) {
        if (entry.is_directory()) {
            stereo_folders.push_back(entry.path().string());
        }
    }

    std::sort(stereo_folders.begin(), stereo_folders.end());

    for (int i = 0; i < std::min(num_pairs, static_cast<int>(stereo_folders.size())); ++i) {
        std::string folder_path = stereo_folders[i];

        std::string left_image_path = folder_path + "/im0.png";
        std::string right_image_path = folder_path + "/im1.png";

        cv::Mat left_image = cv::imread(left_image_path, cv::IMREAD_GRAYSCALE);
        cv::Mat right_image = cv::imread(right_image_path, cv::IMREAD_GRAYSCALE);

        if (!left_image.empty() && !right_image.empty()) {
            StereoImageData entry;
            entry.folder_path = folder_path;
            entry.left_image_path = left_image_path;
            entry.right_image_path = right_image_path;
            stereo_image_data.push_back(entry);

            image_pairs.emplace_back(left_image, right_image);
        } else {
            std::cerr << "ERROR: Could not load images from folder: " << folder_path << std::endl;
        }
    }

    return image_pairs;
}

std::vector<cv::Mat> Dataset::LoadETH3DLeftReferenceMaps(const std::string &stereo_pairs_path, int num_maps) {
    std::vector<cv::Mat> disparity_maps;
    std::vector<std::string> stereo_folders;

    for (const auto &entry : std::filesystem::directory_iterator(stereo_pairs_path)) {
        if (entry.is_directory()) {
            stereo_folders.push_back(entry.path().string());
        }
    }

    std::sort(stereo_folders.begin(), stereo_folders.end());

    for (int i = 0; i < std::min(num_maps, static_cast<int>(stereo_folders.size())); i++) {
        std::string folder_path = stereo_folders[i];
        std::string disparity_csv_path = folder_path + "/disparity_map.csv";
        std::string disparity_bin_path = folder_path + "/disparity_map.bin";

        cv::Mat disparity_map;

        if (std::filesystem::exists(disparity_bin_path)) {
            // std::cout << "Loading disparity data from: " << disparity_bin_path << std::endl;
            disparity_map = ReadDisparityFromBinary(disparity_bin_path);
        } else {
            // std::cout << "Parsing and storing disparity data from: " << disparity_csv_path << std::endl;
            disparity_map = LoadDisparityFromCSV(disparity_csv_path);
            if (!disparity_map.empty()) {
                WriteDisparityToBinary(disparity_bin_path, disparity_map);
                // std::cout << "Saved disparity data to: " << disparity_bin_path << std::endl;
            }
        }

        if (!disparity_map.empty()) {
            disparity_maps.push_back(disparity_map);
        }
    }

    return disparity_maps;
}

void Dataset::WriteDisparityToBinary(const std::string& filepath, const cv::Mat& disparity_map) {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "ERROR: Could not write disparity to: " << filepath << std::endl;
        return;
    }

    int rows = disparity_map.rows;
    int cols = disparity_map.cols;
    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    ofs.write(reinterpret_cast<const char*>(disparity_map.ptr<float>(0)), sizeof(float) * rows * cols);
}

cv::Mat Dataset::ReadDisparityFromBinary(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "ERROR: Could not read disparity from: " << filepath << std::endl;
        return {};
    }

    int rows, cols;
    ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    cv::Mat disparity_map(rows, cols, CV_32F);
    ifs.read(reinterpret_cast<char*>(disparity_map.ptr<float>(0)), sizeof(float) * rows * cols);

    return disparity_map;
}

cv::Mat Dataset::LoadDisparityFromCSV(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open disparity CSV: " << path << std::endl;
        return {};
    }

    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        while (std::getline(ss, value, ',')) {
            try {
                float d = std::stof(value);

                if (value == "nan" || value == "NaN") {
                    d = std::numeric_limits<float>::quiet_NaN();
                } else if (value == "inf" || value == "Inf") {
                    d = std::numeric_limits<float>::infinity();
                } else if (value == "-inf" || value == "-Inf") {
                    d = -std::numeric_limits<float>::infinity();
                }

                row.push_back(d);
            } catch (const std::exception &e) {
                    std::cerr << "WARNING: Invalid value in file: " << path << " -> " << value << std::endl;
                    row.push_back(std::numeric_limits<float>::quiet_NaN());
            }
        }

        if (!row.empty()) data.push_back(row);
    }

    int rows = data.size();
    int cols = data[0].size();
    cv::Mat disparity(rows, cols, CV_32F);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            disparity.at<float>(r, c) = data[r][c];
        }
    }

    return disparity;
}

void Dataset::WriteEdgesToBinary(const std::string& filepath,
                                  const std::vector<cv::Point2d>& locations,
                                  const std::vector<double>& orientations) {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "ERROR: Could not open binary file for writing: " << filepath << std::endl;
        return;
    }

    size_t size = locations.size();
    ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
    ofs.write(reinterpret_cast<const char*>(locations.data()), sizeof(cv::Point2d) * size);
    ofs.write(reinterpret_cast<const char*>(orientations.data()), sizeof(double) * size);
}

void Dataset::ReadEdgesFromBinary(const std::string& filepath,
                                   std::vector<cv::Point2d>& locations,
                                   std::vector<double>& orientations) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "ERROR: Could not open binary file for reading: " << filepath << std::endl;
        return;
    }

    size_t size = 0;
    ifs.read(reinterpret_cast<char*>(&size), sizeof(size));

    locations.resize(size);
    orientations.resize(size);

    ifs.read(reinterpret_cast<char*>(locations.data()), sizeof(cv::Point2d) * size);
    ifs.read(reinterpret_cast<char*>(orientations.data()), sizeof(double) * size);
}

void Dataset::ProcessEdges(const cv::Mat& image,
                           const std::string& filepath,
                           std::shared_ptr<ThirdOrderEdgeDetectionCPU>& toed,
                           std::vector<cv::Point2d>& locations,
                           std::vector<double>& orientations) {
    std::string path = filepath + ".bin";

    if (std::filesystem::exists(path)) {
        // std::cout << "Loading edge data from: " << path << std::endl;
        ReadEdgesFromBinary(path, locations, orientations);
    } else {
        // std::cout << "Running third-order edge detector..." << std::endl;
        toed->get_Third_Order_Edges(image);
        locations = toed->toed_locations;
        orientations = toed->toed_orientations;

        WriteEdgesToBinary(path, locations, orientations);
        // std::cout << "Saved edge data to: " << path << std::endl;
    }
}

void Dataset::Load_GT_Poses( std::string GT_Poses_File_Path ) {
   std::ifstream gt_pose_file(GT_Poses_File_Path);
   if (!gt_pose_file.is_open()) {
       LOG_FILE_ERROR(GT_Poses_File_Path);
       exit(1);
   }

   std::string line;
   bool b_first_line = true;
   if (dataset_type == "EuRoC") {
       Eigen::Matrix4d Transf_frame2body;
       Eigen::Matrix4d inv_Transf_frame2body;
       Transf_frame2body.setIdentity();
       Transf_frame2body.block<3,3>(0,0) = rot_frame2body_left;
       Transf_frame2body.block<3,1>(0,3) = transl_frame2body_left;
       inv_Transf_frame2body = Transf_frame2body.inverse();

       Eigen::Matrix4d Transf_Poses;
       Eigen::Matrix4d inv_Transf_Poses;
       Transf_Poses.setIdentity();

       Eigen::Matrix4d frame2world;

       while (std::getline(gt_pose_file, line)) {
           if (b_first_line) {
               b_first_line = false;
               continue;
           }

           std::stringstream ss(line);
           std::string gt_val;
           std::vector<double> csv_row_val;

           while (std::getline(ss, gt_val, ',')) {
               try {
                   csv_row_val.push_back(std::stod(gt_val));
               } catch (const std::invalid_argument& e) {
                   std::cerr << "Invalid argument: " << e.what() << " for value (" << gt_val << ") from the file " << GT_Poses_File_Path << std::endl;
               } catch (const std::out_of_range& e) {
                    std::cerr << "Out of range exception: " << e.what() << " for value: " << gt_val << std::endl;
               }
           }

           GT_time_stamps.push_back(csv_row_val[0]);
           Eigen::Vector3d transl_val( csv_row_val[1], csv_row_val[2], csv_row_val[3] );
           Eigen::Quaterniond quat_val( csv_row_val[4], csv_row_val[5], csv_row_val[6], csv_row_val[7] );
           Eigen::Matrix3d rot_from_quat = quat_val.toRotationMatrix();

           Transf_Poses.block<3,3>(0,0) = rot_from_quat;
           Transf_Poses.block<3,1>(0,3) = transl_val;
           inv_Transf_Poses = Transf_Poses.inverse();

           frame2world = (inv_Transf_frame2body*inv_Transf_Poses).inverse();

           unaligned_GT_Rot.push_back(frame2world.block<3,3>(0,0));
           unaligned_GT_Transl.push_back(frame2world.block<3,1>(0,3));
       }
   }
   else {
       LOG_ERROR("Dataset type not supported!");
   }
}

void Dataset::Align_Images_and_GT_Poses() {
   std::vector<double> time_stamp_diff_val;
   std::vector<unsigned> time_stamp_diff_indx;
   for (double img_time_stamp : Img_time_stamps) {
       time_stamp_diff_val.clear();
       for ( double gt_time_stamp : GT_time_stamps) {
           time_stamp_diff_val.push_back(std::abs(img_time_stamp - gt_time_stamp));
       }
       auto min_diff = std::min_element(std::begin(time_stamp_diff_val), std::end(time_stamp_diff_val));
       int min_index;
       if (min_diff != time_stamp_diff_val.end()) {
           min_index = std::distance(std::begin(time_stamp_diff_val), min_diff);
       } else {
           LOG_ERROR("Empty vector for time stamp difference vector");
       }

       aligned_GT_Rot.push_back(unaligned_GT_Rot[min_index]);
       aligned_GT_Transl.push_back(unaligned_GT_Transl[min_index]);
   }

}

Eigen::Matrix3d Dataset::ConvertToEigenMatrix(const std::vector<std::vector<double>>& matrix) {
   Eigen::Matrix3d eigen_matrix;
   for (int i = 0; i < 3; i++) {
       for (int j = 0; j < 3; j++) {
           eigen_matrix(i, j) = matrix[i][j];
       }
   }
   return eigen_matrix;
}

void Dataset::PrintDatasetInfo() {
    std::cout << "Left Camera Resolution: " << left_res[0] << "x" << left_res[1] << std::endl;
    std::cout << "\nRight Camera Resolution: " << right_res[0] << "x" << right_res[1] << std::endl;

    std::cout << "\nLeft Camera Intrinsics: ";
    for (const auto& value : left_intr) std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nRight Camera Intrinsics: ";
    for (const auto& value : right_intr) std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nStereo Extrinsic Parameters (Left to Right): \n";

    std::cout << "\nRotation Matrix: \n";
    for (const auto& row : rot_mat_21) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nTranslation Vector: \n";
    for (const auto& value : trans_vec_21) std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nFundamental Matrix: \n";
    for (const auto& row : fund_mat_21) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nStereo Extrinsic Parameters (Right to Left): \n";

    std::cout << "\nRotation Matrix: \n";
    for (const auto& row : rot_mat_12) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nTranslation Vector: \n";
    for (const auto& value : trans_vec_12) std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nFundamental Matrix: \n";
    for (const auto& row : fund_mat_12) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nStereo Camera Parameters: \n";
    std::cout << "Focal Length: " << focal_length << " pixels" << std::endl;
    std::cout << "Baseline: " << baseline << " meters" << std::endl;

    std::cout << "\n" << std::endl;
}

void Dataset::onMouse(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_MOUSEMOVE) {
        if (merged_visualization_global.empty()) return;

        int left_width = merged_visualization_global.cols / 2; 

        std::string coord_text;
        if (x < left_width) {
            coord_text = "Left Image: (" + std::to_string(x) + ", " + std::to_string(y) + ")";
        } else {
            int right_x = x - left_width;
            coord_text = "Right Image: (" + std::to_string(right_x) + ", " + std::to_string(y) + ")";
        }

        cv::Mat display = merged_visualization_global.clone();
        cv::putText(display, coord_text, cv::Point(x, y), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        cv::imshow("Edge Matching Using NCC & Bidirectional Consistency", display);
    }
}

std::ofstream Dataset::OpenCsvFile(
    const std::filesystem::path& directory,
    const std::string& filename,
    const std::string& header
) {
    std::filesystem::create_directories(directory);

    std::filesystem::path filepath = directory / filename;

    std::ofstream csv_file(filepath.string());
    if (!csv_file) {
        std::cerr << "WARNING: Failed to open CSV file: " << filepath << std::endl;
        return {};
    }

    csv_file << header;
    return csv_file;
}

#endif