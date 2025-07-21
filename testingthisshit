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


cv::Mat merged_visualization_global;

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
    int num_pairs = 100000;
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;
    std::vector<cv::Mat> left_ref_disparity_maps;
    // std::vector<cv::Mat> right_ref_disparity_maps;
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
        // right_ref_disparity_maps = LoadETH3DRightReferenceMaps(stereo_pairs_path, num_pairs);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    LOG_INFO("Start looping over all image pairs");

    std::cout << "There are " << image_pairs.size() << " image pairs" << std::endl;

    //Figure out a solution for this (will skip the last image pair)
    for (size_t i = 0; i < image_pairs.size()-1; ++i) {
        const cv::Mat& curr_left_img = image_pairs[i].first;
        const cv::Mat& curr_right_img = image_pairs[i].second;

        const cv::Mat& left_ref_map = left_ref_disparity_maps[i]; 
        // const cv::Mat& right_ref_map = right_ref_disparity_maps[i];

        const cv::Mat& next_left_img = image_pairs[i + 1].first;
        const cv::Mat& next_right_img = image_pairs[i + 1].second;

        std::vector<cv::Mat> curr_left_pyramid, curr_right_pyramid;
        std::vector<cv::Mat> next_left_pyramid, next_right_pyramid;
        int pyramid_levels = 4;

        BuildImagePyramids(
            curr_left_img,
            curr_right_img,
            next_left_img,
            next_right_img,
            pyramid_levels,
            curr_left_pyramid,
            curr_right_pyramid,
            next_left_pyramid,
            next_right_pyramid
        );

        ncc_one_vs_err.clear();
        ncc_two_vs_err.clear();
        ground_truth_right_edges_after_lowe.clear();

        std::cout << "Image Pair #" << i << "\n";
        cv::Mat left_calib = (cv::Mat_<double>(3, 3) << left_intr[0], 0, left_intr[2], 0, left_intr[1], left_intr[3], 0, 0, 1);
        cv::Mat right_calib = (cv::Mat_<double>(3, 3) << right_intr[0], 0, right_intr[2], 0, right_intr[1], right_intr[3], 0, 0, 1);
        cv::Mat left_dist_coeff_mat(left_dist_coeffs);
        cv::Mat right_dist_coeff_mat(right_dist_coeffs);

        cv::Mat left_undistorted, right_undistorted;
        cv::undistort(curr_left_img, left_undistorted, left_calib, left_dist_coeff_mat);
        cv::undistort(curr_right_img, right_undistorted, right_calib, right_dist_coeff_mat);

        if (Total_Num_Of_Imgs == 0) {
            img_height = left_undistorted.rows;
            img_width  = left_undistorted.cols;
            TOED = std::shared_ptr<ThirdOrderEdgeDetectionCPU>(new ThirdOrderEdgeDetectionCPU( img_height, img_width ));
        }

        std::string edge_dir = output_path + "/edges";
        std::filesystem::create_directories(edge_dir);

        std::string left_edge_path = edge_dir + "/left_edges_" + std::to_string(i);
        std::string right_edge_path = edge_dir + "/right_edges_" + std::to_string(i);

        ProcessEdges(left_undistorted, left_edge_path, TOED, left_third_order_edges_locations, left_third_order_edges_orientation);
        std::cout << "Number of edges on the left image: " << left_third_order_edges_locations.size() << std::endl;

        ProcessEdges(right_undistorted, right_edge_path, TOED, right_third_order_edges_locations, right_third_order_edges_orientation);
        std::cout << "Number of edges on the right image: " << right_third_order_edges_locations.size() << std::endl;

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
        // CalculateGTLeftEdge(right_third_order_edges_locations, right_third_order_edges_orientation, right_ref_map, left_edge_map, right_edge_map);

        StereoMatchResult match_result = DisplayMatches(
            left_undistorted,
            right_undistorted,
            right_third_order_edges_locations,
            right_third_order_edges_orientation
        );

#if 0
        std::vector<cv::Point3d> points_opencv = Calculate3DPoints(match_result.confirmed_matches);
        std::vector<cv::Point3d> points_linear = LinearTriangulatePoints(match_result.confirmed_matches);
        std::vector<Eigen::Vector3d> orientations_3d = Calculate3DOrientations(match_result.confirmed_matches);

        std::vector<OrientedPoint3D> oriented_points;

        for (size_t i = 0; i < points_opencv.size(); ++i) {
            OrientedPoint3D op;
            op.position = points_opencv[i];
            op.orientation = orientations_3d[i];
            oriented_points.push_back(op);
        }


        if (points_opencv.size() != points_linear.size()) {
            std::cerr << "Mismatch in number of 3D points: OpenCV=" << points_opencv.size()
                    << ", Linear=" << points_linear.size() << "\n";
        } else {
            std::cout << "Comparing " << points_opencv.size() << " 3D points...\n";

            double total_error = 0.0;
            for (size_t i = 0; i < points_opencv.size(); ++i) {
                const auto& pt1 = points_opencv[i];
                const auto& pt2 = points_linear[i];

                double error = cv::norm(pt1 - pt2);
                total_error += error;

                std::cout << "Point " << i << ": OpenCV = [" << pt1 << "], "
                        << "Linear = [" << pt2 << "], "
                        << "Error = " << error << "\n";
            }

            std::cout << "Average triangulation error: "
                    << (total_error / points_opencv.size()) << " units.\n";
        }
#endif

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

StereoMatchResult Dataset::DisplayMatches(const cv::Mat& left_image, const cv::Mat& right_image, std::vector<cv::Point2d> right_edge_coords, std::vector<double> right_edge_orientations) {
    ///////////////////////////////FORWARD DIRECTION///////////////////////////////
    std::vector<cv::Point2d> left_edge_coords;
    std::vector<cv::Point2d> ground_truth_right_edges;
    std::vector<double> left_edge_orientations;

    for (const auto& data : forward_gt_data) {
        left_edge_coords.push_back(std::get<0>(data)); 
        ground_truth_right_edges.push_back(std::get<1>(data)); 
        left_edge_orientations.push_back(std::get<2>(data)); 
    }

    auto [left_orthogonal_one, left_orthogonal_two] = CalculateOrthogonalShifts(left_edge_coords, left_edge_orientations, ORTHOGONAL_SHIFT_MAG);

    std::vector<cv::Point2d> filtered_left_edges;
    std::vector<double> filtered_left_orientations;
    std::vector<cv::Point2d> filtered_ground_truth_right_edges;

    std::vector<cv::Mat> left_patch_set_one;
    std::vector<cv::Mat> left_patch_set_two;

    ExtractPatches(
        PATCH_SIZE,
        left_image,
        left_edge_coords,
        left_edge_orientations,
        left_orthogonal_one,
        left_orthogonal_two,
        filtered_left_edges,
        filtered_left_orientations,
        left_patch_set_one,
        left_patch_set_two,
        &ground_truth_right_edges,
        &filtered_ground_truth_right_edges
    );

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
        right_image,
        filtered_ground_truth_right_edges
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

    auto [right_orthogonal_one, right_orthogonal_two] = CalculateOrthogonalShifts(reverse_primary_edges, reverse_primary_orientations, ORTHOGONAL_SHIFT_MAG);

    std::vector<cv::Point2d> filtered_right_edges;
    std::vector<double> filtered_right_orientations;
    std::vector<cv::Point2d> filtered_ground_truth_left_edges;

    std::vector<cv::Mat> right_patch_set_one;
    std::vector<cv::Mat> right_patch_set_two;

    ExtractPatches(
        PATCH_SIZE,
        right_image,
        reverse_primary_edges,
        reverse_primary_orientations,
        right_orthogonal_one,
        right_orthogonal_two,
        filtered_right_edges,
        filtered_right_orientations,
        right_patch_set_one,
        right_patch_set_two,
        nullptr,
        nullptr
    );

    std::vector<Eigen::Vector3d> epipolar_lines_left = CalculateEpipolarLine(fundamental_matrix_12, filtered_right_edges);

    EdgeMatchResult reverse_match = CalculateMatches(
        filtered_right_edges,
        filtered_right_orientations,
        left_edge_coords,
        left_edge_orientations,
        right_patch_set_one,
        right_patch_set_two,
        epipolar_lines_left,
        left_image
    );

    std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>> confirmed_matches;

    int matches_before_bct = static_cast<int>(forward_match.edge_to_cluster_matches.size());
    std::cout << "Number of matches before BCT: " << matches_before_bct << std::endl;

    auto bct_start = std::chrono::high_resolution_clock::now();

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

    // //> Measure the recall of bidirectional consistency test
    // int bct_true_positive = 0;
    // for (int i = 0; i < confirmed_matches.size(); i++) {
    //     ConfirmedMatchEdge right_confirmed = confirmed_matches[i].second;
    //     cv::Point2d right_edge_location = right_confirmed.position;
    //     cv::Point2d GT_right_edge_location = ground_truth_right_edges_after_lowe[i];
    //     if (cv::norm(right_edge_location - GT_right_edge_location) <= MATCH_TOL) {
    //         bct_true_positive++;
    //     }
    // }
    std::cout << "BCT true positives: " << bct_true_positive << std::endl;

    auto bct_end = std::chrono::high_resolution_clock::now();
    double total_time_bct = std::chrono::duration<double, std::milli>(bct_end - bct_start).count();

    double per_image_bct_time = (matches_before_bct > 0) ? total_time_bct / matches_before_bct : 0.0;

    int matches_after_bct = static_cast<int>(confirmed_matches.size());
    std::cout << "Number of matches after BCT: " << matches_after_bct << std::endl;
    std::cout << "Number of stacked GT right edges: " << ground_truth_right_edges_after_lowe.size() << std::endl;

    // double per_image_bct_precision = (matches_before_bct > 0) ? static_cast<double>(matches_after_bct) / matches_before_bct: 0.0;
    double per_image_bct_precision = (matches_before_bct > 0) ? bct_true_positive / (double)(matches_after_bct) : 0.0;
    std::cout << "BCT precision = " << per_image_bct_precision << std::endl;

    int bct_denonimator = forward_match.recall_metrics.lowe_true_positive + forward_match.recall_metrics.lowe_false_negative;
    // int bct_true_positives = static_cast<int>(confirmed_matches.size());

    double bct_recall = (bct_denonimator > 0) ? bct_true_positive / (double)(bct_denonimator) : 0.0;
    std::cout << "BCT recall = " << bct_recall << std::endl;

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
    const cv::Mat& secondary_image, const std::vector<cv::Point2d>& selected_ground_truth_edges) {
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

#pragma omp parallel
{
    int thread_id = omp_get_thread_num();    

    cv::Point2d ground_truth_edge;

    //MAKE SURE TO UPDATE THIS ACCORDINGLY
    // int skip = (!selected_ground_truth_edges.empty()) ? 100 : 1;
    const int skip = 1;

    //> Start looping over left edges
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

        bool primary_passes_tangency = (abs(angle_diff_deg - 0) > EPIP_TENGENCY_ORIENT_THRESH && abs(angle_diff_deg - 180) > EPIP_TENGENCY_ORIENT_THRESH) ? (true) : (false);
        if (!primary_passes_tangency) {
            continue;
        }

        ///////////////////////////////EPIPOLAR DISTANCE THRESHOLD///////////////////////////////
#if MEASURE_TIMINGS
        auto start_epi = std::chrono::high_resolution_clock::now();
#endif

        std::pair<std::vector<cv::Point2d>, std::vector<double>> secondary_candidates_data = ExtractEpipolarEdges(epipolar_line, secondary_edge_coords, secondary_edge_orientations, 0.5);
        std::vector<cv::Point2d> secondary_candidate_edges = secondary_candidates_data.first;
        std::vector<double> secondary_candidate_orientations = secondary_candidates_data.second;

        std::pair<std::vector<cv::Point2d>, std::vector<double>> test_secondary_candidates_data = ExtractEpipolarEdges(epipolar_line, secondary_edge_coords, secondary_edge_orientations, 3);
        std::vector<cv::Point2d> test_secondary_candidate_edges = test_secondary_candidates_data.first;

        // epi_input_counts.push_back(secondary_edge_coords.size());
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
                    // break;
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
        
        // epi_output_counts.push_back(secondary_candidate_edges.size());
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

        // disp_input_counts.push_back(secondary_candidate_edges.size());
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
                    // break;
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

        // disp_output_counts.push_back(filtered_secondary_edge_coords.size());
        local_disp_output_counts[thread_id].push_back(filtered_secondary_edge_coords.size());

        std::vector<cv::Point2d> shifted_secondary_edge_coords;
        std::vector<double> shifted_secondary_edge_orientations;
        std::vector<double> epipolar_coefficients = {a, b, c};

        for (size_t j = 0; j < filtered_secondary_edge_coords.size(); j++) {
            bool secondary_passes_tangency = false;

            cv::Point2d shifted_edge = PerformEpipolarShift(filtered_secondary_edge_coords[j], filtered_secondary_edge_orientations[j], epipolar_coefficients, secondary_passes_tangency);
            if (secondary_passes_tangency){
                shifted_secondary_edge_coords.push_back(shifted_edge);
                shifted_secondary_edge_orientations.push_back(filtered_secondary_edge_orientations[j]);
            }
        }

        // shift_input_counts.push_back(filtered_secondary_edge_coords.size());
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
                if (cv::norm(shifted_candidate - ground_truth_edge) <= 3.0){
                    shift_precision_numerator++;
                    shift_match_found = true;
                    // break;
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

        // shift_output_counts.push_back(shifted_secondary_edge_coords.size());
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

        // clust_input_counts.push_back(shifted_secondary_edge_coords.size());
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
                if (cv::norm(cluster.center_coord - ground_truth_edge) <= 3.0) {
                    clust_precision_numerator++;
                    cluster_match_found = true;
                    // break;
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

        // clust_output_counts.push_back(cluster_centers.size());
        local_clust_output_counts[thread_id].push_back(cluster_centers.size());

        std::vector<cv::Point2d> cluster_coords;
        std::vector<double> cluster_orientations;

        for (const auto& cluster : cluster_centers) {
            cluster_coords.push_back(cluster.center_coord);
            cluster_orientations.push_back(cluster.center_orientation);
        }

        auto [secondary_orthogonal_one, secondary_orthogonal_two] = CalculateOrthogonalShifts(
            cluster_coords,
            cluster_orientations,
            ORTHOGONAL_SHIFT_MAG
        );

        std::vector<EdgeCluster> filtered_cluster_centers;
        std::vector<cv::Mat> secondary_patch_set_one;
        std::vector<cv::Mat> secondary_patch_set_two;

        ExtractClusterPatches(
            PATCH_SIZE,
            secondary_image,
            cluster_centers,
            nullptr,
            secondary_orthogonal_one,
            secondary_orthogonal_two,
            filtered_cluster_centers,
            nullptr,
            secondary_patch_set_one,
            secondary_patch_set_two
        );

        // patch_input_counts.push_back(cluster_centers.size());
        local_patch_input_counts[thread_id].push_back(cluster_centers.size());

#if MEASURE_TIMINGS
        time_patch_edges_evaluated++;
        auto end_patch = std::chrono::high_resolution_clock::now();
        time_patch += std::chrono::duration<double, std::milli>(end_patch - start_patch).count();
        //> MARK: NCC
       ///////////////////////////////NCC THRESHOLD/////////////////////////////////////////////////////
       auto start_ncc = std::chrono::high_resolution_clock::now();
#endif

    //    patch_output_counts.push_back(filtered_cluster_centers.size());
       local_patch_output_counts[thread_id].push_back(filtered_cluster_centers.size());

       int ncc_precision_numerator = 0;

       bool ncc_match_found = false;
       std::vector<EdgeMatch> passed_ncc_matches;

       if (!primary_patch_one.empty() && !primary_patch_two.empty() &&
           !secondary_patch_set_one.empty() && !secondary_patch_set_two.empty()) {

           for (size_t i = 0; i < filtered_cluster_centers.size(); ++i) {
               double ncc_one = ComputeNCC(primary_patch_one, secondary_patch_set_one[i]);
               double ncc_two = ComputeNCC(primary_patch_two, secondary_patch_set_two[i]);
               double ncc_three = ComputeNCC(primary_patch_one, secondary_patch_set_two[i]);
               double ncc_four = ComputeNCC(primary_patch_two, secondary_patch_set_one[i]);

               double score_one = std::min(ncc_one, ncc_two);
               double score_two = std::min(ncc_three, ncc_four);
               double final_score = std::max(score_one, score_two);

#if DEBUG_COLLECT_NCC_AND_ERR
               double err_to_gt = cv::norm(filtered_cluster_centers[i].center_coord - ground_truth_edge);
               std::pair<double, double> pair_ncc_one_err(err_to_gt, ncc_one);
               std::pair<double, double> pair_ncc_two_err(err_to_gt, ncc_two);
               ncc_one_vs_err.push_back(pair_ncc_one_err);
               ncc_two_vs_err.push_back(pair_ncc_two_err);
#endif
               if (ncc_one >= NCC_THRESH_STRONG_BOTH_SIDES && ncc_two >= NCC_THRESH_STRONG_BOTH_SIDES) {
                    EdgeMatch info;
                    info.coord = filtered_cluster_centers[i].center_coord;
                    info.orientation = filtered_cluster_centers[i].center_orientation;
                    info.final_score = final_score;
                    info.contributing_edges = filtered_cluster_centers[i].contributing_edges;
                    info.contributing_orientations = filtered_cluster_centers[i].contributing_orientations;
                    passed_ncc_matches.push_back(info);

                    if (!selected_ground_truth_edges.empty()) {
                        if (cv::norm(filtered_cluster_centers[i].center_coord - ground_truth_edge) <= 3.0) {
                            ncc_match_found = true;
                            ncc_precision_numerator++;
                        }
                    }
               }
               else if (ncc_one >= NCC_THRESH_STRONG_ONE_SIDE || ncc_two >= NCC_THRESH_STRONG_ONE_SIDE) {
                    EdgeMatch info;
                    info.coord = filtered_cluster_centers[i].center_coord;
                    info.orientation = filtered_cluster_centers[i].center_orientation;
                    info.final_score = final_score;
                    info.contributing_edges = filtered_cluster_centers[i].contributing_edges;
                    info.contributing_orientations = filtered_cluster_centers[i].contributing_orientations;
                    passed_ncc_matches.push_back(info);

                    if (!selected_ground_truth_edges.empty()) {
                        if (cv::norm(filtered_cluster_centers[i].center_coord - ground_truth_edge) <= 3.0) {
                            ncc_match_found = true;
                            ncc_precision_numerator++;
                        }
                    }
               }
               else if (ncc_one >= NCC_THRESH_WEAK_BOTH_SIDES && ncc_two >= NCC_THRESH_WEAK_BOTH_SIDES && filtered_cluster_centers.size() == 1) {
                    EdgeMatch info;
                    info.coord = filtered_cluster_centers[i].center_coord;
                    info.orientation = filtered_cluster_centers[i].center_orientation;
                    info.final_score = final_score;
                    info.contributing_edges = filtered_cluster_centers[i].contributing_edges;
                    info.contributing_orientations = filtered_cluster_centers[i].contributing_orientations;
                    passed_ncc_matches.push_back(info);

                    if (!selected_ground_truth_edges.empty()) {
                        if (cv::norm(filtered_cluster_centers[i].center_coord - ground_truth_edge) <= 3.0) {
                            ncc_match_found = true;
                            ncc_precision_numerator++;
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
    //    ncc_input_counts.push_back(filtered_cluster_centers.size());
    //    ncc_output_counts.push_back(passed_ncc_matches.size());

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
        // lowe_input_counts.push_back(passed_ncc_matches.size());
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
                    if (cv::norm(best_match.coord - ground_truth_edge) <= 3.0) {
                        lowe_precision_numerator++;
                        lowe_true_positive++;
                    }
                    else {
                        lowe_false_negative++;
                    }
                }
                SourceEdge source_edge {primary_edge, primary_orientation};
                // final_matches.emplace_back(source_edge, best_match);
                local_final_matches[thread_id].emplace_back(source_edge, best_match);
                // lowe_output_counts.push_back(1);
                local_lowe_output_counts[thread_id].push_back(1);
            }
            else {
                lowe_false_negative++;
                // lowe_output_counts.push_back(0);
                local_lowe_output_counts[thread_id].push_back(0);
            }
        }   
        else if (passed_ncc_matches.size() == 1){
            best_match = passed_ncc_matches[0];

            if (!selected_ground_truth_edges.empty()) {
                local_GT_right_edges_after_lowe[thread_id].push_back(ground_truth_edge);
                if (cv::norm(best_match.coord - ground_truth_edge) <= 3.0) {
                    lowe_precision_numerator++;
                    lowe_true_positive++;
                } else {
                    lowe_false_negative++;
                }
            }
            
            SourceEdge source_edge {primary_edge, primary_orientation};
            // final_matches.emplace_back(source_edge, best_match);
            local_final_matches[thread_id].emplace_back(source_edge, best_match);
            // lowe_output_counts.push_back(1);
            local_lowe_output_counts[thread_id].push_back(1);
        }
        else {
            lowe_false_negative++;
            // lowe_output_counts.push_back(0); 
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

    // std::cout << "Epipolar Distance Recall: " << std::fixed << std::setprecision(2) << epi_distance_recall * 100 << "%" << std::endl;
    // std::cout << "Max Disparity Threshold Recall: " << std::fixed << std::setprecision(2) << max_disparity_recall * 100 << "%" << std::endl;
    // std::cout << "Epipolar Shift Threshold Recall: " << std::fixed << std::setprecision(2) << epi_shift_recall * 100 << "%" << std::endl;
    // std::cout << "Epipolar Cluster Threshold Recall: " << std::fixed << std::setprecision(2) << epi_cluster_recall * 100 << "%" << std::endl;
    // std::cout << "NCC Threshold Recall: " << std::fixed << std::setprecision(2) << ncc_recall * 100 << "%" << std::endl; 
    // std::cout << "LRT Threshold Recall: " << std::fixed << std::setprecision(2) << lowe_recall * 100 << "%" << std::endl;

    double per_image_epi_precision   = (epi_edges_evaluated > 0)   ? (per_edge_epi_precision / epi_edges_evaluated)     : (0.0);
    double per_image_disp_precision  = (disp_edges_evaluated > 0)  ? (per_edge_disp_precision / disp_edges_evaluated)   : (0.0);
    double per_image_shift_precision = (shift_edges_evaluated > 0) ? (per_edge_shift_precision / shift_edges_evaluated) : (0.0);
    double per_image_clust_precision = (clust_edges_evaluated > 0) ? (per_edge_clust_precision / clust_edges_evaluated) : (0.0);
    double per_image_ncc_precision   = (ncc_edges_evaluated > 0)   ? (per_edge_ncc_precision / ncc_edges_evaluated)     : (0.0);
    double per_image_lowe_precision  = (lowe_edges_evaluated > 0)  ? (per_edge_lowe_precision / lowe_edges_evaluated)   : (0.0);

    // std::cout << "Epipolar Distance Precision: " 
    //     << std::fixed << std::setprecision(2) 
    //     << per_image_epi_precision * 100 << "%" << std::endl;
    // std::cout << "Maximum Disparity Precision: " 
    //     << std::fixed << std::setprecision(2) 
    //     << per_image_disp_precision * 100 << "%" << std::endl;
    // std::cout << "Epipolar Shift Precision: " 
    //     << std::fixed << std::setprecision(2) 
    //     << per_image_shift_precision * 100 << "%" << std::endl;
    // std::cout << "Epipolar Cluster Precision: " 
    //     << std::fixed << std::setprecision(2) 
    //     << per_image_clust_precision * 100 << "%" << std::endl;
    // std::cout << "NCC Precision: " 
    //     << std::fixed << std::setprecision(2) 
    //     << per_image_ncc_precision * 100 << "%" << std::endl;
    // std::cout << "LRT Precision: " 
    //     << std::fixed << std::setprecision(2) 
    //     << per_image_lowe_precision * 100 << "%" << std::endl;

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

std::vector<cv::Point3d> Dataset::Calculate3DPoints(
    const std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>>& confirmed_matches
) {
    std::vector<cv::Point3d> points_3d;

    if (confirmed_matches.empty()) {
        std::cerr << "WARNING: No confirmed matches to triangulate.\n";
        return points_3d;
    }

    cv::Mat K_left = (cv::Mat_<double>(3, 3) <<
        left_intr[0], 0,            left_intr[2],
        0,           left_intr[1], left_intr[3],
        0,           0,            1);

    cv::Mat K_right = (cv::Mat_<double>(3, 3) <<
        right_intr[0], 0,             right_intr[2],
        0,            right_intr[1], right_intr[3],
        0,            0,             1);

    cv::Mat R_left = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat T_left = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat R_right(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R_right.at<double>(i, j) = rot_mat_21[i][j];

    cv::Mat T_right = (cv::Mat_<double>(3, 1) <<
        trans_vec_21[0],
        trans_vec_21[1],
        trans_vec_21[2]);

    cv::Mat extrinsic_left, extrinsic_right;
    cv::hconcat(R_left, T_left, extrinsic_left);
    cv::hconcat(R_right, T_right, extrinsic_right);

    cv::Mat P_left = K_left * extrinsic_left;
    cv::Mat P_right = K_right * extrinsic_right;

    std::vector<cv::Point2f> points_left, points_right;
    for (const auto& [left, right] : confirmed_matches) {
        points_left.emplace_back(static_cast<cv::Point2f>(left.position));
        points_right.emplace_back(static_cast<cv::Point2f>(right.position));
    }

    cv::Mat points_4d_homogeneous;
    cv::triangulatePoints(P_left, P_right, points_left, points_right, points_4d_homogeneous);

    int skipped = 0;
    for (int i = 0; i < points_4d_homogeneous.cols; ++i) {
        float w = points_4d_homogeneous.at<float>(3, i);
        if (std::abs(w) > 1e-5) {
            points_3d.emplace_back(
                points_4d_homogeneous.at<float>(0, i) / w,
                points_4d_homogeneous.at<float>(1, i) / w,
                points_4d_homogeneous.at<float>(2, i) / w
            );
        } else {
            ++skipped;
        }
    }

    int total = static_cast<int>(points_4d_homogeneous.cols);
    if (skipped > 0.1 * total) {
        std::cerr << "WARNING: " << skipped << " out of " << total
                << " triangulated points had near-zero depth (w ≈ 0) and were skipped.\n";
    }

    return points_3d;
}

std::vector<Eigen::Vector3d> Dataset::Calculate3DOrientations(
    const std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>>& confirmed_matches
) {
    std::vector<Eigen::Vector3d> tangent_vectors;

    if (confirmed_matches.empty()) {
        std::cerr << "WARNING: No confirmed matches to compute 3D orientations.\n";
        return tangent_vectors;
    }

    Eigen::Matrix3d K;
    K << left_intr[0], 0, left_intr[2],
         0, left_intr[1], left_intr[3],
         0, 0, 1;

    Eigen::Matrix3d R21;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R21(i, j) = rot_mat_21[i][j];

    for (const auto& [left_edge, right_edge] : confirmed_matches) {
        Eigen::Vector3d gamma1 = K.inverse() * Eigen::Vector3d(left_edge.position.x, left_edge.position.y, 1.0);
        Eigen::Vector3d gamma2 = K.inverse() * Eigen::Vector3d(right_edge.position.x, right_edge.position.y, 1.0);

        double theta1 = left_edge.orientation;
        double theta2 = right_edge.orientation;

        Eigen::Vector3d t1(std::cos(theta1), std::sin(theta1), 0.0);
        Eigen::Vector3d t2(std::cos(theta2), std::sin(theta2), 0.0);

        Eigen::Vector3d n1 = gamma1.cross(t1);
        Eigen::Vector3d n2 = R21.transpose() * (t2.cross(gamma2));

        Eigen::Vector3d T = n1.cross(n2);
        T.normalize(); 

        tangent_vectors.push_back(T);
    }

    return tangent_vectors;
}

void Dataset::BuildImagePyramids(
    const cv::Mat& curr_left_image,
    const cv::Mat& curr_right_image,
    const cv::Mat& next_left_image,
    const cv::Mat& next_right_image,
    int num_levels,
    std::vector<cv::Mat>& curr_left_pyramid,
    std::vector<cv::Mat>& curr_right_pyramid,
    std::vector<cv::Mat>& next_left_pyramid,
    std::vector<cv::Mat>& next_right_pyramid
) {
    curr_left_pyramid.clear();
    curr_right_pyramid.clear();
    next_left_pyramid.clear();
    next_right_pyramid.clear();

    curr_left_pyramid.reserve(num_levels);
    curr_right_pyramid.reserve(num_levels);
    next_left_pyramid.reserve(num_levels);
    next_right_pyramid.reserve(num_levels);

    cv::buildPyramid(curr_left_image, curr_left_pyramid, num_levels - 1);
    cv::buildPyramid(curr_right_image, curr_right_pyramid, num_levels - 1);
    cv::buildPyramid(next_left_image, next_left_pyramid, num_levels - 1);
    cv::buildPyramid(next_right_image, next_right_pyramid, num_levels - 1);
}

std::vector<cv::Point3d> Dataset::LinearTriangulatePoints(
    const std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>>& confirmed_matches
) {
    std::vector<cv::Point3d> triangulated_points;

    if (confirmed_matches.empty()) {
        std::cerr << "WARNING: No confirmed matches to triangulate using linear method.\n";
        return triangulated_points;
    }

    Eigen::Matrix3d K;
    K << left_intr[0], 0, left_intr[2],
         0, left_intr[1], left_intr[3],
         0, 0, 1;

    Eigen::Matrix3d R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R(i, j) = rot_mat_21[i][j];

    Eigen::Vector3d T(trans_vec_21[0], trans_vec_21[1], trans_vec_21[2]);

    for (const auto& [left_edge, right_edge] : confirmed_matches) {
        std::vector<Eigen::Vector2d> pts;
        pts.emplace_back(left_edge.position.x, left_edge.position.y);
        pts.emplace_back(right_edge.position.x, right_edge.position.y);

        std::vector<Eigen::Vector2d> pts_meters;
        for (const auto& pt : pts) {
            Eigen::Vector3d homo_pt(pt.x(), pt.y(), 1.0);
            Eigen::Vector3d pt_cam = K.inverse() * homo_pt;
            pts_meters.emplace_back(pt_cam.x(), pt_cam.y());
        }

        Eigen::MatrixXd A(4, 4); 

        A.row(0) << 0.0, -1.0, pts_meters[0].y(), 0.0;
        A.row(1) << 1.0,  0.0, -pts_meters[0].x(), 0.0;

        Eigen::Matrix3d Rp = R;
        Eigen::Vector3d Tp = T;
        Eigen::Vector2d mp = pts_meters[1];

        A.row(2) << mp.y() * Rp(2, 0) - Rp(1, 0),
                    mp.y() * Rp(2, 1) - Rp(1, 1),
                    mp.y() * Rp(2, 2) - Rp(1, 2),
                    mp.y() * Tp.z()   - Tp.y();

        A.row(3) << Rp(0, 0) - mp.x() * Rp(2, 0),
                    Rp(0, 1) - mp.x() * Rp(2, 1),
                    Rp(0, 2) - mp.x() * Rp(2, 2),
                    Tp.x()   - mp.x() * Tp.z();

        Eigen::Matrix4d ATA = A.transpose() * A;
        Eigen::Vector4d gamma = ATA.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3);

        if (std::abs(gamma(3)) > 1e-5) {
            gamma /= gamma(3);
            triangulated_points.emplace_back(gamma(0), gamma(1), gamma(2));
        }
    }

    return triangulated_points;
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

std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> Dataset::CalculateOrthogonalShifts(const std::vector<cv::Point2d>& edge_points, const std::vector<double>& orientations, double shift_magnitude){
   std::vector<cv::Point2d> shifted_points_one;
   std::vector<cv::Point2d> shifted_points_two;

   if (edge_points.size() != orientations.size()) {
       std::cerr << "ERROR: Mismatch between number of edge points and orientations." << std::endl;
       return {shifted_points_one, shifted_points_two};
   }

   for (size_t i = 0; i < edge_points.size(); ++i) {
       const auto& edge_point = edge_points[i];
       double theta = orientations[i];

       double orthogonal_x1 = std::sin(theta);
       double orthogonal_y1 = -std::cos(theta);
       double orthogonal_x2 = -std::sin(theta);
       double orthogonal_y2 = std::cos(theta);

       double shifted_x1 = edge_point.x + shift_magnitude * orthogonal_x1;
       double shifted_y1 = edge_point.y + shift_magnitude * orthogonal_y1;
       double shifted_x2 = edge_point.x + shift_magnitude * orthogonal_x2;
       double shifted_y2 = edge_point.y + shift_magnitude * orthogonal_y2;

       shifted_points_one.emplace_back(shifted_x1, shifted_y1);
       shifted_points_two.emplace_back(shifted_x2, shifted_y2);
   }

   return {shifted_points_one, shifted_points_two};
}

void Dataset::ExtractClusterPatches(
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
)
{
    int half_patch = std::ceil(patch_size / 2);

    for (int i = 0; i < shifted_one.size(); i++) {
        double x1 = shifted_one[i].x;
        double y1 = shifted_one[i].y;
        double x2 = shifted_two[i].x;
        double y2 = shifted_two[i].y;

        bool in_bounds_one = (x1 - half_patch >= 0 && x1 + half_patch < image.cols &&
                            y1 - half_patch >= 0 && y1 + half_patch < image.rows);
        bool in_bounds_two = (x2 - half_patch >= 0 && x2 + half_patch < image.cols &&
                            y2 - half_patch >= 0 && y2 + half_patch < image.rows);

        if (in_bounds_one && in_bounds_two) {
            cv::Point2f center1(static_cast<float>(x1), static_cast<float>(y1));
            cv::Point2f center2(static_cast<float>(x2), static_cast<float>(y2));
            cv::Size size(patch_size, patch_size);

            cv::Mat patch1, patch2;
            cv::getRectSubPix(image, size, center1, patch1);
            cv::getRectSubPix(image, size, center2, patch2);

            if (patch1.type() != CV_32F) {
                patch1.convertTo(patch1, CV_32F);
            }
            if (patch2.type() != CV_32F) {
                patch2.convertTo(patch2, CV_32F);
            }

            patch_set_one_out.push_back(patch1);
            patch_set_two_out.push_back(patch2);
            cluster_centers_out.push_back(cluster_centers[i]);

            if (right_edges && filtered_right_edges_out) {
                filtered_right_edges_out->push_back((*right_edges)[i]);
            }
        }
    }
}

void Dataset::ExtractPatches(
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
    const std::vector<cv::Point2d>* ground_truth_edges, 
    std::vector<cv::Point2d>* filtered_gt_edges_out
)
{
    int half_patch = std::ceil(patch_size / 2);

    for (int i = 0; i < shifted_one.size(); i++) {
        double x1 = shifted_one[i].x;
        double y1 = shifted_one[i].y;
        double x2 = shifted_two[i].x;
        double y2 = shifted_two[i].y;

        bool in_bounds_one = (x1 - half_patch >= 0 && x1 + half_patch < image.cols &&
                            y1 - half_patch >= 0 && y1 + half_patch < image.rows);
        bool in_bounds_two = (x2 - half_patch >= 0 && x2 + half_patch < image.cols &&
                            y2 - half_patch >= 0 && y2 + half_patch < image.rows);

        if (in_bounds_one && in_bounds_two) {
            cv::Point2f center1(static_cast<float>(x1), static_cast<float>(y1));
            cv::Point2f center2(static_cast<float>(x2), static_cast<float>(y2));
            cv::Size size(patch_size, patch_size);

            cv::Mat patch1, patch2;
            cv::getRectSubPix(image, size, center1, patch1);
            cv::getRectSubPix(image, size, center2, patch2);

            if (patch1.type() != CV_32F) {
                patch1.convertTo(patch1, CV_32F);
            }
            if (patch2.type() != CV_32F) {
                patch2.convertTo(patch2, CV_32F);
            }

            filtered_edges_out.push_back(edges[i]);
            filtered_orientations_out.push_back(orientations[i]);
            patch_set_one_out.push_back(patch1);
            patch_set_two_out.push_back(patch2);

            if (ground_truth_edges && filtered_gt_edges_out) {
                filtered_gt_edges_out->push_back((*ground_truth_edges)[i]);
            }
        }
    }
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

std::pair<std::vector<cv::Point2d>, std::vector<double>>Dataset::ExtractEpipolarEdges(const Eigen::Vector3d& epipolar_line, const std::vector<cv::Point2d>& edge_locations, const std::vector<double>& edge_orientations, double distance_threshold) {
   std::vector<cv::Point2d> extracted_edges;
   std::vector<double> extracted_orientations;

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
       }
   }

   return {extracted_edges, extracted_orientations};
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
    int half_patch = patch_size / 2;

    if (edges.size() != orientations.size() || edges.size() != ground_truth_right_edges.size()) {
        throw std::runtime_error("Edge locations, orientations, and ground truth edges size mismatch.");
    }

    for (size_t i = 0; i < edges.size(); ++i) {
        const auto& edge = edges[i];
        if (edge.x >= half_patch && edge.x < img_width - half_patch &&
            edge.y >= half_patch && edge.y < img_height - half_patch) {
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

cv::Point2d Dataset::PerformEpipolarShift( 
    cv::Point2d original_edge_location, double edge_orientation,
    std::vector<double> epipolar_line_coeffs, bool& b_pass_epipolar_tengency_check )
{
    cv::Point2d corrected_edge;
    assert(epipolar_line_coeffs.size() == 3);
    double EL_coeff_A = epipolar_line_coeffs[0];
    double EL_coeff_B = epipolar_line_coeffs[1];
    double EL_coeff_C = epipolar_line_coeffs[2];
    double a1_line  = -epipolar_line_coeffs[0] / epipolar_line_coeffs[1];
    double b1_line  = -1;
    double c1_line  = -epipolar_line_coeffs[2] / epipolar_line_coeffs[1];
    
    //> Parameters of the line passing through the original edge along its direction (tangent) vector
    double a_edgeH2 = tan(edge_orientation);
    double b_edgeH2 = -1;
    double c_edgeH2 = -(a_edgeH2*original_edge_location.x - original_edge_location.y); //−(a⋅x2−y2)

    //> Find the intersected point of the two lines
    corrected_edge.x = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    corrected_edge.y = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    
    //> Find (i) the displacement between the original edge and the corrected edge, and
    //       (ii) the intersection angle between the epipolar line and the line passing through the original edge along its direction vector
    double epipolar_shift_displacement = cv::norm(corrected_edge - original_edge_location);
    double m_epipolar = -a1_line / b1_line; //> Slope of epipolar line
    double angle_diff_rad = abs(edge_orientation - atan(m_epipolar));
    double angle_diff_deg = angle_diff_rad * (180.0 / M_PI);
    if (angle_diff_deg > 180){
        angle_diff_deg -= 180;
    }

    //> check if the corrected edge passes the epoplar tengency test (intersection angle < 4 degrees and displacement < 6 pixels)
    b_pass_epipolar_tengency_check = (epipolar_shift_displacement < EPIP_TENGENCY_PROXIM_THRESH && abs(angle_diff_deg - 0) > EPIP_TENGENCY_ORIENT_THRESH && abs(angle_diff_deg - 180) > EPIP_TENGENCY_ORIENT_THRESH) ? (true) : (false);
    
    return corrected_edge;
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

double Bilinear_Interpolation(const cv::Mat &meshGrid, cv::Point2d P) {
    cv::Point2d Q12(floor(P.x), floor(P.y));
    cv::Point2d Q22(ceil(P.x), floor(P.y));
    cv::Point2d Q11(floor(P.x), ceil(P.y));
    cv::Point2d Q21(ceil(P.x), ceil(P.y));

    if (Q11.x < 0 || Q11.y < 0 || Q21.x >= meshGrid.cols || Q21.y >= meshGrid.rows ||
        Q12.x < 0 || Q12.y < 0 || Q22.x >= meshGrid.cols || Q22.y >= meshGrid.rows) {
        return std::numeric_limits<double>::quiet_NaN(); 
    }

    double fQ11 = meshGrid.at<float>(Q11.y, Q11.x);
    double fQ21 = meshGrid.at<float>(Q21.y, Q21.x);
    double fQ12 = meshGrid.at<float>(Q12.y, Q12.x);
    double fQ22 = meshGrid.at<float>(Q22.y, Q22.x);

    double f_x_y1 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * fQ11 + ((P.x - Q11.x) / (Q21.x - Q11.x)) * fQ21;
    double f_x_y2 = ((Q21.x - P.x) / (Q21.x - Q11.x)) * fQ12 + ((P.x - Q11.x) / (Q21.x - Q11.x)) * fQ22;
    return ((Q12.y - P.y) / (Q12.y - Q11.y)) * f_x_y1 + ((P.y - Q11.y) / (Q12.y - Q11.y)) * f_x_y2;
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

// std::vector<cv::Mat> Dataset::LoadETH3DRightReferenceMaps(const std::string &stereo_pairs_path, int num_maps) {
//     std::vector<cv::Mat> disparity_maps;
//     std::vector<std::string> stereo_folders;

//     for (const auto &entry : std::filesystem::directory_iterator(stereo_pairs_path)) {
//         if (entry.is_directory()) {
//             stereo_folders.push_back(entry.path().string());
//         }
//     }

//     std::sort(stereo_folders.begin(), stereo_folders.end());

//     for (int i = 0; i < std::min(num_maps, static_cast<int>(stereo_folders.size())); i++) {
//         std::string folder_path = stereo_folders[i];
//         std::string disparity_csv_path = folder_path + "/disparity_map_right.csv";
//         std::string disparity_bin_path = folder_path + "/disparity_map_right.bin";

//         cv::Mat disparity_map;

//         if (std::filesystem::exists(disparity_bin_path)) {
//             disparity_map = ReadDisparityFromBinary(disparity_bin_path);
//         } else {
//             disparity_map = LoadDisparityFromCSV(disparity_csv_path);
//             if (!disparity_map.empty()) {
//                 WriteDisparityToBinary(disparity_bin_path, disparity_map);
//             }
//         }

//         if (!disparity_map.empty()) {
//             disparity_maps.push_back(disparity_map);
//         }
//     }

//     return disparity_maps;
// }

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

#endif