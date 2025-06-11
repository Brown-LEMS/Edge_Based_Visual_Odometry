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
#include "Matches.h"

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

Dataset::Dataset(YAML::Node config_map, bool use_GCC_filter) : config_file(config_map), compute_grad_depth(use_GCC_filter)
{

#if USE_DEFINED_NUM_OF_CORES
    omp_threads = USE_NUM_CORES_FOR_OMP;
#else
    omp_threads = omp_get_num_procs();
#endif

    dataset_path = config_file["dataset_dir"].as<std::string>();
    output_path = config_file["output_dir"].as<std::string>();
    sequence_name = config_file["sequence_name"].as<std::string>();
    dataset_type = config_file["dataset_type"].as<std::string>();

    if (dataset_type == "EuRoC")
    {
        try
        {
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

            if (stereo["R21"] && stereo["T21"] && stereo["F21"])
            {
                for (const auto &row : stereo["R21"])
                {
                    rot_mat_21.push_back(row.as<std::vector<double>>());
                }

                trans_vec_21 = stereo["T21"].as<std::vector<double>>();

                for (const auto &row : stereo["F21"])
                {
                    fund_mat_21.push_back(row.as<std::vector<double>>());
                }
            }
            else
            {
                std::cerr << "ERROR: Missing left-to-right stereo parameters (R21, T21, F21) in YAML file!" << std::endl;
            }

            if (stereo["R12"] && stereo["T12"] && stereo["F12"])
            {
                for (const auto &row : stereo["R12"])
                {
                    rot_mat_12.push_back(row.as<std::vector<double>>());
                }
                trans_vec_12 = stereo["T12"].as<std::vector<double>>();

                for (const auto &row : stereo["F12"])
                {
                    fund_mat_12.push_back(row.as<std::vector<double>>());
                }
            }
            else
            {
                std::cerr << "ERROR: Missing right-to-left stereo parameters (R12, T12, F12) in YAML file!" << std::endl;
            }

            if (frame_to_body["rotation"] && frame_to_body["translation"])
            {
                rot_frame2body_left = Eigen::Map<Eigen::Matrix3d>(frame_to_body["rotation"].as<std::vector<double>>().data()).transpose();
                transl_frame2body_left = Eigen::Map<Eigen::Vector3d>(frame_to_body["translation"].as<std::vector<double>>().data());
            }
            else
            {
                LOG_ERROR("Missing relative rotation and translation from the left camera to the body coordinate (should be given by cam0/sensor.yaml)");
            }
        }
        catch (const YAML::Exception &e)
        {
            std::cerr << "ERROR: Could not parse YAML file! " << e.what() << std::endl;
        }
    }
    else if (dataset_type == "ETH3D")
        try
        {
            YAML::Node left_cam = config_file["left_camera"];
            YAML::Node right_cam = config_file["right_camera"];
            YAML::Node stereo = config_file["stereo"];

            left_res = left_cam["resolution"].as<std::vector<int>>();
            left_intr = left_cam["intrinsics"].as<std::vector<double>>();
            left_dist_coeffs = left_cam["distortion_coefficients"].as<std::vector<double>>();

            right_res = right_cam["resolution"].as<std::vector<int>>();
            right_intr = right_cam["intrinsics"].as<std::vector<double>>();
            right_dist_coeffs = right_cam["distortion_coefficients"].as<std::vector<double>>();

            if (stereo["R21"] && stereo["T21"] && stereo["F21"])
            {
                for (const auto &row : stereo["R21"])
                {
                    rot_mat_21.push_back(row.as<std::vector<double>>());
                }

                trans_vec_21 = stereo["T21"].as<std::vector<double>>();

                for (const auto &row : stereo["F21"])
                {
                    fund_mat_21.push_back(row.as<std::vector<double>>());
                }
            }
            else
            {
                std::cerr << "ERROR: Missing left-to-right stereo parameters (R21, T21, F21) in YAML file!" << std::endl;
            }

            if (stereo["R12"] && stereo["T12"] && stereo["F12"])
            {
                for (const auto &row : stereo["R12"])
                {
                    rot_mat_12.push_back(row.as<std::vector<double>>());
                }
                trans_vec_12 = stereo["T12"].as<std::vector<double>>();

                for (const auto &row : stereo["F12"])
                {
                    fund_mat_12.push_back(row.as<std::vector<double>>());
                }
            }
            else
            {
                std::cerr << "ERROR: Missing right-to-left stereo parameters (R12, T12, F12) in YAML file!" << std::endl;
            }
            if (stereo["focal_length"] && stereo["baseline"])
            {
                focal_length = stereo["focal_length"].as<double>();
                baseline = stereo["baseline"].as<double>();
            }
            else
            {
                std::cerr << "ERROR: Missing stereo parameters (focal_length, baseline) in YAML file!" << std::endl;
            }
        }
        catch (const YAML::Exception &e)
        {
            std::cerr << "ERROR: Could not parse YAML file! " << e.what() << std::endl;
        }

    Total_Num_Of_Imgs = 0;
}

void load_dataset(std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs, std::string dataset_type, int num_pairs)
{
    if (dataset_type == "EuRoC")
    {
        std::string left_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data/";
        std::string right_path = dataset_path + "/" + sequence_name + "/mav0/cam1/data/";
        std::string image_csv_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data.csv";
        std::string ground_truth_path = dataset_path + "/" + sequence_name + "/mav0/state_groundtruth_estimate0/data.csv";

        image_pairs = LoadEuRoCImages(image_csv_path, left_path, right_path, num_pairs);

        Load_GT_Poses(ground_truth_path);
        Align_Images_and_GT_Poses();
    }
    else if (dataset_type == "ETH3D")
    {
        std::string stereo_pairs_path = dataset_path + "/" + sequence_name + "/stereo_pairs";
        image_pairs = LoadETH3DImages(stereo_pairs_path, num_pairs);
        left_ref_disparity_maps = LoadETH3DLeftReferenceMaps(stereo_pairs_path, num_pairs);
        // right_ref_disparity_maps = LoadETH3DRightReferenceMaps(stereo_pairs_path, num_pairs);
    }
}

void Dataset::VisualizeGTRightEdge(const cv::Mat &left_image, const cv::Mat &right_image, const std::vector<std::pair<cv::Point2d, cv::Point2d>> &left_right_edges)
{
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
        cv::Scalar(255, 20, 147)};

    std::vector<std::pair<cv::Point2d, cv::Point2d>> sampled_pairs;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, left_right_edges.size() - 1);

    int num_samples = std::min(10, static_cast<int>(left_right_edges.size()));
    for (int i = 0; i < num_samples; ++i)
    {
        sampled_pairs.push_back(left_right_edges[distr(gen)]);
    }

    for (size_t i = 0; i < sampled_pairs.size(); ++i)
    {
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

void Dataset::CalculateGTRightEdge(const std::vector<cv::Point2d> &left_third_order_edges_locations, const std::vector<double> &left_third_order_edges_orientation, const cv::Mat &disparity_map, const cv::Mat &left_image, const cv::Mat &right_image)
{
    forward_gt_data.clear();

    static size_t total_rows_written = 0;
    static int file_index = 1;
    static std::ofstream csv_file;
    static const size_t max_rows_per_file = 1'000'000;

    if (!csv_file.is_open())
    {
        std::string filename = "valid_disparities_part_" + std::to_string(file_index) + ".csv";
        csv_file.open(filename, std::ios::out);
    }

    for (size_t i = 0; i < left_third_order_edges_locations.size(); i++)
    {
        const cv::Point2d &left_edge = left_third_order_edges_locations[i];
        double orientation = left_third_order_edges_orientation[i];

        double disparity = Bilinear_Interpolation(disparity_map, left_edge);

        if (std::isnan(disparity) || std::isinf(disparity) || disparity < 0)
        {
            continue;
        }

        cv::Point2d right_edge(left_edge.x - disparity, left_edge.y);
        forward_gt_data.emplace_back(left_edge, right_edge, orientation);

        if (total_rows_written >= max_rows_per_file)
        {
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

void Dataset::CalculateGTLeftEdge(const std::vector<cv::Point2d> &right_third_order_edges_locations, const std::vector<double> &right_third_order_edges_orientation, const cv::Mat &disparity_map_right_reference, const cv::Mat &left_image, const cv::Mat &right_image)
{
    reverse_gt_data.clear();

    static size_t total_rows_written = 0;
    static int file_index = 1;
    static std::ofstream csv_file;
    static const size_t max_rows_per_file = 1'000'000;

    if (!csv_file.is_open())
    {
        std::string filename = "valid_reverse_disparities_part_" + std::to_string(file_index) + ".csv";
        csv_file.open(filename, std::ios::out);
    }

    for (size_t i = 0; i < right_third_order_edges_locations.size(); i++)
    {
        const cv::Point2d &right_edge = right_third_order_edges_locations[i];
        double orientation = right_third_order_edges_orientation[i];

        double disparity = Bilinear_Interpolation(disparity_map_right_reference, right_edge);

        if (std::isnan(disparity) || std::isinf(disparity) || disparity < 0)
        {
            continue;
        }

        cv::Point2d left_edge(right_edge.x + disparity, right_edge.y);

        reverse_gt_data.emplace_back(right_edge, left_edge, orientation);

        if (total_rows_written >= max_rows_per_file)
        {
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

std::vector<std::pair<cv::Mat, cv::Mat>> Dataset::LoadEuRoCImages(const std::string &csv_path, const std::string &left_path, const std::string &right_path,
                                                                  int num_pairs)
{
    std::ifstream csv_file(csv_path);
    if (!csv_file.is_open())
    {
        std::cerr << "ERROR: Could not open the CSV file located at " << csv_path << "!" << std::endl;
        return {};
    }

    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;
    std::string line;
    bool first_line = true;

    while (std::getline(csv_file, line) && image_pairs.size() < num_pairs)
    {
        if (first_line)
        {
            first_line = false;
            continue;
        }

        std::istringstream line_stream(line);
        std::string timestamp;
        std::getline(line_stream, timestamp, ',');

        Img_time_stamps.push_back(std::stod(timestamp));

        std::string left_img_path = left_path + timestamp + ".png";
        std::string right_img_path = right_path + timestamp + ".png";

        cv::Mat curr_left_img = cv::imread(left_img_path, cv::IMREAD_GRAYSCALE);
        cv::Mat curr_right_img = cv::imread(right_img_path, cv::IMREAD_GRAYSCALE);

        if (curr_left_img.empty() || curr_right_img.empty())
        {
            std::cerr << "ERROR: Could not load the images: " << left_img_path << " or " << right_img_path << "!" << std::endl;
            continue;
        }

        image_pairs.emplace_back(curr_left_img, curr_right_img);
    }

    csv_file.close();
    return image_pairs;
}

std::vector<std::pair<cv::Mat, cv::Mat>> Dataset::LoadETH3DImages(const std::string &stereo_pairs_path, int num_pairs)
{
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;

    std::vector<std::string> stereo_folders;
    for (const auto &entry : std::filesystem::directory_iterator(stereo_pairs_path))
    {
        if (entry.is_directory())
        {
            stereo_folders.push_back(entry.path().string());
        }
    }

    std::sort(stereo_folders.begin(), stereo_folders.end());

    for (int i = 0; i < std::min(num_pairs, static_cast<int>(stereo_folders.size())); ++i)
    {
        std::string folder_path = stereo_folders[i];

        std::string left_image_path = folder_path + "/im0.png";
        std::string right_image_path = folder_path + "/im1.png";

        cv::Mat left_image = cv::imread(left_image_path, cv::IMREAD_GRAYSCALE);
        cv::Mat right_image = cv::imread(right_image_path, cv::IMREAD_GRAYSCALE);

        if (!left_image.empty() && !right_image.empty())
        {
            image_pairs.emplace_back(left_image, right_image);
        }
        else
        {
            std::cerr << "ERROR: Could not load images from folder: " << folder_path << std::endl;
        }
    }

    return image_pairs;
}

std::vector<cv::Mat> Dataset::LoadETH3DLeftReferenceMaps(const std::string &stereo_pairs_path, int num_maps)
{
    std::vector<cv::Mat> disparity_maps;
    std::vector<std::string> stereo_folders;

    for (const auto &entry : std::filesystem::directory_iterator(stereo_pairs_path))
    {
        if (entry.is_directory())
        {
            stereo_folders.push_back(entry.path().string());
        }
    }

    std::sort(stereo_folders.begin(), stereo_folders.end());

    for (int i = 0; i < std::min(num_maps, static_cast<int>(stereo_folders.size())); i++)
    {
        std::string folder_path = stereo_folders[i];
        std::string disparity_csv_path = folder_path + "/disparity_map.csv";
        std::string disparity_bin_path = folder_path + "/disparity_map.bin";

        cv::Mat disparity_map;

        if (std::filesystem::exists(disparity_bin_path))
        {
            // std::cout << "Loading disparity data from: " << disparity_bin_path << std::endl;
            disparity_map = ReadDisparityFromBinary(disparity_bin_path);
        }
        else
        {
            // std::cout << "Parsing and storing disparity data from: " << disparity_csv_path << std::endl;
            disparity_map = LoadDisparityFromCSV(disparity_csv_path);
            if (!disparity_map.empty())
            {
                WriteDisparityToBinary(disparity_bin_path, disparity_map);
                // std::cout << "Saved disparity data to: " << disparity_bin_path << std::endl;
            }
        }

        if (!disparity_map.empty())
        {
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

void Dataset::WriteDisparityToBinary(const std::string &filepath, const cv::Mat &disparity_map)
{
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open())
    {
        std::cerr << "ERROR: Could not write disparity to: " << filepath << std::endl;
        return;
    }

    int rows = disparity_map.rows;
    int cols = disparity_map.cols;
    ofs.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    ofs.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    ofs.write(reinterpret_cast<const char *>(disparity_map.ptr<float>(0)), sizeof(float) * rows * cols);
}

cv::Mat Dataset::ReadDisparityFromBinary(const std::string &filepath)
{
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open())
    {
        std::cerr << "ERROR: Could not read disparity from: " << filepath << std::endl;
        return {};
    }

    int rows, cols;
    ifs.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    ifs.read(reinterpret_cast<char *>(&cols), sizeof(cols));

    cv::Mat disparity_map(rows, cols, CV_32F);
    ifs.read(reinterpret_cast<char *>(disparity_map.ptr<float>(0)), sizeof(float) * rows * cols);

    return disparity_map;
}

cv::Mat Dataset::LoadDisparityFromCSV(const std::string &path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "ERROR: Could not open disparity CSV: " << path << std::endl;
        return {};
    }

    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        while (std::getline(ss, value, ','))
        {
            try
            {
                float d = std::stof(value);

                if (value == "nan" || value == "NaN")
                {
                    d = std::numeric_limits<float>::quiet_NaN();
                }
                else if (value == "inf" || value == "Inf")
                {
                    d = std::numeric_limits<float>::infinity();
                }
                else if (value == "-inf" || value == "-Inf")
                {
                    d = -std::numeric_limits<float>::infinity();
                }

                row.push_back(d);
            }
            catch (const std::exception &e)
            {
                std::cerr << "WARNING: Invalid value in file: " << path << " -> " << value << std::endl;
                row.push_back(std::numeric_limits<float>::quiet_NaN());
            }
        }

        if (!row.empty())
            data.push_back(row);
    }

    int rows = data.size();
    int cols = data[0].size();
    cv::Mat disparity(rows, cols, CV_32F);

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            disparity.at<float>(r, c) = data[r][c];
        }
    }

    return disparity;
}

void Dataset::WriteEdgesToBinary(const std::string &filepath,
                                 const std::vector<cv::Point2d> &locations,
                                 const std::vector<double> &orientations)
{
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open())
    {
        std::cerr << "ERROR: Could not open binary file for writing: " << filepath << std::endl;
        return;
    }

    size_t size = locations.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
    ofs.write(reinterpret_cast<const char *>(locations.data()), sizeof(cv::Point2d) * size);
    ofs.write(reinterpret_cast<const char *>(orientations.data()), sizeof(double) * size);
}

void Dataset::ReadEdgesFromBinary(const std::string &filepath,
                                  std::vector<cv::Point2d> &locations,
                                  std::vector<double> &orientations)
{
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open())
    {
        std::cerr << "ERROR: Could not open binary file for reading: " << filepath << std::endl;
        return;
    }

    size_t size = 0;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size));

    locations.resize(size);
    orientations.resize(size);

    ifs.read(reinterpret_cast<char *>(locations.data()), sizeof(cv::Point2d) * size);
    ifs.read(reinterpret_cast<char *>(orientations.data()), sizeof(double) * size);
}

void Dataset::ProcessEdges(const cv::Mat &image,
                           const std::string &filepath,
                           std::shared_ptr<ThirdOrderEdgeDetectionCPU> &toed,
                           std::vector<cv::Point2d> &locations,
                           std::vector<double> &orientations)
{
    std::string path = filepath + ".bin";

    if (std::filesystem::exists(path))
    {
        // std::cout << "Loading edge data from: " << path << std::endl;
        ReadEdgesFromBinary(path, locations, orientations);
    }
    else
    {
        // std::cout << "Running third-order edge detector..." << std::endl;
        toed->get_Third_Order_Edges(image);
        locations = toed->toed_locations;
        orientations = toed->toed_orientations;

        WriteEdgesToBinary(path, locations, orientations);
        // std::cout << "Saved edge data to: " << path << std::endl;
    }
}

void Dataset::Load_GT_Poses(std::string GT_Poses_File_Path)
{
    // define the gtpose file
    std::ifstream gt_pose_file(GT_Poses_File_Path);
    if (!gt_pose_file.is_open())
    {
        LOG_FILE_ERROR(GT_Poses_File_Path);
        exit(1);
    }
    // read line by line, and see if the first line is read
    std::string line;
    bool b_first_line = true;
    if (dataset_type == "EuRoC")
    {
        Eigen::Matrix4d Transf_frame2body;
        Eigen::Matrix4d inv_Transf_frame2body;
        Transf_frame2body.setIdentity();
        Transf_frame2body.block<3, 3>(0, 0) = rot_frame2body_left;
        Transf_frame2body.block<3, 1>(0, 3) = transl_frame2body_left;
        // can be made to decompose  to get inverse
        inv_Transf_frame2body = Transf_frame2body.inverse();

        Eigen::Matrix4d Transf_Poses;
        Eigen::Matrix4d inv_Transf_Poses;
        Transf_Poses.setIdentity();

        Eigen::Matrix4d frame2world;

        while (std::getline(gt_pose_file, line))
        {
            if (b_first_line)
            {
                b_first_line = false;
                continue;
            }

            std::stringstream ss(line);
            std::string gt_val;
            std::vector<double> csv_row_val;

            while (std::getline(ss, gt_val, ','))
            {
                try
                {
                    csv_row_val.push_back(std::stod(gt_val));
                }
                catch (const std::invalid_argument &e)
                {
                    std::cerr << "Invalid argument: " << e.what() << " for value (" << gt_val << ") from the file " << GT_Poses_File_Path << std::endl;
                }
                catch (const std::out_of_range &e)
                {
                    std::cerr << "Out of range exception: " << e.what() << " for value: " << gt_val << std::endl;
                }
            }

            GT_time_stamps.push_back(csv_row_val[0]);
            Eigen::Vector3d transl_val(csv_row_val[1], csv_row_val[2], csv_row_val[3]);
            Eigen::Quaterniond quat_val(csv_row_val[4], csv_row_val[5], csv_row_val[6], csv_row_val[7]);
            Eigen::Matrix3d rot_from_quat = quat_val.toRotationMatrix();

            Transf_Poses.block<3, 3>(0, 0) = rot_from_quat;
            Transf_Poses.block<3, 1>(0, 3) = transl_val;
            inv_Transf_Poses = Transf_Poses.inverse();

            frame2world = (inv_Transf_frame2body * inv_Transf_Poses).inverse();

            unaligned_GT_Rot.push_back(frame2world.block<3, 3>(0, 0));
            unaligned_GT_Transl.push_back(frame2world.block<3, 1>(0, 3));
        }
    }
    else
    {
        LOG_ERROR("Dataset type not supported!");
    }
}

void Dataset::Align_Images_and_GT_Poses()
{
    std::vector<double> time_stamp_diff_val;
    std::vector<unsigned> time_stamp_diff_indx;
    for (double img_time_stamp : Img_time_stamps)
    {
        time_stamp_diff_val.clear();
        for (double gt_time_stamp : GT_time_stamps)
        {
            time_stamp_diff_val.push_back(std::abs(img_time_stamp - gt_time_stamp));
        }
        auto min_diff = std::min_element(std::begin(time_stamp_diff_val), std::end(time_stamp_diff_val));
        int min_index;
        if (min_diff != time_stamp_diff_val.end())
        {
            min_index = std::distance(std::begin(time_stamp_diff_val), min_diff);
        }
        else
        {
            LOG_ERROR("Empty vector for time stamp difference vector");
        }

        aligned_GT_Rot.push_back(unaligned_GT_Rot[min_index]);
        aligned_GT_Transl.push_back(unaligned_GT_Transl[min_index]);
    }
}

void Dataset::PrintDatasetInfo()
{
    std::cout << "Left Camera Resolution: " << left_res[0] << "x" << left_res[1] << std::endl;
    std::cout << "\nRight Camera Resolution: " << right_res[0] << "x" << right_res[1] << std::endl;

    std::cout << "\nLeft Camera Intrinsics: ";
    for (const auto &value : left_intr)
        std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nRight Camera Intrinsics: ";
    for (const auto &value : right_intr)
        std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nStereo Extrinsic Parameters (Left to Right): \n";

    std::cout << "\nRotation Matrix: \n";
    for (const auto &row : rot_mat_21)
    {
        for (const auto &value : row)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nTranslation Vector: \n";
    for (const auto &value : trans_vec_21)
        std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nFundamental Matrix: \n";
    for (const auto &row : fund_mat_21)
    {
        for (const auto &value : row)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nStereo Extrinsic Parameters (Right to Left): \n";

    std::cout << "\nRotation Matrix: \n";
    for (const auto &row : rot_mat_12)
    {
        for (const auto &value : row)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nTranslation Vector: \n";
    for (const auto &value : trans_vec_12)
        std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nFundamental Matrix: \n";
    for (const auto &row : fund_mat_12)
    {
        for (const auto &value : row)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nStereo Camera Parameters: \n";
    std::cout << "Focal Length: " << focal_length << " pixels" << std::endl;
    std::cout << "Baseline: " << baseline << " meters" << std::endl;

    std::cout << "\n"
              << std::endl;
}

void Dataset::onMouse(int event, int x, int y, int, void *)
{
    if (event == cv::EVENT_MOUSEMOVE)
    {
        if (merged_visualization_global.empty())
            return;

        int left_width = merged_visualization_global.cols / 2;

        std::string coord_text;
        if (x < left_width)
        {
            coord_text = "Left Image: (" + std::to_string(x) + ", " + std::to_string(y) + ")";
        }
        else
        {
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