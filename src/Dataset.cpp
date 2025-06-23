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

auto load_matrix = [](const YAML::Node &node) -> Eigen::Matrix3d
{
    Eigen::Matrix3d mat;
    int i = 0;
    for (const auto &row : node)
    {
        mat.row(i++) = Eigen::Map<const Eigen::Vector3d>(row.as<std::vector<double>>().data());
    }
    return mat;
};

Dataset::Dataset(YAML::Node config_map, bool use_GCC_filter) : config_file(config_map), compute_grad_depth(use_GCC_filter)
{

#if USE_DEFINED_NUM_OF_CORES
    omp_threads = USE_NUM_CORES_FOR_OMP;
#else
    omp_threads = omp_get_num_procs();
#endif

    file_info.dataset_type = config_file["dataset_type"].as<std::string>();
    file_info.dataset_path = config_file["dataset_dir"].as<std::string>();
    file_info.output_path = config_file["output_dir"].as<std::string>();
    file_info.sequence_name = config_file["sequence_name"].as<std::string>();
    if (file_info.dataset_type == "EuRoC")
    {
        file_info.GT_file_name = config_file["state_GT_estimate_file_name"].as<std::string>();
    }

    try
    {

        YAML::Node left_cam = config_file["left_camera"];
        YAML::Node right_cam = config_file["right_camera"];
        YAML::Node stereo = config_file["stereo"];
        YAML::Node frame_to_body = config_file["frame_to_body"];

        auto load_matrix = [](const YAML::Node &node) -> Eigen::Matrix3d
        {
            Eigen::Matrix3d mat;
            int i = 0;
            for (const auto &row : node)
            {
                mat.row(i++) = Eigen::Map<const Eigen::Vector3d>(row.as<std::vector<double>>().data());
            }
            return mat;
        };
        // Left camera
        camera_info.left.resolution = left_cam["resolution"].as<std::vector<int>>();
        camera_info.left.intrinsics = left_cam["intrinsics"].as<std::vector<double>>();
        camera_info.left.distortion = left_cam["distortion_coefficients"].as<std::vector<double>>();

        // Right camera
        camera_info.right.resolution = right_cam["resolution"].as<std::vector<int>>();
        camera_info.right.intrinsics = right_cam["intrinsics"].as<std::vector<double>>();
        camera_info.right.distortion = right_cam["distortion_coefficients"].as<std::vector<double>>();

        // Stereo right-from-left (R21, T21, F21)
        if (stereo["R21"] && stereo["T21"] && stereo["F21"])
        {
            camera_info.left.R = load_matrix(stereo["R21"]);
            camera_info.left.T = Eigen::Map<const Eigen::Vector3d>(stereo["T21"].as<std::vector<double>>().data());
            camera_info.left.F = load_matrix(stereo["F21"]);
        }
        else
        {
            std::cerr << "ERROR: Missing left-to-right stereo parameters R21/T21/F21" << std::endl;
        }

        // Stereo left-from-right (R12, T12, F12)
        if (stereo["R12"] && stereo["T12"] && stereo["F12"])
        {
            camera_info.right.R = load_matrix(stereo["R12"]);
            camera_info.right.T = Eigen::Map<const Eigen::Vector3d>(stereo["T12"].as<std::vector<double>>().data());
            camera_info.right.F = load_matrix(stereo["F12"]);
        }
        else
        {
            std::cerr << "ERROR: Missing right-to-left stereo parameters R12/T12/F12" << std::endl;
        }

        // ETH3D stereo focal length and baseline
        if (file_info.dataset_type == "ETH3D")
        {
            if (stereo["focal_length"] && stereo["baseline"])
            {
                camera_info.focal_length = stereo["focal_length"].as<double>();
                camera_info.baseline = stereo["baseline"].as<double>();
            }
            else
            {
                std::cerr << "ERROR: Missing stereo parameters (focal_length, baseline) in YAML file!" << std::endl;
            }
        }

        // EuRoc
        else if (file_info.dataset_type == "EuRoC")
        {
            if (frame_to_body["rotation"] && frame_to_body["translation"])
            {
                camera_info.rot_frame2body_left = Eigen::Map<Eigen::Matrix3d>(frame_to_body["rotation"].as<std::vector<double>>().data()).transpose();
                camera_info.transl_frame2body_left = Eigen::Map<Eigen::Vector3d>(frame_to_body["translation"].as<std::vector<double>>().data());
            }
            else
            {
                LOG_ERROR("Missing relative rotation and translation from the left camera to the body coordinate (should be given by cam0/sensor.yaml)");
            }
        }
    }
    catch (const YAML::Exception &e)
    {
        std::cerr << "ERROR: Could not parse YAML file! " << e.what() << std::endl;
    }

    Total_Num_Of_Imgs = 0;
}

void Dataset::load_dataset(const std::string &dataset_type,
                           std::vector<cv::Mat> &left_ref_disparity_maps,
                           int num_pairs)
{
    if (dataset_type == "EuRoC")
    {
        std::string base = file_info.dataset_path + "/" + file_info.sequence_name + "/mav0/";

        std::string cam0_path = base + "cam0/data/";
        std::string cam1_path = base + "cam1/data/";
        std::string csv_path = base + "cam0/data.csv";
        std::string gt_path = base + "state_groundtruth_estimate0/data.csv";

        stereo_iterator = Iterators::createAlignedEuRoCIterator(
            csv_path,
            cam0_path,
            cam1_path,
            gt_path);
    }
    else if (dataset_type == "ETH3D")
    {
        std::string stereo_pairs_path = file_info.dataset_path + "/" + file_info.sequence_name + "/stereo_pairs";
        stereo_iterator = Iterators::createETH3DIterator(stereo_pairs_path);
        left_ref_disparity_maps = LoadETH3DLeftReferenceMaps(stereo_pairs_path, num_pairs);
    }
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

void Dataset::PrintDatasetInfo()
{
    std::cout << "Left Camera Resolution: " << camera_info.left.resolution[0] << "x" << camera_info.left.resolution[1] << std::endl;
    std::cout << "\nRight Camera Resolution: " << camera_info.right.resolution[0] << "x" << camera_info.right.resolution[1] << std::endl;

    std::cout << "\nLeft Camera Intrinsics: ";
    for (const auto &value : camera_info.left.intrinsics)
        std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nRight Camera Intrinsics: ";
    for (const auto &value : camera_info.right.intrinsics)
        std::cout << value << " ";
    std::cout << std::endl;

    std::cout << "\nStereo Extrinsic Parameters (Left to Right): \n";

    std::cout << "\nRotation Matrix: \n";
    std::cout << camera_info.left.R << std::endl;

    std::cout << "\nTranslation Vector: \n";
    std::cout << camera_info.left.T << std::endl;
    std::cout << std::endl;

    std::cout << "\nFundamental Matrix: \n";
    std::cout << camera_info.left.F << std::endl;

    std::cout << "\nStereo Extrinsic Parameters (Right to Left): \n";

    std::cout << "\nRotation Matrix: \n";
    std::cout << camera_info.right.R << std::endl;

    std::cout << "\nTranslation Vector: \n";
    std::cout << camera_info.right.T << std::endl;

    std::cout << "\nFundamental Matrix: \n";
    std::cout << camera_info.right.F << std::endl;

    std::cout << "\nStereo Camera Parameters: \n";
    std::cout << "Focal Length: " << camera_info.focal_length << " pixels" << std::endl;
    std::cout << "Baseline: " << camera_info.baseline << " meters" << std::endl;

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