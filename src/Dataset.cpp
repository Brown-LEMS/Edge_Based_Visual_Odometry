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
#include "Stereo_Matches.h"

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

Dataset::Dataset(YAML::Node config_map) : config_file(config_map)
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
        file_info.has_gt = true;
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
        Eigen::Matrix3d calib_left = Eigen::Matrix3d::Identity();
        calib_left(0, 0) = camera_info.left.intrinsics[0];
        calib_left(0, 2) = camera_info.left.intrinsics[2];
        calib_left(1, 1) = camera_info.left.intrinsics[1];
        calib_left(1, 2) = camera_info.left.intrinsics[3];
        camera_info.left.K = calib_left;

        // Right camera
        camera_info.right.resolution = right_cam["resolution"].as<std::vector<int>>();
        camera_info.right.intrinsics = right_cam["intrinsics"].as<std::vector<double>>();
        camera_info.right.distortion = right_cam["distortion_coefficients"].as<std::vector<double>>();
        Eigen::Matrix3d calib_right = Eigen::Matrix3d::Identity();
        calib_right(0, 0) = camera_info.right.intrinsics[0];
        calib_right(0, 2) = camera_info.right.intrinsics[2];
        calib_right(1, 1) = camera_info.right.intrinsics[1];
        calib_right(1, 2) = camera_info.right.intrinsics[3];
        camera_info.right.K = calib_right;

        // Stereo right-from-left (R21, T21, F21)
        if (stereo["R21"] && stereo["T21"] && stereo["F21"])
        {
            camera_info.left.R = load_matrix(stereo["R21"]);
            camera_info.left.T = Eigen::Map<const Eigen::Vector3d>(stereo["T21"].as<std::vector<double>>().data());
            camera_info.left.F = load_matrix(stereo["F21"]);
        }
        else
        {
            LOG_ERROR("ERROR: Missing left-to-right stereo parameters R21/T21/F21");
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
            LOG_ERROR("ERROR: Missing right-to-left stereo parameters R12/T12/F12");
        }

        // ETH3D stereo focal length and baseline
        if (file_info.dataset_type == "ETH3D")
        {
            file_info.has_gt = true;
            if (stereo["focal_length"] && stereo["baseline"])
            {
                camera_info.focal_length = stereo["focal_length"].as<double>();
                camera_info.baseline = stereo["baseline"].as<double>();
            }
            else
            {
                LOG_ERROR("ERROR: Missing stereo parameters (focal_length, baseline) in YAML file!");
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
                           std::vector<cv::Mat> &right_ref_disparity_maps,
                           std::vector<cv::Mat> &left_occlusion_masks,
                           std::vector<cv::Mat> &right_occlusion_masks)
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
        LoadETH3DDisparityMaps(stereo_pairs_path, left_ref_disparity_maps, right_ref_disparity_maps);
        left_occlusion_masks = LoadETH3DOcclusionMasks(stereo_pairs_path, true);
        right_occlusion_masks = LoadETH3DOcclusionMasks(stereo_pairs_path, false);
    }
}

std::vector<cv::Mat> Dataset::LoadETH3DOcclusionMasks(const std::string &stereo_pairs_path, bool left)
{
    std::vector<cv::Mat> occlusion_masks;
    std::vector<std::string> stereo_folders;

    for (const auto &entry : std::filesystem::directory_iterator(stereo_pairs_path))
    {
        if (entry.is_directory())
        {
            stereo_folders.push_back(entry.path().string());
        }
    }

    std::sort(stereo_folders.begin(), stereo_folders.end());

    for (int i = 0; i < static_cast<int>(stereo_folders.size()); i++)
    {
        std::string folder_path = stereo_folders[i];
        std::string mask_filename = left ? "mask0nocc.png" : "mask1nocc.png";
        std::string mask_path = folder_path + "/" + mask_filename;

        cv::Mat mask;
        if (std::filesystem::exists(mask_path))
        {
            mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
            if (mask.empty())
            {
                std::cerr << "Warning: Could not load occlusion mask: " << mask_path << std::endl;
            }
        }
        else
        {
            std::cerr << "Warning: Occlusion mask not found: " << mask_path << std::endl;
        }
        occlusion_masks.push_back(mask);
    }

    return occlusion_masks;
}
void Dataset::LoadETH3DDisparityMaps(const std::string &stereo_pairs_path, std::vector<cv::Mat> &left_disparity_maps, std::vector<cv::Mat> &right_disparity_maps)
{
    std::vector<std::string> stereo_folders;

    for (const auto &entry : std::filesystem::directory_iterator(stereo_pairs_path))
    {
        if (entry.is_directory())
        {
            stereo_folders.push_back(entry.path().string());
        }
    }

    std::sort(stereo_folders.begin(), stereo_folders.end());

    for (int i = 0; i < static_cast<int>(stereo_folders.size()); i++)
    {
        std::string folder_path = stereo_folders[i];
        std::string disparity_pfm_left_path = folder_path + "/disp0GT.pfm";
        std::string disparity_pfm_right_path = folder_path + "/disp1GT.pfm";

        cv::Mat left_disparity_map, right_disparity_map;

        // Load left disparity
        if (std::filesystem::exists(disparity_pfm_left_path))
        {
            try
            {
                left_disparity_map = readPFM(disparity_pfm_left_path);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error reading left PFM file: " << e.what() << std::endl;
            }
        }
        else
        {
            std::cerr << "Warning: Left PFM file not found: " << disparity_pfm_left_path << std::endl;
        }

        // Load right disparity
        if (std::filesystem::exists(disparity_pfm_right_path))
        {
            try
            {
                right_disparity_map = readPFM(disparity_pfm_right_path);
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error reading right PFM file: " << e.what() << std::endl;
            }
        }
        else
        {
            std::cerr << "Warning: Right PFM file not found: " << disparity_pfm_right_path << std::endl;
        }

        if (!left_disparity_map.empty())
        {
            left_disparity_maps.push_back(left_disparity_map);
        }

        if (!right_disparity_map.empty())
        {
            right_disparity_maps.push_back(right_disparity_map);
        }
    }
#if DATASET_LOAD_VERBOSE
    std::cout << "Loaded " << left_disparity_maps.size() << " left disparity maps and " << right_disparity_maps.size() << " right disparity maps" << std::endl;
#endif
}

cv::Mat Dataset::readPFM(const std::string &file_path)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open PFM file: " + file_path);
    }

    // Read header line (PF or Pf)
    std::string header;
    std::getline(file, header);

    bool is_color = false;
    if (header == "PF")
    {
        is_color = true;
    }
    else if (header == "Pf")
    {
        is_color = false;
    }
    else
    {
        throw std::runtime_error("Not a PFM file: " + file_path);
    }

    // Read dimensions (width height)
    std::string dim_line;
    std::getline(file, dim_line);

    int width, height;
    std::istringstream dim_stream(dim_line);
    if (!(dim_stream >> width >> height))
    {
        throw std::runtime_error("Malformed PFM header (dimensions): " + file_path);
    }

    // Read scale (determines endianness)
    std::string scale_line;
    std::getline(file, scale_line);
    float scale = std::stof(scale_line);

    bool little_endian = (scale < 0);
    if (scale < 0)
    {
        scale = -scale;
    }

    // Read binary float data
    int num_channels = is_color ? 3 : 1;
    int num_elements = width * height * num_channels;

    std::vector<float> data(num_elements);
    file.read(reinterpret_cast<char *>(data.data()), num_elements * sizeof(float));

    if (file.gcount() != num_elements * sizeof(float))
    {
        throw std::runtime_error("Failed to read all data from PFM file: " + file_path);
    }

    // Handle endianness conversion if needed
    // Check system endianness
    union
    {
        uint32_t i;
        char c[4];
    } test_endian = {0x01020304};
    bool system_is_little_endian = (test_endian.c[0] == 0x04);

    // If file endianness doesn't match system endianness, swap bytes
    if (little_endian != system_is_little_endian)
    {
        for (int i = 0; i < num_elements; i++)
        {
            char *bytes = reinterpret_cast<char *>(&data[i]);
            std::swap(bytes[0], bytes[3]);
            std::swap(bytes[1], bytes[2]);
        }
    }

    // Reshape data into Mat (height, width, channels)
    cv::Mat mat;
    if (is_color)
    {
        mat = cv::Mat(height, width, CV_32FC3, data.data()).clone();
    }
    else
    {
        mat = cv::Mat(height, width, CV_32FC1, data.data()).clone();
    }

    // Flip vertically (PFM format stores top-to-bottom, OpenCV expects bottom-to-top)
    cv::flip(mat, mat, 0);

    return mat;
}

bool Dataset::readDispMiddlebury(const std::string &disp_file_path, cv::Mat &disparity, cv::Mat &valid_mask)
{
    try
    {
        // Extract base path and construct mask path
        // disp_file_path: .../disp0GT.pfm
        // mask_path: .../mask0nocc.png
        std::string mask_path = disp_file_path;
        size_t pos = mask_path.find("disp0.pfm");
        if (pos == std::string::npos)
        {
            std::cerr << "Error: disp_file_path must contain 'disp0.pfm'" << std::endl;
            return false;
        }
        mask_path.replace(pos, 11, "mask0nocc.png");

        // Read disparity from PFM file
        cv::Mat disp_full = readPFM(disp_file_path);

        // Ensure it's single channel (2D)
        if (disp_full.channels() != 1)
        {
            // If it's 3-channel, take first channel or convert to grayscale
            if (disp_full.channels() == 3)
            {
                std::vector<cv::Mat> channels;
                cv::split(disp_full, channels);
                disparity = channels[0].clone();
            }
            else
            {
                std::cerr << "Error: Unexpected number of channels in PFM file" << std::endl;
                return false;
            }
        }
        else
        {
            disparity = disp_full.clone();
        }

        // Convert to float32 if needed
        if (disparity.type() != CV_32F)
        {
            disparity.convertTo(disparity, CV_32F);
        }

        // Read non-occlusion mask from PNG
        cv::Mat mask_img = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        if (mask_img.empty())
        {
            std::cerr << "Error: Cannot read mask file: " << mask_path << std::endl;
            return false;
        }

        // Create validity mask: mask == 255 (non-occluded pixels)
        // Mirrors: nocc_pix = imageio.imread(nocc_pix) == 255
        valid_mask = (mask_img == 255);
        valid_mask.convertTo(valid_mask, CV_8U); // Convert bool to uint8 (0 or 255)

        // Verify mask has valid pixels
        if (cv::countNonZero(valid_mask) == 0)
        {
            std::cerr << "Warning: No valid pixels in mask" << std::endl;
            return false;
        }

        // Ensure disparity and mask have same dimensions
        if (disparity.size() != valid_mask.size())
        {
            std::cerr << "Error: Disparity and mask size mismatch" << std::endl;
            return false;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading Middlebury disparity: " << e.what() << std::endl;
        return false;
    }
}

bool Dataset::readDispETH3D(const std::string &disp_file_path, cv::Mat &disparity, cv::Mat &valid_mask)
{
    try
    {
        // Extract base path and construct mask path
        // disp_file_path: .../disp0GT.pfm
        // mask_path: .../mask0nocc.png
        std::string mask_path = disp_file_path;
        size_t pos = mask_path.find("disp0GT.pfm");
        if (pos == std::string::npos)
        {
            std::cerr << "Error: disp_file_path must contain 'disp0GT.pfm'" << std::endl;
            return false;
        }
        mask_path.replace(pos, 11, "mask0nocc.png");

        // Read disparity from PFM file
        cv::Mat disp_full = readPFM(disp_file_path);

        // Ensure it's single channel (2D)
        if (disp_full.channels() != 1)
        {
            // If it's 3-channel, take first channel or convert to grayscale
            if (disp_full.channels() == 3)
            {
                std::vector<cv::Mat> channels;
                cv::split(disp_full, channels);
                disparity = channels[0].clone();
            }
            else
            {
                std::cerr << "Error: Unexpected number of channels in PFM file" << std::endl;
                return false;
            }
        }
        else
        {
            disparity = disp_full.clone();
        }

        // Convert to float32 if needed
        if (disparity.type() != CV_32F)
        {
            disparity.convertTo(disparity, CV_32F);
        }

        // Read non-occlusion mask from PNG
        cv::Mat mask_img = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
        if (mask_img.empty())
        {
            std::cerr << "Error: Cannot read mask file: " << mask_path << std::endl;
            return false;
        }

        // Create validity mask: mask == 255 (non-occluded pixels)
        // This mirrors the Python code: occ_mask == 255
        valid_mask = (mask_img == 255);
        valid_mask.convertTo(valid_mask, CV_8U); // Convert bool to uint8 (0 or 255)

        // Ensure disparity and mask have same dimensions
        if (disparity.size() != valid_mask.size())
        {
            std::cerr << "Error: Disparity and mask size mismatch" << std::endl;
            return false;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading ETH3D disparity: " << e.what() << std::endl;
        return false;
    }
}

bool Dataset::read_edges_and_disparities_from_file(const std::string &file_path, std::vector<Edge> &edges, std::vector<double> &edge_disparities)
{
    try
    {
        std::ifstream file(file_path);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file: " << file_path << std::endl;
            return false;
        }

        edges.clear();
        edge_disparities.clear();

        std::string line;
        int edge_index = 0;

        while (std::getline(file, line))
        {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#')
            {
                continue;
            }

            // Parse the line - support both space and tab separators
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;

            while (iss >> token)
            {
                tokens.push_back(token);
            }

            // Expect 4 columns: x, y, orientation, disparity
            if (tokens.size() < 4)
            {
                std::cerr << "Warning: Skipping line with insufficient columns: " << line << std::endl;
                continue;
            }

            try
            {
                double y = std::stod(tokens[0]);
                double x = std::stod(tokens[1]);
                double orientation = std::stod(tokens[2]);
                double disparity = std::stod(tokens[3]);

                // Create Edge object
                cv::Point2d location(x, y);
                Edge edge(location, orientation, false, 0); // b_isEmpty=false, frame_source=0
                edge.index = edge_index;

                edges.push_back(edge);
                edge_disparities.push_back(disparity);

                edge_index++;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Warning: Error parsing line: " << line << " - " << e.what() << std::endl;
                continue;
            }
        }

        file.close();

        if (edges.empty())
        {
            std::cerr << "Warning: No valid edge data found in file: " << file_path << std::endl;
            return false;
        }
#if DATASET_LOAD_VERBOSE
        std::cout << "Successfully loaded " << edges.size() << " third-order edges from: " << file_path << std::endl;
#endif
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading third-order edges from file: " << e.what() << std::endl;
        return false;
    }
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
    std::cout << "Baseline: " << camera_info.baseline << " meters" << std::endl
              << std::endl;
}

#endif