#include "DatasetLoader.h"
#include "ThirdOrderEdgeDetectionCPU.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <stdexcept>
#include <limits>

#if USE_GLOGS
#include <glog/logging.h>
#define LOG_ERROR(msg) LOG(ERROR) << msg
#define LOG_INFO(msg) LOG(INFO) << msg
#define LOG_FILE_ERROR(filepath) LOG(ERROR) << "Could not open file: " << filepath
#else
#define LOG_ERROR(msg) std::cerr << "ERROR: " << msg << std::endl
#define LOG_INFO(msg) std::cout << "INFO: " << msg << std::endl
#define LOG_FILE_ERROR(filepath) std::cerr << "ERROR: Could not open file: " << filepath << std::endl
#endif



DatasetLoader::DatasetLoader(const YAML::Node& config_map) {
    dataset_path = config_map["dataset_dir"].as<std::string>();
    output_path = config_map["output_dir"].as<std::string>();
    sequence_name = config_map["sequence_name"].as<std::string>();
    dataset_type = config_map["dataset_type"].as<std::string>();
    
    LoadCalibration(config_map);
}

bool DatasetLoader::LoadCalibration(const YAML::Node& config_map) {
    try {
        if (dataset_type == "EuRoC") {
            YAML::Node left_cam = config_map["left_camera"];
            YAML::Node right_cam = config_map["right_camera"];
            YAML::Node stereo = config_map["stereo"];
            YAML::Node frame_to_body = config_map["frame_to_body"];

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
                LOG_ERROR("Missing left-to-right stereo parameters (R21, T21, F21) in YAML file!");
                return false;
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
                LOG_ERROR("Missing right-to-left stereo parameters (R12, T12, F12) in YAML file!");
                return false;
            }

            if (frame_to_body["rotation"] && frame_to_body["translation"]) {
                rot_frame2body_left = Eigen::Map<Eigen::Matrix3d>(
                    frame_to_body["rotation"].as<std::vector<double>>().data()).transpose();
                transl_frame2body_left = Eigen::Map<Eigen::Vector3d>(
                    frame_to_body["translation"].as<std::vector<double>>().data());
            } else {
                LOG_ERROR("Missing relative rotation and translation from the left camera to the body coordinate");
                return false;
            }
        }
        else if (dataset_type == "ETH3D") {
            YAML::Node left_cam = config_map["left_camera"];
            YAML::Node right_cam = config_map["right_camera"];
            YAML::Node stereo = config_map["stereo"];

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
                LOG_ERROR("Missing left-to-right stereo parameters (R21, T21, F21) in YAML file!");
                return false;
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
                LOG_ERROR("Missing right-to-left stereo parameters (R12, T12, F12) in YAML file!");
                return false;
            }
            
            if (stereo["focal_length"] && stereo["baseline"]) {
                focal_length = stereo["focal_length"].as<double>();
                baseline = stereo["baseline"].as<double>();
            } else {
                LOG_ERROR("Missing stereo parameters (focal_length, baseline) in YAML file!");
                return false;
            }
        }
        return true;
    } catch (const YAML::Exception &e) {
        LOG_ERROR("Could not parse YAML file! " + std::string(e.what()));
        return false;
    }
}

bool DatasetLoader::LoadDataset(int max_pairs) {
    left_ref_disparity_maps.clear();
    Img_time_stamps.clear();
    
    if (dataset_type == "EuRoC") {
        std::string left_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data/";
        std::string right_path = dataset_path + "/" + sequence_name + "/mav0/cam1/data/";
        std::string image_csv_path = dataset_path + "/" + sequence_name + "/mav0/cam0/data.csv";
        std::string ground_truth_path = dataset_path + "/" + sequence_name + "/mav0/state_groundtruth_estimate0/data.csv";

        euroc_stream = std::make_unique<EuRoCImageIterator>(image_csv_path, left_path, right_path);
        InitGTPoseStream(ground_truth_path);
        return euroc_stream != nullptr;
    }
    else if (dataset_type == "ETH3D") {
        std::string stereo_pairs_path = dataset_path + "/" + sequence_name + "/stereo_pairs";
        image_pairs = LoadETH3DImages(stereo_pairs_path, max_pairs);
        left_ref_disparity_maps = LoadETH3DLeftReferenceMaps(stereo_pairs_path, max_pairs);
    }
    
    current_image_index = 0;
    return !image_pairs.empty();
}

std::pair<cv::Mat, cv::Mat> DatasetLoader::GetImagePair(size_t index) const {
    if (index >= image_pairs.size()) {
        LOG_ERROR("Image pair index out of bounds: " + std::to_string(index));
        return {cv::Mat(), cv::Mat()};
    }
    return image_pairs[index];
}

cv::Mat DatasetLoader::GetLeftDisparityMap(size_t index) const {
    if (index >= left_ref_disparity_maps.size()) {
        LOG_ERROR("Disparity map index out of bounds: " + std::to_string(index));
        return cv::Mat();
    }
    return left_ref_disparity_maps[index];
}

bool DatasetLoader::HasNextImagePair() const {
    return current_image_index < image_pairs.size();
}

std::pair<cv::Mat, cv::Mat> DatasetLoader::GetNextImagePair() {
    if (!HasNextImagePair()) {
        return {cv::Mat(), cv::Mat()};
    }
    return image_pairs[current_image_index++];
}

cv::Mat DatasetLoader::GetNextLeftDisparityMap() {
    if (current_image_index >= left_ref_disparity_maps.size()) {
        return cv::Mat();
    }
    return left_ref_disparity_maps[current_image_index - 1]; // Match the image index
}

std::pair<std::vector<cv::Point2d>, std::vector<double>> DatasetLoader::GetOrLoadEdges(
    const cv::Mat& image, 
    const std::string& cache_path, 
    std::shared_ptr<ThirdOrderEdgeDetectionCPU>& detector) {
    
    std::vector<cv::Point2d> locations;
    std::vector<double> orientations;
    
    std::string bin_path = cache_path + ".bin";
    
    if (std::filesystem::exists(bin_path)) {
        ReadEdgesFromBinary(bin_path, locations, orientations);
    } else {
        // Create the output directory if it doesn't exist
        std::filesystem::create_directories(std::filesystem::path(bin_path).parent_path());
        
        detector->get_Third_Order_Edges(image);
        locations = detector->toed_locations;
        orientations = detector->toed_orientations;
        
        WriteEdgesToBinary(bin_path, locations, orientations);
    }
    
    return {locations, orientations};
}
/*
    Below is the implementation of the EuRoCImageIterator class, 
    which reads image pairs from a CSV file and loads corresponding images from specified directories.
*/
EuRoCImageIterator::EuRoCImageIterator(const std::string& csv_path, const std::string& left_path, const std::string& right_path) 
    : left_path(left_path), right_path(right_path) {
    csv_file.open(csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "ERROR: Could not open the CSV file located at " << csv_path << "!" << std::endl;
    }
}

EuRoCImageIterator::~EuRoCImageIterator() {
    if (csv_file.is_open()) {
        csv_file.close();
    }
}

bool EuRoCImageIterator::hasNext() const {
    return csv_file.is_open() && !csv_file.eof();
}

std::pair<cv::Mat, cv::Mat> EuRoCImageIterator::getNext(double& timestamp) {
    if (!hasNext()) {
        return {cv::Mat(), cv::Mat()};
    }
    
    std::string line;
    if (!std::getline(csv_file, line)) {
        return {cv::Mat(), cv::Mat()};
    }
    
    // Skip header line
    if (!first_line_skipped) {
        first_line_skipped = true;
        return getNext(timestamp);
    }
    
    std::istringstream line_stream(line);
    std::string timestamp_str;
    std::getline(line_stream, timestamp_str, ',');
    
    timestamp = std::stod(timestamp_str);
    
    std::string left_img_path = left_path + timestamp_str + ".png";
    std::string right_img_path = right_path + timestamp_str + ".png";
    
    cv::Mat left_img = cv::imread(left_img_path, cv::IMREAD_GRAYSCALE);
    cv::Mat right_img = cv::imread(right_img_path, cv::IMREAD_GRAYSCALE);
    
    if (left_img.empty() || right_img.empty()) {
        std::cerr << "ERROR: Could not load the images: " << left_img_path << " or " << right_img_path << "!" << std::endl;
        return getNext(timestamp); // Skip this pair and try the next one
    }
    
    return {left_img, right_img};
}

//just copied the align_images from dataset and changed it int atomic, need to be revised. 
void Dataset::Align_Images_and_GT_Poses(double img_time_stamp) {
   std::vector<double> time_stamp_diff_val;
   std::vector<unsigned> time_stamp_diff_indx;

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
    //is aligned_GT_Rot an attribute of Dataset class? Or this is something to return?
    aligned_GT_Rot.push_back(unaligned_GT_Rot[min_index]);
    aligned_GT_Transl.push_back(unaligned_GT_Transl[min_index]);

}

void DatasetLoader::InitGTPoseStream(const std::string& pose_file_path) {
    GT_time_stamps.clear();
    unaligned_GT_Rot.clear();
    unaligned_GT_Transl.clear();
    aligned_GT_Rot.clear();
    aligned_GT_Transl.clear();
    
    gt_pose_iterator = std::make_unique<GTPoseIterator>(
        pose_file_path, 
        rot_frame2body_left, 
        transl_frame2body_left, 
        dataset_type
    );
}
bool DatasetLoader::HasNextGTPose() const {
    return gt_pose_iterator && gt_pose_iterator->hasNext();
}

bool DatasetLoader::GetNextGTPose(double& timestamp, Eigen::Matrix3d& rotation, Eigen::Vector3d& translation) {
    if (!HasNextGTPose()) {
        return false;
    }
    
    if (gt_pose_iterator->getNext(timestamp, rotation, translation)) {
        GT_time_stamps.push_back(timestamp);
        unaligned_GT_Rot.push_back(rotation);
        unaligned_GT_Transl.push_back(translation);
        return true;
    }
    
    return false;
}