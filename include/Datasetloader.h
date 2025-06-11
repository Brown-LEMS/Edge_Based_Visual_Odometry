#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>

class ThirdOrderEdgeDetectionCPU;

class EuRoCImageIterator {
private:
    std::ifstream csv_file;
    std::string left_path;
    std::string right_path;
    bool first_line_skipped = false;
    
public:
    EuRoCImageIterator(const std::string& csv_path, const std::string& left_path, const std::string& right_path);
    ~EuRoCImageIterator();
    
    bool hasNext() const;
    std::pair<cv::Mat, cv::Mat> getNext(double& timestamp);
};

class DatasetLoader {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    DatasetLoader(YAML::Node, bool);
    
    // Load images and disparity maps
    bool LoadDataset(int max_pairs = 100000);
    bool LoadCalibration();
    bool LoadGroundTruth();

    std::pair<cv::Mat, cv::Mat> GetImagePair(size_t index) const;
    cv::Mat GetLeftDisparityMap(size_t index) const;
    
    //stream-based
    bool HasNextImagePair() const;
    std::pair<cv::Mat, cv::Mat> GetNextImagePair();
    cv::Mat GetNextLeftDisparityMap();

    // Edge processing methods
    std::pair<std::vector<cv::Point2d>, std::vector<double>> GetOrLoadEdges(
        const cv::Mat& image, 
        const std::string& cache_path, 
        std::shared_ptr<ThirdOrderEdgeDetectionCPU>& detector);
    
    // Calibration access
    const std::vector<double>& GetLeftIntrinsics() const { return left_intr; }
    const std::vector<double>& GetRightIntrinsics() const { return right_intr; }
    const std::vector<std::vector<double>>& GetRotationMatrix21() const { return rot_mat_21; }
    const std::vector<double>& GetTranslationVector21() const { return trans_vec_21; }
    const std::vector<std::vector<double>>& GetFundamentalMatrix21() const { return fund_mat_21; }
    const std::vector<std::vector<double>>& GetRotationMatrix12() const { return rot_mat_12; }
    const std::vector<double>& GetTranslationVector12() const { return trans_vec_12; }
    const std::vector<std::vector<double>>& GetFundamentalMatrix12() const { return fund_mat_12; }
    
    // Ground truth access
    std::vector<Eigen::Matrix3d> GetAlignedRotations() const { return aligned_GT_Rot; }
    std::vector<Eigen::Vector3d> GetAlignedTranslations() const { return aligned_GT_Transl; }
    
    // Utility methods
    Eigen::Matrix3d ConvertToEigenMatrix(const std::vector<std::vector<double>>& matrix) const;
    void PrintDatasetInfo() const;

private:
    // Dataset type and paths
    std::string dataset_type;
    std::string dataset_path;
    std::string sequence_name;
    std::string output_path;
    
    // Current image index for streaming
    size_t current_image_index = 0;
    
    // Calibration parameters
    std::vector<int> left_res;
    std::vector<int> right_res;
    std::vector<double> left_intr;
    std::vector<double> right_intr;
    std::vector<double> left_dist_coeffs;
    std::vector<double> right_dist_coeffs;
    std::vector<std::vector<double>> rot_mat_21;
    std::vector<double> trans_vec_21;
    std::vector<std::vector<double>> fund_mat_21;
    std::vector<std::vector<double>> rot_mat_12;
    std::vector<double> trans_vec_12;
    std::vector<std::vector<double>> fund_mat_12;
    double focal_length;
    double baseline;
    
    // Frame to body transform for EuRoC
    Eigen::Matrix3d rot_frame2body_left;
    Eigen::Vector3d transl_frame2body_left;
     // Ground truth data
    std::vector<double> GT_time_stamps;
    std::vector<double> Img_time_stamps;
    std::vector<Eigen::Matrix3d> unaligned_GT_Rot;
    std::vector<Eigen::Vector3d> unaligned_GT_Transl;
    std::vector<Eigen::Matrix3d> aligned_GT_Rot;
    std::vector<Eigen::Vector3d> aligned_GT_Transl;
    
    // Loaded image data
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;
    std::vector<cv::Mat> left_ref_disparity_maps;
    
    // Helper methods
    std::vector<std::pair<cv::Mat, cv::Mat>> LoadEuRoCImages(
        const std::string& csv_path, 
        const std::string& left_path, 
        const std::string& right_path,
        int num_pairs);
        
    std::vector<std::pair<cv::Mat, cv::Mat>> LoadETH3DImages(
        const std::string& stereo_pairs_path, 
        int num_pairs);
        
    std::vector<cv::Mat> LoadETH3DLeftReferenceMaps(
        const std::string& stereo_pairs_path, 
        int num_maps);
        
    void LoadGTPoses(const std::string& GT_Poses_File_Path);
    void AlignImagesAndGTPoses();
    
     // Disparity map I/O
    cv::Mat LoadDisparityFromCSV(const std::string& path);
    void WriteDisparityToBinary(const std::string& filepath, const cv::Mat& disparity_map);
    cv::Mat ReadDisparityFromBinary(const std::string& filepath);
    
    // Edge I/O
    void WriteEdgesToBinary(
        const std::string& filepath,
        const std::vector<cv::Point2d>& locations,
        const std::vector<double>& orientations);
        
    void ReadEdgesFromBinary(
        const std::string& filepath,
        std::vector<cv::Point2d>& locations,
        std::vector<double>& orientations);
};

#endif // DATASET_LOADER_H
