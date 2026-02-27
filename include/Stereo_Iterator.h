#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "toed/cpu_toed.hpp"

// =====================================================================================================================
// class Stereo_Iterator: image iterator to load images from dataset once at a time
//
// ChangeLogs
//    Jue  25-06-14    Initially created.
//
//> (c) LEMS, Brown University
//> Jue Han (jhan192@brown.edu)
// ======================================================================================================================

struct StereoFrame
{
    cv::Mat left_image;
    cv::Mat right_image;
    cv::Mat left_image_undistorted;
    cv::Mat right_image_undistorted;
    double timestamp;

    //> gradients in x and y directions
    cv::Mat left_image_gradients_x;
    cv::Mat right_image_gradients_x;
    cv::Mat left_image_gradients_y;
    cv::Mat right_image_gradients_y;

    //> third-order edges
    std::vector<Edge> left_edges;
    std::vector<Edge> right_edges;

    //> disparity maps
    cv::Mat left_disparity_map;
    cv::Mat right_disparity_map;

    // bool has_gt = false;
    Eigen::Matrix3d gt_rotation;
    Eigen::Vector3d gt_translation;
};

struct GTPose
{
    double timestamp;
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;

    GTPose(double ts, const Eigen::Matrix3d &R, const Eigen::Vector3d &T)
        : timestamp(ts), rotation(R), translation(T) {}

    bool operator<(const GTPose &other) const
    {
        return timestamp < other.timestamp;
    }
};

// ---------------------------------------------------------------------------------------------------------------------

class StereoIterator
{
public:
    virtual bool hasNext() = 0;
    virtual bool getNext(StereoFrame &frame) = 0;
    virtual void reset() = 0;
    virtual ~StereoIterator() = default;
};

class EuRoCIterator : public StereoIterator
{
public:
    EuRoCIterator(const std::string &csv_path,
                  const std::string &left_path,
                  const std::string &right_path);

    bool hasNext() override;
    bool getNext(StereoFrame &frame) override;
    void reset() override;

private:
    std::ifstream csv_file;
    std::string left_path, right_path, csv_path;
    bool first_line_skipped = false;
};

class ETH3DIterator : public StereoIterator
{
public:
    ETH3DIterator(const std::string &stereo_pairs_path);

    bool hasNext() override;
    bool getNext(StereoFrame &frame) override;
    void reset() override;
    bool readETH3DGroundTruth(const std::string &images_file, StereoFrame &frame);

private:
    std::vector<std::string> folders;
    size_t current_index = 0;
};

class ETH3DSLAMIterator : public StereoIterator
{
public:
    ETH3DSLAMIterator(const std::string &dataset_path);

    bool hasNext() override;
    bool getNext(StereoFrame &frame) override;
    void reset() override;

private:
    bool loadImageList();
    bool loadGroundTruth();
    bool findClosestGTPose(double timestamp, Eigen::Matrix3d &R, Eigen::Vector3d &T);

    std::string dataset_path;
    std::vector<std::pair<double, std::string>> image_list; // (timestamp, filename)
    std::vector<GTPose> gt_poses;
    size_t current_index = 0;
};

// ---------------------------------------------------------------------------------------------------------------------

class GTPoseIterator
{
public:
    virtual bool hasNext() = 0;
    virtual bool getNext(Eigen::Matrix3d &R, Eigen::Vector3d &T, double &timestamp) = 0;
    virtual ~GTPoseIterator() = default;
};

class EuRoCGTPoseIterator : public GTPoseIterator
{
public:
    EuRoCGTPoseIterator(const std::string &gt_file,
                        const Eigen::Matrix3d &R_frame2body,
                        const Eigen::Vector3d &T_frame2body);

    bool hasNext() override;
    bool getNext(Eigen::Matrix3d &R, Eigen::Vector3d &T, double &timestamp) override;

private:
    std::ifstream gt_stream;
    Eigen::Matrix4d inv_T_frame2body;
    bool first_line_skipped = false;
};

// ---------------------------------------------------------------------------------------------------------------------

class GTPoseAligner
{
public:
    GTPoseAligner(std::unique_ptr<GTPoseIterator> gt_iterator);
    bool getAlignedGT(double img_timestamp, Eigen::Matrix3d &R, Eigen::Vector3d &T);

private:
    std::vector<GTPose> poses;
};

// ---------------------------------------------------------------------------------------------------------------------

class AlignedStereoIterator : public StereoIterator
{
public:
    AlignedStereoIterator(
        std::unique_ptr<StereoIterator> image_iterator,
        std::unique_ptr<GTPoseAligner> gt_aligner);

    bool hasNext() override;
    bool getNext(StereoFrame &frame) override;
    void reset() override;

private:
    std::unique_ptr<StereoIterator> image_iterator;
    std::unique_ptr<GTPoseAligner> gt_aligner;
};

// ---------------------------------------------------------------------------------------------------------------------

namespace Iterators
{
    std::unique_ptr<StereoIterator> createEuRoCIterator(
        const std::string &csv_path,
        const std::string &left_path,
        const std::string &right_path);

    std::unique_ptr<StereoIterator> createETH3DIterator(
        const std::string &stereo_pairs_path);

    std::unique_ptr<StereoIterator> createETH3DSLAMIterator(
        const std::string &dataset_path);

    std::unique_ptr<GTPoseIterator> createEuRoCGTPoseIterator(
        const std::string &gt_file,
        const Eigen::Matrix3d &R_frame2body = Eigen::Matrix3d::Identity(),
        const Eigen::Vector3d &T_frame2body = Eigen::Vector3d::Zero());

    std::unique_ptr<StereoIterator> createAlignedEuRoCIterator(
        const std::string &csv_path,
        const std::string &left_path,
        const std::string &right_path,
        const std::string &gt_file,
        const Eigen::Matrix3d &R_frame2body = Eigen::Matrix3d::Identity(),
        const Eigen::Vector3d &T_frame2body = Eigen::Vector3d::Zero());
}