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

struct alignas(32) Camera_Pose {
    //> Credits: https://github.com/PoseLib/PoseLib/blob/master/PoseLib/camera_pose.h
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //> Rotation is represented as a 3x3 rotation matrix or a quaternion (qw, qx, qy, qz)
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;

    // Constructors (Defaults to identity camera)
    Camera_Pose() : R(Eigen::Matrix3d::Identity()), t(0.0, 0.0, 0.0), q(Eigen::Quaterniond(R).normalized()) { }
    Camera_Pose(const Eigen::Quaterniond &qq, const Eigen::Vector3d &tt) : q(qq), t(tt) { R = q.normalized().toRotationMatrix(); }
    Camera_Pose(const Eigen::Matrix3d &R, const Eigen::Vector3d &tt) : R(R), t(tt) { q = Eigen::Quaterniond(R).normalized(); }

    // Helper functions
    inline Eigen::Matrix3d quat_to_R() const { return q.normalized().toRotationMatrix(); }
    inline Eigen::Matrix<double, 3, 4> make_Rt_in_3x4() const { 
        Eigen::Matrix<double, 3, 4> tmp = Eigen::Matrix<double, 3, 4>::Zero();
        tmp.block<3, 3>(0, 0) = R;
        tmp.col(3) = t;
        return tmp;
    }
    inline Eigen::Matrix<double, 4, 4> make_Rt_in_4x4() const {
        Eigen::Matrix<double, 4, 4> tmp = Eigen::Matrix<double, 4, 4>::Identity();
        tmp.block<3, 3>(0, 0) = R;
        tmp.block<3, 1>(0, 3) = t;
        return tmp;
    }
    inline Eigen::Matrix<double, 4, 4> inverse_in_4x4() const {
        Eigen::Matrix<double, 4, 4> tmp = Eigen::Matrix<double, 4, 4>::Identity();
        tmp.block<3, 3>(0, 0) = R.transpose();
        tmp.block<3, 1>(0, 3) = -R.transpose() * t;
        return tmp;
    }
    inline Eigen::Vector3d rotate(const Eigen::Vector3d &p) const { return R * p; }
    inline Eigen::Vector3d detransform(const Eigen::Vector3d &p) const { return R.transpose() * (p - t); }
    inline Eigen::Vector3d transform(const Eigen::Vector3d &p) const { return rotate(p) + t; }

    inline Eigen::Vector3d center() const { return -detransform(t); }

    inline void print_Camera_Pose(const std::string &pose_name) const {
        std::cout << pose_name << ": " << std::endl;
        std::cout << "- Rotation:\n" << R << std::endl;
        std::cout << "- Translation:\n" << t.transpose() << std::endl << std::endl;
    }
};

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
    Camera_Pose gt_camera_pose;
};

//> TODO: combine GTPose and Camera_Pose structure
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