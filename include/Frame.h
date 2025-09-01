#ifndef FRAME_H
#define FRAME_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Geometry>

// =======================================================================================================
// class Frame: structurize data associate to a frame/camera/view
//
// ChangeLogs
//    Chien  23-01-17    Initially created.
//    Chien  23-01-18    Add (SIFT) features and matches as part of the frame data
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// =======================================================================================================

class Frame {

public:
    //> Make the class shareable as a pointer
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    //> Constructors
    Frame(long, double, Eigen::Matrix3d, Eigen::Vector3d, cv::Mat);             //> ID, time_Stamp, R, T, image
    Frame(long, double, Eigen::Matrix3d, Eigen::Vector3d, cv::Mat, cv::Mat);    //> ID, time_Stamp, R, T, image, Depth
    Frame() {}

    //> Destructor
    //> TODO

    static std::shared_ptr<Frame> Create_Frame();

    //> Information of a frame
    cv::Mat Image;                              //> gray image
    cv::Mat Depth;                              //> depth image
    Eigen::Matrix3d Rel_Rot;                    //> relative rotation matrix
    Eigen::Vector3d Rel_Transl;                 //> relative translation vector
    Eigen::Matrix3d Abs_Rot;                    //> absolute rotation matrix
    Eigen::Vector3d Abs_Transl;                 //> absolute translation vector
    Eigen::Matrix3d K;                          //> Calibration matrix
    Eigen::Matrix3d inv_K;                      //> Inverse of Calibration matrix
    unsigned long ID = 0;                       //> frame id
    unsigned long KeyFrame_ID = 0;              //> keyframe id, if we have keyframe selection strategy
    bool is_KeyFrame = false;                   //> whether the frame is a keyframe or not
    double time_stamp;                          //> time stamp, only required by tum dataset type
    
    std::vector<cv::KeyPoint> SIFT_Locations;               //> a list of SIFT feature locations, used for matching
    cv::Mat SIFT_Descriptors;                               //> a list of SIFT feature descriptors used for matching
    std::vector<Eigen::Vector3d> SIFT_Match_Locations_Pixels;    //> a list of SIFT homogeneous locations in pixels. Indices are aligned with the 3D-2D correspondences.
    std::vector<bool> is_SIFT_Outlier;              //> whether the corresponding SIFT list index is an outlier or not
    std::vector<Eigen::Vector3d> Gamma;             //> 3D points arise from the depth map

    //> Used in GCC filter
    bool need_depth_grad = false;
    cv::Mat grad_Depth_xi;
    cv::Mat grad_Depth_eta;
    std::vector< std::pair<double, double>> gradient_Depth_at_Features;
    
private:
    
};


#endif
