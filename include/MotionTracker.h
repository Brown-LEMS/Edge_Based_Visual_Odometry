#ifndef MOTION_TRACKER_H
#define MOTION_TRACKER_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "utility.h"

// =============================================================================================================================
// class MotionTracker: track camera motion, i.e., estimate camera poses, similar to "tracking" used in ORB-SLAM or OpenVSLAM,
//                      but the name aims to differentiate "camera motion tracks" from "feature tracks".
//
// ChangeLogs
//    Chien  24-05-18    Initially created. Add relative pose under a RANSAC loop with depth prior.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ======================================================================================================================

class MotionTracker {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MotionTracker> Ptr;

    //> Constructor (nothing special)
    MotionTracker();


    Eigen::Matrix3d Final_Rel_Rot;
    Eigen::Vector3d Final_Rel_Transl;    
    int Final_Num_Of_Inlier_Support;

private:
    
    template<typename T>
    T Uniform_Random_Number_Generator(T range_from, T range_to) {
        std::random_device                                          rand_dev;
        std::mt19937                                                rng(rand_dev());
        std::uniform_int_distribution<std::mt19937::result_type>    distr(range_from, range_to);
        return distr(rng);
    }

    //> Pointers to the classes
    Utility::Ptr utility_tool = nullptr;

    Eigen::Matrix3d Estimated_Rel_Rot;
    Eigen::Vector3d Estimated_Rel_Transl;
};


#endif
