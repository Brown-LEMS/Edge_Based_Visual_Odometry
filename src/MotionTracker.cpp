#ifndef MOTION_TRACKER_CPP
#define MOTION_TRACKER_CPP

#include <limits>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include "MotionTracker.h"
#include "definitions.h"

// ============================================================================================================================
// class MotionTracker: track camera motion, i.e., estimate camera poses, similar to "tracking" used in ORB-SLAM or OpenVSLAM,
//                      but the name aims to differentiate "camera motion tracks" from "feature tracks".
//
// ChangeLogs
//    Chien  24-01-17    Initially created.
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ============================================================================================================================

MotionTracker::MotionTracker() {
    Estimated_Rel_Rot = Eigen::Matrix3d::Identity();
    Estimated_Rel_Transl << 0.0, 0.0, 0.0;
    Final_Rel_Rot = Eigen::Matrix3d::Identity();
    Final_Rel_Transl << 0.0, 0.0, 0.0;
    Final_Num_Of_Inlier_Support = 0;
}


#endif