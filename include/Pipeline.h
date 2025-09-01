#ifndef PIPELINE_H
#define PIPELINE_H

#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "Frame.h"
#include "utility.h"
#include "MotionTracker.h"

// =====================================================================================================================
// class Pipeline: visual odometry pipeline 
//
// ChangeLogs
//    Chien  23-01-17    Initially created. 
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ======================================================================================================================

//> status of the visual odometry pipeline
enum class PipelineStatus { STATUS_INITIALIZATION, \
                            STATUS_GET_AND_MATCH_SIFT, \
                            STATUS_ESTIMATE_RELATIVE_POSE };

//> Defined operator used to create a rank-ordered list of feature correspondences
struct less_than_Eucl_Dist {
    inline bool operator() ( const cv::DMatch& Des1, const cv::DMatch& Des2 ) {
        return ( Des1.distance < Des2.distance );
    }
};

class Pipeline {

public:
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Pipeline> Ptr;

    //> Constructor (nothing special)
    Pipeline();

    //> When new frame is created, jump to the pipeline status
    bool Add_Frame(Frame::Ptr frame);

    //> get the pipeline status
    PipelineStatus get_Status() const { return status_; }

    //> Print pipeline status
    std::string print_Status() const {
        if      (status_ == PipelineStatus::STATUS_INITIALIZATION)          return std::string("STATUS_INITIALIZATION");
        else if (status_ == PipelineStatus::STATUS_GET_AND_MATCH_SIFT)      return std::string("STATUS_GET_AND_MATCH_SIFT");
        else if (status_ == PipelineStatus::STATUS_ESTIMATE_RELATIVE_POSE)  return std::string("STATUS_ESTIMATE_RELATIVE_POSE");
        LOG_ERROR("[Developer Error] Need to apend status string in print_Status() function!");
        return std::string("STATUS_UNKNOWN");
    }

    //> Interact with the main function
    bool send_control_to_main = true;

    //> For now these member variables are used for dubugging purposes
    int Num_Of_SIFT_Features;
    int Num_Of_Good_Feature_Matches;

private:
    /**
     * Extract features of the current frame
     * @return Num of SIFT features
     */
    int get_Features();

    /**
     * Find the corresponding features in right image of current_frame_
     * @return Num of SIFT feature correspondences
     */
    int get_Feature_Correspondences();

    /**
     * Estimate camera poses
     * @return true if success (defined as having sufficient number of inliers)
     */
    bool track_Camera_Motion();

    //> Status of the Visual Odometry Pipeline
    PipelineStatus status_ = PipelineStatus::STATUS_INITIALIZATION;

    //> Pointers to the classes
    Frame::Ptr Current_Frame  = nullptr;
    Frame::Ptr Previous_Frame = nullptr;
    Utility::Ptr utility_tool = nullptr;
    MotionTracker::Ptr Camera_Motion_Estimate = nullptr;
};


#endif
