#ifndef PIPELINE_CPP
#define PIPELINE_CPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Pipeline.h"
#include "definitions.h"


// =====================================================================================================================
// class Pipeline: visual odometry pipeline 
//
// ChangeLogs
//    Chien  24-01-17    Initially created.
//    Chien  24-01-18    Add (SIFT) feature detection, matching, and creating a rank-order list of correspondences
//
//> (c) LEMS, Brown University
//> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
// ======================================================================================================================

Pipeline::Pipeline() {

    //> Set zeros
    Num_Of_SIFT_Features = 0;
    Num_Of_Good_Feature_Matches = 0;
}

bool Pipeline::Add_Frame(Frame::Ptr frame) {
    //> Set the frame pointer
    Current_Frame = frame;

    do {
        switch (status_) {
            case PipelineStatus::STATUS_INITIALIZATION:
                //> Initialization: extract SIFT features on the first frame
                LOG_STATUS("INITIALIZATION");
                Num_Of_SIFT_Features = get_Features();
                send_control_to_main = true;
                break;
            case PipelineStatus::STATUS_GET_AND_MATCH_SIFT:
                //> Extract and match SIFT features of the current frame with the previous frame
                LOG_STATUS("GET_AND_MATCH_SIFT");
                Num_Of_SIFT_Features = get_Features();
                Num_Of_Good_Feature_Matches = get_Feature_Correspondences();
                std::cout << "Number of good feature correspondences: " << Num_Of_Good_Feature_Matches << std::endl;
                send_control_to_main = false;
                break;
            case PipelineStatus::STATUS_ESTIMATE_RELATIVE_POSE:
                LOG_STATUS("ESTIMATE_RELATIVE_POSE");
                track_Camera_Motion();
                send_control_to_main = true;
                break;
        }
    } while ( !send_control_to_main );

    //> Swap the frame
    Previous_Frame = Current_Frame;

    return true;
}


int Pipeline::get_Features() {

    //> SIFT feature detector. Parameters same as the VLFeat default values, defined in Macros.
    //cv::Ptr<cv::xfeatures2d::SIFT> sift;
    cv::Ptr<cv::SIFT> sift;
    sift = cv::SIFT::create(SIFT_NFEATURES, SIFT_NOCTAVE_LAYERS, SIFT_CONTRAST_THRESHOLD, \
                            SIFT_EDGE_THRESHOLD, SIFT_GAUSSIAN_SIGMA);

    //> Detect SIFT keypoints and extract SIFT descriptor from the image
    std::vector<cv::KeyPoint> SIFT_Keypoints;
    cv::Mat SIFT_KeyPoint_Descriptors;
    sift->detect(Current_Frame->Image, SIFT_Keypoints);
    sift->compute(Current_Frame->Image, SIFT_Keypoints, SIFT_KeyPoint_Descriptors);

    //> Copy to the class Frame
    Current_Frame->SIFT_Locations = SIFT_Keypoints;
    Current_Frame->SIFT_Descriptors = SIFT_KeyPoint_Descriptors;

    //> Change to the next status
    status_ = PipelineStatus::STATUS_GET_AND_MATCH_SIFT;
    
    return SIFT_Keypoints.size();
}

int Pipeline::get_Feature_Correspondences() {

    //> Match SIFT features via OpenCV built-in KNN approach
    //  Matching direction: from previous frame -> current frame
    cv::BFMatcher matcher;
    std::vector< std::vector< cv::DMatch > > feature_matches;
    matcher.knnMatch( Previous_Frame->SIFT_Descriptors, Current_Frame->SIFT_Descriptors, feature_matches, K_IN_KNN_MATCHING );
    
    //> Apply Lowe's ratio test
    std::vector< cv::DMatch > Good_Matches;
    for (int i = 0; i < feature_matches.size(); ++i) {
        if (feature_matches[i][0].distance < LOWES_RATIO * feature_matches[i][1].distance) {
            Good_Matches.push_back(feature_matches[i][0]);
        }
    }

    //> Sort the matches based on their matching score (Euclidean distances of feature descriptors)
    std::sort( Good_Matches.begin(), Good_Matches.end(), less_than_Eucl_Dist() );

    //> Push back the "valid" match feature locations of the previous and current frames
    std::vector< std::pair<int, int> > Valid_Good_Matches_Index;
    for (int fi = 0; fi < Good_Matches.size(); fi++) {
        cv::DMatch f = Good_Matches[fi];

        //> Push back the match feature locations of the previous and current frames
        cv::Point2d previous_pt = Previous_Frame->SIFT_Locations[f.queryIdx].pt;
        cv::Point2d current_pt = Current_Frame->SIFT_Locations[f.trainIdx].pt;

        //> Rule out points unable to do bilinear interpolation
        if (previous_pt.x < 1 || previous_pt.y < 1 || current_pt.x < 1 || current_pt.y < 1) continue;
        if (previous_pt.x > (Current_Frame->Image.cols-1) || previous_pt.y > (Current_Frame->Image.rows-1) \
         || current_pt.x > (Current_Frame->Image.cols-1) || current_pt.y > (Current_Frame->Image.rows-1)) 
            continue;

        Eigen::Vector3d Previous_Match_Location = {previous_pt.x, previous_pt.y, 1.0};
        Eigen::Vector3d Current_Match_Location = {current_pt.x, current_pt.y, 1.0};

        //> Calculate gradient depth at corresponding feature locations, if required
        if (Current_Frame->need_depth_grad) {

            //> Check if the depth value is valid or not
            if ( Previous_Frame->Depth.at<double>( round(previous_pt.y), round(previous_pt.x) ) - 0.0 < EPSILON ) 
                continue;
            
            if ( Current_Frame->Depth.at<double>( round(current_pt.y), round(current_pt.x) ) - 0.0 < EPSILON ) 
                continue;

            double previous_depth = utility_tool->get_Interpolated_Depth(Previous_Frame, previous_pt);
            double current_depth  = utility_tool->get_Interpolated_Depth(Current_Frame,  current_pt );

            Eigen::Vector3d Previous_Match_gamma = Previous_Frame->inv_K * Previous_Match_Location;
            Eigen::Vector3d Current_Match_gamma = Current_Frame->inv_K * Current_Match_Location;

            //> Current frame features
            double grad_x = utility_tool->get_Interpolated_Gradient_Depth( Current_Frame, current_pt, "xi" );
            double grad_y = utility_tool->get_Interpolated_Gradient_Depth( Current_Frame, current_pt, "eta" );
            Current_Frame->gradient_Depth_at_Features.push_back( std::make_pair(grad_x, grad_y) );
            //> Previous frame features
            grad_x = utility_tool->get_Interpolated_Gradient_Depth( Previous_Frame, previous_pt, "xi" );
            grad_y = utility_tool->get_Interpolated_Gradient_Depth( Previous_Frame, previous_pt, "eta" );
            Previous_Frame->gradient_Depth_at_Features.push_back( std::make_pair(grad_x, grad_y) );

            Previous_Frame->Gamma.push_back(previous_depth * Previous_Match_gamma);
            Current_Frame->Gamma.push_back(current_depth * Current_Match_gamma);
        }

        //> 2D-3D matches of the previous and current frames
        Previous_Frame->SIFT_Match_Locations_Pixels.push_back(Previous_Match_Location);
        Current_Frame->SIFT_Match_Locations_Pixels.push_back(Current_Match_Location);
        
        //> Push back indices of valid feature matches
        Valid_Good_Matches_Index.push_back(std::make_pair(f.queryIdx, f.trainIdx));
    }

#if OPENCV_DISPLAY_CORRESPONDENCES
    utility_tool->Display_Feature_Correspondences(Previous_Frame->Image, Current_Frame->Image, \
                                                  Previous_Frame->SIFT_Locations, Current_Frame->SIFT_Locations, \
                                                  Good_Matches ) ;
#endif

    //> Get 3D matches
    //std::string ty = UTILITY_TOOLS::cvMat_Type( Current_Frame->Depth.type() );
    //printf("Depth Image: %s %dx%d \n", ty.c_str(), Current_Frame->Depth.cols, Current_Frame->Depth.rows );
    //std::cout << Current_Frame->Depth.at<float>(320, 240) << std::endl;

    //> Change to the next status
    status_ = PipelineStatus::STATUS_ESTIMATE_RELATIVE_POSE;

    return Valid_Good_Matches_Index.size();
} //> End of get_Feature_Correspondences

bool Pipeline::track_Camera_Motion() {

    // std::cout << "Need depth gradient? " << Current_Frame->need_depth_grad << std::endl;
    Camera_Motion_Estimate = std::shared_ptr<MotionTracker>(new MotionTracker());
    if (Current_Frame->need_depth_grad) {

        //> Estimate camera relative pose with depth prior
        Camera_Motion_Estimate->get_Relative_Pose_from_RANSAC(Current_Frame, Previous_Frame, Num_Of_Good_Feature_Matches, true);

        std::cout << "RANSAC Estimation:" << std::endl;
        std::cout << "- Rotation:" << std::endl;
        std::cout << Camera_Motion_Estimate->Final_Rel_Rot << std::endl;
        std::cout << "- Translation:" << std::endl;
        std::cout << Camera_Motion_Estimate->Final_Rel_Transl << std::endl;
        double inlier_ratio = (double)(Camera_Motion_Estimate->Final_Num_Of_Inlier_Support) / (double)(Num_Of_Good_Feature_Matches);
        std::cout << "Number of Supporting Inliers:" << Camera_Motion_Estimate->Final_Num_Of_Inlier_Support << "(" << inlier_ratio << "%)" << std::endl;
    }
    return true;
}


#endif