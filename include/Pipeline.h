#ifndef PIPELINE_H
#define PIPELINE_H

#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "Dataset.h"
#include "toed/cpu_toed.hpp"
#include "definitions.h"
#include "Frame.h"
#include "utility.h"
#include "MotionTracker.h"
#include "Stereo_Matches.h"
#include "Temporal_Matches.h"

//> status of the visual odometry pipeline
enum class PipelineStatus { STATUS_IMG_PREPARATION, \
                            STATUS_GET_STEREO_EDGE_CORRESPONDENCES, \
                            STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES, \
                            STATUS_TRACK_CAMERA_MOTION };

class Pipeline {

public:
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Pipeline> Ptr;

    //> Constructor
    Pipeline(Dataset::Ptr dataset);

    //> When new frame is created, jump to the pipeline status
    bool Add_Stereo_Frame();

    void prepare_Stereo_Images();
    void get_Stereo_Edge_Correspondences();
    void get_Temporal_Edge_Correspondences();

    //> setters
    void set_Stereo_Frame_Index(size_t frame_idx) { stereo_frame_idx = frame_idx; }

    //> get the pipeline status
    PipelineStatus get_Status() const { return status_; }

    //> Print pipeline status
    std::string print_Status() const {
        if      (status_ == PipelineStatus::STATUS_IMG_PREPARATION)                      return std::string("STATUS_IMG_PREPARATION");
        else if (status_ == PipelineStatus::STATUS_GET_STEREO_EDGE_CORRESPONDENCES)      return std::string("STATUS_GET_STEREO_EDGE_CORRESPONDENCES");
        else if (status_ == PipelineStatus::STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES)    return std::string("STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES");
        else if (status_ == PipelineStatus::STATUS_TRACK_CAMERA_MOTION)                  return std::string("STATUS_TRACK_CAMERA_MOTION");
        LOG_ERROR("[Developer Error] Need to apend status string in print_Status() function!");
        return std::string("STATUS_UNKNOWN");
    }

    //> Interact with the main function
    bool send_control_to_main = true;

    //> Stereo frames
    StereoFrame keyframe, current_frame;

private:
    size_t stereo_frame_idx;
    SpatialGrid left_spatial_grids;
    SpatialGrid right_spatial_grids;

    void initialize_TOED_and_Spatial_Grids() {
        //> Set the image dimensions
        dataset_->set_height(current_frame.left_image_undistorted.rows);
        dataset_->set_width(current_frame.left_image_undistorted.cols);

        //> Initialize the third-order edge detector class pointer
        TOED = ThirdOrderEdgeDetectionCPU::Ptr(new ThirdOrderEdgeDetectionCPU(dataset_->get_height(), dataset_->get_width()));

        //> Initialize the spatial grids with a cell size of defined GRID_SIZE
        left_spatial_grids = SpatialGrid(dataset_->get_width(), dataset_->get_height(), GRID_SIZE);
        right_spatial_grids = SpatialGrid(dataset_->get_width(), dataset_->get_height(), GRID_SIZE);
    };

    void ProcessEdges(const cv::Mat &image, std::vector<Edge> &edges);

    void set_Keyframe() {
        keyframe = current_frame;
        keyframe_stereo_edge_mates = current_frame_stereo_edge_mates;
        keyframe_stereo_left_constructor = current_frame_stereo_left_constructor;
        keyframe_stereo_left_constructor.stereo_frame = &keyframe;
        keyframe_stereo_left_constructor.left_disparity_map = keyframe.left_disparity_map;
        keyframe_stereo_left_constructor.right_disparity_map = keyframe.right_disparity_map;

        //> reset current_frame
        current_frame = StereoFrame();
        current_frame_stereo_edge_mates.clear();
        current_frame_stereo_left_constructor.clean_up_vector_data_structures();
    }

    void set_Stereo_Left_Constructor() {
        current_frame_stereo_edge_mates.clear();
        current_frame_stereo_left_constructor.clean_up_vector_data_structures();
        current_frame_stereo_left_constructor.stereo_frame = &current_frame;
        current_frame_stereo_left_constructor.left_disparity_map = current_frame.left_disparity_map;
        current_frame_stereo_left_constructor.right_disparity_map = current_frame.right_disparity_map;
    }

    /**
     * Estimate camera poses
     * @return true if success (defined as having sufficient number of inliers)
     */
    bool track_Camera_Motion();

    //> Evaluations are enabled if the dataset has ground truth disparity maps
    std::vector<cv::Mat> left_ref_disparity_maps, right_ref_disparity_maps;
    std::vector<cv::Mat> left_occlusion_masks, right_occlusion_masks;

    //> Status of the Visual Odometry Pipeline
    PipelineStatus status_ = PipelineStatus::STATUS_IMG_PREPARATION;

    //> Stereo edge pairs constructor (processing stereo edge correspondences before finalizing the 1-1 stereo edge mapping)
    Stereo_Edge_Pairs keyframe_stereo_left_constructor;
    Stereo_Edge_Pairs current_frame_stereo_left_constructor;

    //> Final 1-1 stereo edge pairs for keyframe and current frame
    std::vector<final_stereo_edge_pair> keyframe_stereo_edge_mates;
    std::vector<final_stereo_edge_pair> current_frame_stereo_edge_mates;

    //> Keyframe <-> current frame edge pairs
    std::vector<temporal_edge_pair> left_temporal_edge_mates;
    std::vector<temporal_edge_pair> right_temporal_edge_mates;

    //> Pointers to the classes
    Dataset::Ptr dataset_ = nullptr;
    ThirdOrderEdgeDetectionCPU::Ptr TOED = nullptr;
    Stereo_Matches::Ptr stereo_matches_engine = nullptr;
    Temporal_Matches::Ptr temporal_matches_engine = nullptr;
    Frame::Ptr Current_Frame  = nullptr;
    Frame::Ptr Previous_Frame = nullptr;
    Utility::Ptr utility_tool = nullptr;
    MotionTracker::Ptr Camera_Motion_Estimate = nullptr;
};


#endif
