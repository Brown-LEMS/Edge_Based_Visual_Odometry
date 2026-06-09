#ifndef PIPELINE_H
#define PIPELINE_H

#include <deque>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "Dataset.h"
#include "Stereo_Iterator.h"
#include "toed/cpu_toed.hpp"
#include "definitions.h"
#include "utility.h"
#include "MotionTracker.h"
#include "Stereo_Matches.h"
#include "Temporal_Matches.h"

//> status of the visual odometry pipeline
enum class PipelineStatus
{
    STATUS_IMG_PREPARATION,
    STATUS_GET_STEREO_EDGE_CORRESPONDENCES,
    STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES,
    STATUS_TRACK_CAMERA_MOTION,
    STATUS_GET_POSE_FROM_QUAD_PAIRS,
    STATUS_KEYFRAME_DECISION,
    STATUS_SYSTEM_EXIT
};

class Pipeline
{

public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Pipeline> Ptr;

    //> Constructor
    Pipeline(Dataset::Ptr dataset);

    //> Destructor
    ~Pipeline();

    //> When new frame is created, jump to the pipeline status
    bool Add_Stereo_Frame();

    void prepare_Stereo_Images();
    void get_Stereo_Edge_Correspondences();
    void get_Temporal_Edge_Correspondences();
    void get_Pose_From_Quad_Pairs();
    void make_Keyframe_Decision();

    void Print_Stereo_Matches_Metrics_Statistics();
    void Print_Temporal_Matches_Metrics_Statistics();

    //> setters
    void set_Current_Stereo_Frame_Index(size_t frame_idx) { stereo_current_frame_idx = frame_idx; }
    void increment_Current_Stereo_Frame_Index() { stereo_current_frame_idx++; }

    //> get the pipeline status
    PipelineStatus get_Status() const { return status_; }
    size_t get_Keyframe_Index() const { return stereo_key_frame_idx; }
    size_t get_Current_Frame_Index() const { return stereo_current_frame_idx; }

    //> Print pipeline status
    std::string print_Status() const
    {
        if (status_ == PipelineStatus::STATUS_IMG_PREPARATION)
            return std::string("STATUS_IMG_PREPARATION");
        else if (status_ == PipelineStatus::STATUS_GET_STEREO_EDGE_CORRESPONDENCES)
            return std::string("STATUS_GET_STEREO_EDGE_CORRESPONDENCES");
        else if (status_ == PipelineStatus::STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES)
            return std::string("STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES");
        else if (status_ == PipelineStatus::STATUS_TRACK_CAMERA_MOTION)
            return std::string("STATUS_TRACK_CAMERA_MOTION");
        else if (status_ == PipelineStatus::STATUS_GET_POSE_FROM_QUAD_PAIRS)
            return std::string("STATUS_GET_POSE_FROM_QUAD_PAIRS");
        else if (status_ == PipelineStatus::STATUS_KEYFRAME_DECISION)
            return std::string("STATUS_KEYFRAME_DECISION");
        else if (status_ == PipelineStatus::STATUS_SYSTEM_EXIT)
            return std::string("STATUS_SYSTEM_EXIT");
        LOG_ERROR("[Developer Error] Need to apend status string in print_Status() function!");
        return std::string("STATUS_UNKNOWN");
    }

    //> Interact with the main function
    bool send_control_to_main = true;

    //> Stereo frames
    StereoFrame keyframe, current_frame;

    std::vector<Camera_Pose> estimated_poses;
    std::vector<Camera_Pose> cumulated_poses;

    void save_Current_Estimated_Pose() {
        estimated_abs_poses.push_back(std::make_pair(current_frame.timestamp, current_frame.estimated_camera_pose));
    }
    void save_KF_Estimated_Pose() {
        estimated_KF_abs_poses.push_back(std::make_pair(keyframe.timestamp, keyframe.estimated_camera_pose));
    }
    
    //> The estimated poses are stored as (timestamp, pose) in the world-to-rig convention.
    std::vector<std::pair<std::string, Camera_Pose>> get_Estimated_Poses() const { return estimated_abs_poses; }  
    std::vector<std::pair<std::string, Camera_Pose>> get_Estimated_KF_Poses() const { return estimated_KF_abs_poses; }
    Camera_Pose get_KF_abs_pose() const { return keyframe.estimated_camera_pose; }

    //> system exit message
    std::string system_exit_message;

    //> for debugging purpose
    bool b_end_pipeline_for_debug = false;

private:
    size_t stereo_key_frame_idx;
    size_t stereo_current_frame_idx;
    SpatialGrid left_spatial_grids;
    SpatialGrid right_spatial_grids;

    //> Results
    std::vector<std::pair<std::string, Camera_Pose>> estimated_abs_poses;
    std::vector<std::pair<std::string, Camera_Pose>> estimated_KF_abs_poses;

    void reset_spatial_grids() { left_spatial_grids.reset(); right_spatial_grids.reset(); }
    void initialize_TOED_and_Spatial_Grids()
    {
        //> Set the image dimensions
        dataset_->set_left_height(current_frame.left_image_undistorted.rows);
        dataset_->set_left_width(current_frame.left_image_undistorted.cols);
        dataset_->set_right_height(current_frame.right_image_undistorted.rows);
        dataset_->set_right_width(current_frame.right_image_undistorted.cols);

        //> Initialize the third-order edge detector class pointer
        TOED_left = ThirdOrderEdgeDetectionCPU::Ptr(new ThirdOrderEdgeDetectionCPU(dataset_->get_left_height(), dataset_->get_left_width()));
        TOED_right = ThirdOrderEdgeDetectionCPU::Ptr(new ThirdOrderEdgeDetectionCPU(dataset_->get_right_height(), dataset_->get_right_width()));

        //> Initialize the spatial grids with a cell size of defined GRID_SIZE
        left_spatial_grids = SpatialGrid(dataset_->get_left_width(), dataset_->get_left_height(), GRID_SIZE);
        right_spatial_grids = SpatialGrid(dataset_->get_right_width(), dataset_->get_right_height(), GRID_SIZE);
    };

    void ProcessEdges(const cv::Mat &image, std::vector<Edge> &edges, bool is_left);

    void set_Keyframe()
    {
        stereo_key_frame_idx = stereo_current_frame_idx;

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

    void set_Stereo_Left_Constructor()
    {
        current_frame_stereo_edge_mates.clear();
        current_frame_stereo_left_constructor.clean_up_vector_data_structures();
        current_frame_stereo_left_constructor.stereo_frame = &current_frame;
        current_frame_stereo_left_constructor.left_disparity_map = current_frame.left_disparity_map;
        current_frame_stereo_left_constructor.right_disparity_map = current_frame.right_disparity_map;
    }

    void Memory_clear_for_keyframe_stereo_left_constructor()
    {
        //> Clear and shrink keyframe edge patches
        keyframe_stereo_left_constructor.left_edge_patches.clear();
        keyframe_stereo_left_constructor.left_edge_patches.shrink_to_fit();

        //> Clear and shrink keyframe edge descriptors
        keyframe_stereo_left_constructor.left_edge_descriptors.clear();
        keyframe_stereo_left_constructor.left_edge_descriptors.shrink_to_fit();

        //> Clear and shrink keyframe matching clusters
        for (auto &cluster_list : keyframe_stereo_left_constructor.matching_edge_clusters)
        {
            cluster_list.edge_clusters.clear();
            cluster_list.edge_clusters.shrink_to_fit();
            cluster_list.refine_final_scores.clear();
            cluster_list.refine_confidences.clear();
            cluster_list.refine_validities.clear();
        }
        keyframe_stereo_left_constructor.matching_edge_clusters.clear();
        keyframe_stereo_left_constructor.matching_edge_clusters.shrink_to_fit();

        //> Clear and shrink keyframe veridical data
        keyframe_stereo_left_constructor.veridical_right_edges_indices.clear();
        keyframe_stereo_left_constructor.veridical_right_edges_indices.shrink_to_fit();
        keyframe_stereo_left_constructor.GT_locations_from_left_edges.clear();
        keyframe_stereo_left_constructor.GT_locations_from_left_edges.shrink_to_fit();
    }

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

    //> Final temporal quads by Keyframe
    std::vector<KF_Temporal_Edge_Quads> temporal_quads_by_kf;

    //> Rolling buffer of adjacent-frame veridical quads for intersecting chains (source -> ... -> dest).
    struct AdjacentQuadSnapshot
    {
        size_t kf_frame_idx = 0;
        size_t cf_frame_idx = 0;
        std::vector<final_stereo_edge_pair> kf_stereo_mates;
        std::vector<final_stereo_edge_pair> cf_stereo_mates;
        std::vector<KF_Temporal_Edge_Quads> quads;
    };

    std::deque<AdjacentQuadSnapshot> adjacent_quad_history_;

    void push_adjacent_veridical_snapshot();
    bool try_propagate_multihop_veridical_quads();
    const AdjacentQuadSnapshot *find_adjacent_snapshot(size_t kf_frame, size_t cf_frame) const;

    std::vector<Frame_Evaluation_Metrics> all_stereo_matches_metrics;
    std::vector<Frame_Evaluation_Metrics> all_temporal_matches_metrics;

    //> Pointers to the classes
    Dataset::Ptr dataset_ = nullptr;
    ThirdOrderEdgeDetectionCPU::Ptr TOED_left = nullptr;
    ThirdOrderEdgeDetectionCPU::Ptr TOED_right = nullptr;
    Stereo_Matches::Ptr stereo_matches_engine = nullptr;
    Temporal_Matches::Ptr temporal_matches_engine = nullptr;
    Utility::Ptr utility_tool = nullptr;
    MotionTracker::Ptr motion_tracker_engine = nullptr;

    //> file stream for writing timing statistics
    std::ofstream timing_statistics_file;
    std::ofstream stereo_matches_timing_statistics_file;
    std::ofstream temporal_matches_timing_statistics_file;
};

#endif
