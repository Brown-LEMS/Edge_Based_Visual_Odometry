#ifndef PIPELINE_CPP
#define PIPELINE_CPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Pipeline.h"
#include "definitions.h"

Pipeline::Pipeline(Dataset::Ptr dataset) : dataset_(dataset) {

    //> Loading the dataset
    dataset_->load_dataset(dataset_->get_dataset_type(), left_ref_disparity_maps, right_ref_disparity_maps, left_occlusion_masks, right_occlusion_masks);

    //> Initialize the pointers to the classes
    stereo_matches_engine = std::make_shared<Stereo_Matches>();
    temporal_matches_engine = std::make_shared<Temporal_Matches>(dataset_);
    utility_tool = std::make_shared<Utility>();
    Camera_Motion_Estimate = std::make_shared<MotionTracker>();
}

void Pipeline::ProcessEdges(const cv::Mat &image, std::vector<Edge> &edges)
{
    std::cout << "Running third-order edge detector..." << std::endl;
    TOED->get_Third_Order_Edges(image);
    edges = TOED->toed_edges;
}

bool Pipeline::Add_Stereo_Frame() {

    do {
        switch (status_) {
            case PipelineStatus::STATUS_IMG_PREPARATION:
                //> Preparing images
                LOG_STATUS("IMG_PREPARATION");
                prepare_Stereo_Images();
                break;
            case PipelineStatus::STATUS_GET_STEREO_EDGE_CORRESPONDENCES:
                //> Get stereo edge correspondences
                LOG_STATUS("GET_STEREO_EDGE_CORRESPONDENCES");
                get_Stereo_Edge_Correspondences();
                break;
            case PipelineStatus::STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES:
                //> Get temporal edge correspondences (keyframe <-> current frame)
                LOG_STATUS("GET_TEMPORAL_EDGE_CORRESPONDENCES");
                get_Temporal_Edge_Correspondences();
                break;
        }
    } while ( !send_control_to_main );

    return true;
}

void Pipeline::prepare_Stereo_Images() {

    current_frame.left_disparity_map = (stereo_frame_idx < left_ref_disparity_maps.size()) ? left_ref_disparity_maps[stereo_frame_idx] : cv::Mat();
    current_frame.right_disparity_map = (stereo_frame_idx < right_ref_disparity_maps.size()) ? right_ref_disparity_maps[stereo_frame_idx] : cv::Mat();

    // //> This is optional
    // const cv::Mat &left_occlusion_mask = (stereo_frame_idx < left_occlusion_masks.size()) ? left_occlusion_masks[stereo_frame_idx] : cv::Mat();
    // const cv::Mat &right_occlusion_mask = (stereo_frame_idx < right_occlusion_masks.size()) ? right_occlusion_masks[stereo_frame_idx] : cv::Mat();

    std::cout << std::endl << "Stereo Image Pair #" << stereo_frame_idx << std::endl;

    cv::Mat left_cur_undistorted, right_cur_undistorted;
    cv::undistort(current_frame.left_image, left_cur_undistorted, dataset_->get_left_calib_matrix_cvMat(), dataset_->get_left_dist_coeff_mat());
    cv::undistort(current_frame.right_image, right_cur_undistorted, dataset_->get_right_calib_matrix_cvMat(), dataset_->get_right_dist_coeff_mat());
    current_frame.left_image_undistorted = left_cur_undistorted;
    current_frame.right_image_undistorted = right_cur_undistorted;

    util_compute_Img_Gradients(current_frame.left_image_undistorted, current_frame.left_image_gradients_x, current_frame.left_image_gradients_y);
    util_compute_Img_Gradients(current_frame.right_image_undistorted, current_frame.right_image_gradients_x, current_frame.right_image_gradients_y);

    //> initialize the pointers of the third-order edge detector and the spatial grids
    if (dataset_->get_num_imgs() == 0)
        initialize_TOED_and_Spatial_Grids();

    ProcessEdges(left_cur_undistorted, dataset_->left_edges);
    std::cout << "Number of edges on the left image: " << dataset_->left_edges.size() << std::endl;
    current_frame.left_edges = dataset_->left_edges;

    ProcessEdges(right_cur_undistorted, dataset_->right_edges);
    std::cout << "Number of edges on the right image: " << dataset_->right_edges.size() << std::endl;
    current_frame.right_edges = dataset_->right_edges;

    dataset_->increment_num_imgs();
    std::cout << std::endl;

    //> Shift to the next status
    status_ = PipelineStatus::STATUS_GET_STEREO_EDGE_CORRESPONDENCES;
    send_control_to_main = false;
}

void Pipeline::get_Stereo_Edge_Correspondences() {

    //> Set the stereo left constructor
    set_Stereo_Left_Constructor();

    //> For each left edge, get the corresponding GT location (not right edge) on the right image, and the triangulated 3D point in the left camera coordinate
    stereo_matches_engine->Find_Stereo_GT_Locations(dataset_, current_frame.left_disparity_map, current_frame, current_frame_stereo_left_constructor, true);
    std::cout << "Complete calculating GT locations for left edges of the current_frame..." << std::endl;
    //> Construct a GT stereo edge pool
    stereo_matches_engine->get_Stereo_Edge_GT_Pairs(dataset_, current_frame, current_frame_stereo_left_constructor, true);
    std::cout << "Size of stereo edge correspondences pool = " << current_frame_stereo_left_constructor.focused_edge_indices.size() << std::endl;
    
    //> construct stereo edge correspondences for the current_frame
    Frame_Evaluation_Metrics metrics = stereo_matches_engine->get_Stereo_Edge_Pairs(dataset_, current_frame_stereo_left_constructor, stereo_frame_idx);

    //> Finalize the stereo edge pairs for the current_frame
    stereo_matches_engine->finalize_stereo_edge_mates(current_frame_stereo_left_constructor, current_frame_stereo_edge_mates);

    //> If the current frame is the first frame, make current frame the keyframe
    if (stereo_frame_idx == 0) {
        set_Keyframe();
        status_ = PipelineStatus::STATUS_IMG_PREPARATION;
        send_control_to_main = true;
    }
    else {
        //> Shift to the next status
        status_ = PipelineStatus::STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES;
        send_control_to_main = false;
    }
}

void Pipeline::get_Temporal_Edge_Correspondences() {

    //> construct spatial grids for the current stereo edge mates
    temporal_matches_engine->add_edges_to_spatial_grid(current_frame_stereo_edge_mates, left_spatial_grids, right_spatial_grids);
    
    //> Construct correspondences structure between last keyframe and the current frame
    temporal_matches_engine->Find_Veridical_Edge_Correspondences_on_CF(
        left_temporal_edge_mates,
        keyframe_stereo_edge_mates,
        current_frame_stereo_edge_mates,
        keyframe_stereo_left_constructor, current_frame_stereo_left_constructor,
        left_spatial_grids, true, 1.0);
    std::cout << "Size of veridical edge pairs (left) = " << left_temporal_edge_mates.size() << std::endl;
    temporal_matches_engine->Find_Veridical_Edge_Correspondences_on_CF(
        right_temporal_edge_mates,
        keyframe_stereo_edge_mates,
        current_frame_stereo_edge_mates,
        keyframe_stereo_left_constructor, current_frame_stereo_left_constructor,
        right_spatial_grids, false, 1.0);
    std::cout << "Size of veridical edge pairs (right) = " << right_temporal_edge_mates.size() << std::endl;

    std::vector<KF_Veridical_Quads> veridical_quads_by_kf = temporal_matches_engine->find_Veridical_Quads(left_temporal_edge_mates, right_temporal_edge_mates, current_frame_stereo_edge_mates);
    size_t num_veridical_quads = 0;
    for (const auto &kvq : veridical_quads_by_kf)
        num_veridical_quads += kvq.veridical_CF_stereo_mates.size();
    std::cout << "Veridical quads: " << veridical_quads_by_kf.size() << " KF stereo correspondences, " << num_veridical_quads << " total quads" << std::endl;

    temporal_matches_engine->get_Temporal_Edge_Pairs(
        current_frame_stereo_edge_mates,
        left_temporal_edge_mates, right_temporal_edge_mates,
        left_spatial_grids, right_spatial_grids,
        keyframe, current_frame,
        stereo_frame_idx);

    send_control_to_main = true;
}

#endif