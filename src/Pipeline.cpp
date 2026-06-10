#ifndef PIPELINE_CPP
#define PIPELINE_CPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Pipeline.h"
#include "definitions.h"

Pipeline::Pipeline(Dataset::Ptr dataset) : dataset_(dataset)
{
    //> Loading the dataset
    dataset_->load_dataset(dataset_->get_dataset_type(), left_ref_disparity_maps, right_ref_disparity_maps, left_occlusion_masks, right_occlusion_masks);

    //> Initialize the pointers to the classes
    stereo_matches_engine = std::make_shared<Stereo_Matches>();
    temporal_matches_engine = std::make_shared<Temporal_Matches>(dataset_);
    utility_tool = std::make_shared<Utility>();
    motion_tracker_engine = std::make_shared<MotionTracker>(dataset_);

    //> Open the file stream for writing timing statistics
    std::string timing_statistics_file_name = dataset_->get_output_path() + "/timing_statistics.txt";
    std::string stereo_matches_timing_statistics_file_name = dataset_->get_output_path() + "/stereo_matches_timing_statistics.txt";
    std::string temporal_matches_timing_statistics_file_name = dataset_->get_output_path() + "/temporal_matches_timing_statistics.txt";
    timing_statistics_file.open(timing_statistics_file_name);
    stereo_matches_timing_statistics_file.open(stereo_matches_timing_statistics_file_name);
    temporal_matches_timing_statistics_file.open(temporal_matches_timing_statistics_file_name);
}

bool Pipeline::Add_Stereo_Frame()
{
    do
    {
        if (b_end_pipeline_for_debug) {
            break;
        }
        switch (status_)
        {
        case PipelineStatus::STATUS_IMG_PREPARATION:
            //> Preparing images
            LOG_STATUS("IMG_PREPARATION");
            prepare_Stereo_Images();
            break;
        case PipelineStatus::STATUS_GET_STEREO_EDGE_CORRESPONDENCES:
            //> Get stereo edge correspondences
            LOG_STATUS("GET_STEREO_EDGE_CORRESPONDENCES");
            // get_Stereo_Edge_Correspondences();
            get_Stereo_Edge_Correspondences_GPU();
            break;
        case PipelineStatus::STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES:
            //> Get temporal edge correspondences (keyframe <-> current frame)
            LOG_STATUS("GET_TEMPORAL_EDGE_CORRESPONDENCES");
            // get_Temporal_Edge_Correspondences();
            get_Temporal_Edge_Correspondences_GPU();
            break;
        case PipelineStatus::STATUS_GET_POSE_FROM_QUAD_PAIRS:
            //> Get pose from quad pairs
            LOG_STATUS("GET_POSE_FROM_QUAD_PAIRS");
            get_Pose_From_Quad_Pairs();
            break;
        case PipelineStatus::STATUS_KEYFRAME_DECISION:
            //> Make keyframe decision
            LOG_STATUS("KEYFRAME_DECISION");
            make_Keyframe_Decision();
            break;
        case PipelineStatus::STATUS_SYSTEM_EXIT:
            //> System exit
            LOG_STATUS("SYSTEM_EXIT");
            break;
        }
    } while (!send_control_to_main);

    save_Current_Estimated_Pose();

    return true;
}

void Pipeline::ProcessEdges(const cv::Mat &image, std::vector<Edge> &edges, bool is_left)
{
    std::cout << "Running third-order edge detector..." << std::endl;
    if (is_left) {
        TOED_left->get_Third_Order_Edges(image);
    }
    else {
        TOED_right->get_Third_Order_Edges(image);
    }
    edges = (is_left) ? TOED_left->toed_edges : TOED_right->toed_edges;
}

void Pipeline::prepare_Stereo_Images()
{
    if (dataset_->has_gt()) {
        current_frame.left_disparity_map = (stereo_current_frame_idx < left_ref_disparity_maps.size()) ? left_ref_disparity_maps[stereo_current_frame_idx] : cv::Mat();
        current_frame.right_disparity_map = (stereo_current_frame_idx < right_ref_disparity_maps.size()) ? right_ref_disparity_maps[stereo_current_frame_idx] : cv::Mat();
    }

    std::cout << std::endl << "Stereo Image Pair #" << stereo_current_frame_idx << std::endl;

    cv::Mat left_cur_undistorted, right_cur_undistorted;
    cv::undistort(current_frame.left_image, left_cur_undistorted, dataset_->get_left_calib_matrix_cvMat(), dataset_->get_left_dist_coeff_mat());
    cv::undistort(current_frame.right_image, right_cur_undistorted, dataset_->get_right_calib_matrix_cvMat(), dataset_->get_right_dist_coeff_mat());
    current_frame.left_image_undistorted = left_cur_undistorted;
    current_frame.right_image_undistorted = right_cur_undistorted;

    util_compute_Img_Gradients(current_frame.left_image_undistorted, current_frame.left_image_gradients_x, current_frame.left_image_gradients_y);
    util_compute_Img_Gradients(current_frame.right_image_undistorted, current_frame.right_image_gradients_x, current_frame.right_image_gradients_y);

    //> initialize the pointers of the third-order edge detector and the spatial grids
    if (get_Current_Frame_Index() == 0)
        initialize_TOED_and_Spatial_Grids();

    auto start_time_left = std::chrono::high_resolution_clock::now();
    ProcessEdges(left_cur_undistorted, dataset_->left_edges, true);
    auto end_time_left = std::chrono::high_resolution_clock::now();
    auto duration_left = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_left - start_time_left);
    // std::cout << "Time taken by the third-order edge detector for the left image: " << duration_left.count() << " milliseconds" << std::endl;
    std::cout << "Number of edges on the left image: " << dataset_->left_edges.size() << std::endl;
    current_frame.left_edges = dataset_->left_edges;

    auto start_time_right = std::chrono::high_resolution_clock::now();
    ProcessEdges(right_cur_undistorted, dataset_->right_edges, false);
    auto end_time_right = std::chrono::high_resolution_clock::now();
    auto duration_right = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_right - start_time_right);
    // std::cout << "Time taken by the third-order edge detector for the right image: " << duration_right.count() << " milliseconds" << std::endl;
    std::cout << "Number of edges on the right image: " << dataset_->right_edges.size() << std::endl;
    current_frame.right_edges = dataset_->right_edges;

    timing_statistics_file << duration_left.count() + duration_right.count() << " ";

    //> Shift to the next status
    status_ = PipelineStatus::STATUS_GET_STEREO_EDGE_CORRESPONDENCES;
    send_control_to_main = false;
}

void Pipeline::get_Stereo_Edge_Correspondences_GPU() 
{
    //> Set up the flattened fundamental matrix, left and right images, and the GPU device
    prepare_for_GPU_engine = std::make_shared<Prepare_For_GPU_Pipeline>(dataset_, current_frame, DEVICE_ID);
    left_cf_img_texture_ = prepare_for_GPU_engine->left_image_texture_;
    right_cf_img_texture_ = prepare_for_GPU_engine->right_image_texture_;

    //> Reset before allocate: frees the previous frame's GPU pipeline before this frame's cudaMalloc traffic (avoids a brief 2x VRAM spike).
    stereo_matches_engine_GPU.reset();

    Stereo_Matches_GPU_Timing_Statistics stereo_GPU_times;

    //> h_F, left_image_texture_, right_image_texture_ are created and passed to the stereo matches engine
    stereo_matches_engine_GPU = std::make_shared<Stereo_Matches_GPU_Pipeline>(dataset_, prepare_for_GPU_engine, current_frame, DEVICE_ID);
    stereo_GPU_times = stereo_matches_engine_GPU->get_stereo_edge_matching_GPU(stereo_current_frame_idx);
    collect_stereo_GPU_times.push_back(stereo_GPU_times);

    //> Host export and file write (We do not need this if we do GPU stereo + GPU temporal)
    stereo_matches_engine_GPU->retrieve_stereo_mates(current_frame, current_frame_stereo_edge_mates);
    std::cout << "Number of stereo edge matches: " << current_frame_stereo_edge_mates.size() << std::endl;

    //> output the results to files
    stereo_matches_engine_GPU->write_finalized_matches_to_file(stereo_current_frame_idx);

    //> Transfer the ownership from the finished stereo GPU to Pipeline class members
    set_CF_Stereo_Matches_GPU_Pointers();

    //> Release GPU immediately; nothing else uses stereo_matches_engine_GPU until the next stereo pass.
    stereo_matches_engine_GPU.reset();

    //> If the current frame is the first frame, make current frame the keyframe
    if (get_Current_Frame_Index() == 0)
    {
        //> The pose of the first frame is the identity pose
        current_frame.estimated_camera_pose = Camera_Pose(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());

        set_Keyframe();
        increment_Current_Stereo_Frame_Index();
        status_ = PipelineStatus::STATUS_IMG_PREPARATION;
        send_control_to_main = true;
    }
    else
    {
        //> Shift to the next status
        status_ = PipelineStatus::STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES;
        send_control_to_main = false;
    }
}

void Pipeline::get_Stereo_Edge_Correspondences()
{
    //> Set the stereo left constructor
    set_Stereo_Left_Constructor();
    Timing_Statistics stereo_timing_statistics;

    //> For each left edge, get the corresponding GT location (not right edge) on the right image, and the triangulated 3D point in the left camera coordinate
    //> For dataset without GT, we still need these functions to set up `current_frame_stereo_left_constructor` but we will not use the GT locations
    stereo_matches_engine->Find_Stereo_GT_Locations(dataset_, current_frame.left_disparity_map, current_frame, current_frame_stereo_left_constructor, true);
    //> Construct a GT stereo edge pool
    stereo_matches_engine->get_Stereo_Edge_GT_Pairs(dataset_, current_frame, current_frame_stereo_left_constructor, true);

    //> construct stereo edge correspondences for the current_frame
    Frame_Evaluation_Metrics metrics = stereo_matches_engine->get_Stereo_Edge_Pairs(dataset_, current_frame_stereo_left_constructor, stereo_current_frame_idx, stereo_timing_statistics);
    //> Only accumulate evaluation metrics when GT is available
    if (dataset_->has_gt()) {
        all_stereo_matches_metrics.push_back(metrics);
    }

    //> Finalize the stereo edge pairs for the current_frame
    stereo_matches_engine->finalize_stereo_edge_mates(current_frame_stereo_left_constructor, current_frame_stereo_edge_mates);

    //> write stereo matching timings to a file
    stereo_matches_engine->write_timings_to_file(stereo_matches_timing_statistics_file, stereo_timing_statistics);
    timing_statistics_file << stereo_timing_statistics.total_time << " ";

    // stereo_matches_engine->write_finalized_stereo_edge_pairs_to_file(dataset_, current_frame_stereo_edge_mates, stereo_current_frame_idx);

    //> If the current frame is the first frame, make current frame the keyframe
    if (get_Current_Frame_Index() == 0)
    {
        //> The pose of the first frame is the identity pose
        current_frame.estimated_camera_pose = Camera_Pose(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        
        // estimated_poses.push_back(current_frame.gt_camera_pose);
        // cumulated_poses.push_back(current_frame.gt_camera_pose);
        set_Keyframe();
        increment_Current_Stereo_Frame_Index();
        status_ = PipelineStatus::STATUS_IMG_PREPARATION;
        send_control_to_main = true;
    }
    else
    {
        //> Shift to the next status
        status_ = PipelineStatus::STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES;
        send_control_to_main = false;
    }
}

void Pipeline::get_Temporal_Edge_Correspondences_GPU()
{
    std::cout << "Finding temporal edge correspondences (GPU) (" << get_Keyframe_Index() << "->" << get_Current_Frame_Index() << ")" << std::endl;

    //> Reset before allocate: avoids a brief 2x VRAM spike
    temporal_matches_engine_GPU.reset();

    Temporal_Matches_GPU_Timing_Statistics temporal_GPU_times;
    temporal_matches_engine_GPU = std::make_shared<Temporal_Matches_GPU_Pipeline>(
        dataset_, current_frame, 
        num_of_kf_stereo_matches, num_of_cf_stereo_matches,
        d_kf_stereo_matches, d_cf_stereo_matches,
        d_kf_left_patches, d_cf_left_patches,
        d_kf_right_patches, d_cf_right_patches,
        d_kf_left_sift_descriptors, d_cf_left_sift_descriptors,
        left_kf_img_texture_.texObj, right_kf_img_texture_.texObj,
        right_cf_img_texture_.texObj, left_cf_img_texture_.texObj,
        DEVICE_ID);

    //> Main temporal edge matching pipeline in GPU
    temporal_GPU_times = temporal_matches_engine_GPU->get_temporal_edge_matching_GPU();

    //> Retrieve final (kf_mate_idx, cf_mate_idx) pairs for downstream use
    std::vector<Match_by_Edge_Index> final_quad_matches;
    temporal_matches_engine_GPU->retrieve_final_matches(final_quad_matches);

    //> output the results to files
    temporal_matches_engine_GPU->write_finalized_matches_to_file(stereo_key_frame_idx, stereo_current_frame_idx);

    collect_temporal_GPU_times.push_back(temporal_GPU_times);
    temporal_matches_engine_GPU.reset();

    set_Keyframe();

    status_ = PipelineStatus::STATUS_IMG_PREPARATION;
    send_control_to_main = true;
    b_end_pipeline_for_debug = true;
}

void Pipeline::get_Temporal_Edge_Correspondences()
{
    std::cout << "Finding the termpoal edge correspondences (" << stereo_key_frame_idx << "->" << stereo_current_frame_idx << ")" << std::endl;
    temporal_quads_by_kf.clear();
    temporal_quads_by_kf.shrink_to_fit();

    //> Reset spatial grids so they contain only the current frame's edges (otherwise candidates accumulate across frames and memory explodes)
    reset_spatial_grids();
    Timing_Statistics temporal_timing_statistics;

    //> construct spatial grids for the current stereo edge mates
    temporal_matches_engine->add_edges_to_spatial_grid(current_frame_stereo_edge_mates, left_spatial_grids, right_spatial_grids);

    bool used_chained_veridical_quads = false;
#if ENABLE_QUAD_PROPAGATION
    if (dataset_->has_gt() && stereo_current_frame_idx > stereo_key_frame_idx + 1)
        used_chained_veridical_quads = try_propagate_multihop_veridical_quads();
#endif

    //> `temporal_quads_by_kf` is a struct that stores quads from KF stereo edge pairs
    //> One KF stereo edge pair could pair up with multiple veridical CF stereo edge pairs.
    //> `build_Veridical_Quads` fills veridical quads when reference disparity GT exists; otherwise it still runs and seeds one
    //> `KF_Temporal_Edge_Quads` per KF stereo mate (see Temporal_Matches::build_Veridical_Quads).
    if (!used_chained_veridical_quads)
        temporal_matches_engine->build_Veridical_Quads(temporal_quads_by_kf, keyframe_stereo_edge_mates, current_frame_stereo_edge_mates, keyframe_stereo_left_constructor, current_frame_stereo_left_constructor, left_spatial_grids, right_spatial_grids);

    //> Store adjacent quads for multi-hop propagation (propagation path still requires has_gt() today).
    if (!used_chained_veridical_quads && stereo_current_frame_idx == stereo_key_frame_idx + 1)
        push_adjacent_veridical_snapshot();

    //> Quad-centric pipeline: build veridical quads, apply filters, optionally convert to temporal pairs for backward compatibility
    Frame_Evaluation_Metrics metrics = temporal_matches_engine->get_Temporal_Edge_Pairs_from_Quads(
        temporal_quads_by_kf,
        keyframe_stereo_edge_mates,
        current_frame_stereo_edge_mates,
        left_spatial_grids, right_spatial_grids,
        keyframe_stereo_left_constructor, current_frame_stereo_left_constructor,
        keyframe, current_frame,
        stereo_key_frame_idx, stereo_current_frame_idx, temporal_timing_statistics);
    //> Only accumulate temporal evaluation metrics when GT is available
    if (dataset_->has_gt())
        all_temporal_matches_metrics.push_back(metrics);

    //> for testing
    // std::vector<Quad_for_Pose_Solution> quads_for_pose_solution = motion_tracker_engine->get_Quad_for_Pose_Solution(temporal_quads_by_kf);
    // motion_tracker_engine->save_Quad_for_Pose_Solution_to_File(*dataset_, temporal_quads_by_kf, quads_for_pose_solution, "rank_ordered_quads_for_pose_solution");

    //> write quads to a file
    // temporal_matches_engine->write_quads_to_file(temporal_quads_by_kf, stereo_key_frame_idx, stereo_current_frame_idx);

    // temporal_matches_engine->test_Constraints_from_Two_Oriented_Points( temporal_quads_by_kf, stereo_key_frame_idx, stereo_current_frame_idx);

    //> Print temporal metrics only when GT is available
    // if (dataset_->has_gt())
    //     Print_Temporal_Matches_Metrics_Statistics();

    timing_statistics_file << temporal_timing_statistics.total_time << " ";
    temporal_matches_engine->write_timings_to_file(temporal_matches_timing_statistics_file, temporal_timing_statistics);

    status_ = PipelineStatus::STATUS_GET_POSE_FROM_QUAD_PAIRS;
    send_control_to_main = false;
}

void Pipeline::get_Pose_From_Quad_Pairs()
{
    Ransac_Options opt;
    Ransac_State state;

    Camera_Pose KF_GT_pose = keyframe.gt_camera_pose;
    Camera_Pose CF_GT_pose = current_frame.gt_camera_pose;
    Camera_Pose rel_pose_GT = utility_tool->get_Relative_Pose(KF_GT_pose, CF_GT_pose);
    rel_pose_GT.print_Camera_Pose("Ground-truth relative pose from KF to CF");

    //> TEST
    std::vector<Quad_for_Pose_Solution> quads_for_pose_solution = motion_tracker_engine->get_Quad_for_Pose_Solution(temporal_quads_by_kf);
    std::cout << "Temporal quads by KF size: " << temporal_quads_by_kf.size() << std::endl;
    std::cout << "Number of quads for pose solution: " << quads_for_pose_solution.size() << std::endl;
    std::vector<size_t> inlier_indices;
    motion_tracker_engine->score_Pose_Hypothesis(rel_pose_GT, quads_for_pose_solution, temporal_quads_by_kf, opt, inlier_indices);
    std::cout << "Inlier ratio from GT relative pose: " << static_cast<double>(inlier_indices.size()) / static_cast<double>(quads_for_pose_solution.size()) << std::endl;

#if WRITE_GT_REPROJECTION_MATLAB_FILE
    {
        const std::string reproj_csv = dataset_->get_output_path() + "/gt_reproj_cf" + std::to_string(stereo_current_frame_idx) + ".txt";
        motion_tracker_engine->save_GT_reprojection_table_for_matlab(rel_pose_GT, quads_for_pose_solution, temporal_quads_by_kf, opt,
            reproj_csv, stereo_current_frame_idx);
    }
#endif

    //> Add a timer for the motion tracker
    auto start_time_motion_tracker = std::chrono::high_resolution_clock::now();

    Camera_Pose best_pose_hypothesis;
    if( motion_tracker_engine->estimate_Relative_Pose_From_Quad_Pairs(temporal_quads_by_kf, opt, state) )
    {
        best_pose_hypothesis = state.best_pose_hypothesis;
        std::cout << "Inlier ratio: " << state.inlier_ratio << std::endl;
    }
    else {
        status_ = PipelineStatus::STATUS_SYSTEM_EXIT;
        send_control_to_main = true;
        system_exit_message = "Failed to estimate relative pose from quad pairs";
        return;
    }

    auto end_time_motion_tracker = std::chrono::high_resolution_clock::now();
    auto duration_motion_tracker = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_motion_tracker - start_time_motion_tracker);
    timing_statistics_file << duration_motion_tracker.count() << "\n";

    //> print the ground-truth relative pose from KF to CF
    // Camera_Pose KF_GT_pose = keyframe.gt_camera_pose;
    // Camera_Pose CF_GT_pose = current_frame.gt_camera_pose;
    // CF_GT_pose.print_Camera_Pose("Ground-truth pose of the current frame");
    // Camera_Pose rel_pose_GT = utility_tool->get_Relative_Pose(KF_GT_pose, CF_GT_pose);
    // rel_pose_GT.print_Camera_Pose("Ground-truth relative pose from KF to CF");

    // //> Apply the relative pose to the GT poses
    // Eigen::Matrix3d applied_R = best_pose_hypothesis.R * keyframe.gt_camera_pose.R;
    // Eigen::Vector3d applied_t = best_pose_hypothesis.R * keyframe.gt_camera_pose.t + best_pose_hypothesis.t;
    // Camera_Pose applied_pose = Camera_Pose(applied_R, applied_t);
    // applied_pose.print_Camera_Pose("Applied pose of the current frame");

    // //> get the last pose from estimated_poses
    // Camera_Pose last_pose = estimated_poses.back();
    // Eigen::Matrix3d cumulative_R = best_pose_hypothesis.R * last_pose.R;
    // Eigen::Vector3d cumulative_t = best_pose_hypothesis.R * last_pose.t + best_pose_hypothesis.t;
    // Camera_Pose acc_pose = Camera_Pose(cumulative_R, cumulative_t);
    // acc_pose.print_Camera_Pose("Cumulative pose of the current frame");
    // cumulated_poses.push_back(acc_pose);

    current_frame.estimated_camera_pose = best_pose_hypothesis;
    Camera_Pose KF_abs_pose = get_KF_abs_pose();
    Camera_Pose rel_pose_estimated = utility_tool->get_Relative_Pose(KF_abs_pose, best_pose_hypothesis);
    rel_pose_estimated.print_Camera_Pose("Estimated relative pose from KF to CF");

    double rot_err = utility_tool->evaluate_Relative_Rotation_Accuracy(rel_pose_estimated, rel_pose_GT);
    double trans_err = utility_tool->evaluate_Relative_Translation_Accuracy(rel_pose_estimated, rel_pose_GT);
    std::cout << "Rotation error: " << rot_err << " degrees" << std::endl;
    std::cout << "Translation error: " << trans_err << " meters" << std::endl;
    
    //> TODO: Transform to the world coordinate




    // status_ = PipelineStatus::STATUS_KEYFRAME_DECISION;
    // send_control_to_main = false;

    //> Memory cleanup: free memory from keyframe structures that are no longer needed
    Memory_clear_for_keyframe_stereo_left_constructor();

    set_Keyframe();
    increment_Current_Stereo_Frame_Index();
    status_ = PipelineStatus::STATUS_IMG_PREPARATION;
    send_control_to_main = true; 
    b_end_pipeline_for_debug = false;   
}

void Pipeline::make_Keyframe_Decision()
{
    
}

void Pipeline::Print_Stereo_Matches_Metrics_Statistics()
{
    stereo_matches_engine->Stereo_Matches_Metrics_Statistics(all_stereo_matches_metrics);
}

void Pipeline::Print_Temporal_Matches_Metrics_Statistics()
{
    temporal_matches_engine->Temporal_Matches_Metrics_Statistics(all_temporal_matches_metrics);
}

const Pipeline::AdjacentQuadSnapshot *Pipeline::find_adjacent_snapshot(size_t kf_frame, size_t cf_frame) const
{
    for (const auto &s : adjacent_quad_history_)
    {
        if (s.kf_frame_idx == kf_frame && s.cf_frame_idx == cf_frame)
            return &s;
    }
    return nullptr;
}

void Pipeline::push_adjacent_veridical_snapshot()
{
    AdjacentQuadSnapshot snap;
    snap.kf_frame_idx = stereo_key_frame_idx;
    snap.cf_frame_idx = stereo_current_frame_idx;
    snap.kf_stereo_mates = keyframe_stereo_edge_mates;
    snap.cf_stereo_mates = current_frame_stereo_edge_mates;
    snap.quads = temporal_quads_by_kf;
    for (auto &kvq : snap.quads)
    {
        ptrdiff_t ki = kvq.KF_stereo_mate - keyframe_stereo_edge_mates.data();
        if (ki >= 0 && ki < static_cast<ptrdiff_t>(keyframe_stereo_edge_mates.size()))
            kvq.KF_stereo_mate = snap.kf_stereo_mates.data() + ki;
    }
    adjacent_quad_history_.push_back(std::move(snap));
    while (adjacent_quad_history_.size() > QUAD_PROPAGATION_HISTORY_MAX)
        adjacent_quad_history_.pop_front();
}

bool Pipeline::try_propagate_multihop_veridical_quads()
{
#if !ENABLE_QUAD_PROPAGATION
    return false;
#else
    if (!dataset_->has_gt())
        return false;
    const size_t gap = stereo_current_frame_idx - stereo_key_frame_idx;
    if (gap <= 1)
        return false;

    std::vector<const AdjacentQuadSnapshot *> chain;
    chain.reserve(gap);
    for (size_t f = stereo_key_frame_idx; f < stereo_current_frame_idx; ++f)
    {
        const AdjacentQuadSnapshot *s = find_adjacent_snapshot(f, f + 1);
        if (!s)
            return false;
        chain.push_back(s);
    }

    std::vector<KF_Temporal_Edge_Quads> acc = chain[0]->quads;

    for (size_t i = 1; i < chain.size(); ++i)
    {
        std::vector<KF_Temporal_Edge_Quads> next_out;
        Temporal_Matches::propagate_veridical_quads_one_hop(
            acc,
            chain[0]->kf_stereo_mates,
            chain[i]->kf_stereo_mates,
            chain[i]->quads,
            chain[i]->cf_stereo_mates,
            next_out);
        acc = std::move(next_out);
    }

    if (acc.empty())
        return false;

    temporal_quads_by_kf = std::move(acc);

    for (auto &kvq : temporal_quads_by_kf)
    {
        if (!kvq.KF_stereo_mate)
            continue;
        ptrdiff_t ki = kvq.KF_stereo_mate - chain[0]->kf_stereo_mates.data();
        if (ki >= 0 && ki < static_cast<ptrdiff_t>(keyframe_stereo_edge_mates.size()))
            kvq.KF_stereo_mate = keyframe_stereo_edge_mates.data() + ki;
    }

    size_t total_v = 0;
    for (const auto &k : temporal_quads_by_kf)
        total_v += k.veridical_quads.size();
    std::cout << "Quad propagation (" << stereo_key_frame_idx << "->" << stereo_current_frame_idx << "): "
              << temporal_quads_by_kf.size() << " KF groups, " << total_v << " chained veridical quads" << std::endl;

    return true;
#endif
}

//> Destructor
Pipeline::~Pipeline()
{
    //> Texture memory is owned/released by Prepare_For_GPU_Pipeline instances.
    left_kf_img_texture_ = {nullptr, 0};
    right_kf_img_texture_ = {nullptr, 0};
    left_cf_img_texture_ = {nullptr, 0};
    right_cf_img_texture_ = {nullptr, 0};
    
    //> free GPU memory
    auto* released_kf_stereo_matches = d_kf_stereo_matches;
    if (released_kf_stereo_matches != nullptr) {
        cudacheck_noexcept( cudaFree(d_kf_stereo_matches) );
        d_kf_stereo_matches = nullptr;
    }
    if (d_cf_stereo_matches != nullptr && d_cf_stereo_matches != released_kf_stereo_matches) {
        cudacheck_noexcept( cudaFree(d_cf_stereo_matches) );
        d_cf_stereo_matches = nullptr;
    }
    d_cf_stereo_matches = nullptr;

    auto* released_kf_left_patches = d_kf_left_patches;
    if (released_kf_left_patches != nullptr) {
        cudacheck_noexcept( cudaFree(d_kf_left_patches) );
        d_kf_left_patches = nullptr;
    }
    if (d_cf_left_patches != nullptr && d_cf_left_patches != released_kf_left_patches) {
        cudacheck_noexcept( cudaFree(d_cf_left_patches) );
        d_cf_left_patches = nullptr;
    }
    d_cf_left_patches = nullptr;

    auto* released_kf_right_patches = d_kf_right_patches;
    if (released_kf_right_patches != nullptr) {
        cudacheck_noexcept( cudaFree(d_kf_right_patches) );
        d_kf_right_patches = nullptr;
    }
    if (d_cf_right_patches != nullptr && d_cf_right_patches != released_kf_right_patches) {
        cudacheck_noexcept( cudaFree(d_cf_right_patches) );
        d_cf_right_patches = nullptr;
    }
    d_cf_right_patches = nullptr;

    auto* released_kf_left_sift_descriptors = d_kf_left_sift_descriptors;
    if (released_kf_left_sift_descriptors != nullptr) {
        cudacheck_noexcept( cudaFree(d_kf_left_sift_descriptors) );
        d_kf_left_sift_descriptors = nullptr;
    }
    if (d_cf_left_sift_descriptors != nullptr && d_cf_left_sift_descriptors != released_kf_left_sift_descriptors) {
        cudacheck_noexcept( cudaFree(d_cf_left_sift_descriptors) );
        d_cf_left_sift_descriptors = nullptr;
    }
    d_cf_left_sift_descriptors = nullptr;

    timing_statistics_file.close();
    stereo_matches_timing_statistics_file.close();
    temporal_matches_timing_statistics_file.close();
}



#endif