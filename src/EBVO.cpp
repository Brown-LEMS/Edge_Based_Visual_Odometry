#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <numeric>
#include <cmath>
#include <chrono>
#include <random>
#include <omp.h>
#include "EBVO.h"
#include "EdgeClusterer.h"
#include "definitions.h"
#include <opencv2/core/eigen.hpp>

EBVO::EBVO(YAML::Node config_map) : dataset(config_map) {}

//> MARK: MAIN CODE OF EDGE VO
void EBVO::PerformEdgeBasedVO()
{
    // The main processing function that performs edge-based visual odometry on a sequence of stereo image pairs.
    // It loads images, detects edges, matches them, and evaluates the matching performance.
    int num_pairs = 100000;
    std::vector<cv::Mat> left_ref_disparity_maps;
    std::vector<cv::Mat> right_ref_disparity_maps;
    std::vector<cv::Mat> left_occlusion_masks, right_occlusion_masks;

    //> load stereo iterator and read disparity
    dataset.load_dataset(dataset.get_dataset_type(), left_ref_disparity_maps, right_ref_disparity_maps, left_occlusion_masks, right_occlusion_masks, num_pairs);

    std::vector<double> left_intr = dataset.left_intr();
    std::vector<double> right_intr = dataset.right_intr();

    cv::Mat left_calib = (cv::Mat_<double>(3, 3) << left_intr[0], 0, left_intr[2], 0, left_intr[1], left_intr[3], 0, 0, 1);
    cv::Mat right_calib = (cv::Mat_<double>(3, 3) << right_intr[0], 0, right_intr[2], 0, right_intr[1], right_intr[3], 0, 0, 1);

    cv::Mat left_dist_coeff_mat = (cv::Mat_<double>(1, 4) << dataset.left_dist_coeffs()[0], dataset.left_dist_coeffs()[1], dataset.left_dist_coeffs()[2], dataset.left_dist_coeffs()[3]);
    cv::Mat right_dist_coeff_mat = (cv::Mat_<double>(1, 4) << dataset.right_dist_coeffs()[0], dataset.right_dist_coeffs()[1], dataset.right_dist_coeffs()[2], dataset.right_dist_coeffs()[3]);

    LOG_INFO("Start looping over all image pairs");
    // now we change the logic, we will do previous frame and current frame instead
    StereoFrame last_keyframe, current_frame;
    //> Initialize
    Stereo_Edge_Pairs last_keyframe_stereo_left, current_frame_stereo_left;

    //> Engine for constructing stereo edge correspondences
    Stereo_Matches stereo_edge_matcher;

    //> Final 1-1 stereo edge pairs for keyframe and current frame
    std::vector<final_stereo_edge_pair> keyframe_stereo_edge_mates;
    std::vector<final_stereo_edge_pair> current_frame_stereo_edge_mates;

    //> Keyframe <-> current frame edge pairs
    std::vector<temporal_edge_pair> left_temporal_edge_mates;
    std::vector<temporal_edge_pair> right_temporal_edge_mates;

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    std::string out_file = dataset.get_output_path() + "/edge_numbers.txt";
    std::string temporal_matches_file = dataset.get_output_path() + "/temporal_matches.txt";

    //> Check for checkpoint recovery: read last processed frame from temporal_matches.txt
    size_t start_frame_idx = 0;
    size_t resume_keyframe_idx = 0;
    bool resume_from_checkpoint = false;

    std::ifstream checkpoint_check(temporal_matches_file);
    if (checkpoint_check.is_open())
    {
        std::string last_line, line;
        while (std::getline(checkpoint_check, line))
        {
            if (!line.empty())
                last_line = line;
        }
        checkpoint_check.close();

        if (!last_line.empty())
        {
            //> Parse line format: "(KF:31, CF:32), left temporal matches: 6207, right temporal matches: 6207"
            size_t kf_pos = last_line.find("KF:");
            size_t cf_pos = last_line.find("CF:");
            if (kf_pos != std::string::npos && cf_pos != std::string::npos)
            {
                size_t kf_end = last_line.find(",", kf_pos);
                size_t cf_end = last_line.find(")", cf_pos);

                std::string kf_str = last_line.substr(kf_pos + 3, kf_end - kf_pos - 3);
                std::string cf_str = last_line.substr(cf_pos + 3, cf_end - cf_pos - 3);

                resume_keyframe_idx = std::stoull(kf_str);
                size_t last_cf_idx = std::stoull(cf_str);

                //> Resume from the last CF as the new keyframe
                start_frame_idx = last_cf_idx;
                resume_from_checkpoint = true;

                std::cout << "\n========================================" << std::endl;
                std::cout << "CHECKPOINT RECOVERY ENABLED" << std::endl;
                std::cout << "Last processed: KF:" << resume_keyframe_idx << " -> CF:" << last_cf_idx << std::endl;
                std::cout << "Resuming from frame " << start_frame_idx << " as new keyframe" << std::endl;
                std::cout << "========================================\n"
                          << std::endl;
            }
        }
    }

    //> Open output files (append mode if resuming)
    std::ofstream record_file(out_file, resume_from_checkpoint ? std::ios::app : std::ios::out);
    std::ofstream temporal_record_file(temporal_matches_file, resume_from_checkpoint ? std::ios::app : std::ios::out);

    bool b_is_keyframe = true;
    std::vector<Frame_Evaluation_Metrics> all_metrics;
    size_t frame_idx = 0;
    size_t keyframe_idx = resume_from_checkpoint ? start_frame_idx : 0;
    while (dataset.stereo_iterator->hasNext() && num_pairs - frame_idx >= 0)
    {
        if (!dataset.stereo_iterator->getNext(current_frame))
        {
            std::cout << "No more image pairs to process" << std::endl;
            break;
        }

        //> Skip frames until we reach the checkpoint resume point
        if (frame_idx < start_frame_idx)
        {
            frame_idx++;
            continue;
        }

        const cv::Mat &left_disparity_map = (frame_idx < left_ref_disparity_maps.size()) ? left_ref_disparity_maps[frame_idx] : cv::Mat();
        const cv::Mat &right_disparity_map = (frame_idx < right_ref_disparity_maps.size()) ? right_ref_disparity_maps[frame_idx] : cv::Mat();

        //> FOr now, this is optional
        const cv::Mat &left_occlusion_mask = (frame_idx < left_occlusion_masks.size()) ? left_occlusion_masks[frame_idx] : cv::Mat();
        const cv::Mat &right_occlusion_mask = (frame_idx < right_occlusion_masks.size()) ? right_occlusion_masks[frame_idx] : cv::Mat();

        std::cout << std::endl
                  << "Image Pair #" << frame_idx << std::endl;

        cv::Mat left_cur_undistorted, right_cur_undistorted;

        //> Parallelize image undistortion and gradient computation
#pragma omp parallel sections
        {
#pragma omp section
            {
                cv::undistort(current_frame.left_image, left_cur_undistorted, left_calib, left_dist_coeff_mat);
                current_frame.left_image_undistorted = left_cur_undistorted;
                util_compute_Img_Gradients(current_frame.left_image_undistorted, current_frame.left_image_gradients_x, current_frame.left_image_gradients_y);
            }
#pragma omp section
            {
                cv::undistort(current_frame.right_image, right_cur_undistorted, right_calib, right_dist_coeff_mat);
                current_frame.right_image_undistorted = right_cur_undistorted;
                util_compute_Img_Gradients(current_frame.right_image_undistorted, current_frame.right_image_gradients_x, current_frame.right_image_gradients_y);
            }
        }

        if (dataset.get_num_imgs() == 0)
        {
            dataset.set_height(left_cur_undistorted.rows);
            dataset.set_width(left_cur_undistorted.cols);

            TOED = std::shared_ptr<ThirdOrderEdgeDetectionCPU>(new ThirdOrderEdgeDetectionCPU(dataset.get_height(), dataset.get_width()));

            // Initialize the spatial grids with a cell size of defined GRID_SIZE
            left_spatial_grids = SpatialGrid(dataset.get_width(), dataset.get_height(), GRID_SIZE);
            right_spatial_grids = SpatialGrid(dataset.get_width(), dataset.get_height(), GRID_SIZE);
        }

        ProcessEdges(left_cur_undistorted, TOED, dataset.left_edges);
        std::cout << "Number of edges on the left image: " << dataset.left_edges.size() << std::endl;
        current_frame.left_edges = dataset.left_edges;

        ProcessEdges(right_cur_undistorted, TOED, dataset.right_edges);
        std::cout << "Number of edges on the right image: " << dataset.right_edges.size() << std::endl;
        current_frame.right_edges = dataset.right_edges;
        dataset.increment_num_imgs();
        std::cout << std::endl;

        if (b_is_keyframe)
        {
            //> Update last keyframe
            last_keyframe = current_frame;
            last_keyframe_stereo_left.clean_up_vector_data_structures();
            last_keyframe_stereo_left.stereo_frame = &last_keyframe;
            last_keyframe_stereo_left.left_disparity_map = left_disparity_map;
            last_keyframe_stereo_left.right_disparity_map = right_disparity_map;
            record_file << "Frame:" << frame_idx << ", left edge numbers: " << current_frame.left_edges.size() << " right edge numbers: " << current_frame.right_edges.size() << std::endl;
            //> For each left edge, get the corresponding GT location (not right edge) on the right image, and the triangulated 3D point in the left camera coordinate
            stereo_edge_matcher.Find_Stereo_GT_Locations(dataset, left_disparity_map, last_keyframe, last_keyframe_stereo_left, true);
            std::cout << "Complete calculating GT locations for left edges of the keyframe (previous frame)..." << std::endl;
            //> Construct a GT stereo edge pool
            stereo_edge_matcher.get_Stereo_Edge_GT_Pairs(dataset, last_keyframe, last_keyframe_stereo_left, true);
            std::cout << "Size of stereo edge correspondences pool = " << last_keyframe_stereo_left.focused_edge_indices.size() << std::endl;
            last_keyframe_stereo_left.construct_toed_left_id_to_Stereo_Edge_Pairs_left_id_map();
            Construct_final_stereo_edge_pairs_with_stereo(last_keyframe_stereo_left, keyframe_stereo_edge_mates);
            std::cout << "Finish constructing final stereo edge pairs for the keyframe with " << keyframe_stereo_edge_mates.size() << " pairs" << std::endl;
            b_is_keyframe = false;
        }
        else
        {
            Construct_final_stereo_edge_pairs(current_frame, current_frame_stereo_edge_mates);
            std::cout << "Finish constructing final stereo edge pairs for the current frame with " << current_frame_stereo_edge_mates.size() << " pairs" << std::endl;
            //> Assign edges to spatial grids
            add_edges_to_spatial_grid(current_frame_stereo_edge_mates, left_spatial_grids, right_spatial_grids);
            std::cout << "Finish adding left and right edges of the current frame to spatial grid with cell size " << GRID_SIZE << std::endl;

            //> Set current frame stereo pointer so Find_Veridical can access gt_rotation/gt_translation
            current_frame_stereo_left.stereo_frame = &current_frame;

            Find_Veridical_Edge_Correspondences_on_CF(
                left_temporal_edge_mates,
                keyframe_stereo_edge_mates,
                current_frame_stereo_edge_mates,
                last_keyframe_stereo_left, current_frame_stereo_left,
                left_spatial_grids, true, 1.0);
            std::cout << "Size of veridical edge pairs (left) = " << left_temporal_edge_mates.size() << std::endl;

            //> Now that the GT edge correspondences are constructed between the keyframe and the current frame, we can apply various filters from the beginning
            //> Stage 1: Apply spatial grid to the current frame
            apply_spatial_grid_filtering(left_temporal_edge_mates, current_frame_stereo_edge_mates, left_spatial_grids, 30.0, true);
            Evaluate_KF_CF_Edge_Correspondences(left_temporal_edge_mates, frame_idx, "Limited Disparity", "Left");

            //> Stage 2: Do orientation filtering (parallelized left/right)

            apply_orientation_filtering(left_temporal_edge_mates, current_frame_stereo_edge_mates, 10.0, true);
            Evaluate_KF_CF_Edge_Correspondences(left_temporal_edge_mates, frame_idx, "Orientation Filtering", "Left");

            apply_NCC_filtering(left_temporal_edge_mates, current_frame_stereo_edge_mates, 0.8, last_keyframe.left_image, current_frame.left_image, true);
            Evaluate_KF_CF_Edge_Correspondences(left_temporal_edge_mates, frame_idx, "NCC Filtering", "Left");

            apply_SIFT_filtering(left_temporal_edge_mates, current_frame_stereo_edge_mates, 200.0, true);
            Evaluate_KF_CF_Edge_Correspondences(left_temporal_edge_mates, frame_idx, "SIFT Filtering", "Left");

            apply_best_nearly_best_filtering(left_temporal_edge_mates, 0.9, "NCC");
            Evaluate_KF_CF_Edge_Correspondences(left_temporal_edge_mates, frame_idx, "BNB NCC Filtering", "Left");

            apply_best_nearly_best_filtering(left_temporal_edge_mates, 0.8, "SIFT");
            Evaluate_KF_CF_Edge_Correspondences(left_temporal_edge_mates, frame_idx, "BNB SIFT Filtering", "Left");

            apply_photometric_refinement(left_temporal_edge_mates, current_frame_stereo_edge_mates, last_keyframe, current_frame, true);
            Evaluate_KF_CF_Edge_Correspondences(left_temporal_edge_mates, frame_idx, "Photometric Refinement", "Left");

            apply_temporal_edge_clustering(left_temporal_edge_mates, true);
            Evaluate_KF_CF_Edge_Correspondences(left_temporal_edge_mates, frame_idx, "Temporal Edge Clustering", "Left");

            apply_best_nearly_best_filtering(left_temporal_edge_mates, 0.8, "SIFT");
            Evaluate_KF_CF_Edge_Correspondences(left_temporal_edge_mates, frame_idx, "BNB SIFT Filtering", "Left");

            cleaning_temporal_edge_mates(left_temporal_edge_mates);
            Evaluate_KF_CF_Edge_Correspondences(left_temporal_edge_mates, frame_idx, "Cleaning Temporal Edge Mates", "Left");

            //> Record temporal match statistics (parallelized counting)
            size_t left_surviving_clusters = 0;
            size_t right_surviving_clusters = 0;

#pragma omp parallel for reduction(+ : left_surviving_clusters)
            for (int i = 0; i < static_cast<int>(left_temporal_edge_mates.size()); ++i)
            {
                if (!left_temporal_edge_mates[i].matching_CF_edge_clusters.empty())
                {
                    left_surviving_clusters++;
                }
            }

#pragma omp parallel for reduction(+ : right_surviving_clusters)
            for (int i = 0; i < static_cast<int>(right_temporal_edge_mates.size()); ++i)
            {
                if (!right_temporal_edge_mates[i].matching_CF_edge_clusters.empty())
                {
                    right_surviving_clusters++;
                }
            }

            temporal_record_file << "(KF:" << keyframe_idx << ", CF:" << frame_idx << "), "
                                 << "left temporal matches: " << left_surviving_clusters << ", "
                                 << "right temporal matches: " << right_surviving_clusters << std::endl;

            std::cout << "\n=== Temporal Match Summary ===" << std::endl;
            std::cout << "KF: " << keyframe_idx << " -> CF: " << frame_idx << std::endl;
            std::cout << "Left temporal matches (surviving clusters): " << left_surviving_clusters << std::endl;
            std::cout << "Right temporal matches (surviving clusters): " << right_surviving_clusters << std::endl;
            std::cout << "==============================\n"
                      << std::endl;

            //> Static KF experiment: keep keyframe (frame 0) fixed, only clean up CF data

            //> Clean up current frame stereo structures
            current_frame_stereo_left.clean_up_vector_data_structures();
            current_frame_stereo_left.stereo_frame = nullptr;
            current_frame_stereo_left.left_disparity_map = cv::Mat();
            current_frame_stereo_left.right_disparity_map = cv::Mat();

            //> Clear current frame stereo edge mates
            current_frame_stereo_edge_mates.clear();
            current_frame_stereo_edge_mates.shrink_to_fit();

            //> Clear temporal matches for next iteration
            left_temporal_edge_mates.clear();
            left_temporal_edge_mates.shrink_to_fit();
            right_temporal_edge_mates.clear();
            right_temporal_edge_mates.shrink_to_fit();

            //> Clear spatial grids for reuse
            left_spatial_grids.reset();
            right_spatial_grids.reset();

            //> keyframe_idx stays fixed (frame 0 remains KF)
        }

        frame_idx++;
    }

    if (!all_metrics.empty())
    {
        // Collect all stage names across all frames
        std::set<std::string> all_stage_names;
        for (const auto &frame_metrics : all_metrics)
        {
            for (const auto &[stage_name, metrics] : frame_metrics.stages)
            {
                all_stage_names.insert(stage_name);
            }
        }

        // Compute averages for each stage
        for (const auto &stage_name : all_stage_names)
        {
            double avg_recall = 0.0, avg_precision = 0.0, avg_ambiguity = 0.0;
            int count = 0;

            for (const auto &frame_metrics : all_metrics)
            {
                auto it = frame_metrics.stages.find(stage_name);
                if (it != frame_metrics.stages.end())
                {
                    avg_recall += it->second.recall;
                    avg_precision += it->second.precision;
                    avg_ambiguity += it->second.ambiguity;
                    count++;
                }
            }

            if (count > 0)
            {
                avg_recall /= count;
                avg_precision /= count;
                avg_ambiguity /= count;

                std::cout << "Stage: " << stage_name << std::endl;
                std::cout << "  - Average Recall:     " << std::fixed << std::setprecision(8) << avg_recall << std::endl;
                std::cout << "  - Average Precision:  " << std::fixed << std::setprecision(8) << avg_precision << std::endl;
                std::cout << "  - Average Ambiguity:  " << std::fixed << std::setprecision(8) << avg_ambiguity << std::endl;
                std::cout << std::endl;
            }
        }
    }
    record_file.close();
    temporal_record_file.close();
}

void EBVO::Construct_final_stereo_edge_pairs_with_stereo(Stereo_Edge_Pairs &stereo_edge_pairs, std::vector<final_stereo_edge_pair> &stereo_edge_mates)
{
    // Only works for left edges for purpose of experiment
    // also, for KF frames which we are constucting, we also dont have half of the structures ;).
    stereo_edge_mates.clear();
    cv::Mat left_image = stereo_edge_pairs.stereo_frame->left_image_undistorted;
    if (left_image.type() != CV_64F)
        left_image.convertTo(left_image, CV_64F);
    const size_t num_edges = stereo_edge_pairs.focused_edge_indices.size();
    if (num_edges == 0)
        return;

    // Pre-size the output vector
    stereo_edge_mates.resize(num_edges);

#pragma omp parallel
    {

        Utility thread_util;
        cv::Ptr<cv::SIFT> thread_sift = cv::SIFT::create();

#pragma omp for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(num_edges); ++i)
        {
            const Edge &left_edge = dataset.left_edges[stereo_edge_pairs.focused_edge_indices[i]];
            final_stereo_edge_pair mate;
            mate.left_edge = left_edge;
            mate.left_edge_patches = thread_util.get_edge_patches(left_edge, left_image);
            // We also have Gamma now
            mate.Gamma_in_left_cam_coord = stereo_edge_pairs.Gamma_in_left_cam_coord[i];
            // Extract SIFT descriptors at orthogonally shifted points
            std::pair<cv::Point2d, cv::Point2d> shifted_points = thread_util.get_Orthogonal_Shifted_Points(left_edge, 8);
            std::vector<cv::KeyPoint> keypoints;
            keypoints.reserve(2);
            keypoints.emplace_back(cv::KeyPoint(shifted_points.first, 1.0f, static_cast<float>(180.0 / M_PI * left_edge.orientation)));
            keypoints.emplace_back(cv::KeyPoint(shifted_points.second, 1.0f, static_cast<float>(180.0 / M_PI * left_edge.orientation)));

            cv::Mat descriptors;
            thread_sift->compute(stereo_edge_pairs.stereo_frame->left_image_undistorted, keypoints, descriptors);
            if (descriptors.rows == 2)
            {
                mate.left_edge_descriptors = std::make_pair(descriptors.row(0).clone(), descriptors.row(1).clone());
            }
            else
            {
                mate.left_edge_descriptors = std::make_pair(cv::Mat(), cv::Mat());
            }
            stereo_edge_mates[i] = mate;
        }
    }
}
void EBVO::Construct_final_stereo_edge_pairs(const StereoFrame &frame, std::vector<final_stereo_edge_pair> &stereo_edge_mates)
{
    // Only works for left edges for purpose of experiment
    // also, for CF frames which we are constucting, we dont have half of the structures.
    stereo_edge_mates.clear();
    cv::Mat left_image = frame.left_image_undistorted;
    if (left_image.type() != CV_64F)
        left_image.convertTo(left_image, CV_64F);
    const size_t num_edges = dataset.left_edges.size();
    if (num_edges == 0)
        return;

    // Pre-size the output vector
    stereo_edge_mates.resize(num_edges);

#pragma omp parallel
    {

        Utility thread_util;
        cv::Ptr<cv::SIFT> thread_sift = cv::SIFT::create();

#pragma omp for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(num_edges); ++i)
        {
            const Edge &left_edge = dataset.left_edges[i];
            final_stereo_edge_pair mate;
            mate.left_edge = left_edge;
            mate.left_edge_patches = thread_util.get_edge_patches(left_edge, left_image);

            // Extract SIFT descriptors at orthogonally shifted points
            std::pair<cv::Point2d, cv::Point2d> shifted_points = thread_util.get_Orthogonal_Shifted_Points(left_edge, 8);
            std::vector<cv::KeyPoint> keypoints;
            keypoints.reserve(2);
            keypoints.emplace_back(cv::KeyPoint(shifted_points.first, 1.0f, static_cast<float>(180.0 / M_PI * left_edge.orientation)));
            keypoints.emplace_back(cv::KeyPoint(shifted_points.second, 1.0f, static_cast<float>(180.0 / M_PI * left_edge.orientation)));

            cv::Mat descriptors;
            thread_sift->compute(frame.left_image_undistorted, keypoints, descriptors);
            if (descriptors.rows == 2)
            {
                mate.left_edge_descriptors = std::make_pair(descriptors.row(0).clone(), descriptors.row(1).clone());
            }
            else
            {
                mate.left_edge_descriptors = std::make_pair(cv::Mat(), cv::Mat());
            }
            stereo_edge_mates[i] = mate;
        }
    }
}

void EBVO::add_edges_to_spatial_grid(const std::vector<final_stereo_edge_pair> &stereo_edge_mates, SpatialGrid &left_spatial_grids, SpatialGrid &right_spatial_grids)
{
    left_spatial_grids.reset();

    std::vector<std::pair<int, int>> left_edge_to_grid(stereo_edge_mates.size());

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(stereo_edge_mates.size()); ++i)
    {
        const cv::Point2d &left_loc = stereo_edge_mates[i].left_edge.location;
        int lx = static_cast<int>(left_loc.x) / left_spatial_grids.cell_size;
        int ly = static_cast<int>(left_loc.y) / left_spatial_grids.cell_size;
        if (lx >= 0 && lx < left_spatial_grids.grid_width && ly >= 0 && ly < left_spatial_grids.grid_height)
            left_edge_to_grid[i] = {i, ly * left_spatial_grids.grid_width + lx};
        else
            left_edge_to_grid[i] = {i, -1};
    }

    for (size_t i = 0; i < stereo_edge_mates.size(); ++i)
    {
        int left_grid_idx = left_edge_to_grid[i].second;
        if (left_grid_idx >= 0 && left_grid_idx < static_cast<int>(left_spatial_grids.grid.size()))
            left_spatial_grids.grid[left_grid_idx].push_back(i);
    }
}

void EBVO::Find_Veridical_Edge_Correspondences_on_CF(
    std::vector<temporal_edge_pair> &temporal_edge_mates,
    const std::vector<final_stereo_edge_pair> &KF_stereo_edge_mates,
    const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
    Stereo_Edge_Pairs &last_keyframe_stereo, Stereo_Edge_Pairs &current_frame_stereo,
    SpatialGrid &spatial_grid, bool b_is_left, double gt_dist_threshold)
{
    temporal_edge_mates.clear();

    Eigen::Matrix3d R_left_ego = current_frame_stereo.stereo_frame->gt_rotation * last_keyframe_stereo.stereo_frame->gt_rotation.transpose();
    Eigen::Vector3d t_left_ego = current_frame_stereo.stereo_frame->gt_translation - R_left_ego * last_keyframe_stereo.stereo_frame->gt_translation;

    const double orientation_threshold = 30.0;                   // degrees
    const double search_radius = 15.0 + gt_dist_threshold + 3.0; // +3 for safety margin
    const int img_margin = 10;

    int num_threads_corr = omp_get_max_threads();
    std::vector<std::vector<temporal_edge_pair>> thread_temporal_edge_mates(num_threads_corr);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, 128)
        for (int i = 0; i < static_cast<int>(KF_stereo_edge_mates.size()); ++i)
        {
            const final_stereo_edge_pair *kf_mate = &KF_stereo_edge_mates[i];

            Eigen::Vector3d Gamma_in_left_KF = kf_mate->Gamma_in_left_cam_coord;
            Eigen::Vector3d Gamma_in_left_CF = R_left_ego * Gamma_in_left_KF + t_left_ego;
            Eigen::Vector3d projected_point;
            if (b_is_left)
            {
                projected_point = dataset.get_left_calib_matrix() * Gamma_in_left_CF;
            }
            else
            {
                Eigen::Vector3d Gamma_in_right_CF = dataset.get_relative_rot_left_to_right() * Gamma_in_left_CF + dataset.get_relative_transl_left_to_right();
                projected_point = dataset.get_right_calib_matrix() * Gamma_in_right_CF;
            }
            projected_point /= projected_point.z();

            double projected_orientation = orientation_mapping(
                kf_mate->left_edge,
                kf_mate->right_edge,
                projected_point, b_is_left, *last_keyframe_stereo.stereo_frame, *current_frame_stereo.stereo_frame, dataset);

            cv::Point2d projected_point_cv(projected_point.x(), projected_point.y());
            if (projected_point.x() <= img_margin || projected_point.y() <= img_margin ||
                projected_point.x() >= dataset.get_width() - img_margin || projected_point.y() >= dataset.get_height() - img_margin)
                continue;

            std::vector<int> current_candidate_edge_indices = spatial_grid.getCandidatesWithinRadius(projected_point_cv, search_radius);
            std::vector<int> veridical_CF_stereo_edge_mate_indices;
            std::vector<Temporal_CF_Edge_Cluster> clusters;
            clusters.reserve(current_candidate_edge_indices.size());

            for (const int curr_e_index : current_candidate_edge_indices)
            {
                if (curr_e_index < 0 || curr_e_index >= static_cast<int>(CF_stereo_edge_mates.size()))
                    continue;

                const Edge &cf_edge = b_is_left ? CF_stereo_edge_mates[curr_e_index].left_edge : CF_stereo_edge_mates[curr_e_index].right_edge;
                double dist = cv::norm(cf_edge.location - projected_point_cv);
                double orientation_diff = std::abs(rad_to_deg<double>(projected_orientation - cf_edge.orientation));
                if (orientation_diff > 180.0)
                    orientation_diff = 360.0 - orientation_diff;

                scores score;
                score.ncc_score = -1.0;
                score.sift_score = 900.0;

                Temporal_CF_Edge_Cluster cluster;
                cluster.cf_stereo_edge_mate_index = curr_e_index;
                cluster.contributing_cf_stereo_indices = {curr_e_index};
                cluster.center_edge = cf_edge;
                cluster.matching_scores = score;
                clusters.push_back(std::move(cluster));

                if ((dist < gt_dist_threshold) && (orientation_diff < orientation_threshold ||
                                                   std::abs(orientation_diff - 180.0) < orientation_threshold))
                {
                    veridical_CF_stereo_edge_mate_indices.push_back(curr_e_index);
                }
            }

            //> Only include this KF mate in temporal_edge_mates if it has at least one veridical CF match
            if (!veridical_CF_stereo_edge_mate_indices.empty())
            {
                temporal_edge_pair tp;
                tp.KF_stereo_edge_mate = kf_mate;
                tp.projected_point = projected_point;
                tp.projected_orientation = projected_orientation;
                tp.veridical_CF_stereo_edge_mate_indices = std::move(veridical_CF_stereo_edge_mate_indices);
                tp.matching_CF_edge_clusters = std::move(clusters);
                thread_temporal_edge_mates[tid].push_back(std::move(tp));
            }
        }
    }

    //> Merge thread-local temporal edge mates into temporal_edge_mates
    for (int tid = 0; tid < num_threads_corr; ++tid)
    {
        temporal_edge_mates.insert(temporal_edge_mates.end(),
                                   std::make_move_iterator(thread_temporal_edge_mates[tid].begin()),
                                   std::make_move_iterator(thread_temporal_edge_mates[tid].end()));
    }
}

void EBVO::ProcessEdges(const cv::Mat &image,
                        std::shared_ptr<ThirdOrderEdgeDetectionCPU> &toed,
                        std::vector<Edge> &edges)
{
    std::cout << "Running third-order edge detector..." << std::endl;
    toed->get_Third_Order_Edges(image);
    edges = toed->toed_edges;
}

double EBVO::orientation_mapping(const Edge &e_left, const Edge &e_right, const Eigen::Vector3d projected_point, bool is_left_cam, const StereoFrame &last_keyframe, const StereoFrame &current_frame, Dataset &dataset)
{
    // Step 1: Get the stereo baseline rotation (Left -> Right)
    Eigen::Matrix3d R_stereo = dataset.get_relative_rot_left_to_right();

    // Step 2: Reconstruct 3D direction T_1 in Left KF
    Eigen::Vector3d t1(cos(e_left.orientation), sin(e_left.orientation), 0);
    Eigen::Vector3d t2(cos(e_right.orientation), sin(e_right.orientation), 0);

    Eigen::Vector3d gamma_1(e_left.location.x, e_left.location.y, 1.0);
    gamma_1 = dataset.get_left_calib_matrix().inverse() * gamma_1;
    Eigen::Vector3d gamma_2(e_right.location.x, e_right.location.y, 1.0);
    gamma_2 = dataset.get_right_calib_matrix().inverse() * gamma_2;

    Eigen::Vector3d T_1 = -(gamma_2.dot(t2.cross(R_stereo * t1))) * gamma_1 + (gamma_2.dot(t2.cross(R_stereo * gamma_1))) * t1;
    T_1 = -T_1;
    T_1.normalize();

    // Step 3: Transform T_1 to current frame
    Eigen::Matrix3d R_temporal = current_frame.gt_rotation * last_keyframe.gt_rotation.transpose(); // Left KF -> Left CF

    Eigen::Vector3d T_2;
    if (is_left_cam)
    {
        T_2 = R_temporal * T_1; // Left KF -> Left CF
    }
    else
    {
        T_2 = R_stereo * R_temporal * T_1; // Left KF -> Left CF -> Right CF
    }

    // Step 4: Project T_2 to image
    Eigen::Vector3d gamma_cf = projected_point / projected_point.z();
    if (is_left_cam)
        gamma_cf = dataset.get_left_calib_matrix().inverse() * gamma_cf;
    else
        gamma_cf = dataset.get_right_calib_matrix().inverse() * gamma_cf;

    Eigen::Vector3d t = T_2 - T_2.z() * gamma_cf;
    t.normalize();

    return atan2(t.y(), t.x());
}

void EBVO::record_Temporal_Ambiguity_Distribution(const std::string &stage_name,
                                                  const std::vector<temporal_edge_pair> &temporal_edge_mates,
                                                  const std::string &output_dir,
                                                  size_t frame_idx,
                                                  bool b_is_left)
{
    if (output_dir.empty())
        return;

    std::string side = b_is_left ? "left" : "right";
    // Create subdirectory: output_files/ambiguity/{stage_name}/
    std::string ambig_dir = output_dir + "/ambiguity/temporal_" + stage_name + "_" + side;
    std::filesystem::create_directories(ambig_dir);

    // Create file in the subdirectory
    std::string filename = ambig_dir + "/frame_" + std::to_string(frame_idx) + ".txt";

    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write header
    file << "# Stage: temporal_" << stage_name << "_" << side << "\n";
    file << "# Frame: " << frame_idx << "\n";
    file << "edge_index\tnum_candidates\tis_gt_present\n";

    // Write data
    for (size_t i = 0; i < temporal_edge_mates.size(); ++i)
    {
        int num_candidates = temporal_edge_mates[i].matching_CF_edge_clusters.size();
        // Check if any cluster is veridical (within GT threshold)
        int is_gt_present = 0;
        cv::Point2d gt_location(temporal_edge_mates[i].projected_point.x(), temporal_edge_mates[i].projected_point.y());
        for (const auto &cluster : temporal_edge_mates[i].matching_CF_edge_clusters)
        {
            if (cv::norm(cluster.center_edge.location - gt_location) < DIST_TO_GT_THRESH_TEMP)
            {
                is_gt_present = 1;
                break;
            }
        }
        file << i << "\t" << num_candidates << "\t" << is_gt_present << "\n";
    }

    file.close();
    std::cout << "Recorded temporal ambiguity (" << side << ") for " << temporal_edge_mates.size()
              << " edges to: " << filename << std::endl;
}

void EBVO::apply_spatial_grid_filtering(std::vector<temporal_edge_pair> &temporal_edge_mates, const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, SpatialGrid &spatial_grid, double grid_radius, bool b_is_left)
{
    //> For each temporal edge pair (KF mate + projected point on CF), find candidate CF stereo edge mate indices using the spatial grid

    //> Combined storage: ALL veridical edges + sampled KF edges' ALL CF edges
    std::vector<double> all_location_errors;
    std::vector<int> all_location_labels;
    int num_random_edges_for_distribution = 10;
    //> Select random KF edges to record ALL CF edges (before any filtering)
    std::vector<int> sampled_kf_indices;
    if (num_random_edges_for_distribution > 0 && temporal_edge_mates.size() > 0)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, temporal_edge_mates.size() - 1);

        std::unordered_set<int> unique_indices;
        while (unique_indices.size() < std::min(num_random_edges_for_distribution,
                                                (int)temporal_edge_mates.size()))
        {
            unique_indices.insert(dis(gen));
        }
        sampled_kf_indices.assign(unique_indices.begin(), unique_indices.end());
        std::sort(sampled_kf_indices.begin(), sampled_kf_indices.end());
    }

#pragma omp parallel
    {
        //> Thread-local storage for combined data
        std::vector<double> thread_location_errors;
        std::vector<int> thread_location_labels;

#pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(temporal_edge_mates.size()); ++i)
        {
            const final_stereo_edge_pair *kf_mate = temporal_edge_mates[i].KF_stereo_edge_mate;
            cv::Point2d query_location = b_is_left ? kf_mate->left_edge.location : kf_mate->right_edge.location;
            cv::Point2d gt_location(temporal_edge_mates[i].projected_point.x(), temporal_edge_mates[i].projected_point.y());

            //> Check if this KF edge is in the sampled set
            bool should_record_all_cf_edges = std::find(sampled_kf_indices.begin(),
                                                        sampled_kf_indices.end(), i) != sampled_kf_indices.end();

            //> For sampled KF edges: record location error to ALL CF edges (before any filtering)
            if (should_record_all_cf_edges)
            {
                for (size_t cf_idx = 0; cf_idx < CF_stereo_edge_mates.size(); ++cf_idx)
                {
                    const Edge &cf_edge = b_is_left ? CF_stereo_edge_mates[cf_idx].left_edge : CF_stereo_edge_mates[cf_idx].right_edge;
                    double location_error = cv::norm(cf_edge.location - gt_location);
                    double location_difference = cv::norm(cf_edge.location - query_location);
                    bool is_veridical = location_error < DIST_TO_GT_THRESH_TEMP;

                    thread_location_errors.push_back(location_difference);
                    thread_location_labels.push_back(is_veridical ? 1 : 0);
                }
            }

            //> Now apply spatial grid filter to get candidate edges for processing
            std::vector<int> candidates = spatial_grid.getCandidatesWithinRadius(query_location, grid_radius);
            std::vector<Temporal_CF_Edge_Cluster> clusters;
            clusters.reserve(candidates.size());

            //> For NON-sampled KF edges: only record veridical edges that passed the filter
            if (!should_record_all_cf_edges)
            {
                for (int cf_stereo_idx : candidates)
                {
                    const Edge &cf_edge = b_is_left ? CF_stereo_edge_mates[cf_stereo_idx].left_edge : CF_stereo_edge_mates[cf_stereo_idx].right_edge;
                    double location_error = cv::norm(cf_edge.location - gt_location);
                    double location_difference = cv::norm(cf_edge.location - query_location);
                    bool is_veridical = location_error < DIST_TO_GT_THRESH_TEMP;

                    //> Only record veridical edges from non-sampled KF edges
                    if (is_veridical)
                    {
                        thread_location_errors.push_back(location_difference);
                        thread_location_labels.push_back(1);
                    }
                }
            }

            //> Create clusters from filtered candidates (unchanged)
            for (int cf_stereo_idx : candidates)
            {
                Temporal_CF_Edge_Cluster cluster;
                cluster.cf_stereo_edge_mate_index = cf_stereo_idx;
                cluster.contributing_cf_stereo_indices = {cf_stereo_idx};
                cluster.center_edge = b_is_left ? CF_stereo_edge_mates[cf_stereo_idx].left_edge : CF_stereo_edge_mates[cf_stereo_idx].right_edge;
                cluster.matching_scores = scores{-1.0, 900.0};
                clusters.push_back(std::move(cluster));
            }
            temporal_edge_mates[i].matching_CF_edge_clusters = std::move(clusters);
        }

#pragma omp critical
        {
            all_location_errors.insert(all_location_errors.end(),
                                       thread_location_errors.begin(),
                                       thread_location_errors.end());
            all_location_labels.insert(all_location_labels.end(),
                                       thread_location_labels.begin(),
                                       thread_location_labels.end());
        }
    }
#if RECORD_FILTER_DISTRIBUTIONS
    //> Record combined distribution: ALL veridical (from all KF edges) + ALL CF edges (from sampled KF edges)
    std::string output_dir = dataset.get_output_path();
    size_t frame_idx = 0; // TODO: Pass this as parameter
    if (!output_dir.empty() && !all_location_errors.empty())
    {
        std::string side = b_is_left ? "left" : "right";
        Stereo_Matches stereo_matcher;
        stereo_matcher.record_Filter_Distribution("temporal_location_error_" + side, all_location_errors, all_location_labels, output_dir, frame_idx);

        int veridical_count = std::count(all_location_labels.begin(), all_location_labels.end(), 1);
        int non_veridical_count = all_location_errors.size() - veridical_count;

        std::cout << "Recorded temporal location error distribution (" << side << "):" << std::endl;
        std::cout << "  - Veridical: " << veridical_count << " edges (from all KF edges)" << std::endl;
        std::cout << "  - Non-veridical: " << non_veridical_count << " edges (from "
                  << sampled_kf_indices.size() << " sampled KF edges × ~"
                  << CF_stereo_edge_mates.size() << " CF edges)" << std::endl;

        record_Temporal_Ambiguity_Distribution("spatial_grid", temporal_edge_mates, output_dir, frame_idx, b_is_left);
    }
#endif
}

void EBVO::apply_SIFT_filtering(std::vector<temporal_edge_pair> &temporal_edge_mates,
                                const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
                                double sift_dist_threshold, bool b_is_left)
{
    //> Thread-local storage for SIFT distances
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<double>> thread_local_sift_distances(num_threads);
    std::vector<std::vector<int>> thread_local_is_veridical(num_threads);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(temporal_edge_mates.size()); ++i)
        {
            temporal_edge_pair &tp = temporal_edge_mates[i];
            const final_stereo_edge_pair *kf_mate = tp.KF_stereo_edge_mate;
            std::pair<cv::Mat, cv::Mat> kf_edge_descriptors = (b_is_left) ? kf_mate->left_edge_descriptors : kf_mate->right_edge_descriptors;

            std::vector<Temporal_CF_Edge_Cluster> filtered_clusters;
            filtered_clusters.reserve(tp.matching_CF_edge_clusters.size());

            for (Temporal_CF_Edge_Cluster &cluster : tp.matching_CF_edge_clusters)
            {
                int cf_stereo_mate_idx = cluster.cf_stereo_edge_mate_index;
                if (cf_stereo_mate_idx < 0 || cf_stereo_mate_idx >= static_cast<int>(CF_stereo_edge_mates.size()))
                    continue;

                std::pair<cv::Mat, cv::Mat> cf_edge_descriptors = (b_is_left) ? CF_stereo_edge_mates[cf_stereo_mate_idx].left_edge_descriptors : CF_stereo_edge_mates[cf_stereo_mate_idx].right_edge_descriptors;

                if (cf_edge_descriptors.first.empty() || cf_edge_descriptors.second.empty())
                {
                    filtered_clusters.push_back(std::move(cluster));
                }
                else
                {
                    double sift_dist_1 = cv::norm(kf_edge_descriptors.first, cf_edge_descriptors.first, cv::NORM_L2);
                    double sift_dist_2 = cv::norm(kf_edge_descriptors.first, cf_edge_descriptors.second, cv::NORM_L2);
                    double sift_dist_3 = cv::norm(kf_edge_descriptors.second, cf_edge_descriptors.first, cv::NORM_L2);
                    double sift_dist_4 = cv::norm(kf_edge_descriptors.second, cf_edge_descriptors.second, cv::NORM_L2);
                    double min_sift_dist = std::min({sift_dist_1, sift_dist_2, sift_dist_3, sift_dist_4});

                    //> Record SIFT distance
                    thread_local_sift_distances[tid].push_back(min_sift_dist);

                    //> Check if veridical
                    cv::Point2d gt_location(tp.projected_point.x(), tp.projected_point.y());
                    const Edge &cf_edge = b_is_left ? CF_stereo_edge_mates[cf_stereo_mate_idx].left_edge : CF_stereo_edge_mates[cf_stereo_mate_idx].right_edge;
                    bool is_gt = cv::norm(cf_edge.location - gt_location) < DIST_TO_GT_THRESH_TEMP;
                    thread_local_is_veridical[tid].push_back(is_gt ? 1 : 0);

                    if (min_sift_dist < sift_dist_threshold)
                    {
                        cluster.matching_scores.sift_score = min_sift_dist;
                        filtered_clusters.push_back(std::move(cluster));
                    }
                }
            }
            tp.matching_CF_edge_clusters = std::move(filtered_clusters);
        }
    }

    //> Merge thread-local data
    std::vector<double> sift_distances;
    std::vector<int> is_veridical;
    for (int tid = 0; tid < num_threads; ++tid)
    {
        sift_distances.insert(sift_distances.end(), thread_local_sift_distances[tid].begin(), thread_local_sift_distances[tid].end());
        is_veridical.insert(is_veridical.end(), thread_local_is_veridical[tid].begin(), thread_local_is_veridical[tid].end());
    }

    std::string output_dir = dataset.get_output_path();
    size_t frame_idx = 0; // TODO: Pass this as parameter
#if RECORD_FILTER_DISTRIBUTIONS
    if (!output_dir.empty() && !sift_distances.empty())
    {
        std::string side = b_is_left ? "left" : "right";
        Stereo_Matches stereo_matcher;
        stereo_matcher.record_Filter_Distribution("temporal_sift_distance_" + side, sift_distances, is_veridical, output_dir, frame_idx);
        record_Temporal_Ambiguity_Distribution("sift", temporal_edge_mates, output_dir, frame_idx, b_is_left);
    }
#endif
}

// void EBVO::apply_best_nearly_best_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, double threshold, bool is_NCC)
// {
//     std::string test_name = is_NCC ? "BNB_NCC" : "BNB_SIFT";

//     // Get the correct edge vector based on is_left flag
//     const std::vector<Edge> &kf_edges = KF_CF_edge_pairs.is_left ? kf_edges_left : kf_edges_right;

// #pragma omp parallel
//     {
// #pragma omp for schedule(dynamic)
//         for (int i = 0; i < static_cast<int>(KF_CF_edge_pairs.kf_edges.size()); ++i)
//         {

//             int edge_idx = KF_CF_edge_pairs.kf_edges[i];
//             // Bounds check to prevent segfault
//             if (edge_idx < 0 || edge_idx >= static_cast<int>(kf_edges.size()))
//             {
//                 continue;
//             }

//             Edge left_edge = kf_edges[edge_idx];
//             int stereo_idx = KF_CF_edge_pairs.key_frame_pairs->get_Stereo_Edge_Pairs_left_id_index(KF_CF_edge_pairs.kf_edges[i]);
//             auto &m_ind = KF_CF_edge_pairs.matching_cf_edges_indices[i];
//             size_t num_clusters = m_ind.size();

//             if (num_clusters < 2)
//                 continue;

//             // 1. Create an index map to sort clusters based on score without losing original indices
//             // Assuming higher score is better (NCC). If SIFT (lower better), flip the comparison logic.
//             std::vector<size_t> indices(num_clusters);
//             std::iota(indices.begin(), indices.end(), 0);
//             if (is_NCC)
//                 std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
//                           { return KF_CF_edge_pairs.matching_scores[i][a].ncc_score > KF_CF_edge_pairs.matching_scores[i][b].ncc_score; });
//             else
//                 std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
//                           { return KF_CF_edge_pairs.matching_scores[i][a].sift_score < KF_CF_edge_pairs.matching_scores[i][b].sift_score; });
//             // 2. Determine how many candidates pass the recursive ratio test
//             size_t keep_count = 1; // Always keep the best

//             double best_score = is_NCC ? KF_CF_edge_pairs.matching_scores[i][indices[0]].ncc_score : KF_CF_edge_pairs.matching_scores[i][indices[0]].sift_score;

//             for (size_t j = 0; j < num_clusters - 1; ++j)
//             {
//                 double next_score = is_NCC ? KF_CF_edge_pairs.matching_scores[i][indices[j + 1]].ncc_score : KF_CF_edge_pairs.matching_scores[i][indices[j + 1]].sift_score;
//                 if (best_score == 0)
//                     break;
//                 // Ratio: next_best / current_best
//                 // If the ratio is high (e.g., 0.9), they are "nearly best."
//                 // If the ratio is low (e.g., 0.4), the next one is significantly worse.

//                 double ratio = is_NCC ? next_score / best_score : best_score / next_score;
//                 if (ratio >= threshold)
//                 {
//                     keep_count++;
//                 }
//                 else
//                 {
//                     break; // Significant drop detected, stop including further matches
//                 }
//             }

//             // 3. If we aren't keeping everything, rebuild the vectors
//             if (keep_count < num_clusters)
//             {

//                 std::vector<int> surviving_cf_matches;
//                 std::vector<scores> surviving_scores;

//                 for (size_t k = 0; k < keep_count; ++k)
//                 {
//                     size_t idx = indices[k];
//                     surviving_cf_matches.push_back(KF_CF_edge_pairs.matching_cf_edges_indices[i][idx]);
//                     surviving_scores.push_back(KF_CF_edge_pairs.matching_scores[i][idx]);
//                 }
//                 KF_CF_edge_pairs.matching_cf_edges_indices[i] = std::move(surviving_cf_matches);
//                 KF_CF_edge_pairs.matching_scores[i] = std::move(surviving_scores);
//             }
//         }
//     }
// }

void EBVO::apply_best_nearly_best_filtering(std::vector<temporal_edge_pair> &temporal_edge_mates, double threshold, const std::string scoring_type)
{
    bool is_NCC = scoring_type == "NCC";
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(temporal_edge_mates.size()); ++i)
    {
        temporal_edge_pair &tp = temporal_edge_mates[i];
        size_t num_of_clusters = tp.matching_CF_edge_clusters.size();

        if (num_of_clusters < 2)
            continue;

        std::vector<size_t> indices(num_of_clusters);
        std::iota(indices.begin(), indices.end(), 0);
        if (is_NCC)
        {
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
                      { return tp.matching_CF_edge_clusters[a].matching_scores.ncc_score > tp.matching_CF_edge_clusters[b].matching_scores.ncc_score; });
        }
        else
        {
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
                      { return tp.matching_CF_edge_clusters[a].matching_scores.sift_score < tp.matching_CF_edge_clusters[b].matching_scores.sift_score; });
        }

        size_t keep_count = 1;
        double best_score = is_NCC ? tp.matching_CF_edge_clusters[indices[0]].matching_scores.ncc_score : tp.matching_CF_edge_clusters[indices[0]].matching_scores.sift_score;
        for (size_t j = 0; j < num_of_clusters - 1; ++j)
        {
            double next_score = is_NCC ? tp.matching_CF_edge_clusters[indices[j + 1]].matching_scores.ncc_score : tp.matching_CF_edge_clusters[indices[j + 1]].matching_scores.sift_score;
            if (best_score == 0)
                break;
            double ratio = is_NCC ? next_score / best_score : best_score / next_score;
            if (ratio >= threshold)
                keep_count++;
            else
                break;
        }

        if (keep_count < num_of_clusters)
        {
            std::vector<Temporal_CF_Edge_Cluster> surviving_clusters;
            surviving_clusters.reserve(keep_count);
            for (size_t k = 0; k < keep_count; ++k)
            {
                surviving_clusters.push_back(std::move(tp.matching_CF_edge_clusters[indices[k]]));
            }
            tp.matching_CF_edge_clusters = std::move(surviving_clusters);
        }
    }
}

void EBVO::apply_photometric_refinement(std::vector<temporal_edge_pair> &temporal_edge_mates,
                                        const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
                                        const StereoFrame &keyframe, const StereoFrame &current_frame,
                                        bool b_is_left)
{
    cv::Mat kf_img_32f, cf_img_32f;
    const cv::Mat &kf_img = b_is_left ? keyframe.left_image_undistorted : keyframe.right_image_undistorted;
    const cv::Mat &cf_img = b_is_left ? current_frame.left_image_undistorted : current_frame.right_image_undistorted;
    kf_img.convertTo(kf_img_32f, CV_32F);
    cf_img.convertTo(cf_img_32f, CV_32F);

    const cv::Mat &cf_grad_x = b_is_left ? current_frame.left_image_gradients_x : current_frame.right_image_gradients_x;
    const cv::Mat &cf_grad_y = b_is_left ? current_frame.left_image_gradients_y : current_frame.right_image_gradients_y;

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < static_cast<int>(temporal_edge_mates.size()); ++i)
    {
        temporal_edge_pair &tp = temporal_edge_mates[i];
        const final_stereo_edge_pair *kf_mate = tp.KF_stereo_edge_mate;

        Edge kf_edge = b_is_left ? kf_mate->left_edge : kf_mate->right_edge;

        for (Temporal_CF_Edge_Cluster &cluster : tp.matching_CF_edge_clusters)
        {
            int cf_idx = cluster.cf_stereo_edge_mate_index;
            if (cf_idx < 0 || cf_idx >= static_cast<int>(CF_stereo_edge_mates.size()))
            {
                cluster.refine_final_score = 1e6;
                cluster.refine_validity = false;
                continue;
            }

            const Edge &cf_edge = b_is_left ? CF_stereo_edge_mates[cf_idx].left_edge : CF_stereo_edge_mates[cf_idx].right_edge;
            Eigen::Vector2d init_disp(kf_edge.location.x - cf_edge.location.x, kf_edge.location.y - cf_edge.location.y);
            Eigen::Vector2d refined_disp;
            double refined_score;
            bool refined_validity;
            std::vector<double> residual_log;
            min_Edge_Photometric_Residual_by_Gauss_Newton(
                kf_edge, cf_edge, init_disp, kf_img_32f, cf_img_32f, cf_grad_x, cf_grad_y,
                refined_disp, refined_score, refined_validity, residual_log,
                20, 1e-3, 3.0, false);

            cluster.refine_final_score = refined_score;
            cluster.refine_validity = refined_validity;
            if (refined_validity)
            {
                cluster.center_edge.location.x = kf_edge.location.x - refined_disp[0];
                cluster.center_edge.location.y = kf_edge.location.y - refined_disp[1];
            }
        }
    }
}

void EBVO::apply_temporal_edge_clustering(std::vector<temporal_edge_pair> &temporal_edge_mates, bool b_cluster_by_orientation)
{
    //> Cluster nearby CF edge candidates per temporal pair, mirroring consolidate_redundant_edge_hypothesis in stereo
    for (temporal_edge_pair &tp : temporal_edge_mates)
    {
        if (tp.matching_CF_edge_clusters.size() < 2)
            continue;

        std::vector<Edge> shifted_edges;
        std::vector<int> cf_indices;
        std::vector<double> refine_scores;
        shifted_edges.reserve(tp.matching_CF_edge_clusters.size());
        cf_indices.reserve(tp.matching_CF_edge_clusters.size());
        refine_scores.reserve(tp.matching_CF_edge_clusters.size());

        for (const Temporal_CF_Edge_Cluster &cluster : tp.matching_CF_edge_clusters)
        {
            shifted_edges.push_back(cluster.center_edge);
            cf_indices.push_back(cluster.cf_stereo_edge_mate_index);
            refine_scores.push_back(cluster.refine_final_score);
        }

        EdgeClusterer edge_cluster_engine(shifted_edges, cf_indices, b_cluster_by_orientation);
        edge_cluster_engine.setRefineScores(refine_scores);
        edge_cluster_engine.performClustering();

        //> Map EdgeCluster output back to Temporal_CF_Edge_Cluster structure; pick primary cf_index from merged cluster (best refine score)
        std::vector<Temporal_CF_Edge_Cluster> new_clusters;
        new_clusters.reserve(edge_cluster_engine.returned_clusters.size());

        for (const EdgeCluster &ec : edge_cluster_engine.returned_clusters)
        {
            Temporal_CF_Edge_Cluster tc;
            tc.center_edge = ec.center_edge;
            tc.contributing_edges = ec.contributing_edges;

            //> Collect all cf_indices that contributed to this merged cluster
            std::unordered_set<int> merged_cf_indices;
            int best_orig_idx = -1;
            double best_score = 1e10;
            for (const Edge &contrib : ec.contributing_edges)
            {
                for (size_t i = 0; i < shifted_edges.size(); ++i)
                {
                    if (cv::norm(contrib.location - shifted_edges[i].location) < 1e-3)
                    {
                        merged_cf_indices.insert(cf_indices[i]);
                        if (refine_scores[i] < best_score)
                        {
                            best_score = refine_scores[i];
                            best_orig_idx = static_cast<int>(i);
                        }
                        break;
                    }
                }
            }
            tc.contributing_cf_stereo_indices.assign(merged_cf_indices.begin(), merged_cf_indices.end());
            if (best_orig_idx >= 0)
            {
                const Temporal_CF_Edge_Cluster &orig = tp.matching_CF_edge_clusters[best_orig_idx];
                tc.cf_stereo_edge_mate_index = orig.cf_stereo_edge_mate_index;
                tc.matching_scores = orig.matching_scores;
                tc.refine_final_score = orig.refine_final_score;
                tc.refine_validity = orig.refine_validity;
            }
            else
            {
                tc.cf_stereo_edge_mate_index = tc.contributing_cf_stereo_indices.empty() ? -1 : tc.contributing_cf_stereo_indices[0];
            }
            new_clusters.push_back(std::move(tc));
        }
        tp.matching_CF_edge_clusters = std::move(new_clusters);
    }
}

void EBVO::apply_length_constraint(std::vector<temporal_edge_pair> &left_temporal_edge_mates,
                                   std::vector<temporal_edge_pair> &right_temporal_edge_mates,
                                   const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates)
{
}
void EBVO::apply_mate_consistency_filtering(std::vector<temporal_edge_pair> &left_temporal_edge_mates,
                                            std::vector<temporal_edge_pair> &right_temporal_edge_mates)
{
    //> For each left temporal pair, its stereo mate (right) must have the same CF candidate in the right temporal pair.
    //> Build KF_stereo_edge_mate -> right temporal pair index map
    std::unordered_map<const final_stereo_edge_pair *, size_t> kf_mate_to_right_idx;
    for (size_t i = 0; i < right_temporal_edge_mates.size(); ++i)
        kf_mate_to_right_idx[right_temporal_edge_mates[i].KF_stereo_edge_mate] = i;

    //> Build KF_stereo_edge_mate -> left temporal pair index map
    std::unordered_map<const final_stereo_edge_pair *, size_t> kf_mate_to_left_idx;
    for (size_t i = 0; i < left_temporal_edge_mates.size(); ++i)
        kf_mate_to_left_idx[left_temporal_edge_mates[i].KF_stereo_edge_mate] = i;

    //> For each KF mate that appears in both left and right, keep only the intersection of candidates (by cf_stereo_edge_mate_index)
    for (const auto &kv : kf_mate_to_left_idx)
    {
        const final_stereo_edge_pair *kf_mate = kv.first;
        size_t left_idx = kv.second;
        auto it_right = kf_mate_to_right_idx.find(kf_mate);
        if (it_right == kf_mate_to_right_idx.end())
        {
            left_temporal_edge_mates[left_idx].matching_CF_edge_clusters.clear();
            continue;
        }
        size_t right_idx = it_right->second;

        temporal_edge_pair &left_tp = left_temporal_edge_mates[left_idx];
        temporal_edge_pair &right_tp = right_temporal_edge_mates[right_idx];

        std::unordered_set<int> right_cf_indices;
        for (const Temporal_CF_Edge_Cluster &c : right_tp.matching_CF_edge_clusters)
        {
            if (!c.contributing_cf_stereo_indices.empty())
                // if it has contributing indices, add all of them; otherwise add the primary mate index
                for (int idx : c.contributing_cf_stereo_indices)
                    right_cf_indices.insert(idx);
            else
                right_cf_indices.insert(c.cf_stereo_edge_mate_index);
        }

        std::vector<Temporal_CF_Edge_Cluster> left_filtered;
        left_filtered.reserve(left_tp.matching_CF_edge_clusters.size());
        for (Temporal_CF_Edge_Cluster &cluster : left_tp.matching_CF_edge_clusters)
        {
            bool has_overlap = false;
            const auto &left_indices = cluster.contributing_cf_stereo_indices.empty()
                                           ? std::vector<int>{cluster.cf_stereo_edge_mate_index}
                                           : cluster.contributing_cf_stereo_indices;
            for (int idx : left_indices)
                if (right_cf_indices.count(idx))
                {
                    has_overlap = true;
                    break;
                }
            if (has_overlap)
                left_filtered.push_back(std::move(cluster));
        }
        left_tp.matching_CF_edge_clusters = std::move(left_filtered);

        std::unordered_set<int> left_cf_indices;
        for (const Temporal_CF_Edge_Cluster &c : left_tp.matching_CF_edge_clusters)
        {
            if (!c.contributing_cf_stereo_indices.empty())
                for (int idx : c.contributing_cf_stereo_indices)
                    left_cf_indices.insert(idx);
            else
                left_cf_indices.insert(c.cf_stereo_edge_mate_index);
        }

        std::vector<Temporal_CF_Edge_Cluster> right_filtered;
        right_filtered.reserve(right_tp.matching_CF_edge_clusters.size());
        for (Temporal_CF_Edge_Cluster &cluster : right_tp.matching_CF_edge_clusters)
        {
            bool has_overlap = false;
            const auto &right_indices = cluster.contributing_cf_stereo_indices.empty()
                                            ? std::vector<int>{cluster.cf_stereo_edge_mate_index}
                                            : cluster.contributing_cf_stereo_indices;
            for (int idx : right_indices)
                if (left_cf_indices.count(idx))
                {
                    has_overlap = true;
                    break;
                }
            if (has_overlap)
                right_filtered.push_back(std::move(cluster));
        }
        right_tp.matching_CF_edge_clusters = std::move(right_filtered);
    }

    for (const auto &kv : kf_mate_to_right_idx)
    {
        if (kf_mate_to_left_idx.count(kv.first) == 0)
            right_temporal_edge_mates[kv.second].matching_CF_edge_clusters.clear();
    }
}

void EBVO::apply_NCC_filtering(std::vector<temporal_edge_pair> &temporal_edge_mates,
                               const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
                               double ncc_val_threshold, const cv::Mat &keyframe_image, const cv::Mat &current_image, bool b_is_left)
{
    Utility util{};
    cv::Mat kf_image_64f, cf_image_64f;
    keyframe_image.convertTo(kf_image_64f, CV_64F);
    current_image.convertTo(cf_image_64f, CV_64F);

    //> Thread-local storage for NCC scores
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<double>> thread_local_ncc_scores(num_threads);
    std::vector<std::vector<int>> thread_local_is_veridical(num_threads);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(temporal_edge_mates.size()); ++i)
        {
            temporal_edge_pair &tp = temporal_edge_mates[i];
            const final_stereo_edge_pair *kf_mate = tp.KF_stereo_edge_mate;

            std::pair<cv::Mat, cv::Mat> kf_edge_patches = (b_is_left) ? kf_mate->left_edge_patches : kf_mate->right_edge_patches;

            std::vector<Temporal_CF_Edge_Cluster> filtered_clusters;
            filtered_clusters.reserve(tp.matching_CF_edge_clusters.size());

            for (Temporal_CF_Edge_Cluster &cluster : tp.matching_CF_edge_clusters)
            {
                int cf_stereo_mate_idx = cluster.cf_stereo_edge_mate_index;
                if (cf_stereo_mate_idx < 0 || cf_stereo_mate_idx >= static_cast<int>(CF_stereo_edge_mates.size()))
                    continue;

                std::pair<cv::Mat, cv::Mat> cf_edge_patches = (b_is_left) ? CF_stereo_edge_mates[cf_stereo_mate_idx].left_edge_patches : CF_stereo_edge_mates[cf_stereo_mate_idx].right_edge_patches;

                double sim_pp = util.get_patch_similarity(kf_edge_patches.first, cf_edge_patches.first);
                double sim_nn = util.get_patch_similarity(kf_edge_patches.second, cf_edge_patches.second);
                double sim_pn = util.get_patch_similarity(kf_edge_patches.first, cf_edge_patches.second);
                double sim_np = util.get_patch_similarity(kf_edge_patches.second, cf_edge_patches.first);
                double final_SIM_score = std::max({sim_pp, sim_nn, sim_pn, sim_np});

                //> Record NCC score
                thread_local_ncc_scores[tid].push_back(final_SIM_score);

                //> Check if veridical
                cv::Point2d gt_location(tp.projected_point.x(), tp.projected_point.y());
                const Edge &cf_edge = b_is_left ? CF_stereo_edge_mates[cf_stereo_mate_idx].left_edge : CF_stereo_edge_mates[cf_stereo_mate_idx].right_edge;
                bool is_gt = cv::norm(cf_edge.location - gt_location) < DIST_TO_GT_THRESH_TEMP;
                thread_local_is_veridical[tid].push_back(is_gt ? 1 : 0);

                if (final_SIM_score > ncc_val_threshold)
                {
                    cluster.matching_scores.ncc_score = final_SIM_score;
                    cluster.matching_scores.sift_score = 900.0;
                    filtered_clusters.push_back(std::move(cluster));
                }
            }
            tp.matching_CF_edge_clusters = std::move(filtered_clusters);
        }
    }

    //> Merge thread-local data
    std::vector<double> ncc_scores;
    std::vector<int> is_veridical;
    for (int tid = 0; tid < num_threads; ++tid)
    {
        ncc_scores.insert(ncc_scores.end(), thread_local_ncc_scores[tid].begin(), thread_local_ncc_scores[tid].end());
        is_veridical.insert(is_veridical.end(), thread_local_is_veridical[tid].begin(), thread_local_is_veridical[tid].end());
    }

    std::string output_dir = dataset.get_output_path();
    size_t frame_idx = 0; // TODO: Pass this as parameter
#if RECORD_FILTER_DISTRIBUTIONS
    if (!output_dir.empty() && !ncc_scores.empty())
    {
        std::string side = b_is_left ? "left" : "right";
        Stereo_Matches stereo_matcher;
        stereo_matcher.record_Filter_Distribution("temporal_ncc_score_" + side, ncc_scores, is_veridical, output_dir, frame_idx);
        record_Temporal_Ambiguity_Distribution("ncc", temporal_edge_mates, output_dir, frame_idx, b_is_left);
    }
#endif
}

void EBVO::min_Edge_Photometric_Residual_by_Gauss_Newton(
    /* inputs */
    Edge kf_edge, Edge cf_edge, Eigen::Vector2d init_disp, const cv::Mat &kf_image_undistorted,
    const cv::Mat &cf_image_undistorted, const cv::Mat &cf_image_gradients_x, const cv::Mat &cf_image_gradients_y,
    /* outputs */
    Eigen::Vector2d &refined_disparity, double &refined_final_score, bool &refined_validity, std::vector<double> &residual_log,
    /* optional inputs */
    int max_iter, double tol, double huber_delta, bool b_verbose)
{
    cv::Point2d t(std::cos(kf_edge.orientation), std::sin(kf_edge.orientation));
    cv::Point2d n(-t.y, t.x);
    double side_shift = (PATCH_SIZE / 2.0) + 1.0;
    cv::Point2d c_plus = kf_edge.location + n * side_shift;
    cv::Point2d c_minus = kf_edge.location - n * side_shift;

    std::vector<cv::Point2d> cLplus, cLminus;
    util_make_rotated_patch_coords(c_plus, kf_edge.orientation, cLplus);
    util_make_rotated_patch_coords(c_minus, kf_edge.orientation, cLminus);

    std::vector<double> pLplus_f, pLminus_f;
    util_sample_patch_at_coords(kf_image_undistorted, cLplus, pLplus_f);
    util_sample_patch_at_coords(kf_image_undistorted, cLminus, pLminus_f);
    double mLplus = util_vector_mean<double>(pLplus_f);
    double mLminus = util_vector_mean<double>(pLminus_f);
    std::vector<double> Lplus, Lminus;
    for (double x : pLplus_f)
    {
        Lplus.push_back(x - mLplus);
    }
    for (double x : pLminus_f)
    {
        Lminus.push_back(x - mLminus);
    }

    Edge cf_edge_iterated = cf_edge;
    cv::Point2d t_cf(std::cos(cf_edge_iterated.orientation), std::sin(cf_edge_iterated.orientation));
    cv::Point2d n_cf(-t_cf.y, t_cf.x);

    Eigen::Vector2d d = init_disp;
    double init_RMS = 0.0;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        //> Compute the right patch coordinates
        cf_edge_iterated.location = cv::Point2d(kf_edge.location.x - d[0], kf_edge.location.y - d[1]);
        cv::Point2d cRplus = cf_edge_iterated.location + n_cf * side_shift;
        cv::Point2d cRminus = cf_edge_iterated.location - n_cf * side_shift;

        std::vector<cv::Point2d> cRplusC, cRminusC;
        util_make_rotated_patch_coords(cRplus, cf_edge_iterated.orientation, cRplusC);
        util_make_rotated_patch_coords(cRminus, cf_edge_iterated.orientation, cRminusC);

        //> Sample right intensities and right gradient X at these coords
        std::vector<double> pRplus_f, pRminus_f, gxRplus_f, gxRminus_f, gyRplus_f, gyRminus_f;
        util_sample_patch_at_coords(cf_image_undistorted, cRplusC, pRplus_f);
        util_sample_patch_at_coords(cf_image_undistorted, cRminusC, pRminus_f);
        util_sample_patch_at_coords(cf_image_gradients_x, cRplusC, gxRplus_f);
        util_sample_patch_at_coords(cf_image_gradients_x, cRminusC, gxRminus_f);
        util_sample_patch_at_coords(cf_image_gradients_y, cRplusC, gyRplus_f);
        util_sample_patch_at_coords(cf_image_gradients_y, cRminusC, gyRminus_f);

        //> Compute means of the right patches
        double mRplus = util_vector_mean<double>(pRplus_f);
        double mRminus = util_vector_mean<double>(pRminus_f);

        //> Build residuals r = (L - meanL) - (R - meanR)  which centers both patches
        //> Build gradient which is the derivative of the residual with respect to the disparity g = dr / dd
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        double cost = 0.0;
        auto accumulate_patch = [&](const std::vector<double> &Lc, const std::vector<double> &Rf,
                                    const std::vector<double> &gxRf, const std::vector<double> &gyRf, double meanR)
        {
            for (size_t k = 0; k < Lc.size(); ++k)
            {
                double r = Lc[k] - (Rf[k] - meanR);
                Eigen::Vector2d J = Eigen::Vector2d(gxRf[k], gyRf[k]);
                double absr = std::abs(r);
                double w = (absr < huber_delta) ? 1.0 : huber_delta / absr;

                H += w * J * J.transpose();
                H += 1e-6 * Eigen::Matrix2d::Identity();
                b += w * J * r;
                cost += w * r * r;
            }
        };
        accumulate_patch(Lplus, pRplus_f, gxRplus_f, gyRplus_f, mRplus);
        accumulate_patch(Lminus, pRminus_f, gxRminus_f, gyRminus_f, mRminus);

        //> Update delta
        Eigen::Vector2d delta = -H.ldlt().solve(b);
        d += delta;

        double rms = std::sqrt(cost / (Lplus.size() + Lminus.size()));
        if (iter == 0)
            init_RMS = rms;
        if (b_verbose)
        {
            std::cout << "iter " << iter << ": disp =" << d
                      << "  Δ =" << delta
                      << "  RMS =" << rms
                      << "  cost =" << cost << std::endl;
        }
        residual_log.push_back(rms);

        bool is_outlier = (rms > huber_delta * 2.0) || (residual_log.size() < 2);

        //> Early stopping if the update is too small
        if (delta.norm() < tol || iter == max_iter - 1)
        {
            refined_validity = (is_outlier) ? false : true;
            refined_final_score = rms;
            break;
        }
    }

    refined_disparity = d;
}

void EBVO::apply_orientation_filtering(std::vector<temporal_edge_pair> &temporal_edge_mates,
                                       const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
                                       double orientation_threshold, bool b_is_left)
{
    //> Thread-local storage for orientation differences
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<double>> thread_local_orientation_diffs(num_threads);
    std::vector<std::vector<int>> thread_local_is_veridical(num_threads);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(temporal_edge_mates.size()); ++i)
        {
            temporal_edge_pair &tp = temporal_edge_mates[i];
            double ref_orientation = tp.projected_orientation;

            std::vector<Temporal_CF_Edge_Cluster> filtered_clusters;
            filtered_clusters.reserve(tp.matching_CF_edge_clusters.size());

            for (Temporal_CF_Edge_Cluster &cluster : tp.matching_CF_edge_clusters)
            {
                int cf_stereo_mate_idx = cluster.cf_stereo_edge_mate_index;
                const Edge &cf_edge = b_is_left ? CF_stereo_edge_mates[cf_stereo_mate_idx].left_edge : CF_stereo_edge_mates[cf_stereo_mate_idx].right_edge;
                double orientation_diff = std::abs(rad_to_deg<double>(ref_orientation - cf_edge.orientation));
                if (orientation_diff > 180.0)
                    orientation_diff = 360.0 - orientation_diff;
                double normalized_orientation_diff = orientation_diff;
                if (normalized_orientation_diff > 90.0)
                    normalized_orientation_diff = std::abs(normalized_orientation_diff - 180.0);

                //> Record orientation difference
                thread_local_orientation_diffs[tid].push_back(normalized_orientation_diff);

                //> Check if veridical (GT location check)
                cv::Point2d gt_location(tp.projected_point.x(), tp.projected_point.y());
                bool is_gt = cv::norm(cf_edge.location - gt_location) < DIST_TO_GT_THRESH_TEMP;
                thread_local_is_veridical[tid].push_back(is_gt ? 1 : 0);

                if (orientation_diff < orientation_threshold || std::abs(orientation_diff - 180.0) < orientation_threshold)
                {
                    cluster.center_edge = cf_edge;
                    filtered_clusters.push_back(std::move(cluster));
                }
            }
            tp.matching_CF_edge_clusters = std::move(filtered_clusters);
        }
    }

    //> Merge thread-local data
    std::vector<double> orientation_differences;
    std::vector<int> is_veridical;
    for (int tid = 0; tid < num_threads; ++tid)
    {
        orientation_differences.insert(orientation_differences.end(), thread_local_orientation_diffs[tid].begin(), thread_local_orientation_diffs[tid].end());
        is_veridical.insert(is_veridical.end(), thread_local_is_veridical[tid].begin(), thread_local_is_veridical[tid].end());
    }

    std::string output_dir = dataset.get_output_path();
    size_t frame_idx = 0; // TODO: Pass this as parameter
#if RECORD_FILTER_DISTRIBUTIONS
    if (!output_dir.empty() && !orientation_differences.empty())
    {
        std::string side = b_is_left ? "left" : "right";
        Stereo_Matches stereo_matcher;
        stereo_matcher.record_Filter_Distribution("temporal_orientation_difference_" + side, orientation_differences, is_veridical, output_dir, frame_idx);
        record_Temporal_Ambiguity_Distribution("orientation", temporal_edge_mates, output_dir, frame_idx, b_is_left);
    }
#endif
}
void EBVO::Evaluate_Temporal_Edge_Pairs_on_Quads(const std::vector<temporal_edge_pair> &left_edge_mates, const std::vector<temporal_edge_pair> &right_edge_mates,
                                                 size_t frame_idx)
{
    if (left_edge_mates.empty() || right_edge_mates.empty())
    {
        std::cout << "Temporal Edge Correspondences Evaluation: Stage: Quad numbers | Frame: " << frame_idx << std::endl;
        std::cout << " (NO PAIRS FOUND)" << std::endl;
        std::cout << "========================================================\n"
                  << std::endl;
        return;
    }
    double recall_per_temporal_image = 0.0;
    double precision_per_temporal_image = 0.0;
    double ambiguity_per_temporal_image = 0.0;
    double precision_pre_pair_temp_image = 0.0;
    int num_of_KF_stereo_TP_edge_mates = 0;
    int num_of_KF_steroe_TP_edge_mates_with_clusters = 0;
    std::unordered_map<const final_stereo_edge_pair *, size_t> kf_mate_to_right_idx;
    for (size_t i = 0; i < right_edge_mates.size(); ++i)
        kf_mate_to_right_idx[right_edge_mates[i].KF_stereo_edge_mate] = i;

    //> Build KF_stereo_edge_mate -> left temporal pair index map
    std::unordered_map<const final_stereo_edge_pair *, size_t> kf_mate_to_left_idx;
    for (size_t i = 0; i < left_edge_mates.size(); ++i)
        kf_mate_to_left_idx[left_edge_mates[i].KF_stereo_edge_mate] = i;

    for (const auto &kv : kf_mate_to_left_idx)
    {
        const final_stereo_edge_pair *kf_mate = kv.first;
        size_t left_idx = kv.second;
        auto it_right = kf_mate_to_right_idx.find(kf_mate);
        if (it_right == kf_mate_to_right_idx.end())
        {
            continue;
        }
        size_t right_idx = it_right->second;

        const temporal_edge_pair &left_tp = left_edge_mates[left_idx];
        const temporal_edge_pair &right_tp = right_edge_mates[right_idx];
        if (!left_tp.KF_stereo_edge_mate->b_is_TP)
            continue;
        if (!right_tp.KF_stereo_edge_mate->b_is_TP)
            continue;
        cv::Point2d gt_location_left(left_tp.projected_point.x(), left_tp.projected_point.y());
        cv::Point2d gt_location_right(right_tp.projected_point.x(), right_tp.projected_point.y());
        int num_TP_centers = 0;
        int num_quad_pairs = 0; // Count left clusters that form at least one quad
        std::unordered_map<int, cv::Point2d> right_cf_indices;
        for (const Temporal_CF_Edge_Cluster &c : right_tp.matching_CF_edge_clusters)
        {
            if (!c.contributing_cf_stereo_indices.empty())
                // if it has contributing indices, add all of them; otherwise add the primary mate index
                for (int idx : c.contributing_cf_stereo_indices)
                    right_cf_indices.insert({idx, c.center_edge.location});
            else
                right_cf_indices.insert({c.cf_stereo_edge_mate_index, c.center_edge.location});
        }
        for (const Temporal_CF_Edge_Cluster &c : left_tp.matching_CF_edge_clusters)
        {
            bool cluster_has_valid_match = false;
            bool cluster_forms_quad = false;
            std::vector<int> left_indices = c.contributing_cf_stereo_indices.empty() ? std::vector<int>{c.cf_stereo_edge_mate_index} : c.contributing_cf_stereo_indices;
            for (int idx : left_indices)
            {
                if (right_cf_indices.count(idx))
                {
                    cluster_forms_quad = true;
                    cv::Point2d right_cluster_center = right_cf_indices[idx];
                    double dist_left = cv::norm(c.center_edge.location - gt_location_left);
                    double dist_right = cv::norm(right_cluster_center - gt_location_right);
                    if (dist_left < DIST_TO_GT_THRESH_TEMP && dist_right < DIST_TO_GT_THRESH_TEMP)
                    {
                        // This cluster has at least one valid quad match
                        cluster_has_valid_match = true;
                        break; // No need to check other indices for this cluster
                    }
                }
            }
            if (cluster_forms_quad)
                num_quad_pairs++;
            if (cluster_has_valid_match)
                num_TP_centers++;
        }
        size_t num_clusters = num_quad_pairs;
        double recall_per_edge = (num_TP_centers >= 1) ? 1.0 : 0.0;
        double precision_per_edge = (num_clusters == 0) ? 0.0 : (static_cast<double>(num_TP_centers) / static_cast<double>(num_clusters));
        double ambiguity_per_edge = static_cast<double>(num_clusters);

        recall_per_temporal_image += recall_per_edge;
        precision_per_temporal_image += precision_per_edge;
        ambiguity_per_temporal_image += ambiguity_per_edge;
        precision_pre_pair_temp_image += precision_per_edge;
        if (num_clusters > 0)
            num_of_KF_steroe_TP_edge_mates_with_clusters++;
        num_of_KF_stereo_TP_edge_mates++;
    }
    double recall_per_image = recall_per_temporal_image / static_cast<double>(num_of_KF_stereo_TP_edge_mates);
    double precision_per_image = precision_per_temporal_image / static_cast<double>(num_of_KF_stereo_TP_edge_mates);
    double ambiguity_avg = ambiguity_per_temporal_image / static_cast<double>(num_of_KF_steroe_TP_edge_mates_with_clusters);
    double precision_pre_pair_temp_image_avg = precision_pre_pair_temp_image / static_cast<double>(num_of_KF_steroe_TP_edge_mates_with_clusters);

    std::cout << "Temporal Edge Correspondences Evaluation: Stage: Quad number | Frame: " << frame_idx << std::endl;
    std::cout << "- Recall rate:       " << std::fixed << std::setprecision(8) << recall_per_image << std::endl;
    std::cout << "- Precision rate:    " << std::fixed << std::setprecision(8) << precision_per_image << std::endl;
    std::cout << "- Average ambiguity: " << std::fixed << std::setprecision(8) << ambiguity_avg << std::endl;
    std::cout << "- Precision per pair: " << std::fixed << std::setprecision(8) << precision_pre_pair_temp_image_avg << std::endl;
    std::cout << "========================================================\n"
              << std::endl;
}

void EBVO::Evaluate_KF_CF_Edge_Correspondences(const std::vector<temporal_edge_pair> &temporal_edge_mates,
                                               size_t frame_idx, const std::string &stage_name, const std::string which_side_of_temporal_edge_mates)
{
    if (temporal_edge_mates.empty())
    {
        std::cout << "Temporal Edge Correspondences Evaluation: Stage: " << stage_name << " | Frame: " << frame_idx << std::endl;
        std::cout << " (NO PAIRS FOUND)" << std::endl;
        std::cout << "========================================================\n"
                  << std::endl;
        return;
    }

    double recall_per_temporal_image = 0.0;
    double precision_per_temporal_image = 0.0;
    double ambiguity_per_temporal_image = 0.0;
    double precision_pre_pair_temp_image = 0.0;
    int num_of_KF_stereo_TP_edge_mates = 0;
    int num_of_KF_steroe_TP_edge_mates_with_clusters = 0;
    for (const temporal_edge_pair &tp : temporal_edge_mates)
    {
        if (!tp.KF_stereo_edge_mate->b_is_TP)
            continue;

        cv::Point2d gt_location(tp.projected_point.x(), tp.projected_point.y());
        int num_TP_centers = 0;
        for (const Temporal_CF_Edge_Cluster &cluster : tp.matching_CF_edge_clusters)
        {
            double dist = cv::norm(cluster.center_edge.location - gt_location);
            if (dist < DIST_TO_GT_THRESH_TEMP)
                num_TP_centers++;
        }

        size_t num_clusters = tp.matching_CF_edge_clusters.size();
        double recall_per_edge = (num_TP_centers >= 1) ? 1.0 : 0.0;
        double precision_per_edge = (num_clusters == 0) ? 0.0 : (static_cast<double>(num_TP_centers) / static_cast<double>(num_clusters));
        double ambiguity_per_edge = static_cast<double>(num_clusters);

        recall_per_temporal_image += recall_per_edge;
        precision_per_temporal_image += precision_per_edge;
        ambiguity_per_temporal_image += ambiguity_per_edge;
        precision_pre_pair_temp_image += precision_per_edge;
        if (num_clusters > 0)
            num_of_KF_steroe_TP_edge_mates_with_clusters++;
        num_of_KF_stereo_TP_edge_mates++;
    }

    double recall_per_image = recall_per_temporal_image / static_cast<double>(num_of_KF_stereo_TP_edge_mates);
    double precision_per_image = precision_per_temporal_image / static_cast<double>(num_of_KF_stereo_TP_edge_mates);
    double ambiguity_avg = ambiguity_per_temporal_image / static_cast<double>(num_of_KF_steroe_TP_edge_mates_with_clusters);
    double precision_pre_pair_temp_image_avg = precision_pre_pair_temp_image / static_cast<double>(num_of_KF_steroe_TP_edge_mates_with_clusters);

    std::cout << "Temporal Edge Correspondences Evaluation: Stage: " << stage_name << " | Frame: " << frame_idx;
    std::cout << " (Side: " << which_side_of_temporal_edge_mates << ")" << std::endl;
    std::cout << "- Recall rate:       " << std::fixed << std::setprecision(8) << recall_per_image << std::endl;
    std::cout << "- Precision rate:    " << std::fixed << std::setprecision(8) << precision_per_image << std::endl;
    std::cout << "- Average ambiguity: " << std::fixed << std::setprecision(8) << ambiguity_avg << std::endl;
    std::cout << "- Precision per pair: " << std::fixed << std::setprecision(8) << precision_pre_pair_temp_image_avg << std::endl;
    std::cout << "========================================================\n"
              << std::endl;
}

void EBVO::cleaning_temporal_edge_mates(std::vector<temporal_edge_pair> &temporal_edge_mates)
{
    std::vector<int> indices_to_remove;
    for (int i = 0; i < temporal_edge_mates.size(); i++)
    {
        if (temporal_edge_mates[i].matching_CF_edge_clusters.empty())
        {
            indices_to_remove.push_back(i);
        }
    }

    //> Remove the left edges from the stereo_frame structure if there is no right edge correspondences close to the GT edge
    if (!indices_to_remove.empty())
    {
        //> First sort the indices in an descending order
        std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());
        for (size_t no_GT_index : indices_to_remove)
        {
            temporal_edge_mates.erase(temporal_edge_mates.begin() + no_GT_index);
        }

        //> Free excess memory capacity
        temporal_edge_mates.shrink_to_fit();
    }
}
