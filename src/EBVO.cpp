#include <filesystem>
#include <unordered_set>
#include <numeric>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "EBVO.h"
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
    std::vector<final_stereo_edge_pair> keyframe_stereo_edge_pairs;
    std::vector<final_stereo_edge_pair> current_frame_stereo_edge_pairs;

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    bool b_is_keyframe = true;

    size_t frame_idx = 0;
    while (dataset.stereo_iterator->hasNext() && num_pairs - frame_idx >= 0)
    {
        if (!dataset.stereo_iterator->getNext(current_frame))
        {
            std::cout << "No more image pairs to process" << std::endl;
            break;
        }

        const cv::Mat &left_disparity_map = (frame_idx < left_ref_disparity_maps.size()) ? left_ref_disparity_maps[frame_idx] : cv::Mat();
        const cv::Mat &right_disparity_map = (frame_idx < right_ref_disparity_maps.size()) ? right_ref_disparity_maps[frame_idx] : cv::Mat();

        //> FOr now, this is optional
        const cv::Mat &left_occlusion_mask = (frame_idx < left_occlusion_masks.size()) ? left_occlusion_masks[frame_idx] : cv::Mat();
        const cv::Mat &right_occlusion_mask = (frame_idx < right_occlusion_masks.size()) ? right_occlusion_masks[frame_idx] : cv::Mat();

        std::cout << std::endl
                  << "Image Pair #" << frame_idx << std::endl;

        cv::Mat left_cur_undistorted, right_cur_undistorted;
        cv::undistort(current_frame.left_image, left_cur_undistorted, left_calib, left_dist_coeff_mat);
        cv::undistort(current_frame.right_image, right_cur_undistorted, right_calib, right_dist_coeff_mat);
        current_frame.left_image_undistorted = left_cur_undistorted;
        current_frame.right_image_undistorted = right_cur_undistorted;

        util_compute_Img_Gradients(current_frame.left_image_undistorted, current_frame.left_image_gradients_x, current_frame.left_image_gradients_y);
        util_compute_Img_Gradients(current_frame.right_image_undistorted, current_frame.right_image_gradients_x, current_frame.right_image_gradients_y);

        if (dataset.get_num_imgs() == 0)
        {
            dataset.set_height(left_cur_undistorted.rows);
            dataset.set_width(left_cur_undistorted.cols);

            TOED = std::shared_ptr<ThirdOrderEdgeDetectionCPU>(new ThirdOrderEdgeDetectionCPU(dataset.get_height(), dataset.get_width()));

            // Initialize the spatial grids with a cell size of defined GRID_SIZE
            left_grid = SpatialGrid(dataset.get_width(), dataset.get_height(), GRID_SIZE);
            right_grid = SpatialGrid(dataset.get_width(), dataset.get_height(), GRID_SIZE);
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
            kf_edges_left = dataset.left_edges;
            last_keyframe_stereo_left.clean_up_vector_data_structures();
            last_keyframe_stereo_left.stereo_frame = &last_keyframe;
            last_keyframe_stereo_left.left_disparity_map = left_disparity_map;
            last_keyframe_stereo_left.right_disparity_map = right_disparity_map;

            //> For each left edge, get the corresponding GT location (not right edge) on the right image, and the triangulated 3D point in the left camera coordinate
            stereo_edge_matcher.Find_Stereo_GT_Locations(dataset, left_disparity_map, last_keyframe, last_keyframe_stereo_left, true);
            std::cout << "Complete calculating GT locations for left edges of the keyframe (previous frame)..." << std::endl;
            //> Construct a GT stereo edge pool
            stereo_edge_matcher.get_Stereo_Edge_GT_Pairs(dataset, last_keyframe, last_keyframe_stereo_left, true);
            std::cout << "Size of stereo edge correspondences pool = " << last_keyframe_stereo_left.focused_edge_indices.size() << std::endl;

            last_keyframe_stereo_left.construct_toed_left_id_to_Stereo_Edge_Pairs_left_id_map();

            //> construct stereo edge correspondences for the keyframe frame
            Frame_Evaluation_Metrics metrics = stereo_edge_matcher.get_Stereo_Edge_Pairs(dataset, last_keyframe_stereo_left, frame_idx);

            //> Finalize the stereo edge pairs for the keyframe
            stereo_edge_matcher.construct_candidate_set(last_keyframe_stereo_left, keyframe_stereo_edge_pairs);

            if (!last_keyframe_stereo_left.b_is_size_consistent())
                last_keyframe_stereo_left.print_size_consistency();
            b_is_keyframe = false;
        }
        else
        {
            cf_edges_left = dataset.left_edges;
            current_frame_stereo_left.clean_up_vector_data_structures();
            current_frame_stereo_left.stereo_frame = &current_frame;
            current_frame_stereo_left.left_disparity_map = left_disparity_map;
            current_frame_stereo_left.right_disparity_map = right_disparity_map;

            stereo_edge_matcher.Find_Stereo_GT_Locations(dataset, left_disparity_map, current_frame, current_frame_stereo_left, true);
            std::cout << "Complete calculating GT locations for left edges of the current frame..." << current_frame_stereo_left.focused_edge_indices.size() << std::endl;

            //> Construct a GT stereo edge pool
            stereo_edge_matcher.get_Stereo_Edge_GT_Pairs(dataset, current_frame, current_frame_stereo_left, true);
            std::cout << "Size of stereo edge correspondences pool for left edges= " << current_frame_stereo_left.focused_edge_indices.size() << std::endl;

            // Construct TOED-to-Stereo_Edge_Pairs mapping for the current frame
            current_frame_stereo_left.construct_toed_left_id_to_Stereo_Edge_Pairs_left_id_map();

            //> construct stereo edge correspondences for the current frame
            Frame_Evaluation_Metrics metrics = stereo_edge_matcher.get_Stereo_Edge_Pairs(dataset, current_frame_stereo_left, frame_idx);

            //> Finalize the stereo edge pairs for the keyframe
            stereo_edge_matcher.construct_candidate_set(current_frame_stereo_left, current_frame_stereo_edge_pairs);
            // stereo_edge_matcher.construct_candidate_set(current_frame_stereo_left, cf_edges_right, cf_right_eval);
            // std::cout << "Size of candidate set = " << cf_edges_right.size() << std::endl;
            //> extract SIFT descriptor for each left edge of current_frame_stereo
            // stereo_edge_matcher.augment_Edge_Data(current_frame_stereo_left, true);
            if (!current_frame_stereo_left.b_is_size_consistent())
                current_frame_stereo_left.print_size_consistency();

            add_edges_to_spatial_grid(current_frame_stereo_left, left_grid, true);
            std::cout << "Finish adding left edges of the current frame to spatial grid with cell size " << GRID_SIZE << std::endl;
            add_edges_to_spatial_grid(current_frame_stereo_left, right_grid, false);
            std::cout << "Finish adding right edges of the current frame to spatial grid with cell size " << GRID_SIZE << std::endl;
            //> Construct correspondences structure between last keyframe and the current frame
            KF_CF_EdgeCorrespondence KF_CF_edge_pairs_left, KF_CF_edge_pairs_right;

            // Use ALL-edges grid for temporal matching
            Find_Veridical_Edge_Correspondences_on_CF(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, left_grid, true);
            std::cout << "Size of veridical edge pairs (left) = " << KF_CF_edge_pairs_left.kf_edges.size() << std::endl;
            // Use ALL-edges grid for temporal matching
            Find_Veridical_Edge_Correspondences_on_CF(KF_CF_edge_pairs_right, last_keyframe_stereo_left, current_frame_stereo_left, right_grid, false);
            std::cout << "Size of veridical edge pairs (right) = " << KF_CF_edge_pairs_right.kf_edges.size() << std::endl;

            //> Now that the GT edge correspondences are constructed between the keyframe and the current frame, we can apply various filters from the beginning
            //> Stage 1: Apply spatial grid to the current frame
            apply_spatial_grid_filtering(KF_CF_edge_pairs_left, left_grid, 30.0);
            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "Spatial Grid");
            apply_spatial_grid_filtering(KF_CF_edge_pairs_right, right_grid, 30.0);
            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_right, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "Spatial Grid");

            // //> Stage 2: Do orientation filtering
            apply_orientation_filtering(KF_CF_edge_pairs_left, 35.0, true);
            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "Orientation Filtering");
            apply_orientation_filtering(KF_CF_edge_pairs_right, 35.0, false);
            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_right, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "Orientation Filtering");

            //> Stage 3: Do NCC
            apply_NCC_filtering(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, 0.6, last_keyframe.left_image, current_frame.left_image, true);
            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "NCC Filtering");
            apply_NCC_filtering(KF_CF_edge_pairs_right, last_keyframe_stereo_left, current_frame_stereo_left, 0.6, last_keyframe.right_image, current_frame.right_image, false);
            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_right, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "NCC Filtering");

            // augment_all_Edge_Data(current_frame_stereo_left, current_frame_descriptors_left, true);
            // augment_all_Edge_Data(current_frame_stereo_left, current_frame_descriptors_right, false);
            // apply_SIFT_filtering(KF_CF_edge_pairs_left, 500.0, true);
            // Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "SIFT Filtering");

            // apply_best_nearly_best_filtering(KF_CF_edge_pairs_left, 0.8, true);
            // Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "BNB NCC Filtering");
            // apply_best_nearly_best_filtering(KF_CF_edge_pairs_left, 0.3, false);
            // Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "BNB SIFT Filtering");

            // //> Stage 4: Stereo consistency filtering
            // apply_stereo_filtering(KF_CF_edge_pairs_left, KF_CF_edge_pairs_right,
            //                        last_keyframe_stereo_left, current_frame_stereo_left,
            //                        last_keyframe_stereo_right, current_frame_stereo_right,
            //                        frame_idx);
            // Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "Mate consistency Filtering");
            break;
        }

        frame_idx++;
        if (frame_idx > 1)
        {
            break;
        }
    }
}

void EBVO::add_edges_to_spatial_grid(Stereo_Edge_Pairs &stereo_frame, SpatialGrid &spatial_grid, bool is_left)
{
    //> Add left edges to spatial grid. This is done on the current image only.
    const std::vector<int> &left_toed_index = stereo_frame.focused_edge_indices;
    stereo_frame.grid_indices.resize(left_toed_index.size());

    // Pre-compute grid cell assignments in parallel (read-only, no race conditions)
    std::vector<std::pair<int, int>> edge_to_grid(left_toed_index.size()); // <edge_idx, grid_cell_idx>

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(left_toed_index.size()); ++i)
    {
        cv::Point2d edge_location = stereo_frame.get_focused_edge_by_Stereo_Edge_Pairs_index(i).location;
        is_left ? edge_location = edge_location : edge_location = stereo_frame.matching_edge_clusters[i].edge_clusters[0].center_edge.location; //> at this point left edge has one and only one correspondence in the right edge
        int grid_x = static_cast<int>(edge_location.x) / spatial_grid.cell_size;
        int grid_y = static_cast<int>(edge_location.y) / spatial_grid.cell_size;

        if (grid_x >= 0 && grid_x < spatial_grid.grid_width &&
            grid_y >= 0 && grid_y < spatial_grid.grid_height)
        {
            int grid_idx = grid_y * spatial_grid.grid_width + grid_x;
            edge_to_grid[i] = {i, grid_idx};
            stereo_frame.grid_indices[i] = grid_idx;
        }
        else
        {
            edge_to_grid[i] = {i, -1}; // Mark as invalid
            stereo_frame.grid_indices[i] = -1;
        }
    }

    for (int i = 0; i < static_cast<int>(left_toed_index.size()); ++i)
    {
        int grid_idx = edge_to_grid[i].second;
        if (grid_idx >= 0 && grid_idx < static_cast<int>(spatial_grid.grid.size()))
        {
            int idx_to_store = is_left ? left_toed_index[i] : i;
            spatial_grid.grid[grid_idx].push_back(idx_to_store); // Store actual TOED edge index or the representative indices
        }
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

double orientation_mapping(const Edge &e_left, const Edge &e_right, const Eigen::Vector3d projected_point, bool is_left_cam, const StereoFrame &last_keyframe, const StereoFrame &current_frame, Dataset &dataset)
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

void EBVO::Find_Veridical_Edge_Correspondences_on_CF(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs,
                                                     Stereo_Edge_Pairs &last_keyframe_stereo, Stereo_Edge_Pairs &current_frame_stereo,
                                                     SpatialGrid &spatial_grid, bool is_left, double gt_dist_threshold)
{
    std::string filename = dataset.get_output_path() + "/veridical_edges_frame" + std::to_string(is_left) + ".txt";
    std::ofstream outfile(filename);
    KF_CF_edge_pairs.is_left = is_left;
    KF_CF_edge_pairs.key_frame_pairs = &last_keyframe_stereo;
    KF_CF_edge_pairs.current_frame_pairs = &current_frame_stereo;
    //> Ground-truth relative pose of the current frame with respect to the keyframe
    Eigen::Matrix3d R1 = last_keyframe_stereo.stereo_frame->gt_rotation;
    Eigen::Vector3d t1 = last_keyframe_stereo.stereo_frame->gt_translation;
    t1 = is_left ? t1 : t1 + dataset.get_relative_transl_left_to_right();
    Eigen::Matrix3d R2 = current_frame_stereo.stereo_frame->gt_rotation;
    Eigen::Vector3d t2 = current_frame_stereo.stereo_frame->gt_translation;
    t2 = is_left ? t2 : t2 + dataset.get_relative_transl_left_to_right();

    Eigen::Matrix3d R_left = R2 * R1.transpose(); // R_left
    Eigen::Matrix3d R21 = is_left ? R_left : dataset.get_relative_rot_left_to_right() * R_left * dataset.get_relative_rot_left_to_right().transpose();

    Eigen::Vector3d t21 = t2 - R2 * R1.transpose() * t1; // t_left
    t21 = is_left ? t21 : -dataset.get_relative_rot_left_to_right() * R_left * dataset.get_relative_rot_left_to_right().transpose() * dataset.get_relative_transl_left_to_right() + dataset.get_relative_rot_left_to_right() * R_left * t21 + dataset.get_relative_transl_left_to_right();

    int size = is_left ? last_keyframe_stereo.focused_edge_indices.size() : kf_edges_right.size(); // they should be the same.

    std::vector<Edge> &current_kf_edges = is_left ? kf_edges_left : kf_edges_right;
    std::vector<Edge> &current_cf_edges = is_left ? cf_edges_left : cf_edges_right;
    std::vector<Edge> &other_keyframe_edges = is_left ? kf_edges_right : kf_edges_left;
    // right now we have populated cf_edges_right and kf_edges_right that stores the representative edges
    // Thread-local storage for parallel correspondence accumulation
    int num_threads_corr = omp_get_max_threads();

    std::vector<std::vector<int>> thread_kf_edges(num_threads_corr);
    std::vector<std::vector<int>> thread_stereo_frame_indices(num_threads_corr); // Track stereo frame indices for GT lookups
    std::vector<std::vector<double>> thread_gt_orientations(num_threads_corr);
    std::vector<std::vector<cv::Point2d>> thread_gt_locations(num_threads_corr);
    std::vector<std::vector<std::vector<int>>> thread_veridical_cf_edges(num_threads_corr);
    std::vector<std::vector<std::vector<scores>>> thread_scores(num_threads_corr);

    //> For each left edge in the keyframe, find the GT location on the current image
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, 128)
        for (int i = 0; i < size; ++i)
        {

            Eigen::Matrix3d calib_matrix;
            if (is_left)
            {
                calib_matrix = dataset.get_left_calib_matrix();
            }
            else
            {
                calib_matrix = dataset.get_right_calib_matrix();
            }

            int kf_idx = is_left ? last_keyframe_stereo.focused_edge_indices[i] : i;

            // 1. Calculate raw Left KF -> Left CF ego-motion directly from GT
            Eigen::Matrix3d R_left_ego = current_frame_stereo.stereo_frame->gt_rotation * last_keyframe_stereo.stereo_frame->gt_rotation.transpose();
            Eigen::Vector3d t_left_ego = current_frame_stereo.stereo_frame->gt_translation - R_left_ego * last_keyframe_stereo.stereo_frame->gt_translation;

            Eigen::Vector3d projected_point;

            if (is_left)
            {
                // Left KF -> Left CF
                Eigen::Vector3d pt_in_left_cf = R_left_ego * last_keyframe_stereo.Gamma_in_left_cam_coord[i] + t_left_ego;
                projected_point = calib_matrix * pt_in_left_cf;
            }
            else
            {
                // Left KF -> Left CF
                Eigen::Vector3d pt_in_left_cf = R_left_ego * last_keyframe_stereo.Gamma_in_left_cam_coord[i] + t_left_ego;

                // Left CF -> Right CF (Apply stereo baseline shift)
                Eigen::Vector3d pt_in_right_cf = dataset.get_relative_rot_left_to_right() * pt_in_left_cf + dataset.get_relative_transl_left_to_right();

                projected_point = calib_matrix * pt_in_right_cf;
            }

            projected_point /= projected_point.z();
            int kf_veridical_edge_indx = is_left ? last_keyframe_stereo.veridical_right_edges_indices[i][0] : last_keyframe_stereo.focused_edge_indices[i];
            Edge e_left = is_left ? current_kf_edges[kf_idx] : other_keyframe_edges[kf_veridical_edge_indx];
            Edge e_right = is_left ? other_keyframe_edges[kf_veridical_edge_indx] : current_kf_edges[kf_idx];

            double ori = orientation_mapping(
                e_left,
                e_right,
                projected_point,
                is_left,
                *last_keyframe_stereo.stereo_frame,
                *current_frame_stereo.stereo_frame,
                dataset);
            cv::Point2d projected_point_cv(projected_point.x(), projected_point.y());

            if (projected_point.x() > 10 && projected_point.y() > 10 && projected_point.x() < dataset.get_width() - 10 && projected_point.y() < dataset.get_height() - 10)
            {
                std::vector<int> current_candidate_edge_indices;
                double search_radius = 15.0 + gt_dist_threshold + 3.0; // +3 for safety margin
                current_candidate_edge_indices = spatial_grid.getCandidatesWithinRadius(projected_point_cv, search_radius);

                std::vector<int> CF_veridical_edges_indices;
                std::vector<scores> CF_veridical_edges_scores;
                for (const auto &curr_e_index : current_candidate_edge_indices)
                {
                    //> Check if the Euclidean distance is less than some threshold
                    double dist = cv::norm(current_cf_edges[curr_e_index].location - projected_point_cv);
                    double orientation_diff = std::abs(rad_to_deg<double>(ori - current_cf_edges[curr_e_index].orientation));
                    if (orientation_diff > 180.0)
                    {
                        orientation_diff = 360.0 - orientation_diff;
                    }
                    double orientation_threshold = 30.0; // degrees
                    if ((dist < gt_dist_threshold) && (orientation_diff < orientation_threshold ||
                                                       std::abs(orientation_diff - 180.0) < orientation_threshold))
                    {
                        CF_veridical_edges_indices.push_back(curr_e_index);
                    }
                    scores score;
                    score.ncc_score = -1.0;
                    score.sift_score = 900.0;
                    CF_veridical_edges_scores.push_back(score);
                }

                // Only record KF edges that have at least one veridical correspondence
                if (!CF_veridical_edges_indices.empty())
                {
                    thread_kf_edges[tid].push_back(kf_idx);
                    thread_stereo_frame_indices[tid].push_back(i);
                    thread_gt_orientations[tid].push_back(ori);
                    thread_gt_locations[tid].push_back(projected_point_cv);
                    thread_veridical_cf_edges[tid].push_back(CF_veridical_edges_indices);
                    thread_scores[tid].push_back(CF_veridical_edges_scores);
                }
            }
        }
    }

    // Merge thread-local correspondence data into the main structure
    for (int tid = 0; tid < num_threads_corr; ++tid)
    {
        KF_CF_edge_pairs.kf_edges.insert(KF_CF_edge_pairs.kf_edges.end(),
                                         thread_kf_edges[tid].begin(), thread_kf_edges[tid].end());
        KF_CF_edge_pairs.gt_orientation_on_cf.insert(KF_CF_edge_pairs.gt_orientation_on_cf.end(),
                                                     thread_gt_orientations[tid].begin(), thread_gt_orientations[tid].end());
        KF_CF_edge_pairs.gt_location_on_cf.insert(KF_CF_edge_pairs.gt_location_on_cf.end(),
                                                  thread_gt_locations[tid].begin(), thread_gt_locations[tid].end());
        KF_CF_edge_pairs.veridical_cf_edges_indices.insert(KF_CF_edge_pairs.veridical_cf_edges_indices.end(),
                                                           thread_veridical_cf_edges[tid].begin(), thread_veridical_cf_edges[tid].end());
        KF_CF_edge_pairs.matching_scores.insert(KF_CF_edge_pairs.matching_scores.end(),
                                                thread_scores[tid].begin(), thread_scores[tid].end());
    }

    for (int i = 0; i < KF_CF_edge_pairs.kf_edges.size(); ++i)
    {
        int kf_edge_index = KF_CF_edge_pairs.kf_edges[i];
        cv::Point2d gt_location = KF_CF_edge_pairs.gt_location_on_cf[i];
        double gt_orientation = KF_CF_edge_pairs.gt_orientation_on_cf[i];
        std::vector<int> veridical_cf_edges = KF_CF_edge_pairs.veridical_cf_edges_indices[i];

        outfile << "KF Edge Index: " << kf_edge_index << " (Location: (" << current_kf_edges[kf_edge_index].location.x << ", " << current_kf_edges[kf_edge_index].location.y << ")"

                << ", GT Location on CF: (" << gt_location.x << ", " << gt_location.y << ")"
                << ", GT Orientation on CF: " << gt_orientation
                << "\n Veridical CF Edges [";
        for (size_t j = 0; j < veridical_cf_edges.size(); ++j)
        {
            int cf_edge_index = veridical_cf_edges[j];
            outfile << cf_edge_index << " (Location: (" << current_cf_edges[cf_edge_index].location.x << ", " << current_cf_edges[cf_edge_index].location.y << ")"
                    << ", Orientation: " << rad_to_deg<double>(current_cf_edges[cf_edge_index].orientation) << "°)";
            if (j < veridical_cf_edges.size() - 1)
            {
                outfile << ", ";
            }
        }
        if (!is_left)
        {
            outfile << "]; is " << kf_right_eval[kf_edge_index];
        }
        outfile << "\n";
    }
    outfile.close();
}

void EBVO::apply_spatial_grid_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, SpatialGrid &spatial_grid, double grid_radius)
{
    //> For each edge in the keyframe, find candidate edges in the current frame using spatial grid
    const std::string &output_dir = dataset.get_output_path();
    size_t frame_idx = 0;
    // Pre-allocate the matching_cf_edges_indices vector to avoid segfault
    KF_CF_edge_pairs.matching_cf_edges_indices.resize(KF_CF_edge_pairs.kf_edges.size());

    // Get the correct edge vector based on is_left flag
    const std::vector<Edge> &kf_edges = KF_CF_edge_pairs.is_left ? kf_edges_left : kf_edges_right;

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < static_cast<int>(KF_CF_edge_pairs.kf_edges.size()); ++i)
    {
        int edge_idx = KF_CF_edge_pairs.kf_edges[i];
        // Bounds check to prevent segfault
        if (edge_idx < 0 || edge_idx >= static_cast<int>(kf_edges.size()))
        {
            std::cerr << "ERROR: Edge index " << edge_idx << " out of bounds. Vector size: " << kf_edges.size() << std::endl;
            continue;
        }

        Edge kf_edge = kf_edges[edge_idx];
        std::vector<int> candidates = spatial_grid.getCandidatesWithinRadius(kf_edge.location, grid_radius);
        KF_CF_edge_pairs.matching_cf_edges_indices[i] = candidates;
    }
}

void EBVO::apply_SIFT_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, double sift_dist_threshold, bool is_left)
{
    //> For each edge in the keyframe, compare its SIFT descriptor with the SIFT descriptors of the edges in the current frame
    //> Filter out edges that don't meet the SIFT distance threshold
    std::string output_dir = dataset.get_output_path();
    size_t frame_idx = 0;
    // Thread-local storage for parallel SIFT distance accumulation
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<double>> thread_local_sift_distances(num_threads);
    std::vector<std::vector<int>> thread_local_is_veridical(num_threads);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < KF_CF_edge_pairs.kf_edges.size(); ++i)
        {
            std::vector<int> filtered_cf_edges_indices;
            std::vector<scores> filtered_scores;
            // Get SIFT descriptors for the keyframe edge (two descriptors per edge)
            int kf_toed_index = KF_CF_edge_pairs.kf_edges[i];
            int kf_stereo_index = KF_CF_edge_pairs.key_frame_pairs->get_Stereo_Edge_Pairs_left_id_index(kf_toed_index);
            std::pair<cv::Mat, cv::Mat> kf_edge_descriptors = KF_CF_edge_pairs.key_frame_pairs->left_edge_descriptors[kf_stereo_index];
            // left and right descriptors should be similar for the same match
            //  Iterate through all matching CF edges for this KF edge
            for (int j = 0; j < KF_CF_edge_pairs.matching_cf_edges_indices[i].size(); ++j)
            {
                int cf_edge_idx = KF_CF_edge_pairs.matching_cf_edges_indices[i][j];
                // Get SIFT descriptors for the current frame edge
                std::pair<cv::Mat, cv::Mat> cf_edge_descriptors = is_left ? current_frame_descriptors_left[cf_edge_idx] : current_frame_descriptors_right[cf_edge_idx];
                scores &score = KF_CF_edge_pairs.matching_scores[i][j];
                if (cf_edge_descriptors.first.empty() || cf_edge_descriptors.second.empty())
                {
                    filtered_cf_edges_indices.push_back(cf_edge_idx); // If no descriptors, keep the edge for now (or could choose to discard)
                    filtered_scores.push_back(score);
                }
                else
                {
                    // Compare all combinations of descriptors (2x2 = 4 combinations)
                    // Take the minimum distance to get the best match
                    double sift_dist_1 = cv::norm(kf_edge_descriptors.first, cf_edge_descriptors.first, cv::NORM_L2);
                    double sift_dist_2 = cv::norm(kf_edge_descriptors.first, cf_edge_descriptors.second, cv::NORM_L2);
                    double sift_dist_3 = cv::norm(kf_edge_descriptors.second, cf_edge_descriptors.first, cv::NORM_L2);
                    double sift_dist_4 = cv::norm(kf_edge_descriptors.second, cf_edge_descriptors.second, cv::NORM_L2);

                    double min_sift_dist = std::min({sift_dist_1, sift_dist_2, sift_dist_3, sift_dist_4});

                    // Check if this edge is veridical (in ground truth)
                    bool is_gt = std::find(KF_CF_edge_pairs.veridical_cf_edges_indices[i].begin(),
                                           KF_CF_edge_pairs.veridical_cf_edges_indices[i].end(),
                                           cf_edge_idx) != KF_CF_edge_pairs.veridical_cf_edges_indices[i].end();
                    thread_local_is_veridical[tid].push_back(is_gt ? 1 : 0);
                    thread_local_sift_distances[tid].push_back(min_sift_dist);

                    // Keep the edge if it passes the threshold
                    if (min_sift_dist < sift_dist_threshold)
                    {
                        filtered_cf_edges_indices.push_back(cf_edge_idx);
                        scores score_refined = score;
                        score_refined.sift_score = min_sift_dist;
                        filtered_scores.push_back(score_refined);
                    }
                }
            }

            // Update the matching indices with filtered results
            KF_CF_edge_pairs.matching_cf_edges_indices[i] = filtered_cf_edges_indices;
            KF_CF_edge_pairs.matching_scores[i] = filtered_scores;
        }
    }

    // Merge thread-local data
    std::vector<double> sift_distances;
    std::vector<int> is_veridical;
    for (int tid = 0; tid < num_threads; ++tid)
    {
        sift_distances.insert(sift_distances.end(), thread_local_sift_distances[tid].begin(), thread_local_sift_distances[tid].end());
        is_veridical.insert(is_veridical.end(), thread_local_is_veridical[tid].begin(), thread_local_is_veridical[tid].end());
    }
}

void EBVO::apply_best_nearly_best_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, double threshold, bool is_NCC)
{
    std::string test_name = is_NCC ? "BNB_NCC" : "BNB_SIFT";

    // Get the correct edge vector based on is_left flag
    const std::vector<Edge> &kf_edges = KF_CF_edge_pairs.is_left ? kf_edges_left : kf_edges_right;

#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(KF_CF_edge_pairs.kf_edges.size()); ++i)
        {

            int edge_idx = KF_CF_edge_pairs.kf_edges[i];
            // Bounds check to prevent segfault
            if (edge_idx < 0 || edge_idx >= static_cast<int>(kf_edges.size()))
            {
                continue;
            }

            Edge left_edge = kf_edges[edge_idx];
            int stereo_idx = KF_CF_edge_pairs.key_frame_pairs->get_Stereo_Edge_Pairs_left_id_index(KF_CF_edge_pairs.kf_edges[i]);
            auto &m_ind = KF_CF_edge_pairs.matching_cf_edges_indices[i];
            size_t num_clusters = m_ind.size();

            if (num_clusters < 2)
                continue;

            // 1. Create an index map to sort clusters based on score without losing original indices
            // Assuming higher score is better (NCC). If SIFT (lower better), flip the comparison logic.
            std::vector<size_t> indices(num_clusters);
            std::iota(indices.begin(), indices.end(), 0);
            if (is_NCC)
                std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
                          { return KF_CF_edge_pairs.matching_scores[i][a].ncc_score > KF_CF_edge_pairs.matching_scores[i][b].ncc_score; });
            else
                std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
                          { return KF_CF_edge_pairs.matching_scores[i][a].sift_score < KF_CF_edge_pairs.matching_scores[i][b].sift_score; });
            // 2. Determine how many candidates pass the recursive ratio test
            size_t keep_count = 1; // Always keep the best

            double best_score = is_NCC ? KF_CF_edge_pairs.matching_scores[i][indices[0]].ncc_score : KF_CF_edge_pairs.matching_scores[i][indices[0]].sift_score;

            for (size_t j = 0; j < num_clusters - 1; ++j)
            {
                double next_score = is_NCC ? KF_CF_edge_pairs.matching_scores[i][indices[j + 1]].ncc_score : KF_CF_edge_pairs.matching_scores[i][indices[j + 1]].sift_score;
                if (best_score == 0)
                    break;
                // Ratio: next_best / current_best
                // If the ratio is high (e.g., 0.9), they are "nearly best."
                // If the ratio is low (e.g., 0.4), the next one is significantly worse.

                double ratio = is_NCC ? next_score / best_score : best_score / next_score;
                if (ratio >= threshold)
                {
                    keep_count++;
                }
                else
                {
                    break; // Significant drop detected, stop including further matches
                }
            }

            // 3. If we aren't keeping everything, rebuild the vectors
            if (keep_count < num_clusters)
            {

                std::vector<int> surviving_cf_matches;
                std::vector<scores> surviving_scores;

                for (size_t k = 0; k < keep_count; ++k)
                {
                    size_t idx = indices[k];
                    surviving_cf_matches.push_back(KF_CF_edge_pairs.matching_cf_edges_indices[i][idx]);
                    surviving_scores.push_back(KF_CF_edge_pairs.matching_scores[i][idx]);
                }
                KF_CF_edge_pairs.matching_cf_edges_indices[i] = std::move(surviving_cf_matches);
                KF_CF_edge_pairs.matching_scores[i] = std::move(surviving_scores);
            }
        }
    }
}

void EBVO::apply_NCC_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, const Stereo_Edge_Pairs &keyframe_stereo, const Stereo_Edge_Pairs &current_stereo, double ncc_val_threshold,
                               const cv::Mat &keyframe_image, const cv::Mat &current_image, bool is_left)
{
    Utility util{};
    //> For each edge in the keyframe, compute the NCC score with the edges in the current frame
    //> Filter out edges that don't meet the NCC threshold

    // Get the appropriate edge vectors based on left/right
    const std::vector<Edge> &kf_edges = is_left ? kf_edges_left
                                                : kf_edges_right;
    const std::vector<Edge> &cf_edges = is_left ? cf_edges_left
                                                : cf_edges_right;

    // Convert images to CV_64F for patch extraction
    cv::Mat kf_image_64f, cf_image_64f;
    keyframe_image.convertTo(kf_image_64f, CV_64F);
    current_image.convertTo(cf_image_64f, CV_64F);

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < KF_CF_edge_pairs.kf_edges.size(); ++i)
    {
        int kf_edge_idx = KF_CF_edge_pairs.kf_edges[i];
        const Edge &kf_edge = kf_edges[kf_edge_idx];

        // Extract patches for the keyframe edge (once per KF edge)
        std::pair<cv::Mat, cv::Mat> kf_edge_patches = util.get_edge_patches(kf_edge, kf_image_64f);

        std::vector<int> filtered_cf_edges_indices;
        std::vector<scores> filtered_sim_scores;
        // Iterate through all matching CF edges for this KF edge
        for (int j = 0; j < KF_CF_edge_pairs.matching_cf_edges_indices[i].size(); ++j)
        {
            int cf_edge_idx = KF_CF_edge_pairs.matching_cf_edges_indices[i][j];
            if (cf_edge_idx >= 0 && cf_edge_idx < cf_edges.size())
            {
                const Edge &cf_edge = cf_edges[cf_edge_idx];
                scores s = KF_CF_edge_pairs.matching_scores[i][j];
                // Extract patches for the current frame edge
                std::pair<cv::Mat, cv::Mat> cf_edge_patches = util.get_edge_patches(cf_edge, cf_image_64f);

                // Calculate the similarity between the KF edge patches and the CF edge patches
                double sim_pp = util.get_patch_similarity(kf_edge_patches.first, cf_edge_patches.first);   //> (A+, B+)
                double sim_nn = util.get_patch_similarity(kf_edge_patches.second, cf_edge_patches.second); //> (A-, B-)
                double sim_pn = util.get_patch_similarity(kf_edge_patches.first, cf_edge_patches.second);  //> (A+, B-)
                double sim_np = util.get_patch_similarity(kf_edge_patches.second, cf_edge_patches.first);  //> (A-, B+)
                double final_SIM_score = std::max({sim_pp, sim_nn, sim_pn, sim_np});

                // Keep the edge if it passes the threshold
                if (final_SIM_score > ncc_val_threshold)
                {
                    filtered_cf_edges_indices.push_back(cf_edge_idx);
                    scores refined_score;
                    refined_score.ncc_score = final_SIM_score;
                    refined_score.sift_score = s.sift_score; // Preserve SIFT score for potential later use
                    filtered_sim_scores.push_back(refined_score);
                }
            }
        }

        // Update the matching indices with filtered results
        KF_CF_edge_pairs.matching_cf_edges_indices[i] = filtered_cf_edges_indices;
        KF_CF_edge_pairs.matching_scores[i] = filtered_sim_scores;
    }
}

void EBVO::min_Edge_Photometric_Residual_by_Gauss_Newton(
    /* inputs */
    Edge left_edge, Eigen::Vector2d init_disp, const cv::Mat &left_image_undistorted,
    const cv::Mat &right_image_undistorted, const cv::Mat &right_image_gradients_x, const cv::Mat &right_image_gradients_y,
    /* outputs */
    Eigen::Vector2d &refined_disparity, double &refined_final_score, double &refined_confidence, bool &refined_validity, std::vector<double> &residual_log,
    /* optional inputs */
    int max_iter, double tol, double huber_delta, bool b_verbose)
{
    cv::Point2d t(std::cos(left_edge.orientation), std::sin(left_edge.orientation));
    cv::Point2d n(-t.y, t.x);
    double side_shift = (PATCH_SIZE / 2.0) + 1.0;
    cv::Point2d c_plus = left_edge.location + n * side_shift;
    cv::Point2d c_minus = left_edge.location - n * side_shift;

    std::vector<cv::Point2d> cLplus, cLminus;
    util_make_rotated_patch_coords(c_plus, left_edge.orientation, cLplus);
    util_make_rotated_patch_coords(c_minus, left_edge.orientation, cLminus);

    std::vector<double> pLplus_f, pLminus_f;
    util_sample_patch_at_coords(left_image_undistorted, cLplus, pLplus_f);
    util_sample_patch_at_coords(left_image_undistorted, cLminus, pLminus_f);
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

    Eigen::Vector2d d = init_disp;
    double init_RMS = 0.0;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        //> Compute the right patch coordinates
        cv::Point2d cRplus = c_plus - cv::Point2d(d[0], d[1]);
        cv::Point2d cRminus = c_minus - cv::Point2d(d[0], d[1]);

        std::vector<cv::Point2d> cRplusC, cRminusC;
        util_make_rotated_patch_coords(cRplus, left_edge.orientation, cRplusC);
        util_make_rotated_patch_coords(cRminus, left_edge.orientation, cRminusC);

        //> Sample right intensities and right gradient X at these coords
        std::vector<double> pRplus_f, pRminus_f, gxRplus_f, gxRminus_f, gyRplus_f, gyRminus_f;
        util_sample_patch_at_coords(right_image_undistorted, cRplusC, pRplus_f);
        util_sample_patch_at_coords(right_image_undistorted, cRminusC, pRminus_f);
        util_sample_patch_at_coords(right_image_gradients_x, cRplusC, gxRplus_f);
        util_sample_patch_at_coords(right_image_gradients_x, cRminusC, gxRminus_f);
        util_sample_patch_at_coords(right_image_gradients_y, cRplusC, gyRplus_f);
        util_sample_patch_at_coords(right_image_gradients_y, cRminusC, gyRminus_f);

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
                Eigen::Vector2d J = -Eigen::Vector2d(gxRf[k], gyRf[k]);
                double absr = std::abs(r);
                double w = (absr > huber_delta) ? 1.0 : huber_delta / absr;

                H += w * J * J.transpose();
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
            refined_confidence = std::exp(-rms / huber_delta);
            break;
        }
        else if (iter == max_iter - 1)
        {
            refined_validity = (is_outlier) ? false : true;
            refined_final_score = rms;
            refined_confidence = 1.0 - (rms / init_RMS); //> optional
        }
    }

    refined_disparity = d;
}

void EBVO::apply_stereo_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs_left, KF_CF_EdgeCorrespondence &KF_CF_edge_pairs_right,
                                  const Stereo_Edge_Pairs &last_keyframe_stereo_left, const Stereo_Edge_Pairs &current_frame_stereo_left,
                                  const Stereo_Edge_Pairs &last_keyframe_stereo_right, const Stereo_Edge_Pairs &current_frame_stereo_right,
                                  size_t frame_idx)
{
    // // Stereo consistency tolerance: edges within this distance are considered matching
    // const double STEREO_SPATIAL_TOLERANCE = 3.0; // pixels

    // // Use pre-built maps for O(1) lookups
    // const std::unordered_map<int, int> &right_edge_to_stereo_idx = current_frame_stereo_right.edge_idx_to_stereo_frame_idx;
    // const std::unordered_map<int, int> &left_edge_to_stereo_idx = current_frame_stereo_left.edge_idx_to_stereo_frame_idx;

    // // we do the same thing to the stereo right edges: use keyframe_stereo.GT_corresponding_edges, apply spatial grid, sift, ncc to filter the right edges
    // std::string output_dir = dataset.get_output_path();
    // std::filesystem::create_directories(output_dir);
    // std::string stereo_filter_filename = output_dir + "/stereo_filtering_debug_frame_" + std::to_string(frame_idx) + ".csv";
    // std::string failed_edges_filename = output_dir + "/stereo_filtering_failed_edges_frame_" + std::to_string(frame_idx) + ".csv";
    // std::ofstream stereo_csv(stereo_filter_filename);
    // std::ofstream failed_edges_csv(failed_edges_filename);

    // // stereo_csv << "examining the 1000th KF-CF pair in left frames\n";ji
    // // std::cout << "Applying stereo filtering on KF-CF edge correspondences..." << std::endl;
    // auto write_edge_pair_to_csv = [&](const std::string &stage, int pair_idx, int edge_index, const EdgeCorrespondenceData &edge_pair, const std::vector<Edge> &kf_edges, const std::vector<Edge> &cf_edges)
    // {
    //     stereo_csv << "kf_edge_index,loc_x,loc_y\n";
    //     stereo_csv << edge_index << ","
    //                << std::fixed << std::setprecision(2) << kf_edges[edge_index].location.x << ","
    //                << std::fixed << std::setprecision(2) << kf_edges[edge_index].location.y << "\n";
    //     stereo_csv << "number of veridical CF edges\n";
    //     stereo_csv << edge_pair.veridical_cf_edges_indices.size() << "\n";
    //     stereo_csv << "number of matching CF edges after " << stage << "\n";
    //     stereo_csv << edge_pair.matching_cf_edges_indices.size() << "\n";

    //     stereo_csv << "veridical_cf_edges_indices\n";
    //     // Write veridical indices
    //     for (size_t i = 0; i < edge_pair.veridical_cf_edges_indices.size(); ++i)
    //     {
    //         if (i > 0)
    //             stereo_csv << ";";
    //         if (i % 10 == 0 && i > 0)
    //             stereo_csv << "\n"; // New line every 10 entries for readability
    //         int edge_n = edge_pair.veridical_cf_edges_indices[i];
    //         stereo_csv << edge_n << ":(" << cf_edges[edge_n].location.x << " " << cf_edges[edge_n].location.y << ")";
    //     }
    //     stereo_csv << "\nmatching_cf_edges_indices\n";

    //     // Write matching indices
    //     for (size_t i = 0; i < edge_pair.matching_cf_edges_indices.size(); ++i)
    //     {
    //         if (i > 0)
    //             stereo_csv << ";";
    //         if (i % 10 == 0 && i > 0)
    //             stereo_csv << "\n"; // New line every 10 entries for readability
    //         int edge_n = edge_pair.matching_cf_edges_indices[i];
    //         stereo_csv << edge_n << ":(" << cf_edges[edge_n].location.x << " " << cf_edges[edge_n].location.y << ")";

    //         // stereo_csv << "(" << edges[edge_pair.matching_cf_edges_indices[i]].location.x << " " << edges[edge_pair.matching_cf_edges_indices[i]].location.y << ")";
    //     }
    //     stereo_csv << "\n";
    // };

    // auto write_stereo_info_to_csv = [&](int edge_index, const std::vector<int> &veridical_edge_indices)
    // {
    //     stereo_csv << "index from CF\n";
    //     stereo_csv << edge_index << "\n";

    //     // Write veridical indices
    //     for (size_t i = 0; i < veridical_edge_indices.size(); ++i)
    //     {
    //         if (i > 0)
    //             stereo_csv << ";";
    //         stereo_csv << veridical_edge_indices[i]; // New line every 10 entries for readability
    //     }

    //     stereo_csv << "\n";
    // };
    // try
    // {
    //     int i = 0;

    //     // Write header to failed edges CSV
    //     failed_edges_csv << "kf_edge_index,cf_edge_index,kf_location_x,kf_location_y,kf_orientation,cf_location_x,cf_location_y,cf_orientation,gt_location_x,gt_location_y,num_veridical_edges,num_matched_edges_before_stereo\n";

    //     for (auto it = KF_CF_edge_pairs_left.begin(); it != KF_CF_edge_pairs_left.end(); ++it)
    //     {
    //         // A- -> B-
    //         std::unordered_set<int> stereo_left_to_right_mapping;
    //         int left_kf_edge_index = it->first;         // represents the 3rd order edge index in the keyframe left image
    //         bool debug = (left_kf_edge_index == 49863); // debug for a specific edge
    //         EdgeCorrespondenceData *left_pair_it = &it->second;
    //         if (debug)
    //         {
    //             write_edge_pair_to_csv("initial", i, it->first, it->second, kf_edges_left, cf_edges_left);
    //         }

    //         // for each left edge in the keyframe, we find the closest corresponding right stereo edge
    //         // since the cloest doesnt have best accuracy, we will consider all the verdical right edges
    //         int kf_veridical_size = last_keyframe_stereo_left.GT_corresponding_edges[left_pair_it->stereo_frame_idx].size();
    //         for (int j = 0; j < kf_veridical_size; j++)
    //         {
    //             // get B-
    //             int left_kf_veridical_edge_index = last_keyframe_stereo_left.GT_corresponding_edges[left_pair_it->stereo_frame_idx][j];
    //             std::vector<int> &matching_cf_right = KF_CF_edge_pairs_right[left_kf_veridical_edge_index].matching_cf_edges_indices;

    //             for (const auto &right_cf_edge_index : matching_cf_right)
    //             {
    //                 stereo_left_to_right_mapping.insert(right_cf_edge_index); // insert all B+
    //             }
    //         }
    //         // now we go from A- to A+
    //         //  Build spatial index of stereo mates for proximity matching
    //         std::vector<cv::Point2d> stereo_mate_locations;
    //         stereo_mate_locations.reserve(stereo_left_to_right_mapping.size());
    //         for (int stereo_mate_idx : stereo_left_to_right_mapping)
    //         {
    //             stereo_mate_locations.push_back(cf_edges_right[stereo_mate_idx].location);
    //         }

    //         // std::cout << "Number of mapped right edges from left edges: " << stereo_left_to_right_mapping.size() << std::endl;
    //         std::vector<int> filtered_cf_left_edges;
    //         // std::vector<int> failed_edges; // Track edges that don't have veridical candidates
    //         for (const auto &left_cf_edge_index : left_pair_it->matching_cf_edges_indices)
    //         {
    //             // Check if edge has stereo GT
    //             if (left_edge_to_stereo_idx.find(left_cf_edge_index) == left_edge_to_stereo_idx.end())
    //             {
    //                 // No stereo GT (could be occluded) - keep it to be safe
    //                 filtered_cf_left_edges.push_back(left_cf_edge_index);
    //                 continue;
    //             }

    //             // Check for spatial proximity match instead of exact index match
    //             // we got A+, we then go find B+
    //             int stereo_idx = left_edge_to_stereo_idx.at(left_cf_edge_index);
    //             const std::vector<int> &right_cf_edge_veridical_indices = current_frame_stereo_left.GT_corresponding_edges[stereo_idx];

    //             bool any_b_plus_matches = false;
    //             for (int k = 0; k < right_cf_edge_veridical_indices.size(); k++)
    //             {
    //                 int veridical_right_cf_edge_index = right_cf_edge_veridical_indices[k];
    //                 const cv::Point2d &right_ver_edge_loc = cf_edges_right[veridical_right_cf_edge_index].location;

    //                 for (const cv::Point2d &stereo_mate_loc : stereo_mate_locations)
    //                 {
    //                     double dist = cv::norm(right_ver_edge_loc - stereo_mate_loc);
    //                     if (dist < STEREO_SPATIAL_TOLERANCE)
    //                     {
    //                         any_b_plus_matches = true;
    //                         break;
    //                     }
    //                 }

    //                 if (any_b_plus_matches)
    //                 {
    //                     break; // Found a match, no need to check other B+ mates
    //                 }
    //             }

    //             if (any_b_plus_matches)
    //             {
    //                 filtered_cf_left_edges.push_back(left_cf_edge_index);
    //             }
    //         }
    //         left_pair_it->matching_cf_edges_indices = filtered_cf_left_edges;
    //         if (debug)
    //         {
    //             write_edge_pair_to_csv("stereo_filtering", i, left_kf_edge_index, *left_pair_it, kf_edges_left, cf_edges_left);
    //         }

    //         i++;
    //     }

    //     stereo_csv.close();
    //     failed_edges_csv.close();
    //     std::cout << "Stereo filtering debug data saved to: " << stereo_filter_filename << std::endl;
    //     std::cout << "Failed edges data saved to: " << failed_edges_filename << std::endl;
    // }
    // catch (const std::exception &e)
    // {
    //     std::cerr << "Exception in apply_stereo_filtering: " << e.what() << std::endl;
    //     throw;
    // }
    // catch (...)
    // {
    //     std::cerr << "Unknown exception in apply_stereo_filtering" << std::endl;
    //     throw;
    // }
}

void EBVO::debug_veridical(int edge_idx, const KF_CF_EdgeCorrespondence &KF_CF_edge_pairs_left, const KF_CF_EdgeCorrespondence &KF_CF_edge_pairs_right, const Stereo_Edge_Pairs &keyframe_stereo, const Stereo_Edge_Pairs &current_stereo_left, const Stereo_Edge_Pairs &current_stereo_right, bool is_left)
{
    // std::string output_dir = dataset.get_output_path();
    // std::string debug_filename = output_dir + "/veridical_debug_" + (is_left ? "left" : "right") + ".csv";
    // std::ofstream debug_file(debug_filename);
    // auto it = KF_CF_edge_pairs_left.find(edge_idx);
    // if (it != KF_CF_edge_pairs_left.end())
    // {
    //     const EdgeCorrespondenceData &edge_data = it->second;
    //     debug_file << "(A-)KF Edge Index,Location_X,Location_Y,Orientation\n";
    //     const Edge &kf_edge = is_left ? kf_edges_left[edge_idx]
    //                                   : kf_edges_right[edge_idx];
    //     debug_file << edge_idx << ","
    //                << std::fixed << std::setprecision(2) << kf_edge.location.x << ","
    //                << std::fixed << std::setprecision(2) << kf_edge.location.y << ","
    //                << std::fixed << std::setprecision(2) << kf_edge.orientation << "\n";

    //     // Output A- to A+ GT projection location
    //     debug_file << "(A- to A+) left_gt_projected_location_on_cf_x,left_gt_projected_location_on_cf_y,left_gt_orientation_on_cf\n";
    //     debug_file << std::fixed << std::setprecision(2) << edge_data.gt_location_on_cf.x << ","
    //                << std::fixed << std::setprecision(2) << edge_data.gt_location_on_cf.y << ","
    //                << std::fixed << std::setprecision(2) << edge_data.gt_orientation_on_cf << "\n";

    //     // now going from B- to B+
    //     int size = keyframe_stereo.GT_corresponding_edges[edge_data.stereo_frame_idx].size();
    //     for (int i = 0; i < size; ++i)
    //     {
    //         debug_file << "(B-)GT Veridical Stereo KF Edges:\n";
    //         debug_file << "Index,Location_X,Location_Y,Orientation\n";
    //         int veridical_kf_edge_idx = keyframe_stereo.GT_corresponding_edges[edge_data.stereo_frame_idx][i];
    //         const Edge &veridical_kf_edge = kf_edges_right[veridical_kf_edge_idx];
    //         debug_file << veridical_kf_edge_idx << ","
    //                    << std::fixed << std::setprecision(2) << veridical_kf_edge.location.x << ","
    //                    << std::fixed << std::setprecision(2) << veridical_kf_edge.location.y << ","
    //                    << std::fixed << std::setprecision(2) << veridical_kf_edge.orientation << "\n";

    //         auto it_cf = KF_CF_edge_pairs_right.find(veridical_kf_edge_idx);
    //         if (it_cf != KF_CF_edge_pairs_right.end())
    //         {
    //             debug_file << "(B- to B+) right_gt_projected_location_on_cf_x,right_gt_projected_location_on_cf_y,right_gt_orientation_on_cf\n";
    //             debug_file << std::fixed << std::setprecision(2) << it_cf->second.gt_location_on_cf.x << ","
    //                        << std::fixed << std::setprecision(2) << it_cf->second.gt_location_on_cf.y << ","
    //                        << std::fixed << std::setprecision(2) << it_cf->second.gt_orientation_on_cf << "\n";
    //             debug_file << "(B+)GT Veridical Stereo CF Edges:\n";
    //             debug_file << "Index,Location_X,Location_Y,Orientation\n";
    //             const EdgeCorrespondenceData &edge_data_cf = it_cf->second;
    //             for (const auto &veridical_cf_edge_idx : edge_data_cf.veridical_cf_edges_indices)
    //             {
    //                 const Edge &veridical_cf_edge = cf_edges_right[veridical_cf_edge_idx];
    //                 debug_file << veridical_cf_edge_idx << ","
    //                            << std::fixed << std::setprecision(2) << veridical_cf_edge.location.x << ","
    //                            << std::fixed << std::setprecision(2) << veridical_cf_edge.location.y << ","
    //                            << std::fixed << std::setprecision(2) << veridical_cf_edge.orientation << "\n";
    //             }
    //         }
    //     }
    //     // now going from A- to A+

    //     for (const auto &veridical_cf_edge_idx : edge_data.veridical_cf_edges_indices)
    //     {
    //         debug_file << "(A+)GT Veridical CF Edges:\n";
    //         debug_file << "Index,Location_X,Location_Y,Orientation\n";
    //         const Edge &veridical_cf_edge = is_left ? cf_edges_left[veridical_cf_edge_idx]
    //                                                 : cf_edges_right[veridical_cf_edge_idx];
    //         debug_file << veridical_cf_edge_idx << ","
    //                    << std::fixed << std::setprecision(2) << veridical_cf_edge.location.x << ","
    //                    << std::fixed << std::setprecision(2) << veridical_cf_edge.location.y << ","
    //                    << std::fixed << std::setprecision(2) << veridical_cf_edge.orientation << "\n";
    //         debug_file << "(B+) 2nd GT Veridical Stereo CF Edges:\n";
    //         debug_file << "Index,Location_X,Location_Y,Orientation\n";

    //         // Check if edge exists in stereo GT map
    //         auto map_it = current_stereo_left.edge_idx_to_stereo_frame_idx.find(veridical_cf_edge_idx);
    //         if (map_it == current_stereo_left.edge_idx_to_stereo_frame_idx.end())
    //         {
    //             // Edge doesn't have stereo GT (occluded or invalid disparity)
    //             continue;
    //         }

    //         int stereo_frame_idx_left = map_it->second;
    //         int c_size = current_stereo_left.GT_corresponding_edges[stereo_frame_idx_left].size();
    //         for (int j = 0; j < c_size; ++j)
    //         {
    //             int veridical_cf_right_edge_idx = current_stereo_left.GT_corresponding_edges[stereo_frame_idx_left][j];
    //             const Edge &veridical_cf_edge = cf_edges_right[veridical_cf_right_edge_idx];
    //             debug_file << veridical_cf_right_edge_idx << ","
    //                        << std::fixed << std::setprecision(2) << veridical_cf_edge.location.x << ","
    //                        << std::fixed << std::setprecision(2) << veridical_cf_edge.location.y << ","
    //                        << std::fixed << std::setprecision(2) << veridical_cf_edge.orientation << "\n";
    //         }
    //     }
    // }
    // debug_file.close();
}

void EBVO::apply_orientation_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, double orientation_threshold, bool is_left)
{
    //> for each edge in the keyframe, compare its orientation with the edges in the current frame

    // Get the correct edge vectors based on is_left flag
    const std::vector<Edge> &kf_edges = KF_CF_edge_pairs.is_left ? kf_edges_left : kf_edges_right;
    const std::vector<Edge> &cf_edges = KF_CF_edge_pairs.is_left ? cf_edges_left : cf_edges_right;

#pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < KF_CF_edge_pairs.kf_edges.size(); ++idx)
    {
        int kf_edge_idx = KF_CF_edge_pairs.kf_edges[idx];
        // Bounds check
        if (kf_edge_idx < 0 || kf_edge_idx >= static_cast<int>(kf_edges.size()))
        {
            continue;
        }

        Edge kf_edge = kf_edges[kf_edge_idx];

        std::vector<int> filtered_cf_edges_indices;
        for (auto &m_edge_idx : KF_CF_edge_pairs.matching_cf_edges_indices[idx])
        {
            // Bounds check for CF edge
            if (m_edge_idx < 0 || m_edge_idx >= static_cast<int>(cf_edges.size()))
            {
                continue;
            }

            Edge cf_edge = cf_edges[m_edge_idx];
            double orientation_diff = std::abs(rad_to_deg<double>(kf_edge.orientation - cf_edge.orientation));
            if (orientation_diff > 180.0)
            {
                orientation_diff = 360.0 - orientation_diff;
            }

            // Keep edge if orientation is similar (< threshold) OR opposite (~180°)
            if (orientation_diff < orientation_threshold ||
                std::abs(orientation_diff - 180.0) < orientation_threshold)
            {
                filtered_cf_edges_indices.push_back(m_edge_idx);
            }
        }
        KF_CF_edge_pairs.matching_cf_edges_indices[idx] = filtered_cf_edges_indices;
    }
}

void EBVO::Evaluate_KF_CF_Edge_Correspondences(const KF_CF_EdgeCorrespondence &KF_CF_edge_pairs,
                                               Stereo_Edge_Pairs &keyframe_stereo, Stereo_Edge_Pairs &current_stereo,
                                               size_t frame_idx, const std::string &stage_name)
{
    int total_num_of_true_positives_for_recall = 0;
    int total_num_of_true_positives_for_precision = 0;
    std::vector<double> num_of_target_edges_per_source_edge;
    std::vector<double> precision_per_edge;
    std::vector<double> precision_pair_edge;

    //> Do we need this?
    int kf_edge_count = 0;
    if (KF_CF_edge_pairs.is_left)
    {
        kf_edge_count = KF_CF_edge_pairs.kf_edges.size();
    }
    else
    {
        // Only count right edges that actually have a temporal Ground Truth AND are stereo-veridical
        for (int kf_edge_idx : KF_CF_edge_pairs.kf_edges)
        {
            if (kf_right_eval[kf_edge_idx])
            {
                kf_edge_count++;
            }
        }
    }

    num_of_target_edges_per_source_edge.reserve(kf_edge_count); //> compailable to both left and right edges
    precision_per_edge.reserve(kf_edge_count);

    for (int i = 0; i < KF_CF_edge_pairs.kf_edges.size(); ++i)
    {
        if (!KF_CF_edge_pairs.is_left && !kf_right_eval[i])
            continue;
        bool fail = true;
        total_num_of_true_positives_for_precision = 0;
        //> Find if there is at least one edge index in edge_pair.matching_cf_edges_indices is found in edge_pair.veridical_cf_edges_indices
        if (KF_CF_edge_pairs.matching_cf_edges_indices[i].size() > 0)
        {
            int total_num_of_candidates = KF_CF_edge_pairs.matching_cf_edges_indices[i].size();

            // Get the appropriate edge vector based on left/right
            const std::vector<Edge> &cf_edges = KF_CF_edge_pairs.is_left
                                                    ? cf_edges_left
                                                    : cf_edges_right;
            for (const auto &matched_idx : KF_CF_edge_pairs.matching_cf_edges_indices[i])
            {
                // matched_idx is a TOED edge index, access the edge directly
                if (matched_idx >= 0 && matched_idx < cf_edges.size())
                {
                    const Edge &cf_edge = cf_edges[matched_idx];
                    if (cv::norm(cf_edge.location - KF_CF_edge_pairs.gt_location_on_cf[i]) < 1.0)
                    {
                        total_num_of_true_positives_for_precision++;
                    }
                }
            }
            if (total_num_of_true_positives_for_precision > 0)
            {
                total_num_of_true_positives_for_recall++;
            }
            else
            {
            }
            precision_per_edge.push_back(static_cast<double>(total_num_of_true_positives_for_precision) / static_cast<double>(total_num_of_candidates));
            precision_pair_edge.push_back(static_cast<double>(total_num_of_true_positives_for_precision) / static_cast<double>(total_num_of_candidates));
            num_of_target_edges_per_source_edge.push_back(total_num_of_candidates);
        }
    }

    double recall_per_image = static_cast<double>(total_num_of_true_positives_for_recall) / kf_edge_count;
    double precision_per_image = std::accumulate(precision_per_edge.begin(), precision_per_edge.end(), 0.0) / precision_per_edge.size();
    double precision_pair_per_image = std::accumulate(precision_pair_edge.begin(), precision_pair_edge.end(), 0.0) / precision_pair_edge.size();
    double num_of_target_edges_per_source_edge_avg = std::accumulate(num_of_target_edges_per_source_edge.begin(), num_of_target_edges_per_source_edge.end(), 0.0) / num_of_target_edges_per_source_edge.size();

    std::cout << "Stereo Edge Correspondences Evaluation: Stage: " << stage_name << " | Frame: " << frame_idx << std::endl;
    std::cout << "- Recall rate:       " << std::fixed << std::setprecision(8) << recall_per_image << std::endl;
    std::cout << "- Precision rate:    " << std::fixed << std::setprecision(8) << precision_per_image << std::endl;
    std::cout << "- Average ambiguity: " << std::fixed << std::setprecision(8) << num_of_target_edges_per_source_edge_avg << std::endl;
    std::cout << "========================================================\n"
              << std::endl;
}

void EBVO::augment_all_Edge_Data(Stereo_Edge_Pairs &stereo_frame_edge_pairs, std::vector<std::pair<cv::Mat, cv::Mat>> &edge_descriptors, bool is_left)
{
    Utility util{};
    edge_descriptors.clear();
    int size = is_left ? stereo_frame_edge_pairs.focused_edge_indices.size() : cf_edges_right.size();

    // Pre-fill with empty Mats so dropped keypoints are handled safely
    edge_descriptors.resize(size, std::make_pair(cv::Mat(), cv::Mat()));

    const cv::Mat &image = is_left ? stereo_frame_edge_pairs.stereo_frame->left_image_undistorted
                                   : stereo_frame_edge_pairs.stereo_frame->right_image_undistorted;

    // 1. Gather all valid keypoints into a single vector
    std::vector<cv::KeyPoint> all_kps;
    all_kps.reserve(size * 2);

    for (int i = 0; i < size; ++i)
    {
        Edge le = is_left ? stereo_frame_edge_pairs.get_focused_edge_by_toed_index(i) : cf_edges_right[i];

        // Safety check to prevent OpenCV crashing on invalid math
        if (std::isnan(le.location.x) || std::isnan(le.orientation))
            continue;

        std::pair<cv::Point2d, cv::Point2d> shifted_points = util.get_Orthogonal_Shifted_Points(le, 8);

        float angle = 180.0f / M_PI * le.orientation;

        // We embed the original edge index 'i' into class_id.
        // class_id = (i * 2) for the first point, (i * 2 + 1) for the second.
        all_kps.emplace_back(shifted_points.first, 1.0f, angle, 0, 0, i * 2);
        all_kps.emplace_back(shifted_points.second, 1.0f, angle, 0, 0, i * 2 + 1);
    }

    // 2. Compute EVERYTHING in one single, safe OpenCV call
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::Mat all_descriptors;

    // OpenCV will safely multi-thread this internally! No OpenMP needed.
    sift->compute(image, all_kps, all_descriptors);

    // 3. Unpack the batch results back into your edge_descriptors array
    // Note: If OpenCV drops an out-of-bounds keypoint, it naturally removes it
    // from all_kps. Our class_id check ensures we still map them to the correct edge!
    for (int j = 0; j < all_kps.size(); ++j)
    {
        int original_id = all_kps[j].class_id;
        int edge_index = original_id / 2;
        int point_index = original_id % 2;

        if (point_index == 0)
        {
            edge_descriptors[edge_index].first = all_descriptors.row(j).clone();
        }
        else
        {
            edge_descriptors[edge_index].second = all_descriptors.row(j).clone();
        }
    }
}