#include <filesystem>
#include <unordered_set>
#include <numeric>
#include <cmath>
#include <omp.h>
#include "EBVO.h"
#include "Dataset.h"
#include "Matches.h"
#include <opencv2/core/eigen.hpp>

EBVO::EBVO(YAML::Node config_map) : dataset(config_map) {}

//> MARK: MAIN CODE OF EDGE VO
void EBVO::PerformEdgeBasedVO()
{
    // The main processing function that performs edge-based visual odometry on a sequence of stereo image pairs.
    // It loads images, detects edges, matches them, and evaluates the matching performance.
    int num_pairs = 100000;
    std::vector<std::pair<cv::Mat, cv::Mat>> image_pairs;
    std::vector<cv::Mat> left_ref_disparity_maps;
    // std::vector<cv::Mat> right_ref_disparity_maps;
    std::vector<double> max_disparity_values;

    std::vector<double> per_image_avg_before_epi;
    std::vector<double> per_image_avg_after_epi;

    std::vector<double> per_image_avg_before_disp;
    std::vector<double> per_image_avg_after_disp;

    std::vector<double> per_image_avg_before_shift;
    std::vector<double> per_image_avg_after_shift;

    std::vector<double> per_image_avg_before_clust;
    std::vector<double> per_image_avg_after_clust;

    std::vector<double> per_image_avg_before_patch;
    std::vector<double> per_image_avg_after_patch;

    std::vector<double> per_image_avg_before_ncc;
    std::vector<double> per_image_avg_after_ncc;

    std::vector<double> per_image_avg_before_lowe;
    std::vector<double> per_image_avg_after_lowe;

    std::vector<double> per_image_avg_before_bct;
    std::vector<double> per_image_avg_after_bct;

    std::vector<RecallMetrics> all_forward_recall_metrics;
    std::vector<BidirectionalMetrics> all_bct_metrics;

    //> load stereo iterator and read disparity
    dataset.load_dataset(dataset.get_dataset_type(), left_ref_disparity_maps, num_pairs);

    std::vector<double> left_intr = dataset.left_intr();
    std::vector<double> right_intr = dataset.right_intr();

    cv::Mat left_calib = (cv::Mat_<double>(3, 3) << left_intr[0], 0, left_intr[2], 0, left_intr[1], left_intr[3], 0, 0, 1);
    cv::Mat right_calib = (cv::Mat_<double>(3, 3) << right_intr[0], 0, right_intr[2], 0, right_intr[1], right_intr[3], 0, 0, 1);

    cv::Mat left_dist_coeff_mat = (cv::Mat_<double>(1, 4) << dataset.left_dist_coeffs()[0], dataset.left_dist_coeffs()[1], dataset.left_dist_coeffs()[2], dataset.left_dist_coeffs()[3]);
    cv::Mat right_dist_coeff_mat = (cv::Mat_<double>(1, 4) << dataset.right_dist_coeffs()[0], dataset.right_dist_coeffs()[1], dataset.right_dist_coeffs()[2], dataset.right_dist_coeffs()[3]);

    auto start_time = std::chrono::high_resolution_clock::now();

    LOG_INFO("Start looping over all image pairs");
    // now we change the logic, we will do previous frame and current frame instead
    StereoFrame previous_frame, current_frame;
    StereoFrame last_keyframe;
    //> Initialize
    StereoEdgeCorrespondencesGT last_keyframe_stereo_left, current_frame_stereo_left;
    StereoEdgeCorrespondencesGT last_keyframe_stereo_right, current_frame_stereo_right;

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    bool b_is_keyframe = true;

    size_t frame_idx = 0;
    while (dataset.stereo_iterator->hasNext() && num_pairs - frame_idx >= 0)
    {
        if (!dataset.stereo_iterator->getNext(current_frame))
        {
            break;
        }

        // If the current frame has ground truth, we can use it (current development: using GT for evaluation)
        if (dataset.has_gt())
        {
        }
        const cv::Mat &left_disparity_map = (frame_idx < left_ref_disparity_maps.size()) ? left_ref_disparity_maps[frame_idx] : cv::Mat();

        dataset.ncc_one_vs_err.clear();
        dataset.ncc_two_vs_err.clear();
        dataset.ground_truth_right_edges_after_lowe.clear();

        std::cout << "Image Pair #" << frame_idx << "\n";

        cv::Mat left_prev_undistorted, right_prev_undistorted, left_cur_undistorted, right_cur_undistorted;

        cv::undistort(current_frame.left_image, left_cur_undistorted, left_calib, left_dist_coeff_mat);
        cv::undistort(current_frame.right_image, right_cur_undistorted, right_calib, right_dist_coeff_mat);

        current_frame.left_image_undistorted = left_cur_undistorted;
        current_frame.right_image_undistorted = right_cur_undistorted;

        if (dataset.get_num_imgs() == 0)
        {
            dataset.set_height(left_cur_undistorted.rows);
            dataset.set_width(left_cur_undistorted.cols);

            TOED = std::shared_ptr<ThirdOrderEdgeDetectionCPU>(new ThirdOrderEdgeDetectionCPU(dataset.get_height(), dataset.get_width()));

            // Initialize the spatial grids with a cell size of defined GRID_SIZE
            left_grid = SpatialGrid(dataset.get_width(), dataset.get_height(), GRID_SIZE);
            right_grid = SpatialGrid(dataset.get_width(), dataset.get_height(), GRID_SIZE);
        }

        std::string edge_dir = dataset.get_output_path() + "/edges";
        std::filesystem::create_directories(edge_dir);

        std::string left_edge_path = edge_dir + "/left_edges_" + std::to_string(frame_idx + 1);
        std::string right_edge_path = edge_dir + "/right_edges_" + std::to_string(frame_idx + 1);
        // getting toed edges for left and right images
        ProcessEdges(left_cur_undistorted, left_edge_path, TOED, dataset.left_edges);
        std::cout << "Number of edges on the left image: " << dataset.left_edges.size() << std::endl;
        std::string third_order_edges_file = dataset.get_output_path() + "/third_order_edges_frame_" + std::to_string(frame_idx + 1) + ".txt";
        WriteThirdOrderEdgesToFile(frame_idx, third_order_edges_file);
        std::cout << "Wrote third-order edges of frame " << frame_idx << " to: " << third_order_edges_file << std::endl;
        ProcessEdges(right_cur_undistorted, right_edge_path, TOED, dataset.right_edges);
        std::cout << "Number of edges on the right image: " << dataset.right_edges.size() << std::endl;

        dataset.increment_num_imgs();

        if (b_is_keyframe)
        {
            //> Update last keyframe
            last_keyframe = current_frame;
            kf_edges_left = dataset.left_edges;
            kf_edges_right = dataset.right_edges;

            //> For each left edge, get the corresponding GT location (not right edge) on the right image, and the triangulated 3D point in the left camera coordinate
            Find_Stereo_GT_Locations(left_disparity_map, true, last_keyframe_stereo_left, kf_edges_left);
            std::cout << "Complete calculating GT locations for left edges of the keyframe (previous frame)..." << std::endl;
            Find_Stereo_GT_Locations(left_disparity_map, false, last_keyframe_stereo_right, kf_edges_right);
            std::cout << "Complete calculating GT locations for right edges of the keyframe (previous frame)..." << std::endl;

            //> Construct a GT stereo edge pool
            get_Stereo_Edge_GT_Pairs(dataset, last_keyframe_stereo_left, kf_edges_right, true);
            std::cout << "Size of stereo edge correspondences pool = " << last_keyframe_stereo_left.focused_edges.size() << std::endl;
            get_Stereo_Edge_GT_Pairs(dataset, last_keyframe_stereo_right, kf_edges_left, false);
            std::cout << "Size of stereo edge correspondences pool = " << last_keyframe_stereo_right.focused_edges.size() << std::endl;

            //> construct stereo edge correspondences

            //> extract SIFT descriptor for each left and right edge of last_keyframe_stereo
            augment_Edge_Data(last_keyframe_stereo_left, last_keyframe.left_image_undistorted, true);
            augment_Edge_Data(last_keyframe_stereo_right, last_keyframe.right_image_undistorted, false);

            if (!last_keyframe_stereo_left.b_is_size_consistent())
                last_keyframe_stereo_left.print_size_consistency();
            if (!last_keyframe_stereo_right.b_is_size_consistent())
                last_keyframe_stereo_right.print_size_consistency();
            b_is_keyframe = false;
        }
        else
        {
            cf_edges_left = dataset.left_edges;
            cf_edges_right = dataset.right_edges;
            //> For each left edge, get the corresponding GT location (not right edge) on the right image, and the triangulated 3D point in the left camera coordinate
            Find_Stereo_GT_Locations(left_disparity_map, true, current_frame_stereo_left, cf_edges_left);
            std::cout << "Complete calculating GT locations for left edges of the current frame..." << std::endl;
            Find_Stereo_GT_Locations(left_disparity_map, false, current_frame_stereo_right, cf_edges_right);
            std::cout << "Complete calculating GT locations for right edges of the current frame..." << std::endl;

            //> Construct a GT stereo edge pool
            get_Stereo_Edge_GT_Pairs(dataset, current_frame_stereo_left, cf_edges_right, true);
            std::cout << "Size of stereo edge correspondences pool for left edges= " << current_frame_stereo_left.focused_edges.size() << std::endl;
            get_Stereo_Edge_GT_Pairs(dataset, current_frame_stereo_right, cf_edges_left, false);
            std::cout << "Size of stereo edge correspondences pool for right edges= " << current_frame_stereo_right.focused_edges.size() << std::endl;

            //> extract SIFT descriptor for each left edge of current_frame_stereo
            augment_Edge_Data(current_frame_stereo_left, current_frame.left_image_undistorted, true);
            augment_Edge_Data(current_frame_stereo_right, current_frame.right_image_undistorted, false);

            if (!current_frame_stereo_left.b_is_size_consistent())
                current_frame_stereo_left.print_size_consistency();
            if (!current_frame_stereo_right.b_is_size_consistent())
                current_frame_stereo_right.print_size_consistency();

            add_edges_to_spatial_grid(current_frame_stereo_left, left_grid, cf_edges_left);
            add_edges_to_spatial_grid(current_frame_stereo_right, right_grid, cf_edges_right);
            //> Construct correspondences structure between last keyframe and the current frame
            KF_CF_EdgeCorrespondenceMap KF_CF_edge_pairs_left, KF_CF_edge_pairs_right;
            Find_Veridical_Edge_Correspondences_on_CF(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, last_keyframe, current_frame, left_grid, true);
            Find_Veridical_Edge_Correspondences_on_CF(KF_CF_edge_pairs_right, last_keyframe_stereo_right, current_frame_stereo_right, last_keyframe, current_frame, right_grid, false);

            std::cout << "Size of veridical edge pairs (left) = " << KF_CF_edge_pairs_left.size() << std::endl;
            std::cout << "Size of veridical edge pairs (right) = " << KF_CF_edge_pairs_right.size() << std::endl;

            //> Now that the GT edge correspondences are constructed between the keyframe and the current frame, we can apply various filters from the beginning
            //> Stage 1: Apply spatial grid to the current frame
            apply_spatial_grid_filtering(KF_CF_edge_pairs_left, last_keyframe_stereo_left, kf_edges_left, left_grid, 1.0);
            apply_spatial_grid_filtering(KF_CF_edge_pairs_right, last_keyframe_stereo_right, kf_edges_right, right_grid, 1.0);

            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "Spatial Grid");
            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_right, last_keyframe_stereo_right, current_frame_stereo_right, frame_idx, "Spatial Grid");

            //> Stage 3: Do NCC
            apply_NCC_filtering(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, 0.25, last_keyframe.left_image, current_frame.left_image, true);
            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "NCC Filtering");

            apply_NCC_filtering(KF_CF_edge_pairs_right, last_keyframe_stereo_right, current_frame_stereo_right, 0.25, last_keyframe.right_image, current_frame.right_image, false);
            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_right, last_keyframe_stereo_right, current_frame_stereo_right, frame_idx, "NCC Filtering");
            //> Stage 2: Do SIFT descriptor comparison between the last keyframe and the current frame from the KC_edge_correspondences
            apply_SIFT_filtering(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, 700.0);
            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "SIFT Filtering");

            apply_SIFT_filtering(KF_CF_edge_pairs_right, last_keyframe_stereo_right, current_frame_stereo_right, 700.0, false);
            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_right, last_keyframe_stereo_right, current_frame_stereo_right, frame_idx, "SIFT Filtering");

            //> Stage 4: Stereo consistency filtering
            apply_stereo_filtering(KF_CF_edge_pairs_left, KF_CF_edge_pairs_right,
                                   last_keyframe_stereo_left, current_frame_stereo_left,
                                   last_keyframe_stereo_right, current_frame_stereo_right,
                                   frame_idx);
            Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "Stereo Consistency Filtering");
            // Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_right, last_keyframe_stereo_right, current_frame_stereo_right, frame_idx, "Stereo Consistency Filtering");

            // StereoMatchResult match_result = get_Stereo_Edge_Pairs(
            //     left_cur_undistorted,
            //     right_cur_undistorted,
            //     last_keyframe_stereo_left,
            //     dataset, frame_idx);

            break;
        }

        frame_idx++;
        if (frame_idx > 1)
        {
            break;
        }

        // previous_frame = current_frame;
        // previous_edge_loc is now updated in the frame_idx > 0 block above
    }
}
// void EBCO::bidirection_check(){

// }
void EBVO::augment_Edge_Data(StereoEdgeCorrespondencesGT &stereo_frame, const cv::Mat image, bool is_left)
{
    //> Pre-allocate thread-local storage based on number of threads
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<cv::KeyPoint>> thread_local_keypoints(num_threads);
    std::vector<cv::KeyPoint> edge_keypoints;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
#pragma omp for schedule(dynamic)
        for (const auto &i : stereo_frame.focused_edges)
        {
            const Edge &le = is_left ? dataset.left_edges[i] : dataset.right_edges[i];
            cv::KeyPoint edge_kp(le.location, 1, 180 / M_PI * le.orientation);
            thread_local_keypoints[thread_id].push_back(edge_kp);
        }
    }

    //> Merge all thread-local keypoints
    for (const auto &local_keypoints : thread_local_keypoints)
    {
        edge_keypoints.insert(edge_keypoints.end(), local_keypoints.begin(), local_keypoints.end());
    }

    //> Compute SIFT descriptors for all left edges
    cv::Mat edge_desc;
    sift->compute(image, edge_keypoints, edge_desc);

    //> Loop over all the rows of edge_desc and assign each to the stereo_frame.edge_descriptors
    for (int i = 0; i < edge_desc.rows; ++i)
    {
        stereo_frame.edge_descriptors.push_back(edge_desc.row(i));
    }

    // std::cout << "Descriptors matrix size: " << edge_desc.rows << " x " << edge_desc.cols << std::endl;
}

void EBVO::add_edges_to_spatial_grid(StereoEdgeCorrespondencesGT &stereo_frame, SpatialGrid &spatial_grid, const std::vector<Edge> &edges)
{
    //> Add left edges to spatial grid. This is done on the current image only.
    // TODO: check if this can be done in parallel for faster computation
    for (int i = 0; i < stereo_frame.focused_edges.size(); ++i)
    {
        int grid_index = spatial_grid.add_edge_to_grids(stereo_frame.focused_edges[i], edges[stereo_frame.focused_edges[i]].location);
        stereo_frame.grid_indices.push_back(grid_index);
    }
}

void EBVO::ProcessEdges(const cv::Mat &image,
                        const std::string &filepath,
                        std::shared_ptr<ThirdOrderEdgeDetectionCPU> &toed,
                        std::vector<Edge> &edges)
{
    std::string path = filepath + ".bin";

    // if (std::filesystem::exists(path))
    // {
    //     // std::cout << "Loading edge data from: " << path << std::endl;
    //     ReadEdgesFromBinary(path, edges);
    // }
    // else
    // {
    std::cout << "Running third-order edge detector..." << std::endl;
    toed->get_Third_Order_Edges(image);
    edges = toed->toed_edges;

    // WriteEdgesToBinary(path, edges);
    // std::cout << "Saved edge data to: " << path << std::endl;
    // }
}

/*
    Pick a random edge
*/
std::tuple<std::vector<cv::Point2d>, std::vector<double>, std::vector<cv::Point2d>> EBVO::PickRandomEdges(int patch_size, const std::vector<cv::Point2d> &edges, const std::vector<cv::Point2d> &ground_truth_right_edges, const std::vector<double> &orientations, size_t num_points, int img_width, int img_height)
{
    std::vector<cv::Point2d> valid_edges;
    std::vector<double> valid_orientations;
    std::vector<cv::Point2d> valid_ground_truth_edges;
    int half_patch = patch_size / 2;

    if (edges.size() != orientations.size() || edges.size() != ground_truth_right_edges.size())
    {
        throw std::runtime_error("Edge locations, orientations, and ground truth edges size mismatch.");
    }

    for (size_t i = 0; i < edges.size(); ++i)
    {
        const auto &edge = edges[i];
        if (edge.x >= half_patch && edge.x < img_width - half_patch &&
            edge.y >= half_patch && edge.y < img_height - half_patch)
        {
            valid_edges.push_back(edge);
            valid_orientations.push_back(orientations[i]);
            valid_ground_truth_edges.push_back(ground_truth_right_edges[i]);
        }
    }

    num_points = std::min(num_points, valid_edges.size());

    std::vector<cv::Point2d> selected_points;
    std::vector<double> selected_orientations;
    std::vector<cv::Point2d> selected_ground_truth_points;
    std::unordered_set<int> used_indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, valid_edges.size() - 1);

    while (selected_points.size() < num_points)
    {
        int index = dis(gen);
        if (used_indices.find(index) == used_indices.end())
        {
            selected_points.push_back(valid_edges[index]);
            selected_orientations.push_back(valid_orientations[index]);
            selected_ground_truth_points.push_back(valid_ground_truth_edges[index]);
            used_indices.insert(index);
        }
    }

    return {selected_points, selected_orientations, selected_ground_truth_points};
}

Eigen::Vector3d two2three(const Edge &e, bool left, double d, Dataset &dataset)
{
    // FIXME:: if doing bidirectional, need to careful with the disparity, which is redo the bilinear interpolation!
    //> Convert cv::Point2d to Eigen::Vector3d
    Eigen::Vector3d e_location_eigen(e.location.x, e.location.y, 1.0);
    double focal_length, baseline;
    Eigen::Matrix3d calib_matrix;
    if (left)
    {
        focal_length = dataset.get_left_focal_length();
        baseline = dataset.get_left_baseline();
        calib_matrix = dataset.get_left_calib_matrix();
    }
    else
    {
        focal_length = dataset.get_right_focal_length();
        baseline = dataset.get_right_baseline();
        calib_matrix = dataset.get_right_calib_matrix();
    }

    double rho = focal_length * baseline / d;
    double rho_1 = (rho < 0.0) ? (-rho) : (rho);
    Eigen::Vector3d gamma_1 = calib_matrix.inverse() * e_location_eigen;
    Eigen::Vector3d Gamma_1 = rho_1 * gamma_1;

    return Gamma_1;
}

void EBVO::Find_Stereo_GT_Locations(const cv::Mat left_disparity_map, bool left, StereoEdgeCorrespondencesGT &prev_stereo_frame, const std::vector<Edge> &left_edges)
{
    Utility util;

    // left_edges here are actually the focused edges (change it after)
    for (int i = 0; i < left_edges.size(); ++i)
    {
        const Edge &e = left_edges[i];
        double disparity = Bilinear_Interpolation(left_disparity_map, e.location);

        if (std::isnan(disparity) || std::isinf(disparity) || disparity < 0)
        {
            continue;
        }
        double x_loc = left ? e.location.x - disparity : e.location.x + disparity;
        cv::Point2d GT_location(x_loc, e.location.y);

        //> Insert the data to the structure
        prev_stereo_frame.focused_edges.push_back(i);
        prev_stereo_frame.GT_locations_from_disparity.push_back(GT_location);

        //> Convert cv::Point2d to Eigen::Vector3d
        Eigen::Vector3d GT_location_eigen(GT_location.x, GT_location.y, 1.0);

        Eigen::Vector3d Gamma_1 = two2three(e, left, disparity, dataset);

        //> Compute the corresponding 3D point locations under the left camera coordinate
        // Eigen::Vector3d triangulated_point = util.two_view_linear_triangulation(
        //     e_location_eigen, GT_location_eigen,
        //     dataset.get_left_calib_matrix(), dataset.get_right_calib_matrix(),
        //     dataset.get_relative_rot_right_to_left(), dataset.get_relative_transl_right_to_left());

        prev_stereo_frame.Gamma_in_cam_coord.push_back(Gamma_1);
    }
#if WRITE_KF_CF_GT_EDGE_PAIRS
    std::string left_str = left ? "left" : "right";

    std::string stereo_gt_filename = dataset.get_output_path() + "/stereo_GT_loc_" + left_str + ".txt";
    std::cout << "Writing Stereo GT to : " << stereo_gt_filename << std::endl;
    std::ofstream stereo_gt_file(stereo_gt_filename);
    int gt_edge_idx = 0;
    stereo_gt_file << "#Stereo_frame_edge_indx\tEdge_index\tEdge_X\tEdge_Y\tEdge_Orientation\tGT_Edge_X\tGT_Edge_Y\tGamma_X\tGamma_Y\tGamma_Z\n";
    for (int i = 0; i < prev_stereo_frame.focused_edges.size(); i++)
    {
        const Edge &e = left_edges[i];
        stereo_gt_file << gt_edge_idx << "\t"
                       << prev_stereo_frame.focused_edges[i] << "\t"
                       << e.location.x << "\t"
                       << e.location.y << "\t"
                       << e.orientation << "\t"
                       << prev_stereo_frame.GT_locations_from_disparity[i].x << "\t"
                       << prev_stereo_frame.GT_locations_from_disparity[i].y << "\n";
        stereo_gt_file << prev_stereo_frame.Gamma_in_cam_coord[i].x() << "\t"
                       << prev_stereo_frame.Gamma_in_cam_coord[i].y() << "\t"
                       << prev_stereo_frame.Gamma_in_cam_coord[i].z() << "\n";
        gt_edge_idx++;
    }
    stereo_gt_file.close();
#endif
}

void EBVO::Find_Veridical_Edge_Correspondences_on_CF(KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs,
                                                     StereoEdgeCorrespondencesGT &last_keyframe_stereo, StereoEdgeCorrespondencesGT &current_frame_stereo,
                                                     StereoFrame &last_keyframe, StereoFrame &current_frame, SpatialGrid &spatial_grid, bool is_left, double gt_dist_threshold)
{
    //> Ground-truth relative pose of the current frame with respect to the keyframe
    Eigen::Matrix3d R1 = last_keyframe.gt_rotation;
    Eigen::Vector3d t1 = last_keyframe.gt_translation;
    Eigen::Matrix3d R2 = current_frame.gt_rotation;
    Eigen::Vector3d t2 = current_frame.gt_translation;

    Eigen::Matrix3d R21 = R2 * R1.transpose();
    Eigen::Vector3d t21 = t2 - R2 * R1.transpose() * t1;
    std::vector<Edge> &current_kf_edges = is_left ? kf_edges_left : kf_edges_right;
    std::vector<Edge> &current_cf_edges = is_left ? cf_edges_left : cf_edges_right;
    std::vector<Edge> &other_keyframe_edges = is_left ? kf_edges_right : kf_edges_left;
    //> For each left edge in the keyframe, find the GT location on the current image
    for (int i = 0; i < last_keyframe_stereo.focused_edges.size(); ++i)
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
        Eigen::Vector3d projected_point = calib_matrix * (R21 * last_keyframe_stereo.Gamma_in_cam_coord[i] + t21);
        projected_point /= projected_point.z();
        cv::Point2d projected_point_cv(projected_point.x(), projected_point.y());

        if (projected_point.x() > 10 && projected_point.y() > 10 && projected_point.x() < dataset.get_width() - 10 && projected_point.y() < dataset.get_height() - 10)
        {
            std::vector<int> current_candidate_edge_indices;
            current_candidate_edge_indices = spatial_grid.getCandidatesWithinRadius(current_kf_edges[last_keyframe_stereo.focused_edges[i]].location, 1); // clear: we use the kf edge stereoframe order, to get the candaiates
            std::vector<int> CF_veridical_edges_indices;
            for (const auto &curr_e_index : current_candidate_edge_indices)
            {
                //> Check if the Euclidean distance is less than some threshold
                if (cv::norm(current_cf_edges[curr_e_index].location - projected_point_cv) < gt_dist_threshold)
                {
                    CF_veridical_edges_indices.push_back(curr_e_index);
                }
            }
            if (!CF_veridical_edges_indices.empty())
            {
                EdgeCorrespondenceData edge_pair;
                edge_pair.stereo_frame_idx = i;
                edge_pair.gt_location_on_cf = projected_point_cv;
                edge_pair.veridical_cf_edges_indices = CF_veridical_edges_indices;

                //> FIXME: This should be obtained from the stereo edge matching result
                // edge_pair.veridical_stereo_right_edges_for_cf.push_back(current_frame_stereo.GT_corresponding_edges[curr_e_index]);
                // edge_pair.veridical_stereo_right_edges_for_kf.push_back(last_keyframe_stereo.GT_corresponding_edges[i]);
                KF_CF_edge_pairs[last_keyframe_stereo.focused_edges[i]] = edge_pair;
            }
        }
    }
#if WRITE_KF_CF_GT_EDGE_PAIRS
    std::string left = is_left ? "left" : "right";
    std::string kf_cf_gt_edge_pairs_last_kf_filename = dataset.get_output_path() + "/kf_cf_gt_edge_pairs_" + left + "_KF.txt";
    std::cout << "Writing KF-CF-GT edge pairs for KF to: " << kf_cf_gt_edge_pairs_last_kf_filename << std::endl;
    std::ofstream kf_cf_gt_edge_pairs_kf_file(kf_cf_gt_edge_pairs_last_kf_filename);
    int gt_edge_idx = 0;
    kf_cf_gt_edge_pairs_kf_file << "#Stereo_frame_edge_indx\tEdge_index\tKF_Edge_X\tKF_Edge_Y\tKF_Edge_Orientation\tDouble_check_frame_idx\tStereo_GT_index\tStereo_GT_X\tStereo_GT_Y\tCF_GT_Location_X\tCF_GT_Location_Y\n";
    for (const auto &[kf_edge_index, ep] : KF_CF_edge_pairs)
    {
        int other_kf_idx = last_keyframe_stereo.GT_corresponding_edges[ep.stereo_frame_idx][last_keyframe_stereo.Closest_GT_veridical_edges[ep.stereo_frame_idx]];
        Edge &other_kf_edge = other_keyframe_edges[other_kf_idx];
        kf_cf_gt_edge_pairs_kf_file << gt_edge_idx << "\t"
                                    << kf_edge_index << "\t"
                                    << current_kf_edges[kf_edge_index].location.x << "\t"
                                    << current_kf_edges[kf_edge_index].location.y << "\t"
                                    << current_kf_edges[kf_edge_index].orientation << "\t"
                                    << ep.stereo_frame_idx << "\t"
                                    << other_kf_idx << "\t"
                                    << other_kf_edge.location.x << "\t" << other_kf_edge.location.y << "\t"

                                    << ep.gt_location_on_cf.x << "\t"
                                    << ep.gt_location_on_cf.y << "\n";
        gt_edge_idx++;
    }
    kf_cf_gt_edge_pairs_kf_file.close();

    std::string kf_cf_gt_edge_pairs_curr_kf_filename = dataset.get_output_path() + "/kf_cf_gt_edge_pairs_" + left + "_CF.txt";
    std::cout << "Writing KF-CF-GT edge pairs for CF to: " << kf_cf_gt_edge_pairs_curr_kf_filename << std::endl;
    std::ofstream kf_cf_gt_edge_pairs_cf_file(kf_cf_gt_edge_pairs_curr_kf_filename);
    gt_edge_idx = 0;
    for (const auto &[kf_edge_index, ep] : KF_CF_edge_pairs)
    {
        kf_cf_gt_edge_pairs_cf_file << "#GT_Edge_Index_on_KF\tCF_Edge_X\tCF_Edge_Y\tCF_Edge_Orientation\n";
        for (const auto &v_edge_idx : ep.veridical_cf_edges_indices)
        {
            kf_cf_gt_edge_pairs_cf_file << gt_edge_idx << "\t"
                                        << current_cf_edges[v_edge_idx].location.x << "\t"
                                        << current_cf_edges[v_edge_idx].location.y << "\t"
                                        << current_cf_edges[v_edge_idx].orientation << "\n";
        }
        gt_edge_idx++;
    }
    kf_cf_gt_edge_pairs_cf_file.close();
#endif
}

// void EBVO::Right_Edges_Stereo_Reconstruction(const StereoEdgeCorrespondencesGT &stereo_left, StereoEdgeCorrespondencesGT &stereo_right, StereoFrame &current_frame)
// {
//     stereo_right.clear_all();

//     if (stereo_left.focused_edges.empty())
//     {
//         std::cerr << "Warning: No focused edges in left stereo data" << std::endl;
//         return;
//     }

//     const size_t num_edges = stereo_left.focused_edges.size();
//     stereo_right.focused_edges.reserve(num_edges);
//     stereo_right.GT_locations_from_disparity.reserve(num_edges);
//     stereo_right.GT_corresponding_edges.reserve(num_edges);
//     stereo_right.Closest_GT_veridical_edges.reserve(num_edges);
//     stereo_right.Gamma_in_cam_coord.reserve(num_edges);

//     for (size_t i = 0; i < num_edges; ++i)
//     {
//         // Bounds checking
//         if (i >= stereo_left.Closest_GT_veridical_edges.size() ||
//             i >= stereo_left.GT_corresponding_edges.size() ||
//             i >= stereo_left.Gamma_in_cam_coord.size())
//         {
//             std::cerr << "Index out of bounds at i=" << i << std::endl;
//             continue;
//         }

//         if (stereo_left.GT_corresponding_edges[i].empty())
//         {
//             std::cerr << "Empty GT_corresponding_edges for edge " << i << std::endl;
//             continue;
//         }

//         int closest_ind = stereo_left.Closest_GT_veridical_edges[i];

//         // Validate closest_ind
//         if (closest_ind < 0 ||
//             static_cast<size_t>(closest_ind) >= stereo_left.GT_corresponding_edges[i].size())
//         {
//             std::cerr << "Invalid closest_ind=" << closest_ind
//                       << " for edge " << i
//                       << " (GT_corresponding_edges size: " << stereo_left.GT_corresponding_edges[i].size() << ")" << std::endl;
//             continue;
//         }

//         const Edge &focus_edge = stereo_left.GT_corresponding_edges[i][closest_ind];

//         stereo_right.focused_edges.push_back(focus_edge);
//         stereo_right.GT_locations_from_disparity.push_back(stereo_left.focused_edges[i].location);
//         stereo_right.GT_corresponding_edges.push_back({stereo_left.focused_edges[i]});
//         stereo_right.Closest_GT_veridical_edges.push_back(0);
//         stereo_right.Gamma_in_cam_coord.push_back(stereo_left.Gamma_in_cam_coord[i]);
//     }
//     if (!stereo_right.focused_edges.empty())
//     {
//         try
//         {
//             std::cout << "Extracting SIFT descriptors for right edges..." << std::endl;
//             augment_Edge_Data(stereo_right, current_frame.right_image_undistorted);
//         }
//         catch (const std::exception &e)
//         {
//             std::cerr << "Error in augment_Edge_Data: " << e.what() << std::endl;
//             return;
//         }
//     }
// }

void EBVO::apply_spatial_grid_filtering(KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs, const StereoEdgeCorrespondencesGT &keyframe_stereo, const std::vector<Edge> &edges, SpatialGrid &spatial_grid, double grid_radius)
{
    //> For each edge in the keyframe, find candidate edges in the current frame using spatial grid
    // #pragma omp parallel for schedule(dynamic)
    for (auto it = KF_CF_edge_pairs.begin(); it != KF_CF_edge_pairs.end(); ++it)
    {
        int kf_edge_idx = it->first;
        EdgeCorrespondenceData &edge_data = it->second;
        edge_data.matching_cf_edges_indices = spatial_grid.getCandidatesWithinRadius(edges[kf_edge_idx], grid_radius);
    }
}

void EBVO::apply_SIFT_filtering(KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs, const StereoEdgeCorrespondencesGT &keyframe_stereo, const StereoEdgeCorrespondencesGT &current_stereo, double sift_dist_threshold, bool is_left)
{
    //> for each edge in the keyframe, compare its SIFT descriptor with the SIFT descriptors of the edges in the current frame
    // #pragma omp parallel for schedule(dynamic)
    std::unordered_map<int, int> edge_to_stereo_idx;
    for (int i = 0; i < current_stereo.focused_edges.size(); ++i)
    {
        edge_to_stereo_idx[current_stereo.focused_edges[i]] = i;
    }

    for (auto it = KF_CF_edge_pairs.begin(); it != KF_CF_edge_pairs.end(); ++it)
    {
        int kf_edge_idx = it->first;
        EdgeCorrespondenceData &edge_data = it->second;
        std::vector<int> filtered_cf_edges_indices;
        for (auto &m_edge_idx : edge_data.matching_cf_edges_indices)
        {
            auto stereo_it = edge_to_stereo_idx.find(m_edge_idx);
            if (stereo_it != edge_to_stereo_idx.end())
            {
                int stereo_frame_index = stereo_it->second;
                double sift_desc_distance = cv::norm(keyframe_stereo.edge_descriptors[edge_data.stereo_frame_idx], current_stereo.edge_descriptors[stereo_frame_index], cv::NORM_L2);
                if (sift_desc_distance < sift_dist_threshold)
                {
                    filtered_cf_edges_indices.push_back(m_edge_idx);
                }
            }
        }
        edge_data.matching_cf_edges_indices = filtered_cf_edges_indices;
    }
#if WRITE_KF_CF_GT_EDGE_PAIRS
    std::string left = is_left ? "left" : "right";
    std::vector<Edge> &current_cf_edges = is_left ? cf_edges_left : cf_edges_right;
    std::string kf_cf_gt_edge_pairs_curr_kf_filename = dataset.get_output_path() + "/kf_cf_gt_edge_pairs_after_SIFT_" + left + "_CF.txt";
    std::cout << "Writing KF-CF-GT edge pairs for CF after SIFT filtering to: " << kf_cf_gt_edge_pairs_curr_kf_filename << std::endl;
    std::ofstream kf_cf_gt_edge_pairs_cf_file(kf_cf_gt_edge_pairs_curr_kf_filename);
    int gt_edge_idx = 0;
    for (const auto &[kf_edge_index, ep] : KF_CF_edge_pairs)
    {
        for (const auto &v_edge_idx : ep.veridical_cf_edges_indices)
        {
            kf_cf_gt_edge_pairs_cf_file << gt_edge_idx << "\t"
                                        << kf_edge_index << "\t"
                                        << current_stereo.focused_edges[ep.stereo_frame_idx] << "\t"
                                        << current_cf_edges[v_edge_idx].location.x << "\t"
                                        << current_cf_edges[v_edge_idx].location.y << "\t"
                                        << current_cf_edges[v_edge_idx].orientation << "\n";
        }
        gt_edge_idx++;
    }
    kf_cf_gt_edge_pairs_cf_file.close();
#endif
}
void EBVO::apply_NCC_filtering(KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs, const StereoEdgeCorrespondencesGT &keyframe_stereo, const StereoEdgeCorrespondencesGT &current_stereo, double ncc_val_threshold,
                               const cv::Mat &keyframe_image, const cv::Mat &current_image, bool is_left)
{

    //> for each edge in the keyframe, compute the NCC score with the edges in the current frame
    // #pragma omp parallel for schedule(dynamic)
    for (auto it = KF_CF_edge_pairs.begin(); it != KF_CF_edge_pairs.end(); ++it)
    {
        int kf_edge_idx = it->first;
        EdgeCorrespondenceData &edge_data = it->second;
        std::vector<int> filtered_cf_edges_indices;
        for (auto &m_edge_idx : edge_data.matching_cf_edges_indices)
        {
            const Edge &kf_edge = is_left ? kf_edges_left[kf_edge_idx]
                                          : kf_edges_right[kf_edge_idx];
            const Edge &cf_edge = is_left ? cf_edges_left[m_edge_idx]
                                          : cf_edges_right[m_edge_idx];

            double ncc_score = edge_patch_similarity(kf_edge, cf_edge, keyframe_image, current_image);
            // std::cout << "NCC score between KF edge at (" << kf_edge.location.x << ", " << kf_edge.location.y << ") and CF edge at (" << cf_edge.location.x << ", " << cf_edge.location.y << ") is: " << ncc_score << std::endl;
            if (ncc_score > ncc_val_threshold)
            {
                filtered_cf_edges_indices.push_back(m_edge_idx);
            }
        }
        edge_data.matching_cf_edges_indices = filtered_cf_edges_indices;
    }
}

void EBVO::apply_stereo_filtering(KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs_left, KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs_right,
                                  const StereoEdgeCorrespondencesGT &last_keyframe_stereo_left, const StereoEdgeCorrespondencesGT &current_frame_stereo_left,
                                  const StereoEdgeCorrespondencesGT &last_keyframe_stereo_right, const StereoEdgeCorrespondencesGT &current_frame_stereo_right,
                                  size_t frame_idx)
{
    std::unordered_map<int, int> right_edge_to_stereo_idx;
    for (int i = 0; i < current_frame_stereo_right.focused_edges.size(); ++i)
    {
        right_edge_to_stereo_idx[current_frame_stereo_right.focused_edges[i]] = i;
    }
    // we do the same thing to the stereo right edges: use keyframe_stereo.GT_corresponding_edges, apply spatial grid, sift, ncc to filter the right edges
    std::string output_dir = dataset.get_output_path();
    std::filesystem::create_directories(output_dir);
    std::string stereo_filter_filename = output_dir + "/stereo_filtering_debug_frame_" + std::to_string(frame_idx) + ".csv";
    std::ofstream stereo_csv(stereo_filter_filename);
    // stereo_csv << "examining the 1000th KF-CF pair in left frames\n";
    // std::cout << "Applying stereo filtering on KF-CF edge correspondences..." << std::endl;
    auto write_edge_pair_to_csv = [&](const std::string &stage, int pair_idx, int edge_index, const EdgeCorrespondenceData &edge_pair)
    {
        stereo_csv << "kf_edge_index,gt_location_x,gt_location_y\n";
        stereo_csv << edge_index << ","
                   << std::fixed << std::setprecision(2) << edge_pair.gt_location_on_cf.x << ","
                   << std::fixed << std::setprecision(2) << edge_pair.gt_location_on_cf.y << "\n";
        stereo_csv << "number of veridical CF edges\n";
        stereo_csv << edge_pair.veridical_cf_edges_indices.size() << "\n";
        stereo_csv << "number of matching CF edges after " << stage << "\n";
        stereo_csv << edge_pair.matching_cf_edges_indices.size() << "\n";

        stereo_csv << "veridical_cf_edges_indices\n";
        // Write veridical indices
        for (size_t i = 0; i < edge_pair.veridical_cf_edges_indices.size(); ++i)
        {
            if (i > 0)
                stereo_csv << ";";
            if (i % 10 == 0 && i > 0)
                stereo_csv << "\n"; // New line every 10 entries for readability
            stereo_csv << edge_pair.veridical_cf_edges_indices[i];
        }
        stereo_csv << "\nmatching_cf_edges_indices\n";

        // Write matching indices
        for (size_t i = 0; i < edge_pair.matching_cf_edges_indices.size(); ++i)
        {
            if (i > 0)
                stereo_csv << ";";
            if (i % 10 == 0 && i > 0)
                stereo_csv << "\n"; // New line every 10 entries for readability
            stereo_csv << edge_pair.matching_cf_edges_indices[i];
        }
        stereo_csv << "\n";
    };
    try
    {
        int i = 0;

        for (auto it = KF_CF_edge_pairs_left.begin(); it != KF_CF_edge_pairs_left.end(); ++it)
        {
            bool debug = (i == 587);
            std::unordered_set<int> stereo_left_to_right_mapping;
            int left_kf_edge_index = it->first; // represents the 3rd order edge index in the keyframe left image
            EdgeCorrespondenceData *left_pair_it = &it->second;
            if (debug)
            {
                // std::cout << "trying logging initial left edge pair state before stereo filtering..." << std::endl;
                write_edge_pair_to_csv("initial", i, it->first, it->second);
            }

            // for each left edge in the keyframe, we find the closest corresponding right stereo edge
            int right_closest_kf_edge_index = last_keyframe_stereo_left.GT_corresponding_edges[left_pair_it->stereo_frame_idx][last_keyframe_stereo_left.Closest_GT_veridical_edges[left_pair_it->stereo_frame_idx]];
            std::vector<int> &matching_cf_right = KF_CF_edge_pairs_right[right_closest_kf_edge_index].matching_cf_edges_indices;
            if (debug)
            {
                std::cout << "Matching CF right edges size: " << matching_cf_right.size() << std::endl;
                write_edge_pair_to_csv("initial_right", i, right_closest_kf_edge_index, KF_CF_edge_pairs_right[right_closest_kf_edge_index]);
            }
            for (const auto &right_cf_edge_index : matching_cf_right)
            {
                auto stereo_it = right_edge_to_stereo_idx.find(right_cf_edge_index);
                if (stereo_it != right_edge_to_stereo_idx.end())
                {
                    int stereo_frame_idx = stereo_it->second;
                    const std::vector<int> &right_veridical_edges = current_frame_stereo_right.GT_corresponding_edges[stereo_frame_idx];

                    for (auto &veridical_edge : right_veridical_edges)
                    {
                        stereo_left_to_right_mapping.insert(veridical_edge);
                    }
                }

                if (debug)
                {
                    std::cout << "Matching CF right edges size: " << matching_cf_right.size() << std::endl;
                }
            }
            // std::cout << "Number of mapped right edges from left edges: " << stereo_left_to_right_mapping.size() << std::endl;
            std::vector<int> filtered_cf_left_edges;
            for (const auto &left_cf_edge_index : left_pair_it->matching_cf_edges_indices)
            {
                if (stereo_left_to_right_mapping.find(left_cf_edge_index) != stereo_left_to_right_mapping.end())
                {
                    filtered_cf_left_edges.push_back(left_cf_edge_index);
                }
            }
            left_pair_it->matching_cf_edges_indices = filtered_cf_left_edges;
            if (debug)
            {
                write_edge_pair_to_csv("stereo_filtering", i, left_kf_edge_index, *left_pair_it);
            }
            i++;
        }

        stereo_csv.close();
        std::cout << "Stereo filtering debug data saved to: " << stereo_filter_filename << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in apply_stereo_filtering: " << e.what() << std::endl;
        throw;
    }
    catch (...)
    {
        std::cerr << "Unknown exception in apply_stereo_filtering" << std::endl;
        throw;
    }
}

void EBVO::Evaluate_KF_CF_Edge_Correspondences(const KF_CF_EdgeCorrespondenceMap &KF_CF_edge_pairs,
                                               StereoEdgeCorrespondencesGT &keyframe_stereo, StereoEdgeCorrespondencesGT &current_stereo,
                                               size_t frame_idx, const std::string &stage_name)
{
    //> FIXME: Write the results to a file
    // std::string output_dir = dataset.get_output_path();
    // std::string eval_filename = output_dir + "/edge_match_evaluation_" + stage_name + "_frame_" + std::to_string(frame_idx) + ".txt";
    // std::ofstream eval_PR(eval_filename);

    //> For each left edge from the keyframe, see if the filtered edges are found in the pool of matched veridical edges on the current frame
    int total_num_of_true_positives = 0;
    std::vector<double> num_of_cf_edges_per_kf_edge;
    std::vector<double> precision_per_edge;
    num_of_cf_edges_per_kf_edge.reserve(KF_CF_edge_pairs.size());
    precision_per_edge.reserve(KF_CF_edge_pairs.size());
    for (const auto &[edge_idx, edge_pair] : KF_CF_edge_pairs)
    {
        //> Find if there is at least one edge index in edge_pair.matching_cf_edges_indices is found in edge_pair.veridical_cf_edges_indices
        for (const auto &v_edge_idx : edge_pair.veridical_cf_edges_indices)
        {
            if (std::find(edge_pair.matching_cf_edges_indices.begin(), edge_pair.matching_cf_edges_indices.end(), v_edge_idx) != edge_pair.matching_cf_edges_indices.end())
            {
                total_num_of_true_positives++;
                precision_per_edge.push_back(static_cast<double>(edge_pair.veridical_cf_edges_indices.size()) / static_cast<double>(edge_pair.matching_cf_edges_indices.size()));
                break;
            }
        }
        num_of_cf_edges_per_kf_edge.push_back(edge_pair.matching_cf_edges_indices.size());
    }

    double recall_per_image = static_cast<double>(total_num_of_true_positives) / KF_CF_edge_pairs.size();
    double precision_per_image = std::accumulate(precision_per_edge.begin(), precision_per_edge.end(), 0.0) / precision_per_edge.size();
    double num_of_cf_edges_per_kf_edge_avg = std::accumulate(num_of_cf_edges_per_kf_edge.begin(), num_of_cf_edges_per_kf_edge.end(), 0.0) / num_of_cf_edges_per_kf_edge.size();

    std::cout << "Stage: " << stage_name << " | Frame: " << frame_idx << std::endl;
    std::cout << "- Recall rate:       " << std::fixed << std::setprecision(8) << recall_per_image << std::endl;
    std::cout << "- Precision rate:    " << std::fixed << std::setprecision(8) << precision_per_image << std::endl;
    std::cout << "- Average ambiguity: " << std::fixed << std::setprecision(8) << num_of_cf_edges_per_kf_edge_avg << std::endl;
    std::cout << "========================================================\n"
              << std::endl;
}

void EBVO::EvaluateEdgeMatchPerformance(const std::unordered_map<Edge, std::vector<Edge>> &Edge_match,
                                        const std::unordered_map<Edge, EdgeGTMatchInfo> &gt_correspondences,
                                        size_t frame_idx,
                                        const std::string &stage_name,
                                        double distance_threshold,
                                        const StereoFrame &previous_frame,
                                        const StereoFrame &current_frame)
{
    int true_positives = 0;
    int false_positives = 0;
    int false_negatives = 0;
    int total_predictions = 0;
    int total_individual_matches = 0;
    int correct_individual_matches = 0;
    int edges_evaluated = 0;
    int total_gt_correspondences = gt_correspondences.size();

    std::string output_dir = dataset.get_output_path();
    std::string eval_filename = output_dir + "/edge_match_evaluation_" + stage_name + "_frame_" + std::to_string(frame_idx) + ".csv";
    std::ofstream eval_csv(eval_filename);

    // Create a separate CSV file for false positives
    std::string fp_filename = output_dir + "/false_positives_" + stage_name + "_frame_" + std::to_string(frame_idx) + ".csv";
    std::ofstream fp_csv(fp_filename);
    fp_csv << "edge_idx,prev_edge_x,prev_edge_y,matched_edge_x,matched_edge_y,gt_curr_x,gt_curr_y,error_distance,threshold\n";

    eval_csv << "edge_idx,prev_edge_x,prev_edge_y,matched_edge_x,matched_edge_y,is_correct,gt_curr_x,gt_curr_y,error_distance\n";

    // Print header for console output
    std::cout << "\n=== FALSE POSITIVES DETAILS (" << stage_name << ") FRAME " << frame_idx << " ===" << std::endl;
    std::cout << "Threshold: " << distance_threshold << " pixels" << std::endl;
    std::cout << "Format: Prev_edge -> Matched_edge (GT_edge) | Error: X.XX pixels" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    int fp_count = 0;
    int idx = 0;

    std::string debug_all_edge_filename = output_dir + "/ncc_scores_all_edges_frame_" + std::to_string(frame_idx) + stage_name + ".csv";

    std::ofstream debug_all_edge_csv(debug_all_edge_filename);

    debug_all_edge_csv << "prev_edge_idx,prev_x,prev_y,prev_orientation,candidate_type,candidate_idx,candidate_x,candidate_y,candidate_orientation,gt_x,gt_y,ncc_score,is_gt_match\n";
    auto first_match = Edge_match.begin();

    const Edge &prev_edge = first_match->first;
    const std::vector<Edge> &candidates = first_match->second;

    int candidate_idx = 0;
    for (const Edge &candidate : candidates)
    {
        double ncc_score = edge_patch_similarity(prev_edge, candidate,
                                                 previous_frame.left_image,
                                                 current_frame.left_image);

        debug_all_edge_csv << "0" << ","
                           << prev_edge.location.x << ","
                           << prev_edge.location.y << ","
                           << prev_edge.orientation << ","
                           << "Candidate" << ","
                           << candidate_idx << ","
                           << candidate.location.x << ","
                           << candidate.location.y << ","
                           << candidate.orientation << ","
                           << "-1" << "," // No GT coords for all edges
                           << "-1" << ","
                           << ncc_score << ","
                           << "0" << "\n"; // is_gt_match = false

        candidate_idx++;
    }
    // Evaluate each edge match
    for (const auto &edge_pair : Edge_match)
    {
        const Edge &prev_edge = edge_pair.first;
        const std::vector<Edge> &matched_edges = edge_pair.second;

        auto gt_it = gt_correspondences.find(prev_edge);
        bool has_gt = (gt_it != gt_correspondences.end());
        idx++;
        // Skip edges that don't have ground truth correspondence
        if (!has_gt)
        {
            continue;
        }

        cv::Point2d gt_curr_location = gt_it->second.gt_edge.location;
        total_predictions++;

        bool found_correct_match = false;
        double best_error_distance = std::numeric_limits<double>::max();
        Edge best_matched_edge;

        // Evaluate each matched edge for this previous edge
        for (const Edge &matched_edge : matched_edges)
        {
            total_individual_matches++;

            double error_distance = cv::norm(matched_edge.location - gt_curr_location);
            bool is_correct = (error_distance <= distance_threshold);

            // Track the best match for this previous edge
            if (error_distance < best_error_distance)
            {
                best_error_distance = error_distance;
                best_matched_edge = matched_edge;
            }

            if (is_correct)
            {
                correct_individual_matches++; // NEW: Count every correct individual match
                found_correct_match = true;
                break; // Found a correct match, no need to check others for this edge
            }
        }

        if (found_correct_match)
        {
            true_positives++;
        }
        else
        {
            false_positives++;

            fp_csv << std::fixed << std::setprecision(2)
                   << idx - 1 << ","
                   << prev_edge.location.x << ","
                   << prev_edge.location.y << ","
                   << best_matched_edge.location.x << ","
                   << best_matched_edge.location.y << ","
                   << gt_curr_location.x << ","
                   << gt_curr_location.y << ","
                   << best_error_distance << ","
                   << distance_threshold << "\n";
        }

        // Write the best match to main CSV
        if (!matched_edges.empty())
        {
            const Edge &csv_matched_edge = (best_error_distance != std::numeric_limits<double>::max())
                                               ? best_matched_edge
                                               : matched_edges[0];

            eval_csv << std::fixed << std::setprecision(2)
                     << idx - 1 << ","
                     << prev_edge.location.x << ","
                     << prev_edge.location.y << ","
                     << csv_matched_edge.location.x << ","
                     << csv_matched_edge.location.y << ","
                     << (found_correct_match ? 1 : 0) << ","
                     << gt_curr_location.x << ","
                     << gt_curr_location.y << ","
                     << best_error_distance << "\n";
        }
    }

    // New: Calculate false negatives (GT correspondences not found in Edge_match)
    for (const auto &gt_pair : gt_correspondences)
    {
        const Edge &gt_prev_edge = gt_pair.first;
        auto edge_match_it = Edge_match.find(gt_prev_edge);

        if (edge_match_it == Edge_match.end() || edge_match_it->second.empty())
        {
            false_negatives++;
        }
    }

    eval_csv.close();
    fp_csv.close();

    // Print summary of false positives
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Total False Positives: " << false_positives << std::endl;
    std::cout << "False positives saved to: " << fp_filename << std::endl;
    std::cout << "================================================================\n"
              << std::endl;

    double match_level_precision = (total_individual_matches > 0) ? (double)correct_individual_matches / total_individual_matches : 0.0;
    double edge_level_precision = (total_predictions > 0) ? (double)true_positives / total_predictions : 0.0;
    double recall = (total_gt_correspondences > 0) ? (double)true_positives / total_gt_correspondences : 0.0;
    double f1_score = (edge_level_precision + recall > 0) ? 2.0 * edge_level_precision * recall / (edge_level_precision + recall) : 0.0;

    // Print results with corrected metrics
    std::cout << "\n=== EDGE MATCH EVALUATION (" << stage_name << ") FRAME " << frame_idx << " ===" << std::endl;
    std::cout << "Distance threshold: " << distance_threshold << " pixels" << std::endl;
    std::cout << "Total Edge_match predictions (edges): " << total_predictions << std::endl;
    std::cout << "Total individual matches evaluated: " << total_individual_matches << std::endl;
    std::cout << "Total GT correspondences: " << total_gt_correspondences << std::endl;
    std::cout << "True Positives (edges): " << true_positives << std::endl;
    std::cout << "False Positives (edges): " << false_positives << std::endl;
    std::cout << "False Negatives (edges): " << false_negatives << std::endl;
    std::cout << "Precision (match-level): " << std::fixed << std::setprecision(4) << match_level_precision * 100 << "%" << std::endl;
    std::cout << "Precision (edge-level): " << std::fixed << std::setprecision(4) << edge_level_precision * 100 << "%" << std::endl;
    std::cout << "Recall: " << std::fixed << std::setprecision(4) << recall * 100 << "%" << std::endl;
    std::cout << "F1-Score: " << std::fixed << std::setprecision(4) << f1_score * 100 << "%" << std::endl;
    std::cout << "========================================================\n"
              << std::endl;

    // Save summary to file with additional metrics
    std::string summary_filename = output_dir + "/edge_match_metrics_summary.csv";
    std::ofstream summary_csv;

    bool file_exists = std::filesystem::exists(summary_filename);
    summary_csv.open(summary_filename, std::ios::app);

    if (!file_exists)
    {
        summary_csv << "frame_idx,stage_name,distance_threshold,total_predictions,total_matches_evaluated,total_gt_correspondences,true_positives,false_positives,false_negatives,precision_match_level,precision_edge_level,recall,f1_score\n";
    }

    summary_csv << frame_idx << ","
                << stage_name << ","
                << distance_threshold << ","
                << total_predictions << ","
                << total_individual_matches << "," // Fixed: use total_individual_matches instead of precision_denominator
                << total_gt_correspondences << ","
                << true_positives << ","
                << false_positives << ","
                << false_negatives << ","
                << std::fixed << std::setprecision(4) << match_level_precision << "," // Fixed: use match_level_precision instead of precision
                << std::fixed << std::setprecision(4) << edge_level_precision << ","  // Fixed: use edge_level_precision instead of per_edge_precision
                << std::fixed << std::setprecision(4) << recall << ","
                << std::fixed << std::setprecision(4) << f1_score << "\n";

    summary_csv.close();
}