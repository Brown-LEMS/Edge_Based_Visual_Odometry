#include <filesystem>
#include <unordered_set>
#include <numeric>
#include <cmath>
#include <chrono>
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
    std::vector<cv::Mat> right_ref_disparity_maps;
    std::vector<cv::Mat> left_occlusion_masks, right_occlusion_masks;

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
    StereoFrame previous_frame, current_frame;
    StereoFrame last_keyframe;
    //> Initialize
    Stereo_Edge_Pairs last_keyframe_stereo_left, current_frame_stereo_left;
    Stereo_Edge_Pairs last_keyframe_stereo_right, current_frame_stereo_right;

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
        const cv::Mat &right_disparity_map = (frame_idx < right_ref_disparity_maps.size()) ? right_ref_disparity_maps[frame_idx] : cv::Mat();

        const cv::Mat &left_occlusion_mask = (frame_idx < left_occlusion_masks.size()) ? left_occlusion_masks[frame_idx] : cv::Mat();
        const cv::Mat &right_occlusion_mask = (frame_idx < right_occlusion_masks.size()) ? right_occlusion_masks[frame_idx] : cv::Mat();

        dataset.ncc_one_vs_err.clear();
        dataset.ncc_two_vs_err.clear();
        dataset.ground_truth_right_edges_after_lowe.clear();

        std::cout << "Image Pair #" << frame_idx << "\n";

        cv::Mat left_prev_undistorted, right_prev_undistorted, left_cur_undistorted, right_cur_undistorted;

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

        std::string edge_dir = dataset.get_output_path() + "/edges";
        std::filesystem::create_directories(edge_dir);

        std::string left_edge_path = edge_dir + "/left_edges_" + std::to_string(frame_idx + 1);
        std::string right_edge_path = edge_dir + "/right_edges_" + std::to_string(frame_idx + 1);

        ProcessEdges(left_cur_undistorted, left_edge_path, TOED, dataset.left_edges);
        std::cout << "Number of edges on the left image: " << dataset.left_edges.size() << std::endl;
        current_frame.left_edges = dataset.left_edges;

        ProcessEdges(right_cur_undistorted, right_edge_path, TOED, dataset.right_edges);
        std::cout << "Number of edges on the right image: " << dataset.right_edges.size() << std::endl;
        current_frame.right_edges = dataset.right_edges;
        dataset.increment_num_imgs();

        if (b_is_keyframe)
        {
            //> Update last keyframe
            last_keyframe = current_frame;
            kf_edges_left = dataset.left_edges;
            kf_edges_right = dataset.right_edges;
            last_keyframe_stereo_left.clean_up_vector_data_structures();
            last_keyframe_stereo_left.stereo_frame = &last_keyframe;
            last_keyframe_stereo_left.left_disparity_map = left_disparity_map;
            last_keyframe_stereo_left.right_disparity_map = right_disparity_map;

            last_keyframe_stereo_right.clean_up_vector_data_structures();
            last_keyframe_stereo_right.stereo_frame = &last_keyframe;
            last_keyframe_stereo_right.left_disparity_map = right_disparity_map;
            last_keyframe_stereo_right.right_disparity_map = left_disparity_map;

            //> For each left edge, get the corresponding GT location (not right edge) on the right image, and the triangulated 3D point in the left camera coordinate
            Find_Stereo_GT_Locations(left_disparity_map, left_occlusion_mask, true, last_keyframe_stereo_left);
            std::cout << "Complete calculating GT locations for left edges of the keyframe (previous frame)..." << std::endl;
            Find_Stereo_GT_Locations(left_disparity_map, left_occlusion_mask, false, last_keyframe_stereo_right);
            std::cout << "Complete calculating GT locations for right edges of the keyframe (previous frame)..." << std::endl;

            //> Construct a GT stereo edge pool
            get_Stereo_Edge_GT_Pairs(dataset, current_frame, last_keyframe_stereo_left, true);
            std::cout << "Size of stereo edge correspondences pool = " << last_keyframe_stereo_left.focused_edge_indices.size() << std::endl;
            get_Stereo_Edge_GT_Pairs(dataset, current_frame, last_keyframe_stereo_right, false);
            std::cout << "Size of stereo edge correspondences pool = " << last_keyframe_stereo_right.focused_edge_indices.size() << std::endl;

            last_keyframe_stereo_left.construct_toed_left_id_to_Stereo_Edge_Pairs_left_id_map();
            last_keyframe_stereo_right.construct_toed_left_id_to_Stereo_Edge_Pairs_left_id_map();

            //> extract SIFT descriptor for each left and right edge of last_keyframe_stereo
            augment_Edge_Data(last_keyframe_stereo_left, true);
            augment_Edge_Data(last_keyframe_stereo_right, false);

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
            current_frame_stereo_left.clean_up_vector_data_structures();
            current_frame_stereo_left.stereo_frame = &current_frame;
            current_frame_stereo_left.left_disparity_map = left_disparity_map;
            current_frame_stereo_left.right_disparity_map = right_disparity_map;

            current_frame_stereo_right.clean_up_vector_data_structures();
            current_frame_stereo_right.stereo_frame = &current_frame;
            current_frame_stereo_right.left_disparity_map = right_disparity_map;
            current_frame_stereo_right.right_disparity_map = left_disparity_map;
            //> For each left edge, get the corresponding GT location (not right edge) on the right image, and the triangulated 3D point in the left camera coordinate

            Find_Stereo_GT_Locations(left_disparity_map, left_occlusion_mask, true, current_frame_stereo_left);
            std::cout << "Complete calculating GT locations for left edges of the current frame..." << current_frame_stereo_left.focused_edge_indices.size() << std::endl;
            Find_Stereo_GT_Locations(right_disparity_map, right_occlusion_mask, false, current_frame_stereo_right);
            std::cout << "Complete calculating GT locations for right edges of the current frame..." << std::endl;

            //> Construct a GT stereo edge pool
            get_Stereo_Edge_GT_Pairs(dataset, current_frame, current_frame_stereo_left, true);
            std::cout << "Size of stereo edge correspondences pool for left edges= " << current_frame_stereo_left.focused_edge_indices.size() << std::endl;
            get_Stereo_Edge_GT_Pairs(dataset, current_frame, current_frame_stereo_right, false);
            std::cout << "Size of stereo edge correspondences pool for right edges= " << current_frame_stereo_right.focused_edge_indices.size() << std::endl;

            // //> extract SIFT descriptor for each left edge of current_frame_stereo

            // if (!current_frame_stereo_left.b_is_size_consistent())
            //     current_frame_stereo_left.print_size_consistency();
            // if (!current_frame_stereo_right.b_is_size_consistent())
            //     current_frame_stereo_right.print_size_consistency();

            // add_edges_to_spatial_grid(current_frame_stereo_left, left_grid);
            // add_edges_to_spatial_grid(current_frame_stereo_right, right_grid);

            // // Create separate spatial grids with ALL CF edges (not just those with stereo GT)
            // // This allows temporal matching to find edges even if they lack stereo correspondences
            // SpatialGrid left_grid_all_edges(dataset.get_width(), dataset.get_height(), GRID_SIZE);
            // SpatialGrid right_grid_all_edges(dataset.get_width(), dataset.get_height(), GRID_SIZE);

            // std::cout << "Building spatial grids with ALL CF edges (for temporal matching)..." << std::endl;
            // for (int i = 0; i < cf_edges_left.size(); ++i)
            // {
            //     left_grid_all_edges.addEdge(i, cf_edges_left[i].location);
            // }
            // for (int i = 0; i < cf_edges_right.size(); ++i)
            // {
            //     right_grid_all_edges.addEdge(i, cf_edges_right[i].location);
            // }
            // std::cout << "Total edges in left_grid_all: " << cf_edges_left.size()
            //           << ", in left_grid (stereo-valid): " << current_frame_stereo_left.focused_edges.size() << std::endl;

            // //> Construct correspondences structure between last keyframe and the current frame
            // KF_CF_EdgeCorrespondence KF_CF_edge_pairs_left, KF_CF_edge_pairs_right;

            // // Use ALL-edges grid for temporal matching
            // Find_Veridical_Edge_Correspondences_on_CF(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, left_grid_all_edges, true);
            // // Use ALL-edges grid for temporal matching
            // Find_Veridical_Edge_Correspondences_on_CF(KF_CF_edge_pairs_right, last_keyframe_stereo_right, current_frame_stereo_right, right_grid_all_edges, false);

            // std::cout << "Size of veridical edge pairs (left) = " << KF_CF_edge_pairs_left.kf_edges.size() << std::endl;
            // std::cout << "Size of veridical edge pairs (right) = " << KF_CF_edge_pairs_right.kf_edges.size() << std::endl;

            //> Now that the GT edge correspondences are constructed between the keyframe and the current frame, we can apply various filters from the beginning
            //> Stage 1: Apply spatial grid to the current frame
            // apply_spatial_grid_filtering(KF_CF_edge_pairs_left, last_keyframe_stereo_left, kf_edges_left, left_grid_all_edges, 30.0);
            // apply_spatial_grid_filtering(KF_CF_edge_pairs_right, last_keyframe_stereo_right, kf_edges_right, right_grid_all_edges, 30.0);

            // Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "Spatial Grid");
            // Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_right, last_keyframe_stereo_right, current_frame_stereo_right, frame_idx, "Spatial Grid");

            // //> Stage 2: Do orientation filtering
            // apply_orientation_filtering(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, 35.0, true);
            // Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "Orientation Filtering");
            // //> Stage 3: Do NCC
            // apply_NCC_filtering(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, 0.25, last_keyframe.left_image, current_frame.left_image, true);
            // Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs_left, last_keyframe_stereo_left, current_frame_stereo_left, frame_idx, "NCC Filtering");

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

        // previous_frame = current_frame;
        // previous_edge_loc is now updated in the frame_idx > 0 block above
    }
}

void EBVO::add_edges_to_spatial_grid(Stereo_Edge_Pairs &stereo_frame, SpatialGrid &spatial_grid)
{
    //> Add left edges to spatial grid. This is done on the current image only.
    //> Pre-allocate grid_indices to avoid race conditions
    stereo_frame.grid_indices.resize(stereo_frame.focused_edge_indices.size());

#pragma omp parallel for schedule(static)
    for (int i = 0; i < stereo_frame.focused_edge_indices.size(); ++i)
    {
        int grid_index;
#pragma omp critical
        {
            cv::Point2d edge_location = stereo_frame.get_left_edge_by_StereoFrame_index(i).location;
            grid_index = spatial_grid.add_edge_to_grids(stereo_frame.focused_edge_indices[i], edge_location);
        }
        stereo_frame.grid_indices[i] = grid_index;
    }
}

void EBVO::ProcessEdges(const cv::Mat &image,
                        const std::string &filepath,
                        std::shared_ptr<ThirdOrderEdgeDetectionCPU> &toed,
                        std::vector<Edge> &edges)
{
    std::string path = filepath + ".bin";

    std::cout << "Running third-order edge detector..." << std::endl;
    toed->get_Third_Order_Edges(image);
    edges = toed->toed_edges;
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

double orientation_mapping(const Edge &e1, const Edge &e2, const Eigen::Vector3d projected_point, bool left, const StereoFrame &last_keyframe, const StereoFrame &current_frame, Dataset &dataset)
{
    //> Map the edge orientation from keyframe to current frame using stereo pair + temporal transformation
    // Step 1: Get the stereo baseline rotation (left<->right in same frame)
    Eigen::Matrix3d R_stereo = dataset.get_relative_rot_left_to_right();

    // Step 2: Reconstruct 3D direction T_1 using Eq. (186) from stereo pair (e1 in left, e2 in right)
    // t1 and t2 are unit tangent vectors in image plane
    Eigen::Vector3d t1(cos(e1.orientation), sin(e1.orientation), 0); // KF left edge orientation
    Eigen::Vector3d t2(cos(e2.orientation), sin(e2.orientation), 0); // KF right edge orientation (stereo mate)

    // Normalize image points to get γ (homogeneous coordinates with z=1)
    Eigen::Vector3d gamma_1(e1.location.x, e1.location.y, 1.0);
    gamma_1 = dataset.get_left_calib_matrix().inverse() * gamma_1;
    Eigen::Vector3d gamma_2(e2.location.x, e2.location.y, 1.0);
    gamma_2 = dataset.get_right_calib_matrix().inverse() * gamma_2;

    // Eq. (186): Reconstruct 3D tangent T_1 in KF left camera coordinate
    Eigen::Vector3d T_1 = -(gamma_2.dot(t2.cross(R_stereo * t1))) * gamma_1 + (gamma_2.dot(t2.cross(R_stereo * gamma_1))) * t1;
    T_1 = -T_1; // Flip sign to correct 180° orientation error
    T_1.normalize();

    // Step 3: Transform T_1 to current frame using temporal rotation (Eq. 176: T_2 = R_21 * T_1)
    Eigen::Matrix3d R_kf = last_keyframe.gt_rotation;
    Eigen::Matrix3d R_cf = current_frame.gt_rotation;
    Eigen::Matrix3d R_temporal = R_cf * R_kf.transpose(); // Temporal rotation from KF to CF

    // For right camera: need to account for stereo baseline in both frames
    Eigen::Matrix3d R_21;
    if (left)
    {
        R_21 = R_temporal; // Left KF -> Left CF
    }
    else
    {
        R_21 = R_stereo * R_temporal * R_stereo.transpose();
    }

    Eigen::Vector3d T_2 = R_21 * T_1; // 3D tangent in CF camera coordinate

    // Step 4: Project T_2 to image using Eq. (169): t = (T - (e_3^T * T) * γ) / ||...||
    // Ensure projected_point is normalized homogeneous (z=1)
    Eigen::Vector3d gamma_cf = projected_point / projected_point.z(); // γ = Γ / (e_3^T Γ)
    if (left)
        gamma_cf = dataset.get_left_calib_matrix().inverse() * gamma_cf;
    else
        gamma_cf = dataset.get_right_calib_matrix().inverse() * gamma_cf;
    // Eq. (169): t = (T - (e_3^T T) γ) / ||T - (e_3^T T) γ||
    // e_3^T T = T.z() (third component)
    Eigen::Vector3d t = T_2 - T_2.z() * gamma_cf;
    t.normalize();

    // Extract orientation angle from unit tangent vector t = [cos(θ), sin(θ), 0]
    double theta = atan2(t.y(), t.x());
    return theta;
}

void EBVO::Find_Stereo_GT_Locations(const cv::Mat left_disparity_map, const cv::Mat occlusion_mask, bool is_left, Stereo_Edge_Pairs &stereo_frame_edge_pairs)
{
    Utility util;

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<int>> thread_focused_edges(num_threads);
    std::vector<std::vector<cv::Point2d>> thread_gt_locations(num_threads);
    std::vector<std::vector<Eigen::Vector3d>> thread_gamma(num_threads);

    int edge_size = is_left ? stereo_frame_edge_pairs.stereo_frame->left_edges.size() : stereo_frame_edge_pairs.stereo_frame->right_edges.size();
    stereo_frame_edge_pairs.is_left_to_right = is_left;
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, 256)
        for (int left_edge_index = 0; left_edge_index < edge_size; left_edge_index++)
        {
            Edge e = stereo_frame_edge_pairs.stereo_frame->left_edges[left_edge_index]; //> we are abuseing the term left_edges here, it's universal for both left and right edges
            e = is_left ? e : stereo_frame_edge_pairs.stereo_frame->right_edges[left_edge_index];
            //> omit if the edge orientation is close to 0, pi, or -pi
            if (std::abs(rad_to_deg<double>(e.orientation)) < 4 || std::abs(rad_to_deg<double>(e.orientation) - 180.0) < 4 || std::abs(rad_to_deg<double>(e.orientation) + 180.0) < 4)
            {
                continue;
            }

            //> Use bilinear interpolation for sub-pixel accurate disparity

            double disparity = Bilinear_Interpolation<float>(left_disparity_map, e.location);

            if (std::isnan(disparity) || std::isinf(disparity) || disparity < 0)
            {
                continue;
            }
            // For both left and right edges, we subtract the disparity to find the correspondence
            // Left edges: move right (positive disparity) -> x - d
            // Right edges: move left (positive disparity) -> x - d
            cv::Point2d GT_location(e.location.x - disparity, e.location.y);
            Eigen::Vector3d e_location_eigen(e.location.x, e.location.y, 1.0);
            Eigen::Vector3d GT_location_eigen(GT_location.x, GT_location.y, 1.0);

            double focal_length, baseline;
            Eigen::Matrix3d calib_matrix;
            if (is_left)
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

            double rho = focal_length * baseline / disparity;
            double rho_1 = (rho < 0.0) ? (-rho) : (rho);

            Eigen::Vector3d gamma_1 = calib_matrix.inverse() * e_location_eigen;
            Eigen::Vector3d Gamma_1 = rho_1 * gamma_1;
            //> insert data to the stereo_frame_edge_pairs structure
            thread_focused_edges[tid].push_back(left_edge_index);
            thread_gt_locations[tid].push_back(GT_location);
            thread_gamma[tid].push_back(Gamma_1);
        }
    }
    for (int tid = 0; tid < num_threads; ++tid)
    {
        stereo_frame_edge_pairs.focused_edge_indices.insert(stereo_frame_edge_pairs.focused_edge_indices.end(),
                                                            thread_focused_edges[tid].begin(), thread_focused_edges[tid].end());
        stereo_frame_edge_pairs.GT_locations_from_left_edges.insert(stereo_frame_edge_pairs.GT_locations_from_left_edges.end(),
                                                                    thread_gt_locations[tid].begin(), thread_gt_locations[tid].end());
        stereo_frame_edge_pairs.Gamma_in_left_cam_coord.insert(stereo_frame_edge_pairs.Gamma_in_left_cam_coord.end(),
                                                               thread_gamma[tid].begin(), thread_gamma[tid].end());
    }
}

void EBVO::Find_Veridical_Edge_Correspondences_on_CF(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs,
                                                     Stereo_Edge_Pairs &last_keyframe_stereo, Stereo_Edge_Pairs &current_frame_stereo,
                                                     SpatialGrid &spatial_grid, bool is_left, double gt_dist_threshold)
{
    KF_CF_edge_pairs.key_frame = last_keyframe_stereo.stereo_frame;
    KF_CF_edge_pairs.current_frame = current_frame_stereo.stereo_frame;
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

    std::vector<Edge> &current_kf_edges = is_left ? kf_edges_left : kf_edges_right;
    std::vector<Edge> &current_cf_edges = is_left ? cf_edges_left : cf_edges_right;
    std::vector<Edge> &other_keyframe_edges = is_left ? kf_edges_right : kf_edges_left;

    // Record distances between KF location and projected point for proximity analysis
    std::vector<double> kf_to_projection_distances;

    // Thread-local storage for parallel correspondence accumulation
    int num_threads_corr = omp_get_max_threads();

    std::vector<std::vector<int>> thread_kf_edges(num_threads_corr);
    std::vector<std::vector<int>> thread_stereo_frame_indices(num_threads_corr); // Track stereo frame indices for GT lookups
    std::vector<std::vector<double>> thread_gt_orientations(num_threads_corr);
    std::vector<std::vector<cv::Point2d>> thread_gt_locations(num_threads_corr);
    std::vector<std::vector<std::vector<int>>> thread_veridical_cf_edges(num_threads_corr);
    std::vector<std::vector<double>> thread_distances(num_threads_corr); // Thread-local proximity distances

    //> For each left edge in the keyframe, find the GT location on the current image
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, 128)
        for (int i = 0; i < last_keyframe_stereo.focused_edge_indices.size(); ++i)
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
            Eigen::Vector3d projected_point = calib_matrix * (R21 * last_keyframe_stereo.Gamma_in_left_cam_coord[i] + t21);
            projected_point /= projected_point.z();
            int kf_veridical_edge_indx = last_keyframe_stereo.veridical_right_edges_indices[i][0];
            double ori = orientation_mapping(
                current_kf_edges[last_keyframe_stereo.focused_edge_indices[i]],
                other_keyframe_edges[kf_veridical_edge_indx],
                projected_point,
                is_left,
                *last_keyframe_stereo.stereo_frame,
                *current_frame_stereo.stereo_frame,
                dataset);
            cv::Point2d projected_point_cv(projected_point.x(), projected_point.y());

            // Record distance from KF edge location to projected CF location (thread-safe)
            double kf_to_proj_dist = cv::norm(current_kf_edges[last_keyframe_stereo.focused_edge_indices[i]].location - projected_point_cv);
            thread_distances[tid].push_back(kf_to_proj_dist);

            if (projected_point.x() > 10 && projected_point.y() > 10 && projected_point.x() < dataset.get_width() - 10 && projected_point.y() < dataset.get_height() - 10)
            {
                std::vector<int> current_candidate_edge_indices;
                double search_radius = 15.0 + gt_dist_threshold + 3.0; // +3 for safety margin
                current_candidate_edge_indices = spatial_grid.getCandidatesWithinRadius(projected_point_cv, search_radius);

                std::vector<int> CF_veridical_edges_indices;
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
                }

                // Debug: check if specific edges are in candidates

                if (!CF_veridical_edges_indices.empty())
                {
                    thread_kf_edges[tid].push_back(last_keyframe_stereo.focused_edge_indices[i]);
                    thread_stereo_frame_indices[tid].push_back(i);
                    thread_gt_orientations[tid].push_back(ori);
                    thread_gt_locations[tid].push_back(projected_point_cv);
                    thread_veridical_cf_edges[tid].push_back(CF_veridical_edges_indices);
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

        // Merge thread-local distances
        kf_to_projection_distances.insert(kf_to_projection_distances.end(),
                                          thread_distances[tid].begin(),
                                          thread_distances[tid].end());
    }
}

void EBVO::apply_spatial_grid_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, const Stereo_Edge_Pairs &keyframe_stereo, const std::vector<Edge> &edges, SpatialGrid &spatial_grid, double grid_radius)
{
    //> For each edge in the keyframe, find candidate edges in the current frame using spatial grid

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < KF_CF_edge_pairs.kf_edges.size(); ++i)
    {
        int kf_edge_idx = KF_CF_edge_pairs.kf_edges[i];

        KF_CF_edge_pairs.matching_cf_edges_indices[i] = spatial_grid.getCandidatesWithinRadius(edges[kf_edge_idx].location, grid_radius);
    }
}

void EBVO::apply_SIFT_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, const Stereo_Edge_Pairs &keyframe_stereo, const Stereo_Edge_Pairs &current_stereo, double sift_dist_threshold, bool is_left)
{
    //> For each edge in the keyframe, compare its SIFT descriptor with the SIFT descriptors of the edges in the current frame
    //> Filter out edges that don't meet the SIFT distance threshold

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < KF_CF_edge_pairs.kf_edges.size(); ++i)
    {
        std::vector<int> filtered_cf_edges_indices;

        // Get SIFT descriptors for the keyframe edge (two descriptors per edge)
        std::pair<cv::Mat, cv::Mat> kf_edge_descriptors = keyframe_stereo.left_edge_descriptors[i];

        // Iterate through all matching CF edges for this KF edge
        for (int cf_edge_idx : KF_CF_edge_pairs.matching_cf_edges_indices[i])
        {
            // Get the index in the Stereo_Edge_Pairs structure
            int stereo_idx = current_stereo.get_Stereo_Edge_Pairs_left_id_index(cf_edge_idx);

            if (stereo_idx >= 0 && stereo_idx < current_stereo.left_edge_descriptors.size())
            {
                // Get SIFT descriptors for the current frame edge
                std::pair<cv::Mat, cv::Mat> cf_edge_descriptors = current_stereo.left_edge_descriptors[stereo_idx];

                // Compare all combinations of descriptors (2x2 = 4 combinations)
                // Take the minimum distance to get the best match
                double sift_dist_1 = cv::norm(kf_edge_descriptors.first, cf_edge_descriptors.first, cv::NORM_L2);
                double sift_dist_2 = cv::norm(kf_edge_descriptors.first, cf_edge_descriptors.second, cv::NORM_L2);
                double sift_dist_3 = cv::norm(kf_edge_descriptors.second, cf_edge_descriptors.first, cv::NORM_L2);
                double sift_dist_4 = cv::norm(kf_edge_descriptors.second, cf_edge_descriptors.second, cv::NORM_L2);

                double min_sift_dist = std::min({sift_dist_1, sift_dist_2, sift_dist_3, sift_dist_4});

                // Keep the edge if it passes the threshold
                if (min_sift_dist < sift_dist_threshold)
                {
                    filtered_cf_edges_indices.push_back(cf_edge_idx);
                }
            }
        }

        // Update the matching indices with filtered results
        KF_CF_edge_pairs.matching_cf_edges_indices[i] = filtered_cf_edges_indices;
    }
}

void EBVO::apply_NCC_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, const Stereo_Edge_Pairs &keyframe_stereo, const Stereo_Edge_Pairs &current_stereo, double ncc_val_threshold,
                               const cv::Mat &keyframe_image, const cv::Mat &current_image, bool is_left)
{
    //> For each edge in the keyframe, compute the NCC score with the edges in the current frame
    //> Filter out edges that don't meet the NCC threshold

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < KF_CF_edge_pairs.kf_edges.size(); ++i)
    {
        int kf_edge_idx = KF_CF_edge_pairs.kf_edges[i];
        const Edge &kf_edge = keyframe_stereo.get_focused_edge_by_Stereo_Edge_Pairs_index(kf_edge_idx);

        std::vector<int> filtered_cf_edges_indices;

        // Iterate through all matching CF edges for this KF edge
        for (int cf_edge_idx : KF_CF_edge_pairs.matching_cf_edges_indices[i])
        {
            const Edge &cf_edge = current_stereo.get_focused_edge_by_Stereo_Edge_Pairs_index(cf_edge_idx);

            // Compute NCC score between edge patches
            double ncc_score = edge_patch_similarity(kf_edge, cf_edge, keyframe_image, current_image);

            // Keep the edge if it passes the threshold
            if (ncc_score > ncc_val_threshold)
            {
                filtered_cf_edges_indices.push_back(cf_edge_idx);
            }
        }

        // Update the matching indices with filtered results
        KF_CF_edge_pairs.matching_cf_edges_indices[i] = filtered_cf_edges_indices;
    }
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

void EBVO::apply_orientation_filtering(KF_CF_EdgeCorrespondence &KF_CF_edge_pairs, const Stereo_Edge_Pairs &keyframe_stereo, const Stereo_Edge_Pairs &current_stereo, double orientation_threshold, bool is_left)
{
    //> for each edge in the keyframe, compare its orientation with the edges in the current frame
    // Convert map to vector for parallel iteration

#pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < KF_CF_edge_pairs.kf_edges.size(); ++idx)
    {
        Edge kf_edge = keyframe_stereo.get_focused_edge_by_Stereo_Edge_Pairs_index(idx);
        std::vector<int> filtered_cf_edges_indices;
        for (auto &m_edge_idx : KF_CF_edge_pairs.matching_cf_edges_indices[idx])
        {
            int cf_edge_idx = current_stereo.get_Stereo_Edge_Pairs_left_id_index(m_edge_idx);
            const Edge &cf_edge = current_stereo.get_focused_edge_by_Stereo_Edge_Pairs_index(cf_edge_idx);
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
    //> FIXME: Write the results to a file
    // std::string output_dir = dataset.get_output_path();
    // std::string eval_filename = output_dir + "/edge_match_evaluation_" + stage_name + "_frame_" + std::to_string(frame_idx) + ".txt";
    // std::ofstream eval_PR(eval_filename);
    std::string output_dir = dataset.get_output_path();
    std::string stereo_filter_filename = output_dir + "/match_debug_frame_" + std::to_string(frame_idx) + ".csv";
    std::ofstream stereo_csv(stereo_filter_filename);

    int total_num_of_true_positives_for_recall = 0;
    int total_num_of_true_positives_for_precision = 0;
    std::vector<double> num_of_target_edges_per_source_edge;
    std::vector<double> precision_per_edge;
    std::vector<double> precision_pair_edge;
    //> For each left edge from the keyframe, see if the filtered edges are found in the pool of matched veridical edges on the current frame
    bool debug = (stage_name == "Mate consistency Filtering");
    std::cout << "is it debugging? " << debug << std::endl;
    num_of_target_edges_per_source_edge.reserve(keyframe_stereo.focused_edge_indices.size()); //> compailable to both left and right edges
    precision_per_edge.reserve(keyframe_stereo.focused_edge_indices.size());

    for (int i = 0; i < KF_CF_edge_pairs.kf_edges.size(); ++i)
    {
        bool fail = true;
        total_num_of_true_positives_for_precision = 0;
        //> Find if there is at least one edge index in edge_pair.matching_cf_edges_indices is found in edge_pair.veridical_cf_edges_indices
        if (KF_CF_edge_pairs.matching_cf_edges_indices[i].size() > 0)
        {
            int total_num_of_candidates = KF_CF_edge_pairs.matching_cf_edges_indices[i].size();
            for (const auto &matched_idx : KF_CF_edge_pairs.matching_cf_edges_indices[i])
            {
                int toed_cf = current_stereo.get_Stereo_Edge_Pairs_left_id_index(matched_idx);
                Edge cf_edge = current_stereo.get_focused_edge_by_Stereo_Edge_Pairs_index(toed_cf);
                if (cv::norm(cf_edge.location - KF_CF_edge_pairs.gt_location_on_cf[i]) < 3.0)
                {
                    total_num_of_true_positives_for_precision++;
                }
            }
            if (total_num_of_true_positives_for_precision > 0)
            {
                total_num_of_true_positives_for_recall++;
            }
            precision_per_edge.push_back(static_cast<double>(total_num_of_true_positives_for_precision) / static_cast<double>(total_num_of_candidates));
            precision_pair_edge.push_back(static_cast<double>(total_num_of_true_positives_for_precision) / static_cast<double>(total_num_of_candidates));
            num_of_target_edges_per_source_edge.push_back(total_num_of_candidates);
        }
    }

    double recall_per_image = static_cast<double>(total_num_of_true_positives_for_recall) / keyframe_stereo.focused_edge_indices.size();
    double precision_per_image = std::accumulate(precision_per_edge.begin(), precision_per_edge.end(), 0.0) / precision_per_edge.size();
    double precision_pair_per_image = std::accumulate(precision_pair_edge.begin(), precision_pair_edge.end(), 0.0) / precision_pair_edge.size();
    double num_of_target_edges_per_source_edge_avg = std::accumulate(num_of_target_edges_per_source_edge.begin(), num_of_target_edges_per_source_edge.end(), 0.0) / num_of_target_edges_per_source_edge.size();

    std::cout << "Stereo Edge Correspondences Evaluation: Stage: " << stage_name << " | Frame: " << frame_idx << std::endl;
    std::cout << "- Recall rate:       " << std::fixed << std::setprecision(8) << recall_per_image << std::endl;
    std::cout << "- Precision rate:    " << std::fixed << std::setprecision(8) << precision_per_image << std::endl;
    std::cout << "- Average ambiguity: " << std::fixed << std::setprecision(8) << num_of_target_edges_per_source_edge_avg << std::endl;
    std::cout << "========================================================\n"
              << std::endl;
    stereo_csv.close();
}
