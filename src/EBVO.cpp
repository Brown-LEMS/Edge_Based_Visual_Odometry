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
    std::vector<cv::Mat> left_ref_disparity_maps;

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

    //> Initialize
    StereoFrame last_keyframe, current_frame;
    
    // StereoEdgeCorrespondencesGT last_keyframe, current_frame;
    Stereo_Edge_Pairs current_frame_stereo_edge_pairs;
    Stereo_Edge_Pairs last_keyframe_stereo_edge_pairs;

    cv::Mat left_calib_inv = left_calib.inv();
    cv::Mat right_calib_inv = right_calib.inv();

    cv::Mat descriptors_t0, descriptors_t1;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    bool b_is_keyframe = true;

    size_t frame_idx = 0;
    while (dataset.stereo_iterator->hasNext() && num_pairs - frame_idx >= 0)
    {
        if (!dataset.stereo_iterator->getNext(current_frame))
        {
            break;
        }

        // If the current frame has ground truth, we can use it
        if (dataset.has_gt()) { }
        const cv::Mat &left_disparity_map = (frame_idx < left_ref_disparity_maps.size()) ? left_ref_disparity_maps[frame_idx] : cv::Mat();

        dataset.ncc_one_vs_err.clear();
        dataset.ncc_two_vs_err.clear();
        dataset.ground_truth_right_edges_after_lowe.clear();

        std::cout << std::endl << "Image Pair #" << frame_idx << std::endl;

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

            // Initialize the spatial grid with a cell size of defined GRID_SIZE
            spatial_grid = SpatialGrid(dataset.get_width(), dataset.get_height(), GRID_SIZE);
        }

        std::string edge_dir = dataset.get_output_path() + "/edges";
        std::filesystem::create_directories(edge_dir);

        std::string left_edge_path = edge_dir + "/left_edges_" + std::to_string(frame_idx + 1);
        std::string right_edge_path = edge_dir + "/right_edges_" + std::to_string(frame_idx + 1);

        //> Detect third-order edges for the keyframe    
        ProcessEdges(left_cur_undistorted, left_edge_path, TOED, current_frame.left_edges);
        std::cout << "Number of edges on the left image: " << current_frame.left_edges.size() << std::endl;

        ProcessEdges(right_cur_undistorted, right_edge_path, TOED, current_frame.right_edges);
        std::cout << "Number of edges on the right image: " << current_frame.right_edges.size() << std::endl;

        dataset.increment_num_imgs();
        std::cout << std::endl;

        

        if (b_is_keyframe)
        {
            last_keyframe_stereo_edge_pairs.stereo_frame = &current_frame;
            last_keyframe = current_frame;

            //> For each left edge, get the corresponding GT location (not right edge) on the right image, and the triangulated 3D point in the left camera coordinate
            Find_Stereo_GT_Locations(dataset, left_disparity_map, last_keyframe, last_keyframe_stereo_edge_pairs);
            std::cout << "Complete calculating GT locations for left edges of the keyframe..." << std::endl;
            std::cout << "Size of left edges with GT locations on the right image = " << last_keyframe_stereo_edge_pairs.size() << std::endl;
            
            //> Construct a GT stereo edge pool
            get_Stereo_Edge_GT_Pairs(dataset, last_keyframe, last_keyframe_stereo_edge_pairs);
            std::cout << "Size of stereo edge correspondences pool = " << last_keyframe_stereo_edge_pairs.size() << std::endl;

            //> construct stereo edge correspondences
            // StereoMatchResult match_result = get_Stereo_Edge_Pairs(
            //     left_cur_undistorted,
            //     right_cur_undistorted,
            //     last_keyframe,
            //     dataset, frame_idx
            // );

            //> extract SIFT descriptor for each left edge of last_keyframe
            augment_Edge_Data(last_keyframe_stereo_edge_pairs, last_keyframe.left_image_undistorted);
            if (!last_keyframe_stereo_edge_pairs.b_is_size_consistent()) last_keyframe_stereo_edge_pairs.print_size_consistency();

            b_is_keyframe = false;
        }
        else
        {
            current_frame_stereo_edge_pairs.stereo_frame = &current_frame;
            
            //> For each left edge, get the corresponding GT location (not right edge) on the right image, and the triangulated 3D point in the left camera coordinate
            Find_Stereo_GT_Locations(dataset, left_disparity_map, current_frame, current_frame_stereo_edge_pairs);
            std::cout << "Complete calculating GT locations for left edges of the current frame..." << std::endl;
            std::cout << "Size of left edges with GT locations on the right image = " << current_frame_stereo_edge_pairs.size() << std::endl;
            
            //> Construct a GT stereo edge pool
            get_Stereo_Edge_GT_Pairs(dataset, current_frame, current_frame_stereo_edge_pairs);
            std::cout << "Size of stereo edge correspondences pool = " << current_frame_stereo_edge_pairs.size() << std::endl;

            //> extract SIFT descriptor for each left edge of current_frame_stereo
            augment_Edge_Data(current_frame_stereo_edge_pairs, current_frame.left_image_undistorted);
            if (!current_frame_stereo_edge_pairs.b_is_size_consistent()) current_frame_stereo_edge_pairs.print_size_consistency();

            add_edges_to_spatial_grid(current_frame_stereo_edge_pairs);

            //> Construct correspondences structure between last keyframe and the current frame
            KF_CF_Edge_Pairs kf_cf_edge_pairs(&last_keyframe, &current_frame, &last_keyframe_stereo_edge_pairs, &current_frame_stereo_edge_pairs);

            // std::vector<KF_CF_Edge_Correspondences> kf_cf_edge_correspondences;
            Find_Veridical_Edge_Correspondences_on_CF(dataset, kf_cf_edge_pairs);

            std::cout << "Size of KF-CF veridical edge pairs = " << kf_cf_edge_pairs.get_left_kf_edge_pairs_size() << std::endl;
            
            // //> Now that the GT edge correspondences are constructed between the keyframe and the current frame, we can apply various filters from the beginning
            // //> Stage 1: Apply spatial grid to the current frame
            // apply_spatial_grid_filtering(KF_CF_edge_pairs, last_keyframe_stereo, 1.0);
            // Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs, last_keyframe_stereo, current_frame_stereo, frame_idx, "Spatial Grid");

            // //> Stage 2: Do SIFT descriptor comparison between the last keyframe and the current frame from the KC_edge_correspondences
            // apply_SIFT_filtering(KF_CF_edge_pairs, last_keyframe_stereo, current_frame_stereo, 700.0);
            // Evaluate_KF_CF_Edge_Correspondences(KF_CF_edge_pairs, last_keyframe_stereo, current_frame_stereo, frame_idx, "SIFT Filtering");

            //> Stage 3: Do NCC 

            break;
            
        }

        // break;

        // // Declare variables at proper scope level
        // std::vector<cv::DMatch> SIFT_matches;
        // std::unordered_map<int, cv::Mat> current_descriptors_cache;
        // if (frame_idx > 0)
        // {
        //     spatial_grid.reset();
        //     current_edge_loc.clear();

        //     std::unordered_map<Edge, std::vector<Edge>> Edge_match;
        //     #pragma omp parallel for schedule(dynamic)
        //     for (int current_idx = 0; current_idx < dataset.left_edges.size(); ++current_idx)
        //     {
        //         #pragma omp critical(spatial_grid)
        //         {
        //             spatial_grid.addEdge(current_idx, dataset.left_edges[current_idx].location);
        //         }

        //         Edge &current_edge = dataset.left_edges[current_idx];
        //         cv::KeyPoint current_kp(current_edge.location, 1, 180 / M_PI * current_edge.orientation);
        //         #pragma omp critical(edge_loc)
        //         {
        //             current_edge_loc.push_back(current_kp);
        //         }
        //         std::vector<cv::KeyPoint> edge_kp = {current_kp};
        //         cv::Mat edge_desc;
        //         sift->compute(current_frame.left_image, edge_kp, edge_desc);

        //         if (!edge_desc.empty())
        //         {
        //             #pragma omp critical(descriptor_cache)
        //             {
        //                 current_descriptors_cache[current_idx] = edge_desc.clone();
        //             }
        //         }
        //     }
        //     std::vector<Edge> gt_edges;
        //     std::unordered_map<Edge, EdgeGTMatchInfo> left_edges_GT_Info;
        //     GetGTEdges(frame_idx, previous_frame, current_frame, previous_frame_edges,
        //                left_disparity_map, left_calib_inv, left_calib, gt_edges, left_edges_GT_Info);

        //     std::cout << "Stage 1: Populating Edge_match with spatial candidates..." << std::endl;
        //     #pragma omp parallel for schedule(dynamic)
        //     for (int i = 0; i < previous_frame_edges.size(); ++i)
        //     {
        //         const Edge &prev_edge = previous_frame_edges[i];
        //         std::vector<int> indices = spatial_grid.getCandidatesWithinRadius(prev_edge, GRID_SIZE);

        //         // Debug: Check if this is the problematic edge from your example
        //         if (abs(prev_edge.location.x - 799.6) < 1.0 && abs(prev_edge.location.y - 132.9) < 1.0)
        //         {
        //             std::cout << "=== DEBUGGING PROBLEMATIC EDGE ===" << std::endl;
        //             std::cout << "Prev edge: (" << prev_edge.location.x << "," << prev_edge.location.y << ")" << std::endl;
        //             std::cout << "GRID_SIZE: " << GRID_SIZE << std::endl;
        //             std::cout << "Found " << indices.size() << " candidates:" << std::endl;

        //             bool found_gt = false;
        //             for (int idx : indices)
        //             {
        //                 const Edge &candidate = dataset.left_edges[idx];
        //                 double dist = cv::norm(candidate.location - cv::Point2d(806.2, 135.4));
        //                 // std::cout << "  Candidate " << idx << ": (" << candidate.location.x << "," << candidate.location.y << ") dist_to_GT=" << dist << std::endl;
        //                 if (dist < 2.0)
        //                     found_gt = true;
        //             }

        //             std::cout << "GT edge found in candidates: " << (found_gt ? "YES" : "NO") << std::endl;
        //             std::cout << "====================================" << std::endl;
        //         }

        //         // Add all spatial candidates to Edge_match
        //         for (int idx : indices)
        //         {
        //             const Edge &curr_edge = dataset.left_edges[idx];
        //             #pragma omp critical(edge_match)
        //             {
        //                 Edge_match[prev_edge].push_back(curr_edge);
        //             }
        //         }
        //     }

        //     // Evaluate Edge_match after spatial grid stage
        //     std::cout << "Evaluating Edge_match after spatial grid stage..." << std::endl;
        //     EvaluateEdgeMatchPerformance(Edge_match, left_edges_GT_Info, frame_idx, "spatial_grid", 5.0);

        //     // Stage 2: Filter Edge_match based on SIFT descriptors
        //     std::cout << "Stage 2: Filtering Edge_match with SIFT descriptors..." << std::endl;
        //     std::unordered_map<Edge, std::vector<Edge>> Filtered_Edge_match;

        //     // Create a vector to store matches from parallel threads
        //     std::vector<std::vector<cv::DMatch>> thread_matches(omp_get_max_threads());

        //     #pragma omp parallel for schedule(dynamic)
        //     for (int i = 0; i < previous_frame_edges.size(); ++i)
        //     {
        //         int thread_id = omp_get_thread_num();
        //         const Edge &prev_edge = previous_frame_edges[i];

        //         // Check if we have candidates for this previous edge
        //         auto edge_match_it = Edge_match.find(prev_edge);
        //         if (edge_match_it == Edge_match.end())
        //         {
        //             continue;
        //         }

        //         cv::Mat prev_descriptor = previous_frame_descriptors_cache.at(i);
        //         if (prev_descriptor.empty())
        //         {
        //             continue; // Skip if descriptor is empty
        //         }

        //         float threshold = 600.0f;
        //         std::vector<Edge> filtered_candidates;

        //         // Apply SIFT filtering to spatial candidates
        //         for (const Edge &candidate_edge : edge_match_it->second)
        //         {
        //             // Find the index of this candidate edge
        //             int idx = candidate_edge.index; // Assuming Edge has an index field

        //             // Check if we have a descriptor for the current frame candidate
        //             if (current_descriptors_cache.find(idx) == current_descriptors_cache.end())
        //             {
        //                 continue;
        //             }

        //             cv::Mat curr_descriptor = current_descriptors_cache.at(idx);
        //             if (curr_descriptor.empty())
        //             {
        //                 continue;
        //             }

        //             // Calculate L2 distance between descriptors
        //             float distance = cv::norm(prev_descriptor, curr_descriptor, cv::NORM_L2);

        //             if (distance < threshold)
        //             {
        //                 filtered_candidates.push_back(candidate_edge);

        //                 cv::DMatch match;
        //                 match.queryIdx = i;   // Previous frame edge index
        //                 match.trainIdx = idx; // Current frame edge index
        //                 match.distance = distance;
        //                 thread_matches[thread_id].push_back(match);
        //             }
        //         }

        //         // Add filtered candidates to the new Edge_match
        //         if (!filtered_candidates.empty())
        //         {
        //             #pragma omp critical(filtered_edge_match)
        //             {
        //                 Filtered_Edge_match[prev_edge] = filtered_candidates;
        //             }
        //         }
        //     }

        //     // Replace Edge_match with filtered version
        //     Edge_match = Filtered_Edge_match;

        //     // Evaluate Edge_match after SIFT filtering
        //     std::cout << "Evaluating Edge_match after SIFT filtering..." << std::endl;
        //     EvaluateEdgeMatchPerformance(Edge_match, left_edges_GT_Info, frame_idx, "sift_filtered", 5.0);

        //     // Combine matches from all threads
        //     for (const auto &thread_match_vec : thread_matches)
        //     {
        //         SIFT_matches.insert(SIFT_matches.end(), thread_match_vec.begin(), thread_match_vec.end());
        //     }

        //     if (!SIFT_matches.empty())
        //     {
        //         // std::vector<cv::DMatch> selected_matches;
        //         std::cout << "Found " << SIFT_matches.size() << " raw matches" << std::endl;
        //     }
        //     std::vector<std::pair<Edge, std::vector<Edge> *>> edge_match_vector;
        //     edge_match_vector.reserve(Edge_match.size());
        //     for (auto &match : Edge_match)
        //     {
        //         edge_match_vector.push_back({match.first, &match.second});
        //     }

        //     #pragma omp parallel for schedule(dynamic)
        //     for (int i = 0; i < edge_match_vector.size(); i++)
        //     {
        //         const Edge &curr_edge = edge_match_vector[i].first;
        //         std::vector<Edge> *candidates_ptr = edge_match_vector[i].second;
        //         std::vector<Edge> &candidates = *candidates_ptr;

        //         std::vector<Edge> filtered_candidates;

        //         for (const Edge &candidate : candidates)
        //         {
        //             double score = edge_patch_similarity(curr_edge, candidate, previous_frame.left_image, current_frame.left_image);
        //             if (!std::isnan(score) && score >= NCC_THRESH_WEAK_BOTH_SIDES)
        //             {
        //                 filtered_candidates.push_back(candidate);
        //             }
        //         }

        //         // No need for critical section since each thread works on its own portion of the map
        //         candidates = std::move(filtered_candidates);
        //     }

        //     std::cout << "Evaluating Edge_match after patch similarity filtering..." << std::endl;
        //     DebugNCCScoresWithGT(left_edges_GT_Info, frame_idx, previous_frame, current_frame);
        //     // EvaluateEdgeMatchPerformance(Edge_match, left_edges_GT_Info, frame_idx, "ncc_filtered", 5.0);

        //     // Update caches for next iteration
        //     previous_frame_descriptors_cache.clear();
        //     previous_frame_descriptors_cache = std::move(current_descriptors_cache);

        //     // Update previous edge locations for next iteration
        //     previous_edge_loc = current_edge_loc;
        //     previous_frame_edges = dataset.left_edges; // Store current edges as previous for next iteration
        // }
        // else
        // {
        //     //> MARK: Initialize for the first frame
        //     //> CH TODO: We should just loop over the left edges that "have" the corresponding right edges. This should save a lot of time.

        //     for (int i = 0; i < dataset.left_edges.size(); ++i)
        //     {
        //         const Edge &edge = dataset.left_edges[i];
        //         previous_edge_loc.push_back(cv::KeyPoint(edge.location, 1, 180 / M_PI * edge.orientation));
        //         spatial_grid.addEdge(i, edge.location);

        //         // Compute and cache descriptor for the first frame
        //         std::vector<cv::KeyPoint> edge_kp = {cv::KeyPoint(edge.location, 1, 180 / M_PI * edge.orientation)};
        //         cv::Mat edge_desc;
        //         sift->compute(current_frame.left_image, edge_kp, edge_desc);

        //         if (!edge_desc.empty())
        //         {
        //             //> SHould cache only the left edges that "have" the corresponding right edges.
        //             previous_frame_descriptors_cache[i] = edge_desc.clone();
        //         }
        //     }
        //     previous_frame_edges = dataset.left_edges; // Store first frame edges
        // }

        frame_idx++;
        if (frame_idx >= 3)
        {
            break;
        }

        // previous_frame = current_frame;
        // previous_edge_loc is now updated in the frame_idx > 0 block above
    }
}

void EBVO::augment_Edge_Data(Stereo_Edge_Pairs& stereo_frame_edge_pairs, const cv::Mat image) 
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
        for (int le : stereo_frame_edge_pairs.left_edge_indices)
        {
            const Edge left_edge = stereo_frame_edge_pairs.get_left_edge_by_StereoFrame_index(le);
            cv::KeyPoint left_edge_kp(left_edge.location, 1, 180 / M_PI * left_edge.orientation);
            thread_local_keypoints[thread_id].push_back(left_edge_kp);
        }
    }
    
    //> Merge all thread-local keypoints
    for (const auto& local_keypoints : thread_local_keypoints) {
        edge_keypoints.insert(edge_keypoints.end(), local_keypoints.begin(), local_keypoints.end());
    }

    //> Compute SIFT descriptors for all left edges
    cv::Mat edge_desc;
    sift->compute(image, edge_keypoints, edge_desc);

    //> Loop over all the rows of edge_desc and assign each to the stereo_frame.left_edge_descriptors
    for (int i = 0; i < edge_desc.rows; ++i) {
        stereo_frame_edge_pairs.left_edge_descriptors.push_back(edge_desc.row(i));
    }

    // std::cout << "Descriptors matrix size: " << edge_desc.rows << " x " << edge_desc.cols << std::endl;
}

void EBVO::add_edges_to_spatial_grid(Stereo_Edge_Pairs& stereo_frame_edge_pairs)
{
    //> Add left edges to spatial grid. This is done on the current image only. 
    // TODO: check if this can be done in parallel for faster computation
    for (int i = 0; i < stereo_frame_edge_pairs.left_edge_indices.size(); ++i)
    {
        const Edge &left_edge = stereo_frame_edge_pairs.get_left_edge_by_Stereo_Edge_Pairs_index(i);
        int grid_index = spatial_grid.add_edge_to_grids(i, left_edge.location);
        stereo_frame_edge_pairs.grid_indices.push_back(grid_index);
    }
}

void EBVO::ProcessEdges(const cv::Mat &image,
                        const std::string &filepath,
                        std::shared_ptr<ThirdOrderEdgeDetectionCPU> &toed,
                        std::vector<Edge> &edges)
{
    // std::string path = filepath + ".bin";
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

void EBVO::Find_Veridical_Edge_Correspondences_on_CF(Dataset &dataset, KF_CF_Edge_Pairs& kf_cf_edge_pairs, double gt_dist_threshold)
{
    //> Ground-truth relative pose of the current frame with respect to the keyframe
    Eigen::Matrix3d R1 = kf_cf_edge_pairs.keyframe->gt_rotation;
    Eigen::Vector3d t1 = kf_cf_edge_pairs.keyframe->gt_translation;
    Eigen::Matrix3d R2 = kf_cf_edge_pairs.current_frame->gt_rotation;
    Eigen::Vector3d t2 = kf_cf_edge_pairs.current_frame->gt_translation;

    Eigen::Matrix3d R21 = R2 * R1.transpose();
    Eigen::Vector3d t21 = t2 - R2 * R1.transpose() * t1;

    std::cout << "stereo_keyframe_edge_pairs left edge indices size: " << kf_cf_edge_pairs.stereo_keyframe_edge_pairs->left_edge_indices.size() << std::endl;

    //> For each left edge in the keyframe, find the GT location on the current image
    for (int i = 0; i < kf_cf_edge_pairs.stereo_keyframe_edge_pairs->left_edge_indices.size(); ++i)
    {
        Eigen::Vector3d projected_point = dataset.get_left_calib_matrix() * (R21 * kf_cf_edge_pairs.stereo_keyframe_edge_pairs->Gamma_in_left_cam_coord[i] + t21);
        projected_point /= projected_point.z();
        cv::Point2d projected_point_cv(projected_point.x(), projected_point.y());

        if (projected_point.x() > 10 && projected_point.y() > 10 && projected_point.x() < dataset.get_width() - 10 && projected_point.y() < dataset.get_height() - 10)
        {
            // std::cout << "stereo_keyframe_edge_pairs left edge index: " << i << std::endl;
            // std::cout << "stereo_keyframe_edge_pairs left edge location: " << kf_cf_edge_pairs.stereo_keyframe_edge_pairs->get_left_edge_by_Stereo_Edge_Pairs_index(i).location.x << ", " << kf_cf_edge_pairs.stereo_keyframe_edge_pairs->get_left_edge_by_Stereo_Edge_Pairs_index(i).location.y << std::endl;
            // std::cout << "projected point on CF: " << projected_point_cv.x << ", " << projected_point_cv.y << std::endl;
            std::vector<int> current_candidate_edge_indices = spatial_grid.getCandidatesWithinRadius(kf_cf_edge_pairs.stereo_keyframe_edge_pairs->get_left_edge_by_Stereo_Edge_Pairs_index(i).location, 1);
            // std::cout << "current candidate edge indices: " << current_candidate_edge_indices.size() << std::endl;
            // for (const auto& curr_e_index : current_candidate_edge_indices)
            // {
            //     std::cout << "current candidate edge location: " << kf_cf_edge_pairs.stereo_current_frame_edge_pairs->get_left_edge_by_Stereo_Edge_Pairs_index(curr_e_index).location.x << ", " << kf_cf_edge_pairs.stereo_current_frame_edge_pairs->get_left_edge_by_Stereo_Edge_Pairs_index(curr_e_index).location.y << std::endl;
            // }
            // break;

            std::vector<int> CF_veridical_edges_indices;
            for (const auto& curr_e_index : current_candidate_edge_indices)
            {
                //> Check if the Euclidean distance is less than some threshold
                if (cv::norm(kf_cf_edge_pairs.stereo_current_frame_edge_pairs->get_left_edge_by_Stereo_Edge_Pairs_index(curr_e_index).location - projected_point_cv) < gt_dist_threshold)
                {
                    size_t veridical_index = kf_cf_edge_pairs.stereo_current_frame_edge_pairs->get_left_edge_index_in_StereoFrame_given_Stereo_Edge_Pairs_index(curr_e_index);
                    CF_veridical_edges_indices.push_back(veridical_index);
                }
            }
            if (!CF_veridical_edges_indices.empty())
            {
                int toed_edge_index = kf_cf_edge_pairs.stereo_keyframe_edge_pairs->get_left_edge_index_in_StereoFrame_given_Stereo_Edge_Pairs_index(i);
                kf_cf_edge_pairs.left_kf_edge_indices.push_back(toed_edge_index);
                kf_cf_edge_pairs.left_gt_locations_on_cf.push_back(projected_point_cv);
                kf_cf_edge_pairs.left_veridical_cf_edges_indices.push_back(CF_veridical_edges_indices);
            }
        }
    }
#if WRITE_KF_CF_GT_EDGE_PAIRS
    std::string kf_cf_gt_edge_pairs_last_kf_filename = dataset.get_output_path() + "/kf_cf_gt_edge_pairs_KF.txt";
    std::cout << "Writing KF-CF-GT edge pairs for KF to: " << kf_cf_gt_edge_pairs_last_kf_filename << std::endl;
    std::ofstream kf_cf_gt_edge_pairs_kf_file(kf_cf_gt_edge_pairs_last_kf_filename);
    int gt_edge_idx = 0;
    for (int i = 0; i < kf_cf_edge_pairs.left_kf_edge_indices.size(); ++i)
    {
        kf_cf_gt_edge_pairs_kf_file << gt_edge_idx << "\t" \
                                    << kf_cf_edge_pairs.keyframe->left_edges[kf_cf_edge_pairs.left_kf_edge_indices[i]].location.x << "\t" \
                                    << kf_cf_edge_pairs.keyframe->left_edges[kf_cf_edge_pairs.left_kf_edge_indices[i]].location.y << "\t" \
                                    << kf_cf_edge_pairs.keyframe->left_edges[kf_cf_edge_pairs.left_kf_edge_indices[i]].orientation << "\t" \
                                    << kf_cf_edge_pairs.left_gt_locations_on_cf[i].x << "\t" \
                                    << kf_cf_edge_pairs.left_gt_locations_on_cf[i].y << "\n";
        gt_edge_idx++;
    }
    kf_cf_gt_edge_pairs_kf_file.close();

    std::string kf_cf_gt_edge_pairs_curr_kf_filename = dataset.get_output_path() + "/kf_cf_gt_edge_pairs_CF.txt";
    std::cout << "Writing KF-CF-GT edge pairs for CF to: " << kf_cf_gt_edge_pairs_curr_kf_filename << std::endl;
    std::ofstream kf_cf_gt_edge_pairs_cf_file(kf_cf_gt_edge_pairs_curr_kf_filename);
    gt_edge_idx = 0;
    for (int i = 0; i < kf_cf_edge_pairs.left_kf_edge_indices.size(); ++i)
    {
        for (const auto& v_edge_idx : kf_cf_edge_pairs.left_veridical_cf_edges_indices[i])
        {
            kf_cf_gt_edge_pairs_cf_file << gt_edge_idx << "\t" \
                                        << kf_cf_edge_pairs.current_frame->left_edges[v_edge_idx].location.x << "\t" \
                                        << kf_cf_edge_pairs.current_frame->left_edges[v_edge_idx].location.y << "\t" \
                                        << kf_cf_edge_pairs.current_frame->left_edges[v_edge_idx].orientation << "\n";
        }
        gt_edge_idx++;
    }
    kf_cf_gt_edge_pairs_cf_file.close();
#endif
}

void EBVO::apply_spatial_grid_filtering(std::vector<KF_CF_Edge_Correspondences>& KF_CF_edge_pairs, const StereoEdgeCorrespondencesGT& keyframe_stereo, double grid_radius)
{
    #pragma omp parallel for schedule(dynamic)
    for (auto& edge_pair : KF_CF_edge_pairs)
    {
        edge_pair.matching_cf_edges_indices = spatial_grid.getCandidatesWithinRadius(keyframe_stereo.left_edges[edge_pair.kf_edge_index].location, grid_radius);
    }
}

void EBVO::apply_SIFT_filtering(std::vector<KF_CF_Edge_Correspondences>& KF_CF_edge_pairs, const StereoEdgeCorrespondencesGT& keyframe_stereo, const StereoEdgeCorrespondencesGT& current_stereo, double sift_dist_threshold)
{
    //> for each edge in the keyframe, compare its SIFT descriptor with the SIFT descriptors of the edges in the current frame
    #pragma omp parallel for schedule(dynamic)
    for (auto& edge_pair : KF_CF_edge_pairs)
    {
        std::vector<int> filtered_cf_edges_indices;
        for (auto& m_edge_idx : edge_pair.matching_cf_edges_indices)
        {   
            double sift_desc_distance = cv::norm(keyframe_stereo.left_edge_descriptors[edge_pair.kf_edge_index], current_stereo.left_edge_descriptors[m_edge_idx], cv::NORM_L2);
            if (sift_desc_distance < sift_dist_threshold)
            {
                filtered_cf_edges_indices.push_back(m_edge_idx);
            }
        }
        edge_pair.matching_cf_edges_indices = filtered_cf_edges_indices;
    }
}

Edge EBVO::GetGTEdge(bool left, StereoFrame &current_frame, StereoFrame &next_frame,
                     const cv::Mat &disparity_map, const cv::Mat &K_inverse, const cv::Mat &K,
                     const Edge &edge)
{
    double disparity = Bilinear_Interpolation(disparity_map, edge.location);
    if (std::isnan(disparity) || std::isinf(disparity) || disparity < 0)
    {
        // std::cout << "Invalid disparity value: " << disparity << " for edge at location: " << edge.location << std::endl;
        return Edge(); // Return an empty edge if disparity is invalid
    }

    double rho = dataset.get_left_focal_length() * dataset.get_left_baseline() / disparity;
    double rho_1 = (rho < 0.0) ? (-rho) : (rho);

    // Convert cv::Mat to Eigen::Matrix3d
    Eigen::Matrix3d K_inverse_eigen, K_eigen;
    cv::cv2eigen(K_inverse, K_inverse_eigen);
    cv::cv2eigen(K, K_eigen);

    Eigen::Vector3d gamma_1 = K_inverse_eigen * Eigen::Vector3d(edge.location.x, edge.location.y, 1.0);
    Eigen::Vector3d Gamma_1 = rho_1 * gamma_1;

    Eigen::Matrix3d R1 = current_frame.gt_rotation;
    Eigen::Matrix3d R2 = next_frame.gt_rotation;
    Eigen::Vector3d t1 = current_frame.gt_translation;
    Eigen::Vector3d t2 = next_frame.gt_translation;

    Eigen::Vector3d Gamma_2 = (R2 * R1.transpose()) * Gamma_1 + t2 - R2 * R1.transpose() * t1;
    Eigen::Vector3d homogeneous_point = K_eigen * Gamma_2;
    Eigen::Vector2d projected_point = homogeneous_point.head<2>() / homogeneous_point.z();

    Edge gt_edge;
    gt_edge.location = cv::Point2d(projected_point.x(), projected_point.y());
    gt_edge.orientation = edge.orientation; // orientation shouldn't change that much.
    gt_edge.b_isEmpty = false;

    return gt_edge;
}

void EBVO::GetGTEdges(size_t &frame_idx, StereoFrame &previous_frame, StereoFrame &current_frame,
                      const std::vector<Edge> &previous_frame_edges,
                      const cv::Mat &left_disparity_map, const cv::Mat &left_calib_inv,
                      const cv::Mat &left_calib, std::vector<Edge> &gt_edges,
                      std::unordered_map<Edge, EdgeGTMatchInfo> &left_edges_GT_Info)
{
    std::string output_dir = dataset.get_output_path();
    std::string csv_filename = output_dir + "/gt_correspondences_frame_" + std::to_string(frame_idx) + ".csv";
    std::ofstream gt_csv(csv_filename);

    // Write CSV header
    gt_csv << "edge_index,prev_x,prev_y,curr_x,curr_y,orientation\n";
    left_edges_GT_Info.clear();

    for (int i = 0; i < previous_frame_edges.size(); ++i)
    {

        Edge GTEdge = GetGTEdge(true, previous_frame, current_frame,
                                left_disparity_map, left_calib_inv, left_calib,
                                previous_frame_edges[i]);
        if (GTEdge.b_isEmpty)
            continue;
        else
        {
            std::vector<int> edge_idx = spatial_grid.getCandidatesWithinRadius(GTEdge, 1);

            EdgeGTMatchInfo match_info;
            double avg_orientation;
            match_info.edge = previous_frame_edges[i];
            match_info.gt_edge = GTEdge;
            for (int idx : edge_idx)
            {
                if (std::abs(dataset.left_edges[idx].location.x - GTEdge.location.x) < 1.0 && std::abs(dataset.left_edges[idx].location.y - GTEdge.location.y) < 1.0)
                {
                    match_info.vertical_edges.push_back(dataset.left_edges[idx]);
                    avg_orientation += dataset.left_edges[idx].orientation;
                }
            }
            if (match_info.vertical_edges.size() == 0)
                continue;

            avg_orientation /= match_info.vertical_edges.size();
            match_info.gt_edge.orientation = avg_orientation;
            left_edges_GT_Info[previous_frame_edges[i]] = match_info;

            const Edge &prev_edge = previous_frame_edges[i];
            gt_csv << i << ","
                   << std::fixed << std::setprecision(2) << prev_edge.location.x << ","
                   << std::fixed << std::setprecision(2) << prev_edge.location.y << ","
                   << std::fixed << std::setprecision(2) << GTEdge.location.x << ","
                   << std::fixed << std::setprecision(2) << GTEdge.location.y << ","
                   << std::fixed << std::setprecision(4) << prev_edge.orientation << "\n";
        }
    }
    gt_csv.close();
    std::cout << "Saved ground truth correspondences to: " << csv_filename << std::endl;

    std::cout << left_edges_GT_Info.size() << std::endl;
}

// Add this function to your EBVO.cpp file to debug NCC scores against GT data
void EBVO::DebugNCCScoresWithGT(const std::unordered_map<Edge, EdgeGTMatchInfo> &gt_correspondences,
                                size_t frame_idx, const StereoFrame &previous_frame,
                                const StereoFrame &current_frame)
{
    std::string output_dir = dataset.get_output_path();
    std::string debug_filename = output_dir + "/ncc_scores_gt_debug_frame_" + std::to_string(frame_idx) + ".csv";
    std::ofstream debug_csv(debug_filename);

    // Write CSV header
    debug_csv << "prev_edge_idx,prev_x,prev_y,prev_orientation,candidate_type,candidate_idx,candidate_x,candidate_y,candidate_orientation,gt_x,gt_y,ncc_score,is_gt_match\n";

    int prev_edge_idx = 0;
    int gt_matches_found = 0;
    int total_matches_with_high_ncc = 0;
    int total_gt_with_high_ncc = 0;

    std::string vis_dir = output_dir + "/patch_visualization";
    std::filesystem::create_directories(vis_dir);

    const int num_edges_to_visualize = 5;
    int edges_visualized = 0;

    for (const auto &gt_pair : gt_correspondences)
    {
        const Edge &prev_edge = gt_pair.first;
        const EdgeGTMatchInfo &match_info = gt_pair.second;
        const Edge &gt_edge = match_info.gt_edge;
        const std::vector<Edge> &vertical_edges = match_info.vertical_edges;

        // Compute NCC score for the ground truth match
        double gt_ncc_score = edge_patch_similarity(prev_edge, gt_edge,
                                                    previous_frame.left_image,
                                                    current_frame.left_image);

        // Record the ground truth match
        debug_csv << prev_edge_idx << ","
                  << std::fixed << std::setprecision(3)
                  << prev_edge.location.x << ","
                  << prev_edge.location.y << ","
                  << prev_edge.orientation << ","
                  << "GT" << "," // This is a ground truth match
                  << "-1" << "," // No candidate index for GT
                  << gt_edge.location.x << ","
                  << gt_edge.location.y << ","
                  << gt_edge.orientation << ","
                  << gt_edge.location.x << "," // GT coords are the same
                  << gt_edge.location.y << ","
                  << gt_ncc_score << ","
                  << "1" << "\n"; // is_gt_match = true

        // If GT has good NCC score, count it
        if (!std::isnan(gt_ncc_score) && gt_ncc_score >= NCC_THRESH_WEAK_BOTH_SIDES)
        {
            total_gt_with_high_ncc++;
        }
        //> CH TODO: Avoid using OpenCV to visualize. Use Python or MATLAB instead.
        // if (edges_visualized < num_edges_to_visualize)
        // {
        //     std::string edge_dir = vis_dir + "/edge_" + std::to_string(prev_edge_idx);
        //     std::filesystem::create_directories(edge_dir);

        //     // 1. Create visualization of previous frame with the edge
        //     cv::Mat prev_frame_vis = previous_frame.left_image.clone();
        //     // cv::cvtColor(prev_frame_vis, prev_frame_vis, cv::COLOR_GRAY2BGR);

        //     // Draw the previous edge point
        //     // cv::circle(prev_frame_vis, cv::Point(prev_edge.location.x, prev_edge.location.y), 5, cv::Scalar(0, 0, 255), -1);

        //     // Get orthogonal shifted points for patch visualization
        //     std::pair<cv::Point2d, cv::Point2d> shifted_points_prev = get_Orthogonal_Shifted_Points(prev_edge);

        //     // Draw orthogonal direction lines
        //     // cv::line(prev_frame_vis, cv::Point(prev_edge.location.x, prev_edge.location.y),
        //     //          cv::Point(prev_edge.location.x + 20 * cos(prev_edge.orientation),
        //     //                    prev_edge.location.y + 20 * sin(prev_edge.orientation)),
        //     //          cv::Scalar(255, 0, 0), 2);

        //     // Draw perpendicular direction
        //     // cv::line(prev_frame_vis, cv::Point(prev_edge.location.x, prev_edge.location.y),
        //     //          cv::Point(prev_edge.location.x + 20 * cos(prev_edge.orientation + M_PI / 2),
        //     //                    prev_edge.location.y + 20 * sin(prev_edge.orientation + M_PI / 2)),
        //     //          cv::Scalar(0, 255, 0), 2);

        //     // Draw patch boundaries
        //     int half_patch = PATCH_SIZE / 2;
        //     // cv::rectangle(prev_frame_vis,
        //     //               cv::Point(shifted_points_prev.first.x - half_patch, shifted_points_prev.first.y - half_patch),
        //     //               cv::Point(shifted_points_prev.first.x + half_patch, shifted_points_prev.first.y + half_patch),
        //     //               cv::Scalar(0, 255, 0), 2);
        //     // cv::rectangle(prev_frame_vis,
        //     //               cv::Point(shifted_points_prev.second.x - half_patch, shifted_points_prev.second.y - half_patch),
        //     //               cv::Point(shifted_points_prev.second.x + half_patch, shifted_points_prev.second.y + half_patch),
        //     //               cv::Scalar(255, 0, 0), 2);

        //     // Add text labels
        //     // std::string edge_info = "Edge " + std::to_string(prev_edge_idx) + " (" +
        //     //                         std::to_string(int(prev_edge.location.x)) + "," +
        //     //                         std::to_string(int(prev_edge.location.y)) + ")";
        //     // cv::putText(prev_frame_vis, edge_info, cv::Point(20, 30),
        //     //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        //     // cv::imwrite(edge_dir + "/1_previous_frame_edge.png", prev_frame_vis);

        //     // 2. Create visualization of current frame with the GT edge
        //     // cv::Mat curr_frame_gt_vis = current_frame.left_image.clone();
        //     // cv::cvtColor(curr_frame_gt_vis, curr_frame_gt_vis, cv::COLOR_GRAY2BGR);

        //     // Draw the GT edge point
        //     // cv::circle(curr_frame_gt_vis, cv::Point(gt_edge.location.x, gt_edge.location.y), 5, cv::Scalar(0, 0, 255), -1);

        //     // Draw orthogonal direction lines for GT
        //     // cv::line(curr_frame_gt_vis, cv::Point(gt_edge.location.x, gt_edge.location.y),
        //     //          cv::Point(gt_edge.location.x + 20 * cos(gt_edge.orientation),
        //     //                    gt_edge.location.y + 20 * sin(gt_edge.orientation)),
        //     //          cv::Scalar(255, 0, 0), 2);

        //     // Draw perpendicular direction
        //     // cv::line(curr_frame_gt_vis, cv::Point(gt_edge.location.x, gt_edge.location.y),
        //     //          cv::Point(gt_edge.location.x + 20 * cos(gt_edge.orientation + M_PI / 2),
        //     //                    gt_edge.location.y + 20 * sin(gt_edge.orientation + M_PI / 2)),
        //     //          cv::Scalar(0, 255, 0), 2);

        //     // Add text labels
        //     // std::string gt_info = "GT Edge (" +
        //     //                       std::to_string(int(gt_edge.location.x)) + "," +
        //     //                       std::to_string(int(gt_edge.location.y)) + ")";
        //     // cv::putText(curr_frame_gt_vis, gt_info, cv::Point(20, 30),
        //     //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        //     // std::string ncc_info = "NCC Score: " + std::to_string(gt_ncc_score);
        //     // cv::putText(curr_frame_gt_vis, ncc_info, cv::Point(20, 60),
        //     //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        //     // cv::imwrite(edge_dir + "/2_current_frame_gt_edge.png", curr_frame_gt_vis);

        //     // 3. Create combined visualization with GT and all vertical edges
        //     // cv::Mat combined_vis = current_frame.left_image.clone();
        //     // cv::cvtColor(combined_vis, combined_vis, cv::COLOR_GRAY2BGR);

        //     // Draw the GT edge in magenta
        //     // cv::circle(combined_vis, cv::Point(gt_edge.location.x, gt_edge.location.y), 5, cv::Scalar(255, 0, 255), -1);

        //     // Add information about number of vertical edges
        //     // std::string vert_count = "Vertical edges: " + std::to_string(vertical_edges.size());
        //     // cv::putText(combined_vis, vert_count, cv::Point(20, 30),
        //                 // cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        //     for (size_t i = 0; i < vertical_edges.size(); i++)
        //     {
        //         const Edge &vert_edge = vertical_edges[i];
        //         double distance = cv::norm(vert_edge.location - gt_edge.location);
        //         double ncc = edge_patch_similarity(prev_edge, vert_edge,
        //                                            previous_frame.left_image,
        //                                            current_frame.left_image);

        //         // Color based on NCC score (green for high, red for low)
        //         cv::Scalar color;
        //         if (!std::isnan(ncc) && ncc >= NCC_THRESH_WEAK_BOTH_SIDES)
        //         {
        //             color = cv::Scalar(0, 255, 0); // Green for good match
        //         }
        //         else
        //         {
        //             color = cv::Scalar(0, 0, 255); // Red for poor match
        //         }

        //         cv::circle(combined_vis, cv::Point(vert_edge.location.x, vert_edge.location.y), 3, color, -1);

        //         // Add index number next to each edge
        //         cv::putText(combined_vis, std::to_string(i),
        //                     cv::Point(vert_edge.location.x + 5, vert_edge.location.y),
        //                     cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

        //         // Create individual visualization for each vertical edge
        //         if (i < 5)
        //         { // Limit to first 5 vertical edges to avoid too many images
        //             cv::Mat vert_vis = current_frame.left_image.clone();
        //             cv::cvtColor(vert_vis, vert_vis, cv::COLOR_GRAY2BGR);

        //             // Draw the vertical edge
        //             cv::circle(vert_vis, cv::Point(vert_edge.location.x, vert_edge.location.y), 5, color, -1);

        //             // Draw GT edge as reference (smaller, semi-transparent)
        //             cv::circle(vert_vis, cv::Point(gt_edge.location.x, gt_edge.location.y), 3,
        //                        cv::Scalar(255, 0, 255), 1);

        //             // Add information text
        //             std::string vert_info = "Vertical Edge " + std::to_string(i) + " (" +
        //                                     std::to_string(int(vert_edge.location.x)) + "," +
        //                                     std::to_string(int(vert_edge.location.y)) + ")";
        //             cv::putText(vert_vis, vert_info, cv::Point(20, 30),
        //                         cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        //             std::string dist_info = "Distance to GT: " + std::to_string(distance);
        //             cv::putText(vert_vis, dist_info, cv::Point(20, 60),
        //                         cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        //             std::string ncc_text = "NCC: " + (std::isnan(ncc) ? "NaN" : std::to_string(ncc));
        //             cv::putText(vert_vis, ncc_text, cv::Point(20, 90),
        //                         cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        //             cv::imwrite(edge_dir + "/3_vertical_edge_" + std::to_string(i) + ".png", vert_vis);
        //         }
        //     }
        //     cv::imwrite(edge_dir + "/4_all_vertical_edges.png", combined_vis);

        //     // Save patch images and statistics
        //     std::ofstream stats_file(edge_dir + "/patch_statistics.txt");
        //     stats_file << "PATCH STATISTICS FOR EDGE " << prev_edge_idx << std::endl;
        //     stats_file << "--------------------------------------" << std::endl;
        //     stats_file << "Previous Edge Location: (" << prev_edge.location.x << ", "
        //                << prev_edge.location.y << ")" << std::endl;
        //     stats_file << "Previous Edge Orientation: " << prev_edge.orientation << std::endl;
        //     stats_file << "GT Edge Location: (" << gt_edge.location.x << ", "
        //                << gt_edge.location.y << ")" << std::endl;
        //     stats_file << "GT Edge Orientation: " << gt_edge.orientation << std::endl;
        //     stats_file << "GT NCC Score: " << gt_ncc_score << std::endl
        //                << std::endl;

        //     // Extract and save patches for both previous and GT edges
        //     std::pair<cv::Point2d, cv::Point2d> shifted_points_gt = get_Orthogonal_Shifted_Points(gt_edge);

        //     cv::Mat patch_coord_x_plus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        //     cv::Mat patch_coord_y_plus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        //     cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        //     cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

        //     cv::Mat prev_patch_plus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        //     cv::Mat prev_patch_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        //     cv::Mat gt_patch_plus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        //     cv::Mat gt_patch_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

        //     // Extract patches
        //     get_patch_on_one_edge_side(shifted_points_prev.first, prev_edge.orientation,
        //                                patch_coord_x_plus, patch_coord_y_plus,
        //                                prev_patch_plus, previous_frame.left_image);

        //     get_patch_on_one_edge_side(shifted_points_prev.second, prev_edge.orientation,
        //                                patch_coord_x_minus, patch_coord_y_minus,
        //                                prev_patch_minus, previous_frame.left_image);

        //     get_patch_on_one_edge_side(shifted_points_gt.first, gt_edge.orientation,
        //                                patch_coord_x_plus, patch_coord_y_plus,
        //                                gt_patch_plus, current_frame.left_image);

        //     get_patch_on_one_edge_side(shifted_points_gt.second, gt_edge.orientation,
        //                                patch_coord_x_minus, patch_coord_y_minus,
        //                                gt_patch_minus, current_frame.left_image);

        //     // Save normalized patch images
        //     cv::Mat norm_patch;

        //     cv::normalize(prev_patch_plus, norm_patch, 0, 255, cv::NORM_MINMAX, CV_8U);
        //     cv::resize(norm_patch, norm_patch, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
        //     cv::imwrite(edge_dir + "/patch_prev_plus.png", norm_patch);

        //     cv::normalize(prev_patch_minus, norm_patch, 0, 255, cv::NORM_MINMAX, CV_8U);
        //     cv::resize(norm_patch, norm_patch, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
        //     cv::imwrite(edge_dir + "/patch_prev_minus.png", norm_patch);

        //     cv::normalize(gt_patch_plus, norm_patch, 0, 255, cv::NORM_MINMAX, CV_8U);
        //     cv::resize(norm_patch, norm_patch, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
        //     cv::imwrite(edge_dir + "/patch_gt_plus.png", norm_patch);

        //     cv::normalize(gt_patch_minus, norm_patch, 0, 255, cv::NORM_MINMAX, CV_8U);
        //     cv::resize(norm_patch, norm_patch, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
        //     cv::imwrite(edge_dir + "/patch_gt_minus.png", norm_patch);

        //     // Calculate and save patch statistics
        //     double mean_prev_plus = cv::mean(prev_patch_plus)[0];
        //     double mean_prev_minus = cv::mean(prev_patch_minus)[0];
        //     double mean_gt_plus = cv::mean(gt_patch_plus)[0];
        //     double mean_gt_minus = cv::mean(gt_patch_minus)[0];

        //     double var_prev_plus = cv::mean((prev_patch_plus - mean_prev_plus).mul(prev_patch_plus - mean_prev_plus))[0];
        //     double var_prev_minus = cv::mean((prev_patch_minus - mean_prev_minus).mul(prev_patch_minus - mean_prev_minus))[0];
        //     double var_gt_plus = cv::mean((gt_patch_plus - mean_gt_plus).mul(gt_patch_plus - mean_gt_plus))[0];
        //     double var_gt_minus = cv::mean((gt_patch_minus - mean_gt_minus).mul(gt_patch_minus - mean_gt_minus))[0];

        //     stats_file << "Patch Statistics:" << std::endl;
        //     stats_file << "  Prev Plus  - Mean: " << mean_prev_plus << ", Variance: " << var_prev_plus << std::endl;
        //     stats_file << "  Prev Minus - Mean: " << mean_prev_minus << ", Variance: " << var_prev_minus << std::endl;
        //     stats_file << "  GT Plus    - Mean: " << mean_gt_plus << ", Variance: " << var_gt_plus << std::endl;
        //     stats_file << "  GT Minus   - Mean: " << mean_gt_minus << ", Variance: " << var_gt_minus << std::endl;
        //     stats_file << std::endl;

        //     // Add vertical edge statistics
        //     stats_file << "Vertical Edges:" << std::endl;
        //     for (size_t i = 0; i < vertical_edges.size(); i++)
        //     {
        //         const Edge &vert_edge = vertical_edges[i];
        //         double ncc = edge_patch_similarity(prev_edge, vert_edge,
        //                                            previous_frame.left_image,
        //                                            current_frame.left_image);

        //         stats_file << "  Edge " << i << " - Location: ("
        //                    << vert_edge.location.x << ", " << vert_edge.location.y
        //                    << "), NCC: " << ncc << std::endl;
        //     }

        //     stats_file.close();
        //     edges_visualized++;
        // }

        // Now check all vertical edges
        int candidate_idx = 0;
        bool found_gt_among_candidates = false;

        for (const Edge &candidate : vertical_edges)
        {
            double distance_to_gt = cv::norm(candidate.location - gt_edge.location);
            bool is_gt_match = (distance_to_gt < 2.0); // Consider matches within 2 pixels as "the same as GT"

            double ncc_score = edge_patch_similarity(prev_edge, candidate,
                                                     previous_frame.left_image,
                                                     current_frame.left_image);

            // Record the candidate match
            debug_csv << prev_edge_idx << ","
                      << std::fixed << std::setprecision(3)
                      << prev_edge.location.x << ","
                      << prev_edge.location.y << ","
                      << prev_edge.orientation << ","
                      << "Candidate" << ","
                      << candidate_idx << ","
                      << candidate.location.x << ","
                      << candidate.location.y << ","
                      << candidate.orientation << ","
                      << gt_edge.location.x << ","
                      << gt_edge.location.y << ","
                      << ncc_score << ","
                      << (is_gt_match ? "1" : "0") << "\n";

            // Track matches that have good NCC scores
            if (!std::isnan(ncc_score) && ncc_score >= NCC_THRESH_WEAK_BOTH_SIDES)
            {
                total_matches_with_high_ncc++;
                if (is_gt_match)
                {
                    found_gt_among_candidates = true;
                }
            }

            candidate_idx++;
        }

        if (found_gt_among_candidates)
        {
            gt_matches_found++;
        }

        prev_edge_idx++;
    }

    debug_csv.close();

    // Print summary statistics
    std::cout << "\n=== NCC SCORE DEBUG ANALYSIS FRAME " << frame_idx << " ===" << std::endl;
    std::cout << "Total GT correspondences analyzed: " << gt_correspondences.size() << std::endl;
    std::cout << "GT correspondences with high NCC scores: " << total_gt_with_high_ncc
              << " (" << std::fixed << std::setprecision(2)
              << (gt_correspondences.size() > 0 ? 100.0 * total_gt_with_high_ncc / gt_correspondences.size() : 0)
              << "%)" << std::endl;
    std::cout << "GT matches found among candidates with high NCC: " << gt_matches_found
              << " (" << std::fixed << std::setprecision(2)
              << (gt_correspondences.size() > 0 ? 100.0 * gt_matches_found / gt_correspondences.size() : 0)
              << "%)" << std::endl;
    std::cout << "Total candidates with high NCC scores: " << total_matches_with_high_ncc << std::endl;
    std::cout << "NCC score analysis saved to: " << debug_filename << std::endl;
    std::cout << "========================================================\n"
              << std::endl;
}

void EBVO::Evaluate_KF_CF_Edge_Correspondences(const std::vector<KF_CF_Edge_Correspondences>& KF_CF_edge_pairs, \
    StereoEdgeCorrespondencesGT& keyframe_stereo, StereoEdgeCorrespondencesGT& current_stereo, \
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
    for (const auto& edge_pair : KF_CF_edge_pairs)
    {        
        //> Find if there is at least one edge index in edge_pair.matching_cf_edges_indices is found in edge_pair.veridical_cf_edges_indices
        for (const auto& v_edge_idx : edge_pair.veridical_cf_edges_indices)
        {
            if (std::find(edge_pair.matching_cf_edges_indices.begin(), edge_pair.matching_cf_edges_indices.end(), v_edge_idx) != edge_pair.matching_cf_edges_indices.end())
            {
                total_num_of_true_positives++;
                precision_per_edge.push_back( static_cast<double>(edge_pair.veridical_cf_edges_indices.size()) / static_cast<double>(edge_pair.matching_cf_edges_indices.size()) );
                break;
            }
        }
        num_of_cf_edges_per_kf_edge.push_back( edge_pair.matching_cf_edges_indices.size() );
    }

    double recall_per_image = static_cast<double>(total_num_of_true_positives) / KF_CF_edge_pairs.size();
    double precision_per_image = std::accumulate(precision_per_edge.begin(), precision_per_edge.end(), 0.0) / precision_per_edge.size();
    double num_of_cf_edges_per_kf_edge_avg = std::accumulate(num_of_cf_edges_per_kf_edge.begin(), num_of_cf_edges_per_kf_edge.end(), 0.0) / num_of_cf_edges_per_kf_edge.size();

    std::cout << "Stage: " << stage_name << " | Frame: " << frame_idx << std::endl;
    std::cout << "- Recall rate:       " << std::fixed << std::setprecision(8) << recall_per_image << std::endl;
    std::cout << "- Precision rate:    " << std::fixed << std::setprecision(8) << precision_per_image << std::endl;
    std::cout << "- Average ambiguity: " << std::fixed << std::setprecision(8) << num_of_cf_edges_per_kf_edge_avg << std::endl;
    std::cout << "========================================================\n" << std::endl;
}

void EBVO::EvaluateEdgeMatchPerformance(const std::unordered_map<Edge, std::vector<Edge>> &Edge_match,
                                        const std::unordered_map<Edge, EdgeGTMatchInfo> &gt_correspondences,
                                        size_t frame_idx,
                                        const std::string &stage_name,
                                        double distance_threshold)
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
            total_individual_matches++; // NEW: Count every individual match

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