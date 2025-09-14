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
    StereoFrame previous_frame, current_frame, next_frame;
    std::vector<cv::KeyPoint> previous_edge_loc, current_edge_loc; // store the edge locations for the previous and current frames
    std::vector<Edge> previous_frame_edges;                        // Store previous frame edges for spatial grid evaluation

    cv::Mat left_calib_inv = left_calib.inv();
    cv::Mat right_calib_inv = right_calib.inv();

    cv::Mat descriptors_t0, descriptors_t1;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Get the first frame from the dataset

    // if (!dataset.stereo_iterator->hasNext() ||
    //     !dataset.stereo_iterator->getNext(current_frame))
    // {
    //     LOG_ERROR("Failed to get first frame from dataset");
    //     return;
    // }

    size_t frame_idx = 0;
    while (dataset.stereo_iterator->hasNext() && num_pairs - frame_idx >= 0)
    {
        if (!dataset.stereo_iterator->getNext(current_frame))
        {
            break;
        }

        const cv::Mat &curr_left_img = current_frame.left_image;
        const cv::Mat &curr_right_img = current_frame.right_image;

        // If the current frame has ground truth, we can use it
        if (dataset.has_gt())
        {
        }
        const cv::Mat &left_ref_map = (frame_idx < left_ref_disparity_maps.size()) ? left_ref_disparity_maps[frame_idx] : cv::Mat();

        dataset.ncc_one_vs_err.clear();
        dataset.ncc_two_vs_err.clear();
        dataset.ground_truth_right_edges_after_lowe.clear();

        std::cout << "Image Pair #" << frame_idx << "\n";

        cv::Mat left_prev_undistorted, right_prev_undistorted, left_cur_undistorted, right_cur_undistorted;

        cv::undistort(curr_left_img, left_cur_undistorted, left_calib, left_dist_coeff_mat);
        cv::undistort(curr_right_img, right_cur_undistorted, right_calib, right_dist_coeff_mat);

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

        ProcessEdges(left_cur_undistorted, left_edge_path, TOED, dataset.left_edges);
        std::cout << "Number of edges on the left image: " << dataset.left_edges.size() << std::endl;

        ProcessEdges(right_cur_undistorted, right_edge_path, TOED, dataset.right_edges);
        std::cout << "Number of edges on the right image: " << dataset.right_edges.size() << std::endl;

        dataset.increment_num_imgs();

        cv::Mat left_edge_map = cv::Mat::zeros(left_cur_undistorted.size(), CV_8UC1);
        cv::Mat right_edge_map = cv::Mat::zeros(right_cur_undistorted.size(), CV_8UC1);

        // for (const auto &edge : dataset.left_edges)
        // {
        //     if (edge.location.x >= 0 && edge.location.x < left_edge_map.cols && edge.location.y >= 0 && edge.location.y < left_edge_map.rows)
        //     {
        //         left_edge_map.at<uchar>(cv::Point(edge.location.x, edge.location.y)) = 255;
        //     }
        // }

        // for (const auto &edge : dataset.right_edges)
        // {
        //     if (edge.location.x >= 0 && edge.location.x < right_edge_map.cols && edge.location.y >= 0 && edge.location.y < right_edge_map.rows)
        //     {
        //         right_edge_map.at<uchar>(cv::Point(edge.location.x, edge.location.y)) = 255;
        //     }
        // }

        CalculateGTRightEdge(dataset.left_edges, left_ref_map, left_edge_map, right_edge_map);
        std::cout << "Complete calculating GT right edges..." << std::endl;

        // Declare variables at proper scope level
        std::vector<cv::DMatch> SIFT_matches;
        std::unordered_map<int, cv::Mat> current_descriptors_cache;
        if (frame_idx > 0)
        {
            spatial_grid.reset();
            current_edge_loc.clear();

            std::unordered_map<Edge, std::vector<Edge>> Edge_match;
#pragma omp parallel for schedule(dynamic)
            for (int current_idx = 0; current_idx < dataset.left_edges.size(); ++current_idx)
            {
#pragma omp critical(spatial_grid)
                {
                    spatial_grid.addEdge(current_idx, dataset.left_edges[current_idx].location);
                }

                Edge &current_edge = dataset.left_edges[current_idx];
                cv::KeyPoint current_kp(current_edge.location, 1, 180 / M_PI * current_edge.orientation);
#pragma omp critical(edge_loc)
                {
                    current_edge_loc.push_back(current_kp);
                }
                std::vector<cv::KeyPoint> edge_kp = {current_kp};
                cv::Mat edge_desc;
                sift->compute(current_frame.left_image, edge_kp, edge_desc);

                if (!edge_desc.empty())
                {
#pragma omp critical(descriptor_cache)
                    {
                        current_descriptors_cache[current_idx] = edge_desc.clone();
                    }
                }
            }
            std::vector<Edge> gt_edges;
            std::unordered_map<Edge, EdgeGTMatchInfo> left_edges_GT_Info;
            GetGTEdges(frame_idx, previous_frame, current_frame, previous_frame_edges,
                       left_ref_map, left_calib_inv, left_calib, gt_edges, left_edges_GT_Info);

            std::cout << "Stage 1: Populating Edge_match with spatial candidates..." << std::endl;
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < previous_frame_edges.size(); ++i)
            {
                const Edge &prev_edge = previous_frame_edges[i];
                std::vector<int> indices = spatial_grid.getCandidatesWithinRadius(prev_edge, GRID_SIZE);

                // Debug: Check if this is the problematic edge from your example
                if (abs(prev_edge.location.x - 799.6) < 1.0 && abs(prev_edge.location.y - 132.9) < 1.0)
                {
                    std::cout << "=== DEBUGGING PROBLEMATIC EDGE ===" << std::endl;
                    std::cout << "Prev edge: (" << prev_edge.location.x << "," << prev_edge.location.y << ")" << std::endl;
                    std::cout << "GRID_SIZE: " << GRID_SIZE << std::endl;
                    std::cout << "Found " << indices.size() << " candidates:" << std::endl;

                    bool found_gt = false;
                    for (int idx : indices)
                    {
                        const Edge &candidate = dataset.left_edges[idx];
                        double dist = cv::norm(candidate.location - cv::Point2d(806.2, 135.4));
                        // std::cout << "  Candidate " << idx << ": (" << candidate.location.x << "," << candidate.location.y << ") dist_to_GT=" << dist << std::endl;
                        if (dist < 2.0)
                            found_gt = true;
                    }

                    std::cout << "GT edge found in candidates: " << (found_gt ? "YES" : "NO") << std::endl;
                    std::cout << "====================================" << std::endl;
                }

                // Add all spatial candidates to Edge_match
                for (int idx : indices)
                {
                    const Edge &curr_edge = dataset.left_edges[idx];
#pragma omp critical(edge_match)
                    {
                        Edge_match[prev_edge].push_back(curr_edge);
                    }
                }
            }

            // Evaluate Edge_match after spatial grid stage
            std::cout << "Evaluating Edge_match after spatial grid stage..." << std::endl;
            EvaluateEdgeMatchPerformance(Edge_match, left_edges_GT_Info, frame_idx, "spatial_grid", 5.0);

            // Stage 2: Filter Edge_match based on SIFT descriptors
            std::cout << "Stage 2: Filtering Edge_match with SIFT descriptors..." << std::endl;
            std::unordered_map<Edge, std::vector<Edge>> Filtered_Edge_match;

            // Create a vector to store matches from parallel threads
            std::vector<std::vector<cv::DMatch>> thread_matches(omp_get_max_threads());

#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < previous_frame_edges.size(); ++i)
            {
                int thread_id = omp_get_thread_num();
                const Edge &prev_edge = previous_frame_edges[i];

                // Check if we have candidates for this previous edge
                auto edge_match_it = Edge_match.find(prev_edge);
                if (edge_match_it == Edge_match.end())
                {
                    continue;
                }

                cv::Mat prev_descriptor = previous_frame_descriptors_cache.at(i);
                if (prev_descriptor.empty())
                {
                    continue; // Skip if descriptor is empty
                }

                float threshold = 600.0f;
                std::vector<Edge> filtered_candidates;

                // Apply SIFT filtering to spatial candidates
                for (const Edge &candidate_edge : edge_match_it->second)
                {
                    // Find the index of this candidate edge
                    int idx = candidate_edge.index; // Assuming Edge has an index field

                    // Check if we have a descriptor for the current frame candidate
                    if (current_descriptors_cache.find(idx) == current_descriptors_cache.end())
                    {
                        continue;
                    }

                    cv::Mat curr_descriptor = current_descriptors_cache.at(idx);
                    if (curr_descriptor.empty())
                    {
                        continue;
                    }

                    // Calculate L2 distance between descriptors
                    float distance = cv::norm(prev_descriptor, curr_descriptor, cv::NORM_L2);

                    if (distance < threshold)
                    {
                        filtered_candidates.push_back(candidate_edge);

                        cv::DMatch match;
                        match.queryIdx = i;   // Previous frame edge index
                        match.trainIdx = idx; // Current frame edge index
                        match.distance = distance;
                        thread_matches[thread_id].push_back(match);
                    }
                }

                // Add filtered candidates to the new Edge_match
                if (!filtered_candidates.empty())
                {
#pragma omp critical(filtered_edge_match)
                    {
                        Filtered_Edge_match[prev_edge] = filtered_candidates;
                    }
                }
            }

            // Replace Edge_match with filtered version
            Edge_match = Filtered_Edge_match;

            // Evaluate Edge_match after SIFT filtering
            std::cout << "Evaluating Edge_match after SIFT filtering..." << std::endl;
            EvaluateEdgeMatchPerformance(Edge_match, left_edges_GT_Info, frame_idx, "sift_filtered", 5.0);

            // Combine matches from all threads
            for (const auto &thread_match_vec : thread_matches)
            {
                SIFT_matches.insert(SIFT_matches.end(), thread_match_vec.begin(), thread_match_vec.end());
            }

            if (!SIFT_matches.empty())
            {
                // std::vector<cv::DMatch> selected_matches;
                std::cout << "Found " << SIFT_matches.size() << " raw matches" << std::endl;
            }
            std::vector<std::pair<Edge, std::vector<Edge> *>> edge_match_vector;
            edge_match_vector.reserve(Edge_match.size());
            for (auto &match : Edge_match)
            {
                edge_match_vector.push_back({match.first, &match.second});
            }

#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < edge_match_vector.size(); i++)
            {
                const Edge &curr_edge = edge_match_vector[i].first;
                std::vector<Edge> *candidates_ptr = edge_match_vector[i].second;
                std::vector<Edge> &candidates = *candidates_ptr;

                std::vector<Edge> filtered_candidates;

                for (const Edge &candidate : candidates)
                {
                    double score = edge_patch_similarity(curr_edge, candidate, previous_frame.left_image, current_frame.left_image);
                    if (!std::isnan(score) && score >= NCC_THRESH_WEAK_BOTH_SIDES)
                    {
                        filtered_candidates.push_back(candidate);
                    }
                }

                // No need for critical section since each thread works on its own portion of the map
                candidates = std::move(filtered_candidates);
            }

            std::cout << "Evaluating Edge_match after patch similarity filtering..." << std::endl;
            DebugNCCScoresWithGT(left_edges_GT_Info, frame_idx, previous_frame, current_frame);
            // EvaluateEdgeMatchPerformance(Edge_match, left_edges_GT_Info, frame_idx, "ncc_filtered", 5.0);

            // Update caches for next iteration
            previous_frame_descriptors_cache.clear();
            previous_frame_descriptors_cache = std::move(current_descriptors_cache);

            // Update previous edge locations for next iteration
            previous_edge_loc = current_edge_loc;
            previous_frame_edges = dataset.left_edges; // Store current edges as previous for next iteration
        }
        else
        {
            // Initialize for first frame
            for (int i = 0; i < dataset.left_edges.size(); ++i)
            {
                const Edge &edge = dataset.left_edges[i];
                previous_edge_loc.push_back(cv::KeyPoint(edge.location, 1, 180 / M_PI * edge.orientation));
                spatial_grid.addEdge(i, edge.location);

                // Compute and cache descriptor for first frame
                std::vector<cv::KeyPoint> edge_kp = {cv::KeyPoint(edge.location, 1, 180 / M_PI * edge.orientation)};
                cv::Mat edge_desc;
                sift->compute(current_frame.left_image, edge_kp, edge_desc);

                if (!edge_desc.empty())
                {
                    previous_frame_descriptors_cache[i] = edge_desc.clone();
                }
            }
            previous_frame_edges = dataset.left_edges; // Store first frame edges
        }

        // StereoMatchResult match_result = get_Stereo_Edge_Pairs(
        //     left_undistorted,
        //     right_undistorted,
        //     dataset, frame_idx);

        frame_idx++;
        if (frame_idx >= 3)
        {
            break;
        }

        previous_frame = current_frame;
        // previous_edge_loc is now updated in the frame_idx > 0 block above
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

void EBVO::CalculateGTRightEdge(const std::vector<Edge> &edges, const cv::Mat &disparity_map, const cv::Mat &left_image, const cv::Mat &right_image)
{
    dataset.forward_gt_data.clear();

    static size_t total_rows_written = 0;
    static int file_index = 1;
    static std::ofstream csv_file;
    static const size_t max_rows_per_file = 1'000'000;

    if (!csv_file.is_open())
    {
        std::string filename = "valid_disparities_part_" + std::to_string(file_index) + ".csv";
        csv_file.open(filename, std::ios::out);
    }

    for (const Edge &e : edges)
    {
        double disparity = Bilinear_Interpolation(disparity_map, e.location);

        if (std::isnan(disparity) || std::isinf(disparity) || disparity < 0)
        {
            continue;
        }

        cv::Point2d right_edge(e.location.x - disparity, e.location.y);
        dataset.forward_gt_data.emplace_back(e.location, right_edge, e.orientation);

        if (total_rows_written >= max_rows_per_file)
        {
            csv_file.close();
            ++file_index;
            total_rows_written = 0;
            std::string next_filename = "valid_disparities_part_" + std::to_string(file_index) + ".csv";
            csv_file.open(next_filename, std::ios::out);
        }

        csv_file << disparity << "\n";
        ++total_rows_written;
    }

    csv_file.flush();
}

void EBVO::ReadEdgesFromBinary(const std::string &filepath,
                               std::vector<Edge> &edges)
{
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open())
    {
        std::cerr << "ERROR: Could not open binary file for reading: " << filepath << std::endl;
        return;
    }

    size_t size = 0;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size));

    edges.resize(size);
    ifs.read(reinterpret_cast<char *>(edges.data()), sizeof(Edge) * size);
}

void EBVO::WriteEdgesToBinary(const std::string &filepath,
                              const std::vector<Edge> &edges)
{
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs.is_open())
    {
        std::cerr << "ERROR: Could not open binary file for writing: " << filepath << std::endl;
        return;
    }

    size_t size = edges.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
    ofs.write(reinterpret_cast<const char *>(edges.data()), sizeof(Edge) * size);
}

void EBVO::WriteEdgeMatchResult(StereoMatchResult &match_result,
                                std::vector<double> &max_disparity_values,
                                std::vector<double> &per_image_avg_before_epi,
                                std::vector<double> &per_image_avg_after_epi,
                                std::vector<double> &per_image_avg_before_disp,
                                std::vector<double> &per_image_avg_after_disp,
                                std::vector<double> &per_image_avg_before_shift,
                                std::vector<double> &per_image_avg_after_shift,
                                std::vector<double> &per_image_avg_before_clust,
                                std::vector<double> &per_image_avg_after_clust,
                                std::vector<double> &per_image_avg_before_patch,
                                std::vector<double> &per_image_avg_after_patch,
                                std::vector<double> &per_image_avg_before_ncc,
                                std::vector<double> &per_image_avg_after_ncc,
                                std::vector<double> &per_image_avg_before_lowe,
                                std::vector<double> &per_image_avg_after_lowe,
                                std::vector<double> &per_image_avg_before_bct,
                                std::vector<double> &per_image_avg_after_bct,
                                std::vector<RecallMetrics> &all_forward_recall_metrics,
                                std::vector<BidirectionalMetrics> &all_bct_metrics)
{
    const RecallMetrics &forward_metrics = match_result.forward_match.recall_metrics;
    all_forward_recall_metrics.push_back(forward_metrics);

    const BidirectionalMetrics &bidirectional_metrics = match_result.bidirectional_metrics;
    all_bct_metrics.push_back(bidirectional_metrics);

    double avg_before_epi = ComputeAverage(forward_metrics.epi_input_counts);
    double avg_after_epi = ComputeAverage(forward_metrics.epi_output_counts);

    double avg_before_disp = ComputeAverage(forward_metrics.disp_input_counts);
    double avg_after_disp = ComputeAverage(forward_metrics.disp_output_counts);

    double avg_before_shift = ComputeAverage(forward_metrics.shift_input_counts);
    double avg_after_shift = ComputeAverage(forward_metrics.shift_output_counts);

    double avg_before_clust = ComputeAverage(forward_metrics.clust_input_counts);
    double avg_after_clust = ComputeAverage(forward_metrics.clust_output_counts);

    double avg_before_patch = ComputeAverage(forward_metrics.patch_input_counts);
    double avg_after_patch = ComputeAverage(forward_metrics.patch_output_counts);

    double avg_before_ncc = ComputeAverage(forward_metrics.ncc_input_counts);
    double avg_after_ncc = ComputeAverage(forward_metrics.ncc_output_counts);

    double avg_before_lowe = ComputeAverage(forward_metrics.lowe_input_counts);
    double avg_after_lowe = ComputeAverage(forward_metrics.lowe_output_counts);

    // Fix: Use bidirectional_metrics instead of forward_metrics for BCT counts
    double avg_before_bct = bidirectional_metrics.matches_before_bct;
    double avg_after_bct = bidirectional_metrics.matches_after_bct;

    per_image_avg_before_epi.push_back(avg_before_epi);
    per_image_avg_after_epi.push_back(avg_after_epi);

    per_image_avg_before_disp.push_back(avg_before_disp);
    per_image_avg_after_disp.push_back(avg_after_disp);

    per_image_avg_before_shift.push_back(avg_before_shift);
    per_image_avg_after_shift.push_back(avg_after_shift);

    per_image_avg_before_clust.push_back(avg_before_clust);
    per_image_avg_after_clust.push_back(avg_after_clust);

    per_image_avg_before_patch.push_back(avg_before_patch);
    per_image_avg_after_patch.push_back(avg_after_patch);

    per_image_avg_before_ncc.push_back(avg_before_ncc);
    per_image_avg_after_ncc.push_back(avg_after_ncc);

    per_image_avg_before_lowe.push_back(avg_before_lowe);
    per_image_avg_after_lowe.push_back(avg_after_lowe);

    per_image_avg_before_bct.push_back(match_result.bidirectional_metrics.matches_before_bct);
    per_image_avg_after_bct.push_back(match_result.bidirectional_metrics.matches_after_bct);

    double total_epi_recall = 0.0;
    double total_disp_recall = 0.0;
    double total_shift_recall = 0.0;
    double total_cluster_recall = 0.0;
    double total_ncc_recall = 0.0;
    double total_lowe_recall = 0.0;
    double total_bct_recall = 0.0;

    double total_epi_precision = 0.0;
    double total_disp_precision = 0.0;
    double total_shift_precision = 0.0;
    double total_cluster_precision = 0.0;
    double total_ncc_precision = 0.0;
    double total_lowe_precision = 0.0;
    double total_bct_precision = 0.0;

    double total_epi_time = 0.0;
    double total_disp_time = 0.0;
    double total_shift_time = 0.0;
    double total_clust_time = 0.0;
    double total_patch_time = 0.0;
    double total_ncc_time = 0.0;
    double total_lowe_time = 0.0;
    double total_image_time = 0.0;
    double total_bct_time = 0.0;

    for (const RecallMetrics &m : all_forward_recall_metrics)
    {
        total_epi_recall += m.epi_distance_recall;
        total_disp_recall += m.max_disparity_recall;
        total_shift_recall += m.epi_shift_recall;
        total_cluster_recall += m.epi_cluster_recall;
        total_ncc_recall += m.ncc_recall;
        total_lowe_recall += m.lowe_recall;

        total_epi_precision += m.per_image_epi_precision;
        total_disp_precision += m.per_image_disp_precision;
        total_shift_precision += m.per_image_shift_precision;
        total_cluster_precision += m.per_image_clust_precision;
        total_ncc_precision += m.per_image_ncc_precision;
        total_lowe_precision += m.per_image_lowe_precision;

        total_epi_time += m.per_image_epi_time;
        total_disp_time += m.per_image_disp_time;
        total_shift_time += m.per_image_shift_time;
        total_clust_time += m.per_image_clust_time;
        total_patch_time += m.per_image_patch_time;
        total_ncc_time += m.per_image_ncc_time;
        total_lowe_time += m.per_image_lowe_time;
        total_image_time += m.per_image_total_time;
    }

    for (const BidirectionalMetrics &m : all_bct_metrics)
    {
        total_bct_recall += m.per_image_bct_recall;
        total_bct_precision += m.per_image_bct_precision;
        total_bct_time += m.per_image_bct_time;
    }

    int total_images = static_cast<int>(all_forward_recall_metrics.size());

    double avg_epi_recall = (total_images > 0) ? total_epi_recall / total_images : 0.0;
    double avg_disp_recall = (total_images > 0) ? total_disp_recall / total_images : 0.0;
    double avg_shift_recall = (total_images > 0) ? total_shift_recall / total_images : 0.0;
    double avg_cluster_recall = (total_images > 0) ? total_cluster_recall / total_images : 0.0;
    double avg_ncc_recall = (total_images > 0) ? total_ncc_recall / total_images : 0.0;
    double avg_lowe_recall = (total_images > 0) ? total_lowe_recall / total_images : 0.0;
    double avg_bct_recall = (total_images > 0) ? total_bct_recall / total_images : 0.0;

    double avg_epi_precision = (total_images > 0) ? total_epi_precision / total_images : 0.0;
    double avg_disp_precision = (total_images > 0) ? total_disp_precision / total_images : 0.0;
    double avg_shift_precision = (total_images > 0) ? total_shift_precision / total_images : 0.0;
    double avg_cluster_precision = (total_images > 0) ? total_cluster_precision / total_images : 0.0;
    double avg_ncc_precision = (total_images > 0) ? total_ncc_precision / total_images : 0.0;
    double avg_lowe_precision = (total_images > 0) ? total_lowe_precision / total_images : 0.0;
    double avg_bct_precision = (total_images > 0) ? total_bct_precision / total_images : 0.0;

    double avg_epi_time = (total_images > 0) ? total_epi_time / total_images : 0.0;
    double avg_disp_time = (total_images > 0) ? total_disp_time / total_images : 0.0;
    double avg_shift_time = (total_images > 0) ? total_shift_time / total_images : 0.0;
    double avg_clust_time = (total_images > 0) ? total_clust_time / total_images : 0.0;
    double avg_patch_time = (total_images > 0) ? total_patch_time / total_images : 0.0;
    double avg_ncc_time = (total_images > 0) ? total_ncc_time / total_images : 0.0;
    double avg_lowe_time = (total_images > 0) ? total_lowe_time / total_images : 0.0;
    double avg_total_time = (total_images > 0) ? total_image_time / total_images : 0.0;
    double avg_bct_time = (total_images > 0) ? total_bct_time / total_images : 0.0;

    std::string edge_stat_dir = dataset.get_output_path() + "/edge_stats";
    std::filesystem::create_directories(edge_stat_dir);

    std::ofstream recall_csv(edge_stat_dir + "/recall_metrics.csv");
    recall_csv << "ImageIndex,EpiDistanceRecall,MaxDisparityRecall,EpiShiftRecall,EpiClusterRecall,NCCRecall,LoweRecall,BidirectionalRecall\n";

    std::ofstream time_elapsed_csv(edge_stat_dir + "/time_elapsed_metrics.csv");
    time_elapsed_csv << "ImageIndex,EpiDistanceTime,MaxDisparityTime,EpiShiftTime,EpiClusterTime,PatchTime,NCCTime,LoweTime,TotalLoopTime,BidirectionalTime\n";

    for (size_t i = 0; i < all_forward_recall_metrics.size(); i++)
    {
        const auto &m = all_forward_recall_metrics[i];
        const auto &bct = all_bct_metrics[i];
        recall_csv << i << ","
                   << std::fixed << std::setprecision(4) << m.epi_distance_recall * 100 << ","
                   << std::fixed << std::setprecision(4) << m.max_disparity_recall * 100 << ","
                   << std::fixed << std::setprecision(4) << m.epi_shift_recall * 100 << ","
                   << std::fixed << std::setprecision(4) << m.epi_cluster_recall * 100 << ","
                   << std::fixed << std::setprecision(4) << m.ncc_recall * 100 << ","
                   << std::fixed << std::setprecision(4) << m.lowe_recall * 100 << ","
                   << std::fixed << std::setprecision(4) << bct.per_image_bct_recall * 100 << "\n";
    }

    recall_csv << "Average,"
               << std::fixed << std::setprecision(4) << avg_epi_recall * 100 << ","
               << std::fixed << std::setprecision(4) << avg_disp_recall * 100 << ","
               << std::fixed << std::setprecision(4) << avg_shift_recall * 100 << ","
               << std::fixed << std::setprecision(4) << avg_cluster_recall * 100 << ","
               << std::fixed << std::setprecision(4) << avg_ncc_recall * 100 << ","
               << std::fixed << std::setprecision(4) << avg_lowe_recall * 100 << ","
               << std::fixed << std::setprecision(4) << avg_bct_recall * 100 << "\n";

    for (size_t i = 0; i < all_forward_recall_metrics.size(); i++)
    {
        const auto &m = all_forward_recall_metrics[i];
        const auto &bct = all_bct_metrics[i];
        time_elapsed_csv << i << ","
                         << std::fixed << std::setprecision(4) << m.per_image_epi_time << ","
                         << std::fixed << std::setprecision(4) << m.per_image_disp_time << ","
                         << std::fixed << std::setprecision(4) << m.per_image_shift_time << ","
                         << std::fixed << std::setprecision(4) << m.per_image_clust_time << ","
                         << std::fixed << std::setprecision(4) << m.per_image_patch_time << ","
                         << std::fixed << std::setprecision(4) << m.per_image_ncc_time << ","
                         << std::fixed << std::setprecision(4) << m.per_image_lowe_time << ","
                         << std::fixed << std::setprecision(4) << m.per_image_total_time << ","
                         << std::fixed << std::setprecision(4) << bct.per_image_bct_time << "\n";
    }

    time_elapsed_csv << "Average,"
                     << std::fixed << std::setprecision(4) << avg_epi_time << ","
                     << std::fixed << std::setprecision(4) << avg_disp_time << ","
                     << std::fixed << std::setprecision(4) << avg_shift_time << ","
                     << std::fixed << std::setprecision(4) << avg_clust_time << ","
                     << std::fixed << std::setprecision(4) << avg_patch_time << ","
                     << std::fixed << std::setprecision(4) << avg_ncc_time << ","
                     << std::fixed << std::setprecision(4) << avg_lowe_time << ","
                     << std::fixed << std::setprecision(4) << avg_total_time << ","
                     << std::fixed << std::setprecision(4) << avg_bct_time << "\n";

    std::ofstream count_csv(edge_stat_dir + "/count_metrics.csv");
    count_csv
        << "before_epi_distance,after_epi_distance,average_before_epi_distance,average_after_epi_distance,"
        << "before_max_disp,after_max_disp,average_before_max_disp,average_after_max_disp,"
        << "before_epi_shift,after_epi_shift,average_before_epi_shift,average_after_epi_shift,"
        << "before_epi_cluster,after_epi_cluster,average_before_epi_cluster,average_after_epi_cluster,"
        << "before_patch, after_patch, average_before_patch, average_after_patch,"
        << "before_ncc,after_ncc,average_before_ncc,average_after_ncc,"
        << "before_lowe,after_lowe,average_before_lowe,after_after_lowe,"
        << "before_bct (PER IMAGE),after_bct (PER IMAGE),average_before_bct (PER IMAGE),after_after_bct (PER IMAGE)\n";

    double total_avg_before_epi = 0.0;
    double total_avg_after_epi = 0.0;

    double total_avg_before_disp = 0.0;
    double total_avg_after_disp = 0.0;

    double total_avg_before_shift = 0.0;
    double total_avg_after_shift = 0.0;

    double total_avg_before_clust = 0.0;
    double total_avg_after_clust = 0.0;

    double total_avg_before_patch = 0.0;
    double total_avg_after_patch = 0.0;

    double total_avg_before_ncc = 0.0;
    double total_avg_after_ncc = 0.0;

    double total_avg_before_lowe = 0.0;
    double total_avg_after_lowe = 0.0;

    double total_avg_before_bct = 0.0;
    double total_avg_after_bct = 0.0;

    size_t num_rows = per_image_avg_before_epi.size();

    for (size_t i = 0; i < num_rows; ++i)
    {
        total_avg_before_epi += per_image_avg_before_epi[i];
        total_avg_after_epi += per_image_avg_after_epi[i];

        total_avg_before_disp += per_image_avg_before_disp[i];
        total_avg_after_disp += per_image_avg_after_disp[i];

        total_avg_before_shift += per_image_avg_before_shift[i];
        total_avg_after_shift += per_image_avg_after_shift[i];

        total_avg_before_clust += per_image_avg_before_clust[i];
        total_avg_after_clust += per_image_avg_after_clust[i];

        total_avg_before_patch += per_image_avg_before_patch[i];
        total_avg_after_patch += per_image_avg_after_patch[i];

        total_avg_before_ncc += per_image_avg_before_ncc[i];
        total_avg_after_ncc += per_image_avg_after_ncc[i];

        total_avg_before_lowe += per_image_avg_before_lowe[i];
        total_avg_after_lowe += per_image_avg_after_lowe[i];

        total_avg_before_bct += per_image_avg_before_bct[i];
        total_avg_after_bct += per_image_avg_after_bct[i];

        count_csv
            << static_cast<int>(std::ceil(per_image_avg_before_epi[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_epi[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_disp[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_disp[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_shift[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_shift[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_clust[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_clust[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_patch[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_patch[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_ncc[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_ncc[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_lowe[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_lowe[i])) << ","
            << ","
            << ","
            << static_cast<int>(std::ceil(per_image_avg_before_bct[i])) << ","
            << static_cast<int>(std::ceil(per_image_avg_after_bct[i])) << ","
            << "\n";
    }

    int avg_of_avgs_before_epi = 0;
    int avg_of_avgs_after_epi = 0;

    int avg_of_avgs_before_disp = 0;
    int avg_of_avgs_after_disp = 0;

    int avg_of_avgs_before_shift = 0;
    int avg_of_avgs_after_shift = 0;

    int avg_of_avgs_before_clust = 0;
    int avg_of_avgs_after_clust = 0;

    int avg_of_avgs_before_patch = 0;
    int avg_of_avgs_after_patch = 0;

    int avg_of_avgs_before_ncc = 0;
    int avg_of_avgs_after_ncc = 0;

    int avg_of_avgs_before_lowe = 0;
    int avg_of_avgs_after_lowe = 0;

    int avg_of_avgs_before_bct = 0;
    int avg_of_avgs_after_bct = 0;

    if (num_rows > 0)
    {
        avg_of_avgs_before_epi = std::ceil(total_avg_before_epi / num_rows);
        avg_of_avgs_after_epi = std::ceil(total_avg_after_epi / num_rows);

        avg_of_avgs_before_disp = std::ceil(total_avg_before_disp / num_rows);
        avg_of_avgs_after_disp = std::ceil(total_avg_after_disp / num_rows);

        avg_of_avgs_before_shift = std::ceil(total_avg_before_shift / num_rows);
        avg_of_avgs_after_shift = std::ceil(total_avg_after_shift / num_rows);

        avg_of_avgs_before_clust = std::ceil(total_avg_before_clust / num_rows);
        avg_of_avgs_after_clust = std::ceil(total_avg_after_clust / num_rows);

        avg_of_avgs_before_patch = std::ceil(total_avg_before_patch / num_rows);
        avg_of_avgs_after_patch = std::ceil(total_avg_after_patch / num_rows);

        avg_of_avgs_before_ncc = std::ceil(total_avg_before_ncc / num_rows);
        avg_of_avgs_after_ncc = std::ceil(total_avg_after_ncc / num_rows);

        avg_of_avgs_before_lowe = std::ceil(total_avg_before_lowe / num_rows);
        avg_of_avgs_after_lowe = std::ceil(total_avg_after_lowe / num_rows);

        avg_of_avgs_before_bct = std::ceil(total_avg_before_bct / num_rows);
        avg_of_avgs_after_bct = std::ceil(total_avg_after_bct / num_rows);
    }

    count_csv
        << ","
        << ","
        << avg_of_avgs_before_epi << ","
        << avg_of_avgs_after_epi << ","
        << ","
        << ","
        << avg_of_avgs_before_disp << ","
        << avg_of_avgs_after_disp << ","
        << ","
        << ","
        << avg_of_avgs_before_shift << ","
        << avg_of_avgs_after_shift << ","
        << ","
        << ","
        << avg_of_avgs_before_clust << ","
        << avg_of_avgs_after_clust << ","
        << ","
        << ","
        << avg_of_avgs_before_patch << ","
        << avg_of_avgs_after_patch << ","
        << ","
        << ","
        << avg_of_avgs_before_ncc << ","
        << avg_of_avgs_after_ncc << ","
        << ","
        << ","
        << avg_of_avgs_before_lowe << ","
        << avg_of_avgs_after_lowe << ","
        << ","
        << ","
        << avg_of_avgs_before_bct << ","
        << avg_of_avgs_after_bct << "\n";

    std::ofstream precision_csv(edge_stat_dir + "/precision_metrics.csv");
    precision_csv << "ImageIndex,EpiDistancePrecision,MaxDisparityPrecision,EpiShiftPrecision,EpiClusterPrecision,NCCPrecision,LowePrecision,BidirectionalPrecision\n";

    for (size_t i = 0; i < all_forward_recall_metrics.size(); i++)
    {
        const auto &m = all_forward_recall_metrics[i];
        const auto &bct = all_bct_metrics[i];
        precision_csv << i << ","
                      << std::fixed << std::setprecision(4) << m.per_image_epi_precision * 100 << ","
                      << std::fixed << std::setprecision(4) << m.per_image_disp_precision * 100 << ","
                      << std::fixed << std::setprecision(4) << m.per_image_shift_precision * 100 << ","
                      << std::fixed << std::setprecision(4) << m.per_image_clust_precision * 100 << ","
                      << std::fixed << std::setprecision(4) << m.per_image_ncc_precision * 100 << ","
                      << std::fixed << std::setprecision(4) << m.per_image_lowe_precision * 100 << ","
                      << std::fixed << std::setprecision(4) << bct.per_image_bct_precision * 100 << "\n";
    }

    precision_csv << "Average,"
                  << std::fixed << std::setprecision(4) << avg_epi_precision * 100 << ","
                  << std::fixed << std::setprecision(4) << avg_disp_precision * 100 << ","
                  << std::fixed << std::setprecision(4) << avg_shift_precision * 100 << ","
                  << std::fixed << std::setprecision(4) << avg_cluster_precision * 100 << ","
                  << std::fixed << std::setprecision(4) << avg_ncc_precision * 100 << ","
                  << std::fixed << std::setprecision(4) << avg_lowe_precision * 100 << ","
                  << std::fixed << std::setprecision(4) << avg_bct_precision * 100 << "\n";
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
                      const cv::Mat &left_ref_map, const cv::Mat &left_calib_inv,
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
                                left_ref_map, left_calib_inv, left_calib,
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