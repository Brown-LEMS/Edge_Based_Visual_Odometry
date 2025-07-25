#include <filesystem>
#include <unordered_set>
#include <numeric>
#include "EBVO.h"
#include "Dataset.h"
#include "Matches.h"
#include <opencv2/core/eigen.hpp>

std::vector<cv::DMatch> FilterMatchesByProximity(
    const std::vector<cv::DMatch> &matches,
    const std::vector<cv::KeyPoint> &previous_keypoints,
    const std::vector<cv::KeyPoint> &current_keypoints,
    double max_distance_threshold = 30.0) // pixels
{
    std::vector<cv::DMatch> filtered_matches;

    for (const cv::DMatch &match : matches)
    {
        // Get the matched keypoints
        cv::Point2f prev_pt = previous_keypoints[match.queryIdx].pt;
        cv::Point2f curr_pt = current_keypoints[match.trainIdx].pt;

        // Calculate spatial distance between matched points
        double spatial_distance = cv::norm(prev_pt - curr_pt);

        // Only keep matches where points are spatially close
        if (spatial_distance <= max_distance_threshold)
        {
            filtered_matches.push_back(match);
        }
    }

    return filtered_matches;
}

EBVO::EBVO(YAML::Node config_map, bool use_GCC_filter) : dataset(config_map, use_GCC_filter) {}

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

        for (const auto &edge : dataset.left_edges)
        {
            if (edge.location.x >= 0 && edge.location.x < left_edge_map.cols && edge.location.y >= 0 && edge.location.y < left_edge_map.rows)
            {
                left_edge_map.at<uchar>(cv::Point(edge.location.x, edge.location.y)) = 255;
            }
        }

        for (const auto &edge : dataset.right_edges)
        {
            if (edge.location.x >= 0 && edge.location.x < right_edge_map.cols && edge.location.y >= 0 && edge.location.y < right_edge_map.rows)
            {
                right_edge_map.at<uchar>(cv::Point(edge.location.x, edge.location.y)) = 255;
            }
        }

        CalculateGTRightEdge(dataset.left_edges, left_ref_map, left_edge_map, right_edge_map);
        std::vector<Edge> gt_edges;
        std::vector<std::pair<Edge, Edge>> left_edges_GT_pair;
        if (frame_idx > 0 && dataset.has_gt())
        {
            GetGTEdges(frame_idx, previous_frame, current_frame, left_ref_map, left_calib_inv, left_calib, gt_edges, left_edges_GT_pair);
        }

        if (frame_idx > 0)
        { // at any frame after the first, we compute SIFT descriptors on the current frame
            current_edge_loc.clear();
            for (const Edge &edge : dataset.left_edges)
            {
                current_edge_loc.push_back(cv::KeyPoint(edge.location, 1.0f)); // Convert to KeyPoint
            }
            sift->compute(current_frame.left_image, current_edge_loc, descriptors_t1);

            if (!descriptors_t0.empty() && !descriptors_t1.empty())
            {
                // Create a matcher
                cv::BFMatcher matcher(cv::NORM_L2); // For SIFT, use L2 norm

                // Perform matching
                std::vector<cv::DMatch> raw_matches;
                matcher.match(descriptors_t0, descriptors_t1, raw_matches);

                std::cout << "Found " << raw_matches.size() << " raw matches" << std::endl;
                std::vector<cv::DMatch> proximity_filtered_matches = FilterMatchesByProximity(
                    raw_matches, previous_edge_loc, current_edge_loc, 25.0); // 25 pixel threshold

                std::cout << "After proximity filtering: " << proximity_filtered_matches.size() << " matches" << std::endl;

                EvaluateSIFTMatches(proximity_filtered_matches, previous_edge_loc, current_edge_loc, left_edges_GT_pair, frame_idx, 5.0);
                // Randomly select up to 100 matches for visualization
                std::vector<cv::DMatch> selected_matches;
                if (proximity_filtered_matches.size() <= 100)
                {
                    selected_matches = proximity_filtered_matches; // Use all filtered matches if we have 100 or fewer
                }
                else
                {
                    std::vector<int> indices(proximity_filtered_matches.size());
                    std::iota(indices.begin(), indices.end(), 0);

                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::shuffle(indices.begin(), indices.end(), gen);

                    selected_matches.reserve(100);
                    for (int i = 0; i < 100; ++i)
                    {
                        selected_matches.push_back(proximity_filtered_matches[indices[i]]);
                    }
                }

                std::cout << "Visualizing " << selected_matches.size() << " randomly selected matches" << std::endl;

                cv::Mat img_matches;
                cv::drawMatches(previous_frame.left_image, previous_edge_loc, // Previous frame & keypoints
                                current_frame.left_image, current_edge_loc,   // Current frame & keypoints
                                selected_matches,                             // Selected matches
                                img_matches,                                  // Output image
                                cv::Scalar::all(-1),                          // Match color (random)
                                cv::Scalar::all(-1),                          // Single point color
                                std::vector<char>(),                          // Matches mask
                                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                // Save the visualization
                std::string output_dir = dataset.get_output_path();
                cv::imwrite(output_dir + "/sift_matches_frame_" + std::to_string(frame_idx) + ".png", img_matches);
            }

            // Update previous keypoints and descriptors for next iteration
            previous_edge_loc = current_edge_loc;    // Copy current keypoints to previous
            descriptors_t0 = descriptors_t1.clone(); // copy the current descriptors to previous for next iteration
        }
        else
        {
            for (const Edge &edge : dataset.left_edges)
            {
                previous_edge_loc.push_back(cv::KeyPoint(edge.location, 1.0f)); // Convert to KeyPoint
            }
            sift->compute(current_frame.left_image, previous_edge_loc, descriptors_t0);
            // now previous_edge_loc stores the frame 0's edge locations, and descriptors_t0 has sift descriptors
        }

        // StereoMatchResult match_result = DisplayMatches(
        //     left_undistorted,
        //     right_undistorted,
        //     dataset, frame_idx);

#if 0
        std::vector<cv::Point3d> points_opencv = Calculate3DPoints(match_result.confirmed_matches);
        std::vector<cv::Point3d> points_linear = LinearTriangulatePoints(match_result.confirmed_matches);
        std::vector<Eigen::Vector3d> orientations_3d = Calculate3DOrientations(match_result.confirmed_matches);

        std::vector<Edge> oriented_points;

        for (size_t i = 0; i < points_opencv.size(); ++i) {
            Edge op;
            op.location = points_opencv[i];
            op.orientation = orientations_3d[i];
            oriented_points.push_back(op);
        }


        if (points_opencv.size() != points_linear.size()) {
            std::cerr << "Mismatch in number of 3D points: OpenCV=" << points_opencv.size()
                    << ", Linear=" << points_linear.size() << "\n";
        } else {
            std::cout << "Comparing " << points_opencv.size() << " 3D points...\n";

            double total_error = 0.0;
            for (size_t i = 0; i < points_opencv.size(); ++i) {
                const auto& pt1 = points_opencv[i];
                const auto& pt2 = points_linear[i];

                double error = cv::norm(pt1 - pt2);
                total_error += error;

                std::cout << "Point " << i << ": OpenCV = [" << pt1 << "], "
                        << "Linear = [" << pt2 << "], "
                        << "Error = " << error << "\n";
            }

            std::cout << "Average triangulation error: "
                    << (total_error / points_opencv.size()) << " units.\n";
        }
#endif

#if DISPLAY_STERO_EDGE_MATCHES
        cv::Mat left_visualization, right_visualization;
        cv::cvtColor(left_edge_map, left_visualization, cv::COLOR_GRAY2BGR);
        cv::cvtColor(right_edge_map, right_visualization, cv::COLOR_GRAY2BGR);

        cv::Mat merged_visualization;
        cv::hconcat(left_visualization, right_visualization, merged_visualization);

        int total_matches = static_cast<int>(match_result.confirmed_matches.size());
        int index = 0;

        for (const auto &[left_edge, right_edge] : match_result.confirmed_matches)
        {
            cv::Scalar color = PickUniqueColor(index, total_matches);

            cv::Point2d left_position = left_edge.position;
            cv::Point2d right_position = right_edge.position;

            cv::circle(merged_visualization, left_position, 4, color, cv::FILLED);
            cv::Point2d right_shifted(right_position.x + left_visualization.cols, right_position.y);
            cv::circle(merged_visualization, right_shifted, 4, color, cv::FILLED);
            cv::line(merged_visualization, left_position, right_shifted, color, 1);

            ++index;
        }

        std::string save_path = output_path + "/edge_matches_image" + std::to_string(i) + ".png";
        cv::imwrite(save_path, merged_visualization);
#endif
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
                      const cv::Mat &left_ref_map, const cv::Mat &left_calib_inv,
                      const cv::Mat &left_calib, std::vector<Edge> &gt_edges,
                      std::vector<std::pair<Edge, Edge>> &left_edges_GT_pair)
{
    std::string output_dir = dataset.get_output_path();
    std::string csv_filename = output_dir + "/gt_correspondences_frame_" + std::to_string(frame_idx) + ".csv";
    std::ofstream gt_csv(csv_filename);

    // Write CSV header
    gt_csv << "edge_index,prev_x,prev_y,curr_x,curr_y,orientation\n";

    for (int i = 0; i < dataset.left_edges.size(); ++i)
    {

        Edge GTEdge = GetGTEdge(true, previous_frame, current_frame,
                                left_ref_map, left_calib_inv, left_calib,
                                dataset.left_edges[i]);
        if (GTEdge.b_isEmpty)
            continue;
        else
        {
            left_edges_GT_pair.push_back(std::make_pair(dataset.left_edges[i], GTEdge));
            gt_edges.push_back(GTEdge);

            const Edge &prev_edge = dataset.left_edges[i];
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

    for (int i = 0; i < left_edges_GT_pair.size(); i++)
    {
        cv::Point2d left_edge_t0 = left_edges_GT_pair[i].first.location;
        cv::Point2d left_edge_t1 = left_edges_GT_pair[i].second.location;
    }
    std::cout << left_edges_GT_pair.size() << std::endl;

    // cv::Mat previous_frame_vis;
    // cv::cvtColor(previous_frame.left_image, previous_frame_vis, cv::COLOR_GRAY2BGR);

    // // Different colors for different selected points
    // std::vector<cv::Scalar> colors;
    // for (int i = 0; i < 100; ++i)
    // {
    //     int hue = (i * 179 / 100);
    //     cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
    //     cv::Mat bgr;
    //     cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    //     cv::Vec3b bgr_pixel = bgr.at<cv::Vec3b>(0, 0);
    //     colors.push_back(cv::Scalar(bgr_pixel[0], bgr_pixel[1], bgr_pixel[2]));
    // }

    // for (int i = 0; i < left_edges_GT_pair.size(); ++i)
    // {
    //     cv::Point2d pt = left_edges_GT_pair[i].first.location;
    //     cv::circle(previous_frame_vis, pt, 8, colors[i], -1);
    //     cv::putText(previous_frame_vis, std::to_string(i),
    //                 cv::Point(pt.x + 10, pt.y - 10),
    //                 cv::FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2);
    // }

    // // Draw GT edges on next frame
    // cv::Mat current_frame_vis;
    // cv::cvtColor(current_frame.left_image, current_frame_vis, cv::COLOR_GRAY2BGR);

    // for (int i = 0; i < left_edges_GT_pair.size(); ++i)
    // {
    //     cv::Point2d pt = left_edges_GT_pair[i].second.location;
    //     if (pt.x >= 0 && pt.x < current_frame_vis.cols &&
    //         pt.y >= 0 && pt.y < current_frame_vis.rows)
    //     {
    //         cv::circle(current_frame_vis, pt, 8, colors[i], -1);
    //         cv::putText(current_frame_vis, std::to_string(i),
    //                     cv::Point(pt.x + 10, pt.y - 10),
    //                     cv::FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2);
    //     }
    // }

    // // Save visualization images
    // cv::imwrite(output_dir + "/current_frame_selected_edges_" + std::to_string(frame_idx - 1) + ".png",
    //             previous_frame_vis);
    // cv::imwrite(output_dir + "/next_frame_gt_edges_" + std::to_string(frame_idx) + ".png",
    //             current_frame_vis);

    // std::cout << "Saved edge tracking visualization for frame " << frame_idx << std::endl;
}

void EBVO::EvaluateSIFTMatches(const std::vector<cv::DMatch> &matches,
                               const std::vector<cv::KeyPoint> &previous_keypoints,
                               const std::vector<cv::KeyPoint> &current_keypoints,
                               const std::vector<std::pair<Edge, Edge>> &gt_correspondences,
                               size_t frame_idx,
                               double distance_threshold)
{

    int true_positives = 0;
    int false_positives = 0;
    int total_gt_correspondences = gt_correspondences.size();

    std::string output_dir = dataset.get_output_path();
    std::string eval_filename = output_dir + "/sift_evaluation_frame_" + std::to_string(frame_idx) + ".csv";
    std::ofstream eval_csv(eval_filename);

    eval_csv << "match_idx,prev_x,prev_y,curr_x,curr_y,distance,is_correct,closest_gt_distance,gt_curr_x,gt_curr_y\n";

    for (size_t match_idx = 0; match_idx < matches.size(); ++match_idx)
    {
        const cv::DMatch &match = matches[match_idx];

        // Get the matched keypoints
        cv::Point2f prev_pt = previous_keypoints[match.queryIdx].pt;
        cv::Point2f curr_pt = current_keypoints[match.trainIdx].pt;

        // Find the closest ground truth correspondence
        double min_distance = std::numeric_limits<double>::max();
        bool is_correct_match = false;
        cv::Point2d closest_gt_prev, closest_gt_curr;

        for (const auto &gt_pair : gt_correspondences)
        {
            cv::Point2d gt_prev = gt_pair.first.location;
            cv::Point2d gt_curr = gt_pair.second.location;

            // Calculate distance between SIFT match and GT correspondence
            double curr_dist = cv::norm(cv::Point2d(curr_pt.x, curr_pt.y) - gt_curr);

            double total_dist = curr_dist;

            if (total_dist < min_distance)
            {
                min_distance = total_dist;
                closest_gt_curr = gt_curr;

                // Consider it correct if both points are within threshold
                is_correct_match = (curr_dist <= distance_threshold);
            }
        }

        if (is_correct_match)
        {
            true_positives++;
        }
        else
        {
            false_positives++;
        }

        eval_csv << match_idx << ","
                 << std::fixed << std::setprecision(2) << prev_pt.x << ","
                 << std::fixed << std::setprecision(2) << prev_pt.y << ","
                 << std::fixed << std::setprecision(2) << curr_pt.x << ","
                 << std::fixed << std::setprecision(2) << curr_pt.y << ","
                 << std::fixed << std::setprecision(2) << match.distance << ","
                 << (is_correct_match ? 1 : 0) << ","
                 << std::fixed << std::setprecision(2) << min_distance << ","
                 << std::fixed << std::setprecision(2) << closest_gt_curr.x << ","
                 << std::fixed << std::setprecision(2) << closest_gt_curr.y << "\n";
    }

    eval_csv.close();

    int total_sift_matches = matches.size();
    double precision = (total_sift_matches > 0) ? (double)true_positives / total_sift_matches : 0.0;
    double recall = (total_gt_correspondences > 0) ? (double)true_positives / total_gt_correspondences : 0.0;
    double f1_score = (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;

    std::cout << "\n=== SIFT Evaluation Frame " << frame_idx << " ===" << std::endl;
    std::cout << "Distance threshold: " << distance_threshold << " pixels" << std::endl;
    std::cout << "Total SIFT matches: " << total_sift_matches << std::endl;
    std::cout << "Total GT correspondences: " << total_gt_correspondences << std::endl;
    std::cout << "True Positives: " << true_positives << std::endl;
    std::cout << "False Positives: " << false_positives << std::endl;
    std::cout << "Precision: " << std::fixed << std::setprecision(4) << precision * 100 << "%" << std::endl;
    std::cout << "Recall: " << std::fixed << std::setprecision(4) << recall * 100 << "%" << std::endl;
    std::cout << "F1-Score: " << std::fixed << std::setprecision(4) << f1_score * 100 << "%" << std::endl;
    std::cout << "=====================================\n"
              << std::endl;

    std::string summary_filename = output_dir + "/sift_metrics_summary.csv";
    std::ofstream summary_csv;

    bool file_exists = std::filesystem::exists(summary_filename);
    summary_csv.open(summary_filename, std::ios::app);

    if (!file_exists)
    {
        summary_csv << "frame_idx,distance_threshold,total_sift_matches,total_gt_correspondences,true_positives,false_positives,precision,recall,f1_score\n";
    }

    summary_csv << frame_idx << ","
                << distance_threshold << ","
                << total_sift_matches << ","
                << total_gt_correspondences << ","
                << true_positives << ","
                << false_positives << ","
                << std::fixed << std::setprecision(4) << precision << ","
                << std::fixed << std::setprecision(4) << recall << ","
                << std::fixed << std::setprecision(4) << f1_score << "\n";

    summary_csv.close();
}