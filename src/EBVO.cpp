#include <filesystem>
#include <unordered_set>
#include <algorithm>
#include <random>
#include "EBVO.h"
#include "Dataset.h"
#include "Matches.h"

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

    cv::Mat left_dist_coeff_mat(dataset.left_dist_coeffs());
    cv::Mat right_dist_coeff_mat(dataset.right_dist_coeffs());

    auto start_time = std::chrono::high_resolution_clock::now();

    LOG_INFO("Start looping over all image pairs");

    StereoFrame current_frame, next_frame;

    // Get the first frame from the dataset
    if (!dataset.stereo_iterator->hasNext() ||
        !dataset.stereo_iterator->getNext(current_frame))
    {
        LOG_ERROR("Failed to get first frame from dataset");
        return;
    }
    std::vector<double> left_intr = dataset.left_intr();
    std::vector<double> right_intr = dataset.right_intr();

    // is it a storage issue of us not making the original one matrix?
    cv::Mat left_calib = (cv::Mat_<double>(3, 3) << left_intr[0], 0, left_intr[2], 0, left_intr[1], left_intr[3], 0, 0, 1);
    cv::Mat right_calib = (cv::Mat_<double>(3, 3) << right_intr[0], 0, right_intr[2], 0, right_intr[1], right_intr[3], 0, 0, 1);

    cv::Mat left_dist_coeff_mat(dataset.left_dist_coeffs());
    cv::Mat right_dist_coeff_mat(dataset.right_dist_coeffs());

    // Temporal tracking variables for optical flow
    std::vector<Edge> tracked_edges;
    bool first_frame = true;

    // // Track history storage for visualization - track ALL edges
    // std::vector<Edge> tracked_edges_filtered;
    // std::vector<int> original_indices;
    // std::vector<std::vector<cv::Point2d>> all_tracks; // Each track is a vector of points for ALL edges
    
    std::vector<cv::Point2f> p0, p1;                  // Previous and next image

    cv::RNG rng(12345);
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < 1000; i++)
    {
        colors.push_back(cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)));
    }
    std::string flow_viz_dir = dataset.get_output_path() + "/flow_tracks_viz";
    std::filesystem::create_directories(flow_viz_dir);

    std::vector<int> tracked_indices;
    std::vector<std::vector<cv::Point2f>> trajectories(100);
    // ============ POSE ESTIMATION VARIABLES ============
    std::vector<cv::Point3d> previous_3d_points;
    std::vector<cv::Point2d> previous_2d_points;
    cv::Mat accumulated_rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat accumulated_translation = cv::Mat::zeros(3, 1, CV_64F);

    // Store camera trajectory for output
    std::vector<cv::Mat> camera_poses;
    std::vector<cv::Point3d> trajectory_points;
    // ================================================

    size_t frame_idx = 0;
    while (dataset.stereo_iterator->hasNext() && num_pairs - frame_idx >= 0)
    {
        if (!dataset.stereo_iterator->getNext(next_frame))
        {
            break;
        }

        const cv::Mat &curr_left_img = current_frame.left_image;
        const cv::Mat &curr_right_img = current_frame.right_image;
        const cv::Mat &next_left_img = next_frame.left_image;
        const cv::Mat &next_right_img = next_frame.right_image;

        const cv::Mat &left_ref_map = (frame_idx < left_ref_disparity_maps.size()) ? left_ref_disparity_maps[frame_idx] : cv::Mat();
        // const cv::Mat& right_ref_map = right_ref_disparity_maps[i];

        dataset.ncc_one_vs_err.clear();
        dataset.ncc_two_vs_err.clear();
        dataset.ground_truth_right_edges_after_lowe.clear();

        std::cout << "Image Pair #" << frame_idx << "\n";

        cv::Mat left_undistorted, right_undistorted;
        cv::undistort(curr_left_img, left_undistorted, left_calib, left_dist_coeff_mat);
        cv::undistort(curr_right_img, right_undistorted, right_calib, right_dist_coeff_mat);

        if (dataset.get_num_imgs() == 0)
        {
            dataset.set_height(left_undistorted.rows);
            dataset.set_width(left_undistorted.cols);

            TOED = std::shared_ptr<ThirdOrderEdgeDetectionCPU>(new ThirdOrderEdgeDetectionCPU(dataset.get_height(), dataset.get_width()));
        }

        std::string edge_dir = dataset.get_output_path() + "/edges";
        std::filesystem::create_directories(edge_dir);

        std::string left_edge_path = edge_dir + "/left_edges_" + std::to_string(frame_idx);
        std::string right_edge_path = edge_dir + "/right_edges_" + std::to_string(frame_idx);

        ProcessEdges(left_undistorted, left_edge_path, TOED, dataset.left_edges);
        std::cout << "Number of edges on the left image: " << dataset.left_edges.size() << std::endl;

        ProcessEdges(right_undistorted, right_edge_path, TOED, dataset.right_edges);
        std::cout << "Number of edges on the right image: " << dataset.right_edges.size() << std::endl;

        dataset.increment_num_imgs();
        std::vector<cv::Point2f> good_new;
        std::vector<int> good_indices;
        if (!first_frame)
        {
            // Get the previous frame's undistorted image for tracking
            cv::Mat prev_left_undistorted, curr_left_undistorted_copy;
            cv::undistort(current_frame.left_image, prev_left_undistorted, left_calib, left_dist_coeff_mat);
            cv::undistort(next_frame.left_image, curr_left_undistorted_copy, left_calib, left_dist_coeff_mat);
  
            // Track points using OpenCV's pyramidal Lucas-Kanade
            // It automatically builds pyramids internally
            std::vector<uchar> status;
            std::vector<float> errors;

            if (!p0.empty())
            {
                cv::calcOpticalFlowPyrLK(
                prev_left_undistorted, curr_left_undistorted_copy, // full resolution images
                p0, p1,                                            // point correspondences
                status, errors,                                    // output status and errors
                cv::Size(15, 15),                                  // window size
                3,                                                 // max pyramid levels
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
                cv::Mat mask = cv::Mat::zeros(current_frame.left_image.size(), CV_8UC3);


                cv::Mat curr_left_bgr;
            if (curr_left_undistorted_copy.channels() == 1)
                cv::cvtColor(curr_left_undistorted_copy, curr_left_bgr, cv::COLOR_GRAY2BGR);
            else
                curr_left_bgr = curr_left_undistorted_copy.clone();
                
            good_new.clear();
            good_indices.clear();
            for (size_t i = 0; i < p0.size() ; i++)
            {
                if (status[i] == 1)
                {
                    
                    good_new.push_back(p1[i]);
                    good_indices.push_back(i);

                }
            }

             for (int i = 0; i < 100; ++i)
    {
        for (size_t j = 1; j < trajectories[i].size(); ++j)
        {
            cv::line(mask, trajectories[i][j - 1], trajectories[i][j], colors[i], 2);
        }
        if (!trajectories[i].empty())
        {
            cv::circle(curr_left_bgr, trajectories[i].back(), 4, colors[i], -1);
        }
    }

            cv::Mat output_frame;
            cv::add(curr_left_bgr, mask, output_frame);

            std::ostringstream out_path;
            out_path << flow_viz_dir << "/frame_" << std::setw(4) << std::setfill('0') << frame_idx << ".png";
            cv::imwrite(out_path.str(), output_frame);

            std::cout<<"written"<<std::endl;
            }
            p0 = good_new;
            std::cout << "[DEBUG] p0 updated to good_new with size: " << p0.size() << std::endl;


            

        }
        else
        {
            // First frame/key frame: initialize tracking with ALL detected left edges
            first_frame = false;


            // Initialize track history for all edges
            p0.reserve(dataset.left_edges.size());
            for (const auto& edge : dataset.left_edges) {
                p0.emplace_back(static_cast<float>(edge.location.x), static_cast<float>(edge.location.y));
            }
            std::vector<int> all_indices(p0.size());
            std::iota(all_indices.begin(), all_indices.end(), 0);

            std::mt19937 gen(12345); // Fixed seed for consistency
            std::shuffle(all_indices.begin(), all_indices.end(), gen);

            tracked_indices.assign(all_indices.begin(), all_indices.begin() + std::min(100, (int)p0.size()));
            trajectories.resize(100);  // make sure this is declared outside the loop

            for (int i = 0; i < 100; ++i)
            {
                int idx = tracked_indices[i];
                if (idx < (int)p0.size()) {
                    trajectories[i].push_back(p0[idx]);
                }
            }

            std::cout << "[DEBUG] Initializing p0 from first frame with " << dataset.left_edges.size() << " edges" << std::endl;

        }

        std::cout << "[DEBUG] p0 initialized with size: " << p0.size() << std::endl;

        // ================================================================

        cv::Mat left_edge_map = cv::Mat::zeros(left_undistorted.size(), CV_8UC1);
        cv::Mat right_edge_map = cv::Mat::zeros(right_undistorted.size(), CV_8UC1);

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
       
        StereoMatchResult match_result;


 
        match_result = DisplayMatches(
            left_undistorted,
            right_undistorted,
            dataset);

        // ================================================================

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

        // Move to next frame
        current_frame = next_frame;

        frame_idx++;

        // Early stop after 3 frames for visualization
        if (frame_idx >= 3)
        {
            std::cout << "Stopping after " << frame_idx << " frames for track visualization..." << std::endl;
            break;
        }
    }



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

void EBVO::ProcessEdges(const cv::Mat &image,
                        const std::string &filepath,
                        std::shared_ptr<ThirdOrderEdgeDetectionCPU> &toed,
                        std::vector<Edge> &edges)
{
    std::string path = filepath + ".bin";

    if (std::filesystem::exists(path))
    {
        // std::cout << "Loading edge data from: " << path << std::endl;
        ReadEdgesFromBinary(path, edges);
    }
    else
    {
        // std::cout << "Running third-order edge detector..." << std::endl;
        toed->get_Third_Order_Edges(image);
        edges = toed->toed_edges;

        WriteEdgesToBinary(path, edges);
        // std::cout << "Saved edge data to: " << path << std::endl;
    }
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

std::vector<Eigen::Vector2f> EBVO::LucasKanadeOpticalFlow(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<Edge> &edges,
    int patch_size)
{
    std::vector<Eigen::Vector2f> flow_vectors;

    if (edges.empty())
    {
        return flow_vectors;
    }

    // Convert images to CV_32F if necessary
    cv::Mat img1_float, img2_float;
    if (img1.type() != CV_32F)
    {
        img1.convertTo(img1_float, CV_32F);
    }
    else
    {
        img1_float = img1;
    }

    if (img2.type() != CV_32F)
    {
        img2.convertTo(img2_float, CV_32F);
    }
    else
    {
        img2_float = img2;
    }

    cv::Mat grad_x, grad_y;
    cv::Sobel(img1_float, grad_x, CV_32F, 1, 0, 3, 1.0 / 8.0);
    cv::Sobel(img1_float, grad_y, CV_32F, 0, 1, 3, 1.0 / 8.0);

    // Compute temporal gradient (I2 - I1)
    cv::Mat grad_t = img2_float - img1_float;

    // Initialize flow vectors
    flow_vectors.reserve(edges.size());

    int w = patch_size / 2;

    // Loop over all points
    for (size_t ci = 0; ci < edges.size(); ++ci)
    {
        // Get point locations
        int cx = static_cast<int>(std::round(edges[ci].location.x));
        int cy = static_cast<int>(std::round(edges[ci].location.y));

        // Check bounds
        if (cx - w < 0 || cx + w >= img1_float.cols ||
            cy - w < 0 || cy + w >= img1_float.rows)
        {
            flow_vectors.emplace_back(0.0f, 0.0f);
            continue;
        }

        // Extract patch around the point
        cv::Rect patch_rect(cx - w, cy - w, 2 * w + 1, 2 * w + 1);
        cv::Mat Ix_patch = grad_x(patch_rect).clone(); // Make continuous
        cv::Mat Iy_patch = grad_y(patch_rect).clone(); // Make continuous
        cv::Mat It_patch = grad_t(patch_rect).clone(); // Make continuous

        // Flatten patches to vectors
        cv::Mat Ix_vec = Ix_patch.reshape(1, (2 * w + 1) * (2 * w + 1));
        cv::Mat Iy_vec = Iy_patch.reshape(1, (2 * w + 1) * (2 * w + 1));
        cv::Mat It_vec = It_patch.reshape(1, (2 * w + 1) * (2 * w + 1));

        // Build matrix A = [Ix_patch, Iy_patch]
        cv::Mat A(Ix_vec.rows, 2, CV_32F);
        Ix_vec.copyTo(A.col(0));
        Iy_vec.copyTo(A.col(1));

        // Solve the Lucas-Kanade equation: A^T * A * [du; dv] = -A^T * It
        cv::Mat ATA = A.t() * A;
        cv::Mat ATb = -A.t() * It_vec; // Note the negative sign

        // Check if ATA is invertible (determinant > threshold)
        double det = cv::determinant(ATA);
        if (std::abs(det) < 1e-5) // Relaxed threshold
        {
            // Matrix is nearly singular, cannot solve reliably
            flow_vectors.emplace_back(0.0f, 0.0f);
            continue;
        }

        // Solve for flow vector
        cv::Mat flow_solution;
        cv::solve(ATA, ATb, flow_solution, cv::DECOMP_LU);

        float du = flow_solution.at<float>(0, 0);
        float dv = flow_solution.at<float>(1, 0);

        // Clamp flow vectors to reasonable bounds to avoid extreme movements
        float max_flow = 50.0f; // Maximum pixel movement per frame
        du = std::max(-max_flow, std::min(max_flow, du));
        dv = std::max(-max_flow, std::min(max_flow, dv));

        // Debug output for first few points
        if (ci < 5)
        {
            std::cout << "    Point " << ci << " at (" << edges[ci].location.x << "," << edges[ci].location.y
                      << ") -> flow: (" << du << "," << dv << ")" << std::endl;
        }

        flow_vectors.emplace_back(du, dv);
    }

    return flow_vectors;
}

void EBVO::VisualizeTracks_OpenCVStyle(
    const std::vector<std::vector<cv::Point2d>> &all_tracks,
    const std::vector<cv::Mat> &left_images,
    int n_tracks)
{
    if (all_tracks.empty() || left_images.empty())
    {
        std::cout << "No tracks or images available for OpenCV-style visualization" << std::endl;
        return;
    }

    // Limit n_tracks to available tracks
    int num_tracks_to_show = std::min(n_tracks, (int)all_tracks.size());

    // Randomly select track indices
    std::vector<int> selected_track_indices;
    if (num_tracks_to_show >= (int)all_tracks.size())
    {
        // If we want to show all tracks, just use all indices
        for (int i = 0; i < (int)all_tracks.size(); i++)
        {
            selected_track_indices.push_back(i);
        }
    }
    else
    {
        // Randomly sample without replacement
        std::vector<int> all_indices;
        for (int i = 0; i < (int)all_tracks.size(); i++)
        {
            all_indices.push_back(i);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(all_indices.begin(), all_indices.end(), gen);

        for (int i = 0; i < num_tracks_to_show; i++)
        {
            selected_track_indices.push_back(all_indices[i]);
        }
    }

    // Generate random colors for tracks
    cv::RNG rng(12345);
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < num_tracks_to_show; i++)
    {
        colors.push_back(cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)));
    }

    std::cout << "OpenCV-style visualization of " << num_tracks_to_show << " tracks across " << left_images.size() << " images" << std::endl;

    // Create output directory
    std::string viz_dir = dataset.get_output_path() + "/track_visualization_opencv_style";
    std::filesystem::create_directories(viz_dir);

    // Initialize mask - this will accumulate all track lines
    cv::Mat mask = cv::Mat::zeros(left_images[0].size(), CV_8UC3);

    // For each frame, draw track segments and points
    for (size_t frame_idx = 0; frame_idx < left_images.size(); ++frame_idx)
    {
        cv::Mat img_vis;
        if (left_images[frame_idx].channels() == 1)
            cv::cvtColor(left_images[frame_idx], img_vis, cv::COLOR_GRAY2BGR);
        else
            img_vis = left_images[frame_idx].clone();

        for (size_t i = 0; i < selected_track_indices.size(); ++i)
        {
            int track_id = selected_track_indices[i];
            const auto &track = all_tracks[track_id];
            if ((int)track.size() <= frame_idx) continue;

            const auto &pt = track[frame_idx];
            if (pt.x >= 0 && pt.y >= 0)  // Optional: skip invalid points
            {
                cv::circle(img_vis, pt, 2, colors[i], -1);

                // Draw line from previous point if exists
                if (frame_idx > 0 && (int)track.size() > frame_idx - 1)
                {
                    const auto &pt_prev = track[frame_idx - 1];
                    if (pt_prev.x >= 0 && pt_prev.y >= 0)
                    {
                        cv::line(img_vis, pt_prev, pt, colors[i], 1);
                    }
                }
            }
        }

        std::stringstream ss;
        ss << viz_dir << "/frame_" << std::setw(4) << std::setfill('0') << frame_idx << ".png";
        cv::imwrite(ss.str(), img_vis);
    }

    std::cout << "OpenCV-style track visualization saved to: " << viz_dir << std::endl;
}