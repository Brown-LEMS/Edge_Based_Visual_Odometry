#include "EBVO.h"
#include "Dataset.h"
#include "Matches.h"

EBVO(YAML::Node config_map, bool use_GCC_filter = false)
{
    dataset = Dataset(config_map, use_GCC_filter);
}

void PerformEdgeBasedVO();
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

    load_dataset(image_pairs, dataset_type, num_pairs);

    auto start_time = std::chrono::high_resolution_clock::now();

    LOG_INFO("Start looping over all image pairs");

    std::cout << "There are " << image_pairs.size() << " image pairs" << std::endl;

    // Figure out a solution for this (will skip the last image pair)
    for (size_t i = 0; i < image_pairs.size() - 1; ++i)
    {
        const cv::Mat &curr_left_img = image_pairs[i].first;
        const cv::Mat &curr_right_img = image_pairs[i].second;

        const cv::Mat &left_ref_map = left_ref_disparity_maps[i];
        // const cv::Mat& right_ref_map = right_ref_disparity_maps[i];

        const cv::Mat &next_left_img = image_pairs[i + 1].first;
        const cv::Mat &next_right_img = image_pairs[i + 1].second;

        std::vector<cv::Mat> curr_left_pyramid, curr_right_pyramid;
        std::vector<cv::Mat> next_left_pyramid, next_right_pyramid;
        int pyramid_levels = 4;

        BuildImagePyramids(
            curr_left_img,
            curr_right_img,
            next_left_img,
            next_right_img,
            pyramid_levels,
            curr_left_pyramid,
            curr_right_pyramid,
            next_left_pyramid,
            next_right_pyramid);

        ncc_one_vs_err.clear();
        ncc_two_vs_err.clear();
        dataset.ground_truth_right_edges_after_lowe.clear();

        std::cout << "Image Pair #" << i << "\n";
        cv::Mat left_calib = (cv::Mat_<double>(3, 3) << left_intr[0], 0, left_intr[2], 0, left_intr[1], left_intr[3], 0, 0, 1);
        cv::Mat right_calib = (cv::Mat_<double>(3, 3) << right_intr[0], 0, right_intr[2], 0, right_intr[1], right_intr[3], 0, 0, 1);
        cv::Mat left_dist_coeff_mat(dataset.left_dist_coeffs);
        cv::Mat right_dist_coeff_mat(dataset.right_dist_coeffs);

        cv::Mat left_undistorted, right_undistorted;
        cv::undistort(curr_left_img, left_undistorted, left_calib, left_dist_coeff_mat);
        cv::undistort(curr_right_img, right_undistorted, right_calib, right_dist_coeff_mat);

        if (dataset.Total_Num_Of_Imgs == 0)
        {
            img_height = left_undistorted.rows;
            img_width = left_undistorted.cols;

            TOED = std::shared_ptr<ThirdOrderEdgeDetectionCPU>(new ThirdOrderEdgeDetectionCPU(img_height, img_width));
        }

        std::string edge_dir = dataset.output_path + "/edges";
        std::filesystem::create_directories(edge_dir);

        std::string left_edge_path = edge_dir + "/left_edges_" + std::to_string(i);
        std::string right_edge_path = edge_dir + "/right_edges_" + std::to_string(i);

        ProcessEdges(left_undistorted, left_edge_path, TOED, dataset.left_third_order_edges_locations, dataset.left_third_order_edges_orientation);
        std::cout << "Number of edges on the left image: " << dataset.left_third_order_edges_locations.size() << std::endl;

        ProcessEdges(right_undistorted, right_edge_path, TOED, dataset.right_third_order_edges_locations, dataset.right_third_order_edges_orientation);
        std::cout << "Number of edges on the right image: " << right_third_order_edges_locations.size() << std::endl;

        dataset.Total_Num_Of_Imgs++;

        cv::Mat left_edge_map = cv::Mat::zeros(left_undistorted.size(), CV_8UC1);
        cv::Mat right_edge_map = cv::Mat::zeros(right_undistorted.size(), CV_8UC1);

        for (const auto &edge : dataset.left_third_order_edges_locations)
        {
            if (edge.x >= 0 && edge.x < left_edge_map.cols && edge.y >= 0 && edge.y < left_edge_map.rows)
            {
                left_edge_map.at<uchar>(cv::Point(edge.x, edge.y)) = 255;
            }
        }

        for (const auto &edge : dataset.right_third_order_edges_locations)
        {
            if (edge.x >= 0 && edge.x < right_edge_map.cols && edge.y >= 0 && edge.y < right_edge_map.rows)
            {
                right_edge_map.at<uchar>(cv::Point(edge.x, edge.y)) = 255;
            }
        }

        CalculateGTRightEdge(dataset.left_third_order_edges_locations, dataset.left_third_order_edges_orientation, left_ref_map, left_edge_map, right_edge_map);
        // CalculateGTLeftEdge(right_third_order_edges_locations, right_third_order_edges_orientation, right_ref_map, left_edge_map, right_edge_map);

        StereoMatchResult match_result = DisplayMatches(
            left_undistorted,
            right_undistorted,
            right_third_order_edges_locations,
            right_third_order_edges_orientation);

#if 0
        std::vector<cv::Point3d> points_opencv = Calculate3DPoints(match_result.confirmed_matches);
        std::vector<cv::Point3d> points_linear = LinearTriangulatePoints(match_result.confirmed_matches);
        std::vector<Eigen::Vector3d> orientations_3d = Calculate3DOrientations(match_result.confirmed_matches);

        std::vector<OrientedPoint3D> oriented_points;

        for (size_t i = 0; i < points_opencv.size(); ++i) {
            OrientedPoint3D op;
            op.position = points_opencv[i];
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

    std::string edge_stat_dir = output_path + "/edge_stats";
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

/*
    Extract patches from the image based on the cluster centers and shifted edge points.
    The function checks if the patches are within bounds before extracting them.
*/
void ExtractClusterPatches(
    int patch_size,
    const cv::Mat &image,
    const std::vector<EdgeCluster> &cluster_centers,
    const std::vector<cv::Point2d> *right_edges,
    const std::vector<cv::Point2d> &shifted_one,
    const std::vector<cv::Point2d> &shifted_two,
    std::vector<EdgeCluster> &cluster_centers_out,
    std::vector<cv::Point2d> *filtered_right_edges_out,
    std::vector<cv::Mat> &patch_set_one_out,
    std::vector<cv::Mat> &patch_set_two_out)
{
    int half_patch = std::ceil(patch_size / 2);

    for (int i = 0; i < shifted_one.size(); i++)
    {
        double x1 = shifted_one[i].x;
        double y1 = shifted_one[i].y;
        double x2 = shifted_two[i].x;
        double y2 = shifted_two[i].y;

        bool in_bounds_one = (x1 - half_patch >= 0 && x1 + half_patch < image.cols &&
                              y1 - half_patch >= 0 && y1 + half_patch < image.rows);
        bool in_bounds_two = (x2 - half_patch >= 0 && x2 + half_patch < image.cols &&
                              y2 - half_patch >= 0 && y2 + half_patch < image.rows);

        if (in_bounds_one && in_bounds_two)
        {
            cv::Point2f center1(static_cast<float>(x1), static_cast<float>(y1));
            cv::Point2f center2(static_cast<float>(x2), static_cast<float>(y2));
            cv::Size size(patch_size, patch_size);

            cv::Mat patch1, patch2;
            cv::getRectSubPix(image, size, center1, patch1);
            cv::getRectSubPix(image, size, center2, patch2);

            if (patch1.type() != CV_32F)
            {
                patch1.convertTo(patch1, CV_32F);
            }
            if (patch2.type() != CV_32F)
            {
                patch2.convertTo(patch2, CV_32F);
            }

            patch_set_one_out.push_back(patch1);
            patch_set_two_out.push_back(patch2);
            cluster_centers_out.push_back(cluster_centers[i]);

            if (right_edges && filtered_right_edges_out)
            {
                filtered_right_edges_out->push_back((*right_edges)[i]);
            }
        }
    }
}

/*
    Cluster the shifted edges based on their proximity and orientation.
    Returns a vector of clusters, where each cluster contains a pair of vectors:
    one for the edge points and one for their corresponding orientations.
    The clustering is based on a distance threshold and an orientation difference threshold.
*/
std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> Dataset::ClusterEpipolarShiftedEdges(std::vector<cv::Point2d> &valid_shifted_edges, std::vector<double> &valid_shifted_orientations)
{
    std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> clusters;

    if (valid_shifted_edges.empty() || valid_shifted_orientations.empty())
    {
        return clusters;
    }

    std::vector<std::pair<cv::Point2d, double>> edge_orientation_pairs;
    for (size_t i = 0; i < valid_shifted_edges.size(); ++i)
    {
        edge_orientation_pairs.emplace_back(valid_shifted_edges[i], valid_shifted_orientations[i]);
    }

    std::sort(edge_orientation_pairs.begin(), edge_orientation_pairs.end(),
              [](const std::pair<cv::Point2d, double> &a, const std::pair<cv::Point2d, double> &b)
              {
                  return a.first.x < b.first.x;
              });

    valid_shifted_edges.clear();
    valid_shifted_orientations.clear();
    for (const auto &pair : edge_orientation_pairs)
    {
        valid_shifted_edges.push_back(pair.first);
        valid_shifted_orientations.push_back(pair.second);
    }

    std::vector<cv::Point2d> current_cluster_edges;
    std::vector<double> current_cluster_orientations;
    current_cluster_edges.push_back(valid_shifted_edges[0]);
    current_cluster_orientations.push_back(valid_shifted_orientations[0]);

    for (size_t i = 1; i < valid_shifted_edges.size(); ++i)
    {
        double distance = cv::norm(valid_shifted_edges[i] - valid_shifted_edges[i - 1]);
        double orientation_difference = std::abs(valid_shifted_orientations[i] - valid_shifted_orientations[i - 1]);

        if (distance <= EDGE_CLUSTER_THRESH && orientation_difference < 5.0)
        {
            current_cluster_edges.push_back(valid_shifted_edges[i]);
            current_cluster_orientations.push_back(valid_shifted_orientations[i]);
        }
        else
        {
            clusters.emplace_back(current_cluster_edges, current_cluster_orientations);
            current_cluster_edges.clear();
            current_cluster_orientations.clear();
            current_cluster_edges.push_back(valid_shifted_edges[i]);
            current_cluster_orientations.push_back(valid_shifted_orientations[i]);
        }
    }

    if (!current_cluster_edges.empty())
    {
        clusters.emplace_back(current_cluster_edges, current_cluster_orientations);
    }

    return clusters;
}

/*
    Extract edges that are close to the epipolar line within a specified distance threshold.
    Returns a pair of vectors: one for the extracted edge locations and one for their orientations.
*/
std::pair<std::vector<cv::Point2d>, std::vector<double>> Dataset::ExtractEpipolarEdges(const Eigen::Vector3d &epipolar_line, const std::vector<cv::Point2d> &edge_locations, const std::vector<double> &edge_orientations, double distance_threshold)
{
    std::vector<cv::Point2d> extracted_edges;
    std::vector<double> extracted_orientations;

    if (edge_locations.size() != edge_orientations.size())
    {
        throw std::runtime_error("Edge locations and orientations size mismatch.");
    }

    for (size_t i = 0; i < edge_locations.size(); ++i)
    {
        const auto &edge = edge_locations[i];
        double x = edge.x;
        double y = edge.y;

        double distance = std::abs(epipolar_line(0) * x + epipolar_line(1) * y + epipolar_line(2)) / std::sqrt((epipolar_line(0) * epipolar_line(0)) + (epipolar_line(1) * epipolar_line(1)));

        if (distance < distance_threshold)
        {
            extracted_edges.push_back(edge);
            extracted_orientations.push_back(edge_orientations[i]);
        }
    }

    return {extracted_edges, extracted_orientations};
}

/*
    Pick a random edge
*/
std::tuple<std::vector<cv::Point2d>, std::vector<double>, std::vector<cv::Point2d>> Dataset::PickRandomEdges(int patch_size, const std::vector<cv::Point2d> &edges, const std::vector<cv::Point2d> &ground_truth_right_edges, const std::vector<double> &orientations, size_t num_points, int img_width, int img_height)
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

/*
    Perform an epipolar shift on the original edge location based on the epipolar line coefficients.
    The function checks if the corrected edge passes the epipolar tengency test.
*/
cv::Point2d Dataset::PerformEpipolarShift(
    cv::Point2d original_edge_location, double edge_orientation,
    std::vector<double> epipolar_line_coeffs, bool &b_pass_epipolar_tengency_check)
{
    cv::Point2d corrected_edge;
    assert(epipolar_line_coeffs.size() == 3);
    double EL_coeff_A = epipolar_line_coeffs[0];
    double EL_coeff_B = epipolar_line_coeffs[1];
    double EL_coeff_C = epipolar_line_coeffs[2];
    double a1_line = -epipolar_line_coeffs[0] / epipolar_line_coeffs[1];
    double b1_line = -1;
    double c1_line = -epipolar_line_coeffs[2] / epipolar_line_coeffs[1];

    //> Parameters of the line passing through the original edge along its direction (tangent) vector
    double a_edgeH2 = tan(edge_orientation);
    double b_edgeH2 = -1;
    double c_edgeH2 = -(a_edgeH2 * original_edge_location.x - original_edge_location.y); // −(a⋅x2−y2)

    //> Find the intersected point of the two lines
    corrected_edge.x = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    corrected_edge.y = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);

    //> Find (i) the displacement between the original edge and the corrected edge, and
    //       (ii) the intersection angle between the epipolar line and the line passing through the original edge along its direction vector
    double epipolar_shift_displacement = cv::norm(corrected_edge - original_edge_location);
    double m_epipolar = -a1_line / b1_line; //> Slope of epipolar line
    double angle_diff_rad = abs(edge_orientation - atan(m_epipolar));
    double angle_diff_deg = angle_diff_rad * (180.0 / M_PI);
    if (angle_diff_deg > 180)
    {
        angle_diff_deg -= 180;
    }

    //> check if the corrected edge passes the epoplar tengency test (intersection angle < 4 degrees and displacement < 6 pixels)
    b_pass_epipolar_tengency_check = (epipolar_shift_displacement < EPIP_TENGENCY_PROXIM_THRESH && abs(angle_diff_deg - 0) > EPIP_TENGENCY_ORIENT_THRESH && abs(angle_diff_deg - 180) > EPIP_TENGENCY_ORIENT_THRESH) ? (true) : (false);

    return corrected_edge;
}
