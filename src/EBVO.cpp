#include <filesystem>
#include <unordered_set>
#include "EBVO.h"
#include "Dataset.h"
#include "Matches.h"
#include <opencv2/core/eigen.hpp>

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

    StereoFrame current_frame, next_frame;

    cv::Mat left_calib_inv = left_calib.inv();
    cv::Mat right_calib_inv = right_calib.inv();

    // Get the first frame from the dataset
    if (!dataset.stereo_iterator->hasNext() ||
        !dataset.stereo_iterator->getNext(current_frame))
    {
        LOG_ERROR("Failed to get first frame from dataset");
        return;
    }

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
        // If the current frame has ground truth, we can use it
        if (dataset.has_gt())
        {
        }
        const cv::Mat &left_ref_map = (frame_idx < left_ref_disparity_maps.size()) ? left_ref_disparity_maps[frame_idx] : cv::Mat();

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

        cv::Mat left_edge_map = cv::Mat::zeros(left_undistorted.size(), CV_8UC1);
        cv::Mat right_edge_map = cv::Mat::zeros(right_undistorted.size(), CV_8UC1);

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
#if DEBUG_EDGE_MATCHES_BETWEEN_LEFT_IMGS
        if (dataset.has_gt())
        {
            int indices[9] = {657, 10000, 10350, 20300, 30200, 35006, 40000, 46741};
            std::vector<Edge> gt_edges;
            std::vector<std::pair<Edge, Edge>> left_edges_GT_pair;
            for (int i = 0; i < 9; ++i)
            {
                int index = indices[i];
                // Get the ground truth edge for the left image
                Edge GTEdge = GetGTEdge(true, current_frame, next_frame,
                                        left_ref_map, left_calib_inv, left_calib,
                                        dataset.left_edges[index]);
                if ( GTEdge.b_isEmpty )
                    continue;
                else 
                {
                    left_edges_GT_pair.push_back(std::make_pair(dataset.left_edges[index], GTEdge));
                    gt_edges.push_back(GTEdge);
                }
            }

            for (int i = 0; i < left_edges_GT_pair.size(); i++)
            {
                cv::Point2d left_edge_t0 = left_edges_GT_pair[i].first.location;
                cv::Point2d left_edge_t1 = left_edges_GT_pair[i].second.location;
                // std::cout << "[" << left_edge_t0.x << ", " << left_edge_t0.y << "] - [" << left_edge_t1.x << ", " << left_edge_t1.y << "]" << std::endl;
            }

            cv::Mat current_frame_vis;
            cv::cvtColor(curr_left_img, current_frame_vis, cv::COLOR_GRAY2BGR);

            // Different colors for different selected points
            std::vector<cv::Scalar> colors = {
                cv::Scalar(0, 0, 255),   // Red
                cv::Scalar(0, 255, 0),   // Green
                cv::Scalar(255, 0, 0),   // Blue
                cv::Scalar(0, 255, 255), // Yellow
                cv::Scalar(255, 0, 255), // Magenta
                cv::Scalar(255, 255, 0), // Cyan
                cv::Scalar(128, 0, 128), // Purple
                cv::Scalar(255, 165, 0), // Orange
                cv::Scalar(255, 20, 147) // Deep Pink
            };

            for (int i = 0; i < 9; ++i)
            {
                int index = indices[i];
                if (index < dataset.left_edges.size())
                {
                    cv::Point2d pt = dataset.left_edges[index].location;
                    cv::circle(current_frame_vis, pt, 8, colors[i], -1);
                    cv::putText(current_frame_vis, std::to_string(i),
                                cv::Point(pt.x + 10, pt.y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2);
                }
            }

            // Draw GT edges on next frame
            cv::Mat next_frame_vis;
            cv::cvtColor(next_left_img, next_frame_vis, cv::COLOR_GRAY2BGR);

            for (int i = 0; i < gt_edges.size(); ++i)
            {
                cv::Point2d gt_pt = gt_edges[i].location;
                // Check if point is within image bounds
                if (gt_pt.x >= 0 && gt_pt.x < next_frame_vis.cols &&
                    gt_pt.y >= 0 && gt_pt.y < next_frame_vis.rows)
                {
                    cv::circle(next_frame_vis, gt_pt, 8, colors[i], -1);
                    cv::putText(next_frame_vis, std::to_string(i),
                                cv::Point(gt_pt.x + 10, gt_pt.y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2);
                }
            }

            // Save visualization images
            std::string output_dir = dataset.get_output_path();
            cv::imwrite(output_dir + "/current_frame_selected_edges_" + std::to_string(frame_idx) + ".png",
                        current_frame_vis);
            cv::imwrite(output_dir + "/next_frame_gt_edges_" + std::to_string(frame_idx) + ".png",
                        next_frame_vis);

            std::cout << "Saved edge tracking visualization for frame " << frame_idx << std::endl;
        }
#endif

        StereoMatchResult match_result = get_Stereo_Edge_Pairs(
            left_undistorted,
            right_undistorted,
            dataset);

        exit(1);

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
        if (frame_idx >= 1)
        {
            break;
        }
        current_frame = next_frame;
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
        std::cout << "Invalid disparity value: " << disparity << " for edge at location: " << edge.location << std::endl;
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