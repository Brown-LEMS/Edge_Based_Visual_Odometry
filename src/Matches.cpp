#include "Matches.h"

/*
    calculate the epipolar line for each edge point using the fundamental matrix.
*/
std::vector<Eigen::Vector3d> Dataset::CalculateEpipolarLine(const Eigen::Matrix3d &fund_mat, const std::vector<cv::Point2d> &edges)
{
    std::vector<Eigen::Vector3d> epipolar_lines;

    for (const auto &point : edges)
    {
        Eigen::Vector3d homo_point(point.x, point.y, 1.0);

        Eigen::Vector3d epipolar_line = fund_mat * homo_point;

        epipolar_lines.push_back(epipolar_line);
    }

    return epipolar_lines;
}

/*
    Calculate orthogonal shifts for edge points based on their orientations.
    Returns two vectors of shifted points, one for each orthogonal direction.
*/
std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> CalculateOrthogonalShifts(const std::vector<cv::Point2d> &edge_points, const std::vector<double> &orientations, double shift_magnitude)
{
    std::vector<cv::Point2d> shifted_points_one;
    std::vector<cv::Point2d> shifted_points_two;

    if (edge_points.size() != orientations.size())
    {
        std::cerr << "ERROR: Mismatch between number of edge points and orientations." << std::endl;
        return {shifted_points_one, shifted_points_two};
    }

    for (size_t i = 0; i < edge_points.size(); ++i)
    {
        const auto &edge_point = edge_points[i];
        double theta = orientations[i];

        double orthogonal_x1 = std::sin(theta);
        double orthogonal_y1 = -std::cos(theta);
        double orthogonal_x2 = -std::sin(theta);
        double orthogonal_y2 = std::cos(theta);

        double shifted_x1 = edge_point.x + shift_magnitude * orthogonal_x1;
        double shifted_y1 = edge_point.y + shift_magnitude * orthogonal_y1;
        double shifted_x2 = edge_point.x + shift_magnitude * orthogonal_x2;
        double shifted_y2 = edge_point.y + shift_magnitude * orthogonal_y2;

        shifted_points_one.emplace_back(shifted_x1, shifted_y1);
        shifted_points_two.emplace_back(shifted_x2, shifted_y2);
    }

    return {shifted_points_one, shifted_points_two};
}

StereoMatchResult DisplayMatches(const cv::Mat &left_image, const cv::Mat &right_image, std::vector<cv::Point2d> right_edge_coords, std::vector<double> right_edge_orientations, Dataset &dataset)
{
    ///////////////////////////////FORWARD DIRECTION///////////////////////////////
    std::vector<cv::Point2d> left_edge_coords;
    std::vector<cv::Point2d> ground_truth_right_edges;
    std::vector<double> left_edge_orientations;

    for (const auto &data : dataset.forward_gt_data)
    {
        left_edge_coords.push_back(std::get<0>(data));
        ground_truth_right_edges.push_back(std::get<1>(data));
        left_edge_orientations.push_back(std::get<2>(data));
    }

    auto [left_orthogonal_one, left_orthogonal_two] = CalculateOrthogonalShifts(left_edge_coords, left_edge_orientations, ORTHOGONAL_SHIFT_MAG);

    std::vector<cv::Point2d> filtered_left_edges;
    std::vector<double> filtered_left_orientations;
    std::vector<cv::Point2d> filtered_ground_truth_right_edges;

    std::vector<cv::Mat> left_patch_set_one;
    std::vector<cv::Mat> left_patch_set_two;

    ExtractPatches(
        PATCH_SIZE,
        left_image,
        left_edge_coords,
        left_edge_orientations,
        left_orthogonal_one,
        left_orthogonal_two,
        filtered_left_edges,
        filtered_left_orientations,
        left_patch_set_one,
        left_patch_set_two,
        &ground_truth_right_edges,
        &filtered_ground_truth_right_edges);

    Eigen::Matrix3d fundamental_matrix_21 = ConvertToEigenMatrix(dataset.fund_mat_21);
    Eigen::Matrix3d fundamental_matrix_12 = ConvertToEigenMatrix(dataset.fund_mat_12);

    std::vector<Eigen::Vector3d> epipolar_lines_right = CalculateEpipolarLine(fundamental_matrix_21, filtered_left_edges);

    EdgeMatchResult forward_match = CalculateMatches(
        filtered_left_edges,
        filtered_left_orientations,
        right_edge_coords,
        right_edge_orientations,
        left_patch_set_one,
        left_patch_set_two,
        epipolar_lines_right,
        right_image,
        filtered_ground_truth_right_edges,
        dataset);

    ///////////////////////////////REVERSE DIRECTION///////////////////////////////
    std::vector<cv::Point2d> reverse_primary_edges;
    std::vector<double> reverse_primary_orientations;

    for (const auto &match_pair : forward_match.edge_to_cluster_matches)
    {
        const EdgeMatch &match_info = match_pair.second;

        for (const auto &edge : match_info.contributing_edges)
        {
            reverse_primary_edges.push_back(edge);
        }
        for (const auto &orientation : match_info.contributing_orientations)
        {
            reverse_primary_orientations.push_back(orientation);
        }
    }

    auto [right_orthogonal_one, right_orthogonal_two] = CalculateOrthogonalShifts(reverse_primary_edges, reverse_primary_orientations, ORTHOGONAL_SHIFT_MAG);

    std::vector<cv::Point2d> filtered_right_edges;
    std::vector<double> filtered_right_orientations;
    std::vector<cv::Point2d> filtered_ground_truth_left_edges;

    std::vector<cv::Mat> right_patch_set_one;
    std::vector<cv::Mat> right_patch_set_two;

    ExtractPatches(
        PATCH_SIZE,
        right_image,
        reverse_primary_edges,
        reverse_primary_orientations,
        right_orthogonal_one,
        right_orthogonal_two,
        filtered_right_edges,
        filtered_right_orientations,
        right_patch_set_one,
        right_patch_set_two,
        nullptr,
        nullptr);

    std::vector<Eigen::Vector3d> epipolar_lines_left = CalculateEpipolarLine(fundamental_matrix_12, filtered_right_edges);

    EdgeMatchResult reverse_match = CalculateMatches(
        filtered_right_edges,
        filtered_right_orientations,
        left_edge_coords,
        left_edge_orientations,
        right_patch_set_one,
        right_patch_set_two,
        epipolar_lines_left,
        left_image,
        dataset);

    std::vector<std::pair<ConfirmedMatchEdge, ConfirmedMatchEdge>> confirmed_matches;

    int matches_before_bct = static_cast<int>(forward_match.edge_to_cluster_matches.size());
    std::cout << "Number of matches before BCT: " << matches_before_bct << std::endl;

    auto bct_start = std::chrono::high_resolution_clock::now();

    int forward_left_index = 0;
    int bct_true_positive = 0;
    for (const auto &[left_oriented_edge, patch_match_forward] : forward_match.edge_to_cluster_matches)
    {
        const cv::Point2d &left_position = left_oriented_edge.position;
        const double left_orientation = left_oriented_edge.orientation;

        const auto &right_contributing_edges = patch_match_forward.contributing_edges;
        const auto &right_contributing_orientations = patch_match_forward.contributing_orientations;
        bool break_flag = false;
        for (size_t i = 0; i < right_contributing_edges.size(); ++i)
        {
            break_flag = false;
            const cv::Point2d &right_position = right_contributing_edges[i];
            const double right_orientation = right_contributing_orientations[i];

            for (const auto &[rev_right_edge, patch_match_rev] : reverse_match.edge_to_cluster_matches)
            {
                if (cv::norm(rev_right_edge.position - right_position) <= MATCH_TOL)
                {

                    for (const auto &rev_contributing_left : patch_match_rev.contributing_edges)
                    {
                        if (cv::norm(rev_contributing_left - left_position) <= MATCH_TOL)
                        {
                            ConfirmedMatchEdge left_confirmed{left_position, left_orientation};
                            ConfirmedMatchEdge right_confirmed{right_position, right_orientation};
                            confirmed_matches.emplace_back(left_confirmed, right_confirmed);

                            cv::Point2d GT_right_edge_location = ground_truth_right_edges_after_lowe[forward_left_index];
                            if (cv::norm(right_position - GT_right_edge_location) <= MATCH_TOL)
                            {
                                bct_true_positive++;
                            }
                            break_flag = true;
                            break;
                        }
                    }
                }
                if (break_flag)
                    break;
            }
            if (break_flag)
                break;
        }
        forward_left_index++;
    }

    // //> Measure the recall of bidirectional consistency test
    // int bct_true_positive = 0;
    // for (int i = 0; i < confirmed_matches.size(); i++) {
    //     ConfirmedMatchEdge right_confirmed = confirmed_matches[i].second;
    //     cv::Point2d right_edge_location = right_confirmed.position;
    //     cv::Point2d GT_right_edge_location = ground_truth_right_edges_after_lowe[i];
    //     if (cv::norm(right_edge_location - GT_right_edge_location) <= MATCH_TOL) {
    //         bct_true_positive++;
    //     }
    // }
    std::cout << "BCT true positives: " << bct_true_positive << std::endl;

    auto bct_end = std::chrono::high_resolution_clock::now();
    double total_time_bct = std::chrono::duration<double, std::milli>(bct_end - bct_start).count();

    double per_image_bct_time = (matches_before_bct > 0) ? total_time_bct / matches_before_bct : 0.0;

    int matches_after_bct = static_cast<int>(confirmed_matches.size());
    std::cout << "Number of matches after BCT: " << matches_after_bct << std::endl;
    std::cout << "Number of stacked GT right edges: " << ground_truth_right_edges_after_lowe.size() << std::endl;

    // double per_image_bct_precision = (matches_before_bct > 0) ? static_cast<double>(matches_after_bct) / matches_before_bct: 0.0;
    double per_image_bct_precision = (matches_before_bct > 0) ? bct_true_positive / (double)(matches_after_bct) : 0.0;
    std::cout << "BCT precision = " << per_image_bct_precision << std::endl;

    int bct_denonimator = forward_match.recall_metrics.lowe_true_positive + forward_match.recall_metrics.lowe_false_negative;
    // int bct_true_positives = static_cast<int>(confirmed_matches.size());

    double bct_recall = (bct_denonimator > 0) ? bct_true_positive / (double)(bct_denonimator) : 0.0;
    std::cout << "BCT recall = " << bct_recall << std::endl;

    BidirectionalMetrics bidirectional_metrics;
    bidirectional_metrics.matches_before_bct = matches_before_bct;
    bidirectional_metrics.matches_after_bct = matches_after_bct;
    bidirectional_metrics.per_image_bct_recall = bct_recall;
    bidirectional_metrics.per_image_bct_precision = per_image_bct_precision;
    bidirectional_metrics.per_image_bct_time = per_image_bct_time;

    return StereoMatchResult{forward_match, reverse_match, confirmed_matches, bidirectional_metrics};
}

//> MARK: Main Edge Pairing
EdgeMatchResult CalculateMatches(const std::vector<cv::Point2d> &selected_primary_edges, const std::vector<double> &selected_primary_orientations, const std::vector<cv::Point2d> &secondary_edge_coords,
                                 const std::vector<double> &secondary_edge_orientations, const std::vector<cv::Mat> &primary_patch_set_one, const std::vector<cv::Mat> &primary_patch_set_two, const std::vector<Eigen::Vector3d> &epipolar_lines_secondary,
                                 const cv::Mat &secondary_image, const std::vector<cv::Point2d> &selected_ground_truth_edges,
                                 Dataset &dataset)
{
    auto total_start = std::chrono::high_resolution_clock::now();

    std::vector<int> epi_input_counts;
    std::vector<int> epi_output_counts;

    std::vector<int> disp_input_counts;
    std::vector<int> disp_output_counts;

    std::vector<int> shift_input_counts;
    std::vector<int> shift_output_counts;

    std::vector<int> clust_input_counts;
    std::vector<int> clust_output_counts;

    std::vector<int> patch_input_counts;
    std::vector<int> patch_output_counts;

    std::vector<int> ncc_input_counts;
    std::vector<int> ncc_output_counts;

    std::vector<int> lowe_input_counts;
    std::vector<int> lowe_output_counts;

    double total_time;

    //> CH: this is a global structure of final_matches
    std::vector<std::pair<SourceEdge, EdgeMatch>> final_matches;

    //> CH: this is local structure of final matches
    std::vector<std::vector<std::pair<SourceEdge, EdgeMatch>>> local_final_matches(omp_get_max_threads());

    //> CH: Local structures of all counts
    std::vector<std::vector<int>> local_epi_input_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_epi_output_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_disp_input_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_disp_output_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_shift_input_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_shift_output_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_clust_input_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_clust_output_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_patch_input_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_patch_output_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_ncc_input_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_ncc_output_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_lowe_input_counts(omp_get_max_threads());
    std::vector<std::vector<int>> local_lowe_output_counts(omp_get_max_threads());

    //> CH: Local structures for GT right edge after Lowe's ratio test
    std::vector<std::vector<cv::Point2d>> local_GT_right_edges_after_lowe(omp_get_max_threads());

    int time_epi_edges_evaluated = 0;
    int time_disp_edges_evaluated = 0;
    int time_shift_edges_evaluated = 0;
    int time_clust_edges_evaluated = 0;
    int time_patch_edges_evaluated = 0;
    int time_ncc_edges_evaluated = 0;
    int time_lowe_edges_evaluated = 0;
    double time_epi = 0.0;
    double time_disp = 0.0;
    double time_shift = 0.0;
    double time_patch = 0.0;
    double time_cluster = 0.0;
    double time_ncc = 0.0;
    double time_lowe = 0.0;

    //> These are global variables for reduction sum
    double per_edge_epi_precision = 0.0;
    double per_edge_disp_precision = 0.0;
    double per_edge_shift_precision = 0.0;
    double per_edge_clust_precision = 0.0;
    double per_edge_ncc_precision = 0.0;
    double per_edge_lowe_precision = 0.0;
    int epi_true_positive = 0;
    int epi_false_negative = 0;
    int epi_true_negative = 0;
    int disp_true_positive = 0;
    int disp_false_negative = 0;
    int shift_true_positive = 0;
    int shift_false_negative = 0;
    int cluster_true_positive = 0;
    int cluster_false_negative = 0;
    int cluster_true_negative = 0;
    int ncc_true_positive = 0;
    int ncc_false_negative = 0;
    int lowe_true_positive = 0;
    int lowe_false_negative = 0;
    int epi_edges_evaluated = 0;
    int disp_edges_evaluated = 0;
    int shift_edges_evaluated = 0;
    int clust_edges_evaluated = 0;
    int ncc_edges_evaluated = 0;
    int lowe_edges_evaluated = 0;

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        cv::Point2d ground_truth_edge;

        // MAKE SURE TO UPDATE THIS ACCORDINGLY
        //  int skip = (!selected_ground_truth_edges.empty()) ? 100 : 1;
        const int skip = 1;

//> Start looping over left edges
#pragma omp for schedule(static, omp_threads) reduction(+ : epi_true_positive, epi_false_negative, epi_true_negative, disp_true_positive, disp_false_negative, shift_true_positive, shift_false_negative, cluster_true_positive, cluster_false_negative, cluster_true_negative, ncc_true_positive, ncc_false_negative, lowe_true_positive, lowe_false_negative, per_edge_epi_precision, per_edge_disp_precision, per_edge_shift_precision, per_edge_clust_precision, per_edge_ncc_precision, per_edge_lowe_precision, epi_edges_evaluated, disp_edges_evaluated, shift_edges_evaluated, clust_edges_evaluated, ncc_edges_evaluated, lowe_edges_evaluated)
        for (size_t i = 0; i < selected_primary_edges.size(); i += skip)
        {
            const auto &primary_edge = selected_primary_edges[i];
            const auto &primary_orientation = selected_primary_orientations[i];

            if (!selected_ground_truth_edges.empty())
            {
                ground_truth_edge = selected_ground_truth_edges[i];
            }

            const auto &epipolar_line = epipolar_lines_secondary[i];
            const auto &primary_patch_one = primary_patch_set_one[i];
            const auto &primary_patch_two = primary_patch_set_two[i];

            double a = epipolar_line(0);
            double b = epipolar_line(1);
            double c = epipolar_line(2);

            if (std::abs(b) < 1e-6)
                continue;

            double a1_line = -a / b;
            double b1_line = -1;

            double m_epipolar = -a1_line / b1_line;
            double angle_diff_rad = abs(primary_orientation - atan(m_epipolar));
            double angle_diff_deg = angle_diff_rad * (180.0 / M_PI);
            if (angle_diff_deg > 180)
            {
                angle_diff_deg -= 180;
            }

            bool primary_passes_tangency = (abs(angle_diff_deg - 0) > EPIP_TENGENCY_ORIENT_THRESH && abs(angle_diff_deg - 180) > EPIP_TENGENCY_ORIENT_THRESH) ? (true) : (false);
            if (!primary_passes_tangency)
            {
                continue;
            }

            ///////////////////////////////EPIPOLAR DISTANCE THRESHOLD///////////////////////////////
#if MEASURE_TIMINGS
            auto start_epi = std::chrono::high_resolution_clock::now();
#endif

            std::pair<std::vector<cv::Point2d>, std::vector<double>> secondary_candidates_data = ExtractEpipolarEdges(epipolar_line, secondary_edge_coords, secondary_edge_orientations, 0.5);
            std::vector<cv::Point2d> secondary_candidate_edges = secondary_candidates_data.first;
            std::vector<double> secondary_candidate_orientations = secondary_candidates_data.second;

            std::pair<std::vector<cv::Point2d>, std::vector<double>> test_secondary_candidates_data = ExtractEpipolarEdges(epipolar_line, secondary_edge_coords, secondary_edge_orientations, 3);
            std::vector<cv::Point2d> test_secondary_candidate_edges = test_secondary_candidates_data.first;

            // epi_input_counts.push_back(secondary_edge_coords.size());
            local_epi_input_counts[thread_id].push_back(secondary_edge_coords.size());

#if MEASURE_TIMINGS
            time_epi_edges_evaluated++;
            auto end_epi = std::chrono::high_resolution_clock::now();
            time_epi += std::chrono::duration<double, std::milli>(end_epi - start_epi).count();
#endif
            //> MARK: Epipolar Distance
            ///////////////////////////////EPIPOLAR DISTANCE THRESHOLD RECALL//////////////////////////
            if (!selected_ground_truth_edges.empty())
            {
                int epi_precision_numerator = 0;
                bool match_found = false;

                for (const auto &candidate : secondary_candidate_edges)
                {
                    if (cv::norm(candidate - ground_truth_edge) <= 0.5)
                    {
                        epi_precision_numerator++;
                        match_found = true;
                        // break;
                    }
                }

                if (match_found)
                {
                    epi_true_positive++;
                }
                else
                {
                    bool gt_match_found = false;
                    for (const auto &test_candidate : test_secondary_candidate_edges)
                    {
                        if (cv::norm(test_candidate - ground_truth_edge) <= 0.5)
                        {
                            gt_match_found = true;
                            break;
                        }
                    }

                    if (!gt_match_found)
                    {
                        epi_true_negative++;
                        continue;
                    }
                    else
                    {
                        epi_false_negative++;
                    }
                }
                if (!secondary_candidate_edges.empty())
                {
                    per_edge_epi_precision += static_cast<double>(epi_precision_numerator) / secondary_candidate_edges.size();
                    epi_edges_evaluated++;
                }
            }
            ///////////////////////////////MAXIMUM DISPARITY THRESHOLD//////////////////////////
#if MEASURE_TIMINGS
            auto start_disp = std::chrono::high_resolution_clock::now();
#endif

            // epi_output_counts.push_back(secondary_candidate_edges.size());
            local_epi_output_counts[thread_id].push_back(secondary_candidate_edges.size());

            std::vector<cv::Point2d> filtered_secondary_edge_coords;
            std::vector<double> filtered_secondary_edge_orientations;

            for (size_t j = 0; j < secondary_candidate_edges.size(); j++)
            {
                const cv::Point2d &secondary_edge = secondary_candidate_edges[j];

                double disparity = (!selected_ground_truth_edges.empty()) ? (primary_edge.x - secondary_edge.x) : (secondary_edge.x - primary_edge.x);

                bool within_horizontal = (disparity >= 0) && (disparity <= MAX_DISPARITY);
                bool within_vertical = std::abs(secondary_edge.y - primary_edge.y) <= MAX_DISPARITY;

                if (within_horizontal && within_vertical)
                {
                    filtered_secondary_edge_coords.push_back(secondary_edge);
                    filtered_secondary_edge_orientations.push_back(secondary_candidate_orientations[j]);
                }
            }

            // disp_input_counts.push_back(secondary_candidate_edges.size());
            local_disp_input_counts[thread_id].push_back(secondary_candidate_edges.size());

#if MEASURE_TIMINGS
            time_disp_edges_evaluated++;
            auto end_disp = std::chrono::high_resolution_clock::now();
            time_disp += std::chrono::duration<double, std::milli>(end_disp - start_disp).count();
#endif
            //> MARK: Maximum Disparity
            ///////////////////////////////MAXIMUM DISPARITY THRESHOLD RECALL//////////////////////////
            if (!selected_ground_truth_edges.empty())
            {
                int disp_precision_numerator = 0;
                bool disp_match_found = false;

                for (const auto &filtered_candidate : filtered_secondary_edge_coords)
                {
                    if (cv::norm(filtered_candidate - ground_truth_edge) <= 0.5)
                    {
                        disp_precision_numerator++;
                        disp_match_found = true;
                        // break;
                    }
                }

                if (disp_match_found)
                {
                    disp_true_positive++;
                }
                else
                {
                    disp_false_negative++;
                }
                if (!filtered_secondary_edge_coords.empty())
                {
                    per_edge_disp_precision += static_cast<double>(disp_precision_numerator) / filtered_secondary_edge_coords.size();
                    disp_edges_evaluated++;
                }
            }
            ///////////////////////////////EPIPOLAR SHIFT THRESHOLD//////////////////////////
#if MEASURE_TIMINGS
            auto start_shift = std::chrono::high_resolution_clock::now();
#endif

            // disp_output_counts.push_back(filtered_secondary_edge_coords.size());
            local_disp_output_counts[thread_id].push_back(filtered_secondary_edge_coords.size());

            std::vector<cv::Point2d> shifted_secondary_edge_coords;
            std::vector<double> shifted_secondary_edge_orientations;
            std::vector<double> epipolar_coefficients = {a, b, c};

            for (size_t j = 0; j < filtered_secondary_edge_coords.size(); j++)
            {
                bool secondary_passes_tangency = false;

                cv::Point2d shifted_edge = PerformEpipolarShift(filtered_secondary_edge_coords[j], filtered_secondary_edge_orientations[j], epipolar_coefficients, secondary_passes_tangency);
                if (secondary_passes_tangency)
                {
                    shifted_secondary_edge_coords.push_back(shifted_edge);
                    shifted_secondary_edge_orientations.push_back(filtered_secondary_edge_orientations[j]);
                }
            }

            // shift_input_counts.push_back(filtered_secondary_edge_coords.size());
            local_shift_input_counts[thread_id].push_back(filtered_secondary_edge_coords.size());

#if MEASURE_TIMINGS
            time_shift_edges_evaluated++;
            auto end_shift = std::chrono::high_resolution_clock::now();
            time_shift += std::chrono::duration<double, std::milli>(end_shift - start_shift).count();
#endif
            //> MARK: Epipolar Shift
            ///////////////////////////////EPIPOLAR SHIFT THRESHOLD RECALL//////////////////////////
            if (!selected_ground_truth_edges.empty())
            {
                int shift_precision_numerator = 0;
                bool shift_match_found = false;

                for (const auto &shifted_candidate : shifted_secondary_edge_coords)
                {
                    if (cv::norm(shifted_candidate - ground_truth_edge) <= 3.0)
                    {
                        shift_precision_numerator++;
                        shift_match_found = true;
                        // break;
                    }
                }

                if (shift_match_found)
                {
                    shift_true_positive++;
                }
                else
                {
                    shift_false_negative++;
                }
                if (!shifted_secondary_edge_coords.empty())
                {
                    per_edge_shift_precision += static_cast<double>(shift_precision_numerator) / shifted_secondary_edge_coords.size();
                    shift_edges_evaluated++;
                }
            }
            ///////////////////////////////EPIPOLAR CLUSTER THRESHOLD//////////////////////////
#if MEASURE_TIMINGS
            auto start_cluster = std::chrono::high_resolution_clock::now();
#endif

            // shift_output_counts.push_back(shifted_secondary_edge_coords.size());
            local_shift_output_counts[thread_id].push_back(shifted_secondary_edge_coords.size());

            std::vector<std::pair<std::vector<cv::Point2d>, std::vector<double>>> clusters = ClusterEpipolarShiftedEdges(shifted_secondary_edge_coords, shifted_secondary_edge_orientations);
            std::vector<EdgeCluster> cluster_centers;

            for (size_t j = 0; j < clusters.size(); j++)
            {
                const auto &cluster_edges = clusters[j].first;
                const auto &cluster_orientations = clusters[j].second;

                if (cluster_edges.empty())
                    continue;

                cv::Point2d sum_point(0.0, 0.0);
                double sum_orientation = 0.0;

                for (size_t j = 0; j < cluster_edges.size(); ++j)
                {
                    sum_point += cluster_edges[j];
                }

                for (size_t j = 0; j < cluster_orientations.size(); ++j)
                {
                    sum_orientation += cluster_orientations[j];
                }

                cv::Point2d avg_point = sum_point * (1.0 / cluster_edges.size());
                double avg_orientation = sum_orientation * (1.0 / cluster_orientations.size());

                EdgeCluster cluster;
                cluster.center_coord = avg_point;
                cluster.center_orientation = avg_orientation;
                cluster.contributing_edges = cluster_edges;
                cluster.contributing_orientations = cluster_orientations;

                cluster_centers.push_back(cluster);
            }

            // clust_input_counts.push_back(shifted_secondary_edge_coords.size());
            local_clust_input_counts[thread_id].push_back(shifted_secondary_edge_coords.size());

#if MEASURE_TIMINGS
            time_clust_edges_evaluated++;
            auto end_cluster = std::chrono::high_resolution_clock::now();
            time_cluster += std::chrono::duration<double, std::milli>(end_cluster - start_cluster).count();
#endif
            //> MARK: Clustering
            ///////////////////////////////EPIPOLAR CLUSTER THRESHOLD RECALL//////////////////////////
            if (!selected_ground_truth_edges.empty())
            {
                int clust_precision_numerator = 0;
                bool cluster_match_found = false;

                for (const auto &cluster : cluster_centers)
                {
                    if (cv::norm(cluster.center_coord - ground_truth_edge) <= 3.0)
                    {
                        clust_precision_numerator++;
                        cluster_match_found = true;
                        // break;
                    }
                }

                if (cluster_match_found)
                {
                    cluster_true_positive++;
                }
                else
                {
                    cluster_false_negative++;
                }
                if (!cluster_centers.empty())
                {
                    per_edge_clust_precision += static_cast<double>(clust_precision_numerator) / cluster_centers.size();
                    clust_edges_evaluated++;
                }
            }
            ///////////////////////////////EXTRACT PATCHES THRESHOLD////////////////////////////////////////////
#if MEASURE_TIMINGS
            auto start_patch = std::chrono::high_resolution_clock::now();
#endif

            // clust_output_counts.push_back(cluster_centers.size());
            local_clust_output_counts[thread_id].push_back(cluster_centers.size());

            std::vector<cv::Point2d> cluster_coords;
            std::vector<double> cluster_orientations;

            for (const auto &cluster : cluster_centers)
            {
                cluster_coords.push_back(cluster.center_coord);
                cluster_orientations.push_back(cluster.center_orientation);
            }

            auto [secondary_orthogonal_one, secondary_orthogonal_two] = CalculateOrthogonalShifts(
                cluster_coords,
                cluster_orientations,
                ORTHOGONAL_SHIFT_MAG);

            std::vector<EdgeCluster> filtered_cluster_centers;
            std::vector<cv::Mat> secondary_patch_set_one;
            std::vector<cv::Mat> secondary_patch_set_two;

            ExtractClusterPatches(
                PATCH_SIZE,
                secondary_image,
                cluster_centers,
                nullptr,
                secondary_orthogonal_one,
                secondary_orthogonal_two,
                filtered_cluster_centers,
                nullptr,
                secondary_patch_set_one,
                secondary_patch_set_two);

            // patch_input_counts.push_back(cluster_centers.size());
            local_patch_input_counts[thread_id].push_back(cluster_centers.size());

#if MEASURE_TIMINGS
            time_patch_edges_evaluated++;
            auto end_patch = std::chrono::high_resolution_clock::now();
            time_patch += std::chrono::duration<double, std::milli>(end_patch - start_patch).count();
            //> MARK: NCC
            ///////////////////////////////NCC THRESHOLD/////////////////////////////////////////////////////
            auto start_ncc = std::chrono::high_resolution_clock::now();
#endif

            //    patch_output_counts.push_back(filtered_cluster_centers.size());
            local_patch_output_counts[thread_id].push_back(filtered_cluster_centers.size());

            int ncc_precision_numerator = 0;

            bool ncc_match_found = false;
            std::vector<EdgeMatch> passed_ncc_matches;

            if (!primary_patch_one.empty() && !primary_patch_two.empty() &&
                !secondary_patch_set_one.empty() && !secondary_patch_set_two.empty())
            {

                for (size_t i = 0; i < filtered_cluster_centers.size(); ++i)
                {
                    double ncc_one = ComputeNCC(primary_patch_one, secondary_patch_set_one[i]);
                    double ncc_two = ComputeNCC(primary_patch_two, secondary_patch_set_two[i]);
                    double ncc_three = ComputeNCC(primary_patch_one, secondary_patch_set_two[i]);
                    double ncc_four = ComputeNCC(primary_patch_two, secondary_patch_set_one[i]);

                    double score_one = std::min(ncc_one, ncc_two);
                    double score_two = std::min(ncc_three, ncc_four);
                    double final_score = std::max(score_one, score_two);

#if DEBUG_COLLECT_NCC_AND_ERR
                    double err_to_gt = cv::norm(filtered_cluster_centers[i].center_coord - ground_truth_edge);
                    std::pair<double, double> pair_ncc_one_err(err_to_gt, ncc_one);
                    std::pair<double, double> pair_ncc_two_err(err_to_gt, ncc_two);
                    ncc_one_vs_err.push_back(pair_ncc_one_err);
                    ncc_two_vs_err.push_back(pair_ncc_two_err);
#endif
                    if (ncc_one >= NCC_THRESH_STRONG_BOTH_SIDES && ncc_two >= NCC_THRESH_STRONG_BOTH_SIDES)
                    {
                        EdgeMatch info;
                        info.coord = filtered_cluster_centers[i].center_coord;
                        info.orientation = filtered_cluster_centers[i].center_orientation;
                        info.final_score = final_score;
                        info.contributing_edges = filtered_cluster_centers[i].contributing_edges;
                        info.contributing_orientations = filtered_cluster_centers[i].contributing_orientations;
                        passed_ncc_matches.push_back(info);

                        if (!selected_ground_truth_edges.empty())
                        {
                            if (cv::norm(filtered_cluster_centers[i].center_coord - ground_truth_edge) <= 3.0)
                            {
                                ncc_match_found = true;
                                ncc_precision_numerator++;
                            }
                        }
                    }
                    else if (ncc_one >= NCC_THRESH_STRONG_ONE_SIDE || ncc_two >= NCC_THRESH_STRONG_ONE_SIDE)
                    {
                        EdgeMatch info;
                        info.coord = filtered_cluster_centers[i].center_coord;
                        info.orientation = filtered_cluster_centers[i].center_orientation;
                        info.final_score = final_score;
                        info.contributing_edges = filtered_cluster_centers[i].contributing_edges;
                        info.contributing_orientations = filtered_cluster_centers[i].contributing_orientations;
                        passed_ncc_matches.push_back(info);

                        if (!selected_ground_truth_edges.empty())
                        {
                            if (cv::norm(filtered_cluster_centers[i].center_coord - ground_truth_edge) <= 3.0)
                            {
                                ncc_match_found = true;
                                ncc_precision_numerator++;
                            }
                        }
                    }
                    else if (ncc_one >= NCC_THRESH_WEAK_BOTH_SIDES && ncc_two >= NCC_THRESH_WEAK_BOTH_SIDES && filtered_cluster_centers.size() == 1)
                    {
                        EdgeMatch info;
                        info.coord = filtered_cluster_centers[i].center_coord;
                        info.orientation = filtered_cluster_centers[i].center_orientation;
                        info.final_score = final_score;
                        info.contributing_edges = filtered_cluster_centers[i].contributing_edges;
                        info.contributing_orientations = filtered_cluster_centers[i].contributing_orientations;
                        passed_ncc_matches.push_back(info);

                        if (!selected_ground_truth_edges.empty())
                        {
                            if (cv::norm(filtered_cluster_centers[i].center_coord - ground_truth_edge) <= 3.0)
                            {
                                ncc_match_found = true;
                                ncc_precision_numerator++;
                            }
                        }
                    }
                }

                if (ncc_match_found)
                {
                    ncc_true_positive++;
                }
                else
                {
                    ncc_false_negative++;
                }
            }

            if (!passed_ncc_matches.empty())
            {
                per_edge_ncc_precision += static_cast<double>(ncc_precision_numerator) / passed_ncc_matches.size();
                ncc_edges_evaluated++;
            }
            //    ncc_input_counts.push_back(filtered_cluster_centers.size());
            //    ncc_output_counts.push_back(passed_ncc_matches.size());

            local_ncc_input_counts[thread_id].push_back(filtered_cluster_centers.size());
            local_ncc_output_counts[thread_id].push_back(passed_ncc_matches.size());

#if MEASURE_TIMINGS
            time_ncc_edges_evaluated++;
            auto end_ncc = std::chrono::high_resolution_clock::now();
            time_ncc += std::chrono::duration<double, std::milli>(end_ncc - start_ncc).count();
            //> MARK: Lowe's Ratio Test
            ///////////////////////////////LOWES RATIO TEST//////////////////////////////////////////////
            auto start_lowe = std::chrono::high_resolution_clock::now();
#endif
            // lowe_input_counts.push_back(passed_ncc_matches.size());
            local_lowe_input_counts[thread_id].push_back(passed_ncc_matches.size());

            int lowe_precision_numerator = 0;

            EdgeMatch best_match;
            double best_score = -1;

            if (passed_ncc_matches.size() >= 2)
            {
                EdgeMatch second_best_match;
                double second_best_score = -1;

                for (const auto &match : passed_ncc_matches)
                {
                    if (match.final_score > best_score)
                    {
                        second_best_score = best_score;
                        second_best_match = best_match;

                        best_score = match.final_score;
                        best_match = match;
                    }
                    else if (match.final_score > second_best_score)
                    {
                        second_best_score = match.final_score;
                        second_best_match = match;
                    }
                }
                double lowe_ratio = second_best_score / best_score;

                if (lowe_ratio < 1)
                {
                    if (!selected_ground_truth_edges.empty())
                    {
                        local_GT_right_edges_after_lowe[thread_id].push_back(ground_truth_edge);
                        if (cv::norm(best_match.coord - ground_truth_edge) <= 3.0)
                        {
                            lowe_precision_numerator++;
                            lowe_true_positive++;
                        }
                        else
                        {
                            lowe_false_negative++;
                        }
                    }
                    SourceEdge source_edge{primary_edge, primary_orientation};
                    // final_matches.emplace_back(source_edge, best_match);
                    local_final_matches[thread_id].emplace_back(source_edge, best_match);
                    // lowe_output_counts.push_back(1);
                    local_lowe_output_counts[thread_id].push_back(1);
                }
                else
                {
                    lowe_false_negative++;
                    // lowe_output_counts.push_back(0);
                    local_lowe_output_counts[thread_id].push_back(0);
                }
            }
            else if (passed_ncc_matches.size() == 1)
            {
                best_match = passed_ncc_matches[0];

                if (!selected_ground_truth_edges.empty())
                {
                    local_GT_right_edges_after_lowe[thread_id].push_back(ground_truth_edge);
                    if (cv::norm(best_match.coord - ground_truth_edge) <= 3.0)
                    {
                        lowe_precision_numerator++;
                        lowe_true_positive++;
                    }
                    else
                    {
                        lowe_false_negative++;
                    }
                }

                SourceEdge source_edge{primary_edge, primary_orientation};
                // final_matches.emplace_back(source_edge, best_match);
                local_final_matches[thread_id].emplace_back(source_edge, best_match);
                // lowe_output_counts.push_back(1);
                local_lowe_output_counts[thread_id].push_back(1);
            }
            else
            {
                lowe_false_negative++;
                // lowe_output_counts.push_back(0);
                local_lowe_output_counts[thread_id].push_back(0);
            }
            per_edge_lowe_precision += (static_cast<double>(lowe_precision_numerator) > 0) ? 1.0 : 0.0;

            if (!passed_ncc_matches.empty())
            {
                lowe_edges_evaluated++;
            }
#if MEASURE_TIMINGS
            time_lowe_edges_evaluated++;
            auto end_lowe = std::chrono::high_resolution_clock::now();
            time_lowe += std::chrono::duration<double, std::milli>(end_lowe - start_lowe).count();
#endif
        } //> MARK: end of looping over left edges
    }

#if MEASURE_TIMINGS
    auto total_end = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
#endif

    double epi_distance_recall = 0.0;
    if ((epi_true_positive + epi_false_negative) > 0)
    {
        epi_distance_recall = static_cast<double>(epi_true_positive) / (epi_true_positive + epi_false_negative);
    }

    double max_disparity_recall = 0.0;
    if ((disp_true_positive + disp_false_negative) > 0)
    {
        max_disparity_recall = static_cast<double>(disp_true_positive) / (disp_true_positive + disp_false_negative);
    }

    double epi_shift_recall = 0.0;
    if ((shift_true_positive + shift_false_negative) > 0)
    {
        epi_shift_recall = static_cast<double>(shift_true_positive) / (shift_true_positive + shift_false_negative);
    }

    double epi_cluster_recall = 0.0;
    if ((cluster_true_positive + cluster_false_negative) > 0)
    {
        epi_cluster_recall = static_cast<double>(cluster_true_positive) / (cluster_true_positive + cluster_false_negative);
    }

    double ncc_recall = 0.0;
    if ((ncc_true_positive + ncc_false_negative) > 0)
    {
        ncc_recall = static_cast<double>(ncc_true_positive) / (ncc_true_positive + ncc_false_negative);
    }

    double lowe_recall = 0.0;
    if ((lowe_true_positive + lowe_false_negative) > 0)
    {
        lowe_recall = static_cast<double>(lowe_true_positive) / (lowe_true_positive + lowe_false_negative);
    }

    // std::cout << "Epipolar Distance Recall: " << std::fixed << std::setprecision(2) << epi_distance_recall * 100 << "%" << std::endl;
    // std::cout << "Max Disparity Threshold Recall: " << std::fixed << std::setprecision(2) << max_disparity_recall * 100 << "%" << std::endl;
    // std::cout << "Epipolar Shift Threshold Recall: " << std::fixed << std::setprecision(2) << epi_shift_recall * 100 << "%" << std::endl;
    // std::cout << "Epipolar Cluster Threshold Recall: " << std::fixed << std::setprecision(2) << epi_cluster_recall * 100 << "%" << std::endl;
    // std::cout << "NCC Threshold Recall: " << std::fixed << std::setprecision(2) << ncc_recall * 100 << "%" << std::endl;
    // std::cout << "LRT Threshold Recall: " << std::fixed << std::setprecision(2) << lowe_recall * 100 << "%" << std::endl;

    double per_image_epi_precision = (epi_edges_evaluated > 0) ? (per_edge_epi_precision / epi_edges_evaluated) : (0.0);
    double per_image_disp_precision = (disp_edges_evaluated > 0) ? (per_edge_disp_precision / disp_edges_evaluated) : (0.0);
    double per_image_shift_precision = (shift_edges_evaluated > 0) ? (per_edge_shift_precision / shift_edges_evaluated) : (0.0);
    double per_image_clust_precision = (clust_edges_evaluated > 0) ? (per_edge_clust_precision / clust_edges_evaluated) : (0.0);
    double per_image_ncc_precision = (ncc_edges_evaluated > 0) ? (per_edge_ncc_precision / ncc_edges_evaluated) : (0.0);
    double per_image_lowe_precision = (lowe_edges_evaluated > 0) ? (per_edge_lowe_precision / lowe_edges_evaluated) : (0.0);

    // std::cout << "Epipolar Distance Precision: "
    //     << std::fixed << std::setprecision(2)
    //     << per_image_epi_precision * 100 << "%" << std::endl;
    // std::cout << "Maximum Disparity Precision: "
    //     << std::fixed << std::setprecision(2)
    //     << per_image_disp_precision * 100 << "%" << std::endl;
    // std::cout << "Epipolar Shift Precision: "
    //     << std::fixed << std::setprecision(2)
    //     << per_image_shift_precision * 100 << "%" << std::endl;
    // std::cout << "Epipolar Cluster Precision: "
    //     << std::fixed << std::setprecision(2)
    //     << per_image_clust_precision * 100 << "%" << std::endl;
    // std::cout << "NCC Precision: "
    //     << std::fixed << std::setprecision(2)
    //     << per_image_ncc_precision * 100 << "%" << std::endl;
    // std::cout << "LRT Precision: "
    //     << std::fixed << std::setprecision(2)
    //     << per_image_lowe_precision * 100 << "%" << std::endl;

    double per_image_epi_time = (time_epi_edges_evaluated > 0) ? (time_epi / time_epi_edges_evaluated) : (0.0);
    double per_image_disp_time = (time_disp_edges_evaluated > 0) ? (time_disp / time_disp_edges_evaluated) : 0.0;
    double per_image_shift_time = (time_shift_edges_evaluated > 0) ? (time_shift / time_shift_edges_evaluated) : 0.0;
    double per_image_clust_time = (time_clust_edges_evaluated > 0) ? (time_cluster / time_clust_edges_evaluated) : 0.0;
    double per_image_patch_time = (time_patch_edges_evaluated > 0) ? (time_patch / time_patch_edges_evaluated) : 0.0;
    double per_image_ncc_time = (time_ncc_edges_evaluated > 0) ? (time_ncc / time_ncc_edges_evaluated) : 0.0;
    double per_image_lowe_time = (time_lowe_edges_evaluated > 0) ? (time_lowe / time_lowe_edges_evaluated) : 0.0;
    double per_image_total_time = (selected_primary_edges.size() > 0) ? (total_time / selected_primary_edges.size()) : 0.0;

    //> CH: stack all local_final_matches to a global final_matches
    for (const auto &local_matches : local_final_matches)
    {
        final_matches.insert(final_matches.end(), local_matches.begin(), local_matches.end());
    }

    for (const auto &local_counts : local_epi_input_counts)
        epi_input_counts.insert(epi_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_epi_output_counts)
        epi_output_counts.insert(epi_output_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_disp_input_counts)
        disp_input_counts.insert(disp_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_disp_output_counts)
        disp_output_counts.insert(disp_output_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_shift_input_counts)
        shift_input_counts.insert(shift_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_shift_output_counts)
        shift_output_counts.insert(shift_output_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_clust_input_counts)
        clust_input_counts.insert(clust_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_clust_output_counts)
        clust_output_counts.insert(clust_output_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_patch_input_counts)
        patch_input_counts.insert(patch_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_patch_output_counts)
        patch_output_counts.insert(patch_output_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_ncc_input_counts)
        ncc_input_counts.insert(ncc_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_ncc_output_counts)
        ncc_output_counts.insert(ncc_output_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_lowe_input_counts)
        lowe_input_counts.insert(lowe_input_counts.end(), local_counts.begin(), local_counts.end());
    for (const auto &local_counts : local_lowe_output_counts)
        lowe_output_counts.insert(lowe_output_counts.end(), local_counts.begin(), local_counts.end());

    for (const auto &local_GT_right_edges_stack : local_GT_right_edges_after_lowe)
    {
        ground_truth_right_edges_after_lowe.insert(ground_truth_right_edges_after_lowe.end(), local_GT_right_edges_stack.begin(), local_GT_right_edges_stack.end());
    }

    return EdgeMatchResult{
        RecallMetrics{
            epi_distance_recall,
            max_disparity_recall,
            epi_shift_recall,
            epi_cluster_recall,
            ncc_recall,
            lowe_recall,
            epi_input_counts,
            epi_output_counts,
            disp_input_counts,
            disp_output_counts,
            shift_input_counts,
            shift_output_counts,
            clust_input_counts,
            clust_output_counts,
            patch_input_counts,
            patch_output_counts,
            ncc_input_counts,
            ncc_output_counts,
            lowe_input_counts,
            lowe_output_counts,
            per_image_epi_precision,
            per_image_disp_precision,
            per_image_shift_precision,
            per_image_clust_precision,
            per_image_ncc_precision,
            per_image_lowe_precision,
            lowe_true_positive,
            lowe_false_negative,
            per_image_epi_time,
            per_image_disp_time,
            per_image_shift_time,
            per_image_clust_time,
            per_image_patch_time,
            per_image_ncc_time,
            per_image_lowe_time,
            per_image_total_time},
        final_matches};
}

/*
    Extract patches from the image based on shifted edge points and orientations.
    The function checks if the patches are within bounds before extracting them.
    It also filters edges and orientations based on the extracted patches.
*/
void ExtractPatches(
    int patch_size,
    const cv::Mat &image,
    const std::vector<cv::Point2d> &edges,
    const std::vector<double> &orientations,
    const std::vector<cv::Point2d> &shifted_one,
    const std::vector<cv::Point2d> &shifted_two,
    std::vector<cv::Point2d> &filtered_edges_out,
    std::vector<double> &filtered_orientations_out,
    std::vector<cv::Mat> &patch_set_one_out,
    std::vector<cv::Mat> &patch_set_two_out,
    const std::vector<cv::Point2d> *ground_truth_edges,
    std::vector<cv::Point2d> *filtered_gt_edges_out)
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

            filtered_edges_out.push_back(edges[i]);
            filtered_orientations_out.push_back(orientations[i]);
            patch_set_one_out.push_back(patch1);
            patch_set_two_out.push_back(patch2);

            if (ground_truth_edges && filtered_gt_edges_out)
            {
                filtered_gt_edges_out->push_back((*ground_truth_edges)[i]);
            }
        }
    }
}
