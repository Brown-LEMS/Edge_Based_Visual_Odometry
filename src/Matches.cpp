#include "Matches.h"
#include <iomanip>

/*
    calculate the epipolar line for each edge point using the fundamental matrix.
*/

std::vector<Eigen::Vector3d> CalculateEpipolarLine(const Eigen::Matrix3d &fund_mat, const std::vector<Edge> &edges)
{
    std::vector<Eigen::Vector3d> epipolar_lines;

    for (const Edge &edge : edges)
    {
        Eigen::Vector3d homo_point(edge.location.x, edge.location.y, 1.0);

        Eigen::Vector3d epipolar_line = fund_mat * homo_point;

        epipolar_lines.push_back(epipolar_line);
    }

    return epipolar_lines;
}

double getNormalDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &epiline_x, double &epiline_y) {
	double a1_line = Epip_Line_Coeffs(0);
	double b1_line = Epip_Line_Coeffs(1);
	double c1_line = Epip_Line_Coeffs(2);

	epiline_x = edge(0) - a1_line * (a1_line * edge(0) + b1_line * edge(1) + c1_line)/(pow(a1_line, 2) + pow(b1_line, 2));
	epiline_y = edge(1) - b1_line * (a1_line * edge(0) + b1_line * edge(1) + c1_line)/(pow(a1_line, 2) + pow(b1_line, 2));

	return sqrt(pow(edge(0) - epiline_x, 2) + pow(edge(1) - epiline_y, 2));
}

double getTangentialDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &x_intersection, double &y_intersection) {
	double a_edgeH2 = tan(edge(2)); //tan(theta2)
	double b_edgeH2 = -1;
	double c_edgeH2 = -(a_edgeH2 * edge(0) - edge(1)); //−(a⋅x2−y2)

	double a1_line = Epip_Line_Coeffs(0);
	double b1_line = Epip_Line_Coeffs(1);
	double c1_line = Epip_Line_Coeffs(2);

	x_intersection = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
	y_intersection = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);

	return sqrt((x_intersection - edge(0)) * (x_intersection - edge(0)) + (y_intersection - edge(1)) * (y_intersection - edge(1)));
}

//> MARK: Perform Epipolar Shift
//> Check that const notation works as intended, otherwise remove
std::vector<Eigen::Vector3d> PerformEpipolarShift(
    const Eigen::Vector3d& edge1,
    const Eigen::MatrixXd& edge2,
    const Eigen::Vector3d& epip_coeffs)
{
    std::vector<Eigen::Vector3d> shifted_edges;

        for (int i = 0; i < edge2.rows(); i++)
        {
            Eigen::Vector3d xy1_H2(edge2(i, 0), edge2(i, 1), 1.0);

            double corrected_x, corrected_y, corrected_theta;
            double epiline_x, epiline_y;

            double normal_distance_epiline = getNormalDistance2EpipolarLine(epip_coeffs, xy1_H2, epiline_x, epiline_y);

            if (normal_distance_epiline < LOCATION_PERTURBATION)
            {
                corrected_x = epiline_x;
                corrected_y = epiline_y;
                corrected_theta = edge2(i, 2);
            }
            else
            {
                double x_intersection, y_intersection;
                Eigen::Vector3d isolated_H2(edge2(i, 0), edge2(i, 1), edge2(i, 2));
                double dist_diff_edg2 = getTangentialDistance2EpipolarLine(epip_coeffs, isolated_H2, x_intersection, y_intersection);

                double theta = edge2(i, 2);

                if (dist_diff_edg2 < EPIP_TANGENCY_DISPL_THRESH)
                {
                    corrected_x = x_intersection;
                    corrected_y = y_intersection;
                    corrected_theta = theta;
                }
                else
                {
                    double p_theta = epip_coeffs(0) * cos(theta) + epip_coeffs(1) * sin(theta);
                    double derivative_p_theta = -epip_coeffs(0) * sin(theta) + epip_coeffs(1) * cos(theta);

                    if (p_theta > 0 && derivative_p_theta < 0) theta -= ORIENT_PERTURBATION;
                    else if (p_theta < 0 && derivative_p_theta < 0) theta -= ORIENT_PERTURBATION;
                    else if (p_theta > 0 && derivative_p_theta > 0) theta += ORIENT_PERTURBATION;
                    else if (p_theta < 0 && derivative_p_theta > 0) theta += ORIENT_PERTURBATION;

                    Eigen::Vector3d isolated_H2_(edge2(i, 0), edge2(i, 1), theta);
                    dist_diff_edg2 = getTangentialDistance2EpipolarLine(epip_coeffs, isolated_H2_, x_intersection, y_intersection);

                    if (dist_diff_edg2 < EPIP_TANGENCY_DISPL_THRESH)
                    {
                        corrected_x = x_intersection;
                        corrected_y = y_intersection;
                        corrected_theta = theta;
                    }
                    else
                    {
                        continue;
                    }
                }
            }

            shifted_edges.emplace_back(corrected_x, corrected_y, corrected_theta);
        }

    return shifted_edges;
}

/*
    Extract edges that are close to the epipolar line within a specified distance threshold.
    Returns a pair of vectors: one for the extracted edge locations and one for their orientations.
*/
std::vector<Edge> ExtractEpipolarEdges(const Eigen::Vector3d &epipolar_line, const std::vector<Edge> &edges, double distance_threshold)
{
    std::vector<Edge> extracted_edges;

    // if (edges.size() != edge_orientations.size())
    // {
    //     throw std::runtime_error("Edge locations and orientations size mismatch.");
    // }

    for (size_t i = 0; i < edges.size(); ++i)
    {
        const auto &edge = edges[i];
        double x = edge.location.x;
        double y = edge.location.y;

        double distance = std::abs(epipolar_line(0) * x + epipolar_line(1) * y + epipolar_line(2)) / std::sqrt((epipolar_line(0) * epipolar_line(0)) + (epipolar_line(1) * epipolar_line(1)));

        if (distance < distance_threshold)
        {
            extracted_edges.push_back(edge);
        }
    }

    return extracted_edges;
}

/*
    Cluster the shifted edges based on their proximity and orientation.
    Returns a vector of clusters, where each cluster contains a pair of vectors:
    one for the edge points and one for their corresponding orientations.
    The clustering is based on a distance threshold and an orientation difference threshold.
*/
std::vector<std::vector<Edge>> ClusterEpipolarShiftedEdges(std::vector<Edge> &valid_shifted_edges)
{
    std::vector<std::vector<Edge>> clusters;

    if (valid_shifted_edges.empty())
    {
        return clusters;
    }

    std::sort(valid_shifted_edges.begin(), valid_shifted_edges.end(),
              [](const Edge &a, const Edge &b)
              {
                  return a.location.x < b.location.x;
              });

    std::vector<Edge> current_cluster;
    current_cluster.push_back(valid_shifted_edges[0]);

    for (size_t i = 1; i < valid_shifted_edges.size(); ++i)
    {
        double distance = cv::norm(valid_shifted_edges[i].location - valid_shifted_edges[i - 1].location);
        double orientation_difference = std::abs(valid_shifted_edges[i].orientation - valid_shifted_edges[i - 1].orientation);

        if (distance <= EDGE_CLUSTER_THRESH && orientation_difference < 5.0)
        {
            current_cluster.push_back(valid_shifted_edges[i]);
        }
        else
        {
            clusters.emplace_back(current_cluster);
            current_cluster.clear();
            current_cluster.push_back(valid_shifted_edges[i]);
        }
    }

    if (!current_cluster.empty())
    {
        clusters.emplace_back(current_cluster);
    }

    return clusters;
}

bool is_patch_in_bounds(const cv::Point2d& point, int image_width, int image_height) {
    return (point.x >= HALF_PATCH_SIZE && point.x < image_width - HALF_PATCH_SIZE &&
            point.y >= HALF_PATCH_SIZE && point.y < image_height - HALF_PATCH_SIZE);
}

std::pair<cv::Point2d, cv::Point2d> get_Orthogonal_Shifted_Points(const Edge edgel)
{
    double shifted_x1 = edgel.location.x + ORTHOGONAL_SHIFT_MAG * (std::sin(edgel.orientation));
    double shifted_y1 = edgel.location.y + ORTHOGONAL_SHIFT_MAG * (-std::cos(edgel.orientation));
    double shifted_x2 = edgel.location.x + ORTHOGONAL_SHIFT_MAG * (-std::sin(edgel.orientation));
    double shifted_y2 = edgel.location.y + ORTHOGONAL_SHIFT_MAG * (std::cos(edgel.orientation));

    cv::Point2d shifted_point_plus(shifted_x1, shifted_y1);
    cv::Point2d shifted_point_minus(shifted_x2, shifted_y2);

    return {shifted_point_plus, shifted_point_minus};
}

void get_patch_on_one_edge_side(cv::Point2d shifted_point, double theta, cv::Mat &patch_coord_x, cv::Mat &patch_coord_y, cv::Mat &patch_val, const cv::Mat img) 
{
    CV_Assert(img.type() == CV_64F);

    for (int i = -HALF_PATCH_SIZE; i <= HALF_PATCH_SIZE; i++) {
        for (int j = -HALF_PATCH_SIZE; j <= HALF_PATCH_SIZE; j++) {

            cv::Point2d rotated_point(cos(theta) * (i) - sin(theta) * (j) + shifted_point.x, sin(theta) * (i) + cos(theta) * (j) + shifted_point.y);
            patch_coord_x.at<double>(i + HALF_PATCH_SIZE, j + HALF_PATCH_SIZE) = rotated_point.x;
            patch_coord_y.at<double>(i + HALF_PATCH_SIZE, j + HALF_PATCH_SIZE) = rotated_point.y;

            double interp_val = Bilinear_Interpolation<double>(img, rotated_point);
            patch_val.at<double>(i + HALF_PATCH_SIZE, j + HALF_PATCH_SIZE) = interp_val;
        }
    }
}

StereoMatchResult DisplayMatches(const cv::Mat &left_image, const cv::Mat &right_image, Dataset &dataset)
{
    cv::Mat left_image_64f, right_image_64f;
    if (left_image.type() != CV_64F)
        left_image.convertTo(left_image_64f, CV_64F);
    else
        left_image_64f = left_image;
    
    if (right_image.type() != CV_64F)
        right_image.convertTo(right_image_64f, CV_64F);
    else
        right_image_64f = right_image;

    size_t image_pair_index = 0;
    ETH3DIterator* eth3d_iter = dynamic_cast<ETH3DIterator*>(dataset.stereo_iterator.get());
    if (eth3d_iter) {
        image_pair_index = eth3d_iter->getCurrentIndex();
    }

    ///////////////////////////////FORWARD DIRECTION///////////////////////////////
    std::vector<Edge> left_edges;
    std::vector<cv::Point2d> ground_truth_right_edges;

    for (const auto &data : dataset.forward_gt_data)
    {
        left_edges.push_back(Edge{std::get<0>(data), std::get<2>(data), false});
        ground_truth_right_edges.push_back(std::get<1>(data));
    }

    std::vector<Edge> filtered_left_edges;
    std::vector<cv::Point2d> filtered_ground_truth_right_edges;

    std::vector<cv::Mat> left_patch_set_one;
    std::vector<cv::Mat> left_patch_set_two;

    int left_img_width  = left_image_64f.cols;
    int left_img_height = left_image_64f.rows;
    
    for (size_t i = 0; i < left_edges.size(); ++i) {
        const Edge &edge = left_edges[i];

        auto [shifted_plus, shifted_minus] = get_Orthogonal_Shifted_Points(edge);

        if (!is_patch_in_bounds(shifted_plus, left_img_width, left_img_height) ||
            !is_patch_in_bounds(shifted_minus, left_img_width, left_img_height)) {
            continue;
        }

        cv::Mat patch_coord_x_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_coord_y_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_plus          = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    
        cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_minus         = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    
        get_patch_on_one_edge_side(
            shifted_plus,
            edge.orientation,
            patch_coord_x_plus,
            patch_coord_y_plus,
            patch_plus,
            left_image_64f
        );
    
        get_patch_on_one_edge_side(
            shifted_minus,
            edge.orientation,
            patch_coord_x_minus,
            patch_coord_y_minus,
            patch_minus,
            left_image_64f
        );
    
        cv::Mat patch_plus_32f, patch_minus_32f;
        if (patch_plus.type() != CV_32F) {
            patch_plus.convertTo(patch_plus_32f, CV_32F);
        }
        else {
            patch_plus_32f = patch_plus;
        }
        
        if (patch_minus.type() != CV_32F) {
            patch_minus.convertTo(patch_minus_32f, CV_32F);
        }
        else {
            patch_minus_32f = patch_minus;
        }

        filtered_left_edges.push_back(edge);
        if (!ground_truth_right_edges.empty()) {
            filtered_ground_truth_right_edges.push_back(ground_truth_right_edges[i]);
        }

        left_patch_set_one.push_back(patch_plus_32f);
        left_patch_set_two.push_back(patch_minus_32f);
    }

    Eigen::Matrix3d fundamental_matrix_21 = dataset.get_fund_mat_21();
    Eigen::Matrix3d fundamental_matrix_12 = dataset.get_fund_mat_12();

    std::vector<Eigen::Vector3d> epipolar_lines_right = CalculateEpipolarLine(fundamental_matrix_21, filtered_left_edges);

    EdgeMatchResult forward_match = CalculateMatches(
        filtered_left_edges,
        dataset.right_edges,
        left_patch_set_one,
        left_patch_set_two,
        epipolar_lines_right,
        right_image_64f,
        dataset,
        filtered_ground_truth_right_edges,
        image_pair_index,
        true
    );

    ///////////////////////////////REVERSE DIRECTION///////////////////////////////
    std::vector<Edge> reverse_primary_edges;

    for (const auto &match_pair : forward_match.edge_to_cluster_matches)
    {
        const EdgeMatch &match_info = match_pair.second;

        for (const auto &edge : match_info.contributing_edges)
        {
            reverse_primary_edges.push_back(edge);
        }
    }

    std::vector<Edge> filtered_right_edges;

    std::vector<cv::Mat> right_patch_set_one;
    std::vector<cv::Mat> right_patch_set_two;

    int right_img_width  = right_image_64f.cols;
    int right_img_height = right_image_64f.rows;

    for (size_t i = 0; i < reverse_primary_edges.size(); ++i) {
        const Edge& edge = reverse_primary_edges[i];

        auto [shifted_plus, shifted_minus] = get_Orthogonal_Shifted_Points(edge);

        if (!is_patch_in_bounds(shifted_plus, right_img_width, right_img_height) ||
            !is_patch_in_bounds(shifted_minus, right_img_width, right_img_height)) {
            continue;
        }

        cv::Mat patch_coord_x_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_coord_y_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_plus          = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    
        cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_minus         = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

        get_patch_on_one_edge_side(
            shifted_plus,
            edge.orientation,
            patch_coord_x_plus,
            patch_coord_y_plus,
            patch_plus,
            right_image_64f
        );

        get_patch_on_one_edge_side(
            shifted_minus,
            edge.orientation,
            patch_coord_x_minus,
            patch_coord_y_minus,
            patch_minus,
            right_image_64f
        );

        cv::Mat patch_plus_32f, patch_minus_32f;
        if (patch_plus.type() != CV_32F) {
            patch_plus.convertTo(patch_plus_32f, CV_32F);
        }
        else {
            patch_plus_32f = patch_plus;
        }

        if (patch_minus.type() != CV_32F) {
            patch_minus.convertTo(patch_minus_32f, CV_32F);
        }
        else {
            patch_minus_32f = patch_minus;
        }

        filtered_right_edges.push_back(edge);

        right_patch_set_one.push_back(patch_plus_32f);
        right_patch_set_two.push_back(patch_minus_32f);
    }

    std::vector<Eigen::Vector3d> epipolar_lines_left = CalculateEpipolarLine(fundamental_matrix_12, filtered_right_edges);

    EdgeMatchResult reverse_match = CalculateMatches(
        filtered_right_edges,
        left_edges,
        right_patch_set_one,
        right_patch_set_two,
        epipolar_lines_left,
        left_image_64f,
        dataset,
        std::vector<cv::Point2d>(),
        image_pair_index,
        false
    );

    std::vector<std::pair<Edge, Edge>> confirmed_matches;

    int matches_before_bct = static_cast<int>(forward_match.edge_to_cluster_matches.size());
    auto bct_start = std::chrono::high_resolution_clock::now();

    ///////////////////////////////BCT///////////////////////////////
    int forward_left_index = 0;
    int bct_true_positive = 0;
    for (const auto &[left_edge, patch_match_forward] : forward_match.edge_to_cluster_matches) 
    {
        const auto &right_contributing_edges = patch_match_forward.contributing_edges;
        bool break_flag = false;
        for (size_t i = 0; i < right_contributing_edges.size(); ++i)
        {
            break_flag = false;
            const Edge &right_edge = right_contributing_edges[i];

            for (const auto &[rev_right_edge, patch_match_rev] : reverse_match.edge_to_cluster_matches)
            {
                if (cv::norm(rev_right_edge.location - right_edge.location) <= MATCH_TOL)
                {
                    for (const auto &rev_contributing_left : patch_match_rev.contributing_edges)
                    {
                        if (cv::norm(rev_contributing_left.location - left_edge.location) <= MATCH_TOL)
                        {
                            confirmed_matches.emplace_back(left_edge, right_edge);

                            cv::Point2d GT_right_edge_location = dataset.ground_truth_right_edges_after_lowe[forward_left_index];
                            if (cv::norm(right_edge.location - GT_right_edge_location) <= MATCH_TOL)
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

    auto bct_end = std::chrono::high_resolution_clock::now();
    double total_time_bct = std::chrono::duration<double, std::milli>(bct_end - bct_start).count();

    double per_image_bct_time = (matches_before_bct > 0) ? total_time_bct / matches_before_bct : 0.0;

    int matches_after_bct = static_cast<int>(confirmed_matches.size());


    double per_image_bct_precision = (matches_after_bct > 0) ? bct_true_positive / (double)(matches_after_bct) : 0.0;

    int bct_denominator = forward_match.recall_metrics.lowe_true_positive + forward_match.recall_metrics.lowe_false_negative;
    double bct_recall = (bct_denominator > 0) ? bct_true_positive / (double)(bct_denominator) : 0.0;



    BidirectionalMetrics bidirectional_metrics;
    bidirectional_metrics.matches_before_bct = matches_before_bct;
    bidirectional_metrics.matches_after_bct = matches_after_bct;
    bidirectional_metrics.per_image_bct_recall = bct_recall;
    bidirectional_metrics.per_image_bct_precision = per_image_bct_precision;
    bidirectional_metrics.per_image_bct_time = per_image_bct_time;

    return StereoMatchResult{forward_match, reverse_match, confirmed_matches, bidirectional_metrics};
}

//> MARK: Main Edge Pairing
EdgeMatchResult CalculateMatches(const std::vector<Edge> &selected_primary_edges, const std::vector<Edge> &secondary_edges,
                                 const std::vector<cv::Mat> &primary_patch_set_one, const std::vector<cv::Mat> &primary_patch_set_two, const std::vector<Eigen::Vector3d> &epipolar_lines_secondary,
                                 const cv::Mat &secondary_image, Dataset &dataset, const std::vector<cv::Point2d> &selected_ground_truth_edges, int image_pair_index, bool forward_direction)
{
    auto total_start = std::chrono::high_resolution_clock::now();
    // bunch of counts
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
    // was  std::vector<std::pair<SourceEdge, EdgeMatch>> final_matches;
    std::vector<std::pair<Edge, EdgeMatch>> final_matches;

    //> CH: this is local structure of final matches
    // was std::vector<std::vector<std::pair<SourceEdge, EdgeMatch>>> local_final_matches(omp_get_max_threads());
    std::vector<std::vector<std::pair<Edge, EdgeMatch>>> local_final_matches(omp_get_max_threads());

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

    std::ofstream veridical_csv;
    std::ofstream nonveridical_csv;

    if (forward_direction) {
        std::filesystem::path ncc_dir = dataset.get_output_path();
        ncc_dir /= "ncc_stats";

        std::filesystem::create_directories(ncc_dir);

        std::filesystem::path veridical_path = ncc_dir / ("image_pair_" + std::to_string(image_pair_index) + "_veridical_edges.csv");
        std::filesystem::path nonveridical_path = ncc_dir / ("image_pair_" + std::to_string(image_pair_index) + "_nonveridical_edges.csv");

        veridical_csv.open(veridical_path.string());
        nonveridical_csv.open(nonveridical_path.string());

        if (!veridical_csv || !nonveridical_csv) {
            std::cerr << "WARNING: Failed to open CSV files for writing.\n" << std::endl;
        }

        veridical_csv << ",left_x,left_y,left_theta,"
                    << "right_x,right_y,right_theta,"
                    << "gt_right_x,gt_right_y,"
                    << "epipolar_a,epipolar_b,epipolar_c,"
                    << "ncc1,ncc2,ncc3,ncc4,score1,score2,final_score\n";

        nonveridical_csv << ",left_x,left_y,left_theta,"
                    << "right_x,right_y,right_theta,"
                    << "gt_right_x,gt_right_y,"
                    << "epipolar_a,epipolar_b,epipolar_c,"
                    << "ncc1,ncc2,ncc3,ncc4,score1,score2,final_score\n";
    }     

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        cv::Point2d ground_truth_edge;

        const int skip = 1;

//> Start looping over left edges
#pragma omp for schedule(static, dataset.get_omp_threads()) reduction(+ : epi_true_positive, epi_false_negative, epi_true_negative, disp_true_positive, disp_false_negative, shift_true_positive, shift_false_negative, cluster_true_positive, cluster_false_negative, cluster_true_negative, ncc_true_positive, ncc_false_negative, lowe_true_positive, lowe_false_negative, per_edge_epi_precision, per_edge_disp_precision, per_edge_shift_precision, per_edge_clust_precision, per_edge_ncc_precision, per_edge_lowe_precision, epi_edges_evaluated, disp_edges_evaluated, shift_edges_evaluated, clust_edges_evaluated, ncc_edges_evaluated, lowe_edges_evaluated)
        for (size_t i = 0; i < selected_primary_edges.size(); i += skip)
        {
            const auto &primary_edge = selected_primary_edges[i];

            if (!selected_ground_truth_edges.empty())
            {
                ground_truth_edge = selected_ground_truth_edges[i];
            }

            const auto &epipolar_line = epipolar_lines_secondary[i];
            const auto &primary_patch_one = primary_patch_set_one[i];
            const auto &primary_patch_two = primary_patch_set_two[i];

            if (!CheckEpipolarTangency(primary_edge, epipolar_line))
            {
                continue;
            }

            ///////////////////////////////EPIPOLAR DISTANCE THRESHOLD///////////////////////////////
#if MEASURE_TIMINGS
            auto start_epi = std::chrono::high_resolution_clock::now();
#endif
            std::vector<Edge> secondary_candidates_data = ExtractEpipolarEdges(epipolar_line, secondary_edges, 0.5);
            std::vector<Edge> test_secondary_candidates_data = ExtractEpipolarEdges(epipolar_line, secondary_edges, 3);

            local_epi_input_counts[thread_id].push_back(secondary_edges.size());

#if MEASURE_TIMINGS
            time_epi_edges_evaluated++;
            auto end_epi = std::chrono::high_resolution_clock::now();
            time_epi += std::chrono::duration<double, std::milli>(end_epi - start_epi).count();
#endif
            //> MARK: Epipolar Distance
            ///////////////////////////////EPIPOLAR DISTANCE THRESHOLD RECALL//////////////////////////
            if (!selected_ground_truth_edges.empty())
            {
                if (FilterByEpipolarDistance(
                        epi_true_positive, epi_false_negative, epi_true_negative, per_edge_epi_precision, epi_edges_evaluated,
                        secondary_candidates_data, test_secondary_candidates_data, ground_truth_edge,
                        0.5
                        ))
                    continue;
            }
            ///////////////////////////////MAXIMUM DISPARITY THRESHOLD//////////////////////////
#if MEASURE_TIMINGS
            auto start_disp = std::chrono::high_resolution_clock::now();
#endif

            // epi_output_counts.push_back(secondary_candidates_data.size());
            local_epi_output_counts[thread_id].push_back(secondary_candidates_data.size());

            std::vector<Edge> filtered_secondary_edges;

            FilterByDisparity(
                filtered_secondary_edges,
                secondary_candidates_data,
                !selected_ground_truth_edges.empty(),
                primary_edge);

            // disp_input_counts.push_back(secondary_candidates_data.size());
            local_disp_input_counts[thread_id].push_back(secondary_candidates_data.size());

#if MEASURE_TIMINGS
            time_disp_edges_evaluated++;
            auto end_disp = std::chrono::high_resolution_clock::now();
            time_disp += std::chrono::duration<double, std::milli>(end_disp - start_disp).count();
#endif
            //> MARK: Maximum Disparity
            ///////////////////////////////MAXIMUM DISPARITY THRESHOLD RECALL//////////////////////////
            if (!selected_ground_truth_edges.empty())
            {
                RecallUpdate(disp_true_positive, disp_false_negative, disp_edges_evaluated,
                             per_edge_disp_precision, filtered_secondary_edges, ground_truth_edge,
                             0.5);
            }
            ///////////////////////////////EPIPOLAR SHIFT THRESHOLD//////////////////////////
#if MEASURE_TIMINGS
            auto start_shift = std::chrono::high_resolution_clock::now();
#endif

            local_disp_output_counts[thread_id].push_back(filtered_secondary_edges.size());

            Eigen::Vector3d eigen_primary_edge(primary_edge.location.x, primary_edge.location.y, primary_edge.orientation);

            Eigen::MatrixXd eigen_secondary_edges(filtered_secondary_edges.size(), 3);
            for (size_t i = 0; i < filtered_secondary_edges.size(); ++i)
            {
                eigen_secondary_edges(i, 0) = filtered_secondary_edges[i].location.x;
                eigen_secondary_edges(i, 1) = filtered_secondary_edges[i].location.y;
                eigen_secondary_edges(i, 2) = filtered_secondary_edges[i].orientation;
            }

            Eigen::Vector3d epip_coeffs(epipolar_line(0), epipolar_line(1), epipolar_line(2));

            std::vector<Eigen::Vector3d> shifted_edges = PerformEpipolarShift(eigen_primary_edge, eigen_secondary_edges, epip_coeffs);

            std::vector<Edge> shifted_secondary_edges;
            for (const auto& shifted_edge : shifted_edges) {
                Edge e;
                e.location = cv::Point2d(shifted_edge(0), shifted_edge(1));
                e.orientation = shifted_edge(2);
                e.b_isEmpty = false;
                shifted_secondary_edges.push_back(e);
            }

            local_shift_input_counts[thread_id].push_back(filtered_secondary_edges.size());

#if MEASURE_TIMINGS
            time_shift_edges_evaluated++;
            auto end_shift = std::chrono::high_resolution_clock::now();
            time_shift += std::chrono::duration<double, std::milli>(end_shift - start_shift).count();
#endif
            //> MARK: Epipolar Shift
            ///////////////////////////////EPIPOLAR SHIFT THRESHOLD RECALL//////////////////////////
            if (!selected_ground_truth_edges.empty())
            {
                RecallUpdate(shift_true_positive, shift_false_negative, shift_edges_evaluated,
                             per_edge_shift_precision, shifted_secondary_edges, ground_truth_edge,
                             GT_SPATIAL_TOL);
            }
            ///////////////////////////////EPIPOLAR CLUSTER THRESHOLD//////////////////////////
#if MEASURE_TIMINGS
            auto start_cluster = std::chrono::high_resolution_clock::now();
#endif

            local_shift_output_counts[thread_id].push_back(shifted_secondary_edges.size());

            std::vector<std::vector<Edge>> clusters = ClusterEpipolarShiftedEdges(shifted_secondary_edges); 
            std::vector<EdgeCluster> cluster_centers;
            FormClusterCenters(cluster_centers, clusters);

            local_clust_input_counts[thread_id].push_back(shifted_secondary_edges.size());

#if MEASURE_TIMINGS
            time_clust_edges_evaluated++;
            auto end_cluster = std::chrono::high_resolution_clock::now();
            time_cluster += std::chrono::duration<double, std::milli>(end_cluster - start_cluster).count();
#endif
            //> MARK: Clustering
            ///////////////////////////////EPIPOLAR CLUSTER THRESHOLD RECALL//////////////////////////
            if (!selected_ground_truth_edges.empty())
            {
                RecallUpdate(cluster_true_positive, cluster_false_negative, clust_edges_evaluated,
                             per_edge_clust_precision, cluster_centers, ground_truth_edge,
                             GT_SPATIAL_TOL);
            }
            ///////////////////////////////EXTRACT PATCHES THRESHOLD////////////////////////////////////////////
#if MEASURE_TIMINGS
            auto start_patch = std::chrono::high_resolution_clock::now();
#endif
            local_clust_output_counts[thread_id].push_back(cluster_centers.size());

            std::vector<EdgeCluster> filtered_cluster_centers;
            std::vector<cv::Mat> secondary_patch_set_one;
            std::vector<cv::Mat> secondary_patch_set_two;

            int secondary_img_width  = secondary_image.cols;
            int secondary_img_height = secondary_image.rows;

            for (const auto& cluster : cluster_centers) {
                Edge edge = cluster.center_edge;

                auto [shifted_plus, shifted_minus] = get_Orthogonal_Shifted_Points(edge);

                if (!is_patch_in_bounds(shifted_plus, secondary_img_width, secondary_img_height) ||
                    !is_patch_in_bounds(shifted_minus, secondary_img_width, secondary_img_height)) {
                    continue;
                }

                cv::Mat patch_coord_x_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
                cv::Mat patch_coord_y_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
                cv::Mat patch_plus          = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

                cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
                cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
                cv::Mat patch_minus         = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

                get_patch_on_one_edge_side(
                    shifted_plus,
                    cluster.center_edge.orientation,
                    patch_coord_x_plus,
                    patch_coord_y_plus,
                    patch_plus,
                    secondary_image
                );

                get_patch_on_one_edge_side(
                    shifted_minus,
                    cluster.center_edge.orientation,
                    patch_coord_x_minus,
                    patch_coord_y_minus,
                    patch_minus,
                    secondary_image
                );

                cv::Mat patch_plus_32f, patch_minus_32f;
                if (patch_plus.type() != CV_32F) {
                    patch_plus.convertTo(patch_plus_32f, CV_32F);
                }
                else {
                    patch_plus_32f = patch_plus;
                }

                if (patch_minus.type() != CV_32F) {
                    patch_minus.convertTo(patch_minus_32f, CV_32F);
                }
                else {
                    patch_minus_32f = patch_minus;
                }

                filtered_cluster_centers.push_back(cluster);

                secondary_patch_set_one.push_back(patch_plus_32f);
                secondary_patch_set_two.push_back(patch_minus_32f);
            }

            local_patch_input_counts[thread_id].push_back(cluster_centers.size());

#if MEASURE_TIMINGS
            time_patch_edges_evaluated++;
            auto end_patch = std::chrono::high_resolution_clock::now();
            time_patch += std::chrono::duration<double, std::milli>(end_patch - start_patch).count();
            //> MARK: NCC
            ///////////////////////////////NCC THRESHOLD/////////////////////////////////////////////////////
            auto start_ncc = std::chrono::high_resolution_clock::now();
#endif
            local_patch_output_counts[thread_id].push_back(filtered_cluster_centers.size());

            std::vector<EdgeMatch> passed_ncc_matches;

            FilterByNCC(
                primary_patch_one,
                primary_patch_two,
                secondary_patch_set_one,
                secondary_patch_set_two,
                ground_truth_edge,
                passed_ncc_matches,
                filtered_cluster_centers,
                !selected_ground_truth_edges.empty(),
                ncc_true_positive,
                ncc_false_negative,
                per_edge_ncc_precision,
                ncc_edges_evaluated,
                NCC_THRESH,
                forward_direction,
                image_pair_index,
                veridical_csv,
                nonveridical_csv,
                primary_edge,
                epipolar_line
            );

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
            FilterByLowe(local_final_matches,
                         local_lowe_input_counts,
                         local_lowe_output_counts,
                         local_GT_right_edges_after_lowe,
                         thread_id,
                         passed_ncc_matches,
                         !selected_ground_truth_edges.empty(),
                         primary_edge,
                         ground_truth_edge,
                         lowe_true_positive,
                         lowe_false_negative,
                         per_edge_lowe_precision,
                         lowe_edges_evaluated,
                         GT_SPATIAL_TOL);
        } //> MARK: end of looping over left edges
    }

    if (forward_direction) {
        veridical_csv.close();
        nonveridical_csv.close();
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
    double per_image_epi_precision = (epi_edges_evaluated > 0) ? (per_edge_epi_precision / epi_edges_evaluated) : (0.0);


    double max_disparity_recall = 0.0;
    if ((disp_true_positive + disp_false_negative) > 0)
    {
        max_disparity_recall = static_cast<double>(disp_true_positive) / (disp_true_positive + disp_false_negative);
    }
    double per_image_disp_precision = (disp_edges_evaluated > 0) ? (per_edge_disp_precision / disp_edges_evaluated) : (0.0);


    double epi_shift_recall = 0.0;
    if ((shift_true_positive + shift_false_negative) > 0)
    {
        epi_shift_recall = static_cast<double>(shift_true_positive) / (shift_true_positive + shift_false_negative);
    }
    double per_image_shift_precision = (shift_edges_evaluated > 0) ? (per_edge_shift_precision / shift_edges_evaluated) : (0.0);


    double epi_cluster_recall = 0.0;
    if ((cluster_true_positive + cluster_false_negative) > 0)
    {
        epi_cluster_recall = static_cast<double>(cluster_true_positive) / (cluster_true_positive + cluster_false_negative);
    }
    double per_image_clust_precision = (clust_edges_evaluated > 0) ? (per_edge_clust_precision / clust_edges_evaluated) : (0.0);


    double ncc_recall = 0.0;
    if ((ncc_true_positive + ncc_false_negative) > 0)
    {
        ncc_recall = static_cast<double>(ncc_true_positive) / (ncc_true_positive + ncc_false_negative);
    }
    double per_image_ncc_precision = (ncc_edges_evaluated > 0) ? (per_edge_ncc_precision / ncc_edges_evaluated) : (0.0);


    double lowe_recall = 0.0;
    if ((lowe_true_positive + lowe_false_negative) > 0)
    {
        lowe_recall = static_cast<double>(lowe_true_positive) / (lowe_true_positive + lowe_false_negative);
    }
    double per_image_lowe_precision = (lowe_edges_evaluated > 0) ? (per_edge_lowe_precision / lowe_edges_evaluated) : (0.0);


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
    // std::cout << "Final matches size: " << final_matches.size() << std::endl;
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
        dataset.ground_truth_right_edges_after_lowe.insert(dataset.ground_truth_right_edges_after_lowe.end(), local_GT_right_edges_stack.begin(), local_GT_right_edges_stack.end());
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

bool CheckEpipolarTangency(const Edge &primary_edge, const Eigen::Vector3d &epipolar_line)
{
    double a = epipolar_line(0);
    double b = epipolar_line(1);
    double c = epipolar_line(2);

    if (std::abs(b) < 1e-6)
        return false;

    double a1_line = -a / b;
    double b1_line = -1;
    double m_epipolar = -a1_line / b1_line;
    double angle_diff_rad = abs(primary_edge.orientation - atan(m_epipolar));
    double angle_diff_deg = angle_diff_rad * (180.0 / M_PI);
    if (angle_diff_deg > 180)
    {
        angle_diff_deg -= 180;
    }

    bool primary_passes_tangency = (abs(angle_diff_deg - 0) > EPIP_TANGENCY_ORIENT_THRESH && abs(angle_diff_deg - 180) > EPIP_TANGENCY_ORIENT_THRESH) ? (true) : (false);
    return primary_passes_tangency;
}

bool FilterByEpipolarDistance(
    int &epi_true_positive,
    int &epi_false_negative,
    int &epi_true_negative,
    double &per_edge_epi_precision,
    int &epi_edges_evaluated,
    const std::vector<Edge> &secondary_edges,
    const std::vector<Edge> &test_secondary_edges,
    cv::Point2d &ground_truth_edge,
    double threshold)
{
    int epi_precision_numerator = 0;
    bool match_found = false;
    for (const auto &candidate : secondary_edges)
    {
        if (cv::norm(candidate.location - ground_truth_edge) <= threshold)
        {
            epi_precision_numerator++;
            match_found = true;
        }
    }
    if (match_found)
    {
        epi_true_positive++;
    }
    else
    {
        bool gt_match_found = false;
        for (const auto &test_candidate : test_secondary_edges)
        {
            if (cv::norm(test_candidate.location - ground_truth_edge) <= threshold)
            {
                gt_match_found = true;
                break;
            }
        }

        if (!gt_match_found)
        {
            epi_true_negative++;
            return true;
        }
        else
        {
            epi_false_negative++;
        }
    }
    if (!secondary_edges.empty())
    {
        per_edge_epi_precision += static_cast<double>(epi_precision_numerator) / secondary_edges.size();
        epi_edges_evaluated++;
    }
    return false;
}

void FilterByDisparity(
    std::vector<Edge> &filtered_secondary_edges,
    const std::vector<Edge> &edge_candidates,
    bool gt,
    const Edge &primary_edge)
{
    for (size_t j = 0; j < edge_candidates.size(); j++)
    {
        const Edge &candidate = edge_candidates[j];
        double disparity = primary_edge.location.x - candidate.location.x;
        if (!gt)
        {
            disparity = -disparity;
        }
        bool within_horizontal = (disparity >= 0) && (disparity <= MAX_DISP);
        bool within_vertical = std::abs(candidate.location.y - primary_edge.location.y) <= MAX_DISP;

        if (within_horizontal && within_vertical)
        {
            filtered_secondary_edges.push_back(candidate);
        }
    }
}

template <typename Container>
void RecallUpdate(int &true_positive,
                  int &false_negative,
                  int &edges_evaluated,
                  double &per_edge_precision,
                  const Container &output_candidates,
                  cv::Point2d &ground_truth_edge,
                  double threshold)
{
    int precision_numerator = 0;
    bool match_found = false;
    for (const auto &candidate : output_candidates)
    {
        cv::Point2d location;
        if constexpr (std::is_same_v<typename Container::value_type, Edge>)
        {
            location = candidate.location;
        }
        else if constexpr (std::is_same_v<typename Container::value_type, EdgeCluster>)
        {
            location = candidate.center_edge.location;
        }
        else if constexpr (std::is_same_v<typename Container::value_type, cv::Point2d>)
        {
            location = candidate;
        }
        else
        {
            std::cerr << "Unsupported type in RecallUpdate" << std::endl;
            return;
        }
        if (cv::norm(location - ground_truth_edge) <= threshold)
        {
            precision_numerator++;
            match_found = true;
        }
    }
    if (match_found)
    {
        true_positive++;
    }
    else
    {
        false_negative++;
    }
    if (!output_candidates.empty())
    {
        per_edge_precision += static_cast<double>(precision_numerator) / output_candidates.size();
        edges_evaluated++;
    }
}

void FormClusterCenters(
    std::vector<EdgeCluster> &cluster_centers,
    std::vector<std::vector<Edge>> &clusters)

{
    cluster_centers.clear();
    for (const auto &cluster_edges : clusters)
    {

        if (cluster_edges.empty())
            continue;

        cv::Point2d sum_point(0.0, 0.0);
        double sum_orientation = 0.0;

        for (size_t j = 0; j < cluster_edges.size(); j++)
        {
            sum_point += cluster_edges[j].location;
            sum_orientation += cluster_edges[j].orientation;
        }

        cv::Point2d avg_point = sum_point * (1.0 / cluster_edges.size());
        double avg_orientation = sum_orientation * (1.0 / cluster_edges.size());

        EdgeCluster cluster;
        cluster.center_edge = Edge{avg_point, avg_orientation, false};
        cluster.contributing_edges = cluster_edges;

        cluster_centers.push_back(cluster);
    }
}

void FilterByNCC(
    const cv::Mat &primary_patch_one,
    const cv::Mat &primary_patch_two,
    const std::vector<cv::Mat> &secondary_patch_set_one,
    const std::vector<cv::Mat> &secondary_patch_set_two,
    const cv::Point2d &ground_truth_edge,
    std::vector<EdgeMatch> &passed_ncc_matches,
    std::vector<EdgeCluster> &filtered_cluster_centers,
    bool gt,
    int &ncc_true_positive,
    int &ncc_false_negative,
    double &per_edge_ncc_precision,
    int &ncc_edges_evaluated,
    double threshold,
    bool forward_direction,
    int image_pair_index,
    std::ofstream &veridical_csv,
    std::ofstream &nonveridical_csv,
    const Edge &primary_edge,
    const Eigen::Vector3d &epipolar_line)
{
    int ncc_precision_numerator = 0;
    bool ncc_match_found = false;
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
               double err_to_gt = cv::norm(filtered_cluster_centers[i].center_edge.location - ground_truth_edge);   
               std::pair<double, double> pair_ncc_one_err(err_to_gt, ncc_one);
               std::pair<double, double> pair_ncc_two_err(err_to_gt, ncc_two);
               ncc_one_vs_err.push_back(pair_ncc_one_err);
               ncc_two_vs_err.push_back(pair_ncc_two_err);
#endif

            if (final_score > threshold) {
                EdgeMatch info;
                info.edge = filtered_cluster_centers[i].center_edge;
                info.final_score = final_score;
                info.contributing_edges = filtered_cluster_centers[i].contributing_edges;
                passed_ncc_matches.push_back(info);

                if (forward_direction) {
                    std::ostream& target_stream = 
                        (cv::norm(info.edge.location - ground_truth_edge) <= GT_SPATIAL_TOL)
                        ? veridical_csv : nonveridical_csv;

                    #pragma omp critical(csv_write)
                    {
                        target_stream << std::fixed << std::setprecision(8) << ","
                        << primary_edge.location.x << "," << primary_edge.location.y << "," << primary_edge.orientation << ","
                        << info.edge.location.x << "," << info.edge.location.y << "," << info.edge.orientation << ","
                        << ground_truth_edge.x << "," << ground_truth_edge.y << ","
                        << epipolar_line(0) << "," << epipolar_line(1) << "," << epipolar_line(2) << ","
                        << ncc_one << "," << ncc_two << "," << ncc_three << "," << ncc_four << ","
                        << score_one << "," << score_two << "," << final_score << "\n";
                    }
                }

                if (cv::norm(info.edge.location - ground_truth_edge) <= GT_SPATIAL_TOL) {
                    ncc_match_found = true;
                    ncc_precision_numerator++;
                }
            }
        }

        if (ncc_match_found) {
            ncc_true_positive++;
        } else {
            ncc_false_negative++;
        }
    }

    if (!passed_ncc_matches.empty()) {
        per_edge_ncc_precision += static_cast<double>(ncc_precision_numerator) / passed_ncc_matches.size();
        ncc_edges_evaluated++;
    }
}

void FilterByLowe(
    std::vector<std::vector<std::pair<Edge, EdgeMatch>>> &local_final_matches,
    std::vector<std::vector<int>> &local_lowe_input_counts,
    std::vector<std::vector<int>> &local_lowe_output_counts,
    std::vector<std::vector<cv::Point2d>> &local_GT_right_edges_after_lowe,
    int thread_id,
    const std::vector<EdgeMatch> &passed_ncc_matches,
    bool gt,
    const Edge &primary_edge,
    cv::Point2d &ground_truth_edge,
    int &lowe_true_positive,
    int &lowe_false_negative,
    double &per_edge_lowe_precision,
    int &lowe_edges_evaluated,
    double threshold)
{
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
            if (gt)
            {
                local_GT_right_edges_after_lowe[thread_id].push_back(ground_truth_edge);
                if (cv::norm(best_match.edge.location - ground_truth_edge) <= threshold)
                {
                    lowe_precision_numerator++;
                    lowe_true_positive++;
                }
                else
                {
                    lowe_false_negative++;
                }
            }
            Edge source_edge = primary_edge;

            local_final_matches[thread_id].emplace_back(source_edge, best_match);
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

        if (gt)
        {
            local_GT_right_edges_after_lowe[thread_id].push_back(ground_truth_edge);
            if (cv::norm(best_match.edge.location - ground_truth_edge) <= threshold)
            {
                lowe_precision_numerator++;
                lowe_true_positive++;
            }
            else
            {
                lowe_false_negative++;
            }
        }
        Edge source_edge = primary_edge;
        // final_matches.emplace_back(primary_edge, best_match);
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
}