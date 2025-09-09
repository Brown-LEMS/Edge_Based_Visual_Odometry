#include <iomanip>
#include "Matches.h"

/*
    Calculate the epipolar line for each edge point using the fundamental matrix.
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

/*
   Calculate the normal distance from an edge point to the epipolar line.
*/
double GetNormalDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &epiline_x, double &epiline_y) {
	double a1_line = Epip_Line_Coeffs(0);
	double b1_line = Epip_Line_Coeffs(1);
	double c1_line = Epip_Line_Coeffs(2);

	epiline_x = edge(0) - a1_line * (a1_line * edge(0) + b1_line * edge(1) + c1_line)/(pow(a1_line, 2) + pow(b1_line, 2));
	epiline_y = edge(1) - b1_line * (a1_line * edge(0) + b1_line * edge(1) + c1_line)/(pow(a1_line, 2) + pow(b1_line, 2));

	return sqrt(pow(edge(0) - epiline_x, 2) + pow(edge(1) - epiline_y, 2));
}

/*
   Calculate the tangential distance from an edge point to the epipolar line.
*/
double GetTangentialDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &x_intersection, double &y_intersection) {
	double a_edgeH2 = tan(edge(2)); 
	double b_edgeH2 = -1;
	double c_edgeH2 = -(a_edgeH2 * edge(0) - edge(1));

	double a1_line = Epip_Line_Coeffs(0);
	double b1_line = Epip_Line_Coeffs(1);
	double c1_line = Epip_Line_Coeffs(2);

	x_intersection = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
	y_intersection = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);

	return sqrt((x_intersection - edge(0)) * (x_intersection - edge(0)) + (y_intersection - edge(1)) * (y_intersection - edge(1)));
}

/*
   Perform epipolar shift for a pair of edges.
*/
Edge PerformEpipolarShift(
    Edge original_edge,
    const Eigen::Vector3d epipolar_line_coeffs, bool &b_pass_epipolar_tengency_check, Utility util)
{
    Eigen::Vector3d xy1_edge( original_edge.location.x, original_edge.location.y, 1.0 );
    double corrected_x, corrected_y, corrected_theta;
    double epiline_x, epiline_y;

    if ( util.getNormalDistance2EpipolarLine( epipolar_line_coeffs, xy1_edge, epiline_x, epiline_y ) < LOCATION_PERTURBATION )
    {
        //> If normal distance is small, move directly to the epipolar line
        cv::Point2d corrected_edge_loc(epiline_x, epiline_y);
        b_pass_epipolar_tengency_check = true;
        return Edge{corrected_edge_loc, original_edge.orientation, false};
    }
    else
    {
        double x_intersection, y_intersection;
        Eigen::Vector3d isolated_edge( original_edge.location.x, original_edge.location.y, original_edge.orientation );

        //> Inner two cases: 
        if ( util.getTangentialDistance2EpipolarLine( epipolar_line_coeffs, isolated_edge, x_intersection, y_intersection ) < EPIP_TANGENCY_DISPL_THRESH ) 
        {
            //> (i) if the displacement after epipolar shift is less than EPIP_TANGENCY_DISPL_THRESH, then feel free to shift it along its direction vector
            cv::Point2d corrected_edge_loc(x_intersection, y_intersection);
            b_pass_epipolar_tengency_check = true;
            return Edge{corrected_edge_loc, original_edge.orientation, false};
        }
        else
        {
            //> (ii) if not, then perturb the edge orientation first before shifting the edge along its direction vector
            // double p_theta = a1_line * cos(theta) + b1_line * sin(theta);
            // double derivative_p_theta = -a1_line * sin(theta) + b1_line * cos(theta);
            double theta = original_edge.orientation;
            double p_theta = epipolar_line_coeffs(0) * cos(theta) + epipolar_line_coeffs(1) * sin(theta);
            double derivative_p_theta = -epipolar_line_coeffs(0) * sin(theta) + epipolar_line_coeffs(1) * cos(theta);

            //> Decide how theta should be perturbed by observing the signs of p_theta and derivative_p_theta
            if (p_theta > 0 && derivative_p_theta < 0) theta -= ORIENT_PERTURBATION;
            else if (p_theta < 0 && derivative_p_theta < 0) theta -= ORIENT_PERTURBATION;
            else if (p_theta > 0 && derivative_p_theta > 0) theta += ORIENT_PERTURBATION;
            else if (p_theta < 0 && derivative_p_theta > 0) theta += ORIENT_PERTURBATION;

            //> Calculate the intersection between the tangent and epipolar line
            Eigen::Vector3d isolated_edge_( original_edge.location.x, original_edge.location.y, theta );
            if ( util.getTangentialDistance2EpipolarLine( epipolar_line_coeffs, isolated_edge_, x_intersection, y_intersection ) < EPIP_TANGENCY_DISPL_THRESH ) 
            {
                cv::Point2d corrected_edge_loc(x_intersection, y_intersection);
                b_pass_epipolar_tengency_check = true;
                return Edge{corrected_edge_loc, theta, false};
            } 
            else 
            {
                b_pass_epipolar_tengency_check = false;
                return Edge{original_edge.location, original_edge.orientation, false};
            }
        }
    }
}

/*
    Extract edges that are close to the epipolar line within a certain threshold.
*/
std::vector<std::pair<Edge, double>> ExtractEpipolarEdges(const Eigen::Vector3d &epipolar_line, const std::vector<Edge> &edges, double distance_threshold)
{
    std::vector<std::pair<Edge, double>> extracted_edges;

    for (size_t i = 0; i < edges.size(); ++i)
    {
        const auto &edge = edges[i];
        double x = edge.location.x;
        double y = edge.location.y;

        double distance = std::abs(epipolar_line(0) * x + epipolar_line(1) * y + epipolar_line(2)) /
                          std::sqrt((epipolar_line(0) * epipolar_line(0)) + (epipolar_line(1) * epipolar_line(1)));

        if (distance < distance_threshold)
        {
            extracted_edges.emplace_back(edge, distance);
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
std::vector<EdgeCluster> ClusterEpipolarShiftedEdges(std::vector<Edge> &valid_shifted_edges)
{
    EdgeClusterer edge_cluster_engine( valid_shifted_edges );
    edge_cluster_engine.performClustering( );

    std::vector<Edge> clustered_edges = edge_cluster_engine.Epip_Correct_Edges;
    const int Num_Of_Clusters = edge_cluster_engine.Num_Of_Clusters;
    std::vector<int> cluster_labels = edge_cluster_engine.cluster_labels;

    //> Renumbering the cluster labels into 0, 1, 2, etc for each epipolar shifted edge
    std::vector<int> renumbered_cluster_labels = cluster_labels;
    std::vector<int> unique_cluster_labels = find_Unique_Sorted_Numbers( cluster_labels );
    for (int i = 0; i < unique_cluster_labels.size(); i++) {
        for (int j = 0; j < cluster_labels.size(); j++) {
            if (cluster_labels[j] == unique_cluster_labels[i]) {
                renumbered_cluster_labels[j] = i;
            }
        }
    }

    //> Construct contributing edges for each cluster
    std::vector<std::vector<Edge>> edges_in_clusters;
    edges_in_clusters.resize(Num_Of_Clusters);
    for (int i = 0; i < renumbered_cluster_labels.size(); i++)
    {
        edges_in_clusters[ renumbered_cluster_labels[i] ].push_back(valid_shifted_edges[i]);
    }

    //> Construct cluster structure
    std::vector<EdgeCluster> cluster_centers;
    cluster_centers.resize(Num_Of_Clusters);
    for (int i = 0; i < Num_Of_Clusters; i++) 
    {
        for (int j = 0; j < renumbered_cluster_labels.size(); j++)
        {
            if (renumbered_cluster_labels[j] == i)
            {
                EdgeCluster cluster;
                cluster.center_edge = clustered_edges[j];
                cluster.contributing_edges = edges_in_clusters[renumbered_cluster_labels[j]];
                cluster_centers[i] = cluster;
                break;
            }
        }
    }

    return cluster_centers;
}

/*
    Get the orthogonal shifted points for a given edge.
*/
std::pair<cv::Point2d, cv::Point2d> GetOrthogonalShiftedPoints(const Edge edgel)
{
    double shifted_x1 = edgel.location.x + ORTHOGONAL_SHIFT_MAG * (std::sin(edgel.orientation));
    double shifted_y1 = edgel.location.y + ORTHOGONAL_SHIFT_MAG * (-std::cos(edgel.orientation));
    double shifted_x2 = edgel.location.x + ORTHOGONAL_SHIFT_MAG * (-std::sin(edgel.orientation));
    double shifted_y2 = edgel.location.y + ORTHOGONAL_SHIFT_MAG * (std::cos(edgel.orientation));

    cv::Point2d shifted_point_plus(shifted_x1, shifted_y1);
    cv::Point2d shifted_point_minus(shifted_x2, shifted_y2);

    return {shifted_point_plus, shifted_point_minus};
}

/*
    Get the patch coordinates and values for a given edge.
*/
void GetPatchForEdge(cv::Point2d shifted_point, double theta, cv::Mat &patch_coord_x, cv::Mat &patch_coord_y, cv::Mat &patch_val, const cv::Mat image)
{
    int half_patch_size = floor(PATCH_SIZE / 2);
    
    for (int i = -half_patch_size; i <= half_patch_size; i++) {
        for (int j = -half_patch_size; j <= half_patch_size; j++) {

            cv::Point2d rotated_point(cos(theta)*(i) - sin(theta)*(j) + shifted_point.x, sin(theta)*(i) + cos(theta)*(j) + shifted_point.y);
            patch_coord_x.at<double>(i + half_patch_size, j + half_patch_size) = rotated_point.x;
            patch_coord_y.at<double>(i + half_patch_size, j + half_patch_size) = rotated_point.y;

            double interp_val = Bilinear_Interpolation<double>(image, rotated_point);
            patch_val.at<double>(i + half_patch_size, j + half_patch_size) = interp_val;
        }
    }
}

/*
    Ensure image is CV_64F
*/
static cv::Mat Ensure64F(const cv::Mat& img) {
    if (img.type() == CV_64F) return img;
    cv::Mat out;
    img.convertTo(out, CV_64F);
    return out;
}

/*
    Get the image pair index from the dataset.
*/
static size_t GetImagePairIndex(const Dataset& dataset) {
    ETH3DIterator* eth3d_iter = dynamic_cast<ETH3DIterator*>(dataset.stereo_iterator.get());
    if (eth3d_iter) {
        return eth3d_iter->getCurrentIndex();
    }
    return 0;
}

/*
    Perform Bidirectional Consistency Test (BCT)
*/
static BidirectionalMetrics PerformBCT(
    const EdgeMatchResult& forward_match,
    const EdgeMatchResult& reverse_match,
    const Dataset& dataset,
    std::vector<std::pair<Edge, Edge>>& confirmed_matches
) {
    int matches_before_bct = static_cast<int>(forward_match.edge_to_cluster_matches.size());
    auto bct_start = std::chrono::high_resolution_clock::now();
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
    return bidirectional_metrics;
}

/*
    Get stereo edge pairs from left and right images using the provided dataset.
*/
StereoMatchResult GetStereoEdgePairs(const cv::Mat &left_image, const cv::Mat &right_image, Dataset &dataset)
{
    Utility util{};

    cv::Mat left_image_64f = Ensure64F(left_image);
    cv::Mat right_image_64f = Ensure64F(right_image);
    size_t image_pair_index = GetImagePairIndex(dataset);

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
    
    for (size_t i = 0; i < left_edges.size(); ++i) {
        const Edge &edge = left_edges[i];

        auto [shifted_plus, shifted_minus] = GetOrthogonalShiftedPoints(edge);

        cv::Mat patch_coord_x_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_coord_y_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_plus          = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    
        cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_minus         = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    
        GetPatchForEdge(
            shifted_plus,
            edge.orientation,
            patch_coord_x_plus,
            patch_coord_y_plus,
            patch_plus,
            left_image_64f
        );
    
        GetPatchForEdge(
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

    for (size_t i = 0; i < reverse_primary_edges.size(); ++i) {
        const Edge& edge = reverse_primary_edges[i];

        auto [shifted_plus, shifted_minus] = GetOrthogonalShiftedPoints(edge);

        cv::Mat patch_coord_x_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_coord_y_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_plus          = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

        cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
        cv::Mat patch_minus         = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

        GetPatchForEdge(
            shifted_plus,
            edge.orientation,
            patch_coord_x_plus,
            patch_coord_y_plus,
            patch_plus,
            right_image_64f
        );

        GetPatchForEdge(
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

    BidirectionalMetrics bidirectional_metrics = PerformBCT(forward_match, reverse_match, dataset, confirmed_matches);
    return StereoMatchResult{forward_match, reverse_match, confirmed_matches, bidirectional_metrics};
}

/*
    Calculate matches between selected primary edges and secondary edges using the provided patches and epipolar lines.
*/
EdgeMatchResult CalculateMatches(const std::vector<Edge> &selected_primary_edges, const std::vector<Edge> &secondary_edges,
                                 const std::vector<cv::Mat> &primary_patch_set_one, const std::vector<cv::Mat> &primary_patch_set_two, const std::vector<Eigen::Vector3d> &epipolar_lines_secondary,
                                 const cv::Mat &secondary_image, Dataset &dataset, const std::vector<cv::Point2d> &selected_ground_truth_edges, int image_pair_index, bool forward_direction)
{
    Utility util{};

    //> CH: Start timing
    auto total_start = std::chrono::high_resolution_clock::now();

    //> CH: Per-edge input/output counts
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

    //> CH: Total time
    double total_time;

    //> CH: Local structures for all passing distances
    std::vector<std::vector<double>> all_passing_distances;
    
    //> CH: Final matches
    std::vector<std::pair<Edge, EdgeMatch>> final_matches;

    //> CH: Local structures for all final matches
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

    //> SLL: Local structure for veridical right edge to epipolar line distances
    std::vector<std::vector<std::vector<double>>> local_all_passing_distances(omp_get_max_threads());

    //> CH: Per-edge timing metrics
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

    //> CH: Per-edge precision metrics
    double per_edge_epi_precision = 0.0;
    double per_edge_disp_precision = 0.0;
    double per_edge_shift_precision = 0.0;
    double per_edge_clust_precision = 0.0;
    double per_edge_patch_precision = 0.0;
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
    int patch_true_positive = 0;
    int patch_false_negative = 0;
    int patch_edges_evaluated = 0;
    int ncc_true_positive = 0;
    int ncc_false_negative = 0;
    int lowe_true_positive = 0;
    int lowe_false_negative = 0;
    int epi_edges_evaluated = 0;
    int disp_edges_evaluated = 0;
    int shift_edges_evaluated = 0;
    int clust_edges_evaluated = 0;
    int patch_edges_evaluated_local = 0;
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
            const auto &epipolar_line = epipolar_lines_secondary[i];
            const auto &primary_patch_one = primary_patch_set_one[i];
            const auto &primary_patch_two = primary_patch_set_two[i];

            if (!selected_ground_truth_edges.empty())
            {
                ground_truth_edge = selected_ground_truth_edges[i];
            }

            if (!CheckEpipolarTangency(primary_edge, epipolar_line))
            {
                continue;
            }

            ///////////////////////////////EPIPOLAR DISTANCE THRESHOLD///////////////////////////////
#if MEASURE_TIMINGS
            auto start_epi = std::chrono::high_resolution_clock::now();
#endif
            auto secondary_candidates_pairs = ExtractEpipolarEdges(epipolar_line, secondary_edges, 0.5);
            std::vector<Edge> secondary_candidate_edges;
            secondary_candidate_edges.reserve(secondary_candidates_pairs.size());
            for (const auto& pair : secondary_candidates_pairs) secondary_candidate_edges.push_back(pair.first);

            auto test_secondary_candidates_pairs = ExtractEpipolarEdges(epipolar_line, secondary_edges, 3);
            std::vector<Edge> test_secondary_candidate_edges;
            test_secondary_candidate_edges.reserve(test_secondary_candidates_pairs.size());
            for (const auto& pair : test_secondary_candidates_pairs) test_secondary_candidate_edges.push_back(pair.first);

            local_epi_input_counts[thread_id].push_back(secondary_edges.size());

#if MEASURE_TIMINGS
            time_epi_edges_evaluated++;
            auto end_epi = std::chrono::high_resolution_clock::now();
            time_epi += std::chrono::duration<double, std::milli>(end_epi - start_epi).count();
#endif

            ///////////////////////////////EPIPOLAR DISTANCE THRESHOLD RECALL//////////////////////////
            if (!selected_ground_truth_edges.empty())
            {
                std::vector<double> passing_distances;
                if (FilterByEpipolarDistance(
                        epi_true_positive, epi_false_negative, epi_true_negative, per_edge_epi_precision, epi_edges_evaluated,
                        secondary_candidates_pairs, test_secondary_candidates_pairs, ground_truth_edge,
                        0.5,
                        passing_distances
                        ))
                    continue;
                    local_all_passing_distances[thread_id].push_back(passing_distances);
            }

            ///////////////////////////////MAXIMUM DISPARITY THRESHOLD//////////////////////////
#if MEASURE_TIMINGS
            auto start_disp = std::chrono::high_resolution_clock::now();
#endif
            local_epi_output_counts[thread_id].push_back(secondary_candidate_edges.size());

            std::vector<Edge> filtered_secondary_edges;

            FilterByDisparity(  
                filtered_secondary_edges,
                secondary_candidate_edges,
                !selected_ground_truth_edges.empty(),
                primary_edge);

            local_disp_input_counts[thread_id].push_back(secondary_candidate_edges.size());

#if MEASURE_TIMINGS
            time_disp_edges_evaluated++;
            auto end_disp = std::chrono::high_resolution_clock::now();
            time_disp += std::chrono::duration<double, std::milli>(end_disp - start_disp).count();
#endif

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

            std::vector<Edge> shifted_secondary_edges;
            EpipolarShiftFilter(filtered_secondary_edges, shifted_secondary_edges, epipolar_line, util);

            local_shift_input_counts[thread_id].push_back(filtered_secondary_edges.size());

#if MEASURE_TIMINGS
            time_shift_edges_evaluated++;
            auto end_shift = std::chrono::high_resolution_clock::now();
            time_shift += std::chrono::duration<double, std::milli>(end_shift - start_shift).count();
#endif

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

            std::vector<EdgeCluster> cluster_centers = ClusterEpipolarShiftedEdges(shifted_secondary_edges);

            local_clust_input_counts[thread_id].push_back(shifted_secondary_edges.size());

#if MEASURE_TIMINGS
            time_clust_edges_evaluated++;
            auto end_cluster = std::chrono::high_resolution_clock::now();
            time_cluster += std::chrono::duration<double, std::milli>(end_cluster - start_cluster).count();
#endif

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

            if (!selected_ground_truth_edges.empty()) {
                RecallUpdate(
                    patch_true_positive,
                    patch_false_negative,
                    patch_edges_evaluated,
                    per_edge_patch_precision,
                    cluster_centers,
                    ground_truth_edge,
                    GT_SPATIAL_TOL
                );
            }

            std::vector<EdgeCluster> filtered_cluster_centers;
            std::vector<cv::Mat> secondary_patch_set_one;
            std::vector<cv::Mat> secondary_patch_set_two;

            for (const auto& cluster : cluster_centers) {
                Edge edge = cluster.center_edge;

                auto [shifted_plus, shifted_minus] = GetOrthogonalShiftedPoints(edge);

                cv::Mat patch_coord_x_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
                cv::Mat patch_coord_y_plus  = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
                cv::Mat patch_plus          = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

                cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
                cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
                cv::Mat patch_minus         = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

                GetPatchForEdge(
                    shifted_plus,
                    edge.orientation,
                    patch_coord_x_plus,
                    patch_coord_y_plus,
                    patch_plus,
                    secondary_image
                );

                GetPatchForEdge(
                    shifted_minus,
                    edge.orientation,
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
        } 
    }
    //> MARK: End of processing for current thread

    if (forward_direction) {
        veridical_csv.close();
        nonveridical_csv.close();
    }

#if MEASURE_TIMINGS
    auto total_end = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
#endif


    RecallPrecision epi_metrics = ComputeRecallAndPrecision(epi_true_positive, epi_false_negative, epi_edges_evaluated, per_edge_epi_precision);
    double epi_distance_recall = epi_metrics.recall;
    double per_image_epi_precision = epi_metrics.precision;

    RecallPrecision disp_metrics = ComputeRecallAndPrecision(disp_true_positive, disp_false_negative, disp_edges_evaluated, per_edge_disp_precision);
    double max_disparity_recall = disp_metrics.recall;
    double per_image_disp_precision = disp_metrics.precision;

    RecallPrecision shift_metrics = ComputeRecallAndPrecision(shift_true_positive, shift_false_negative, shift_edges_evaluated, per_edge_shift_precision);
    double epi_shift_recall = shift_metrics.recall;
    double per_image_shift_precision = shift_metrics.precision;

    RecallPrecision clust_metrics = ComputeRecallAndPrecision(cluster_true_positive, cluster_false_negative, clust_edges_evaluated, per_edge_clust_precision);
    double epi_cluster_recall = clust_metrics.recall;
    double per_image_clust_precision = clust_metrics.precision;

    RecallPrecision patch_metrics = ComputeRecallAndPrecision(patch_true_positive, patch_false_negative, patch_edges_evaluated, per_edge_patch_precision);
    double patch_recall = patch_metrics.recall;
    double per_image_patch_precision_metric = patch_metrics.precision;

    RecallPrecision ncc_metrics = ComputeRecallAndPrecision(ncc_true_positive, ncc_false_negative, ncc_edges_evaluated, per_edge_ncc_precision);
    double ncc_recall = ncc_metrics.recall;
    double per_image_ncc_precision = ncc_metrics.precision;

    RecallPrecision lowe_metrics = ComputeRecallAndPrecision(lowe_true_positive, lowe_false_negative, lowe_edges_evaluated, per_edge_lowe_precision);
    double lowe_recall = lowe_metrics.recall;
    double per_image_lowe_precision = lowe_metrics.precision;

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
        final_matches.insert(final_matches.end(), local_matches.begin(), local_matches.end());
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
        dataset.ground_truth_right_edges_after_lowe.insert(dataset.ground_truth_right_edges_after_lowe.end(), local_GT_right_edges_stack.begin(), local_GT_right_edges_stack.end());
    for (const auto &local_counts : local_all_passing_distances)
        all_passing_distances.insert(all_passing_distances.end(), local_counts.begin(), local_counts.end());

    std::vector<double> edge_to_epi_distances;
    for (const auto& vector : all_passing_distances) {
        edge_to_epi_distances.insert(edge_to_epi_distances.end(), vector.begin(), vector.end());
    }

    return EdgeMatchResult{
        RecallMetrics{
            epi_distance_recall,
            max_disparity_recall,
            epi_shift_recall,
            epi_cluster_recall,
            patch_recall,
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
            per_image_patch_precision_metric,
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
        edge_to_epi_distances,
        final_matches
    };
}

/*
    Apply epipolar shift filtering to the given edges.
*/
void EpipolarShiftFilter(
    const std::vector<Edge> &filtered_edges,
    std::vector<Edge> &shifted_edges,
    const Eigen::Vector3d &epipolar_line,
    Utility util)
{

    for (size_t j = 0; j < filtered_edges.size(); j++)
    {
        bool b_secondary_passes_tangency = false;

        Edge shifted_edge = PerformEpipolarShift(filtered_edges[j], epipolar_line, b_secondary_passes_tangency, util);
        if (b_secondary_passes_tangency)
        {
            shifted_edges.push_back(shifted_edge);
        }
    }
}

/*
    Compute recall and precision metrics.
*/
RecallPrecision ComputeRecallAndPrecision(int true_positive, int false_negative, int edges_evaluated, double per_edge_precision) {
    double recall = 0.0;
    if ((true_positive + false_negative) > 0) {
        recall = static_cast<double>(true_positive) / (true_positive + false_negative);
    }
    double precision = (edges_evaluated > 0) ? (per_edge_precision / edges_evaluated) : 0.0;
    return {recall, precision};
}

/*
    Check if the primary edge is tangent to the epipolar line.
*/
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

/*
    Filter secondary edges by their distance to the ground truth edge.
*/
bool FilterByEpipolarDistance(
    int &epi_true_positive,
    int &epi_false_negative,
    int &epi_true_negative,
    double &per_edge_epi_precision,
    int &epi_edges_evaluated,
    const std::vector<std::pair<Edge, double>> &secondary_edges,
    const std::vector<std::pair<Edge, double>> &test_secondary_edges,
    cv::Point2d &ground_truth_edge,
    double threshold,
    std::vector<double> &passing_distances)
{
    int epi_precision_numerator = 0;
    bool match_found = false;

    passing_distances.clear();

    double min_distance = std::numeric_limits<double>::max();
    double min_value = 0.0;

    for (const auto &candidate : secondary_edges)
    {
        double dist = cv::norm(candidate.first.location - ground_truth_edge);

        if (dist <= threshold)
        {
            if (dist < min_distance) {
                min_distance = dist;
                min_value = candidate.second;
            }
            
            epi_precision_numerator++;
            match_found = true;
        }
    }
    if (match_found)
    {
        passing_distances.push_back(min_value);
        epi_true_positive++;
    }
    else
    {
        bool gt_match_found = false;
        for (const auto &test_candidate : test_secondary_edges)
        {
            if (cv::norm(test_candidate.first.location - ground_truth_edge) <= threshold)
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

/*
    Filter secondary edges by their distance to the ground truth edge.
*/
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

/*
    Update recall and precision metrics.
*/
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

/*
    Form cluster centers from the edge clusters.
*/
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

/*
    Filter secondary edges by their distance to the ground truth edge.
*/
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
    int rotated_ncc_fail_count = 0;
    

    if (!primary_patch_one.empty() && !primary_patch_two.empty() &&
        !secondary_patch_set_one.empty() && !secondary_patch_set_two.empty())
    {
        for (size_t i = 0; i < filtered_cluster_centers.size(); ++i)
        {
            std::cout << filtered_cluster_centers.size() << std::endl;
            const Edge& secondary_edge = filtered_cluster_centers[i].center_edge;

            cv::Mat sec_patch_one = secondary_patch_set_one[i];
            cv::Mat sec_patch_two = secondary_patch_set_two[i];

            double orientation = secondary_edge.orientation;
            double deg = orientation * 180.0 / M_PI;

            // Save original patches before rotation
            cv::Mat sec_patch_one_original = sec_patch_one.clone();
            cv::Mat sec_patch_two_original = sec_patch_two.clone();

            bool rotated = false;
            if (std::abs(deg - 90.0) < 10.0) {
                cv::rotate(sec_patch_one, sec_patch_one, cv::ROTATE_180);
                cv::rotate(sec_patch_two, sec_patch_two, cv::ROTATE_180);
                rotated = true;
            }

            double ncc_one   = ComputeNCC(primary_patch_one, sec_patch_one);
            double ncc_two   = ComputeNCC(primary_patch_two, sec_patch_two);
            double ncc_three = ComputeNCC(primary_patch_one, sec_patch_two);
            double ncc_four  = ComputeNCC(primary_patch_two, sec_patch_one);

            double score_one = std::min(ncc_one, ncc_two);
            double score_two = std::min(ncc_three, ncc_four);
            double final_score = std::max(score_one, score_two);
            // double final_score = std::max({ncc_one, ncc_two, ncc_three, ncc_four});

            // Count rotated edges that fail NCC threshold
            if (rotated && final_score <= threshold) {
                rotated_ncc_fail_count++;
            }

            if (gt && cv::norm(filtered_cluster_centers[i].center_edge.location - ground_truth_edge) <= GT_SPATIAL_TOL && final_score <= threshold && forward_direction)
            {
                std::filesystem::path ncc_dir = std::filesystem::path("ncc_stats");
                std::filesystem::create_directories(ncc_dir);
                std::string patch_csv_name = "image_pair_" + std::to_string(image_pair_index) + "_failpatch_" + std::to_string(i) + ".csv";
                std::filesystem::path patch_csv_path = ncc_dir / patch_csv_name;
                std::ofstream patch_csv(patch_csv_path.string());
                if (patch_csv)
                {
                    patch_csv << "# primary_patch_one\n";
                    for (int r = 0; r < primary_patch_one.rows; ++r) {
                        for (int c = 0; c < primary_patch_one.cols; ++c) {
                            patch_csv << primary_patch_one.at<float>(r, c);
                            if (c < primary_patch_one.cols - 1) patch_csv << ",";
                        }
                        patch_csv << "\n";
                    }
                    patch_csv << "# primary_patch_two\n";
                    for (int r = 0; r < primary_patch_two.rows; ++r) {
                        for (int c = 0; c < primary_patch_two.cols; ++c) {
                            patch_csv << primary_patch_two.at<float>(r, c);
                            if (c < primary_patch_two.cols - 1) patch_csv << ",";
                        }
                        patch_csv << "\n";
                    }
                    // Export original secondary patches
                    patch_csv << "# secondary_patch_set_one_original\n";
                    for (int r = 0; r < sec_patch_one_original.rows; ++r) {
                        for (int c = 0; c < sec_patch_one_original.cols; ++c) {
                            patch_csv << sec_patch_one_original.at<float>(r, c);
                            if (c < sec_patch_one_original.cols - 1) patch_csv << ",";
                        }
                        patch_csv << "\n";
                    }
                    patch_csv << "# secondary_patch_set_two_original\n";
                    for (int r = 0; r < sec_patch_two_original.rows; ++r) {
                        for (int c = 0; c < sec_patch_two_original.cols; ++c) {
                            patch_csv << sec_patch_two_original.at<float>(r, c);
                            if (c < sec_patch_two_original.cols - 1) patch_csv << ",";
                        }
                        patch_csv << "\n";
                    }
                    // Export rotated secondary patches (if rotated)
                    patch_csv << "# secondary_patch_set_one_rotated\n";
                    for (int r = 0; r < sec_patch_one.rows; ++r) {
                        for (int c = 0; c < sec_patch_one.cols; ++c) {
                            patch_csv << sec_patch_one.at<float>(r, c);
                            if (c < sec_patch_one.cols - 1) patch_csv << ",";
                        }
                        patch_csv << "\n";
                    }
                    patch_csv << "# secondary_patch_set_two_rotated\n";
                    for (int r = 0; r < sec_patch_two.rows; ++r) {
                        for (int c = 0; c < sec_patch_two.cols; ++c) {
                            patch_csv << sec_patch_two.at<float>(r, c);
                            if (c < sec_patch_two.cols - 1) patch_csv << ",";
                        }
                        patch_csv << "\n";
                    }
                    patch_csv << "# meta\n";
                    patch_csv << "primary_x," << primary_edge.location.x << "\n";
                    patch_csv << "primary_y," << primary_edge.location.y << "\n";
                    patch_csv << "primary_theta," << primary_edge.orientation << "\n";
                    patch_csv << "secondary_x," << filtered_cluster_centers[i].center_edge.location.x << "\n";
                    patch_csv << "secondary_y," << filtered_cluster_centers[i].center_edge.location.y << "\n";
                    patch_csv << "secondary_theta," << filtered_cluster_centers[i].center_edge.orientation << "\n";
                    patch_csv << "secondary_rotated," << rotated << "\n";
                    patch_csv << "ncc_one," << ncc_one << "\n";
                    patch_csv << "ncc_two," << ncc_two << "\n";
                    patch_csv << "ncc_three," << ncc_three << "\n";
                    patch_csv << "ncc_four," << ncc_four << "\n";
                    patch_csv << "score_one," << score_one << "\n";
                    patch_csv << "score_two," << score_two << "\n";
                    patch_csv << "final_score," << final_score << "\n";
                    patch_csv << "threshold," << threshold << "\n";
                }
            }

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

        // if (!ncc_match_found) {
        //     for (size_t i = 0; i < filtered_cluster_centers.size(); ++i) {
        //         double dist = cv::norm(filtered_cluster_centers[i].center_edge.location - ground_truth_edge);
        //         if (dist <= GT_SPATIAL_TOL) {
        //             double ncc_one = ComputeNCC(primary_patch_one, secondary_patch_set_one[i]);
        //             double ncc_two = ComputeNCC(primary_patch_two, secondary_patch_set_two[i]);
        //             double ncc_three = ComputeNCC(primary_patch_one, secondary_patch_set_two[i]);
        //             double ncc_four = ComputeNCC(primary_patch_two, secondary_patch_set_one[i]);
        //             double score_one = std::min(ncc_one, ncc_two);
        //             double score_two = std::min(ncc_three, ncc_four);
        //             double final_score = std::max(score_one, score_two);
        //             std::cout << "[NCC FAIL] Missed veridical match for primary edge at ("
        //                       << primary_edge.location.x << ", " << primary_edge.location.y << ", theta=" << primary_edge.orientation << ") "
        //                       << "and secondary edge at ("
        //                       << filtered_cluster_centers[i].center_edge.location.x << ", "
        //                       << filtered_cluster_centers[i].center_edge.location.y << ", theta=" << filtered_cluster_centers[i].center_edge.orientation << ")\n"
        //                       << "  Distance to GT: " << dist << "\n"
        //                       << "  NCC scores: " << ncc_one << ", " << ncc_two << ", " << ncc_three << ", " << ncc_four << "\n"
        //                       << "  score_one: " << score_one << ", score_two: " << score_two << ", final_score: " << final_score << "\n"
        //                       << "  NCC threshold: " << threshold << std::endl;
        //         }
        //     }
        // }

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

    // Print the number of rotated edges that failed NCC threshold
    std::cout << "[FilterByNCC] Rotated edges that failed NCC threshold: " << rotated_ncc_fail_count << std::endl;
}

/*
   Filter secondary edges by their distance to the ground truth edge.
*/
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
        local_final_matches[thread_id].emplace_back(source_edge, best_match);
        local_lowe_output_counts[thread_id].push_back(1);
    }
    else
    {
        lowe_false_negative++;
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