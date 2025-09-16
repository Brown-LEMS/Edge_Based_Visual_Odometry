#include "Matches.h"
#include <unordered_map>

// task: finished the SIFT freature detection and matching -> result : 50/60 precision & recall
//  TODO: 1. do location filter/SCC filtering, then combine it with SIFT. Making SIFT location sensitive.
//        2. OpenMP parallelization & code optimization

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

void get_patch_on_one_edge_side(cv::Point2d shifted_point, double theta,
                                cv::Mat &patch_coord_x, cv::Mat &patch_coord_y,
                                cv::Mat &patch_val, const cv::Mat img)
{
    int half_patch_size = floor(PATCH_SIZE / 2);
    for (int i = -half_patch_size; i <= half_patch_size; i++)
    {
        for (int j = -half_patch_size; j <= half_patch_size; j++)
        {
            //> get the rotated coordinate
            cv::Point2d rotated_point(cos(theta) * (i)-sin(theta) * (j) + shifted_point.x, sin(theta) * (i) + cos(theta) * (j) + shifted_point.y);
            patch_coord_x.at<double>(i + half_patch_size, j + half_patch_size) = rotated_point.x;
            patch_coord_y.at<double>(i + half_patch_size, j + half_patch_size) = rotated_point.y;

            //> get the image intensity of the rotated coordinate
            double interp_val = Bilinear_Interpolation<double>(img, rotated_point);
            patch_val.at<double>(i + half_patch_size, j + half_patch_size) = interp_val;
        }
    }
}

double get_similarity(const cv::Mat patch_one, const cv::Mat patch_two)
{
    double mean_one = (cv::mean(patch_one))[0];
    double mean_two = (cv::mean(patch_two))[0];
    double sum_of_squared_one = (cv::sum((patch_one - mean_one).mul(patch_one - mean_one))).val[0];
    double sum_of_squared_two = (cv::sum((patch_two - mean_two).mul(patch_two - mean_two))).val[0];

    const double epsilon = 1e-10;
    if (sum_of_squared_one < epsilon || sum_of_squared_two < epsilon)
        return -1.0;

    double denom_one = sqrt(sum_of_squared_one);
    double denom_two = sqrt(sum_of_squared_two);

    cv::Mat norm_one = (patch_one - mean_one) / denom_one;
    cv::Mat norm_two = (patch_two - mean_two) / denom_two;
    return norm_one.dot(norm_two);
}

double edge_patch_similarity(const Edge target_edge_H1, const Edge target_edge_H2, const cv::Mat gray_img_H1, const cv::Mat gray_img_H2)
{
    std::pair<cv::Point2d, cv::Point2d> shifted_points_H1 = get_Orthogonal_Shifted_Points(target_edge_H1);
    std::pair<cv::Point2d, cv::Point2d> shifted_points_H2 = get_Orthogonal_Shifted_Points(target_edge_H2);

    // std::cout << "Orthogoanl shifted point for H1 edge: (" << shifted_points_H1.first.x << ", " << shifted_points_H1.second.y << ")" << std::endl;
    // std::cout << "Orthogoanl shifted point for H2 edge: (" << shifted_points_H2.first.x << ", " << shifted_points_H2.second.y << ")" << std::endl;

    cv::Mat patch_coord_x_plus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_y_plus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_plus_H1 = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_minus_H1 = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_plus_H2 = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_minus_H2 = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);

    //> get the two patches on the two sides of the edge in H1
    get_patch_on_one_edge_side(shifted_points_H1.first, target_edge_H1.orientation, patch_coord_x_plus, patch_coord_y_plus, patch_plus_H1, gray_img_H1);
    get_patch_on_one_edge_side(shifted_points_H1.second, target_edge_H1.orientation, patch_coord_x_minus, patch_coord_y_minus, patch_minus_H1, gray_img_H1);

    //> get the two patches on the two sides of the edge in H2
    get_patch_on_one_edge_side(shifted_points_H2.first, target_edge_H2.orientation, patch_coord_x_plus, patch_coord_y_plus, patch_plus_H2, gray_img_H2);
    get_patch_on_one_edge_side(shifted_points_H2.second, target_edge_H2.orientation, patch_coord_x_minus, patch_coord_y_minus, patch_minus_H2, gray_img_H2);

    if (patch_plus_H1.type() != CV_32F)
        patch_plus_H1.convertTo(patch_plus_H1, CV_32F);
    if (patch_minus_H1.type() != CV_32F)
        patch_minus_H1.convertTo(patch_minus_H1, CV_32F);
    if (patch_plus_H2.type() != CV_32F)
        patch_plus_H2.convertTo(patch_plus_H2, CV_32F);
    if (patch_minus_H2.type() != CV_32F)
        patch_minus_H2.convertTo(patch_minus_H2, CV_32F);

    double sim_pp = get_similarity(patch_plus_H1, patch_plus_H2);   //> (A+, B+)
    double sim_nn = get_similarity(patch_minus_H1, patch_minus_H2); //> (A-, B-)
    double sim_pn = get_similarity(patch_plus_H1, patch_minus_H2);  //> (A+, B-)
    double sim_np = get_similarity(patch_minus_H1, patch_plus_H2);  //> (A-, B+)
    // std::cout << "- Similarity scores of (pp, nn, pn, np) = (" << sim_pp << ", " << sim_nn << ", " << sim_pn << ", " << sim_np << ")" << std::endl;
    double final_SIM_score = std::max({sim_pp, sim_nn, sim_pn, sim_np});
    return final_SIM_score;
    // std::cout << "  Final similarity score = " << final_SIM_score << std::endl;
}

Eigen::Vector3d getEpipolarLineCoeffs(Eigen::Vector3d point_in_pixel, Eigen::Matrix3d F)
{
    Eigen::Vector3d epip_line = F * point_in_pixel;
    Eigen::Vector3d epip_line_on_img(-epip_line(0) / epip_line(1), -1.0, -epip_line(2) / epip_line(1));
    return epip_line_on_img;
    // return F * point_in_pixel;
}

double getNormalDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &epiline_x, double &epiline_y)
{
    double a1_line = Epip_Line_Coeffs(0);
    double b1_line = Epip_Line_Coeffs(1);
    double c1_line = Epip_Line_Coeffs(2);
    epiline_x = edge(0) - a1_line * (a1_line * edge(0) + b1_line * edge(1) + c1_line) / (pow(a1_line, 2) + pow(b1_line, 2));
    epiline_y = edge(1) - b1_line * (a1_line * edge(0) + b1_line * edge(1) + c1_line) / (pow(a1_line, 2) + pow(b1_line, 2));
    return sqrt(pow(edge(0) - epiline_x, 2) + pow(edge(1) - epiline_y, 2));
}

double getTangentialDistance2EpipolarLine(Eigen::Vector3d Epip_Line_Coeffs, Eigen::Vector3d edge, double &x_intersection, double &y_intersection)
{
    double a_edgeH2 = tan(edge(2)); // tan(theta2)
    double b_edgeH2 = -1;
    double c_edgeH2 = -(a_edgeH2 * edge(0) - edge(1)); // −(a⋅x2−y2)
    double a1_line = Epip_Line_Coeffs(0);
    double b1_line = Epip_Line_Coeffs(1);
    double c1_line = Epip_Line_Coeffs(2);
    x_intersection = (b1_line * c_edgeH2 - b_edgeH2 * c1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    y_intersection = (c1_line * a_edgeH2 - c_edgeH2 * a1_line) / (a1_line * b_edgeH2 - a_edgeH2 * b1_line);
    return sqrt((x_intersection - edge(0)) * (x_intersection - edge(0)) + (y_intersection - edge(1)) * (y_intersection - edge(1)));
}

// /*
//     calculate the epipolar line for each edge point using the fundamental matrix.
// */

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
    Perform an epipolar shift on the original edge location based on the epipolar line coefficients.
    The function checks if the corrected edge passes the epipolar tengency test.
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
        //> CH TODO: Pass the frame ID to the last input argument
        return Edge{corrected_edge_loc, original_edge.orientation, false, 0};
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
            //> CH TODO: Pass the frame ID to the last input argument
            return Edge{corrected_edge_loc, original_edge.orientation, false, 0};
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
                //> CH TODO: Pass the frame ID to the last input argument
                return Edge{corrected_edge_loc, theta, false, 0};
            } 
            else 
            {
                b_pass_epipolar_tengency_check = false;

                //> CH TODO: Pass the frame ID to the last input argument
                return Edge{original_edge.location, original_edge.orientation, false, 0};
            }
        }
    }
}

/*
    Extract edges that are close to the epipolar line within a specified distance threshold.
    Returns a pair of vectors: one for the extracted edge locations and one for their orientations.
*/
std::vector<Edge> ExtractEpipolarEdges(const Eigen::Vector3d &epipolar_line, const std::vector<Edge> &edges, const double dist_tol)
{
    std::vector<Edge> extracted_edges;

    for (const auto& e: edges)
    {
        double x = e.location.x;
        double y = e.location.y;
        double dist_to_epip_line = std::abs(epipolar_line(0) * x + epipolar_line(1) * y + epipolar_line(2)) / std::sqrt((epipolar_line(0) * epipolar_line(0)) + (epipolar_line(1) * epipolar_line(1)));

        if (dist_to_epip_line < dist_tol) 
        {
            extracted_edges.push_back(e);
        }
    }

    return extracted_edges;
}

std::vector<Edge> get_right_edges_close_to_GT_location(Edge target_left_edge, const cv::Point2d GT_location, const std::vector<Edge> constrained_right_edges, const double dist_tol) 
{
    std::vector<Edge> right_edges_close_to_GT_location;
    for (const auto& re : constrained_right_edges) 
    {
        if (cv::norm(GT_location - re.location) < dist_tol) 
        {
            right_edges_close_to_GT_location.push_back(re);
        }
    }
    return right_edges_close_to_GT_location;
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

    // EdgeCluster cluster;
    //     cluster.center_edge = Edge{avg_point, avg_orientation, false};
    //     cluster.contributing_edges = cluster_edges;

    
    // std::vector<std::vector<Edge>> clusters;

    // if (valid_shifted_edges.empty())
    // {
    //     return clusters;
    // }

    // std::sort(valid_shifted_edges.begin(), valid_shifted_edges.end(),
    //           [](const Edge &a, const Edge &b)
    //           {
    //               return a.location.x < b.location.x;
    //           });

    // std::vector<Edge> current_cluster;
    // current_cluster.push_back(valid_shifted_edges[0]);

    // for (size_t i = 1; i < valid_shifted_edges.size(); ++i)
    // {
    //     double distance = cv::norm(valid_shifted_edges[i].location - valid_shifted_edges[i - 1].location);
    //     double orientation_difference = std::abs(valid_shifted_edges[i].orientation - valid_shifted_edges[i - 1].orientation);

    //     if (distance <= EDGE_CLUSTER_THRESH && orientation_difference < 5.0)
    //     {
    //         current_cluster.push_back(valid_shifted_edges[i]);
    //     }
    //     else
    //     {
    //         clusters.emplace_back(current_cluster);
    //         current_cluster.clear();
    //         current_cluster.push_back(valid_shifted_edges[i]);
    //     }
    // }

    // if (!current_cluster.empty())
    // {
    //     clusters.emplace_back(current_cluster);
    // }

    // return clusters;
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
    Calculate orthogonal shifts for edge points based on their orientations.
    Returns two vectors of shifted points, one for each orthogonal direction.
*/
std::pair<std::vector<cv::Point2d>, std::vector<cv::Point2d>> CalculateOrthogonalShifts(const std::vector<Edge> &edge_points, double shift_magnitude, Dataset &dataset)
{
    std::vector<cv::Point2d> shifted_points_one;
    std::vector<cv::Point2d> shifted_points_two;

    for (size_t i = 0; i < edge_points.size(); ++i)
    {
        const auto &edge_point = edge_points[i].location;
        double theta = edge_points[i].orientation;

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

void get_Stereo_Edge_GT_Pairs(Dataset &dataset, StereoEdgeCorrespondencesGT& stereo_frame, const std::vector<Edge> right_edges)
{
    std::vector<Eigen::Vector3d> epip_line_coeffs = CalculateEpipolarLine(dataset.get_fund_mat_21(), stereo_frame.left_edges);
    std::vector<int> indices_to_remove;
    
    //> Pre-allocate the output vector to avoid race conditions
    stereo_frame.GT_right_edges.resize(stereo_frame.left_edges.size());
    
    //> Pre-allocate thread-local storage based on number of threads
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<int>> thread_local_indices_to_remove(num_threads);
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic)
        for (int left_edge_index = 0; left_edge_index < static_cast<int>(stereo_frame.left_edges.size()); ++left_edge_index)
        {
            const auto e_coeffs = epip_line_coeffs[left_edge_index];
            std::vector<Edge> right_candidate_edges = ExtractEpipolarEdges(e_coeffs, right_edges, 0.5);
            right_candidate_edges = get_right_edges_close_to_GT_location(stereo_frame.left_edges[left_edge_index], stereo_frame.GT_locations_from_disparity[left_edge_index], right_candidate_edges, 1.0);

            if (right_candidate_edges.empty()) {
                thread_local_indices_to_remove[thread_id].push_back(left_edge_index);
                continue;
            }
            
            //> Direct assignment to pre-allocated vector to prevent race condition
            stereo_frame.GT_right_edges[left_edge_index] = right_candidate_edges;
        }
    }
    
    //> Merge all thread-local indices_to_remove vectors
    for (const auto& local_indices : thread_local_indices_to_remove) {
        indices_to_remove.insert(indices_to_remove.end(), local_indices.begin(), local_indices.end());
    }

    //> Remove the left edges from the stereo_frame structure if there is no right edge correspondences close to the GT edge
    if (!indices_to_remove.empty()) 
    {
        //> First sort the indices in an descending order
        std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());
        for (size_t no_GT_index : indices_to_remove) {
            stereo_frame.left_edges.erase(stereo_frame.left_edges.begin() + no_GT_index);
        }
    }
}

StereoMatchResult get_Stereo_Edge_Pairs(const cv::Mat &left_image, const cv::Mat &right_image, StereoEdgeCorrespondencesGT prev_stereo_frame, Dataset &dataset, int idx)
{
    Utility util{};

    ///////////////////////////////FORWARD DIRECTION///////////////////////////////

    //> CH TODO: Why do we need this if we have prev_stereo_frame????
    std::vector<Edge> left_edges;
    std::vector<cv::Point2d> ground_truth_right_edges;

    // std::cout << "Froward GT data size = " << dataset.forward_gt_data.size() << std::endl;

    for (const auto &data : dataset.forward_gt_data)
    {
        left_edges.push_back(Edge{std::get<0>(data), std::get<2>(data), false, idx});
        ground_truth_right_edges.push_back(std::get<1>(data));
    }
    //============================= END OF CH TODO =====================

    auto [left_orthogonal_one, left_orthogonal_two] = CalculateOrthogonalShifts(left_edges, ORTHOGONAL_SHIFT_MAG, dataset);

    std::vector<Edge> filtered_left_edges;
    std::vector<cv::Point2d> filtered_ground_truth_right_edges;

    std::vector<cv::Mat> left_patch_set_one;
    std::vector<cv::Mat> left_patch_set_two;

    ExtractPatches(
        PATCH_SIZE,
        left_image,
        left_edges,
        left_orthogonal_one,
        left_orthogonal_two,
        filtered_left_edges,
        left_patch_set_one,
        left_patch_set_two,
        &ground_truth_right_edges,
        &filtered_ground_truth_right_edges);

    Eigen::Matrix3d fundamental_matrix_21 = dataset.get_fund_mat_21();
    Eigen::Matrix3d fundamental_matrix_12 = dataset.get_fund_mat_12();

    std::vector<Eigen::Vector3d> epipolar_lines_right = CalculateEpipolarLine(fundamental_matrix_21, filtered_left_edges);

    EdgeMatchResult forward_match = CalculateMatches(
        filtered_left_edges,
        dataset.right_edges,
        left_patch_set_one,
        left_patch_set_two,
        epipolar_lines_right,
        right_image,
        dataset,
        filtered_ground_truth_right_edges);

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

    auto [right_orthogonal_one, right_orthogonal_two] = CalculateOrthogonalShifts(reverse_primary_edges, ORTHOGONAL_SHIFT_MAG, dataset);

    std::vector<Edge> filtered_right_edges;
    std::vector<cv::Point2d> filtered_ground_truth_left_edges;

    std::vector<cv::Mat> right_patch_set_one;
    std::vector<cv::Mat> right_patch_set_two;

    ExtractPatches(
        PATCH_SIZE,
        right_image,
        reverse_primary_edges,
        right_orthogonal_one,
        right_orthogonal_two,
        filtered_right_edges,
        right_patch_set_one,
        right_patch_set_two,
        nullptr,
        nullptr);

    std::vector<Eigen::Vector3d> epipolar_lines_left = CalculateEpipolarLine(fundamental_matrix_12, filtered_right_edges);

    EdgeMatchResult reverse_match = CalculateMatches(
        filtered_right_edges,
        left_edges,
        right_patch_set_one,
        right_patch_set_two,
        epipolar_lines_left,
        left_image,
        dataset);

    std::vector<std::pair<Edge, Edge>> confirmed_matches;

    int matches_before_bct = static_cast<int>(forward_match.edge_to_cluster_matches.size());
    std::cout << "Number of matches before BCT: " << matches_before_bct << std::endl;

    auto bct_start = std::chrono::high_resolution_clock::now();

    int forward_left_index = 0;
    int bct_true_positive = 0;
    for (const auto &[left_edge, patch_match_forward] : forward_match.edge_to_cluster_matches)
    {

        const auto &right_contributing_edges = patch_match_forward.contributing_edges; // gonna be vector of edges
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

    // //> Measure the recall of bidirectional consistency test
    // int bct_true_positive = 0;
    // for (int i = 0; i < confirmed_matches.size(); i++) {
    //     ConfirmedMatchEdge right_confirmed = confirmed_matches[i].second;
    //     cv::Point2d right_edge_location = right_confirmed.location;
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
    std::cout << "Number of stacked GT right edges: " << dataset.ground_truth_right_edges_after_lowe.size() << std::endl;

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

// EdgeMatchResult FrameMatches(
//     const std::vector<Edge> &prev_edges,
//     const std::vector<Edge> &curr_edges,
//     const cv::Mat &prev_image,
//     const cv::Mat &curr_image,
//     const Dataset &dataset)
// {
//     EdgeMatchResult result;

//     // Step 1: Extract patches from previous frame edges
//     auto [prev_orthogonal_one, prev_orthogonal_two] = CalculateOrthogonalShifts(prev_edges, ORTHOGONAL_SHIFT_MAG, dataset);

//     std::vector<Edge> filtered_prev_edges;
//     std::vector<cv::Mat> prev_patch_set_one, prev_patch_set_two;

//     ExtractPatches(
//         PATCH_SIZE,
//         prev_image,
//         prev_edges,
//         prev_orthogonal_one,
//         prev_orthogonal_two,
//         filtered_prev_edges,
//         prev_patch_set_one,
//         prev_patch_set_two,
//         nullptr,
//         nullptr);

//     // Step 2: For each previous frame edge, find temporal correspondences
//     std::vector<std::pair<Edge, EdgeMatch>> forward_temporal_matches;

//     for (size_t i = 0; i < filtered_prev_edges.size(); ++i) {
//         const Edge &prev_edge = filtered_prev_edges[i];
//         const cv::Mat &prev_patch_one = prev_patch_set_one[i];
//         const cv::Mat &prev_patch_two = prev_patch_set_two[i];

//         // Step 2a: Location/Proximity Filter (temporal constraint)
//         // Assumption: dt is small, so corresponding edges should be nearby
//         std::vector<Edge> temporal_candidates;
//         double TEMPORAL_SEARCH_RADIUS = 5.0; // pixels - adjust based on expected motion

//         for (const Edge &curr_edge : curr_edges) {
//             double distance = cv::norm(prev_edge.location - curr_edge.location);
//             if (distance <= TEMPORAL_SEARCH_RADIUS) {
//                 temporal_candidates.push_back(curr_edge);
//             }
//         }

//         if (temporal_candidates.empty()) continue;

//         // Step 2b: Extract patches for current frame candidates
//         auto [curr_orthogonal_one, curr_orthogonal_two] = CalculateOrthogonalShifts(temporal_candidates, ORTHOGONAL_SHIFT_MAG, dataset);

//         std::vector<Edge> filtered_curr_candidates;
//         std::vector<cv::Mat> curr_patch_set_one, curr_patch_set_two;

//         ExtractPatches(
//             PATCH_SIZE,
//             curr_image,
//             temporal_candidates,
//             curr_orthogonal_one,
//             curr_orthogonal_two,
//             filtered_curr_candidates,
//             curr_patch_set_one,
//             curr_patch_set_two,
//             nullptr,
//             nullptr);

//         // Step 3: NCC/SIFT Threshold Filtering
//         std::vector<EdgeMatch> ncc_passed_matches;

//         for (size_t j = 0; j < filtered_curr_candidates.size(); ++j) {
//             // Compute NCC scores (similar to existing FilterByNCC logic)
//             double ncc_one = ComputeNCC(prev_patch_one, curr_patch_set_one[j]);
//             double ncc_two = ComputeNCC(prev_patch_two, curr_patch_set_two[j]);
//             double ncc_cross_one = ComputeNCC(prev_patch_one, curr_patch_set_two[j]);
//             double ncc_cross_two = ComputeNCC(prev_patch_two, curr_patch_set_one[j]);

//             double score_direct = std::min(ncc_one, ncc_two);
//             double score_cross = std::min(ncc_cross_one, ncc_cross_two);
//             double final_score = std::max(score_direct, score_cross);

//             // Apply NCC threshold
//             if (ncc_one >= NCC_THRESH_TEMPORAL && ncc_two >= NCC_THRESH_TEMPORAL) {
//                 EdgeMatch match;
//                 match.edge = filtered_curr_candidates[j];
//                 match.final_score = final_score;
//                 ncc_passed_matches.push_back(match);
//             }
//         }

//         // Step 4: Lowe's Ratio Test
//         if (ncc_passed_matches.size() >= 2) {
//             // Sort by score (descending)
//             std::sort(ncc_passed_matches.begin(), ncc_passed_matches.end(),
//                      [](const EdgeMatch &a, const EdgeMatch &b) {
//                          return a.final_score > b.final_score;
//                      });

//             double best_score = ncc_passed_matches[0].final_score;
//             double second_best_score = ncc_passed_matches[1].final_score;
//             double lowe_ratio = second_best_score / best_score;

//             if (lowe_ratio < LOWE_RATIO_THRESH) {
//                 forward_temporal_matches.emplace_back(prev_edge, ncc_passed_matches[0]);
//             }
//         }
//         else if (ncc_passed_matches.size() == 1) {
//             // Accept single good match
//             forward_temporal_matches.emplace_back(prev_edge, ncc_passed_matches[0]);
//         }
//     }

//     // Step 5: Bidirectional Consistency Test
//     std::vector<std::pair<Edge, Edge>> confirmed_temporal_matches;

//     // Reverse matching: curr_frame -> prev_frame
//     std::vector<std::pair<Edge, EdgeMatch>> reverse_temporal_matches;
//     // ... (similar process but curr->prev direction)

//     // Check bidirectional consistency
//     for (const auto &[prev_edge, forward_match] : forward_temporal_matches) {
//         const Edge &matched_curr_edge = forward_match.edge;

//         // Find reverse match for this current edge
//         for (const auto &[curr_edge_rev, reverse_match] : reverse_temporal_matches) {
//             if (cv::norm(curr_edge_rev.location - matched_curr_edge.location) <= MATCH_TOL) {
//                 const Edge &reverse_matched_prev = reverse_match.edge;

//                 // Check if reverse match points back to original prev_edge
//                 if (cv::norm(reverse_matched_prev.location - prev_edge.location) <= MATCH_TOL) {
//                     confirmed_temporal_matches.emplace_back(prev_edge, matched_curr_edge);
//                     break;
//                 }
//             }
//         }
//     }

//     // Step 6: Optional Stereo Verification
//     // If stereo pairs available, verify temporal matches using stereo consistency
//     std::vector<std::pair<Edge, Edge>> stereo_verified_matches;

//     for (const auto &[prev_edge, curr_edge] : confirmed_temporal_matches) {
//         // Find stereo correspondences for both prev_edge and curr_edge
//         // Edge prev_stereo_match = FindStereoCorrespondence(prev_edge, prev_stereo_image);
//         // Edge curr_stereo_match = FindStereoCorrespondence(curr_edge, curr_stereo_image);

//         // Verify temporal consistency in stereo space
//         // double stereo_temporal_distance = cv::norm(prev_stereo_match.location - curr_stereo_match.location);
//         // if (stereo_temporal_distance <= STEREO_TEMPORAL_THRESH) {
//         //     stereo_verified_matches.emplace_back(prev_edge, curr_edge);
//         // }
//     }

//     // Prepare result
//     result.edge_to_cluster_matches.clear();
//     for (const auto &[prev_edge, curr_edge] : confirmed_temporal_matches) {
//         EdgeMatch temporal_match;
//         temporal_match.edge = curr_edge;
//         temporal_match.final_score = 1.0; // Could store actual NCC score
//         result.edge_to_cluster_matches.emplace_back(prev_edge, temporal_match);
//     }

//     return result;
// }

//> MARK: Main Edge Pairing
EdgeMatchResult CalculateMatches(const std::vector<Edge> &selected_primary_edges, const std::vector<Edge> &secondary_edges, \
                                 const std::vector<cv::Mat> &primary_patch_set_one, const std::vector<cv::Mat> &primary_patch_set_two, \
                                 const std::vector<Eigen::Vector3d> &epipolar_lines_secondary, \
                                 const cv::Mat &secondary_image, Dataset &dataset, const std::vector<cv::Point2d> &selected_ground_truth_edges)
{
    Utility util{};

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
    std::unordered_map<Edge, std::vector<Edge>> final_lowe_matches;
    std::unordered_map<Edge, std::vector<Edge>> final_reverse_lowe_matches;

    //> CH: this is local structure of final matches
    // was std::vector<std::vector<std::pair<SourceEdge, EdgeMatch>>> local_final_matches(omp_get_max_threads());
    std::vector<std::vector<std::pair<Edge, EdgeMatch>>> local_final_matches(omp_get_max_threads());
    std::vector<std::unordered_map<Edge, std::vector<Edge>>> local_final_lowe_matches(omp_get_max_threads());
    std::vector<std::unordered_map<Edge, std::vector<Edge>>> local_final_reverse_lowe_matches(omp_get_max_threads());

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

            // if (!CheckEpipolarTangency(primary_edge, epipolar_line))
            // {
            //     continue;
            // }

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
                        0.5 // threshold
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

            // disp_output_counts.push_back(filtered_secondary_edges.size());
            local_disp_output_counts[thread_id].push_back(filtered_secondary_edges.size());

            std::vector<Edge> shifted_secondary_edge;
            EpipolarShiftFilter(filtered_secondary_edges, shifted_secondary_edge, epipolar_line, util);

            // shift_input_counts.push_back(filtered_secondary_edges.size());
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
                             per_edge_shift_precision, shifted_secondary_edge, ground_truth_edge,
                             3.0);
            }
            ///////////////////////////////EPIPOLAR CLUSTER THRESHOLD//////////////////////////
#if MEASURE_TIMINGS
            auto start_cluster = std::chrono::high_resolution_clock::now();
#endif
            local_shift_output_counts[thread_id].push_back(shifted_secondary_edge.size());
            std::vector<EdgeCluster> cluster_centers = ClusterEpipolarShiftedEdges(shifted_secondary_edge); // notice: shifted_secondary_edge will be changed inside the function but wouldn't be used later on.;
            // FormClusterCenters(cluster_centers, clusters);

            local_clust_input_counts[thread_id].push_back(shifted_secondary_edge.size());

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
                             3.0);
            }
            ///////////////////////////////EXTRACT PATCHES THRESHOLD////////////////////////////////////////////
#if MEASURE_TIMINGS
            auto start_patch = std::chrono::high_resolution_clock::now();
#endif

            // clust_output_counts.push_back(cluster_centers.size());
            local_clust_output_counts[thread_id].push_back(cluster_centers.size());

            std::vector<Edge> cluster_coords;

            for (const auto &cluster : cluster_centers)
            {
                // checked: std::cout << "Cluster center edge location: " << cluster.center_edge.location << std::endl;
                cluster_coords.push_back(cluster.center_edge);
            }

            auto [secondary_orthogonal_one, secondary_orthogonal_two] = CalculateOrthogonalShifts(
                cluster_coords,
                ORTHOGONAL_SHIFT_MAG,
                dataset);

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
                3.0);

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
                         local_final_lowe_matches,
                         local_final_reverse_lowe_matches,
                         thread_id,
                         passed_ncc_matches,
                         !selected_ground_truth_edges.empty(),
                         primary_edge,
                         ground_truth_edge,
                         lowe_true_positive,
                         lowe_false_negative,
                         per_edge_lowe_precision,
                         lowe_edges_evaluated,
                         3.0);
        } //> MARK: end of looping over left edges
    }

#if MEASURE_TIMINGS
    auto total_end = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
#endif
    cv::Point2d ground_truth_edge;
    for (int i = 0; i < omp_get_max_threads(); i++)
    {
        for (const auto &[src, candidates] : local_final_lowe_matches[i])
        {
            final_lowe_matches[src].insert(final_lowe_matches[src].end(), candidates.begin(), candidates.end());
        }
        for (const auto &[src, candidates] : local_final_reverse_lowe_matches[i])
        {
            final_reverse_lowe_matches[src].insert(final_reverse_lowe_matches[src].end(), candidates.begin(), candidates.end());
        }
    }
    for (const auto &[src, candidates] : final_lowe_matches)
    {
        if (candidates.size() == 1)
        {
            Edge tgt = candidates[0];
            final_reverse_lowe_matches.erase(tgt); // Remove reverse match if it exists, so that others will know this edge is already matched.
            final_lowe_matches.erase(src);         // Remove the source edge from final matches, as it has been matched.
        }
    }

    for (const auto &[src, candidates] : final_reverse_lowe_matches)
    {
        if (candidates.size() >= 2)
        {
            const Edge &candidates_1 = candidates[0];
            const Edge &candidates_2 = candidates[1];
            if (final_reverse_lowe_matches.find(candidates_1) != final_reverse_lowe_matches.end())
            {
                if (final_reverse_lowe_matches.find(candidates_2) == final_reverse_lowe_matches.end())
                {
                    // candidates_2 has been ruled out by other matches
                    if (1)
                    {
                        // local_GT_right_edges_after_lowe[thread_id].push_back(ground_truth_edge);
                        if (cv::norm(candidates_1.location - ground_truth_edge) <= 3.0)
                        {
                            // lowe_precision_numerator++;
                            lowe_true_positive++;
                        }
                        else
                        {
                            lowe_false_negative++;
                        }
                    }
                    EdgeMatch match;
                    match.edge = candidates_1;
                    final_matches.emplace_back(src, match);
                    lowe_output_counts.push_back(1);
                }
                else
                {
                    lowe_false_negative++;
                    lowe_output_counts.push_back(0);
                }
            }
            else
            {
                if (final_reverse_lowe_matches.find(candidates_2) != final_reverse_lowe_matches.end())
                {
                    // candidates_1 has been ruled out by other matches
                    if (!selected_ground_truth_edges.empty())
                    {
                        // local_GT_right_edges_after_lowe[thread_id].push_back(ground_truth_edge);
                        if (cv::norm(candidates_2.location - ground_truth_edge) <= 1)
                        {
                            // lowe_precision_numerator++;
                            lowe_true_positive++;
                        }
                        else
                        {
                            lowe_false_negative++;
                        }
                    }
                    EdgeMatch match;
                    match.edge = candidates_2;
                    final_matches.emplace_back(src, match);
                    lowe_output_counts.push_back(1);
                }
                else
                {
                    // still cannot decide
                    lowe_false_negative++;
                    lowe_output_counts.push_back(0);
                }
            }
        }
    }

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

    std::cout << "Epipolar Distance Recall: " << std::fixed << std::setprecision(2) << epi_distance_recall * 100 << "%" << std::endl;
    std::cout << "Max Disparity Threshold Recall: " << std::fixed << std::setprecision(2) << max_disparity_recall * 100 << "%" << std::endl;
    std::cout << "Epipolar Shift Threshold Recall: " << std::fixed << std::setprecision(2) << epi_shift_recall * 100 << "%" << std::endl;
    std::cout << "Epipolar Cluster Threshold Recall: " << std::fixed << std::setprecision(2) << epi_cluster_recall * 100 << "%" << std::endl;
    std::cout << "NCC Threshold Recall: " << std::fixed << std::setprecision(2) << ncc_recall * 100 << "%" << std::endl;
    std::cout << "LRT Threshold Recall: " << std::fixed << std::setprecision(2) << lowe_recall * 100 << "%" << std::endl;

    double per_image_epi_precision = (epi_edges_evaluated > 0) ? (per_edge_epi_precision / epi_edges_evaluated) : (0.0);
    double per_image_disp_precision = (disp_edges_evaluated > 0) ? (per_edge_disp_precision / disp_edges_evaluated) : (0.0);
    double per_image_shift_precision = (shift_edges_evaluated > 0) ? (per_edge_shift_precision / shift_edges_evaluated) : (0.0);
    double per_image_clust_precision = (clust_edges_evaluated > 0) ? (per_edge_clust_precision / clust_edges_evaluated) : (0.0);
    double per_image_ncc_precision = (ncc_edges_evaluated > 0) ? (per_edge_ncc_precision / ncc_edges_evaluated) : (0.0);
    double per_image_lowe_precision = (lowe_edges_evaluated > 0) ? (per_edge_lowe_precision / lowe_edges_evaluated) : (0.0);

    std::cout << "Epipolar Distance Precision: "
        << std::fixed << std::setprecision(2)
        << per_image_epi_precision * 100 << "%" << std::endl;
    std::cout << "Maximum Disparity Precision: "
        << std::fixed << std::setprecision(2)
        << per_image_disp_precision * 100 << "%" << std::endl;
    std::cout << "Epipolar Shift Precision: "
        << std::fixed << std::setprecision(2)
        << per_image_shift_precision * 100 << "%" << std::endl;
    std::cout << "Epipolar Cluster Precision: "
        << std::fixed << std::setprecision(2)
        << per_image_clust_precision * 100 << "%" << std::endl;
    std::cout << "NCC Precision: "
        << std::fixed << std::setprecision(2)
        << per_image_ncc_precision * 100 << "%" << std::endl;
    std::cout << "LRT Precision: "
        << std::fixed << std::setprecision(2)
        << per_image_lowe_precision * 100 << "%" << std::endl;

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
    std::cout << "Final matches size: " << final_matches.size() << std::endl;
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

/*
    Extract patches from the image based on shifted edge points and orientations.
    The function checks if the patches are within bounds before extracting them.
    It also filters edges and orientations based on the extracted patches.
*/
void ExtractPatches(
    int patch_size,
    const cv::Mat &image,
    const std::vector<Edge> &edges,
    const std::vector<cv::Point2d> &shifted_one,
    const std::vector<cv::Point2d> &shifted_two,
    std::vector<Edge> &filtered_edges_out,
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
            patch_set_one_out.push_back(patch1);
            patch_set_two_out.push_back(patch2);

            if (ground_truth_edges && filtered_gt_edges_out)
            {
                filtered_gt_edges_out->push_back((*ground_truth_edges)[i]);
            }
        }
    }
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

    bool primary_passes_tangency = (abs(angle_diff_deg - 0) > EPIP_TENGENCY_ORIENT_THRESH && abs(angle_diff_deg - 180) > EPIP_TENGENCY_ORIENT_THRESH) ? (true) : (false);
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
        bool within_horizontal = (disparity >= 0) && (disparity <= MAX_DISPARITY);
        bool within_vertical = std::abs(candidate.location.y - primary_edge.location.y) <= MAX_DISPARITY;

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

void EpipolarShiftFilter(
    const std::vector<Edge> &filtered_edges,
    std::vector<Edge> &shifted_edges,
    const Eigen::Vector3d &epipolar_line,
    Utility util)
{
    // std::vector<double> epipolar_coefficients = {epipolar_line(0), epipolar_line(1), epipolar_line(2)};

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
        int frame_idx = cluster_edges[0].frame_source;
        for (size_t j = 0; j < cluster_edges.size(); ++j)
        {
            sum_point += cluster_edges[j].location;
            sum_orientation += cluster_edges[j].orientation;
        }

        cv::Point2d avg_point = sum_point * (1.0 / cluster_edges.size());
        double avg_orientation = sum_orientation * (1.0 / cluster_edges.size());

        EdgeCluster cluster;
        cluster.center_edge = Edge{avg_point, avg_orientation, false, frame_idx};
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
    double threshold)
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

            // double score_one = std::min(ncc_one, ncc_two);
            // double score_two = std::min(ncc_three, ncc_four);
            // double final_score = std::max(score_one, score_two);
            double final_score = std::max({ncc_one, ncc_two, ncc_three, ncc_four});

#if DEBUG_COLLECT_NCC_AND_ERR
            double err_to_gt = cv::norm(filtered_cluster_centers[i].center_edge.location - ground_truth_edge);
            std::pair<double, double> pair_ncc_one_err(err_to_gt, ncc_one);
            std::pair<double, double> pair_ncc_two_err(err_to_gt, ncc_two);
            ncc_one_vs_err.push_back(pair_ncc_one_err);
            ncc_two_vs_err.push_back(pair_ncc_two_err);
#endif
            if (ncc_one >= NCC_THRESH_STRONG_BOTH_SIDES && ncc_two >= NCC_THRESH_STRONG_BOTH_SIDES)
            {
                EdgeMatch info;
                info.edge = filtered_cluster_centers[i].center_edge;
                info.final_score = final_score;
                info.contributing_edges = filtered_cluster_centers[i].contributing_edges;
                passed_ncc_matches.push_back(info);

                if (gt)
                {
                    if (cv::norm(filtered_cluster_centers[i].center_edge.location - ground_truth_edge) <= threshold)
                    {

                        ncc_match_found = true;
                        ncc_precision_numerator++;
                    }
                }
            }
            else if (ncc_one >= NCC_THRESH_STRONG_ONE_SIDE || ncc_two >= NCC_THRESH_STRONG_ONE_SIDE)
            {
                EdgeMatch info;
                info.edge = filtered_cluster_centers[i].center_edge;
                info.final_score = final_score;
                info.contributing_edges = filtered_cluster_centers[i].contributing_edges;
                passed_ncc_matches.push_back(info);

                if (gt)
                {
                    if (cv::norm(filtered_cluster_centers[i].center_edge.location - ground_truth_edge) <= threshold)
                    {
                        ncc_match_found = true;
                        ncc_precision_numerator++;
                    }
                }
            }
            else if (ncc_one >= NCC_THRESH_WEAK_BOTH_SIDES && ncc_two >= NCC_THRESH_WEAK_BOTH_SIDES && filtered_cluster_centers.size() == 1)
            {
                EdgeMatch info;
                info.edge = filtered_cluster_centers[i].center_edge;
                info.final_score = final_score;
                info.contributing_edges = filtered_cluster_centers[i].contributing_edges;
                passed_ncc_matches.push_back(info);

                if (gt)
                {
                    if (cv::norm(filtered_cluster_centers[i].center_edge.location - ground_truth_edge) <= threshold)
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
}

void FilterByLowe(
    std::vector<std::vector<std::pair<Edge, EdgeMatch>>> &local_final_matches,
    std::vector<std::vector<int>> &local_lowe_input_counts,
    std::vector<std::vector<int>> &local_lowe_output_counts,
    std::vector<std::vector<cv::Point2d>> &local_GT_right_edges_after_lowe,
    std::vector<std::unordered_map<Edge, std::vector<Edge>>> &local_final_lowe_matches,
    std::vector<std::unordered_map<Edge, std::vector<Edge>>> &local_final_reverse_lowe_matches,
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
        Edge source_edge = primary_edge;
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

            local_final_lowe_matches[thread_id][source_edge].push_back(best_match.edge);
            local_final_reverse_lowe_matches[thread_id][best_match.edge].push_back(source_edge);

            local_final_matches[thread_id].emplace_back(source_edge, best_match); // already pushed into the final matches
            local_lowe_output_counts[thread_id].push_back(1);
        }
        else
        {
            local_final_lowe_matches[thread_id][source_edge].push_back(best_match.edge);
            local_final_reverse_lowe_matches[thread_id][best_match.edge].push_back(source_edge);

            local_final_lowe_matches[thread_id][source_edge].push_back(second_best_match.edge);
            local_final_reverse_lowe_matches[thread_id][second_best_match.edge].push_back(source_edge);

            // lowe_false_negative++;  --> it should be incremented when really needing it to be incremented
            // lowe_output_counts.push_back(0);
            // local_lowe_output_counts[thread_id].push_back(0);
        }
    }
    else if (passed_ncc_matches.size() == 1)
    {
        best_match = passed_ncc_matches[0];

        if (gt)
        {
            local_GT_right_edges_after_lowe[thread_id].push_back(ground_truth_edge);
            if (cv::norm(best_match.edge.location - ground_truth_edge) <= 3.0)
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

        local_final_lowe_matches[thread_id][source_edge].push_back(best_match.edge);
        local_final_reverse_lowe_matches[thread_id][best_match.edge].push_back(source_edge);

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

#if MEASURE_TIMINGS
    time_lowe_edges_evaluated++;
    auto end_lowe = std::chrono::high_resolution_clock::now();
    time_lowe += std::chrono::duration<double, std::milli>(end_lowe - start_lowe).count();
#endif
    per_edge_lowe_precision += (static_cast<double>(lowe_precision_numerator) > 0) ? 1.0 : 0.0;

    if (!passed_ncc_matches.empty())
    {
        lowe_edges_evaluated++;
    }
}