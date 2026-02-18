#include "Matches.h"
#include <unordered_map>
#include <unordered_set>

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

std::pair<cv::Point2d, cv::Point2d> get_Orthogonal_Shifted_Points(const Edge edgel, double shift_magnitude)
{
    double shifted_x1 = edgel.location.x + shift_magnitude * (std::sin(edgel.orientation));
    double shifted_y1 = edgel.location.y + shift_magnitude * (-std::cos(edgel.orientation));
    double shifted_x2 = edgel.location.x + shift_magnitude * (-std::sin(edgel.orientation));
    double shifted_y2 = edgel.location.y + shift_magnitude * (std::cos(edgel.orientation));

    cv::Point2d shifted_point_plus(shifted_x1, shifted_y1);
    cv::Point2d shifted_point_minus(shifted_x2, shifted_y2);

    return {shifted_point_plus, shifted_point_minus};
}

void get_patch_on_one_edge_side(cv::Point2d shifted_point, double theta,
                                cv::Mat &patch_coord_x, cv::Mat &patch_coord_y,
                                cv::Mat &patch_val, const cv::Mat img)
{
    const int half_patch_size = floor(PATCH_SIZE / 2);
    for (int i = -half_patch_size; i <= half_patch_size; i++)
    {
        for (int j = -half_patch_size; j <= half_patch_size; j++)
        {
            //> get the rotated coordinate
            cv::Point2d rotated_point(cos(theta) * (i)-sin(theta) * (j) + shifted_point.x, sin(theta) * (i) + cos(theta) * (j) + shifted_point.y);
            patch_coord_x.at<double>(i + half_patch_size, j + half_patch_size) = rotated_point.x;
            patch_coord_y.at<double>(i + half_patch_size, j + half_patch_size) = rotated_point.y;

            //> get the image intensity of the rotated coordinate
            // double interp_val = Bilinear_Interpolation<double>(img, rotated_point);
            double interp_val = Bilinear_Interpolation<double>(img, rotated_point);
            patch_val.at<double>(i + half_patch_size, j + half_patch_size) = interp_val;
        }
    }
}

double get_patch_similarity(const cv::Mat patch_one, const cv::Mat patch_two)
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

std::pair<cv::Mat, cv::Mat> get_edge_patches(const Edge edge, const cv::Mat img, bool b_debug)
{
    Utility util{};

    std::pair<cv::Point2d, cv::Point2d> shifted_points = get_Orthogonal_Shifted_Points(edge);
    cv::Mat patch_val_plus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_val_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_x_plus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_y_plus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_x_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    cv::Mat patch_coord_y_minus = cv::Mat_<double>(PATCH_SIZE, PATCH_SIZE);
    get_patch_on_one_edge_side(shifted_points.first, edge.orientation, patch_coord_x_plus, patch_coord_y_plus, patch_val_plus, img);
    get_patch_on_one_edge_side(shifted_points.second, edge.orientation, patch_coord_x_minus, patch_coord_y_minus, patch_val_minus, img);

    if (b_debug)
    {
        std::cout << "Edge location: " << edge.location << std::endl;
        std::cout << "Edge orientation: " << edge.orientation << std::endl;
        std::cout << "Patch (+) coordinates: " << std::endl;
        std::cout << patch_coord_x_plus << std::endl;
        std::cout << patch_coord_y_plus << std::endl;
        std::cout << "Patch (-) coordinates: " << std::endl;
        std::cout << patch_coord_x_minus << std::endl;
        std::cout << patch_coord_y_minus << std::endl;
    }

    if (patch_val_plus.type() != CV_32F)
        patch_val_plus.convertTo(patch_val_plus, CV_32F);
    if (patch_val_minus.type() != CV_32F)
        patch_val_minus.convertTo(patch_val_minus, CV_32F);

    return {patch_val_plus, patch_val_minus};
}
double get_similarity(const cv::Mat &patch_one, const cv::Mat &patch_two)
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
Edge shift_Edge_to_Epipolar_Line(Edge original_edge, const Eigen::Vector3d epipolar_line_coeffs)
{
    Utility util{};
    Eigen::Vector3d xy1_edge(original_edge.location.x, original_edge.location.y, 1.0);
    double corrected_x, corrected_y, corrected_theta;
    double epiline_x, epiline_y;

    if (util.getNormalDistance2EpipolarLine(epipolar_line_coeffs, xy1_edge, epiline_x, epiline_y) < LOCATION_PERTURBATION)
    {
        //> If normal distance is small, move directly to the epipolar line
        cv::Point2d corrected_edge_loc(epiline_x, epiline_y);

        //> CH TODO: Pass the frame ID to the last input argument
        return Edge{corrected_edge_loc, original_edge.orientation, false, 0};
    }
    else
    {
        double x_intersection, y_intersection;
        Eigen::Vector3d isolated_edge(original_edge.location.x, original_edge.location.y, original_edge.orientation);

        //> Inner two cases:
        if (util.getTangentialDistance2EpipolarLine(epipolar_line_coeffs, isolated_edge, x_intersection, y_intersection) < EPIP_TANGENCY_DISPL_THRESH)
        {
            //> (i) if the displacement after epipolar shift is less than EPIP_TANGENCY_DISPL_THRESH, then feel free to shift it along its direction vector
            cv::Point2d corrected_edge_loc(x_intersection, y_intersection);
            // b_pass_epipolar_tengency_check = true;
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
            if (p_theta > 0 && derivative_p_theta < 0)
                theta -= ORIENT_PERTURBATION;
            else if (p_theta < 0 && derivative_p_theta < 0)
                theta -= ORIENT_PERTURBATION;
            else if (p_theta > 0 && derivative_p_theta > 0)
                theta += ORIENT_PERTURBATION;
            else if (p_theta < 0 && derivative_p_theta > 0)
                theta += ORIENT_PERTURBATION;

            //> Calculate the intersection between the tangent and epipolar line
            Eigen::Vector3d isolated_edge_(original_edge.location.x, original_edge.location.y, theta);
            if (util.getTangentialDistance2EpipolarLine(epipolar_line_coeffs, isolated_edge_, x_intersection, y_intersection) < EPIP_TANGENCY_DISPL_THRESH)
            {
                cv::Point2d corrected_edge_loc(x_intersection, y_intersection);
                // b_pass_epipolar_tengency_check = true;
                //> CH TODO: Pass the frame ID to the last input argument
                return Edge{corrected_edge_loc, theta, false, 0};
            }
            else
            {
                //> If the edge does not pass the epipolar tengency test, then return the original edge
                // b_pass_epipolar_tengency_check = false;

                //> CH TODO: Pass the frame ID to the last input argument
                return Edge{original_edge.location, original_edge.orientation, false, 0};
            }
        }
    }
}

std::vector<int> extract_Epipolar_Edge_Indices(const Eigen::Vector3d &epipolar_line, const std::vector<Edge> &edges, const double dist_tol)
{
    int edge_index = 0;
    std::vector<int> extracted_indices;
    for (const auto &e : edges)
    {
        double x = e.location.x;
        double y = e.location.y;
        double dist_to_epip_line = std::abs(epipolar_line(0) * x + epipolar_line(1) * y + epipolar_line(2)) / std::sqrt((epipolar_line(0) * epipolar_line(0)) + (epipolar_line(1) * epipolar_line(1)));

        if (dist_to_epip_line < dist_tol)
        {
            extracted_indices.push_back(edge_index);
        }
        edge_index++;
    }

    return extracted_indices;
}

std::vector<int> get_right_edge_indices_close_to_GT_location(const StereoFrame &stereo_frame, const cv::Point2d GT_location, double GT_orientation,
                                                             const std::vector<int> right_candidate_edge_indices,
                                                             const double dist_tol, const double orient_tol, bool is_left)
{
    std::vector<int> right_edge_indices_close_to_GT_location;
    for (const auto &right_candidate_edge_index : right_candidate_edge_indices) // could be left candidates in the case of right edges
    {
        //> Check if the right candidate edge location is close to the GT location
        cv::Point2d edge_loc = is_left ? stereo_frame.right_edges[right_candidate_edge_index].location : stereo_frame.left_edges[right_candidate_edge_index].location;
        if (cv::norm(GT_location - edge_loc) < dist_tol)
        {
            //> Check if the right candidate edge orientation is close to the corresponding left edge orientation
            double edge_orientation = is_left ? stereo_frame.right_edges[right_candidate_edge_index].orientation : stereo_frame.left_edges[right_candidate_edge_index].orientation;
            if (std::abs(rad_to_deg<double>(edge_orientation) - rad_to_deg<double>(GT_orientation)) < orient_tol)
            {
                right_edge_indices_close_to_GT_location.push_back(right_candidate_edge_index);
            }
        }
    }
    return right_edge_indices_close_to_GT_location;
}

/*
    Pick random edges(return the indices) from the valid edges that can find their GT correspondences.
*/
std::vector<int> PickRandomEdges(int size, const int num_points)
{

    int num_points_clamped = std::min(num_points, size);
    std::vector<int> selected_indices;
    std::unordered_set<int> used_indices;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, size - 1);

    while (selected_indices.size() < num_points_clamped)
    {
        int index = dis(gen);
        if (used_indices.find(index) == used_indices.end())
        {
            used_indices.insert(index);
            selected_indices.push_back(index);
        }
    }

    return selected_indices;
}

void Find_Stereo_GT_Locations(Dataset &dataset, const cv::Mat left_disparity_map, const StereoFrame &stereo_frame, Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left)
{
    Utility util;
    // std::cout << "Disparity map type: " << util.cvMat_Type(left_disparity_map.type()) << std::endl;
    int edge_size = is_left ? stereo_frame.left_edges.size() : stereo_frame.right_edges.size();
    for (int left_edge_index = 0; left_edge_index < edge_size; left_edge_index++)
    {
        Edge e = stereo_frame.left_edges[left_edge_index]; //> we are abuseing the term left_edges here, it's universal for both left and right edges
        is_left ? e = e : e = stereo_frame.right_edges[left_edge_index];
        //> omit if the edge orientation is close to 0, pi, or -pi
        if (std::abs(rad_to_deg<double>(e.orientation)) < 4 || std::abs(rad_to_deg<double>(e.orientation) - 180.0) < 4 || std::abs(rad_to_deg<double>(e.orientation) + 180.0) < 4)
        {
            continue;
        }

        //> Use bilinear interpolation for sub-pixel accurate disparity

        double disparity = Bilinear_Interpolation<float>(left_disparity_map, e.location);

        if (std::isnan(disparity) || std::isinf(disparity) || disparity < 0)
        {
            continue;
        }
        disparity = is_left ? disparity : -disparity; //> in case we need to adjust disparity for right edges in the future
        cv::Point2d GT_location(e.location.x - disparity, e.location.y);

        //> insert data to the stereo_frame_edge_pairs structure
        stereo_frame_edge_pairs.focused_edge_indices.push_back(left_edge_index);
        stereo_frame_edge_pairs.GT_locations_from_left_edges.push_back(GT_location);

        //> Convert cv::Point2d to Eigen::Vector3d
        Eigen::Vector3d e_location_eigen(e.location.x, e.location.y, 1.0);
        Eigen::Vector3d GT_location_eigen(GT_location.x, GT_location.y, 1.0);

        double focal_length, baseline;
        Eigen::Matrix3d calib_matrix;
        if (is_left)
        {
            focal_length = dataset.get_left_focal_length();
            baseline = dataset.get_left_baseline();
            calib_matrix = dataset.get_left_calib_matrix();
        }
        else
        {
            focal_length = dataset.get_right_focal_length();
            baseline = dataset.get_right_baseline();
            calib_matrix = dataset.get_right_calib_matrix();
        }

        double rho = focal_length * baseline / disparity;
        double rho_1 = (rho < 0.0) ? (-rho) : (rho);

        Eigen::Vector3d gamma_1 = calib_matrix.inverse() * e_location_eigen;
        Eigen::Vector3d Gamma_1 = rho_1 * gamma_1;
        stereo_frame_edge_pairs.Gamma_in_left_cam_coord.push_back(Gamma_1);
    }
}

void Find_Stereo_GT_Locations(Dataset &dataset, const std::vector<double> &edge_disparities, const StereoFrame &stereo_frame, Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left)
{
    Utility util{};
    for (int left_edge_index = 0; left_edge_index < stereo_frame.left_edges.size(); left_edge_index++)
    {
        Edge e = stereo_frame.left_edges[left_edge_index];

        //> omit if the edge orientation is close to 0, pi, or -pi
        if (std::abs(rad_to_deg<double>(e.orientation)) < 4 || std::abs(rad_to_deg<double>(e.orientation) - 180.0) < 4 || std::abs(rad_to_deg<double>(e.orientation) + 180.0) < 4)
        {
            continue;
        }

        double disparity = edge_disparities[left_edge_index];
        assert(!(std::isnan(disparity) || std::isinf(disparity) || disparity < 0.0));

        cv::Point2d GT_location(e.location.x - disparity, e.location.y);

        //> insert data to the stereo_frame_edge_pairs structure
        stereo_frame_edge_pairs.focused_edge_indices.push_back(left_edge_index);
        stereo_frame_edge_pairs.GT_locations_from_left_edges.push_back(GT_location);

        //> Convert cv::Point2d to Eigen::Vector3d
        Eigen::Vector3d e_location_eigen(e.location.x, e.location.y, 1.0);
        Eigen::Vector3d GT_location_eigen(GT_location.x, GT_location.y, 1.0);

        double rho = dataset.get_left_focal_length() * dataset.get_left_baseline() / disparity;
        double rho_1 = (rho < 0.0) ? (-rho) : (rho);

        Eigen::Vector3d gamma_1 = dataset.get_left_calib_matrix().inverse() * e_location_eigen;
        Eigen::Vector3d Gamma_1 = rho_1 * gamma_1;

        stereo_frame_edge_pairs.Gamma_in_left_cam_coord.push_back(Gamma_1);
    }
}

void get_Stereo_Edge_GT_Pairs(Dataset &dataset, const StereoFrame &stereo_frame, Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left)
{
    Eigen::Matrix3d fund_mat = is_left ? dataset.get_fund_mat_21() : dataset.get_fund_mat_12();
    std::vector<Eigen::Vector3d> epip_line_coeffs = CalculateEpipolarLine(fund_mat, stereo_frame_edge_pairs.get_focused_edges());
    std::vector<int> indices_to_remove;

    //> Pre-allocate the output vector to avoid race conditions
    stereo_frame_edge_pairs.veridical_right_edges_indices.resize(epip_line_coeffs.size());

    //> Pre-allocate thread-local storage based on number of threads
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<int>> thread_local_indices_to_remove(num_threads);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

#pragma omp for schedule(dynamic)
        for (int i = 0; i < stereo_frame_edge_pairs.focused_edge_indices.size(); i++)
        {
            Edge left_edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i);
            const auto e_coeffs = epip_line_coeffs[i];
            const std::vector<Edge> &candidate_edges = is_left ? stereo_frame.right_edges : stereo_frame.left_edges;
            std::vector<int> right_candidate_edge_indices = extract_Epipolar_Edge_Indices(e_coeffs, candidate_edges, 0.5);
            right_candidate_edge_indices = get_right_edge_indices_close_to_GT_location(stereo_frame, stereo_frame_edge_pairs.GT_locations_from_left_edges[i], left_edge.orientation, right_candidate_edge_indices, 1.0, 5.0, is_left);

            if (right_candidate_edge_indices.empty())
            {
                thread_local_indices_to_remove[thread_id].push_back(i);
            }

            //> Direct assignment to pre-allocated vector to prevent race condition
            stereo_frame_edge_pairs.veridical_right_edges_indices[i] = right_candidate_edge_indices;
        }
    }

    //> Merge all thread-local indices_to_remove vectors
    for (const auto &local_indices : thread_local_indices_to_remove)
    {
        indices_to_remove.insert(indices_to_remove.end(), local_indices.begin(), local_indices.end());
    }

    //> Remove the left edges from the stereo_frame structure if there is no right edge correspondences close to the GT edge
    if (!indices_to_remove.empty())
    {
        //> First sort the indices in an descending order
        std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());
        for (size_t no_GT_index : indices_to_remove)
        {
            stereo_frame_edge_pairs.focused_edge_indices.erase(stereo_frame_edge_pairs.focused_edge_indices.begin() + no_GT_index);

            //> Also remove the corresponding 3D points and GT location from disparity to make the size of the vectors consistent
            stereo_frame_edge_pairs.Gamma_in_left_cam_coord.erase(stereo_frame_edge_pairs.Gamma_in_left_cam_coord.begin() + no_GT_index);
            stereo_frame_edge_pairs.GT_locations_from_left_edges.erase(stereo_frame_edge_pairs.GT_locations_from_left_edges.begin() + no_GT_index);
            stereo_frame_edge_pairs.veridical_right_edges_indices.erase(stereo_frame_edge_pairs.veridical_right_edges_indices.begin() + no_GT_index);
        }
    }
}

void Evaluate_Stereo_Edge_Correspondences(                                                                                                                                          /* Inputs */
                                          Stereo_Edge_Pairs &stereo_frame_edge_pairs, size_t frame_idx, const std::string &stage_name,                                              /* Outputs */
                                          double &recall_per_image, double &precision_per_image, double &precision_pair_per_image, double &num_of_target_edges_per_source_edge_avg, /* Optional Inputs */
                                          Evaluation_Statistics &evaluation_statistics, bool b_store_FN = false, bool b_store_photo_refine_statistics = false)
{
    //> For each left edge from the keyframe, see if the filtered edges are found in the pool of matched veridical edges on the current frame
    int total_num_of_true_positives_for_recall = 0;
    int total_num_of_true_positives_for_precision = 0;
    std::vector<double> num_of_target_edges_per_source_edge;
    std::vector<double> precision_per_edge;
    std::vector<double> precision_pair_edge;
    num_of_target_edges_per_source_edge.reserve(stereo_frame_edge_pairs.focused_edge_indices.size()); //> compailable to both left and right edges
    precision_per_edge.reserve(stereo_frame_edge_pairs.focused_edge_indices.size());
    for (int i = 0; i < stereo_frame_edge_pairs.focused_edge_indices.size(); i++)
    {
        //> Find if there is at least one edge in the matching_edge_clusters is found in the veridical_right_edges_indices
        total_num_of_true_positives_for_precision = 0;

        if (stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters.size() > 0)
        {
            int total_num_of_clusters = stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters.size();
            int j = 0;
            for (const auto &matching_edge_cluster : stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters)
            {
                // evaluation_statistics.focused_edge_indices.push_back(i);
                evaluation_statistics.edge_clusters_in_each_step[stage_name].push_back(matching_edge_cluster);
                evaluation_statistics.edge_clusters_in_each_step[stage_name].back().paired_left_edge_index = i;
                if (cv::norm(matching_edge_cluster.center_edge.location - stereo_frame_edge_pairs.GT_locations_from_left_edges[i]) <= DIST_TO_GT_THRESH)
                {
                    total_num_of_true_positives_for_precision++;
                    evaluation_statistics.edge_clusters_in_each_step[stage_name].back().b_is_TP = true;
                }
                else
                {
                    evaluation_statistics.edge_clusters_in_each_step[stage_name].back().b_is_TP = false;
                }

                //> Optional: Store the refine_final_scores, refine_confidences, and refine_validities of the true positive edge clusters
                if (b_store_photo_refine_statistics)
                {
                    evaluation_statistics.refine_final_scores.push_back(stereo_frame_edge_pairs.matching_edge_clusters[i].refine_final_scores[j]);
                    evaluation_statistics.refine_confidences.push_back(stereo_frame_edge_pairs.matching_edge_clusters[i].refine_confidences[j]);
                    evaluation_statistics.refine_validities.push_back(stereo_frame_edge_pairs.matching_edge_clusters[i].refine_validities[j]);
                }
                j++;
            }
            if (total_num_of_true_positives_for_precision > 0)
            {
                total_num_of_true_positives_for_recall++;
            }
            precision_per_edge.push_back(static_cast<double>(total_num_of_true_positives_for_precision) / static_cast<double>(total_num_of_clusters));
            precision_pair_edge.push_back(static_cast<double>(total_num_of_true_positives_for_precision) / static_cast<double>(total_num_of_clusters));
            num_of_target_edges_per_source_edge.push_back(total_num_of_clusters);
        }
        else
        {
            precision_per_edge.push_back(0.0);
        }

        if (b_store_FN)
        {
            if (total_num_of_true_positives_for_precision == 0)
            {
                int FN_right_edge_cluster_index = -1;
                double min_dist_error_to_GT = std::numeric_limits<double>::max();
                for (int j = 0; j < stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters.size(); j++)
                {
                    const auto &matching_edge_cluster = stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters[j];
                    //> store only the closest edge to the GT edge
                    double dist_error = cv::norm(matching_edge_cluster.center_edge.location - stereo_frame_edge_pairs.GT_locations_from_left_edges[i]);
                    if (dist_error < min_dist_error_to_GT)
                    {
                        min_dist_error_to_GT = dist_error;
                        FN_right_edge_cluster_index = j;
                    }
                }
                //> A check to bypass left edges that do not have any right edge cluster correspondences
                if (FN_right_edge_cluster_index >= 0)
                {
                    evaluation_statistics.FN_dist_error_to_GT.push_back(min_dist_error_to_GT);
                    evaluation_statistics.false_negative_edge_clusters.push_back(stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters[FN_right_edge_cluster_index]);
                    evaluation_statistics.edge_clusters_in_each_step["Epipolar Shift and Clustering"].back().paired_left_edge_index = i;
                }
            }
        }
    }

    recall_per_image = static_cast<double>(total_num_of_true_positives_for_recall) / stereo_frame_edge_pairs.focused_edge_indices.size();
    precision_per_image = std::accumulate(precision_per_edge.begin(), precision_per_edge.end(), 0.0) / precision_per_edge.size();
    precision_pair_per_image = std::accumulate(precision_pair_edge.begin(), precision_pair_edge.end(), 0.0) / precision_pair_edge.size();
    num_of_target_edges_per_source_edge_avg = std::accumulate(num_of_target_edges_per_source_edge.begin(), num_of_target_edges_per_source_edge.end(), 0.0) / num_of_target_edges_per_source_edge.size();

    std::cout << "Stereo Edge Correspondences Evaluation: Stage: " << stage_name << " | Frame: " << frame_idx << std::endl;
    std::cout << "- Recall rate:       " << std::fixed << std::setprecision(8) << recall_per_image << std::endl;
    std::cout << "- Precision rate:    " << std::fixed << std::setprecision(8) << precision_per_image << std::endl;
    std::cout << "- Precision pair rate:    " << std::fixed << std::setprecision(8) << precision_pair_per_image << std::endl;
    std::cout << "- Average ambiguity: " << std::fixed << std::setprecision(8) << num_of_target_edges_per_source_edge_avg << std::endl;
    std::cout << "========================================================\n"
              << std::endl;
}

void apply_Epipolar_Line_Distance_Filtering(Stereo_Edge_Pairs &stereo_frame_edge_pairs, Dataset &dataset, const std::vector<Edge> right_edges, const std::string &output_dir = "", bool is_left = true, size_t frame_idx = 0, int num_random_edges_for_distribution = 0)
{
    // be aware that left_edges may not be actually left edges, depends on is_left

    std::vector<Eigen::Vector3d> epip_line_coeffs = CalculateEpipolarLine(dataset.get_fund_mat_21(), stereo_frame_edge_pairs.get_focused_edges());
    epip_line_coeffs = is_left ? epip_line_coeffs : CalculateEpipolarLine(dataset.get_fund_mat_12(), stereo_frame_edge_pairs.get_focused_edges());

    stereo_frame_edge_pairs.epip_line_coeffs_of_left_edges = epip_line_coeffs; // actually not always left edges, depends on is_left

    //> Pre-allocate the output vector to avoid race conditions
    stereo_frame_edge_pairs.matching_edge_clusters.resize(epip_line_coeffs.size());

    //> Storage for epipolar distances
    std::vector<double> epipolar_distances;
    std::vector<int> is_veridical;

    //> Randomly select left edges for recording ALL right edge distances (before filtering)
    std::vector<int> random_focused_edge_indices;
    std::unordered_set<int> random_left_edge_set;
    if (num_random_edges_for_distribution > 0)
    {
        random_focused_edge_indices = is_left ? PickRandomEdges(stereo_frame_edge_pairs.focused_edge_indices.size(), num_random_edges_for_distribution) : PickRandomEdges(stereo_frame_edge_pairs.candidate_edge_indices.size(), num_random_edges_for_distribution);
        random_left_edge_set.insert(random_focused_edge_indices.begin(), random_focused_edge_indices.end());
        std::cout << "Selected " << random_focused_edge_indices.size() << " random left edges for full distribution recording" << std::endl;
    }

    //> Pre-allocate thread-local storage based on number of threads
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<int>> thread_local_indices_to_remove(num_threads);

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        //> Thread-local storage for epipolar distances
        std::vector<double> thread_local_epipolar_distances;
        std::vector<int> thread_local_is_veridical;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < stereo_frame_edge_pairs.focused_edge_indices.size(); i++)
        {
            const auto e_coeffs = epip_line_coeffs[i];
            const std::vector<Edge> &candidate_edges = is_left ? stereo_frame_edge_pairs.stereo_frame->right_edges : stereo_frame_edge_pairs.stereo_frame->left_edges;
            std::vector<int> right_candidate_edge_indices = extract_Epipolar_Edge_Indices(e_coeffs, candidate_edges, 0.5);

            //> Record epipolar distances for randomly selected left edges (ALL right edges)
            if (random_left_edge_set.find(i) != random_left_edge_set.end())
            {
                //> Record for ALL right edges to show pre-filtering distribution
                for (size_t right_edge_idx = 0; right_edge_idx < candidate_edges.size(); ++right_edge_idx)
                {
                    const auto &right_edge = is_left ? stereo_frame_edge_pairs.stereo_frame->right_edges[right_edge_idx] : stereo_frame_edge_pairs.stereo_frame->left_edges[right_edge_idx];
                    double x = right_edge.location.x;
                    double y = right_edge.location.y;
                    double dist_to_epip_line = std::abs(e_coeffs(0) * x + e_coeffs(1) * y + e_coeffs(2)) /
                                               std::sqrt((e_coeffs(0) * e_coeffs(0)) + (e_coeffs(1) * e_coeffs(1)));
                    thread_local_epipolar_distances.push_back(dist_to_epip_line);

                    //> Check if this edge is veridical
                    bool is_gt = std::find(stereo_frame_edge_pairs.veridical_right_edges_indices[i].begin(),
                                           stereo_frame_edge_pairs.veridical_right_edges_indices[i].end(),
                                           right_edge_idx) != stereo_frame_edge_pairs.veridical_right_edges_indices[i].end();
                    thread_local_is_veridical.push_back(is_gt ? 1 : 0);
                }
            }

            //> each right candidate edge is a cluster
            std::vector<EdgeCluster> right_candidate_edge_clusters;
            for (const auto &right_candidate_edge_index : right_candidate_edge_indices)
            {
                EdgeCluster right_candidate_edge_cluster;
                right_candidate_edge_cluster.center_edge = is_left ? stereo_frame_edge_pairs.stereo_frame->right_edges[right_candidate_edge_index]
                                                                   : stereo_frame_edge_pairs.stereo_frame->left_edges[right_candidate_edge_index];
                is_left ? right_candidate_edge_cluster.contributing_edges.push_back(stereo_frame_edge_pairs.stereo_frame->right_edges[right_candidate_edge_index])
                        : right_candidate_edge_cluster.contributing_edges.push_back(stereo_frame_edge_pairs.stereo_frame->left_edges[right_candidate_edge_index]);
                right_candidate_edge_cluster.contributing_edges_toed_indices.push_back(right_candidate_edge_index);
                right_candidate_edge_clusters.push_back(right_candidate_edge_cluster);
            }
            stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters = right_candidate_edge_clusters;
        }

#pragma omp critical
        {
            epipolar_distances.insert(epipolar_distances.end(), thread_local_epipolar_distances.begin(), thread_local_epipolar_distances.end());
            is_veridical.insert(is_veridical.end(), thread_local_is_veridical.begin(), thread_local_is_veridical.end());
        }
    }

    //> Record epipolar distance distribution if output directory is provided
    if (!output_dir.empty() && !epipolar_distances.empty())
    {
        record_Filter_Distribution("epipolar_distance", epipolar_distances, is_veridical, output_dir, frame_idx);
    }

    //> Record ambiguity distribution
    record_Ambiguity_Distribution("epipolar", stereo_frame_edge_pairs, output_dir, frame_idx);
}

void record_Filter_Distribution(const std::string &filter_name, const std::vector<double> &filter_values, const std::vector<int> &is_veridical, const std::string &output_dir, size_t frame_idx)
{
    //> Create filename with filter name and frame index
    std::string filename = output_dir + "/" + filter_name + "_frame_" + std::to_string(frame_idx) + ".txt";

    std::ofstream outfile(filename);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        std::cerr << "Make sure the directory '" << output_dir << "' exists and is writable." << std::endl;
        return;
    }

    //> Count veridical and non-veridical
    int num_veridical = std::count(is_veridical.begin(), is_veridical.end(), 1);
    int num_non_veridical = is_veridical.size() - num_veridical;

    //> Write header
    outfile << "# " << filter_name << " distribution for frame " << frame_idx << std::endl;
    outfile << "# Total values: " << filter_values.size() << " (Veridical: " << num_veridical << ", Non-veridical: " << num_non_veridical << ")" << std::endl;
    outfile << "filter_value\tis_GT" << std::endl;

    //> Write each filter value with veridical flag
    for (size_t i = 0; i < filter_values.size(); ++i)
    {
        outfile << filter_values[i] << "\t" << is_veridical[i] << std::endl;
    }

    outfile.close();
    std::cout << "Recorded " << filter_values.size() << " " << filter_name << " values (" << num_veridical << " GT, " << num_non_veridical << " non-GT) to: " << filename << std::endl;
}

void record_Ambiguity_Distribution(const std::string &stage_name, const Stereo_Edge_Pairs &stereo_frame_edge_pairs, const std::string &output_dir, size_t frame_idx)
{
    if (output_dir.empty())
        return;

    //> Create filename with stage name and frame index
    std::string filename = output_dir + "/ambiguity_" + stage_name + "_frame_" + std::to_string(frame_idx) + ".txt";

    std::ofstream outfile(filename);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    //> Collect ambiguity values (number of candidates per edge)
    std::vector<int> ambiguity_values;
    for (size_t i = 0; i < stereo_frame_edge_pairs.matching_edge_clusters.size(); ++i)
    {
        int num_candidates = stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters.size();
        ambiguity_values.push_back(num_candidates);
    }

    //> Write header
    outfile << "# Ambiguity distribution for stage: " << stage_name << " | Frame: " << frame_idx << std::endl;
    outfile << "# Total edges: " << ambiguity_values.size() << std::endl;
    outfile << "num_candidates" << std::endl;

    //> Write each ambiguity value
    for (int num_candidates : ambiguity_values)
    {
        outfile << num_candidates << std::endl;
    }

    outfile.close();
    std::cout << "Recorded ambiguity for " << ambiguity_values.size() << " edges to: " << filename << std::endl;
}

void apply_Disparity_Filtering(Stereo_Edge_Pairs &stereo_frame_edge_pairs, const std::string &output_dir = "", size_t frame_idx = 0, bool is_left = true)
{
    //> Storage for location errors (distance to GT)
    std::vector<double> location_errors;
    std::vector<int> is_veridical;

#pragma omp parallel
    {
        //> Thread-local storage for location errors
        std::vector<double> thread_local_location_errors;
        std::vector<int> thread_local_is_veridical;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(stereo_frame_edge_pairs.focused_edge_indices.size()); ++i)
        {
            std::vector<EdgeCluster> surviving_edge_clusters;
            Edge left_edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i);
            cv::Point2d GT_location = stereo_frame_edge_pairs.GT_locations_from_left_edges[i];

            for (auto const &edge_cluster : stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters)
            {
                Edge right_edge = edge_cluster.center_edge;

                //> Record location error for all candidate edges (horizontal disparity difference)
                double location_error = std::abs(right_edge.location.x - left_edge.location.x);
                thread_local_location_errors.push_back(location_error);

                //> Check if this edge is veridical (in ground truth)
                int right_edge_index = edge_cluster.contributing_edges_toed_indices[0];
                bool is_gt = std::find(stereo_frame_edge_pairs.veridical_right_edges_indices[i].begin(),
                                       stereo_frame_edge_pairs.veridical_right_edges_indices[i].end(),
                                       right_edge_index) != stereo_frame_edge_pairs.veridical_right_edges_indices[i].end();
                thread_local_is_veridical.push_back(is_gt ? 1 : 0);

                double candidate_disparity = is_left ? (left_edge.location.x - right_edge.location.x)
                                                     : (right_edge.location.x - left_edge.location.x);
                if (candidate_disparity >= 0 && candidate_disparity <= MAX_DISPARITY)
                {
                    surviving_edge_clusters.push_back(edge_cluster);
                }
            }
            // debug_recall_drop("Disparity Filter", stereo_frame_edge_pairs, surviving_edge_clusters, output_dir, false, i, frame_idx);
            stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters = surviving_edge_clusters;
        }

#pragma omp critical
        {
            location_errors.insert(location_errors.end(), thread_local_location_errors.begin(), thread_local_location_errors.end());
            is_veridical.insert(is_veridical.end(), thread_local_is_veridical.begin(), thread_local_is_veridical.end());
        }
    }

    //> Record location error distribution if output directory is provided
    if (!output_dir.empty() && !location_errors.empty())
    {
        record_Filter_Distribution("location_error", location_errors, is_veridical, output_dir, frame_idx);
    }

    //> Record ambiguity distribution
    record_Ambiguity_Distribution("disparity", stereo_frame_edge_pairs, output_dir, frame_idx);
}

void apply_NCC_Filtering(Stereo_Edge_Pairs &stereo_frame_edge_pairs,
                         const std::string &output_dir, size_t frame_idx, bool is_left = true)
{
    Utility util{};
    // Convert images to CV_64F for template Bilinear_Interpolation<double>
    cv::Mat left_image, right_image;
    if (is_left)
    {
        stereo_frame_edge_pairs.stereo_frame->left_image.convertTo(left_image, CV_64F);
        stereo_frame_edge_pairs.stereo_frame->right_image.convertTo(right_image, CV_64F);
    }
    else
    {
        stereo_frame_edge_pairs.stereo_frame->right_image.convertTo(left_image, CV_64F);
        stereo_frame_edge_pairs.stereo_frame->left_image.convertTo(right_image, CV_64F);
    }

    stereo_frame_edge_pairs.left_edge_patches.resize(stereo_frame_edge_pairs.focused_edge_indices.size());

    //> Storage for NCC values
    std::vector<double> ncc_values;
    std::vector<int> is_veridical;

#pragma omp parallel
    {
        //> Thread-local storage for NCC values
        std::vector<double> thread_local_ncc_values;
        std::vector<int> thread_local_is_veridical;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < stereo_frame_edge_pairs.focused_edge_indices.size(); i++)
        {
            //> Get the left edge patches
            Edge left_edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i);
            stereo_frame_edge_pairs.left_edge_patches[i] = get_edge_patches(left_edge, left_image);

            //> Get the right edge patches
            std::vector<EdgeCluster> surviving_edge_clusters;
            std::vector<double> surviving_edge_cluster_final_scores;
            std::vector<double> surviving_edge_cluster_confidences;
            std::vector<bool> surviving_edge_cluster_validities;
            for (int j = 0; j < stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters.size(); j++)
            {
                const auto &edge_cluster = stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters[j];
                Edge right_edge = edge_cluster.center_edge;
                std::pair<cv::Mat, cv::Mat> right_edge_patches = get_edge_patches(right_edge, right_image);

                //> Calculate the similarity between the left edge patches and the right edge patches
                double sim_pp = get_patch_similarity(stereo_frame_edge_pairs.left_edge_patches[i].first, right_edge_patches.first);   //> (A+, B+)
                double sim_nn = get_patch_similarity(stereo_frame_edge_pairs.left_edge_patches[i].second, right_edge_patches.second); //> (A-, B-)
                double sim_pn = get_patch_similarity(stereo_frame_edge_pairs.left_edge_patches[i].first, right_edge_patches.second);  //> (A+, B-)
                double sim_np = get_patch_similarity(stereo_frame_edge_pairs.left_edge_patches[i].second, right_edge_patches.first);  //> (A-, B+)
                double final_SIM_score = std::max({sim_pp, sim_nn, sim_pn, sim_np});

                //> Record NCC value for all candidate edges
                thread_local_ncc_values.push_back(final_SIM_score);

                //> Check if this edge is veridical (in ground truth)
                int right_edge_index = edge_cluster.contributing_edges_toed_indices[0];
                bool is_gt = std::find(stereo_frame_edge_pairs.veridical_right_edges_indices[i].begin(),
                                       stereo_frame_edge_pairs.veridical_right_edges_indices[i].end(),
                                       right_edge_index) != stereo_frame_edge_pairs.veridical_right_edges_indices[i].end();
                thread_local_is_veridical.push_back(is_gt ? 1 : 0);

                if (final_SIM_score > NCC_THRESH)
                {
                    surviving_edge_clusters.push_back(edge_cluster);
                    surviving_edge_cluster_final_scores.push_back(final_SIM_score);
                    // Preserve existing SIFT confidence if available
                    double existing_confidence = (j < stereo_frame_edge_pairs.matching_edge_clusters[i].refine_confidences.size())
                                                     ? stereo_frame_edge_pairs.matching_edge_clusters[i].refine_confidences[j]
                                                     : 0.0;
                    surviving_edge_cluster_confidences.push_back(existing_confidence);
                    surviving_edge_cluster_validities.push_back(true); //> Placeholder for validity
                }
            }

            // debug_recall_drop("NCC Filter", stereo_frame_edge_pairs, surviving_edge_clusters, output_dir, true, i, frame_idx);

            stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters = std::move(surviving_edge_clusters);
            stereo_frame_edge_pairs.matching_edge_clusters[i].refine_final_scores = std::move(surviving_edge_cluster_final_scores);
            stereo_frame_edge_pairs.matching_edge_clusters[i].refine_confidences = std::move(surviving_edge_cluster_confidences);
            stereo_frame_edge_pairs.matching_edge_clusters[i].refine_validities = std::move(surviving_edge_cluster_validities);
        }

#pragma omp critical
        {
            ncc_values.insert(ncc_values.end(), thread_local_ncc_values.begin(), thread_local_ncc_values.end());
            is_veridical.insert(is_veridical.end(), thread_local_is_veridical.begin(), thread_local_is_veridical.end());
        }
    }

    //> Record NCC distribution if output directory is provided
    if (!output_dir.empty() && !ncc_values.empty())
    {
        record_Filter_Distribution("ncc_score", ncc_values, is_veridical, output_dir, frame_idx);
    }

    //> Record ambiguity distribution
    record_Ambiguity_Distribution("ncc", stereo_frame_edge_pairs, output_dir, frame_idx);
}

void add_edges_to_spatial_grid(Stereo_Edge_Pairs &stereo_frame_edge_pairs, SpatialGrid &spatial_grid, bool is_left)
{
    //> Add edges to spatial grid. This is done on the current image only.
    int size = is_left ? stereo_frame_edge_pairs.focused_edge_indices.size() : stereo_frame_edge_pairs.candidate_edge_indices.size();

    // Thread-safe approach: collect grid assignments in parallel, then populate sequentially
    std::vector<std::pair<int, int>> edge_to_grid(size); // <edge_idx, grid_cell_idx>

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < size; ++i)
    {
        Edge edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i);

        int grid_x = static_cast<int>(edge.location.x) / spatial_grid.cell_size;
        int grid_y = static_cast<int>(edge.location.y) / spatial_grid.cell_size;

        if (grid_x >= 0 && grid_x < spatial_grid.grid_width &&
            grid_y >= 0 && grid_y < spatial_grid.grid_height)
        {
            edge_to_grid[i] = {i, grid_y * spatial_grid.grid_width + grid_x};
        }
        else
        {
            edge_to_grid[i] = {i, -1}; // Mark as invalid
        }
    }

    // Single-threaded population to avoid race conditions
    for (const auto &[edge_idx, grid_idx] : edge_to_grid)
    {
        if (grid_idx >= 0)
        {
            spatial_grid.grid[grid_idx].push_back(edge_idx);
        }
    }
}

void augment_Edge_Data(Stereo_Edge_Pairs &stereo_frame_edge_pairs, bool is_left)
{
    stereo_frame_edge_pairs.left_edge_descriptors.clear();
    stereo_frame_edge_pairs.left_edge_descriptors.resize(stereo_frame_edge_pairs.focused_edge_indices.size());
    const cv::Mat &image = is_left ? stereo_frame_edge_pairs.stereo_frame->left_image_undistorted
                                   : stereo_frame_edge_pairs.stereo_frame->right_image_undistorted;
#pragma omp parallel
    {
        cv::Ptr<cv::SIFT> thread_sift = cv::SIFT::create();

#pragma omp for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(stereo_frame_edge_pairs.focused_edge_indices.size()); ++i)
        {
            Edge le = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i);
            std::pair<cv::Point2d, cv::Point2d> shifted_points = get_Orthogonal_Shifted_Points(le, 8);

            //> Create keypoints for this edge pair
            std::vector<cv::KeyPoint> kps;
            kps.push_back(cv::KeyPoint(shifted_points.first, 1, 180 / M_PI * le.orientation));
            kps.push_back(cv::KeyPoint(shifted_points.second, 1, 180 / M_PI * le.orientation));

            cv::Mat descriptors;
            thread_sift->compute(image, kps, descriptors);

            if (descriptors.rows == 2)
            {
                stereo_frame_edge_pairs.left_edge_descriptors[i] = std::make_pair(descriptors.row(0).clone(), descriptors.row(1).clone());
            }
            else
            {
                stereo_frame_edge_pairs.left_edge_descriptors[i] = std::make_pair(cv::Mat(), cv::Mat());
            }
        }
    }
}

void apply_SIFT_filtering(Stereo_Edge_Pairs &stereo_frame_edge_pairs, double sift_dist_threshold, const std::string &output_dir = "", size_t frame_idx = 0, bool is_left = true)
{
    cv::Mat candidate_image = is_left ? stereo_frame_edge_pairs.stereo_frame->right_image_undistorted
                                      : stereo_frame_edge_pairs.stereo_frame->left_image_undistorted;

    //> Storage for SIFT descriptor distances
    std::vector<double> sift_distances;
    std::vector<int> is_veridical;

#pragma omp parallel
    {
        //> Thread-local storage for SIFT distances
        std::vector<double> thread_local_sift_distances;
        std::vector<int> thread_local_is_veridical;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(stereo_frame_edge_pairs.focused_edge_indices.size()); ++i)
        {
            stereo_frame_edge_pairs.matching_edge_clusters[i].refine_confidences.clear();
            std::vector<EdgeCluster> surviving_edge_clusters;
            std::vector<double> surviving_edge_cluster_confidences;
            Edge left_edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i);
            std::pair<cv::Mat, cv::Mat> left_edge_descriptor = stereo_frame_edge_pairs.left_edge_descriptors[i];
            std::vector<cv::KeyPoint> edge_keypoints;

            cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
            for (auto const &edge_cluster : stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters)
            {
                Edge right_edge = edge_cluster.center_edge;
                std::pair<cv::Point2d, cv::Point2d> shifted_points = get_Orthogonal_Shifted_Points(right_edge, 8); //> augment the edge by adding shifted points along the orthogonal direction
                cv::KeyPoint edge_kp1(shifted_points.first, 1, 180 / M_PI * right_edge.orientation);
                cv::KeyPoint edge_kp2(shifted_points.second, 1, 180 / M_PI * right_edge.orientation);
                edge_keypoints.push_back(edge_kp1);
                edge_keypoints.push_back(edge_kp2);
            }
            cv::Mat right_edge_descriptors;
            sift->compute(candidate_image, edge_keypoints, right_edge_descriptors);

            //> Check if descriptors were computed successfully and match the number of clusters
            if (right_edge_descriptors.rows == static_cast<int>(stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters.size() * 2))
            {
                for (int j = 0; j < right_edge_descriptors.rows; j += 2)
                {
                    cv::Mat right_edge_descriptor_1 = right_edge_descriptors.row(j);
                    cv::Mat right_edge_descriptor_2 = right_edge_descriptors.row(j + 1);
                    double sift_desc_distance_1 = cv::norm(left_edge_descriptor.first, right_edge_descriptor_1, cv::NORM_L2);
                    double sift_desc_distance_2 = cv::norm(left_edge_descriptor.second, right_edge_descriptor_1, cv::NORM_L2);
                    double sift_desc_distance_3 = cv::norm(left_edge_descriptor.first, right_edge_descriptor_2, cv::NORM_L2);
                    double sift_desc_distance_4 = cv::norm(left_edge_descriptor.second, right_edge_descriptor_2, cv::NORM_L2);
                    double sift_desc_distance = std::min({sift_desc_distance_1, sift_desc_distance_2, sift_desc_distance_3, sift_desc_distance_4});

                    //> Record SIFT distance for all candidate edges
                    thread_local_sift_distances.push_back(sift_desc_distance);

                    //> Check if this edge is veridical (in ground truth)
                    int right_edge_index = stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters[j / 2].contributing_edges_toed_indices[0];
                    bool is_gt = std::find(stereo_frame_edge_pairs.veridical_right_edges_indices[i].begin(),
                                           stereo_frame_edge_pairs.veridical_right_edges_indices[i].end(),
                                           right_edge_index) != stereo_frame_edge_pairs.veridical_right_edges_indices[i].end();
                    thread_local_is_veridical.push_back(is_gt ? 1 : 0);

                    if (sift_desc_distance < sift_dist_threshold)
                    {
                        EdgeCluster edge_cluster = stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters[j / 2];
                        surviving_edge_cluster_confidences.push_back(sift_desc_distance);
                        surviving_edge_clusters.push_back(edge_cluster);
                    }
                }
            }
            else
            {
                //> If descriptors don't match, keep all clusters (fail-safe)
                surviving_edge_clusters = stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters;
                // Populate with a large default value since SIFT failed (lower is better for SIFT)
                surviving_edge_cluster_confidences.assign(surviving_edge_clusters.size(), sift_dist_threshold * 2.0);
            }
            // debug_recall_drop("SIFT Filter", stereo_frame_edge_pairs, surviving_edge_clusters, output_dir, true, i, frame_idx);
            stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters = surviving_edge_clusters;
            stereo_frame_edge_pairs.matching_edge_clusters[i].refine_confidences = surviving_edge_cluster_confidences;
        }

#pragma omp critical
        {
            sift_distances.insert(sift_distances.end(), thread_local_sift_distances.begin(), thread_local_sift_distances.end());
            is_veridical.insert(is_veridical.end(), thread_local_is_veridical.begin(), thread_local_is_veridical.end());
        }
    }

    //> Record SIFT distance distribution if output directory is provided
    if (!output_dir.empty() && !sift_distances.empty())
    {
        record_Filter_Distribution("sift_distance", sift_distances, is_veridical, output_dir, frame_idx);
    }

    //> Record ambiguity distribution
    record_Ambiguity_Distribution("sift", stereo_frame_edge_pairs, output_dir, frame_idx);
}

void apply_Best_Nearly_Best_Test(Stereo_Edge_Pairs &stereo_frame_edge_pairs, double lowe_ratio_threshold = LOWES_RATIO, const std::string &output_dir = "", size_t frame_idx = 0, bool is_NCC = true)
{
    std::string test_name = is_NCC ? "BNB_NCC" : "BNB_SIFT";
#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(stereo_frame_edge_pairs.focused_edge_indices.size()); ++i)
        {
            Edge left_edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i);
            auto &cluster_data = stereo_frame_edge_pairs.matching_edge_clusters[i];
            size_t num_clusters = cluster_data.edge_clusters.size();

            if (num_clusters < 2)
                continue;

            // 1. Create an index map to sort clusters based on score without losing original indices
            // Assuming higher score is better (NCC). If SIFT (lower better), flip the comparison logic.
            std::vector<size_t> indices(num_clusters);
            std::iota(indices.begin(), indices.end(), 0);
            if (is_NCC)
                std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
                          { return cluster_data.refine_final_scores[a] > cluster_data.refine_final_scores[b]; });
            else
                std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b)
                          { return cluster_data.refine_confidences[a] < cluster_data.refine_confidences[b]; });
            // 2. Determine how many candidates pass the recursive ratio test
            size_t keep_count = 1; // Always keep the best

            double best_score = is_NCC ? cluster_data.refine_final_scores[indices[0]] : cluster_data.refine_confidences[indices[0]];

            for (size_t j = 0; j < num_clusters - 1; ++j)
            {
                double next_score = is_NCC ? cluster_data.refine_final_scores[indices[j + 1]] : cluster_data.refine_confidences[indices[j + 1]];
                if (best_score == 0)
                    break;
                // Ratio: next_best / current_best
                // If the ratio is high (e.g., 0.9), they are "nearly best."
                // If the ratio is low (e.g., 0.4), the next one is significantly worse.

                double ratio = is_NCC ? next_score / best_score : best_score / next_score;
                if (ratio >= lowe_ratio_threshold)
                {
                    keep_count++;
                }
                else
                {
                    break; // Significant drop detected, stop including further matches
                }
            }

            // 3. If we aren't keeping everything, rebuild the vectors
            if (keep_count < num_clusters)
            {
                std::vector<EdgeCluster> surviving_clusters;
                std::vector<double> surviving_scores;
                std::vector<double> surviving_confidences;
                std::vector<bool> surviving_validities;

                for (size_t k = 0; k < keep_count; ++k)
                {
                    size_t idx = indices[k];
                    surviving_clusters.push_back(cluster_data.edge_clusters[idx]);
                    surviving_scores.push_back(cluster_data.refine_final_scores[idx]);
                    surviving_confidences.push_back(cluster_data.refine_confidences[idx]);
                    surviving_validities.push_back(cluster_data.refine_validities[idx]);
                }

                // debug_recall_drop_refine(test_name, stereo_frame_edge_pairs, surviving_clusters, "output_files", true, i, frame_idx);

                // if (is_NCC && i == 10580)
                // {
                //     std::cout << "BEFORE BNB - index 265(" << left_edge.location.x << ", " << left_edge.location.y << "), orientation: " << left_edge.orientation << std::endl;
                //     std::cout << "Best score: " << best_score << " (threshold: " << lowe_ratio_threshold << ")" << std::endl;
                //     for (int m = 0; m < num_clusters; ++m)
                //     {
                //         size_t idx = indices[m]; // idx points to the sorted cluster
                //         bool is_veridical = cv::norm(cluster_data.edge_clusters[idx].center_edge.location - stereo_frame_edge_pairs.GT_locations_from_left_edges[i]) <= DIST_TO_GT_THRESH;
                //         double score = cluster_data.refine_final_scores[idx];
                //         double ratio = score / best_score;
                //         std::string kept = (m < keep_count) ? "[KEPT]" : "[DROPPED]";
                //         std::cout << "  Rank " << m << " " << kept << " (orig_idx=" << idx << "): loc=("
                //                   << cluster_data.edge_clusters[idx].center_edge.location.x << ", "
                //                   << cluster_data.edge_clusters[idx].center_edge.location.y << "), orientation= "
                //                   << cluster_data.edge_clusters[idx].center_edge.orientation
                //                   << ", Score=" << score << ", Ratio=" << ratio
                //                   << (is_veridical ? " [VERIDICAL]" : "") << std::endl;
                //     }
                // }
                cluster_data.edge_clusters = std::move(surviving_clusters);
                cluster_data.refine_final_scores = std::move(surviving_scores);
                cluster_data.refine_confidences = std::move(surviving_confidences);
                cluster_data.refine_validities = std::move(surviving_validities);
                // if (is_NCC && i == 10580)
                // {
                //     std::cout << "AFTER BNB - index 265 - kept " << cluster_data.edge_clusters.size() << " clusters" << std::endl;
                //     for (size_t m = 0; m < cluster_data.edge_clusters.size(); ++m)
                //     {
                //         bool is_veridical = cv::norm(cluster_data.edge_clusters[m].center_edge.location - stereo_frame_edge_pairs.GT_locations_from_left_edges[i]) <= DIST_TO_GT_THRESH;
                //         double score = cluster_data.refine_final_scores[m];
                //         std::cout << "  Surviving " << m << ": loc=("
                //                   << cluster_data.edge_clusters[m].center_edge.location.x << ", "
                //                   << cluster_data.edge_clusters[m].center_edge.location.y << "), orientation= "
                //                   << cluster_data.edge_clusters[m].center_edge.orientation
                //                   << ", Score=" << score
                //                   << (is_veridical ? " [VERIDICAL]" : "") << std::endl;
                //     }
                // }
            }
        }
    }
    record_Ambiguity_Distribution(test_name, stereo_frame_edge_pairs, output_dir, frame_idx);
}

void apply_Lowe_Ratio_Test(Stereo_Edge_Pairs &stereo_frame_edge_pairs, double lowe_ratio_threshold, const std::string &output_dir, size_t frame_idx)
{

    for (int i = 0; i < static_cast<int>(stereo_frame_edge_pairs.focused_edge_indices.size()); ++i)
    {
        Edge left_edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i);
        auto &cluster_data = stereo_frame_edge_pairs.matching_edge_clusters[i];

        if (cluster_data.edge_clusters.empty())
            continue;

        // Check if this edge has GT match in current candidates
        bool has_gt_before = false;
        int gt_cluster_idx = -1;
        for (int j = 0; j < static_cast<int>(cluster_data.edge_clusters.size()); ++j)
        {
            if (cv::norm(cluster_data.edge_clusters[j].center_edge.location - stereo_frame_edge_pairs.GT_locations_from_left_edges[i]) <= DIST_TO_GT_THRESH)
            {
                has_gt_before = true;
                gt_cluster_idx = j;
                break;
            }
        }

        // 1. Find the index of the absolute best match (highest NCC score)
        int best_idx = 0;
        double max_score = -1.0;

        for (int j = 0; j < static_cast<int>(cluster_data.edge_clusters.size()); ++j)
        {
            if (cluster_data.refine_final_scores[j] > max_score)
            {
                max_score = cluster_data.refine_final_scores[j];
                best_idx = j;
            }
        }

        // 2. Keep ONLY that one best match
        std::vector<EdgeCluster> surviving_clusters = {cluster_data.edge_clusters[best_idx]};
        std::vector<double> surviving_scores = {cluster_data.refine_final_scores[best_idx]};
        std::vector<double> surviving_confidences = {cluster_data.refine_confidences[best_idx]};
        std::vector<bool> surviving_validities = {cluster_data.refine_validities[best_idx]};

        // 3. Update the data
        cluster_data.edge_clusters = std::move(surviving_clusters);
        cluster_data.refine_final_scores = std::move(surviving_scores);
        cluster_data.refine_confidences = std::move(surviving_confidences);
        cluster_data.refine_validities = std::move(surviving_validities);
    }
    // debug_file.close();
}

void apply_bidirectional_test(Stereo_Edge_Pairs &left_frame, Stereo_Edge_Pairs &right_frame, const std::string &output_dir, int frame_idx)
{
    //> Now we have good matches from both direction, need to validate bidirectional consistency
    std::unordered_map<int, std::vector<int>> L2R, R2L, edge2cluster;

    //> Testing left to right now, goal: store everything in the map!!
    for (int i = 0; i < left_frame.focused_edge_indices.size(); i++)
    {
        int left_edge_index = left_frame.get_focused_toed_edge_index(i);
        for (const auto &edge_cluster : left_frame.matching_edge_clusters[i].edge_clusters)
        {
            for (int j = 0; j < edge_cluster.contributing_edges_toed_indices.size(); j++)
            {
                int right_edge_index = edge_cluster.contributing_edges_toed_indices[j];
                L2R[left_edge_index].push_back(right_edge_index);
                R2L[right_edge_index].push_back(left_edge_index);
                edge2cluster[right_edge_index].push_back(left_edge_index);
            }
        }
    }
    // Now L2R and R2L are populated, start to check from right to left
    for (int i = 0; i < right_frame.focused_edge_indices.size(); i++)
    {
        int right_edge_toed_index = right_frame.get_focused_toed_edge_index(i); //> Convert sequential to TOED index

        if (R2L.find(right_edge_toed_index) != R2L.end())
        {
            // if we found this right edge was matched from some left edges, check if it matches back
            for (int j = 0; j < R2L[right_edge_toed_index].size(); j++)
            {
                int left_edge_toed_index = R2L[right_edge_toed_index][j]; //> TOED index of left edge

                //> Check if right edge has this left edge as a match in the right->left direction
                bool found_bidirectional_match = false;
                for (const auto &edge_cluster : right_frame.matching_edge_clusters[i].edge_clusters)
                {
                    for (int k = 0; k < edge_cluster.contributing_edges_toed_indices.size(); k++)
                    {
                        if (edge_cluster.contributing_edges_toed_indices[k] == left_edge_toed_index)
                        {
                            found_bidirectional_match = true;
                            break;
                        }
                    }
                    if (found_bidirectional_match)
                        break;
                }
                //> If no bidirectional match found, remove right edge from L2R[left_edge_toed_index]
                if (!found_bidirectional_match)
                {
                    L2R[left_edge_toed_index].erase(
                        std::remove(L2R[left_edge_toed_index].begin(), L2R[left_edge_toed_index].end(), right_edge_toed_index),
                        L2R[left_edge_toed_index].end());
                }
            }
        }
    }
    //> Now update left_frame's matching_edge_clusters to keep all clusters if any contributing edge passes bidirectionality
    for (int i = 0; i < left_frame.focused_edge_indices.size(); i++)
    {
        int left_edge_index = left_frame.get_focused_toed_edge_index(i);
        if (L2R.find(left_edge_index) == L2R.end() || L2R[left_edge_index].empty())
        {
            // No bidirectional matches for this left edge, clear all clusters
            left_frame.matching_edge_clusters[i].edge_clusters.clear();
            continue;
        }
        std::vector<EdgeCluster> kept_clusters;
        for (auto edge_cluster : left_frame.matching_edge_clusters[i].edge_clusters)
        {
            std::vector<int> bidirectional_contributing_indices;
            std::vector<Edge> bidirectional_contributing_edges;
            for (size_t j = 0; j < edge_cluster.contributing_edges_toed_indices.size(); ++j)
            {
                int contributing_toed_idx = edge_cluster.contributing_edges_toed_indices[j];
                if (std::find(L2R[left_edge_index].begin(), L2R[left_edge_index].end(), contributing_toed_idx) != L2R[left_edge_index].end())
                {
                    bidirectional_contributing_indices.push_back(contributing_toed_idx);
                    if (j < edge_cluster.contributing_edges.size())
                        bidirectional_contributing_edges.push_back(edge_cluster.contributing_edges[j]);
                }
            }
            if (!bidirectional_contributing_indices.empty())
            {
                edge_cluster.contributing_edges_toed_indices = bidirectional_contributing_indices;
                edge_cluster.contributing_edges = bidirectional_contributing_edges;
                kept_clusters.push_back(edge_cluster);
            }
        }
        left_frame.matching_edge_clusters[i].edge_clusters = kept_clusters;
    }
    record_Ambiguity_Distribution("bidirectional", left_frame, output_dir, frame_idx);
}

void consolidate_redundant_edge_hypothesis(Stereo_Edge_Pairs &stereo_frame_edge_pairs, size_t frame_idx, bool b_do_epipolar_shift = true, bool b_do_clustering = true)
{

    std::string debug_dir = "output_files/debug_recall_drop";
    std::string debug_filename = debug_dir + "/frame_" + std::to_string(frame_idx) + "_cluster_selection.txt";
    std::ofstream debug_file(debug_filename);

    // std::vector<std::vector<EdgeCluster>> initial_matching_edge_clusters = stereo_frame_edge_pairs.matching_edge_clusters.edge_clusters;
    for (int i = 0; i < stereo_frame_edge_pairs.focused_edge_indices.size(); i++) // check
    {
        if (stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters.empty()) // check
            continue;

        std::vector<Edge> shifted_edges;
        std::vector<int> toed_indices_of_shifted_edges;

        if (b_do_epipolar_shift)
        {
            const auto e_coeffs = stereo_frame_edge_pairs.epip_line_coeffs_of_left_edges[i]; // check

            //> Shift edges in the edge cluster to the epipolar line
            std::vector<EdgeCluster> shifted_edge_clusters;
            for (const auto &edge_cluster : stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters) // check
            {
                Edge shifted_edge = shift_Edge_to_Epipolar_Line(edge_cluster.center_edge, e_coeffs);
                shifted_edges.push_back(shifted_edge);
                toed_indices_of_shifted_edges.push_back(edge_cluster.contributing_edges_toed_indices[0]);

                EdgeCluster shifted_edge_cluster;
                shifted_edge_cluster.center_edge = shifted_edge;
                shifted_edge_clusters.push_back(shifted_edge_cluster);
            }
            // debug_recall_drop_refine("Epipolar Shift", stereo_frame_edge_pairs, shifted_edge_clusters, "output_files", true, i, frame_idx);
            stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters = shifted_edge_clusters;
        }
        else
        {
            if (stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters.size() == 1)
                continue;
            for (const auto &edge_cluster : stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters)
            {
                shifted_edges.push_back(edge_cluster.center_edge);
                toed_indices_of_shifted_edges.push_back(edge_cluster.contributing_edges_toed_indices[0]);
            }
        }

        if (b_do_clustering)
        {
            //> Cluster the shifted edges
            std::vector<double> scores_for_clustering;
            bool has_gt_before = false;
            bool gt_in_cluster = false;
            for (auto s : stereo_frame_edge_pairs.matching_edge_clusters[i].refine_final_scores)
            {
                scores_for_clustering.push_back(s);
            }
            const Edge &left_edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i);
            Edge e;
            for (auto c : stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters)
            {
                e = c.center_edge;
                if (cv::norm(e.location - stereo_frame_edge_pairs.GT_locations_from_left_edges[i]) <= DIST_TO_GT_THRESH)
                {
                    has_gt_before = true;
                    break;
                }
            }
            EdgeClusterer edge_cluster_engine(shifted_edges, toed_indices_of_shifted_edges, b_do_epipolar_shift);
            edge_cluster_engine.setRefineScores(scores_for_clustering);
            edge_cluster_engine.performClustering();
            // debug_recall_drop_refine("Edge Clustering", stereo_frame_edge_pairs, edge_cluster_engine.returned_clusters, "output_files", true, i, frame_idx);
            stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters = edge_cluster_engine.returned_clusters;
            // Debug: if we're dropping GT match, log details
            if (has_gt_before)
            {
                for (int j = 0; j < stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters.size(); j++)
                {
                    if (cv::norm(stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters[j].center_edge.location - stereo_frame_edge_pairs.GT_locations_from_left_edges[i]) <= DIST_TO_GT_THRESH)
                    {
                        gt_in_cluster = true;
                        break;
                    }
                }
                if (!gt_in_cluster)
                {
                    debug_file << "Edge " << i << " (" << left_edge.location.x << "," << left_edge.location.y << ")" << std::endl;
                    debug_file << "  GT location: (" << stereo_frame_edge_pairs.GT_locations_from_left_edges[i].x << "," << stereo_frame_edge_pairs.GT_locations_from_left_edges[i].y << ")" << std::endl;
                    debug_file << "  Has GT before clustering at" << e.location.x << "," << e.location.y << ")" << std::endl;
                    for (const auto &c : stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters)
                    {
                        Edge ce = c.center_edge;

                        debug_file << "  Clustered edge at: (" << ce.location.x << "," << ce.location.y << ")" << std::endl;
                    }

                    debug_file << std::endl;
                }
            }
        }
    }
    debug_file.close();
}

void min_Edge_Photometric_Residual_by_Gauss_Newton(
    /* inputs */
    Edge left_edge, double init_disp, const cv::Mat &left_image_undistorted, const cv::Mat &right_image_undistorted, const cv::Mat &right_image_gradients_x, /* outputs */
    double &refined_disparity, double &refined_final_score, double &refined_confidence, bool &refined_validity, std::vector<double> &residual_log,           /* optional inputs */
    int max_iter = 20, double tol = 1e-3, double huber_delta = 3.0, bool b_verbose = false)
{
    cv::Point2d t(std::cos(left_edge.orientation), std::sin(left_edge.orientation));
    cv::Point2d n(-t.y, t.x);
    double side_shift = (PATCH_SIZE / 2.0) + 1.0;
    cv::Point2d c_plus = left_edge.location + n * side_shift;
    cv::Point2d c_minus = left_edge.location - n * side_shift;

    std::vector<cv::Point2d> cLplus, cLminus;
    util_make_rotated_patch_coords(c_plus, left_edge.orientation, cLplus);
    util_make_rotated_patch_coords(c_minus, left_edge.orientation, cLminus);

    std::vector<double> pLplus_f, pLminus_f;
    util_sample_patch_at_coords(left_image_undistorted, cLplus, pLplus_f);
    util_sample_patch_at_coords(left_image_undistorted, cLminus, pLminus_f);
    double mLplus = util_vector_mean<double>(pLplus_f);
    double mLminus = util_vector_mean<double>(pLminus_f);
    std::vector<double> Lplus, Lminus;
    for (double x : pLplus_f)
    {
        Lplus.push_back(x - mLplus);
    }
    for (double x : pLminus_f)
    {
        Lminus.push_back(x - mLminus);
    }

    double d = init_disp;
    double init_RMS = 0.0;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        //> Compute the right patch coordinates
        cv::Point2d cRplus = c_plus - cv::Point2d(d, 0);
        cv::Point2d cRminus = c_minus - cv::Point2d(d, 0);

        std::vector<cv::Point2d> cRplusC, cRminusC;
        util_make_rotated_patch_coords(cRplus, left_edge.orientation, cRplusC);
        util_make_rotated_patch_coords(cRminus, left_edge.orientation, cRminusC);

        //> Sample right intensities and right gradient X at these coords
        std::vector<double> pRplus_f, pRminus_f, gxRplus_f, gxRminus_f;
        util_sample_patch_at_coords(right_image_undistorted, cRplusC, pRplus_f);
        util_sample_patch_at_coords(right_image_undistorted, cRminusC, pRminus_f);
        util_sample_patch_at_coords(right_image_gradients_x, cRplusC, gxRplus_f);
        util_sample_patch_at_coords(right_image_gradients_x, cRminusC, gxRminus_f);

        //> Compute means of the right patches
        double mRplus = util_vector_mean<double>(pRplus_f);
        double mRminus = util_vector_mean<double>(pRminus_f);

        //> Build residuals r = (L - meanL) - (R - meanR)  which centers both patches
        //> Build gradient which is the derivative of the residual with respect to the disparity g = dr / dd
        double H = 0.0;
        double b = 0.0;
        double cost = 0.0;
        auto accumulate_patch = [&](const std::vector<double> &Lc, const std::vector<double> &Rf,
                                    const std::vector<double> &gxRf, double meanR)
        {
            for (size_t k = 0; k < Lc.size(); ++k)
            {
                double r = Lc[k] - (Rf[k] - meanR);
                double g = gxRf[k];
                double w = 1.0;
                double absr = std::abs(r);
                if (absr > huber_delta)
                    w = huber_delta / absr;
                H += w * g * g;
                b += w * g * r;
                cost += w * r * r;
            }
        };
        accumulate_patch(Lplus, pRplus_f, gxRplus_f, mRplus);
        accumulate_patch(Lminus, pRminus_f, gxRminus_f, mRminus);

        if (H < 1e-8)
            break;

        //> Update delta
        double delta = -b / H;
        d += delta;

        double rms = std::sqrt(cost / (Lplus.size() + Lminus.size()));
        if (iter == 0)
            init_RMS = rms;
        if (b_verbose)
        {
            std::cout << "iter " << iter << ": disp =" << d
                      << "  Δ =" << delta
                      << "  RMS =" << rms
                      << "  cost =" << cost << std::endl;
        }
        residual_log.push_back(rms);

        bool is_outlier = (rms > huber_delta * 2.0) || (residual_log.size() < 2);

        //> Early stopping if the update is too small
        if (std::abs(delta) < tol || iter == max_iter - 1)
        {
            refined_validity = (is_outlier) ? false : true;
            refined_final_score = rms;
            refined_confidence = std::exp(-rms / huber_delta);
            break;
        }
        else if (iter == max_iter - 1)
        {
            refined_validity = (is_outlier) ? false : true;
            refined_final_score = rms;
            refined_confidence = 1.0 - (rms / init_RMS); //> optional
        }
    }

    refined_disparity = d;
}
void refine_edge_disparity(Stereo_Edge_Pairs &stereo_frame_edge_pairs, size_t frame_idx, bool is_left)
{
    cv::Mat left_cur_undistorted_32F, right_cur_undistorted_32F;
    is_left ? stereo_frame_edge_pairs.stereo_frame->left_image_undistorted.convertTo(left_cur_undistorted_32F, CV_32F) : stereo_frame_edge_pairs.stereo_frame->right_image_undistorted.convertTo(left_cur_undistorted_32F, CV_32F);
    is_left ? stereo_frame_edge_pairs.stereo_frame->right_image_undistorted.convertTo(right_cur_undistorted_32F, CV_32F) : stereo_frame_edge_pairs.stereo_frame->left_image_undistorted.convertTo(right_cur_undistorted_32F, CV_32F);

#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(stereo_frame_edge_pairs.focused_edge_indices.size()); ++i)
        {
            if (stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters.empty())
                continue;

            Edge left_edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i);
            std::vector<EdgeCluster> updated_edge_clusters;
            stereo_frame_edge_pairs.matching_edge_clusters[i].refine_final_scores.clear();
            stereo_frame_edge_pairs.matching_edge_clusters[i].refine_confidences.clear();
            stereo_frame_edge_pairs.matching_edge_clusters[i].refine_validities.clear();

            double gt_right_x = stereo_frame_edge_pairs.GT_locations_from_left_edges[i].x;

            for (auto const &edge_cluster : stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters)
            {
                Edge right_edge = edge_cluster.center_edge;
                bool initial_was_correct = (std::abs(right_edge.location.x - gt_right_x) < DIST_TO_GT_THRESH);

                double initial_disparity = left_edge.location.x - right_edge.location.x;
                double refined_disparity, initial_score, refined_final_score, refined_confidence;
                bool refined_validity;
                std::vector<double> residual_log;
                //> Refine the edge disparity through minimizing the photometric residual of the sum of two patches through Gauss-Newton method
                const cv::Mat &candidate_image_gradients_x = is_left ? stereo_frame_edge_pairs.stereo_frame->right_image_gradients_x
                                                                     : stereo_frame_edge_pairs.stereo_frame->left_image_gradients_x;
                min_Edge_Photometric_Residual_by_Gauss_Newton(
                    left_edge, initial_disparity, left_cur_undistorted_32F, right_cur_undistorted_32F, candidate_image_gradients_x,
                    refined_disparity, refined_final_score, refined_confidence, refined_validity, residual_log);

                stereo_frame_edge_pairs.matching_edge_clusters[i].refine_final_scores.push_back(refined_final_score);
                stereo_frame_edge_pairs.matching_edge_clusters[i].refine_confidences.push_back(refined_confidence);
                stereo_frame_edge_pairs.matching_edge_clusters[i].refine_validities.push_back(refined_validity);

                //> Update the edge cluster center edge location using the refined disparity
                EdgeCluster updated_edge_cluster = edge_cluster;
                updated_edge_cluster.center_edge.location.x = left_edge.location.x - refined_disparity;
                updated_edge_clusters.push_back(updated_edge_cluster);
            }
            stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters = updated_edge_clusters;
        }
    }
}

Frame_Evaluation_Metrics get_Stereo_Edge_Pairs(Dataset &dataset, Stereo_Edge_Pairs &stereo_frame_edge_pairs, size_t frame_idx)
{
    // write_3D_edges_to_file(dataset, stereo_frame_edge_pairs, frame_idx);
    Stereo_Edge_Pairs stereo_edge_pairs_right;
#if BIDIRECTIONAL_FILTERING

    std::vector<double> right_edge_disparities;
    // for now the pairs_right have nothing in it, we want to process edges to it(we dont need to redo it since frame_edge_pairs have all the info)
    stereo_edge_pairs_right.stereo_frame = stereo_frame_edge_pairs.stereo_frame;
    stereo_edge_pairs_right.left_disparity_map = stereo_frame_edge_pairs.right_disparity_map;
    stereo_edge_pairs_right.right_disparity_map = stereo_frame_edge_pairs.left_disparity_map;
    if (dataset.b_get_right_edges_and_disparities_from_file)
    {
        Find_Stereo_GT_Locations(dataset, right_edge_disparities, *stereo_frame_edge_pairs.stereo_frame, stereo_edge_pairs_right, false);
    }
    else
    {
        Find_Stereo_GT_Locations(dataset, stereo_frame_edge_pairs.right_disparity_map, *stereo_frame_edge_pairs.stereo_frame, stereo_edge_pairs_right, false);
    }
    std::cout << "Complete calculating GT locations for right edges of the keyframe..." << std::endl;
    std::cout << "Size of right edges with GT locations on the right image = " << stereo_edge_pairs_right.size() << std::endl;

    get_Stereo_Edge_GT_Pairs(dataset, *stereo_frame_edge_pairs.stereo_frame, stereo_edge_pairs_right, false);
    std::cout << "Size of stereo edge correspondences pool = " << stereo_edge_pairs_right.size() << std::endl;

#endif

    SpatialGrid spatial_left(dataset.get_width(), dataset.get_height(), 5);
    SpatialGrid spatial_right(dataset.get_width(), dataset.get_height(), 5);

    Evaluation_Statistics evaluation_statistics;
    Evaluation_Statistics evaluation_statistics_right;
    Frame_Evaluation_Metrics frame_metrics;

    double recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg;
    double recall_per_image_right, precision_per_image_right, precision_pair_per_image_right, num_of_target_edges_per_source_edge_avg_right;

    //> ============= START OF CH'S EDITIONS =============

    //> Apply epipolar line distance filtering (must be first to extract candidates)
    //> Set num_random_edges_for_distribution to 5-10 to record ALL right edges for random subset of left edges (shows pre-filtering distribution)
    //> Set to 0 to disable distribution recording
    apply_Epipolar_Line_Distance_Filtering(stereo_frame_edge_pairs, dataset, stereo_frame_edge_pairs.stereo_frame->right_edges, "output_files", true, frame_idx, 10);

    Evaluate_Stereo_Edge_Correspondences(stereo_frame_edge_pairs, frame_idx, "Epipolar Line Distance Filtering",
                                         recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg,
                                         evaluation_statistics);
    frame_metrics.stages["Epipolar Line Distance Filtering"] = {recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg};
#if BIDIRECTIONAL_FILTERING
    apply_Epipolar_Line_Distance_Filtering(stereo_edge_pairs_right, dataset, stereo_edge_pairs_right.stereo_frame->left_edges, "output_files/right_edges", false, frame_idx, 10);
    Evaluate_Stereo_Edge_Correspondences(stereo_edge_pairs_right, frame_idx, "Epipolar Line Distance Filtering (Right)",
                                         recall_per_image_right, precision_per_image_right, precision_pair_per_image_right, num_of_target_edges_per_source_edge_avg_right,
                                         evaluation_statistics_right);
#endif

    add_edges_to_spatial_grid(stereo_frame_edge_pairs, spatial_left, true);
    add_edges_to_spatial_grid(stereo_frame_edge_pairs, spatial_right, false);

    //> Apply disparity filtering
    apply_Disparity_Filtering(stereo_frame_edge_pairs, "output_files", frame_idx);
    Evaluate_Stereo_Edge_Correspondences(stereo_frame_edge_pairs, frame_idx, "Maximal Disparity Filtering",
                                         recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg,
                                         evaluation_statistics);
    frame_metrics.stages["Maximal Disparity Filtering"] = {recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg};
#if BIDIRECTIONAL_FILTERING
    apply_Disparity_Filtering(stereo_edge_pairs_right, "output_files/right_edges", frame_idx, false);
    Evaluate_Stereo_Edge_Correspondences(stereo_edge_pairs_right, frame_idx, "Maximal Disparity Filtering (Right)",
                                         recall_per_image_right, precision_per_image_right, precision_pair_per_image_right, num_of_target_edges_per_source_edge_avg_right,
                                         evaluation_statistics_right);
#endif
    //> Augment edge data with SIFT descriptors
    augment_Edge_Data(stereo_frame_edge_pairs, true);

#if BIDIRECTIONAL_FILTERING
    augment_Edge_Data(stereo_edge_pairs_right, false);
#endif

    //> Apply SIFT filtering
    apply_SIFT_filtering(stereo_frame_edge_pairs, SIFT_THRESHOLD, "output_files", frame_idx);
    Evaluate_Stereo_Edge_Correspondences(stereo_frame_edge_pairs, frame_idx, "SIFT Filtering",
                                         recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg,
                                         evaluation_statistics);
    frame_metrics.stages["SIFT Filtering"] = {recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg};

    //> Apply NCC filtering

    apply_NCC_Filtering(stereo_frame_edge_pairs, "output_files", frame_idx);
    Evaluate_Stereo_Edge_Correspondences(stereo_frame_edge_pairs, frame_idx, "NCC Filtering",
                                         recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg,
                                         evaluation_statistics);
    frame_metrics.stages["NCC Filtering"] = {recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg};

    apply_Best_Nearly_Best_Test(stereo_frame_edge_pairs, BNB_NCC, "output_files", frame_idx, true);
    Evaluate_Stereo_Edge_Correspondences(stereo_frame_edge_pairs, frame_idx, "BNB Test NCC",
                                         recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg,
                                         evaluation_statistics);
    frame_metrics.stages["BNB Test"] = {recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg};

    apply_Best_Nearly_Best_Test(stereo_frame_edge_pairs, BNB_SIFT, "output_files", frame_idx, false);
    Evaluate_Stereo_Edge_Correspondences(stereo_frame_edge_pairs, frame_idx, "BNB Test SIFT",
                                         recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg,
                                         evaluation_statistics);
    frame_metrics.stages["BNB Test_Strict"] = {recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg};

#if BIDIRECTIONAL_FILTERING
    apply_NCC_Filtering(stereo_edge_pairs_right, "output_files/right_edges", frame_idx, false);
    Evaluate_Stereo_Edge_Correspondences(stereo_edge_pairs_right, frame_idx, "NCC Filtering (Right)",
                                         recall_per_image_right, precision_per_image_right, precision_pair_per_image_right, num_of_target_edges_per_source_edge_avg_right,
                                         evaluation_statistics_right);

#endif

#if BIDIRECTIONAL_FILTERING
    apply_SIFT_filtering(stereo_edge_pairs_right, sift_dist_threshold, "output_files/right_edges", frame_idx, false);
    Evaluate_Stereo_Edge_Correspondences(stereo_edge_pairs_right, frame_idx, "SIFT Filtering (Right)",
                                         recall_per_image_right, precision_per_image_right, precision_pair_per_image_right, num_of_target_edges_per_source_edge_avg_right,
                                         evaluation_statistics_right);
    apply_bidirectional_test(stereo_frame_edge_pairs, stereo_edge_pairs_right, "output_files", frame_idx);
    Evaluate_Stereo_Edge_Correspondences(stereo_frame_edge_pairs, frame_idx, "Bidirectional Consistency Test",
                                         recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg,
                                         evaluation_statistics);
    frame_metrics.stages["Bidirectional Consistency Test"] = {recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg};
#endif

    //> Shift edges to the epipolar line and cluster them to consolidate redundant edge hypothesis
    bool b_shift_to_epipolar_line = true;
    bool b_cluster_edges = false;
    consolidate_redundant_edge_hypothesis(stereo_frame_edge_pairs, frame_idx, b_shift_to_epipolar_line, b_cluster_edges);

    //> Refine the edge disparity after redundant edge hypothesis are consolidated
    refine_edge_disparity(stereo_frame_edge_pairs, frame_idx, true);
    // evaluation_statistics.reset();
    Evaluate_Stereo_Edge_Correspondences(stereo_frame_edge_pairs, frame_idx, "Photometric Refinement",
                                         recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg,
                                         evaluation_statistics, false, true);
    record_Ambiguity_Distribution("photometric_refinement", stereo_frame_edge_pairs, "output_files", frame_idx);
    frame_metrics.stages["Photometric Refinement"] = {recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg};
#if BIDIRECTIONAL_FILTERING
    consolidate_redundant_edge_hypothesis(stereo_edge_pairs_right, frame_idx, b_shift_to_epipolar_line, b_cluster_edges);
    refine_edge_disparity(stereo_edge_pairs_right, frame_idx, false);
    Evaluate_Stereo_Edge_Correspondences(stereo_edge_pairs_right, frame_idx, "Photometric Refinement (Right)",
                                         recall_per_image_right, precision_per_image_right, precision_pair_per_image_right, num_of_target_edges_per_source_edge_avg_right,
                                         evaluation_statistics_right, false, true);
#endif

    consolidate_redundant_edge_hypothesis(stereo_frame_edge_pairs, false, true);
    Evaluate_Stereo_Edge_Correspondences(stereo_frame_edge_pairs, frame_idx, "Edge Clustering",
                                         recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg,
                                         evaluation_statistics);
    record_Ambiguity_Distribution("edge_clustering", stereo_frame_edge_pairs, "output_files", frame_idx);
    frame_metrics.stages["Edge Clustering"] = {recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg};
#if BIDIRECTIONAL_FILTERING
    consolidate_redundant_edge_hypothesis(stereo_edge_pairs_right, frame_idx, false, true);
    Evaluate_Stereo_Edge_Correspondences(stereo_edge_pairs_right, frame_idx, "Edge Clustering (Right)",
                                         recall_per_image_right, precision_per_image_right, precision_pair_per_image_right, num_of_target_edges_per_source_edge_avg_right,
                                         evaluation_statistics_right);
#endif
    // for (int i = 0; i < stereo_frame_edge_pairs.matching_edge_clusters[664].edge_clusters.size(); i++)
    // {
    //     std::cout << "Pre-Clustering NCC for left edge index 664, cluster " << i << ": " << stereo_frame_edge_pairs.matching_edge_clusters[664].edge_clusters[i].center_edge.location.x << ", orientation:" << stereo_frame_edge_pairs.matching_edge_clusters[664].edge_clusters[i].center_edge.orientation << ", score: " << stereo_frame_edge_pairs.matching_edge_clusters[664].refine_final_scores[i] << std::endl;
    // }
    apply_NCC_Filtering(stereo_frame_edge_pairs, "output_files", frame_idx);
    Evaluate_Stereo_Edge_Correspondences(stereo_frame_edge_pairs, frame_idx, "NCC Filtering(Post-Clustering)",
                                         recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg,
                                         evaluation_statistics);

    frame_metrics.stages["NCC Filtering(Post-Clustering)"] = {recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg};
#if BIDIRECTIONAL_FILTERING
    refine_edge_disparity(stereo_edge_pairs_right, frame_idx, false);
    Evaluate_Stereo_Edge_Correspondences(stereo_edge_pairs_right, frame_idx, "Photometric Refinement (Post-NCC) (Right)",
                                         recall_per_image_right, precision_per_image_right, precision_pair_per_image_right, num_of_target_edges_per_source_edge_avg_right,
                                         evaluation_statistics_right, false, true);
#endif

    // apply_Neighborhood_Consistency_Filter(stereo_frame_edge_pairs, spatial_grid_left, spatial_grid_right, 5.0, "output_files", frame_idx);

    // Evaluate_Stereo_Edge_Correspondences(stereo_frame_edge_pairs, frame_idx, "Neighborhood Consistency",
    //                                      recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg,
    //                                      evaluation_statistics);
    // frame_metrics.stages["Neighborhood Consistency"] = {recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg};

    //> Apply Lowe's ratio test after all filtering to disambiguate final candidates
    apply_Lowe_Ratio_Test(stereo_frame_edge_pairs, LOWES_RATIO, "output_files", frame_idx);
    Evaluate_Stereo_Edge_Correspondences(stereo_frame_edge_pairs, frame_idx, "Best",
                                         recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg,
                                         evaluation_statistics);
    frame_metrics.stages["Best"] = {recall_per_image, precision_per_image, precision_pair_per_image, num_of_target_edges_per_source_edge_avg};

#if BIDIRECTIONAL_FILTERING
    apply_Lowe_Ratio_Test(stereo_edge_pairs_right, LOWES_RATIO, "output_files/right_edges", frame_idx);
    Evaluate_Stereo_Edge_Correspondences(stereo_edge_pairs_right, frame_idx, "Lowe's Ratio Test (Right)",
                                         recall_per_image_right, precision_per_image_right, precision_pair_per_image_right, num_of_target_edges_per_source_edge_avg_right,
                                         evaluation_statistics_right);
#endif
    //> ============= END OF CH'S EDITIONS =============

    return frame_metrics;
}

void record_correspondences_for_visualization(const Stereo_Edge_Pairs &stereo_frame_edge_pairs,
                                              const std::string &output_dir,
                                              size_t frame_idx,
                                              int num_samples)
{
    std::vector<std::pair<int, int>> veridical_pairs;  // ≤1px: (left_edge_idx, cluster_idx)
    std::vector<std::pair<int, int>> inaccurate_pairs; // 1-3px
    std::vector<std::pair<int, int>> incorrect_pairs;  // >3px

    // Collect ALL correspondences categorized by accuracy
    for (size_t i = 0; i < stereo_frame_edge_pairs.matching_edge_clusters.size(); i++)
    {
        const auto &cluster_data = stereo_frame_edge_pairs.matching_edge_clusters[i];

        // Skip if no clusters for this edge
        if (cluster_data.edge_clusters.empty())
            continue;

        // Get the best candidate (first cluster with highest score)
        const EdgeCluster &best_cluster = cluster_data.edge_clusters[0];
        cv::Point2d best_match_location = best_cluster.center_edge.location;

        // Get GT location for this left edge
        cv::Point2d gt_location = stereo_frame_edge_pairs.GT_locations_from_left_edges[i];

        // Categorize by distance to ground truth
        double dist_to_gt = cv::norm(best_match_location - gt_location);

        if (dist_to_gt <= 1.0)
        {
            veridical_pairs.push_back({i, 0}); // Best cluster is at index 0
        }
        else if (dist_to_gt <= 3.0)
        {
            inaccurate_pairs.push_back({i, 0});
        }
        else
        {
            incorrect_pairs.push_back({i, 0});
        }
    }

    // Random sampling
    std::random_device rd;
    std::mt19937 gen(rd());

    auto sample_pairs = [&gen, num_samples](std::vector<std::pair<int, int>> &pairs)
    {
        if (pairs.size() > num_samples)
        {
            std::shuffle(pairs.begin(), pairs.end(), gen);
            pairs.resize(num_samples);
        }
    };

    sample_pairs(veridical_pairs);
    sample_pairs(inaccurate_pairs);
    sample_pairs(incorrect_pairs);

    // Write veridical pairs (≤1px)
    std::string veridical_file = output_dir + "/veridical_correspondences_frame_" + std::to_string(frame_idx) + ".txt";
    std::ofstream veridical_out(veridical_file);
    veridical_out << "# left_x left_y left_orientation right_x right_y right_orientation\n";

    for (const auto &pair : veridical_pairs)
    {
        int left_edge_idx = pair.first;
        int cluster_idx = pair.second;

        Edge left_edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(left_edge_idx);
        const EdgeCluster &cluster = stereo_frame_edge_pairs.matching_edge_clusters[left_edge_idx].edge_clusters[cluster_idx];

        veridical_out << left_edge.location.x << " "
                      << left_edge.location.y << " "
                      << left_edge.orientation << " "
                      << cluster.center_edge.location.x << " "
                      << cluster.center_edge.location.y << " "
                      << cluster.center_edge.orientation << "\n";
    }
    veridical_out.close();

    // Write inaccurate pairs (1-3px)
    std::string inaccurate_file = output_dir + "/inaccurate_correspondences_frame_" + std::to_string(frame_idx) + ".txt";
    std::ofstream inaccurate_out(inaccurate_file);
    inaccurate_out << "# left_x left_y left_orientation right_x right_y right_orientation\n";

    for (const auto &pair : inaccurate_pairs)
    {
        int left_edge_idx = pair.first;
        int cluster_idx = pair.second;

        Edge left_edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(left_edge_idx);
        const EdgeCluster &cluster = stereo_frame_edge_pairs.matching_edge_clusters[left_edge_idx].edge_clusters[cluster_idx];

        inaccurate_out << left_edge.location.x << " "
                       << left_edge.location.y << " "
                       << left_edge.orientation << " "
                       << cluster.center_edge.location.x << " "
                       << cluster.center_edge.location.y << " "
                       << cluster.center_edge.orientation << "\n";
    }
    inaccurate_out.close();

    // Write incorrect pairs (>3px)
    std::string incorrect_file = output_dir + "/incorrect_correspondences_frame_" + std::to_string(frame_idx) + ".txt";
    std::ofstream incorrect_out(incorrect_file);
    incorrect_out << "# left_x left_y left_orientation right_x right_y right_orientation\n";

    for (const auto &pair : incorrect_pairs)
    {
        int left_edge_idx = pair.first;
        int cluster_idx = pair.second;

        Edge left_edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(left_edge_idx);
        const EdgeCluster &cluster = stereo_frame_edge_pairs.matching_edge_clusters[left_edge_idx].edge_clusters[cluster_idx];

        incorrect_out << left_edge.location.x << " "
                      << left_edge.location.y << " "
                      << left_edge.orientation << " "
                      << cluster.center_edge.location.x << " "
                      << cluster.center_edge.location.y << " "
                      << cluster.center_edge.orientation << "\n";
    }
    incorrect_out.close();

    std::cout << "Recorded sampled correspondences to " << output_dir << ":\n"
              << "  Veridical (≤1px): " << veridical_pairs.size() << "\n"
              << "  Inaccurate (1-3px): " << inaccurate_pairs.size() << "\n"
              << "  Incorrect (>3px): " << incorrect_pairs.size() << std::endl;
}
