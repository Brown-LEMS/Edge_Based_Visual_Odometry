#include "../include/EdgeClusterer.h"
#include <unordered_map>
#include <numeric>
#include <limits>
#include <cmath>

EdgeClusterer::EdgeClusterer(std::vector<Edge> edge_set, std::vector<int> toed_indices_of_shifted_edges, bool b_cluster_by_orientation)
    : Epip_Correct_Edges(edge_set), toed_indices_of_shifted_edges(toed_indices_of_shifted_edges), b_cluster_by_orientation(b_cluster_by_orientation)
{
    Num_Of_Epipolar_Corrected_H2_Edges = edge_set.size();
    //> Initialize each H2 edge as a single cluster. The cluster label of edge i is i-1, i.e., cluster_labels[i] = i-1
    cluster_labels.resize(Num_Of_Epipolar_Corrected_H2_Edges);
    std::iota(cluster_labels.begin(), cluster_labels.end(), 0); // Each point starts in its own cluster
}

//> If orientation (radians) is less than -90 + threshold/2, add 180 degrees
double EdgeClusterer::normalizeOrientation(double orientation)
{

    double wrap_threshold = -90.0 + (CLUSTER_ORIENT_THRESH / 2.0);
    // double orientation_degrees = orientation * (180.0 / M_PI);

    double normalized_orientation = orientation;
    if (rad_to_deg<double>(orientation) < wrap_threshold)
    {
        normalized_orientation += M_PI;
    }

    return normalized_orientation;
}

int EdgeClusterer::getClusterSize(int label)
{
    int size = 0;
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i)
    {
        if (cluster_labels[i] == label)
            size++;
    }
    return size;
};

std::tuple<double, double, double> EdgeClusterer::computeGaussianAverage(int label1, int label2)
{
    //> If label2 is -1 by default, compute for single cluster (label1 only)
    //> Otherwise, compute for merged cluster (label1 + label2)

    //> CH DOCUMENT:
    //> Given two clusters, find the corresponding H2 edges (with epipolar shifted).
    //  Then consider the two as a single cluster, calculate the centroid edge location and the distance of each edge member to the centroid edge location
    //  Finally, use the distance from the centroid edge location, calculate weighted average of the orientations from all the edge member
    //  The returned weighted average orientation is in degree

    // Calculate the geometric mean of the cluster(s)
    double sum_x = 0, sum_y = 0;
    int count = 0;

    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i)
    {
        if (cluster_labels[i] == label1 || (label2 != -1 && cluster_labels[i] == label2))
        {
            // sum_x += Epip_Correct_Edges(i, 0);
            // sum_y += Epip_Correct_Edges(i, 1);
            sum_x += Epip_Correct_Edges[i].location.x;
            sum_y += Epip_Correct_Edges[i].location.y;
            count++;
        }
    }

    if (count == 0)
        return std::make_tuple(0.0, 0.0, 0.0);

    double centroid_x = sum_x / count;
    double centroid_y = sum_y / count;

    // Calculate mean shift distance from centroid for this cluster
    double total_shift_from_centroid = 0.0;
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i)
    {
        if (cluster_labels[i] == label1 || (label2 != -1 && cluster_labels[i] == label2))
        {
            double dx = Epip_Correct_Edges[i].location.x - centroid_x;
            double dy = Epip_Correct_Edges[i].location.y - centroid_y;
            double distance_from_centroid = std::sqrt(dx * dx + dy * dy);
            total_shift_from_centroid += distance_from_centroid;
        }
    }
    double mean_shift_from_centroid = total_shift_from_centroid / count;

    // Calculate Gaussian-weighted averages for x, y, and orientation
    double sum_weighted_x = 0;
    double sum_weighted_y = 0;
    double sum_weighted_orientation = 0;
    double total_weight = 0;

    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i)
    {
        if (cluster_labels[i] == label1 || (label2 != -1 && cluster_labels[i] == label2))
        {
            double dx = Epip_Correct_Edges[i].location.x - centroid_x;
            double dy = Epip_Correct_Edges[i].location.y - centroid_y;
            double distance_from_centroid = std::sqrt(dx * dx + dy * dy);
            double gaussian_weight = std::exp(-0.5 * std::pow((distance_from_centroid - mean_shift_from_centroid) / CLUSTER_ORIENT_GAUSS_SIGMA, 2));

            sum_weighted_x += gaussian_weight * Epip_Correct_Edges[i].location.x;
            sum_weighted_y += gaussian_weight * Epip_Correct_Edges[i].location.y;
            sum_weighted_orientation += gaussian_weight * Epip_Correct_Edges[i].orientation;
            total_weight += gaussian_weight;
        }
    }

    double gaussian_weighted_x = sum_weighted_x / total_weight;
    double gaussian_weighted_y = sum_weighted_y / total_weight;
    double gaussian_weighted_orientation = sum_weighted_orientation / total_weight;

    return std::make_tuple(gaussian_weighted_x, gaussian_weighted_y, gaussian_weighted_orientation);
}

void EdgeClusterer::performClustering()
{
    std::vector<Edge> shifted_edges = Epip_Correct_Edges;
    // //> Track average orientations for each cluster in degrees
    // for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i) {
    //     double normalized_orient = normalizeOrientation( Epip_Correct_Edges[i].orientation );
    //     Epip_Correct_Edges[i].orientation = normalized_orient;
    //     cluster_avg_orientations[i] = normalized_orient;
    // }

    //> Merge clusters starting from closest pairs
    bool merged = true;
    while (merged)
    {
        merged = false;

        // For each point, find its nearest neighbor and merge if within threshold
        for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i)
        {
            double min_dist = std::numeric_limits<double>::max();
            int nearest = -1;

            // Find the nearest edge to the current edge
            for (int j = 0; j < Num_Of_Epipolar_Corrected_H2_Edges; ++j)
            {
                if (cluster_labels[i] != cluster_labels[j])
                {
                    // double dist = (Epip_Correct_Edges.row(i).head<2>() - Epip_Correct_Edges.row(j).head<2>()).norm();
                    double dist = cv::norm(Epip_Correct_Edges[i].location - Epip_Correct_Edges[j].location);

                    //> orient_i and orient_j are both in degrees
                    // double orient_i = cluster_avg_orientations[cluster_labels[i]];
                    // double orient_j = cluster_avg_orientations[cluster_labels[j]];
                    double orient_i = Epip_Correct_Edges[i].orientation;
                    double orient_j = Epip_Correct_Edges[j].orientation;
                    if (b_cluster_by_orientation)
                    {
                        if (dist < min_dist && dist < CLUSTER_DIST_THRESH && std::abs(orient_i - orient_j) < deg_to_rad<double>(CLUSTER_ORIENT_THRESH))
                        {
                            min_dist = dist;
                            nearest = j;
                        }
                    }
                    else
                    {
                        if (dist < min_dist && dist < CLUSTER_DIST_THRESH)
                        {
                            min_dist = dist;
                            nearest = j;
                        }
                    }
                }
            }
            // If found a nearest edge within threshold, merge clusters
            if (nearest != -1)
            {
                int old_label = cluster_labels[nearest];
                int new_label = cluster_labels[i];
                int size_old = getClusterSize(old_label);
                int size_new = getClusterSize(new_label);
                if (size_old + size_new <= MAX_CLUSTER_SIZE)
                {
                    //> Calculate new average orientation for the merged cluster
                    std::tuple<double, double, double> result = computeGaussianAverage(old_label, new_label);

                    double merged_orientation = std::get<2>(result);
                    // Update the average orientation of the merged cluster
                    cluster_avg_orientations[new_label] = merged_orientation;

                    // double normalized_merged_orient = normalizeOrientation(merged_orientation);
                    // cluster_avg_orientations[new_label] = normalized_merged_orient;

                    // Update all points in the smaller cluster
                    for (int k = 0; k < Num_Of_Epipolar_Corrected_H2_Edges; ++k)
                    {
                        if (cluster_labels[k] == old_label)
                        {
                            cluster_labels[k] = new_label;
                        }
                    }
                    merged = true;
                    break;
                }
            }
        }
    }

    //> Group hypothesis edge indices by their cluster label
    // example: cluster_labels = [0, 0, 1, 2, 1]
    // result: cluster 0: edge 0 and 1; cluster 1: edge 2 and 4; cluster 2: edge 3
    std::map<int, std::vector<int>> label_to_cluster;
    for (int i = 0; i < Num_Of_Epipolar_Corrected_H2_Edges; ++i)
    {
        label_to_cluster[cluster_labels[i]].push_back(i);
    }

    //////////// push to clusters////////////

    //> CH TODO DOCUMENTATION
    std::map<int, std::vector<int>>::iterator it;
    for (it = label_to_cluster.begin(); it != label_to_cluster.end(); ++it)
    {
        clusters.push_back(it->second);
    }

    Num_Of_Clusters = clusters.size();

    //> For each cluster, compute the Gaussian-weighted average edge and update all edges in the cluster
    for (size_t c = 0; c < clusters.size(); ++c)
    {
        const std::vector<int> &cluster = clusters[c];
        if (cluster.empty())
            continue;

        int cluster_label = cluster_labels[cluster[0]];
        std::tuple<double, double, double> result = computeGaussianAverage(cluster_label);
        double gaussian_average_x = std::get<0>(result);
        double gaussian_average_y = std::get<1>(result);
        double gaussian_average_orientation = std::get<2>(result);

        //> Create the Gaussian-weighted average edge
        cv::Point2d gaussian_edge(gaussian_average_x, gaussian_average_y);

        //> CH TODO: Pass the frame ID to the last input argument
        Edge gaussian_weighted_avg{gaussian_edge, gaussian_average_orientation, false, 0};

        // gaussian_weighted_avg << gaussian_average_x, gaussian_average_y, gaussian_average_orientation;

        // Find the edge closest to the Gaussian-weighted average to use as the representative
        double min_dist = std::numeric_limits<double>::max();
        int closest_idx = -1;
        for (size_t i = 0; i < cluster.size(); ++i)
        {
            int idx = cluster[i];
            // double dist = (Epip_Correct_Edges.row(idx).head<2>() - gaussian_weighted_avg.head<2>()).norm();
            double dist = cv::norm(Epip_Correct_Edges[idx].location - gaussian_weighted_avg.location);
            if (dist < min_dist)
            {
                min_dist = dist;
                closest_idx = idx;
            }
        }

        // Update all edges in the cluster with the average edge
        for (size_t i = 0; i < cluster.size(); ++i)
        {
            int idx = cluster[i];
            Epip_Correct_Edges[idx] = gaussian_weighted_avg;
            // Epip_Correct_Edges.row(idx) = gaussian_weighted_avg;
        }
    }

    std::vector<Edge> clustered_edges = Epip_Correct_Edges;

    //> Renumbering the cluster labels into 0, 1, 2, etc for each epipolar shifted edge
    std::vector<int> renumbered_cluster_labels = cluster_labels;
    std::vector<int> unique_cluster_labels = find_Unique_Sorted_Numbers(cluster_labels);
    for (int i = 0; i < unique_cluster_labels.size(); i++)
    {
        for (int j = 0; j < cluster_labels.size(); j++)
        {
            if (cluster_labels[j] == unique_cluster_labels[i])
            {
                renumbered_cluster_labels[j] = i;
            }
        }
    }

    int cluster_count = 0;
    returned_clusters.resize(Num_Of_Clusters);
    std::vector<bool> cluster_centers_filled(Num_Of_Clusters, false);
    for (int i = 0; i < renumbered_cluster_labels.size(); i++)
    {
        if (!cluster_centers_filled[renumbered_cluster_labels[i]] && renumbered_cluster_labels[i] == cluster_count)
        {
            returned_clusters[cluster_count].center_edge = clustered_edges[i];
            cluster_centers_filled[renumbered_cluster_labels[i]] = true;
            cluster_count++;
        }
        returned_clusters[renumbered_cluster_labels[i]].contributing_edges.push_back(shifted_edges[i]);
        // returned_clusters[renumbered_cluster_labels[i]].contributing_edges_toed_indices.push_back(toed_indices_of_shifted_edges[i]);
        // if (cluster_count == Num_Of_Clusters) break;
    }
}