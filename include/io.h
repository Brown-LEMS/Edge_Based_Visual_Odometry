#ifndef IO_H
#define IO_H

#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "Dataset.h"
#include "utility.h"

//> Truncate incremental-append outputs so each program run starts from an empty file (append_* then writes headers on first line).
void reset_incremental_append_outputs()
{
    const std::string pose_path = "outputs/results_incremental_append.txt";
    const std::string eval_path = "outputs/evaluation_results_incremental_append.txt";
    std::ofstream(pose_path, std::ios::trunc).close();
    std::ofstream(eval_path, std::ios::trunc).close();
}

//> Append one pose line in TUM format; writes header if the file is missing or empty. Flush for crash safety.
void append_pose_to_file_tum_format(const std::pair<std::string, Camera_Pose>& pose, const std::string& file_name)
{
    const std::string path = "outputs/" + file_name + ".txt";
    std::ifstream probe(path, std::ios::ate | std::ios::binary);
    const bool file_nonempty = probe.is_open() && probe.tellg() > 0;
    probe.close();

    std::ofstream file(path, std::ios::app);
    if (!file) {
        LOG_FILE_ERROR("Failed to open the file for append: " + path);
        return;
    }
    //> write the header only if the file is empty
    if (!file_nonempty) {
        file << "timestamp tx ty tz qx qy qz qw" << std::endl;
    }
    const Camera_Pose& p = pose.second;
    file << pose.first << " " << p.t.x() << " " << p.t.y() << " " << p.t.z() << " "
         << p.q.x() << " " << p.q.y() << " " << p.q.z() << " " << p.q.w() << std::endl;
    file.flush();
}

//> Append evaluation results. Flush for crash safety.
void append_evaluation_results_to_file(const std::pair<std::string, Camera_Pose>& pose, const evaluation_results& eval_results, const std::string& file_name)
{
    const std::string path = "outputs/" + file_name + ".txt";
    std::ifstream probe(path, std::ios::ate | std::ios::binary);
    const bool file_nonempty = probe.is_open() && probe.tellg() > 0;
    probe.close();

    std::ofstream file(path, std::ios::app);
    if (!file) {
        LOG_FILE_ERROR("Failed to open the file for appending the evaluation results: " + path);
        return;
    }
    //> write the header only if the file is empty
    if (!file_nonempty) {
        file << "timestamp err_R err_t err_scale" << std::endl;
    }
    file << pose.first << " " << eval_results.err_R << " " << eval_results.err_t << " " << eval_results.err_scale << std::endl;
    file.flush();
}

void save_poses_to_file_tum_format(const std::vector<std::pair<std::string, Camera_Pose>>& poses, const std::string& file_name) 
{
    std::ofstream file;
    file.open("outputs/" + file_name + ".txt");
    file << "timestamp tx ty tz qx qy qz qw" << std::endl;
    for (size_t i = 0; i < poses.size(); ++i) {
        const std::string& time_stamp = poses[i].first;
        Camera_Pose pose = poses[i].second;
        file << time_stamp << " " << pose.t.x() << " " << pose.t.y() << " " << pose.t.z() \
                << " " << pose.q.x() << " " << pose.q.y() << " " << pose.q.z() << " " << pose.q.w() << std::endl;
    }
    LOG_STATUS("Saved poses to file: " + file_name + ".txt");
    file.close();
}

void save_evaluation_results_to_file(const std::vector<std::pair<std::string, evaluation_results>>& evaluation_results_paired_with_timestamp, const std::string& file_name) 
{
    std::ofstream file;
    file.open("outputs/" + file_name + ".txt");
    file << "timestamp err_R err_t err_scale" << std::endl;
    for (size_t i = 0; i < evaluation_results_paired_with_timestamp.size(); ++i) {
        const std::string& time_stamp = evaluation_results_paired_with_timestamp[i].first;
        evaluation_results eval_results = evaluation_results_paired_with_timestamp[i].second;
        file << time_stamp << " " << eval_results.err_R << " " << eval_results.err_t << " " << eval_results.err_scale << std::endl;
    }
    LOG_STATUS("Saved evaluation results to file: " + file_name + ".txt");
    file.close();
}

//> Write evaluation statistics to file, specifically the data from photometric refinement
inline void write_Evaluated_Photometric_Refinement_Data_to_file(Dataset &dataset, Evaluation_Statistics &evaluation_statistics, int frame_idx)
{
    std::string output_dir = dataset.get_output_path();
    std::string photo_refine_eval_filename = output_dir + "/photo_refine_data_from_evaluation_statistics_frame_" + std::to_string(frame_idx) + ".txt";
    std::cout << "[I/O] Writing evaluation statistics of photometric refinement to file: " << photo_refine_eval_filename << std::endl;
    std::ofstream photo_refine_eval_file(photo_refine_eval_filename);
    photo_refine_eval_file << "is_TP, left_edge_index, refine_final_score, refine_confidence, refine_validity" << std::endl;
    for (int i = 0; i < evaluation_statistics.edge_clusters_in_each_step["Photometric Refinement"].size(); i++)
    {
        photo_refine_eval_file << evaluation_statistics.edge_clusters_in_each_step["Photometric Refinement"][i].b_is_TP << " "
                               << evaluation_statistics.edge_clusters_in_each_step["Photometric Refinement"][i].paired_left_edge_index << " "
                               << evaluation_statistics.refine_final_scores[i] << " "
                               << evaluation_statistics.refine_confidences[i] << " "
                               << evaluation_statistics.refine_validities[i] << " "
                               << evaluation_statistics.edge_clusters_in_each_step["Photometric Refinement"][i].center_edge.location.x << " "
                               << evaluation_statistics.edge_clusters_in_each_step["Photometric Refinement"][i].center_edge.location.y << " "
                               << evaluation_statistics.edge_clusters_in_each_step["Photometric Refinement"][i].center_edge.orientation << " "
                               << std::endl;
    }
    photo_refine_eval_file.close();
}

//> Write evaluation statistics to file, specifically the data from photometric refinement
//> Stages:
//  Epipolar Line Distance Filtering, Maximal Disparity Filtering, Epipolar Shift and Clustering, Photometric Refinement
inline void write_Evaluated_Matching_Edge_Clusters_Data_to_file(Dataset &dataset, Stereo_Edge_Pairs &stereo_frame_edge_pairs, Evaluation_Statistics &evaluation_statistics, int frame_idx)
{
    std::string output_dir = dataset.get_output_path();
    std::string matching_edge_clusters_eval_filename = output_dir + "/matching_edge_clusters_data_frame_" + std::to_string(frame_idx) + ".txt";
    std::cout << "[I/O] Writing matching edge clusters data to file: " << matching_edge_clusters_eval_filename << std::endl;
    std::ofstream matching_edge_clusters_eval_file(matching_edge_clusters_eval_filename);
    matching_edge_clusters_eval_file << "left_edge_index, left_edge_location, left_edge_orientation, GT_location, shifting_center_edge_location, shifting_center_edge_orientation, photometric_refinement_center_edge_location, photometric_refinement_center_edge_orientation" << std::endl;
    for (int i = 0; i < evaluation_statistics.edge_clusters_in_each_step["Photometric Refinement"].size(); i++)
    {
        int left_edge_index = evaluation_statistics.edge_clusters_in_each_step["Photometric Refinement"][i].paired_left_edge_index;
        bool b_is_TP_in_shifting = evaluation_statistics.edge_clusters_in_each_step["Epipolar Shift and Clustering"][i].b_is_TP;
        bool b_is_TP_in_photometric_refinement = evaluation_statistics.edge_clusters_in_each_step["Photometric Refinement"][i].b_is_TP;
        if (b_is_TP_in_shifting && !b_is_TP_in_photometric_refinement)
        {
            matching_edge_clusters_eval_file << left_edge_index << " "
                                             << stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(left_edge_index).location.x << " "
                                             << stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(left_edge_index).location.y << " "
                                             << stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(left_edge_index).orientation << " "
                                             << stereo_frame_edge_pairs.GT_locations_from_left_edges[left_edge_index].x << " "
                                             << stereo_frame_edge_pairs.GT_locations_from_left_edges[left_edge_index].y << " "
                                             << evaluation_statistics.edge_clusters_in_each_step["Epipolar Shift and Clustering"][i].center_edge.location.x << " "
                                             << evaluation_statistics.edge_clusters_in_each_step["Epipolar Shift and Clustering"][i].center_edge.location.y << " "
                                             << evaluation_statistics.edge_clusters_in_each_step["Epipolar Shift and Clustering"][i].center_edge.orientation << " "
                                             << evaluation_statistics.edge_clusters_in_each_step["Photometric Refinement"][i].center_edge.location.x << " "
                                             << evaluation_statistics.edge_clusters_in_each_step["Photometric Refinement"][i].center_edge.location.y << " "
                                             << evaluation_statistics.edge_clusters_in_each_step["Photometric Refinement"][i].center_edge.orientation << " "
                                             << std::endl;
        }
    }
    matching_edge_clusters_eval_file.close();
}

// inline void write_3D_edges_to_file(Dataset &dataset, Stereo_Edge_Pairs& stereo_frame_edge_pairs, int frame_idx)
// {
//     std::string output_dir = dataset.get_output_path();
//     std::string edges_in_3D_filename = output_dir + "/3D_edges_frame_" + std::to_string(frame_idx) + ".txt";
//     std::ofstream edges_in_3D_file(edges_in_3D_filename);

//     for (int i = 0; i < stereo_frame_edge_pairs.Gamma_in_left_cam_coord.size(); i++)
//     {
//         edges_in_3D_file << stereo_frame_edge_pairs.Gamma_in_left_cam_coord[i].x() << " " \
//                          << stereo_frame_edge_pairs.Gamma_in_left_cam_coord[i].y() << " " \
//                          << stereo_frame_edge_pairs.Gamma_in_left_cam_coord[i].z() << " " \
//                          << stereo_frame_edge_pairs.triangulated_3D_tangents[i].x() << " " \
//                          << stereo_frame_edge_pairs.triangulated_3D_tangents[i].y() << " " \
//                          << stereo_frame_edge_pairs.triangulated_3D_tangents[i].z() << std::endl;
//     }
//     edges_in_3D_file.close();
// }

// inline void write_left_edge_pairs_and_3D_edges_to_file(Dataset &dataset, Stereo_Edge_Pairs &stereo_frame_edge_pairs, int frame_idx)
// {
//     std::string output_dir = dataset.get_output_path();
//     std::string edge_pairs_and_3D_edges_filename = output_dir + "/edge_pairs_and_3D_edges_frame_" + std::to_string(frame_idx) + ".txt";
//     std::ofstream edge_pairs_and_3D_edges_file(edge_pairs_and_3D_edges_filename);
//     for (int i = 0; i < stereo_frame_edge_pairs.focused_edge_indices.size(); i++)
//     {
//         Edge left_edge = stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i);
//         Eigen::Vector3d e_location_eigen(left_edge.location.x, left_edge.location.y, 1.0);
//         Eigen::Vector3d gamma_1 = dataset.get_left_calib_matrix().inverse() * e_location_eigen;
//         if (stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters.size() > 0)
//         {
//             for (int j = 0; j < stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters.size(); j++)
//             {
//                 Edge matching_cluster_edge = stereo_frame_edge_pairs.matching_edge_clusters[i].edge_clusters[j].center_edge;
//                 double disparity = left_edge.location.x - matching_cluster_edge.location.x;

//                 double rho = dataset.get_left_focal_length() * dataset.get_left_baseline() / disparity;
//                 double rho_1 = (rho < 0.0) ? (-rho) : (rho);
//                 Eigen::Vector3d Gamma_1 = rho_1 * gamma_1;
//                 edge_pairs_and_3D_edges_file << Gamma_1.x() << " " << Gamma_1.y() << " " << Gamma_1.z() << std::endl;
//             }
//         }
//     }
//     edge_pairs_and_3D_edges_file.close();
// }

//> Write false negative edge clusters to file
inline void write_False_Negative_Edge_Clusters_to_file(Dataset &dataset, Stereo_Edge_Pairs &stereo_frame_edge_pairs, Evaluation_Statistics &evaluation_statistics, int frame_idx)
{
    std::cout << "[I/O] Writing false negative edge clusters to file." << std::endl;
    std::string output_dir = dataset.get_output_path();
    std::string false_negative_edge_clusters_filename = output_dir + "/false_negative_edge_clusters_frame_" + std::to_string(frame_idx) + ".txt";
    std::ofstream false_negative_edge_clusters_file(false_negative_edge_clusters_filename);
    false_negative_edge_clusters_file << "left_edge_location, left_edge_orientation, GT_location, center_edge_location, center_edge_orientation, dist_error_to_GT" << std::endl;
    for (int i = 0; i < evaluation_statistics.edge_clusters_in_each_step["Epipolar Shift and Clustering"].size(); i++)
    {
        int left_edge_index = evaluation_statistics.edge_clusters_in_each_step["Epipolar Shift and Clustering"][i].paired_left_edge_index;
        false_negative_edge_clusters_file << stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(left_edge_index).location.x << " "
                                          << stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(left_edge_index).location.y << " "
                                          << stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(left_edge_index).orientation << " "
                                          << stereo_frame_edge_pairs.GT_locations_from_left_edges[left_edge_index].x << " "
                                          << stereo_frame_edge_pairs.GT_locations_from_left_edges[left_edge_index].y << " "
                                          << evaluation_statistics.false_negative_edge_clusters[i].center_edge.location.x << " "
                                          << evaluation_statistics.false_negative_edge_clusters[i].center_edge.location.y << " "
                                          << evaluation_statistics.false_negative_edge_clusters[i].center_edge.orientation << " "
                                          << evaluation_statistics.FN_dist_error_to_GT[i] << std::endl;
    }
    false_negative_edge_clusters_file.close();

    //> write the contributing edges of each false negative edge cluster to file
    std::string contributing_edges_filename = output_dir + "/false_negative_edge_clusters_contributing_edges_frame_" + std::to_string(frame_idx) + ".txt";
    std::ofstream contributing_edges_file(contributing_edges_filename);
    contributing_edges_file << "false_negative_edge_cluster_index, contributing_edge_shifted_location, contributing_edge_shifted_orientation, contributing_toed_location, contributing_toed_orientation" << std::endl;

    for (int i = 0; i < evaluation_statistics.edge_clusters_in_each_step["Epipolar Shift and Clustering"].size(); i++)
    {
        int left_edge_index = evaluation_statistics.edge_clusters_in_each_step["Epipolar Shift and Clustering"][i].paired_left_edge_index;
        for (int j = 0; j < evaluation_statistics.false_negative_edge_clusters[i].contributing_edges.size(); j++)
        {
            int toed_index = evaluation_statistics.false_negative_edge_clusters[i].contributing_edges_toed_indices[j];
            contributing_edges_file << i << " "
                                    << evaluation_statistics.false_negative_edge_clusters[i].contributing_edges[j].location.x << " "
                                    << evaluation_statistics.false_negative_edge_clusters[i].contributing_edges[j].location.y << " "
                                    << evaluation_statistics.false_negative_edge_clusters[i].contributing_edges[j].orientation << " "
                                    << stereo_frame_edge_pairs.stereo_frame->right_edges[toed_index].location.x << " "
                                    << stereo_frame_edge_pairs.stereo_frame->right_edges[toed_index].location.y << " "
                                    << stereo_frame_edge_pairs.stereo_frame->right_edges[toed_index].orientation << std::endl;
        }
    }
    contributing_edges_file.close();
}

//> Write stereo edge pairs to file
inline void write_Stereo_Edge_Pairs_to_file(Dataset::Ptr dataset, Stereo_Edge_Pairs &stereo_frame_edge_pairs, int frame_idx)
{
    std::string output_dir = dataset->get_output_path();
    std::string stereo_frame_edge_pairs_filename = output_dir + "/stereo_frame_edge_pairs_frame_" + std::to_string(frame_idx) + ".txt";
    std::ofstream stereo_frame_edge_pairs_file(stereo_frame_edge_pairs_filename);
    stereo_frame_edge_pairs_file << "focused_edge_indices, GT_locations_from_focused_edges" << std::endl;
    for (int i = 0; i < stereo_frame_edge_pairs.focused_edge_indices.size(); i++)
    {
        stereo_frame_edge_pairs_file << stereo_frame_edge_pairs.focused_edge_indices[i] << " "
                                     << stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i).location.x << " "
                                     << stereo_frame_edge_pairs.get_focused_edge_by_Stereo_Edge_Pairs_index(i).location.y << " "
                                     << stereo_frame_edge_pairs.GT_locations_from_left_edges[i].x << " "
                                     << stereo_frame_edge_pairs.GT_locations_from_left_edges[i].y << std::endl;
    }
    stereo_frame_edge_pairs_file.close();
}

//> write third-order edges to file
inline void write_Third_Order_Edges_to_file(Dataset &dataset, const StereoFrame &stereo_frame, int frame_idx, std::string left_or_right = "left")
{
    std::string output_dir = dataset.get_output_path();
    std::string third_order_edges_filename = output_dir + "/" + left_or_right + "_third_order_edges_frame_" + std::to_string(frame_idx) + ".txt";
    std::ofstream third_order_edges_file(third_order_edges_filename);
    third_order_edges_file << "edge_index, edge_location, edge_orientation" << std::endl;

    if (left_or_right == "left")
    {
        for (int i = 0; i < stereo_frame.left_edges.size(); i++)
        {
            third_order_edges_file << i << " "
                                   << stereo_frame.left_edges[i].location.x << " "
                                   << stereo_frame.left_edges[i].location.y << " "
                                   << stereo_frame.left_edges[i].orientation << std::endl;
        }
    }
    else if (left_or_right == "right")
    {
        for (int i = 0; i < stereo_frame.right_edges.size(); i++)
        {
            third_order_edges_file << i << " "
                                   << stereo_frame.right_edges[i].location.x << " "
                                   << stereo_frame.right_edges[i].location.y << " "
                                   << stereo_frame.right_edges[i].orientation << std::endl;
        }
    }
    third_order_edges_file.close();
}

inline void save_Poses_to_File_TUM_Format(Dataset &dataset, const std::vector<Camera_Pose>& poses, const std::vector<double>& timestamps, const std::string& file_name) 
{
    std::ofstream file;
    std::string output_dir = dataset.get_output_path();
    file.open(output_dir + "/" + file_name + ".txt");
    file << "timestamp tx ty tz qx qy qz qw" << std::endl;
    for (size_t i = 0; i < poses.size(); ++i) {
        int time_stamp = timestamps[i];
        Camera_Pose pose = poses[i];
        file << time_stamp << " " << pose.t.x() << " " << pose.t.y() << " " << pose.t.z() \
             << " " << pose.q.x() << " " << pose.q.y() << " " << pose.q.z() << " " << pose.q.w() << std::endl;
    }
    LOG_INFO("Saved poses to TUM format file: " + file_name + ".txt");
    file.close();
}

inline void save_Poses_to_File_KITTI_Format(Dataset &dataset, const std::vector<Camera_Pose>& poses, const std::string& file_name) 
{
    std::ofstream file;
    std::string output_dir = dataset.get_output_path();
    file.open(output_dir + "/" + file_name + ".txt");
    file << "" << std::endl;
    for (size_t i = 0; i < poses.size(); ++i) {
        Camera_Pose pose = poses[i];
        file << std::setprecision(9) \
             << pose.R(0, 0) << " " << pose.R(0, 1) << " " << pose.R(0, 2) << " " << pose.t.x() << " "
             << pose.R(1, 0) << " " << pose.R(1, 1) << " " << pose.R(1, 2) << " " << pose.t.y() << " "
             << pose.R(2, 0) << " " << pose.R(2, 1) << " " << pose.R(2, 2) << " " << pose.t.z() << "\n";
    }
    LOG_INFO("Saved poses to KITTI format file: " + file_name + ".txt");
    file.close();
}

#endif