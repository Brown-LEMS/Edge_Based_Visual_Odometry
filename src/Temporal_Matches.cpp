#include <filesystem>
#include <unordered_set>
#include <unordered_map>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include "Temporal_Matches.h"
#include "EdgeClusterer.h"
#include "definitions.h"
#include <opencv2/core/eigen.hpp>

Temporal_Matches::Temporal_Matches(Dataset::Ptr dataset) : dataset(std::move(dataset)) {}

void Temporal_Matches::add_edges_to_spatial_grid(const std::vector<final_stereo_edge_pair> &stereo_edge_mates, SpatialGrid &left_spatial_grids, SpatialGrid &right_spatial_grids)
{
    // Pre-compute grid cell assignments in parallel (read-only)
    std::vector<std::pair<int, int>> left_edge_to_grid(stereo_edge_mates.size());
    std::vector<std::pair<int, int>> right_edge_to_grid(stereo_edge_mates.size());

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(stereo_edge_mates.size()); ++i)
    {
        const cv::Point2d &left_loc = stereo_edge_mates[i].left_edge.location;
        const cv::Point2d &right_loc = stereo_edge_mates[i].right_edge.location;

        int lx = static_cast<int>(left_loc.x) / left_spatial_grids.cell_size;
        int ly = static_cast<int>(left_loc.y) / left_spatial_grids.cell_size;
        if (lx >= 0 && lx < left_spatial_grids.grid_width && ly >= 0 && ly < left_spatial_grids.grid_height)
            left_edge_to_grid[i] = {i, ly * left_spatial_grids.grid_width + lx};
        else
            left_edge_to_grid[i] = {i, -1};

        int rx = static_cast<int>(right_loc.x) / right_spatial_grids.cell_size;
        int ry = static_cast<int>(right_loc.y) / right_spatial_grids.cell_size;
        if (rx >= 0 && rx < right_spatial_grids.grid_width && ry >= 0 && ry < right_spatial_grids.grid_height)
            right_edge_to_grid[i] = {i, ry * right_spatial_grids.grid_width + rx};
        else
            right_edge_to_grid[i] = {i, -1};
    }

    for (size_t i = 0; i < stereo_edge_mates.size(); ++i)
    {
        int left_grid_idx = left_edge_to_grid[i].second;
        if (left_grid_idx >= 0 && left_grid_idx < static_cast<int>(left_spatial_grids.grid.size()))
            left_spatial_grids.grid[left_grid_idx].push_back(i);

        int right_grid_idx = right_edge_to_grid[i].second;
        if (right_grid_idx >= 0 && right_grid_idx < static_cast<int>(right_spatial_grids.grid.size()))
            right_spatial_grids.grid[right_grid_idx].push_back(i);
    }
}

void Temporal_Matches::build_Veridical_Quads(
    std::vector<KF_Temporal_Edge_Quads> &out,
    const std::vector<final_stereo_edge_pair> &KF_stereo_edge_mates,
    const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
    Stereo_Edge_Pairs &last_keyframe_stereo, Stereo_Edge_Pairs &current_frame_stereo,
    const SpatialGrid &left_spatial_grids, const SpatialGrid &right_spatial_grids)
{
    Utility util{};
    Camera_Pose rel_pose = util.get_Relative_Pose(last_keyframe_stereo.stereo_frame->gt_camera_pose, current_frame_stereo.stereo_frame->gt_camera_pose);

    const double orientation_threshold = 10.0;
    const double search_radius = 15.0 + DIST_TO_GT_THRESH_QUADS + 3.0;
    const int img_margin = 10;

    int num_threads_corr = omp_get_max_threads();
    std::vector<std::vector<KF_Temporal_Edge_Quads>> thread_quads(num_threads_corr);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, 128)
        for (int i = 0; i < static_cast<int>(KF_stereo_edge_mates.size()); ++i)
        {
            const final_stereo_edge_pair *kf_mate = &KF_stereo_edge_mates[i];

            Eigen::Vector3d Gamma_in_left_KF = kf_mate->Gamma_in_left_cam_coord;
            Eigen::Vector3d Gamma_in_left_CF = rel_pose.R * Gamma_in_left_KF + rel_pose.t;
            Eigen::Vector3d projected_point_left = dataset->get_left_calib_matrix() * Gamma_in_left_CF;
            projected_point_left /= projected_point_left.z();

            Eigen::Vector3d Gamma_in_right_CF = dataset->get_relative_rot_left_to_right() * Gamma_in_left_CF + dataset->get_relative_transl_left_to_right();
            Eigen::Vector3d projected_point_right = dataset->get_right_calib_matrix() * Gamma_in_right_CF;
            projected_point_right /= projected_point_right.z();

            double projected_orientation_left = orientation_mapping(
                kf_mate->left_edge, kf_mate->right_edge,
                projected_point_left, true, *last_keyframe_stereo.stereo_frame, *current_frame_stereo.stereo_frame, *dataset);
            double projected_orientation_right = orientation_mapping(
                kf_mate->left_edge, kf_mate->right_edge,
                projected_point_right, false, *last_keyframe_stereo.stereo_frame, *current_frame_stereo.stereo_frame, *dataset);

            cv::Point2d proj_left_cv(projected_point_left.x(), projected_point_left.y());
            cv::Point2d proj_right_cv(projected_point_right.x(), projected_point_right.y());
            if (projected_point_left.x() <= img_margin || projected_point_left.y() <= img_margin ||
                projected_point_left.x() >= dataset->get_width() - img_margin || projected_point_left.y() >= dataset->get_height() - img_margin)
                continue;
            if (projected_point_right.x() <= img_margin || projected_point_right.y() <= img_margin ||
                projected_point_right.x() >= dataset->get_width() - img_margin || projected_point_right.y() >= dataset->get_height() - img_margin)
                continue;

            std::vector<int> left_candidates = left_spatial_grids.getCandidatesWithinRadius(proj_left_cv, search_radius);
            std::vector<int> right_candidates = right_spatial_grids.getCandidatesWithinRadius(proj_right_cv, search_radius);
            std::unordered_set<int> right_set(right_candidates.begin(), right_candidates.end());

            std::vector<Veridical_Quad_Entry> quads;
            for (int cf_idx : left_candidates)
            {
                if (right_set.find(cf_idx) == right_set.end())
                    continue;
                if (cf_idx < 0 || cf_idx >= static_cast<int>(CF_stereo_edge_mates.size()))
                    continue;

                const Edge &cf_left = CF_stereo_edge_mates[cf_idx].left_edge;
                const Edge &cf_right = CF_stereo_edge_mates[cf_idx].right_edge;

                double dist_left = cv::norm(cf_left.location - proj_left_cv);
                double dist_right = cv::norm(cf_right.location - proj_right_cv);
                double orient_diff_left = std::abs(rad_to_deg<double>(projected_orientation_left - cf_left.orientation));
                if (orient_diff_left > 180.0)
                    orient_diff_left = 360.0 - orient_diff_left;
                double orient_diff_right = std::abs(rad_to_deg<double>(projected_orientation_right - cf_right.orientation));
                if (orient_diff_right > 180.0)
                    orient_diff_right = 360.0 - orient_diff_right;

                bool veridical_left = (dist_left < DIST_TO_GT_THRESH_QUADS) &&
                    (orient_diff_left < orientation_threshold || std::abs(orient_diff_left - 180.0) < orientation_threshold);
                bool veridical_right = (dist_right < DIST_TO_GT_THRESH_QUADS) &&
                    (orient_diff_right < orientation_threshold || std::abs(orient_diff_right - 180.0) < orientation_threshold);

                if (!veridical_left || !veridical_right)
                    continue;

                Veridical_Quad_Entry qe;
                qe.cf_stereo_edge_mate_index = cf_idx;
                qe.left_center = cf_left;
                qe.right_center = cf_right;
                quads.push_back(std::move(qe));
            }

            if (!quads.empty())
            {
                KF_Temporal_Edge_Quads kvq;
                kvq.KF_stereo_mate = kf_mate;
                kvq.projected_point_left = projected_point_left;
                kvq.projected_point_right = projected_point_right;
                kvq.projected_orientation_left = projected_orientation_left;
                kvq.projected_orientation_right = projected_orientation_right;
                kvq.veridical_quads = std::move(quads);
                thread_quads[tid].push_back(std::move(kvq));
            }
        }
    }

    out.clear();
    for (int t = 0; t < num_threads_corr; ++t)
    {
        for (auto &kvq : thread_quads[t])
            out.push_back(std::move(kvq));
    }
}

void Temporal_Matches::get_Temporal_Edge_Pairs_from_Quads(
    std::vector<KF_Temporal_Edge_Quads> &temporal_quads_by_kf,
    const std::vector<final_stereo_edge_pair> &KF_stereo_edge_mates,
    const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates,
    const SpatialGrid &left_spatial_grids, const SpatialGrid &right_spatial_grids,
    Stereo_Edge_Pairs &last_keyframe_stereo, Stereo_Edge_Pairs &current_frame_stereo,
    const StereoFrame &keyframe, const StereoFrame &current_frame,
    size_t keyframe_idx, size_t current_frame_idx)
{
    size_t num_quads = 0;
    for (const auto &kvq : temporal_quads_by_kf)
        num_quads += kvq.veridical_quads.size();
    std::cout << "Veridical quads: " << temporal_quads_by_kf.size() << " KF groups, " << num_quads << " total quads" << std::endl;

    apply_spatial_grid_filtering_quads(temporal_quads_by_kf, CF_stereo_edge_mates, left_spatial_grids, right_spatial_grids, 30.0);
    Evaluate_Temporal_Edge_Pairs_on_Quads(temporal_quads_by_kf, keyframe_idx, current_frame_idx, "Location Proximity Filtering");

    apply_orientation_filtering_quads(temporal_quads_by_kf, CF_stereo_edge_mates, 35.0);
    Evaluate_Temporal_Edge_Pairs_on_Quads(temporal_quads_by_kf, keyframe_idx, current_frame_idx, "Orientation Filtering");

    apply_NCC_filtering_quads(temporal_quads_by_kf, CF_stereo_edge_mates, 0.6,
        keyframe.left_image, keyframe.right_image, current_frame.left_image, current_frame.right_image);
    Evaluate_Temporal_Edge_Pairs_on_Quads(temporal_quads_by_kf, keyframe_idx, current_frame_idx, "NCC Filtering");

    apply_SIFT_filtering_quads(temporal_quads_by_kf, CF_stereo_edge_mates, 500.0);
    Evaluate_Temporal_Edge_Pairs_on_Quads(temporal_quads_by_kf, keyframe_idx, current_frame_idx, "SIFT Filtering");

    apply_best_nearly_best_filtering_quads(temporal_quads_by_kf, 0.8, "NCC");
    Evaluate_Temporal_Edge_Pairs_on_Quads(temporal_quads_by_kf, keyframe_idx, current_frame_idx, "BNB NCC Filtering");

    apply_best_nearly_best_filtering_quads(temporal_quads_by_kf, 0.5, "SIFT");
    Evaluate_Temporal_Edge_Pairs_on_Quads(temporal_quads_by_kf, keyframe_idx, current_frame_idx, "BNB SIFT Filtering");

    apply_photometric_refinement_quads(temporal_quads_by_kf, CF_stereo_edge_mates, keyframe, current_frame);
    Evaluate_Temporal_Edge_Pairs_on_Quads(temporal_quads_by_kf, keyframe_idx, current_frame_idx, "Photometric Refinement");

    apply_temporal_edge_clustering_quads(temporal_quads_by_kf, true);
    Evaluate_Temporal_Edge_Pairs_on_Quads(temporal_quads_by_kf, keyframe_idx, current_frame_idx, "Edge Clustering");

    apply_best_nearly_best_filtering_quads(temporal_quads_by_kf, 0.6, "SIFT");
    Evaluate_Temporal_Edge_Pairs_on_Quads(temporal_quads_by_kf, keyframe_idx, current_frame_idx, "BNB SIFT Filtering");
}

void Temporal_Matches::Evaluate_Temporal_Edge_Pairs_on_Quads(
    std::vector<KF_Temporal_Edge_Quads> &temporal_quads_by_kf,
    const size_t keyframe_idx, const size_t current_frame_idx, const std::string &stage_name)
{
    double recall_per_temporal_image = 0.0;
    double precision_per_temporal_image = 0.0;
    double ambiguity_per_temporal_image = 0.0;
    int num_of_KF_stereo_TP_edge_mates = 0;

    for (auto &kvq : temporal_quads_by_kf)
    {
        //> Skip if the stereo edge pair on KF is itself not a true positive
        if (!kvq.KF_stereo_mate->b_is_TP)
            continue;

        //> GT location of the left and right CF edges
        cv::Point2d gt_CF_left_location(kvq.projected_point_left.x(), kvq.projected_point_left.y());
        cv::Point2d gt_CF_right_location(kvq.projected_point_right.x(), kvq.projected_point_right.y());
        kvq.b_is_TP.resize(kvq.candidate_quads.size(), false);
        int candidate_quad_index = 0;
        int num_TP_centers = 0;
        for (const auto &cq : kvq.candidate_quads)
        {
            cv::Point2d left_center = cq.CF_left->center_edge.location;
            cv::Point2d right_center = cq.CF_right->center_edge.location;
            double dist_left = cv::norm(left_center - gt_CF_left_location);
            double dist_right = cv::norm(right_center - gt_CF_right_location);
            if (dist_left < DIST_TO_GT_THRESH_QUADS && dist_right < DIST_TO_GT_THRESH_QUADS)
            {
                kvq.b_is_TP[candidate_quad_index] = true;
                num_TP_centers++;
            }
            else {
                kvq.b_is_TP[candidate_quad_index] = false;
            }
            candidate_quad_index++;
        }
        size_t num_clusters = kvq.candidate_quads.size();
        double recall_per_edge = (num_TP_centers >= 1) ? 1.0 : 0.0;
        double precision_per_edge = (num_clusters == 0) ? 0.0 : (static_cast<double>(num_TP_centers) / static_cast<double>(num_clusters));
        double ambiguity_per_edge = static_cast<double>(num_clusters);

        recall_per_temporal_image += recall_per_edge;
        precision_per_temporal_image += precision_per_edge;
        ambiguity_per_temporal_image += ambiguity_per_edge;

        num_of_KF_stereo_TP_edge_mates++;
    }

    if (num_of_KF_stereo_TP_edge_mates == 0)
    {
        print_eval_results_with_no_quads(keyframe_idx, current_frame_idx, stage_name);
        return;
    }

    double recall = recall_per_temporal_image / static_cast<double>(num_of_KF_stereo_TP_edge_mates);
    double precision = precision_per_temporal_image / static_cast<double>(num_of_KF_stereo_TP_edge_mates);
    double ambiguity_avg = ambiguity_per_temporal_image / static_cast<double>(num_of_KF_stereo_TP_edge_mates);

    std::cout << "Quads Evaluation: Key Frame (" << keyframe_idx << ") <-> Current Frame (" << current_frame_idx << ") | Stage: " << stage_name << std::endl;
    std::cout << "- Recall rate:       " << std::fixed << std::setprecision(8) << recall << std::endl;
    std::cout << "- Precision rate:    " << std::fixed << std::setprecision(8) << precision << std::endl;
    std::cout << "- Average Ambiguity: " << std::fixed << std::setprecision(8) << ambiguity_avg << std::endl;
    std::cout << "========================================================\n" << std::endl;
}

double Temporal_Matches::orientation_mapping(const Edge &e_left, const Edge &e_right, const Eigen::Vector3d projected_point, bool is_left_cam, const StereoFrame &last_keyframe, const StereoFrame &current_frame, Dataset &dataset)
{
    Utility util{};
    // Step 1: Get the stereo baseline rotation (Left -> Right)
    Eigen::Matrix3d R_stereo = dataset.get_relative_rot_left_to_right();
    Camera_Pose rel_pose = util.get_Relative_Pose(last_keyframe.gt_camera_pose, current_frame.gt_camera_pose);

    // Step 2: Reconstruct 3D direction T_1 in Left KF
    Eigen::Vector3d t1(cos(e_left.orientation), sin(e_left.orientation), 0);
    Eigen::Vector3d t2(cos(e_right.orientation), sin(e_right.orientation), 0);
    t1 = dataset.get_left_calib_matrix().inverse() * t1;
    t2 = dataset.get_right_calib_matrix().inverse() * t2;

    Eigen::Vector3d gamma_1(e_left.location.x, e_left.location.y, 1.0);
    gamma_1 = dataset.get_left_calib_matrix().inverse() * gamma_1;
    Eigen::Vector3d gamma_2(e_right.location.x, e_right.location.y, 1.0);
    gamma_2 = dataset.get_right_calib_matrix().inverse() * gamma_2;

    Eigen::Vector3d T_1 = util.reconstruct_3D_Tangent(R_stereo, gamma_1, gamma_2, t1, t2);

    // Step 3: Transform T_1 to current frame
    // Eigen::Matrix3d R_temporal = current_frame.gt_rotation * last_keyframe.gt_rotation.transpose(); // Left KF -> Left CF

    Eigen::Vector3d T_2;
    if (is_left_cam)
    {
        T_2 = rel_pose.R * T_1; // Left KF -> Left CF
    }
    else
    {
        T_2 = R_stereo * rel_pose.R * T_1; // Left KF -> Left CF -> Right CF
    }

    // Step 4: Project T_2 to image
    Eigen::Vector3d gamma_cf = projected_point / projected_point.z();
    if (is_left_cam)
        gamma_cf = dataset.get_left_calib_matrix().inverse() * gamma_cf;
    else
        gamma_cf = dataset.get_right_calib_matrix().inverse() * gamma_cf;

    Eigen::Vector3d t = util.project_3D_Tangent_to_2D_Tangent(T_2, gamma_cf);
    return atan2(t.y(), t.x());
}

void Temporal_Matches::apply_spatial_grid_filtering_quads( \
    std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, \
    const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, \
    const SpatialGrid &left_spatial_grids, const SpatialGrid &right_spatial_grids, double grid_radius)
{
    //> Quad-forming subset = left_set ∩ right_set (same index = same CF stereo pair).
    //> Each cf_idx gives both left and right; form Candidate_Quad_Entry from index.
    candidate_cluster_pairs_.resize(quads_by_kf.size());
#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < static_cast<int>(quads_by_kf.size()); ++i)
    {
        KF_Temporal_Edge_Quads &kvq = quads_by_kf[i];
        if (kvq.veridical_quads.empty())
            continue;

        cv::Point2d query_left = kvq.KF_stereo_mate->left_edge.location;
        cv::Point2d query_right = kvq.KF_stereo_mate->right_edge.location;
        std::vector<int> left_candidates = left_spatial_grids.getCandidatesWithinRadius(query_left, grid_radius);
        std::vector<int> right_candidates = right_spatial_grids.getCandidatesWithinRadius(query_right, grid_radius);
        std::unordered_set<int> right_set(right_candidates.begin(), right_candidates.end());

        std::vector<std::pair<Temporal_CF_Edge_Cluster, Temporal_CF_Edge_Cluster>> pairs;
        for (int cf_idx : left_candidates)
        {
            if (!right_set.count(cf_idx) || cf_idx < 0 || cf_idx >= static_cast<int>(CF_stereo_edge_mates.size()))
                continue;
            const auto &cf = CF_stereo_edge_mates[cf_idx];
            Temporal_CF_Edge_Cluster left_cl;
            left_cl.cf_stereo_edge_mate_index = cf_idx;
            left_cl.contributing_cf_stereo_indices = {cf_idx};
            left_cl.center_edge = cf.left_edge;
            left_cl.matching_scores = scores{-1.0, 900.0};
            Temporal_CF_Edge_Cluster right_cl;
            right_cl.cf_stereo_edge_mate_index = cf_idx;
            right_cl.contributing_cf_stereo_indices = {cf_idx};
            right_cl.center_edge = cf.right_edge;
            right_cl.matching_scores = scores{-1.0, 900.0};
            pairs.emplace_back(std::move(left_cl), std::move(right_cl));
        }

        candidate_cluster_pairs_[i] = std::move(pairs);
        kvq.candidate_quads.clear();
        for (size_t j = 0; j < candidate_cluster_pairs_[i].size(); ++j)
        {
            auto &p = candidate_cluster_pairs_[i][j];
            kvq.candidate_quads.push_back({&p.first, &p.second});
        }
    }
}

void Temporal_Matches::apply_orientation_filtering_quads(std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, double orientation_threshold)
{
#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < static_cast<int>(quads_by_kf.size()); ++i)
    {
        KF_Temporal_Edge_Quads &kvq = quads_by_kf[i];
        std::vector<std::pair<Temporal_CF_Edge_Cluster, Temporal_CF_Edge_Cluster>> new_pairs;
        for (const auto &cq : kvq.candidate_quads)
        {
            double orient_diff_left = std::abs(rad_to_deg<double>(kvq.KF_stereo_mate->left_edge.orientation - cq.CF_left->center_edge.orientation));
            if (orient_diff_left > 180.0)
                orient_diff_left = 360.0 - orient_diff_left;
            double orient_diff_right = std::abs(rad_to_deg<double>(kvq.KF_stereo_mate->right_edge.orientation - cq.CF_right->center_edge.orientation));
            if (orient_diff_right > 180.0)
                orient_diff_right = 360.0 - orient_diff_right;
            if ((orient_diff_left < orientation_threshold || std::abs(orient_diff_left - 180.0) < orientation_threshold) &&
                (orient_diff_right < orientation_threshold || std::abs(orient_diff_right - 180.0) < orientation_threshold))
            {
                new_pairs.emplace_back(*cq.CF_left, *cq.CF_right);
            }
        }
        candidate_cluster_pairs_[i] = std::move(new_pairs);
        kvq.candidate_quads.clear();
        for (size_t j = 0; j < candidate_cluster_pairs_[i].size(); ++j)
        {
            auto &p = candidate_cluster_pairs_[i][j];
            kvq.candidate_quads.push_back({&p.first, &p.second});
        }
    }
}

void Temporal_Matches::apply_NCC_filtering_quads(std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, double ncc_val_threshold,
    const cv::Mat &keyframe_left_image, const cv::Mat &keyframe_right_image, const cv::Mat &cf_left_image, const cv::Mat &cf_right_image)
{
    Utility util{};
    cv::Mat kf_left_64f, kf_right_64f, cf_left_64f, cf_right_64f;
    keyframe_left_image.convertTo(kf_left_64f, CV_64F);
    keyframe_right_image.convertTo(kf_right_64f, CV_64F);
    cf_left_image.convertTo(cf_left_64f, CV_64F);
    cf_right_image.convertTo(cf_right_64f, CV_64F);

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < static_cast<int>(quads_by_kf.size()); ++i)
    {
        KF_Temporal_Edge_Quads &kvq = quads_by_kf[i];
        std::pair<cv::Mat, cv::Mat> kf_left_patches = kvq.KF_stereo_mate->left_edge_patches;
        std::pair<cv::Mat, cv::Mat> kf_right_patches = kvq.KF_stereo_mate->right_edge_patches;

        std::vector<std::pair<Temporal_CF_Edge_Cluster, Temporal_CF_Edge_Cluster>> new_pairs;
        for (const auto &cq : kvq.candidate_quads)
        {
            int cf_idx = cq.CF_left->cf_stereo_edge_mate_index;
            if (cf_idx < 0 || cf_idx >= static_cast<int>(CF_stereo_edge_mates.size()))
                continue;
            const auto &cf = CF_stereo_edge_mates[cf_idx];
            double sim_left = std::max({
                util.get_patch_similarity(kf_left_patches.first, cf.left_edge_patches.first),
                util.get_patch_similarity(kf_left_patches.first, cf.left_edge_patches.second),
                util.get_patch_similarity(kf_left_patches.second, cf.left_edge_patches.first),
                util.get_patch_similarity(kf_left_patches.second, cf.left_edge_patches.second)
            });
            double sim_right = std::max({
                util.get_patch_similarity(kf_right_patches.first, cf.right_edge_patches.first),
                util.get_patch_similarity(kf_right_patches.first, cf.right_edge_patches.second),
                util.get_patch_similarity(kf_right_patches.second, cf.right_edge_patches.first),
                util.get_patch_similarity(kf_right_patches.second, cf.right_edge_patches.second)
            });
            if (sim_left > ncc_val_threshold && sim_right > ncc_val_threshold)
            {
                Temporal_CF_Edge_Cluster left_cl = *cq.CF_left;
                Temporal_CF_Edge_Cluster right_cl = *cq.CF_right;
                left_cl.matching_scores.ncc_score = sim_left;
                right_cl.matching_scores.ncc_score = sim_right;
                new_pairs.emplace_back(std::move(left_cl), std::move(right_cl));
            }
        }
        candidate_cluster_pairs_[i] = std::move(new_pairs);
        kvq.candidate_quads.clear();
        for (size_t j = 0; j < candidate_cluster_pairs_[i].size(); ++j)
        {
            auto &p = candidate_cluster_pairs_[i][j];
            kvq.candidate_quads.push_back({&p.first, &p.second});
        }
    }
}

void Temporal_Matches::apply_SIFT_filtering_quads(std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, double sift_dist_threshold)
{
#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < static_cast<int>(quads_by_kf.size()); ++i)
    {
        KF_Temporal_Edge_Quads &kvq = quads_by_kf[i];
        std::pair<cv::Mat, cv::Mat> kf_left_desc = kvq.KF_stereo_mate->left_edge_descriptors;
        std::pair<cv::Mat, cv::Mat> kf_right_desc = kvq.KF_stereo_mate->right_edge_descriptors;

        std::vector<std::pair<Temporal_CF_Edge_Cluster, Temporal_CF_Edge_Cluster>> new_pairs;
        auto min_sift = [](const std::pair<cv::Mat, cv::Mat> &a, const std::pair<cv::Mat, cv::Mat> &b) {
            if (a.first.empty() || b.first.empty())
                return 900.0;
            double d1 = cv::norm(a.first, b.first, cv::NORM_L2);
            double d2 = cv::norm(a.first, b.second, cv::NORM_L2);
            double d3 = cv::norm(a.second, b.first, cv::NORM_L2);
            double d4 = cv::norm(a.second, b.second, cv::NORM_L2);
            return std::min({d1, d2, d3, d4});
        };
        for (const auto &cq : kvq.candidate_quads)
        {
            int cf_idx = cq.CF_left->cf_stereo_edge_mate_index;
            if (cf_idx < 0 || cf_idx >= static_cast<int>(CF_stereo_edge_mates.size()))
                continue;
            const auto &cf = CF_stereo_edge_mates[cf_idx];
            double sift_left = min_sift(kvq.KF_stereo_mate->left_edge_descriptors, cf.left_edge_descriptors);
            double sift_right = min_sift(kvq.KF_stereo_mate->right_edge_descriptors, cf.right_edge_descriptors);
            if (sift_left < sift_dist_threshold && sift_right < sift_dist_threshold)
            {
                Temporal_CF_Edge_Cluster left_cl = *cq.CF_left;
                Temporal_CF_Edge_Cluster right_cl = *cq.CF_right;
                left_cl.matching_scores.sift_score = sift_left;
                right_cl.matching_scores.sift_score = sift_right;
                new_pairs.emplace_back(std::move(left_cl), std::move(right_cl));
            }
        }
        candidate_cluster_pairs_[i] = std::move(new_pairs);
        kvq.candidate_quads.clear();
        for (size_t j = 0; j < candidate_cluster_pairs_[i].size(); ++j)
        {
            auto &p = candidate_cluster_pairs_[i][j];
            kvq.candidate_quads.push_back({&p.first, &p.second});
        }
    }
}

void Temporal_Matches::apply_best_nearly_best_filtering_quads(std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, double threshold, const std::string scoring_type)
{
    bool is_NCC = scoring_type == "NCC";
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(quads_by_kf.size()); ++i)
    {
        KF_Temporal_Edge_Quads &kvq = quads_by_kf[i];
        size_t n = kvq.candidate_quads.size();
        if (n < 2)
            continue;

        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        if (is_NCC)
        {
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                return kvq.candidate_quads[a].CF_left->matching_scores.ncc_score > kvq.candidate_quads[b].CF_left->matching_scores.ncc_score;
            });
        }
        else
        {
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                return kvq.candidate_quads[a].CF_left->matching_scores.sift_score < kvq.candidate_quads[b].CF_left->matching_scores.sift_score;
            });
        }

        double best = is_NCC ? kvq.candidate_quads[indices[0]].CF_left->matching_scores.ncc_score : kvq.candidate_quads[indices[0]].CF_left->matching_scores.sift_score;
        size_t keep = 1;
        for (size_t j = 0; j < n - 1; ++j)
        {
            double next = is_NCC ? kvq.candidate_quads[indices[j + 1]].CF_left->matching_scores.ncc_score : kvq.candidate_quads[indices[j + 1]].CF_left->matching_scores.sift_score;
            if (best == 0)
                break;
            double ratio = is_NCC ? next / best : best / next;
            if (ratio >= threshold)
                keep++;
            else
                break;
        }
        std::vector<std::pair<Temporal_CF_Edge_Cluster, Temporal_CF_Edge_Cluster>> new_pairs;
        for (size_t k = 0; k < keep; ++k)
        {
            size_t idx = indices[k];
            new_pairs.emplace_back(*kvq.candidate_quads[idx].CF_left, *kvq.candidate_quads[idx].CF_right);
        }
        candidate_cluster_pairs_[i] = std::move(new_pairs);
        kvq.candidate_quads.clear();
        for (size_t j = 0; j < candidate_cluster_pairs_[i].size(); ++j)
        {
            auto &p = candidate_cluster_pairs_[i][j];
            kvq.candidate_quads.push_back({&p.first, &p.second});
        }
    }
}

void Temporal_Matches::apply_photometric_refinement_quads(std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, const std::vector<final_stereo_edge_pair> &CF_stereo_edge_mates, const StereoFrame &keyframe, const StereoFrame &current_frame)
{
    cv::Mat kf_left_32f, kf_right_32f, cf_left_32f, cf_right_32f;
    keyframe.left_image_undistorted.convertTo(kf_left_32f, CV_32F);
    keyframe.right_image_undistorted.convertTo(kf_right_32f, CV_32F);
    current_frame.left_image_undistorted.convertTo(cf_left_32f, CV_32F);
    current_frame.right_image_undistorted.convertTo(cf_right_32f, CV_32F);

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < static_cast<int>(quads_by_kf.size()); ++i)
    {
        KF_Temporal_Edge_Quads &kvq = quads_by_kf[i];
        const Edge &kf_left = kvq.KF_stereo_mate->left_edge;
        const Edge &kf_right = kvq.KF_stereo_mate->right_edge;

        for (size_t j = 0; j < kvq.candidate_quads.size(); ++j)
        {
            Temporal_CF_Edge_Cluster &left_cl = candidate_cluster_pairs_[i][j].first;
            Temporal_CF_Edge_Cluster &right_cl = candidate_cluster_pairs_[i][j].second;
            int cf_idx = left_cl.cf_stereo_edge_mate_index;
            if (cf_idx < 0 || cf_idx >= static_cast<int>(CF_stereo_edge_mates.size()))
            {
                left_cl.refine_final_score = 1e6;
                left_cl.refine_validity = false;
                right_cl.refine_final_score = 1e6;
                right_cl.refine_validity = false;
                continue;
            }
            const Edge &cf_left = CF_stereo_edge_mates[cf_idx].left_edge;
            const Edge &cf_right = CF_stereo_edge_mates[cf_idx].right_edge;

            Eigen::Vector2d init_left(kf_left.location.x - cf_left.location.x, kf_left.location.y - cf_left.location.y);
            Eigen::Vector2d init_right(kf_right.location.x - cf_right.location.x, kf_right.location.y - cf_right.location.y);
            Eigen::Vector2d refined_left, refined_right;
            double score_left, score_right;
            bool valid_left, valid_right;
            std::vector<double> log_l, log_r;

            min_Edge_Photometric_Residual_by_Gauss_Newton(kf_left, cf_left, init_left, kf_left_32f, cf_left_32f,
                current_frame.left_image_gradients_x, current_frame.left_image_gradients_y,
                refined_left, score_left, valid_left, log_l, 20, 1e-3, 3.0, false);
            min_Edge_Photometric_Residual_by_Gauss_Newton(kf_right, cf_right, init_right, kf_right_32f, cf_right_32f,
                current_frame.right_image_gradients_x, current_frame.right_image_gradients_y,
                refined_right, score_right, valid_right, log_r, 20, 1e-3, 3.0, false);

            bool refine_valid = valid_left && valid_right;
            left_cl.refine_final_score = score_left;
            left_cl.refine_validity = refine_valid;
            right_cl.refine_final_score = score_right;
            right_cl.refine_validity = refine_valid;
            if (valid_left)
            {
                left_cl.center_edge.location.x = kf_left.location.x - refined_left[0];
                left_cl.center_edge.location.y = kf_left.location.y - refined_left[1];
            }
            if (valid_right)
            {
                right_cl.center_edge.location.x = kf_right.location.x - refined_right[0];
                right_cl.center_edge.location.y = kf_right.location.y - refined_right[1];
            }
        }
    }
}

void Temporal_Matches::apply_temporal_edge_clustering_quads(std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, bool b_cluster_by_orientation)
{
#pragma omp parallel for schedule(dynamic, 64)
    for (int ki = 0; ki < static_cast<int>(quads_by_kf.size()); ++ki)
    {
        KF_Temporal_Edge_Quads &kvq = quads_by_kf[ki];
        if (kvq.candidate_quads.size() < 2)
            continue;

        std::vector<Edge> shifted_left;
        std::vector<int> cf_indices;
        std::vector<double> refine_scores;
        shifted_left.reserve(kvq.candidate_quads.size());
        cf_indices.reserve(kvq.candidate_quads.size());
        refine_scores.reserve(kvq.candidate_quads.size());
        for (const auto &cq : kvq.candidate_quads)
        {
            shifted_left.push_back(cq.CF_left->center_edge);
            cf_indices.push_back(cq.CF_left->cf_stereo_edge_mate_index);
            refine_scores.push_back(cq.CF_left->refine_final_score);
        }

        EdgeClusterer edge_cluster_engine(shifted_left, cf_indices, b_cluster_by_orientation);
        edge_cluster_engine.setRefineScores(refine_scores);
        edge_cluster_engine.performClustering();

        std::vector<std::pair<Temporal_CF_Edge_Cluster, Temporal_CF_Edge_Cluster>> merged_pairs;
        merged_pairs.reserve(edge_cluster_engine.returned_clusters.size());

        for (const EdgeCluster &ec : edge_cluster_engine.returned_clusters)
        {
            std::unordered_set<int> merged_cf;
            int best_idx = -1;
            std::vector<Edge> right_edges_sub;
            std::vector<int> cf_sub;

            //> for each contrib, find the closest shifted_left edge (no hard constraint on distance)
            for (const Edge &contrib : ec.contributing_edges)
            {
                int closest_i = -1;
                double closest_dist = std::numeric_limits<double>::max();
                for (size_t i = 0; i < shifted_left.size(); ++i)
                {
                    double d = cv::norm(contrib.location - shifted_left[i].location);
                    if (d < closest_dist)
                    {
                        closest_dist = d;
                        closest_i = static_cast<int>(i);
                    }
                }
                if (closest_i >= 0)
                {
                    merged_cf.insert(cf_indices[closest_i]);
                    right_edges_sub.push_back(kvq.candidate_quads[closest_i].CF_right->center_edge);
                    cf_sub.push_back(cf_indices[closest_i]);
                    best_idx = closest_i;
                }
            }
            if (best_idx < 0 || right_edges_sub.empty())
                continue;

            Edge right_center;
            if (right_edges_sub.size() == 1)
            {
                right_center = right_edges_sub[0];
            }
            else
            {
                double sum_x = 0, sum_y = 0, sum_orientation = 0;
                for (const Edge &e : right_edges_sub)
                {
                    sum_x += e.location.x;
                    sum_y += e.location.y;
                    sum_orientation += e.orientation;
                }
                int n = static_cast<int>(right_edges_sub.size());
                right_center.location.x = sum_x / n;
                right_center.location.y = sum_y / n;
                right_center.orientation = sum_orientation / n;
            }

            Temporal_CF_Edge_Cluster left_cl = *kvq.candidate_quads[best_idx].CF_left;
            Temporal_CF_Edge_Cluster right_cl = *kvq.candidate_quads[best_idx].CF_right;
            left_cl.center_edge = ec.center_edge;
            right_cl.center_edge = right_center;
            left_cl.cf_stereo_edge_mate_index = cf_indices[best_idx];
            right_cl.cf_stereo_edge_mate_index = cf_indices[best_idx];
            merged_pairs.emplace_back(std::move(left_cl), std::move(right_cl));
        }
        candidate_cluster_pairs_[ki] = std::move(merged_pairs);
        kvq.candidate_quads.clear();
        for (size_t j = 0; j < candidate_cluster_pairs_[ki].size(); ++j)
        {
            auto &p = candidate_cluster_pairs_[ki][j];
            kvq.candidate_quads.push_back({&p.first, &p.second});
        }
    }
}

void Temporal_Matches::min_Edge_Photometric_Residual_by_Gauss_Newton(
    /* inputs */
    Edge kf_edge, Edge cf_edge, Eigen::Vector2d init_disp, const cv::Mat &kf_image_undistorted,
    const cv::Mat &cf_image_undistorted, const cv::Mat &cf_image_gradients_x, const cv::Mat &cf_image_gradients_y,
    /* outputs */
    Eigen::Vector2d &refined_disparity, double &refined_final_score, bool &refined_validity, std::vector<double> &residual_log,
    /* optional inputs */
    int max_iter, double tol, double huber_delta, bool b_verbose)
{
    cv::Point2d t(std::cos(kf_edge.orientation), std::sin(kf_edge.orientation));
    cv::Point2d n(-t.y, t.x);
    double side_shift = (PATCH_SIZE / 2.0) + 1.0;
    cv::Point2d c_plus = kf_edge.location + n * side_shift;
    cv::Point2d c_minus = kf_edge.location - n * side_shift;

    std::vector<cv::Point2d> cLplus, cLminus;
    util_make_rotated_patch_coords(c_plus, kf_edge.orientation, cLplus);
    util_make_rotated_patch_coords(c_minus, kf_edge.orientation, cLminus);

    std::vector<double> pLplus_f, pLminus_f;
    util_sample_patch_at_coords(kf_image_undistorted, cLplus, pLplus_f);
    util_sample_patch_at_coords(kf_image_undistorted, cLminus, pLminus_f);
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

    Edge cf_edge_iterated = cf_edge;
    cv::Point2d t_cf(std::cos(cf_edge_iterated.orientation), std::sin(cf_edge_iterated.orientation));
    cv::Point2d n_cf(-t_cf.y, t_cf.x);

    Eigen::Vector2d d = init_disp;
    double init_RMS = 0.0;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        //> Compute the right patch coordinates
        cf_edge_iterated.location = cv::Point2d(kf_edge.location.x - d[0], kf_edge.location.y - d[1]);
        cv::Point2d cRplus = cf_edge_iterated.location + n_cf * side_shift;
        cv::Point2d cRminus = cf_edge_iterated.location - n_cf * side_shift;

        std::vector<cv::Point2d> cRplusC, cRminusC;
        util_make_rotated_patch_coords(cRplus, cf_edge_iterated.orientation, cRplusC);
        util_make_rotated_patch_coords(cRminus, cf_edge_iterated.orientation, cRminusC);

        //> Sample right intensities and right gradient X at these coords
        std::vector<double> pRplus_f, pRminus_f, gxRplus_f, gxRminus_f, gyRplus_f, gyRminus_f;
        util_sample_patch_at_coords(cf_image_undistorted, cRplusC, pRplus_f);
        util_sample_patch_at_coords(cf_image_undistorted, cRminusC, pRminus_f);
        util_sample_patch_at_coords(cf_image_gradients_x, cRplusC, gxRplus_f);
        util_sample_patch_at_coords(cf_image_gradients_x, cRminusC, gxRminus_f);
        util_sample_patch_at_coords(cf_image_gradients_y, cRplusC, gyRplus_f);
        util_sample_patch_at_coords(cf_image_gradients_y, cRminusC, gyRminus_f);

        //> Compute means of the right patches
        double mRplus = util_vector_mean<double>(pRplus_f);
        double mRminus = util_vector_mean<double>(pRminus_f);

        //> Build residuals r = (L - meanL) - (R - meanR)  which centers both patches
        //> Build gradient which is the derivative of the residual with respect to the disparity g = dr / dd
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        double cost = 0.0;
        auto accumulate_patch = [&](const std::vector<double> &Lc, const std::vector<double> &Rf,
                                    const std::vector<double> &gxRf, const std::vector<double> &gyRf, double meanR)
        {
            for (size_t k = 0; k < Lc.size(); ++k)
            {
                double r = Lc[k] - (Rf[k] - meanR);
                Eigen::Vector2d J = Eigen::Vector2d(gxRf[k], gyRf[k]);
                double absr = std::abs(r);
                double w = (absr < huber_delta) ? 1.0 : huber_delta / absr;

                H += w * J * J.transpose();
                H += 1e-6 * Eigen::Matrix2d::Identity();
                b += w * J * r;
                cost += w * r * r;
            }
        };
        accumulate_patch(Lplus, pRplus_f, gxRplus_f, gyRplus_f, mRplus);
        accumulate_patch(Lminus, pRminus_f, gxRminus_f, gyRminus_f, mRminus);

        //> Update delta
        Eigen::Vector2d delta = -H.ldlt().solve(b);
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
        if (delta.norm() < tol || iter == max_iter - 1)
        {
            refined_validity = (is_outlier) ? false : true;
            refined_final_score = rms;
            break;
        }
    }

    refined_disparity = d;
}

void Temporal_Matches::get_Gammas_and_Tangents_From_Quads(\
    const KF_Temporal_Edge_Quads &kvq, const size_t candidate_idx, \
    Eigen::Matrix3d inv_K, Eigen::Vector3d &Gamma, Eigen::Vector3d &Gamma_bar, Eigen::Vector3d &Tangent, Eigen::Vector3d &Tangent_bar)
{
    Utility::Ptr utility = Utility::Ptr(new Utility());
    size_t j = candidate_idx;

    Eigen::Vector3d p_left(kvq.KF_stereo_mate->left_edge.location.x, kvq.KF_stereo_mate->left_edge.location.y, 1.0);
    Eigen::Vector3d p_right(kvq.KF_stereo_mate->right_edge.location.x, kvq.KF_stereo_mate->right_edge.location.y, 1.0);
    Eigen::Vector3d gamma1_left = inv_K * p_left;
    Eigen::Vector3d gamma1_right = inv_K * p_right;
    Gamma = utility->backproject_2D_point_to_3D_point_using_rays( \
        dataset->get_relative_rot_left_to_right(), dataset->get_relative_transl_left_to_right(), \
        gamma1_left, gamma1_right );
    Eigen::Vector3d p_bar_left(kvq.candidate_quads[candidate_idx].CF_left->center_edge.location.x, kvq.candidate_quads[candidate_idx].CF_left->center_edge.location.y, 1.0);
    Eigen::Vector3d p_bar_right(kvq.candidate_quads[candidate_idx].CF_right->center_edge.location.x, kvq.candidate_quads[candidate_idx].CF_right->center_edge.location.y, 1.0);
    Eigen::Vector3d gamma1_bar_left = inv_K * p_bar_left;
    Eigen::Vector3d gamma1_bar_right = inv_K * p_bar_right;
    Gamma_bar = utility->backproject_2D_point_to_3D_point_using_rays( \
        dataset->get_relative_rot_left_to_right(), dataset->get_relative_transl_left_to_right(), \
        gamma1_bar_left, gamma1_bar_right );

    //> Tangent from KF and Tangent_bar from CF candidate quad
    Eigen::Vector3d t1(cos(kvq.KF_stereo_mate->left_edge.orientation), sin(kvq.KF_stereo_mate->left_edge.orientation), 0);
    Eigen::Vector3d t2(cos(kvq.KF_stereo_mate->right_edge.orientation), sin(kvq.KF_stereo_mate->right_edge.orientation), 0);
    Eigen::Vector3d tangent1 = inv_K * t1;
    Eigen::Vector3d tangent2 = inv_K * t2;
    Tangent = utility->reconstruct_3D_Tangent( \
        dataset->get_relative_rot_left_to_right(), \
        gamma1_left, gamma1_right, \
        tangent1, tangent2);
    Eigen::Vector3d t1_bar(cos(kvq.candidate_quads[j].CF_left->center_edge.orientation), sin(kvq.candidate_quads[j].CF_left->center_edge.orientation), 0);
    Eigen::Vector3d t2_bar(cos(kvq.candidate_quads[j].CF_right->center_edge.orientation), sin(kvq.candidate_quads[j].CF_right->center_edge.orientation), 0);
    Eigen::Vector3d tangent1_bar = inv_K * t1_bar;
    Eigen::Vector3d tangent2_bar = inv_K * t2_bar;
    Tangent_bar = utility->reconstruct_3D_Tangent( \
        dataset->get_relative_rot_left_to_right(), \
        gamma1_bar_left, gamma1_bar_right, \
        tangent1_bar, tangent2_bar );
}

void Temporal_Matches::test_Constraints_from_Two_Oriented_Points( \
    const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, \
    const size_t keyframe_idx, const size_t current_frame_idx)
{
    // const Eigen::Matrix3d rel_R, const Eigen::Vector3d rel_T, const Eigen::Vector3d ray1, const Eigen::Vector3d ray2 )
    Utility::Ptr utility = Utility::Ptr(new Utility());
    Eigen::Matrix3d inv_K = dataset->get_left_calib_matrix().inverse();
    //> First split the quads into veridical and non-veridical
    std::vector<Quad_Prepared_for_Constraints_Check> veridical_quads;
    std::vector<Quad_Prepared_for_Constraints_Check> non_veridical_quads;
    for (size_t k = 0; k < quads_by_kf.size(); ++k)
    {
        const auto &kvq = quads_by_kf[k];
        if (!kvq.KF_stereo_mate->b_is_TP)
            continue;

        for (size_t j = 0; j < kvq.candidate_quads.size(); ++j)
        {
            Quad_Prepared_for_Constraints_Check q;
            get_Gammas_and_Tangents_From_Quads(kvq, j, inv_K, q.Gamma, q.Gamma_bar, q.Tangent, q.Tangent_bar);

            if ((k == 1176 && j == 0) || (k == 5046 && j == 0)) {
                std::cout << "(k, j): " << k << " " << j << std::endl;
                std::cout << "Gamma: " << q.Gamma.transpose() << std::endl;
                std::cout << "Gamma_bar: " << q.Gamma_bar.transpose() << std::endl;
                std::cout << "Tangent: " << q.Tangent.transpose() << std::endl;
                std::cout << "Tangent_bar: " << q.Tangent_bar.transpose() << std::endl;
            }

            if (kvq.b_is_TP[j]) {
                veridical_quads.push_back({k, j, q.Gamma, q.Gamma_bar, q.Tangent, q.Tangent_bar});
            }
            else {
                non_veridical_quads.push_back({k, j, q.Gamma, q.Gamma_bar, q.Tangent, q.Tangent_bar});
            }   
        }
    }

    //> write the c1 values to a file
    std::string output_dir = dataset->get_output_path();
    std::string file_name_veridical = output_dir + "/veridical_constraints_kf" + std::to_string(keyframe_idx) + "_cf" + std::to_string(current_frame_idx) + ".txt";
    std::string file_name_non_veridical = output_dir + "/non_veridical_constraints_kf" + std::to_string(keyframe_idx) + "_cf" + std::to_string(current_frame_idx) + ".txt";
    std::ofstream file_input_veridical(file_name_veridical);
    std::ofstream file_input_non_veridical(file_name_non_veridical);
    if (!file_input_veridical.is_open() || !file_input_non_veridical.is_open())
        return;

    const size_t N = 5000;
    for (size_t i = 0; i < N; i++) {
        if (veridical_quads.size() < 2 || non_veridical_quads.size() < 2)
            continue;
        //> randomly pick two distinct veridical quads (resample until q1 != q2)
        size_t idx1_v, idx2_v, idx1_nv, idx2_nv;
        do {
            idx1_v = rand() % veridical_quads.size();
            idx2_v = rand() % veridical_quads.size();
        } while (idx1_v == idx2_v);
        do {
            idx1_nv = rand() % non_veridical_quads.size();
            idx2_nv = rand() % non_veridical_quads.size();
        } while (idx1_nv == idx2_nv);
        const auto &q1_v = veridical_quads[idx1_v];
        const auto &q2_v = veridical_quads[idx2_v];
        const auto &q1_nv = non_veridical_quads[idx1_nv];
        const auto &q2_nv = non_veridical_quads[idx2_nv];

        //> Constraint 1: 
        double length_Gamma_v = (q1_v.Gamma - q2_v.Gamma).norm();
        double length_Gamma_bar_v = (q1_v.Gamma_bar - q2_v.Gamma_bar).norm();
        double veridical_c1_values = std::fabs(length_Gamma_v - length_Gamma_bar_v);
        double veridical_c1_values_normalized = veridical_c1_values / length_Gamma_v;

        double length_Gamma_nv = (q1_nv.Gamma - q2_nv.Gamma).norm();
        double length_Gamma_bar_nv = (q1_nv.Gamma_bar - q2_nv.Gamma_bar).norm();
        double non_veridical_c1_values = std::fabs(length_Gamma_nv - length_Gamma_bar_nv);
        double non_veridical_c1_values_normalized = non_veridical_c1_values / length_Gamma_nv;

        //> Constraint 2:
        double cos_angle_v1 = (q2_v.Gamma - q1_v.Gamma).dot(q1_v.Tangent_bar) / ((q2_v.Gamma - q1_v.Gamma).norm());
        double cos_angle_bar_v1 = (q2_v.Gamma_bar - q1_v.Gamma_bar).dot(q1_v.Tangent) / ((q2_v.Gamma_bar - q1_v.Gamma_bar).norm());
        double veridical_c2_values = std::fabs(std::fabs(cos_angle_v1) - std::fabs(cos_angle_bar_v1));

        double cos_angle_nv1 = (q2_nv.Gamma - q1_nv.Gamma).dot(q1_nv.Tangent_bar) / ((q2_nv.Gamma - q1_nv.Gamma).norm());
        double cos_angle_bar_nv1 = (q2_nv.Gamma_bar - q1_nv.Gamma_bar).dot(q1_nv.Tangent) / ((q2_nv.Gamma_bar - q1_nv.Gamma_bar).norm());
        double non_veridical_c2_values = std::fabs(std::fabs(cos_angle_nv1) - std::fabs(cos_angle_bar_nv1));

        //> Constraint 3:
        double cos_angle_v2 = (q2_v.Gamma - q1_v.Gamma).dot(q2_v.Tangent_bar) / ((q2_v.Gamma - q1_v.Gamma).norm());
        double cos_angle_bar_v2 = (q2_v.Gamma_bar - q1_v.Gamma_bar).dot(q2_v.Tangent) / ((q2_v.Gamma_bar - q1_v.Gamma_bar).norm());
        double veridical_c3_values = std::fabs(std::fabs(cos_angle_v2) - std::fabs(cos_angle_bar_v2));

        double cos_angle_nv2 = (q2_nv.Gamma - q1_nv.Gamma).dot(q2_nv.Tangent_bar) / ((q2_nv.Gamma - q1_nv.Gamma).norm());
        double cos_angle_bar_nv2 = (q2_nv.Gamma_bar - q1_nv.Gamma_bar).dot(q2_nv.Tangent) / ((q2_nv.Gamma_bar - q1_nv.Gamma_bar).norm());
        double non_veridical_c3_values = std::fabs(std::fabs(cos_angle_nv2) - std::fabs(cos_angle_bar_nv2));

        //> Constraint 4:
        double cos_tangent_angle_v = (q1_v.Tangent).dot(q2_v.Tangent_bar);
        double cos_tangent_angle_bar_v = (q1_v.Tangent_bar).dot(q2_v.Tangent);
        double veridical_c4_values = std::fabs(std::fabs(cos_tangent_angle_v) - std::fabs(cos_tangent_angle_bar_v));

        if (veridical_c4_values > 0.98 || veridical_c4_values < 0.02) {
            
            std::cout << "================================================" << std::endl;
            std::cout << "veridical_c4_values = " << veridical_c4_values << std::endl;
            std::cout << "================================================" << std::endl;

            const auto &kvq1 = quads_by_kf[q1_v.KF_stereo_mate_index];
            const auto &kvq2 = quads_by_kf[q2_v.KF_stereo_mate_index];

            std::cout << std::endl;
            std::cout << q1_v.KF_stereo_mate_index << " " << q1_v.candidate_quad_index << std::endl;
            std::cout << "KF gamma-left: " << kvq1.KF_stereo_mate->left_edge.location.x << ", " << kvq1.KF_stereo_mate->left_edge.location.y << " " << kvq1.KF_stereo_mate->left_edge.orientation << std::endl;
            std::cout << "KF gamma-right: " << kvq1.KF_stereo_mate->right_edge.location.x << ", " << kvq1.KF_stereo_mate->right_edge.location.y << " " << kvq1.KF_stereo_mate->right_edge.orientation << std::endl;
            std::cout << "CF gamma-left: " << kvq1.candidate_quads[q1_v.candidate_quad_index].CF_left->center_edge.location.x << ", " << kvq1.candidate_quads[q1_v.candidate_quad_index].CF_left->center_edge.location.y << " " << kvq1.candidate_quads[q1_v.candidate_quad_index].CF_left->center_edge.orientation << std::endl;
            std::cout << "CF gamma-right: " << kvq1.candidate_quads[q1_v.candidate_quad_index].CF_right->center_edge.location.x << ", " << kvq1.candidate_quads[q1_v.candidate_quad_index].CF_right->center_edge.location.y << " " << kvq1.candidate_quads[q1_v.candidate_quad_index].CF_right->center_edge.orientation << std::endl;
            std::cout << "Gamma = " << q1_v.Gamma.transpose() << ", Gamma_bar = " << q1_v.Gamma_bar.transpose() << std::endl;
            std::cout << "Tangent = " << q1_v.Tangent.transpose() << ", Tangent_bar = " << q1_v.Tangent_bar.transpose() << std::endl;
            
            std::cout << q2_v.KF_stereo_mate_index << " " << q2_v.candidate_quad_index << std::endl;
            std::cout << "KF gamma-left: " << kvq2.KF_stereo_mate->left_edge.location.x << ", " << kvq2.KF_stereo_mate->left_edge.location.y << " " << kvq2.KF_stereo_mate->left_edge.orientation << std::endl;
            std::cout << "KF gamma-right: " << kvq2.KF_stereo_mate->right_edge.location.x << ", " << kvq2.KF_stereo_mate->right_edge.location.y << " " << kvq2.KF_stereo_mate->right_edge.orientation << std::endl;
            std::cout << "CF gamma-left: " << kvq2.candidate_quads[q2_v.candidate_quad_index].CF_left->center_edge.location.x << ", " << kvq2.candidate_quads[q2_v.candidate_quad_index].CF_left->center_edge.location.y << " " << kvq2.candidate_quads[q2_v.candidate_quad_index].CF_left->center_edge.orientation << std::endl;
            std::cout << "CF gamma-right: " << kvq2.candidate_quads[q2_v.candidate_quad_index].CF_right->center_edge.location.x << ", " << kvq2.candidate_quads[q2_v.candidate_quad_index].CF_right->center_edge.location.y << " " << kvq2.candidate_quads[q2_v.candidate_quad_index].CF_right->center_edge.orientation << std::endl;
            std::cout << "Gamma = " << q2_v.Gamma.transpose() << ", Gamma_bar = " << q2_v.Gamma_bar.transpose() << std::endl;
            std::cout << "Tangent = " << q2_v.Tangent.transpose() << ", Tangent_bar = " << q2_v.Tangent_bar.transpose() << std::endl;
            
            std::cout << "===> cos_tangent_angle_v = " << cos_tangent_angle_v << ", " << "cos_tangent_angle_bar_v = " << cos_tangent_angle_bar_v << std::endl << std::endl;
            Eigen::Vector3d KF_tangent_left_1(cos(kvq1.KF_stereo_mate->left_edge.orientation), sin(kvq1.KF_stereo_mate->left_edge.orientation), 0);
            Eigen::Vector3d KF_tangent_right_1(cos(kvq1.KF_stereo_mate->right_edge.orientation), sin(kvq1.KF_stereo_mate->right_edge.orientation), 0);
            Eigen::Vector3d KF_tangent_left_2(cos(kvq2.KF_stereo_mate->left_edge.orientation), sin(kvq2.KF_stereo_mate->left_edge.orientation), 0);
            Eigen::Vector3d KF_tangent_right_2(cos(kvq2.KF_stereo_mate->right_edge.orientation), sin(kvq2.KF_stereo_mate->right_edge.orientation), 0);
            Eigen::Vector3d KF_gamma_left_1(kvq1.KF_stereo_mate->left_edge.location.x, kvq1.KF_stereo_mate->left_edge.location.y, 1.0);
            Eigen::Vector3d KF_gamma_right_1(kvq1.KF_stereo_mate->right_edge.location.x, kvq1.KF_stereo_mate->right_edge.location.y, 1.0);
            Eigen::Vector3d KF_gamma_left_2(kvq2.KF_stereo_mate->left_edge.location.x, kvq2.KF_stereo_mate->left_edge.location.y, 1.0);
            Eigen::Vector3d KF_gamma_right_2(kvq2.KF_stereo_mate->right_edge.location.x, kvq2.KF_stereo_mate->right_edge.location.y, 1.0);

            Eigen::Vector3d CF_tangent_left_1(cos(kvq1.candidate_quads[q1_v.candidate_quad_index].CF_left->center_edge.orientation), sin(kvq1.candidate_quads[q1_v.candidate_quad_index].CF_left->center_edge.orientation), 0);
            Eigen::Vector3d CF_tangent_right_1(cos(kvq1.candidate_quads[q1_v.candidate_quad_index].CF_right->center_edge.orientation), sin(kvq1.candidate_quads[q1_v.candidate_quad_index].CF_right->center_edge.orientation), 0);
            Eigen::Vector3d CF_tangent_left_2(cos(kvq2.candidate_quads[q2_v.candidate_quad_index].CF_left->center_edge.orientation), sin(kvq2.candidate_quads[q2_v.candidate_quad_index].CF_left->center_edge.orientation), 0);
            Eigen::Vector3d CF_tangent_right_2(cos(kvq2.candidate_quads[q2_v.candidate_quad_index].CF_right->center_edge.orientation), sin(kvq2.candidate_quads[q2_v.candidate_quad_index].CF_right->center_edge.orientation), 0);
            Eigen::Vector3d CF_gamma_left_1(kvq1.candidate_quads[q1_v.candidate_quad_index].CF_left->center_edge.location.x, kvq1.candidate_quads[q1_v.candidate_quad_index].CF_left->center_edge.location.y, 1.0);
            Eigen::Vector3d CF_gamma_right_1(kvq1.candidate_quads[q1_v.candidate_quad_index].CF_right->center_edge.location.x, kvq1.candidate_quads[q1_v.candidate_quad_index].CF_right->center_edge.location.y, 1.0);
            Eigen::Vector3d CF_gamma_left_2(kvq2.candidate_quads[q2_v.candidate_quad_index].CF_left->center_edge.location.x, kvq2.candidate_quads[q2_v.candidate_quad_index].CF_left->center_edge.location.y, 1.0);
            Eigen::Vector3d CF_gamma_right_2(kvq2.candidate_quads[q2_v.candidate_quad_index].CF_right->center_edge.location.x, kvq2.candidate_quads[q2_v.candidate_quad_index].CF_right->center_edge.location.y, 1.0);
            Eigen::Vector3d KF_n1_left = (KF_tangent_left_1.cross(inv_K * KF_gamma_left_1)).normalized();
            Eigen::Vector3d KF_n1_right = (KF_tangent_right_1.cross(inv_K * KF_gamma_right_1)).normalized();
            Eigen::Vector3d CF_n1_left = (CF_tangent_left_1.cross(inv_K * CF_gamma_left_1)).normalized();
            Eigen::Vector3d CF_n1_right = (CF_tangent_right_1.cross(inv_K * CF_gamma_right_1)).normalized();
            Eigen::Vector3d KF_n2_left = (KF_tangent_left_2.cross(inv_K * KF_gamma_left_2)).normalized();
            Eigen::Vector3d KF_n2_right = (KF_tangent_right_2.cross(inv_K * KF_gamma_right_2)).normalized();
            Eigen::Vector3d CF_n2_left = (CF_tangent_left_2.cross(inv_K * CF_gamma_left_2)).normalized();
            Eigen::Vector3d CF_n2_right = (CF_tangent_right_2.cross(inv_K * CF_gamma_right_2)).normalized();

            double KF_normal_vec_dot_product_1 = KF_n1_left.dot(KF_n1_right);
            double CF_normal_vec_dot_product_1 = CF_n1_left.dot(CF_n1_right);
            double KF_normal_vec_dot_product_2 = KF_n2_left.dot(KF_n2_right);
            double CF_normal_vec_dot_product_2 = CF_n2_left.dot(CF_n2_right);
            std::cout << "KF_normal_vec_dot_product_1 = " << KF_normal_vec_dot_product_1 << std::endl;
            std::cout << "CF_normal_vec_dot_product_1 = " << CF_normal_vec_dot_product_1 << std::endl;
            std::cout << "KF_normal_vec_dot_product_2 = " << KF_normal_vec_dot_product_2 << std::endl;
            std::cout << "CF_normal_vec_dot_product_2 = " << CF_normal_vec_dot_product_2 << std::endl << std::endl;
        }

        double cos_tangent_angle_nv = (q1_nv.Tangent).dot(q2_nv.Tangent_bar);
        double cos_tangent_angle_bar_nv = (q1_nv.Tangent_bar).dot(q2_nv.Tangent);
        double non_veridical_c4_values = std::fabs(std::fabs(cos_tangent_angle_nv) - std::fabs(cos_tangent_angle_bar_nv));

        //> Constraint 5:
        double cos_parallel_angle_v = (q1_v.Tangent).dot((q2_v.Gamma - q1_v.Gamma).normalized());
        double veridical_c5_values = std::fabs(cos_parallel_angle_v) - 1.0;

        double cos_parallel_angle_nv = (q1_nv.Tangent).dot((q2_nv.Gamma - q1_nv.Gamma).normalized());
        double non_veridical_c5_values = std::fabs(cos_parallel_angle_nv) - 1.0;

        //> Constraint 6:
        Eigen::Matrix3d c6_v_LHS, c6_v_RHS, c6_nv_LHS, c6_nv_RHS;
        c6_v_LHS.col(0) = q2_v.Gamma - q1_v.Gamma;
        c6_v_LHS.col(1) = q1_v.Tangent;
        c6_v_LHS.col(2) = q2_v.Tangent;
        c6_v_RHS.col(0) = q2_v.Gamma_bar - q1_v.Gamma_bar;
        c6_v_RHS.col(1) = q1_v.Tangent_bar;
        c6_v_RHS.col(2) = q2_v.Tangent_bar;
        double det_v_LHS = c6_v_LHS.determinant();
        double det_v_RHS = c6_v_RHS.determinant();
        double veridical_c6_values = std::fabs(det_v_LHS - det_v_RHS);

        c6_nv_LHS.col(0) = q2_nv.Gamma - q1_nv.Gamma;
        c6_nv_LHS.col(1) = q1_nv.Tangent;
        c6_nv_LHS.col(2) = q2_nv.Tangent;
        c6_nv_RHS.col(0) = q2_nv.Gamma_bar - q1_nv.Gamma_bar;
        c6_nv_RHS.col(1) = q1_nv.Tangent_bar;
        c6_nv_RHS.col(2) = q2_nv.Tangent_bar;
        double det_nv_LHS = c6_nv_LHS.determinant();
        double det_nv_RHS = c6_nv_RHS.determinant();
        double non_veridical_c6_values = std::fabs(det_nv_LHS - det_nv_RHS);

        //> Write to the files
        file_input_veridical << q1_v.KF_stereo_mate_index << " " << q1_v.candidate_quad_index << " " \
                            << q2_v.KF_stereo_mate_index << " " << q2_v.candidate_quad_index << " " \
                            << veridical_c1_values << " " << veridical_c1_values_normalized << " " \
                            << veridical_c2_values << " " << veridical_c3_values << " " << veridical_c4_values << " " \
                            << veridical_c5_values << " " << veridical_c6_values << std::endl;

        file_input_non_veridical << q1_nv.KF_stereo_mate_index << " " << q1_nv.candidate_quad_index << " " \
                            << q2_nv.KF_stereo_mate_index << " " << q2_nv.candidate_quad_index << " " \
                            << non_veridical_c1_values << " " << non_veridical_c1_values_normalized << " " \
                            << non_veridical_c2_values << " " << non_veridical_c3_values << " " << non_veridical_c4_values << " " \
                            << non_veridical_c5_values << " " << non_veridical_c6_values << std::endl;
    }
    file_input_veridical.close();
    file_input_non_veridical.close();
}

void Temporal_Matches::write_quads_to_file(const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf,
    size_t keyframe_idx, size_t current_frame_idx,
    const std::string &filename_suffix)
{
    std::cout << "Writing quads to file..." << std::endl;
    std::string output_dir = dataset->get_output_path();
    std::cout << "output_dir: " << output_dir << std::endl;

    if (output_dir.empty())
        return;
    std::string file_name = output_dir + "/quads_kf" + std::to_string(keyframe_idx) + "_cf" + std::to_string(current_frame_idx);
    if (!filename_suffix.empty())
        file_name += "_" + filename_suffix;
    file_name += ".txt";
    std::ofstream file_input(file_name);
    if (!file_input.is_open())
        return;

    for (size_t k = 0; k < quads_by_kf.size(); ++k)
    {
        const auto &kvq = quads_by_kf[k];
        if (!kvq.KF_stereo_mate->b_is_TP)
            continue;

        //> write KF stereo of the quad
        file_input << k << " " << -1 << " "
        << kvq.KF_stereo_mate->left_edge.location.x << " " << kvq.KF_stereo_mate->left_edge.location.y << " " << kvq.KF_stereo_mate->left_edge.orientation << " "
        << kvq.KF_stereo_mate->right_edge.location.x << " " << kvq.KF_stereo_mate->right_edge.location.y << " " << kvq.KF_stereo_mate->right_edge.orientation << " "
        << "-1" << "\n";

        //> write GT CF stereo of the quad
        file_input << k << " " << -2 << " "
        << kvq.projected_point_left.x() << " " << kvq.projected_point_left.y() << " " << kvq.projected_orientation_left << " "
        << kvq.projected_point_right.x() << " " << kvq.projected_point_right.y() << " " << kvq.projected_orientation_right << " "
        << "-2" << "\n";

        //> write CF stereo of the candidate quads of the quad
        for (size_t j = 0; j < kvq.candidate_quads.size(); ++j)
        {
            const auto &cq = kvq.candidate_quads[j];
            file_input << k << " " << j << " "
            << cq.CF_left->center_edge.location.x << " " << cq.CF_left->center_edge.location.y << " " << cq.CF_left->center_edge.orientation << " "
            << cq.CF_right->center_edge.location.x << " " << cq.CF_right->center_edge.location.y << " " << cq.CF_right->center_edge.orientation << " "
            << (kvq.b_is_TP[j] ? 1 : 0) << "\n";
        }
    }
}
