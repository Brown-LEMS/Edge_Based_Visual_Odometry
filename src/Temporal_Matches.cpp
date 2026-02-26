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
    Eigen::Matrix3d R_left_ego = current_frame_stereo.stereo_frame->gt_rotation * last_keyframe_stereo.stereo_frame->gt_rotation.transpose();
    Eigen::Vector3d t_left_ego = current_frame_stereo.stereo_frame->gt_translation - R_left_ego * last_keyframe_stereo.stereo_frame->gt_translation;

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
            Eigen::Vector3d Gamma_in_left_CF = R_left_ego * Gamma_in_left_KF + t_left_ego;
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
    // Step 1: Get the stereo baseline rotation (Left -> Right)
    Eigen::Matrix3d R_stereo = dataset.get_relative_rot_left_to_right();

    // Step 2: Reconstruct 3D direction T_1 in Left KF
    Eigen::Vector3d t1(cos(e_left.orientation), sin(e_left.orientation), 0);
    Eigen::Vector3d t2(cos(e_right.orientation), sin(e_right.orientation), 0);

    Eigen::Vector3d gamma_1(e_left.location.x, e_left.location.y, 1.0);
    gamma_1 = dataset.get_left_calib_matrix().inverse() * gamma_1;
    Eigen::Vector3d gamma_2(e_right.location.x, e_right.location.y, 1.0);
    gamma_2 = dataset.get_right_calib_matrix().inverse() * gamma_2;

    Eigen::Vector3d T_1 = -(gamma_2.dot(t2.cross(R_stereo * t1))) * gamma_1 + (gamma_2.dot(t2.cross(R_stereo * gamma_1))) * t1;
    T_1 = -T_1;
    T_1.normalize();

    // Step 3: Transform T_1 to current frame
    Eigen::Matrix3d R_temporal = current_frame.gt_rotation * last_keyframe.gt_rotation.transpose(); // Left KF -> Left CF

    Eigen::Vector3d T_2;
    if (is_left_cam)
    {
        T_2 = R_temporal * T_1; // Left KF -> Left CF
    }
    else
    {
        T_2 = R_stereo * R_temporal * T_1; // Left KF -> Left CF -> Right CF
    }

    // Step 4: Project T_2 to image
    Eigen::Vector3d gamma_cf = projected_point / projected_point.z();
    if (is_left_cam)
        gamma_cf = dataset.get_left_calib_matrix().inverse() * gamma_cf;
    else
        gamma_cf = dataset.get_right_calib_matrix().inverse() * gamma_cf;

    Eigen::Vector3d t = T_2 - T_2.z() * gamma_cf;
    t.normalize();

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
                double sum_x = 0, sum_y = 0, sum_cos = 0, sum_sin = 0;
                for (const Edge &e : right_edges_sub)
                {
                    sum_x += e.location.x;
                    sum_y += e.location.y;
                    sum_cos += std::cos(e.orientation);
                    sum_sin += std::sin(e.orientation);
                }
                int n = static_cast<int>(right_edges_sub.size());
                right_center.location.x = sum_x / n;
                right_center.location.y = sum_y / n;
                right_center.orientation = std::atan2(sum_sin, sum_cos);
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
    // file_input << "kf_idx -1 left_x left_y left_orient right_x right_y right_orient"
    // << "cf_idx left_x left_y left_orient right_x right_y right_orient is_TP\n";

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
