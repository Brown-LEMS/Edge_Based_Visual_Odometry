#ifndef MOTION_TRACKER_CPP
#define MOTION_TRACKER_CPP

#include <limits>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include "MotionTracker.h"
#include "definitions.h"

MotionTracker::MotionTracker(Dataset::Ptr dataset) : dataset(std::move(dataset))
{
    utility_tool = std::make_shared<Utility>();
}

void MotionTracker::get_Gammas_and_Tangents_From_Quads(\
    const KF_Temporal_Edge_Quads &kvq, const size_t candidate_idx, \
    Eigen::Matrix3d inv_K, Eigen::Vector3d &Gamma, Eigen::Vector3d &Gamma_bar, Eigen::Vector3d &Tangent, Eigen::Vector3d &Tangent_bar)
{
    size_t j = candidate_idx;

    Eigen::Vector3d p_left(kvq.KF_stereo_mate->left_edge.location.x, kvq.KF_stereo_mate->left_edge.location.y, 1.0);
    Eigen::Vector3d p_right(kvq.KF_stereo_mate->right_edge.location.x, kvq.KF_stereo_mate->right_edge.location.y, 1.0);
    Eigen::Vector3d gamma1_left = inv_K * p_left;
    Eigen::Vector3d gamma1_right = inv_K * p_right;
    Gamma = utility_tool->backproject_2D_point_to_3D_point_using_rays( \
        dataset->get_relative_rot_left_to_right(), dataset->get_relative_transl_left_to_right(), \
        gamma1_left, gamma1_right );
    Eigen::Vector3d p_bar_left(kvq.candidate_quads[candidate_idx].CF_left->center_edge.location.x, kvq.candidate_quads[candidate_idx].CF_left->center_edge.location.y, 1.0);
    Eigen::Vector3d p_bar_right(kvq.candidate_quads[candidate_idx].CF_right->center_edge.location.x, kvq.candidate_quads[candidate_idx].CF_right->center_edge.location.y, 1.0);
    Eigen::Vector3d gamma1_bar_left = inv_K * p_bar_left;
    Eigen::Vector3d gamma1_bar_right = inv_K * p_bar_right;
    Gamma_bar = utility_tool->backproject_2D_point_to_3D_point_using_rays( \
        dataset->get_relative_rot_left_to_right(), dataset->get_relative_transl_left_to_right(), \
        gamma1_bar_left, gamma1_bar_right );

    //> Tangent from KF and Tangent_bar from CF candidate quad
    Eigen::Vector3d t1(cos(kvq.KF_stereo_mate->left_edge.orientation), sin(kvq.KF_stereo_mate->left_edge.orientation), 0);
    Eigen::Vector3d t2(cos(kvq.KF_stereo_mate->right_edge.orientation), sin(kvq.KF_stereo_mate->right_edge.orientation), 0);
    Eigen::Vector3d tangent1 = inv_K * t1;
    Eigen::Vector3d tangent2 = inv_K * t2;
    Tangent = utility_tool->reconstruct_3D_Tangent_through_intersection_of_planes( \
        dataset->get_relative_rot_left_to_right(), \
        gamma1_left, gamma1_right, \
        tangent1, tangent2);
    Eigen::Vector3d t1_bar(cos(kvq.candidate_quads[j].CF_left->center_edge.orientation), sin(kvq.candidate_quads[j].CF_left->center_edge.orientation), 0);
    Eigen::Vector3d t2_bar(cos(kvq.candidate_quads[j].CF_right->center_edge.orientation), sin(kvq.candidate_quads[j].CF_right->center_edge.orientation), 0);
    Eigen::Vector3d tangent1_bar = inv_K * t1_bar;
    Eigen::Vector3d tangent2_bar = inv_K * t2_bar;
    Tangent_bar = utility_tool->reconstruct_3D_Tangent_through_intersection_of_planes( \
        dataset->get_relative_rot_left_to_right(), \
        gamma1_bar_left, gamma1_bar_right, \
        tangent1_bar, tangent2_bar );
}

std::vector<Quad_for_Pose_Solution> MotionTracker::get_Quad_for_Pose_Solution(const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf) 
{
    Eigen::Matrix3d inv_K = dataset->get_left_calib_matrix().inverse();
    std::vector<Quad_for_Pose_Solution> quads_for_pose_solution;

    for (size_t k = 0; k < quads_by_kf.size(); ++k)
    {
        const auto &kvq = quads_by_kf[k];
        //> If we have GT, only keep quads whose KF stereo edge mate is a TP.
        //> Without GT disparity, b_is_TP is not meaningful, so we keep all.
        if (dataset->has_gt() && !kvq.KF_stereo_mate->b_is_TP)
            continue;

        for (size_t j = 0; j < kvq.candidate_quads.size(); ++j)
        {
            Quad_for_Pose_Solution q;
            get_Gammas_and_Tangents_From_Quads(kvq, j, inv_K, q.Gamma, q.Gamma_bar, q.Tangent, q.Tangent_bar);
            bool is_veridical = dataset->has_gt() && j < kvq.b_is_TP.size() ? kvq.b_is_TP[j] : false;
            quads_for_pose_solution.push_back({k, j, q.Gamma, q.Gamma_bar, q.Tangent, q.Tangent_bar, is_veridical});
        }
    }

    //> Rank-order quads by the number of candidate quads of their keyframe (ascending),
    //> so that quads from more ambiguous KFs appear later (useful for PROSAC-style sampling).
    std::sort(quads_for_pose_solution.begin(), quads_for_pose_solution.end(),
              [&quads_by_kf](const Quad_for_Pose_Solution &a, const Quad_for_Pose_Solution &b)
              {
                  const auto size_a = quads_by_kf[a.KF_stereo_mate_index].candidate_quads.size();
                  const auto size_b = quads_by_kf[b.KF_stereo_mate_index].candidate_quads.size();
                  if (size_a != size_b)
                      return size_a < size_b;  // smaller first
                  // Tie-breaker: lower candidate index first for determinism
                  if (a.KF_stereo_mate_index != b.KF_stereo_mate_index)
                      return a.KF_stereo_mate_index < b.KF_stereo_mate_index;
                  return a.candidate_quad_index < b.candidate_quad_index;
              });

    return quads_for_pose_solution;
}

bool MotionTracker::Apply_Normalized_Length_Constraint(const Quad_for_Pose_Solution &q1, const Quad_for_Pose_Solution &q2) 
{
    double length_Gamma = (q1.Gamma - q2.Gamma).norm();
    double length_Gamma_bar = (q1.Gamma_bar - q2.Gamma_bar).norm();
    return (std::fabs(length_Gamma - length_Gamma_bar) / length_Gamma) < TAU_C1 ? (true) : (false);
}

bool MotionTracker::Apply_T1_Angle_Similarity_Constraint(const Quad_for_Pose_Solution &q1, const Quad_for_Pose_Solution &q2) 
{
    double cos_angle = (q2.Gamma - q1.Gamma).dot(q1.Tangent) / ((q2.Gamma - q1.Gamma).norm());
    double cos_angle_bar = (q2.Gamma_bar - q1.Gamma_bar).dot(q1.Tangent_bar) / ((q2.Gamma_bar - q1.Gamma_bar).norm());
    return (std::fabs(std::fabs(cos_angle) - std::fabs(cos_angle_bar)) < TAU_C2) ? (true) : (false);
}

bool MotionTracker::Apply_T2_Angle_Similarity_Constraint(const Quad_for_Pose_Solution &q1, const Quad_for_Pose_Solution &q2) 
{
    double cos_angle = (q2.Gamma - q1.Gamma).dot(q2.Tangent) / ((q2.Gamma - q1.Gamma).norm());
    double cos_angle_bar = (q2.Gamma_bar - q1.Gamma_bar).dot(q2.Tangent_bar) / ((q2.Gamma_bar - q1.Gamma_bar).norm());
    return (std::fabs(std::fabs(cos_angle) - std::fabs(cos_angle_bar)) < TAU_C3) ? (true) : (false);
}

bool MotionTracker::Apply_Tangent_Angle_Similarity_Constraint(const Quad_for_Pose_Solution &q1, const Quad_for_Pose_Solution &q2) 
{
    double cos_tangent_angle = (q1.Tangent).dot(q2.Tangent);
    double cos_tangent_angle_bar = (q1.Tangent_bar).dot(q2.Tangent_bar);
    return (std::fabs(std::fabs(cos_tangent_angle) - std::fabs(cos_tangent_angle_bar)) < TAU_C4) ? (true) : (false);
}

Camera_Pose MotionTracker::estimate_Pose_From_a_Quad_Pair(const Quad_for_Pose_Solution &q1, const Quad_for_Pose_Solution &q2)
{
    Eigen::Vector3d e1 = (q2.Gamma - q1.Gamma).normalized();
    Eigen::Vector3d e1_bar = (q2.Gamma_bar - q1.Gamma_bar).normalized();
    Eigen::Vector3d u1 = q1.Tangent - (e1.dot(q1.Tangent)) * e1;
    Eigen::Vector3d u1_bar = q1.Tangent_bar - (e1_bar.dot(q1.Tangent_bar)) * e1_bar;
    Eigen::Vector3d e2 = u1.normalized();
    Eigen::Vector3d e2_bar = u1_bar.normalized();
    Eigen::Vector3d e3 = e1.cross(e2);
    Eigen::Vector3d e3_bar = e1_bar.cross(e2_bar);
    Eigen::Matrix3d B;
    B << e1, e2, e3;  // columns 0,1,2
    Eigen::Matrix3d B_bar;
    B_bar << e1_bar, e2_bar, e3_bar;
    Eigen::Matrix3d R = B_bar * B.transpose();
    Eigen::Vector3d t = q1.Gamma_bar - R * q1.Gamma;
    return Camera_Pose(R, t);
}

void MotionTracker::score_Pose_Hypothesis(const Camera_Pose &pose_hypothesis, const std::vector<Quad_for_Pose_Solution> &quads, \
    const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, const Ransac_Options &opt, std::vector<size_t> &inlier_indices)
{
    inlier_indices.clear();
    for (size_t i = 0; i < quads.size(); ++i) {
        Eigen::Vector3d hypothesis_Gamma_bar = pose_hypothesis.transform(quads[i].Gamma);
        Eigen::Vector3d proj_point = dataset->get_left_calib_matrix() * hypothesis_Gamma_bar;
        proj_point.x() /= proj_point.z();
        proj_point.y() /= proj_point.z();
        //> find reprojection error between the projected point and the 2D point on the left CF edge location of the candidate quad
        const auto &kvq = quads_by_kf[quads[i].KF_stereo_mate_index];
        const auto *cf_left = kvq.candidate_quads[quads[i].candidate_quad_index].CF_left;
        cv::Point2d proj_point_cv(proj_point.x(), proj_point.y());
        double reproj_error = cv::norm(proj_point_cv - cf_left->center_edge.location);
        if (reproj_error < opt.max_reprojection_location_error) {
            inlier_indices.push_back(i);
        }
    }
}

void MotionTracker::save_GT_reprojection_table_for_matlab(const Camera_Pose &rel_pose_gt,
    const std::vector<Quad_for_Pose_Solution> &quads, const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf,
    const Ransac_Options &opt, const std::string &file_path, size_t current_frame_index)
{
    std::ofstream out(file_path);
    if (!out) {
        LOG_ERROR(std::string("Could not open GT reprojection table for writing: ") + file_path);
        return;
    }
    out << std::setprecision(17);
    out << "# GT relative pose: reproject KF triangulation Gamma into current left camera, compare to candidate CF_left center.\n";
    out << "# current_frame_index " << current_frame_index << "\n";
    out << "# max_reprojection_location_error_px " << opt.max_reprojection_location_error << "\n";
    out << "# quad_idx,kf_stereo_mate_index,candidate_quad_index,proj_x,proj_y,obs_x,obs_y,reproj_error_px,proj_z,inlier\n";

    for (size_t i = 0; i < quads.size(); ++i) {
        Eigen::Vector3d X_cf = rel_pose_gt.transform(quads[i].Gamma);
        Eigen::Vector3d ph = dataset->get_left_calib_matrix() * X_cf;
        const double z = ph.z();
        double proj_x = 0.0;
        double proj_y = 0.0;
        if (std::abs(z) >= 1e-12) {
            proj_x = ph.x() / z;
            proj_y = ph.y() / z;
        }
        const auto &kvq = quads_by_kf[quads[i].KF_stereo_mate_index];
        const auto *cf_left = kvq.candidate_quads[quads[i].candidate_quad_index].CF_left;
        const double obs_x = cf_left->center_edge.location.x;
        const double obs_y = cf_left->center_edge.location.y;
        const double reproj = std::hypot(proj_x - obs_x, proj_y - obs_y);
        const int inlier = reproj < opt.max_reprojection_location_error ? 1 : 0;
        out << i << " " << quads[i].KF_stereo_mate_index << " " << quads[i].candidate_quad_index << " "
            << proj_x << " " << proj_y << " " << obs_x << " " << obs_y << " " 
            << reproj << " " << z << " " << inlier << "\n";
    }
}

bool MotionTracker::estimate_Relative_Pose_From_Quad_Pairs(const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, const Ransac_Options &opt, Ransac_State &state)
{
    if (quads_by_kf.size() < 2) {
        LOG_ERROR("Insufficient quad pairs, available: " + std::to_string(quads_by_kf.size()));
        LOG_ERROR("Returning identity camera pose");
        state.best_pose_hypothesis = Camera_Pose();
        return false;
    }

    //> prepare quads for pose estimation
    std::vector<Quad_for_Pose_Solution> quads_for_pose_solution = get_Quad_for_Pose_Solution(quads_by_kf);
    
    size_t top_n_rank_ordered_list = static_cast<size_t>(std::ceil(opt.top_rank_ordered_percentage * static_cast<double>(quads_for_pose_solution.size())));
    top_n_rank_ordered_list = std::max<size_t>(2, top_n_rank_ordered_list);
    top_n_rank_ordered_list = std::min(top_n_rank_ordered_list, quads_for_pose_solution.size());

    //> prepare RANSAC settings
    state.dynamic_max_iter = opt.max_iterations;
    state.log_prob_missing_model = std::log(1.0 - opt.success_prob);
    state.best_minimal_inlier_count = 0;

    //> RANSAC main loop
    for (state.iterations = 0; state.iterations < opt.max_iterations; state.iterations++) {

        if (state.iterations > opt.min_iterations && state.iterations > state.dynamic_max_iter) {
            break;
        }

        //> Randomly pick 2 indices
        size_t idx1, idx2;
        do {
            idx1 = rand() % top_n_rank_ordered_list;
            idx2 = rand() % top_n_rank_ordered_list;
        } while (idx1 == idx2);
        const auto &q1 = quads_for_pose_solution[idx1];
        const auto &q2 = quads_for_pose_solution[idx2];

        // //> Apply constraints progressively
        // if (!Apply_Normalized_Length_Constraint(q1, q2)) {
        //     state.iterations = (state.iterations > 0) ? (state.iterations - 1) : 0;
        //     continue;
        // }
        // if (!Apply_T1_Angle_Similarity_Constraint(q1, q2)) {
        //     state.iterations = (state.iterations > 0) ? (state.iterations - 1) : 0;
        //     continue;
        // }
        // if (!Apply_T2_Angle_Similarity_Constraint(q1, q2)) {
        //     state.iterations = (state.iterations > 0) ? (state.iterations - 1) : 0;
        //     continue;
        // }
        // if (!Apply_Tangent_Angle_Similarity_Constraint(q1, q2)) {
        //     state.iterations = (state.iterations > 0) ? (state.iterations - 1) : 0;
        //     continue;
        // }

        //> Pose hypothesis formation
        Camera_Pose pose_hypothesis = estimate_Pose_From_a_Quad_Pair(q1, q2);

        //> Pose hypothesis validation
        std::vector<size_t> inlier_indices;
        score_Pose_Hypothesis(pose_hypothesis, quads_for_pose_solution, quads_by_kf, opt, inlier_indices);
        if (inlier_indices.size() > state.best_minimal_inlier_count) {
            state.best_minimal_inlier_count = inlier_indices.size();
            state.best_pose_hypothesis = pose_hypothesis;
            state.inlier_ratio = static_cast<double>(inlier_indices.size()) / static_cast<double>(quads_for_pose_solution.size());
        }

        //> Update number of iterations
        if (state.inlier_ratio >= 0.85) {
            // this is to avoid log(prob_outlier) = -inf below
            state.dynamic_max_iter = opt.min_iterations;
        } else if (state.inlier_ratio <= 0.05) {
            // this is to avoid log(prob_outlier) = 0 below
            state.dynamic_max_iter = opt.max_iterations;
        } else {
            const double prob_outlier = 1.0 - std::pow(state.inlier_ratio, 2);
            state.dynamic_max_iter = std::ceil(state.log_prob_missing_model / std::log(prob_outlier) * opt.dyn_num_trials_mult);
        }
    }

    return true;
}

bool MotionTracker::Solution_Constrained_Pose_Estimation( \
    const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, \
    std::vector<Quad_for_Pose_Solution> quads_for_pose_solution, \
    std::vector<Quad_for_Pose_Solution> all_quads_for_pose_solution, \
    std::vector<std::pair<size_t, size_t>> picked_indices, \
    const Ransac_Options &opt, Ransac_State &state)
{
    if (quads_by_kf.size() < 2) {
        LOG_ERROR("Insufficient quad pairs, available: " + std::to_string(quads_by_kf.size()));
        LOG_ERROR("Returning identity camera pose");
        state.best_pose_hypothesis = Camera_Pose();
        return false;
    }

    size_t top_n_rank_ordered_list = quads_for_pose_solution.size();

    //> prepare RANSAC settings
    state.dynamic_max_iter = opt.max_iterations;
    state.log_prob_missing_model = std::log(1.0 - opt.success_prob);
    state.best_minimal_inlier_count = 0;

    //> RANSAC main loop
    for (state.iterations = 0; state.iterations < picked_indices.size(); state.iterations++) {

        if (state.iterations > opt.min_iterations && state.iterations > state.dynamic_max_iter) {
            break;
        }

        //> Randomly pick one from from the picked_indices pair
        //> he selected index pair is the one that passes the constraints
        size_t idx = state.iterations;
        const auto &q1 = all_quads_for_pose_solution[picked_indices[idx].first];
        const auto &q2 = all_quads_for_pose_solution[picked_indices[idx].second];

        //> Pose hypothesis formation
        Camera_Pose pose_hypothesis = estimate_Pose_From_a_Quad_Pair(q1, q2);

        //> Pose hypothesis validation
        std::vector<size_t> inlier_indices;
        score_Pose_Hypothesis(pose_hypothesis, all_quads_for_pose_solution, quads_by_kf, opt, inlier_indices);
        if (inlier_indices.size() > state.best_minimal_inlier_count) {
            state.best_minimal_inlier_count = inlier_indices.size();
            state.best_pose_hypothesis = pose_hypothesis;
            state.inlier_ratio = static_cast<double>(inlier_indices.size()) / static_cast<double>(all_quads_for_pose_solution.size());
        }

        // //> Approach 1: Early stopping
        // if (state.inlier_ratio >= 0.55) {
        //     break;
        // }

        //> Approach 2: instead of a hard stop, reach to at least reach a minimum number of iterations
        //> Update number of iterations
        if (state.inlier_ratio >= 0.55) {
            // this is to avoid log(prob_outlier) = -inf below
            state.dynamic_max_iter = opt.min_iterations;
        } else if (state.inlier_ratio <= 0.05) {
            // this is to avoid log(prob_outlier) = 0 below
            state.dynamic_max_iter = opt.max_iterations;
        } else {
            const double prob_outlier = 1.0 - std::pow(state.inlier_ratio, 2);
            state.dynamic_max_iter = std::ceil(state.log_prob_missing_model / std::log(prob_outlier) * opt.dyn_num_trials_mult);
        }
    }

    return true;
}

std::vector<Quad_Pair_Evaluation_Metrics> MotionTracker::Solution_Constraints_Application(const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, const Camera_Pose &ground_truth_pose, const Ransac_Options &opt, Ransac_State &state) 
{
    //> get quads for pose estimation
    std::vector<Quad_for_Pose_Solution> quads_for_pose_solution = get_Quad_for_Pose_Solution(quads_by_kf);

    std::vector<std::pair<size_t, size_t>> indices;
    // std::vector<Quad_for_Pose_Solution> last_quads_for_pose_solution = 

    double precision = 0.0;
    double recall = 0.0;
    size_t num_of_veridical_quads = 0;

    std::vector<std::pair<size_t, size_t>> last_surviving_pair_of_indices;
    std::vector<std::pair<size_t, size_t>> surviving_pair_of_indices;
    const size_t total_num_of_quad_pairs = opt.max_iterations;

    size_t top_n_rank_ordered_list = static_cast<size_t>(opt.top_rank_ordered_percentage * quads_for_pose_solution.size());

    //> RANSAC main loop
    for (state.iterations = 0; state.iterations < opt.max_iterations; state.iterations++) {
        
        //> Randomly pick 2 indices
        size_t idx1, idx2;
        do {
            idx1 = rand() % top_n_rank_ordered_list;
            idx2 = rand() % top_n_rank_ordered_list;
        } while (idx1 == idx2);
        const auto &q1 = quads_for_pose_solution[idx1];
        const auto &q2 = quads_for_pose_solution[idx2];
        std::pair<size_t, size_t> pair_of_indices = {idx1, idx2};
        indices.push_back(pair_of_indices);

        if (q1.b_is_veridical && q2.b_is_veridical) {
            num_of_veridical_quads++;
        } 
    }
    precision = static_cast<double>(num_of_veridical_quads) / static_cast<double>(total_num_of_quad_pairs);
    recall = 1.0;
    size_t initial_num_of_veridical_quads = num_of_veridical_quads;

    //> compute the pose
    Ransac_Options opt_baseline;
    Ransac_State state_baseline;
    Solution_Constrained_Pose_Estimation(quads_by_kf, quads_for_pose_solution, quads_for_pose_solution, indices, opt_baseline, state_baseline);
    double rot_err = utility_tool->evaluate_Relative_Rotation_Accuracy(state_baseline.best_pose_hypothesis, ground_truth_pose);
    double trans_err = utility_tool->evaluate_Relative_Translation_Accuracy(state_baseline.best_pose_hypothesis, ground_truth_pose);

    std::vector<Quad_Pair_Evaluation_Metrics> quad_pair_evaluation_metrics;
    quad_pair_evaluation_metrics.push_back({"Baseline", recall, precision, num_of_veridical_quads, rot_err, trans_err, state_baseline.iterations, state_baseline.inlier_ratio});

    //> Constraint 1: Normalized Length Constraint
    std::vector<Quad_for_Pose_Solution> quads_for_pose_solution_after_normalized_length_constraint;
    num_of_veridical_quads = 0;
    size_t num_of_surviving_quad_pairs = 0;
    surviving_pair_of_indices.clear();
    for (size_t i = 0; i < indices.size(); ++i) {
        const auto &q1 = quads_for_pose_solution[indices[i].first];
        const auto &q2 = quads_for_pose_solution[indices[i].second];
        if (Apply_Normalized_Length_Constraint(q1, q2)) {
            surviving_pair_of_indices.push_back(indices[i]);
            if (q1.b_is_veridical && q2.b_is_veridical) {
                num_of_veridical_quads++;
            }
            num_of_surviving_quad_pairs++;
            quads_for_pose_solution_after_normalized_length_constraint.push_back(q1);
            quads_for_pose_solution_after_normalized_length_constraint.push_back(q2);
        }
    }
    recall = static_cast<double>(num_of_veridical_quads) / static_cast<double>(initial_num_of_veridical_quads);
    precision = (num_of_surviving_quad_pairs == 0) ? 0.0 : static_cast<double>(num_of_veridical_quads) / static_cast<double>(num_of_surviving_quad_pairs);

    //> compute the pose
    Ransac_Options opt_normalized_length_constraint;
    Ransac_State state_normalized_length_constraint;
    Solution_Constrained_Pose_Estimation(quads_by_kf, quads_for_pose_solution_after_normalized_length_constraint, quads_for_pose_solution, surviving_pair_of_indices, opt_normalized_length_constraint, state_normalized_length_constraint);
    rot_err = utility_tool->evaluate_Relative_Rotation_Accuracy(state_normalized_length_constraint.best_pose_hypothesis, ground_truth_pose);
    trans_err = utility_tool->evaluate_Relative_Translation_Accuracy(state_normalized_length_constraint.best_pose_hypothesis, ground_truth_pose);

    quad_pair_evaluation_metrics.push_back({"Normalized Length Constraint", recall, precision, num_of_veridical_quads, rot_err, trans_err, state_normalized_length_constraint.iterations, state_normalized_length_constraint.inlier_ratio});
    last_surviving_pair_of_indices = std::move(surviving_pair_of_indices);

    //> Constraint 2: T1 Angle Similarity Constraint
    std::vector<Quad_for_Pose_Solution> quads_for_pose_solution_after_t1_angle_similarity_constraint;
    num_of_veridical_quads = 0;
    num_of_surviving_quad_pairs = 0;
    surviving_pair_of_indices.clear();
    for (size_t i = 0; i < last_surviving_pair_of_indices.size(); ++i) {
        const auto &q1 = quads_for_pose_solution[last_surviving_pair_of_indices[i].first];
        const auto &q2 = quads_for_pose_solution[last_surviving_pair_of_indices[i].second];
        if (Apply_T1_Angle_Similarity_Constraint(q1, q2)) {
            surviving_pair_of_indices.push_back(last_surviving_pair_of_indices[i]);
            if (q1.b_is_veridical && q2.b_is_veridical) {
                num_of_veridical_quads++;
            }
            num_of_surviving_quad_pairs++;
            quads_for_pose_solution_after_t1_angle_similarity_constraint.push_back(q1);
            quads_for_pose_solution_after_t1_angle_similarity_constraint.push_back(q2);
        }
    }
    recall = static_cast<double>(num_of_veridical_quads) / static_cast<double>(initial_num_of_veridical_quads);
    precision = (num_of_surviving_quad_pairs == 0) ? 0.0 : static_cast<double>(num_of_veridical_quads) / static_cast<double>(num_of_surviving_quad_pairs);

    //> compute the pose
    Ransac_Options opt_t1_angle_similarity_constraint;
    Ransac_State state_t1_angle_similarity_constraint;
    Solution_Constrained_Pose_Estimation(quads_by_kf, quads_for_pose_solution_after_t1_angle_similarity_constraint, quads_for_pose_solution, surviving_pair_of_indices, opt_t1_angle_similarity_constraint, state_t1_angle_similarity_constraint);
    rot_err = utility_tool->evaluate_Relative_Rotation_Accuracy(state_t1_angle_similarity_constraint.best_pose_hypothesis, ground_truth_pose);
    trans_err = utility_tool->evaluate_Relative_Translation_Accuracy(state_t1_angle_similarity_constraint.best_pose_hypothesis, ground_truth_pose);

    quad_pair_evaluation_metrics.push_back({"T1 Angle Similarity Constraint", recall, precision, num_of_veridical_quads, rot_err, trans_err, state_t1_angle_similarity_constraint.iterations, state_t1_angle_similarity_constraint.inlier_ratio});
    last_surviving_pair_of_indices = std::move(surviving_pair_of_indices);

    //> Constraint 3: T2 Angle Similarity Constraint
    std::vector<Quad_for_Pose_Solution> quads_for_pose_solution_after_t2_angle_similarity_constraint;
    num_of_veridical_quads = 0;
    num_of_surviving_quad_pairs = 0;
    surviving_pair_of_indices.clear();
    for (size_t i = 0; i < last_surviving_pair_of_indices.size(); ++i) {
        const auto &q1 = quads_for_pose_solution[last_surviving_pair_of_indices[i].first];
        const auto &q2 = quads_for_pose_solution[last_surviving_pair_of_indices[i].second];
        if (Apply_T2_Angle_Similarity_Constraint(q1, q2)) {
            surviving_pair_of_indices.push_back(last_surviving_pair_of_indices[i]);
            if (q1.b_is_veridical && q2.b_is_veridical) {
                num_of_veridical_quads++;
            }
            num_of_surviving_quad_pairs++;
            quads_for_pose_solution_after_t2_angle_similarity_constraint.push_back(q1);
            quads_for_pose_solution_after_t2_angle_similarity_constraint.push_back(q2);
        }
    }
    recall = static_cast<double>(num_of_veridical_quads) / static_cast<double>(initial_num_of_veridical_quads);
    precision = (num_of_surviving_quad_pairs == 0) ? 0.0 : static_cast<double>(num_of_veridical_quads) / static_cast<double>(num_of_surviving_quad_pairs);

    //> compute the pose
    Ransac_Options opt_t2_angle_similarity_constraint;
    Ransac_State state_t2_angle_similarity_constraint;
    Solution_Constrained_Pose_Estimation(quads_by_kf, quads_for_pose_solution_after_t2_angle_similarity_constraint, quads_for_pose_solution, surviving_pair_of_indices, opt_t2_angle_similarity_constraint, state_t2_angle_similarity_constraint);
    rot_err = utility_tool->evaluate_Relative_Rotation_Accuracy(state_t2_angle_similarity_constraint.best_pose_hypothesis, ground_truth_pose);
    trans_err = utility_tool->evaluate_Relative_Translation_Accuracy(state_t2_angle_similarity_constraint.best_pose_hypothesis, ground_truth_pose);
 
    quad_pair_evaluation_metrics.push_back({"T2 Angle Similarity Constraint", recall, precision, num_of_veridical_quads, rot_err, trans_err, state_t2_angle_similarity_constraint.iterations, state_t2_angle_similarity_constraint.inlier_ratio});
    last_surviving_pair_of_indices = std::move(surviving_pair_of_indices);

    //> Constraint 4: Tangent Angle Similarity Constraint
    std::vector<Quad_for_Pose_Solution> quads_for_pose_solution_after_tangent_angle_similarity_constraint;
    num_of_veridical_quads = 0;
    num_of_surviving_quad_pairs = 0;
    surviving_pair_of_indices.clear();
    for (size_t i = 0; i < last_surviving_pair_of_indices.size(); ++i) {
        const auto &q1 = quads_for_pose_solution[last_surviving_pair_of_indices[i].first];
        const auto &q2 = quads_for_pose_solution[last_surviving_pair_of_indices[i].second];
        if (Apply_Tangent_Angle_Similarity_Constraint(q1, q2)) {
            surviving_pair_of_indices.push_back(last_surviving_pair_of_indices[i]);
            if (q1.b_is_veridical && q2.b_is_veridical) {
                num_of_veridical_quads++;
            }
            num_of_surviving_quad_pairs++;
            quads_for_pose_solution_after_tangent_angle_similarity_constraint.push_back(q1);
            quads_for_pose_solution_after_tangent_angle_similarity_constraint.push_back(q2);
        }
    }
    recall = static_cast<double>(num_of_veridical_quads) / static_cast<double>(initial_num_of_veridical_quads);
    precision = (num_of_surviving_quad_pairs == 0) ? 0.0 : static_cast<double>(num_of_veridical_quads) / static_cast<double>(num_of_surviving_quad_pairs);

    //> compute the pose
    Ransac_Options opt_tangent_angle_similarity_constraint;
    Ransac_State state_tangent_angle_similarity_constraint;
    Solution_Constrained_Pose_Estimation(quads_by_kf, quads_for_pose_solution_after_tangent_angle_similarity_constraint, quads_for_pose_solution, surviving_pair_of_indices, opt_tangent_angle_similarity_constraint, state_tangent_angle_similarity_constraint);
    rot_err = utility_tool->evaluate_Relative_Rotation_Accuracy(state_tangent_angle_similarity_constraint.best_pose_hypothesis, ground_truth_pose);
    trans_err = utility_tool->evaluate_Relative_Translation_Accuracy(state_tangent_angle_similarity_constraint.best_pose_hypothesis, ground_truth_pose);
 
    quad_pair_evaluation_metrics.push_back({"Tangent Angle Similarity Constraint", recall, precision, num_of_veridical_quads, rot_err, trans_err, state_tangent_angle_similarity_constraint.iterations, state_tangent_angle_similarity_constraint.inlier_ratio});

    return quad_pair_evaluation_metrics;
}

void MotionTracker::Print_Quad_Pairs_Metrics_Statistics(const std::vector<std::vector<Quad_Pair_Evaluation_Metrics>> &all_quad_pair_evaluation_metrics, const bool b_print_average_metrics, std::ofstream &file_quad_pair_constraints_metrics)
{
    if (all_quad_pair_evaluation_metrics.empty())
        return;    

    if (b_print_average_metrics) {
        const auto &first = all_quad_pair_evaluation_metrics.front();
        if (first.empty())
            return;

        std::cout << "\n===== Quad Pair Constraints Metrics (Solution Constraints Application) =====" << std::endl;
        std::cout << "               Stage               |         Recall         |        Precision       |       Number of Surviving Veridical Quad Pairs" << std::endl;

        // Assume each inner vector corresponds to one run and contains one entry per stage.
        // Aggregate per-stage metrics over all runs by stage_name.
        for (const auto &ref_metric : first)
        {
            const std::string &stage_name = ref_metric.stage_name;
            double sum_recall = 0.0;
            double sum_precision = 0.0;
            std::size_t sum_num_veridical = 0;
            std::size_t count = 0;

            for (const auto &per_run : all_quad_pair_evaluation_metrics)
            {
                auto it = std::find_if(per_run.begin(), per_run.end(),
                                    [&stage_name](const Quad_Pair_Evaluation_Metrics &m)
                                    {
                                        return m.stage_name == stage_name;
                                    });
                if (it != per_run.end())
                {
                    sum_recall += it->recall;
                    sum_precision += it->precision;
                    sum_num_veridical += it->num_of_surviving_veridical_quad_pairs;
                    ++count;
                }
            }

            if (count > 0)
            {
                double avg_recall = sum_recall / static_cast<double>(count);
                double avg_precision = sum_precision / static_cast<double>(count);
                double avg_num_of_surviving_veridical_quad_pairs = sum_num_veridical / static_cast<double>(count);

                std::cout << std::setw(35) << stage_name << " | "
                        << std::setw(20) << avg_recall << " | "
                        << std::setw(20) << avg_precision << " | "
                        << std::setw(20) << avg_num_of_surviving_veridical_quad_pairs << std::endl;
            }
        }
        std::cout << std::endl;
    }
    else {
        //> get the last metrics
        const auto &last = all_quad_pair_evaluation_metrics.back();
        if (last.empty())
            return;

        // std::cout << "\n===== Quad Pair Constraints Metrics (Solution Constraints Application) =====" << std::endl;
        // std::cout << "               Stage               |         Recall         |        Precision       |       Rotation Error       |       Translation Error       |       Number of Iterations" << std::endl;

        for (const auto &ref_metric : last) {
            // const std::string &stage_name = ref_metric.stage_name;
            // std::cout << std::setw(35) << stage_name << " | " << std::setw(20) << ref_metric.recall << " | " << std::setw(20) << ref_metric.precision << " | " \
            //           << std::setw(20) << ref_metric.rot_err << " | " << std::setw(20) << ref_metric.trans_err << " | " << std::setw(20) << ref_metric.num_of_iterations << std::endl;
            file_quad_pair_constraints_metrics << ref_metric.precision << " " << ref_metric.rot_err << " " << ref_metric.trans_err << " " << ref_metric.num_of_iterations << " " << ref_metric.inlier_ratio << std::endl;
        }
        // std::cout << std::endl;
    }
}

void MotionTracker::save_Quad_for_Pose_Solution_to_File(
    Dataset &dataset, const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf,
    const std::vector<Quad_for_Pose_Solution> &quads_for_pose_solution, const std::string &file_name)
{
    const std::string path = dataset.get_output_path() + "/" + file_name + ".txt";
    std::ofstream file(path);
    if (!file)
    {
        LOG_FILE_ERROR("Failed to open for write: " + path);
        return;
    }
    file << "# columns (space-separated): "
            "kf_lx kf_ly kf_ltheta kf_rx kf_ry kf_rtheta "
            "cf_lx cf_ly cf_ltheta cf_rx cf_ry cf_rtheta "
            "gamma_x gamma_y gamma_z gamma_bar_x gamma_bar_y gamma_bar_z "
            "tangent_x tangent_y tangent_z tangent_bar_x tangent_bar_y tangent_bar_z " << std::endl;
    file << std::scientific << std::setprecision(5);
    for (size_t i = 0; i < quads_for_pose_solution.size(); ++i)
    {
        const Quad_for_Pose_Solution &q = quads_for_pose_solution[i];

        //> Some sanity checks
        if (q.KF_stereo_mate_index >= quads_by_kf.size())
            continue;
        const KF_Temporal_Edge_Quads &kvq = quads_by_kf[q.KF_stereo_mate_index];
        if (!kvq.KF_stereo_mate || q.candidate_quad_index >= kvq.candidate_quads.size())
            continue;
        const Candidate_Quad_Entry &cq = kvq.candidate_quads[q.candidate_quad_index];
        if (!cq.CF_left || !cq.CF_right)
            continue;

        const Edge &kf_l = kvq.KF_stereo_mate->left_edge;
        const Edge &kf_r = kvq.KF_stereo_mate->right_edge;
        const Edge &cf_l = cq.CF_left->center_edge;
        const Edge &cf_r = cq.CF_right->center_edge;

        file << kf_l.location.x << " " << kf_l.location.y << " " << kf_l.orientation << " "
             << kf_r.location.x << " " << kf_r.location.y << " " << kf_r.orientation << " "
             << cf_l.location.x << " " << cf_l.location.y << " " << cf_l.orientation << " "
             << cf_r.location.x << " " << cf_r.location.y << " " << cf_r.orientation << " "
             << q.Gamma.x() << " " << q.Gamma.y() << " " << q.Gamma.z() << " "
             << q.Gamma_bar.x() << " " << q.Gamma_bar.y() << " " << q.Gamma_bar.z() << " "
             << q.Tangent.x() << " " << q.Tangent.y() << " " << q.Tangent.z() << " "
             << q.Tangent_bar.x() << " " << q.Tangent_bar.y() << " " << q.Tangent_bar.z() << std::endl;
    }
    file.close();
    LOG_INFO("Saved rank-ordered quads for pose solution to file: " + path);
}


#endif