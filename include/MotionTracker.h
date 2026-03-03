#ifndef MOTION_TRACKER_H
#define MOTION_TRACKER_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "utility.h"
#include "Dataset.h"
#include "Temporal_Matches.h"

struct Quad_for_Pose_Solution
{
    size_t KF_stereo_mate_index;
    size_t candidate_quad_index;
    Eigen::Vector3d Gamma;
    Eigen::Vector3d Gamma_bar;
    Eigen::Vector3d Tangent;
    Eigen::Vector3d Tangent_bar;

    bool b_is_veridical;
};

struct Quad_Pair_Evaluation_Metrics
{
    std::string stage_name;
    double recall;
    double precision;
    size_t num_of_surviving_veridical_quad_pairs;
};

//> Edited based on PoseLib: https://github.com/PoseLib/PoseLib/blob/master/PoseLib/types.h
struct Ransac_Options {
    size_t max_iterations = 5000;
    size_t min_iterations = 1000;
    double dyn_num_trials_mult = 3.0;
    double success_prob = 0.97;
    double max_reprojection_location_error = 1.5; //> in pixels
    double max_reprojection_orientation_error = 5.0; 
    unsigned long seed = 0; //> 0 for deterministic behavior (uses fixed seed 42); non-zero values use std::random_device for different sequence each run
    size_t max_prosac_iterations = 100000;
    double top_rank_ordered_percentage = 0.7;
};

//> Edited based on PoseLib: https://github.com/PoseLib/PoseLib/blob/master/PoseLib/ransac_impl.h
//  and https://github.com/PoseLib/PoseLib/blob/master/PoseLib/types.h
struct Ransac_State {
    size_t refinements = 0;
    size_t iterations = 0;
    double inlier_ratio = 0;
    Camera_Pose best_pose_hypothesis;
    std::vector<size_t> inlier_indices;

    size_t best_minimal_inlier_count = 0;
    double best_minimal_msac_score = std::numeric_limits<double>::max();
    size_t dynamic_max_iter = 100000;
    double log_prob_missing_model = std::log(1.0 - 0.9999);
};

class MotionTracker {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MotionTracker> Ptr;

    //> Constructor (nothing special)
    MotionTracker(Dataset::Ptr dataset);   

    std::vector<Quad_for_Pose_Solution> get_Quad_for_Pose_Solution(const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf);
    Camera_Pose estimate_Pose_From_a_Quad_Pair(const Quad_for_Pose_Solution &q1, const Quad_for_Pose_Solution &q2);
    bool estimate_Relative_Pose_From_Quad_Pairs(const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, const Ransac_Options &opt, Ransac_State &state);
    void get_Gammas_and_Tangents_From_Quads(const KF_Temporal_Edge_Quads &kvq, const size_t candidate_idx, \
        Eigen::Matrix3d inv_K, Eigen::Vector3d &Gamma, Eigen::Vector3d &Gamma_bar, Eigen::Vector3d &Tangent, Eigen::Vector3d &Tangent_bar);
    
    std::vector<Quad_Pair_Evaluation_Metrics> Solution_Constraints_Application(const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, const Ransac_Options &opt, Ransac_State &state);
    void Print_Quad_Pairs_Metrics_Statistics(const std::vector<std::vector<Quad_Pair_Evaluation_Metrics>> &all_quad_pair_evaluation_metrics);

private:

    //> Constraints applied to pairs of quads for finding the pose solution
    bool Apply_Normalized_Length_Constraint(const Quad_for_Pose_Solution &q1, const Quad_for_Pose_Solution &q2); 
    bool Apply_T1_Angle_Similarity_Constraint(const Quad_for_Pose_Solution &q1, const Quad_for_Pose_Solution &q2); 
    bool Apply_T2_Angle_Similarity_Constraint(const Quad_for_Pose_Solution &q1, const Quad_for_Pose_Solution &q2);
    bool Apply_Tangent_Angle_Similarity_Constraint(const Quad_for_Pose_Solution &q1, const Quad_for_Pose_Solution &q2);

    void score_Pose_Hypothesis(const Camera_Pose &pose_hypothesis, const std::vector<Quad_for_Pose_Solution> &quads, \
        const std::vector<KF_Temporal_Edge_Quads> &quads_by_kf, const Ransac_Options &opt, std::vector<size_t> &inlier_indices);

    unsigned long get_seed_value_for_rng(const Ransac_Options &opt) {
        if (opt.seed == 0) 
        {
            return 42;  // Fixed seed for deterministic behavior
        } else {
            std::random_device rd;
            return rd();  // Random seed for non-deterministic behavior
        }
    }

    //> Pointers to the classes
    Utility::Ptr utility_tool = nullptr;
    Dataset::Ptr dataset;

    Eigen::Matrix3d Estimated_Rel_Rot;
    Eigen::Vector3d Estimated_Rel_Transl;
};


#endif
