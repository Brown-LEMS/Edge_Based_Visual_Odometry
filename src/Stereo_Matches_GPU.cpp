#ifndef STEREO_MATCHES_GPU_CPP
#define STEREO_MATCHES_GPU_CPP

#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "definitions.h"
#include "Stereo_Matches_GPU.h"
#include "gpu_kernels.h"
#include "Dataset.h"

Stereo_Matches_GPU_Pipeline::Stereo_Matches_GPU_Pipeline(Dataset::Ptr dataset, const Prepare_For_GPU_Pipeline::Ptr prepare_for_GPU_engine, 
                                                         const StereoFrame& current_frame, int device)
    : device_id(device)
    ,img_left_height(mat_rows_prefer_undistorted(current_frame.left_image_undistorted, current_frame.left_image))
    ,img_right_height(mat_rows_prefer_undistorted(current_frame.right_image_undistorted, current_frame.right_image))
    ,img_left_width(mat_cols_prefer_undistorted(current_frame.left_image_undistorted, current_frame.left_image))
    ,img_right_width(mat_cols_prefer_undistorted(current_frame.right_image_undistorted, current_frame.right_image))
    ,h_F(prepare_for_GPU_engine->h_F)
    ,left_image_texture_(prepare_for_GPU_engine->left_image_texture_)
    ,right_image_texture_(prepare_for_GPU_engine->right_image_texture_)
    ,h_right_img(prepare_for_GPU_engine->h_right_img)
    ,dataset_(std::move(dataset))
    ,stereo_frame_ptr_(&current_frame)
{
    num_left_edges_ = current_frame.left_edges.size();
    num_right_edges_ = current_frame.right_edges.size();

    //> Sanity check
    if (num_left_edges_ <= 0) {
        LOG_ERROR("Number of edges on the left image is less than or equal to 0");
    }
    if (num_right_edges_ <= 0) {
        LOG_ERROR("Number of edges on the right image is less than or equal to 0");
    }

    //> CAUTION: this is subject to the scene complexity
    max_stereo_edge_matches_after_epipolar_rasterization = num_left_edges_ * FACTOR_OF_NUM_OF_LEFT_EDGES_FOR_STEREO_EDGE_MATCHES_AFTER_EPIPOLAR_RASTERIZATION;

    //> convert edges to float (double -> float)
    h_left_edges = new Edge_GPU[num_left_edges_];
    h_right_edges = new Edge_GPU[num_right_edges_];
    h_cell_ids = new int[num_right_edges_];
    h_matches_out = new Match_by_Edge_Index[max_stereo_edge_matches_after_epipolar_rasterization];

    //> Casting the edges to the proper format for GPU processing
    #pragma omp parallel for
    for (size_t i = 0; i < num_left_edges_; i++) {
        h_left_edges[i].location_x = current_frame.left_edges[i].location.x;
        h_left_edges[i].location_y = current_frame.left_edges[i].location.y;
        h_left_edges[i].orientation = current_frame.left_edges[i].orientation;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < num_right_edges_; i++) {
        h_right_edges[i].location_x = current_frame.right_edges[i].location.x;
        h_right_edges[i].location_y = current_frame.right_edges[i].location.y;
        h_right_edges[i].orientation = current_frame.right_edges[i].orientation;
    }

    num_of_grid_cells_in_height = static_cast<int>(std::ceil(static_cast<double>(img_right_height) / GRID_CELL_SIZE));
    num_of_grid_cells_in_width = static_cast<int>(std::ceil(static_cast<double>(img_right_width) / GRID_CELL_SIZE));
    total_num_of_grid_cells = num_of_grid_cells_in_width * num_of_grid_cells_in_height;

    std::cout << "Right image size: " << img_right_width << " x " << img_right_height << std::endl;
    std::cout << "Number of grid cells in width: " << num_of_grid_cells_in_width << std::endl;

    //> Allocate GPU memory
    cudacheck( cudaMalloc((void**)&d_F,                            9 * sizeof(float))    );
    cudacheck( cudaMalloc((void**)&d_left_edges,    num_left_edges_  * sizeof(Edge_GPU)) );
    cudacheck( cudaMalloc((void**)&d_right_edges,   num_right_edges_ * sizeof(Edge_GPU)) );
    cudacheck( cudaMalloc((void**)&d_cell_ids,      num_right_edges_ * sizeof(int))      );

    cudacheck( cudaMalloc((void**)&d_edge_indices,          num_right_edges_ * sizeof(int))        );
    cudacheck( cudaMalloc((void**)&d_cell_start_idx,        total_num_of_grid_cells * sizeof(int)) );
    cudacheck( cudaMalloc((void**)&d_num_of_edges_in_cell,  total_num_of_grid_cells * sizeof(int)) );

    //> Output returned from the kernels processing EP, LP, and OR filters
    //> These can be allocated first since there is no dependency on the preceding kernels
    cudacheck( cudaMalloc((void**)&d_final_matches_passing_EP_LP_OR_filters,  max_stereo_edge_matches_after_epipolar_rasterization * sizeof(Match_by_Edge_Index)) );
    cudacheck( cudaMalloc((void**)&d_match_count_passing_EP_LP_OR_filters, sizeof(int)) );

    //> cuda event
    cudacheck( cudaEventCreate(&start) );
    cudacheck( cudaEventCreate(&stop) );
}

Stereo_Matches_GPU_Timing_Statistics Stereo_Matches_GPU_Pipeline::get_stereo_edge_matching_GPU(size_t frame_idx) {

    //> Timer holders
    Stereo_Matches_GPU_Timing_Statistics GPU_time_collection;
    GPU_time_collection.total_time = 0.0;

    //> For each edge on the right, construct the spatial grid mapping (result is stored in d_cell_ids)
    GPU_time_collection.time_spatial_grid_mapping = build_edge_spatial_map();
    GPU_time_collection.total_time += GPU_time_collection.time_spatial_grid_mapping;

    //> Do epipolar line rasterization to find right candidate edges that pass EP, LP, and OR filters
    GPU_time_collection.time_epipolar_rasterization = rasterize_epipolar_band_for_matches();
    GPU_time_collection.total_time += GPU_time_collection.time_epipolar_rasterization;
    #if WRITE_GPU_STEREO_STAGE_SNAPSHOTS
    write_gpu_stage_match_by_edge_index_file(device_id, dataset_, h_left_edges, num_left_edges_, h_right_edges, num_right_edges_,
                                           d_final_matches_passing_EP_LP_OR_filters, d_match_count_passing_EP_LP_OR_filters, "ep_lp_or", frame_idx);
    #endif

    //> Precompute the left edge patches photometry
    GPU_time_collection.time_precompute_left_edge_patches_photometry = precompute_left_edge_patches_photometry();
    GPU_time_collection.total_time += GPU_time_collection.time_precompute_left_edge_patches_photometry;

    //> Apply NCC filter
    GPU_time_collection.time_apply_NCC_filter = apply_NCC_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_NCC_filter;
    // std::cout << "Number of matches passing NCC filter: " << h_match_count_passing_NCC_filter << std::endl;
    #if WRITE_GPU_STEREO_STAGE_SNAPSHOTS
    write_gpu_stage_match_by_edge_index_file(device_id, dataset_, h_left_edges, num_left_edges_, h_right_edges, num_right_edges_,
                                           d_final_matches_passing_NCC_filter, d_final_count_passing_NCC_filter, "ncc", frame_idx);
    #endif

    //> Apply GPU-only SIFT-style descriptor filter
    GPU_time_collection.time_apply_SIFT_descriptor_filter = apply_SIFT_descriptor_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_SIFT_descriptor_filter;
    // std::cout << "Number of matches passing GPU SIFT descriptor filter: " << h_match_count_passing_SIFT_filter << std::endl;
    #if WRITE_GPU_STEREO_STAGE_SNAPSHOTS
    write_gpu_stage_match_by_edge_index_file(device_id, dataset_, h_left_edges, num_left_edges_, h_right_edges, num_right_edges_,
                                           d_final_matches_passing_SIFT_filter, d_final_count_passing_SIFT_filter, "sift", frame_idx);
    #endif

    //> Apply Best-Nearly-Best in GPU for NCC and SIFT scores.
    GPU_time_collection.time_apply_BNB_NCC_filter = apply_BNB_NCC_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_BNB_NCC_filter;
    // std::cout << "Number of matches passing GPU BNB-NCC filter: " << h_match_count_passing_BNB_NCC_filter << std::endl;
    #if WRITE_GPU_STEREO_STAGE_SNAPSHOTS
    write_gpu_stage_match_by_edge_index_file(device_id, dataset_, h_left_edges, num_left_edges_, h_right_edges, num_right_edges_,
                                           d_final_matches_passing_BNB_NCC_filter, d_final_count_passing_BNB_NCC_filter, "bnb_ncc", frame_idx);
    #endif

    GPU_time_collection.time_apply_BNB_SIFT_filter = apply_BNB_SIFT_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_BNB_SIFT_filter;
    // std::cout << "Number of matches passing GPU BNB-SIFT filter: " << h_match_count_passing_BNB_SIFT_filter << std::endl;
    #if WRITE_GPU_STEREO_STAGE_SNAPSHOTS
    write_gpu_stage_match_by_edge_index_file(device_id, dataset_, h_left_edges, num_left_edges_, h_right_edges, num_right_edges_,
                                           d_final_matches_passing_BNB_SIFT_filter, d_final_count_passing_BNB_SIFT_filter, "bnb_sift", frame_idx);
    #endif

    //> Shift edge pair hypothesis onto the epipolar line and refine the edge location by minimizing the photometric residual, with consolidation as the final step
    GPU_time_collection.time_apply_epipolar_shift_photometric_refine_and_merge = apply_epipolar_shift_photometric_refine_and_merge();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_epipolar_shift_photometric_refine_and_merge;
    // std::cout << "Number of matches after GPU shift + refine + merge: " << h_match_count_merged_after_refine << std::endl;

    //> Apply the second NCC filter
    GPU_time_collection.time_apply_second_NCC_filter = apply_second_NCC_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_second_NCC_filter;
    // std::cout << "Number of matches passing the second NCC filter: " << h_match_count_passing_NCC_filter << std::endl;

    //> One merged hypothesis per left edge: keep the candidate with the highest second-NCC score
    GPU_time_collection.time_apply_best_second_ncc_per_left = apply_best_second_ncc_per_left_after_second_NCC();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_best_second_ncc_per_left;
    // std::cout << "Number of matches after best second-NCC per left (one-to-one): " << h_match_count_passing_NCC_filter << std::endl;

#if SHOW_TIMING_BREAKDOWN
    //> set the same digits for the timeing breakdown when printing
    std::cout.precision(3);
    std::cout << "Timing breakdown of stereo edge matching: " << std::endl;
    std::cout << " - spatial grid mapping: " << GPU_time_collection.time_spatial_grid_mapping << " (ms)" << std::endl;
    std::cout << " - epipolar rasterization: " << GPU_time_collection.time_epipolar_rasterization << " (ms)" << std::endl;
    std::cout << " - precompute left edge patches photometry: " << GPU_time_collection.time_precompute_left_edge_patches_photometry << " (ms)" << std::endl;
    std::cout << " - NCC filter: " << GPU_time_collection.time_apply_NCC_filter << " (ms)" << std::endl;
    std::cout << " - SIFT descriptor filter: " << GPU_time_collection.time_apply_SIFT_descriptor_filter << " (ms)" << std::endl;
    std::cout << " - BNB NCC filter: " << GPU_time_collection.time_apply_BNB_NCC_filter << " (ms)" << std::endl;
    std::cout << " - BNB SIFT filter: " << GPU_time_collection.time_apply_BNB_SIFT_filter << " (ms)" << std::endl;
    std::cout << " - Epipolar shift + refine + merge: " << GPU_time_collection.time_apply_epipolar_shift_photometric_refine_and_merge << " (ms)" << std::endl;
    std::cout << " - Second NCC filter: " << GPU_time_collection.time_apply_second_NCC_filter << " (ms)" << std::endl;
    std::cout << " - Best second-NCC per left: " << GPU_time_collection.time_apply_best_second_ncc_per_left << " (ms)" << std::endl;
#endif
    LOG_INFO("Total time of GPU stereo edge matching: " + std::to_string(GPU_time_collection.total_time) + " (ms)");
    return GPU_time_collection;
}

float Stereo_Matches_GPU_Pipeline::build_edge_spatial_map() {

    //> Build edge spatial map
    return edge_spatial_map_thrust_pipeline(
        device_id,
        h_right_edges,
        h_cell_ids,
        d_right_edges,
        d_cell_ids,
        d_edge_indices,
        d_cell_start_idx,
        d_num_of_edges_in_cell,
        num_right_edges_,
        total_num_of_grid_cells,
        num_of_grid_cells_in_width,
        num_of_grid_cells_in_height,
        start,
        stop);
}

float Stereo_Matches_GPU_Pipeline::rasterize_epipolar_band_for_matches() {

    return epipolar_rasterization_thick_band_pipeline(
        device_id,
        h_F,
        h_left_edges,
        h_match_count_passing_EP_LP_OR_filters,
        h_matches_out,
        d_left_edges,
        d_right_edges,
        d_edge_indices,
        d_cell_start_idx,
        d_num_of_edges_in_cell,
        d_F,
        d_final_matches_passing_EP_LP_OR_filters,
        d_match_count_passing_EP_LP_OR_filters,
        num_left_edges_,
        img_right_width,
        img_right_height,
        num_of_grid_cells_in_width,
        num_of_grid_cells_in_height,
        max_stereo_edge_matches_after_epipolar_rasterization,
        start,
        stop);
}

float Stereo_Matches_GPU_Pipeline::precompute_left_edge_patches_photometry() {

    return precompute_left_edge_patches_photometry_pipeline(
        device_id,
        left_image_texture_.texObj,
        d_left_patches,
        d_left_edges,
        num_left_edges_,
        start,
        stop
    );
}

float Stereo_Matches_GPU_Pipeline::apply_NCC_filter() {

    //> Use device match count: host h_match_count_* must match d_match_count_* or the NCC kernel launches with
    //> num_candidates == 0 while candidates still live on device (validator then sees CPU passes / GPU empty).
    int num_candidates = 0;
    cudacheck(cudaMemcpy(&num_candidates, d_match_count_passing_EP_LP_OR_filters, sizeof(int), cudaMemcpyDeviceToHost));
    if (num_candidates > max_stereo_edge_matches_after_epipolar_rasterization) {
        LOG_WARNING("NCC: clamping candidate count to epipolar buffer (" + std::to_string(num_candidates) + " -> " +
                    std::to_string(max_stereo_edge_matches_after_epipolar_rasterization) + ")");
        num_candidates = max_stereo_edge_matches_after_epipolar_rasterization;
    }
    if (num_candidates < 0) {
        LOG_ERROR("NCC: invalid negative candidate count from device; treating as 0");
        num_candidates = 0;
    }
    h_match_count_passing_EP_LP_OR_filters = num_candidates;

    size_t ncc_out_bytes = static_cast<size_t>(num_candidates) * sizeof(Match_by_Edge_Index);
    if (ncc_out_bytes == 0)
        ncc_out_bytes = sizeof(Match_by_Edge_Index);
    cudacheck(cudaMalloc(&d_final_matches_passing_NCC_filter, ncc_out_bytes));
    cudacheck(cudaMalloc(&d_final_count_passing_NCC_filter, sizeof(int)));

    // std::cout << "Number of matches passing EP, LP, and OR filters: " << num_candidates << std::endl;
    return apply_NCC_filter_pipeline(
        device_id,
        h_match_count_passing_NCC_filter,
        right_image_texture_.texObj,
        d_final_matches_passing_EP_LP_OR_filters,
        d_left_edges,
        d_right_edges,
        d_final_matches_passing_NCC_filter,
        d_final_count_passing_NCC_filter,
        d_ncc_scores_passing_NCC_filter,
        d_left_patches,
        num_candidates,
        start,
        stop,
        static_cast<float>(NCC_THRESH)
    );
}

float Stereo_Matches_GPU_Pipeline::apply_SIFT_descriptor_filter() {

    //> Just a sanity check
    if (h_match_count_passing_NCC_filter < 0) {
        LOG_ERROR("GPU SIFT: invalid negative NCC count from device; treating as 0");
        h_match_count_passing_NCC_filter = 0;
        return -1.0;
    }

    //> First pre-compute the left edge SIFT descriptors
    float precompute_left_sift_time = 0.0f;
    if (d_left_sift_descriptors == nullptr) {
        precompute_left_sift_time = precompute_left_edge_sift_descriptors_pipeline(
            device_id,
            left_image_texture_.texObj,
            d_left_sift_descriptors,
            d_left_edges,
            img_left_width,
            img_left_height,
            num_left_edges_,
            start,
            stop
        );
    }

    //> The right edge SIFT descriptors are computed on-the-fly during the SIFT descriptor filter application
    const float apply_sift_filter_time = apply_SIFT_descriptor_filter_pipeline(
        device_id,
        h_match_count_passing_SIFT_filter,
        right_image_texture_.texObj,
        d_final_matches_passing_NCC_filter,
        d_left_sift_descriptors,
        d_right_edges,
        d_final_matches_passing_SIFT_filter,
        d_final_count_passing_SIFT_filter,
        d_sift_scores_passing_SIFT_filter,
        d_ncc_scores_passing_NCC_filter,
        d_ncc_scores_passing_SIFT_filter,
        img_right_width,
        img_right_height,
        h_match_count_passing_NCC_filter,
        start,
        stop,
        static_cast<float>(SIFT_THRESHOLD)
    );

    return precompute_left_sift_time + apply_sift_filter_time;
}

float Stereo_Matches_GPU_Pipeline::apply_BNB_NCC_filter() {

    //> BNB-NCC runs on NCC+SIFT survivors; rank by NCC score carried through SIFT.
    if (h_match_count_passing_SIFT_filter <= 0
        || d_final_matches_passing_SIFT_filter == nullptr
        || d_ncc_scores_passing_SIFT_filter == nullptr) {
        LOG_ERROR("Something's wrong: zero matches passing SIFT filter or NCC scores after SIFT are not allocated");
        h_match_count_passing_BNB_NCC_filter = 0;
        return -1.0f;
    }

    return apply_best_nearly_best_filter_pipeline(
        device_id,
        h_match_count_passing_BNB_NCC_filter,
        d_final_matches_passing_SIFT_filter,
        d_ncc_scores_passing_SIFT_filter,
        d_final_matches_passing_BNB_NCC_filter,
        d_ncc_scores_passing_BNB_NCC_filter,
        d_final_count_passing_BNB_NCC_filter,
        h_match_count_passing_SIFT_filter,
        static_cast<int>(num_left_edges_),
        static_cast<float>(BNB_NCC),
        true,
        start,
        stop
    );
}

void Stereo_Matches_GPU_Pipeline::release_merged_refined_outputs()
{
    h_match_count_merged_after_refine = 0;
    if (d_merged_refined_matches != nullptr) {
        cudacheck(cudaFree(d_merged_refined_matches));
        d_merged_refined_matches = nullptr;
    }
    if (d_merged_refined_count != nullptr) {
        cudacheck(cudaFree(d_merged_refined_count));
        d_merged_refined_count = nullptr;
    }
    if (d_final_left_patches != nullptr) {
        cudacheck(cudaFree(d_final_left_patches));
        d_final_left_patches = nullptr;
    }
    if (d_final_right_patches != nullptr) {
        cudacheck(cudaFree(d_final_right_patches));
        d_final_right_patches = nullptr;
    }
    if (d_final_left_sift_descriptors != nullptr) {
        cudacheck(cudaFree(d_final_left_sift_descriptors));
        d_final_left_sift_descriptors = nullptr;
    }
}

float Stereo_Matches_GPU_Pipeline::apply_BNB_SIFT_filter() {

    //> Sanity check
    if (h_match_count_passing_SIFT_filter <= 0 || d_sift_scores_passing_SIFT_filter == nullptr) {
        LOG_ERROR("Something's wrong: zero matches passing SIFT filter or SIFT scores are not allocated");
        h_match_count_passing_BNB_SIFT_filter = 0;
        return 0.0f;
    }

    //> Reuse the best-and-nearly-best filter kernel for SIFT similarities
    return apply_best_nearly_best_filter_pipeline(
        device_id,
        h_match_count_passing_BNB_SIFT_filter,
        d_final_matches_passing_SIFT_filter,
        d_sift_scores_passing_SIFT_filter,
        d_final_matches_passing_BNB_SIFT_filter,
        d_sift_scores_passing_BNB_SIFT_filter,
        d_final_count_passing_BNB_SIFT_filter,
        h_match_count_passing_SIFT_filter,
        static_cast<int>(num_left_edges_),
        static_cast<float>(BNB_SIFT),
        false,
        start,
        stop
    );
}

float Stereo_Matches_GPU_Pipeline::apply_epipolar_shift_photometric_refine_and_merge()
{
    release_merged_refined_outputs();
    h_match_count_merged_after_refine = 0;

    if (h_match_count_passing_BNB_SIFT_filter <= 0 || d_final_matches_passing_BNB_SIFT_filter == nullptr || d_left_patches == nullptr) {
        LOG_ERROR("Something's wrong: zero matches passing SIFT filter or SIFT scores are not allocated");
        return -1.0f;
    }

    float time_of_epipolar_shift_refine_and_merge = 
    combined_edge_epipolar_shift_and_refine_and_merge_pipeline(
        device_id,
        right_image_texture_.texObj,
        d_F,
        d_left_edges,
        d_right_edges,
        d_left_patches,
        d_final_matches_passing_BNB_SIFT_filter,
        d_merged_refined_matches,
        d_merged_refined_count,
        h_match_count_passing_BNB_SIFT_filter,
        img_right_width,
        img_right_height,
        start,
        stop
    );

    if (d_merged_refined_count != nullptr)
        cudacheck(cudaMemcpy(&h_match_count_merged_after_refine, d_merged_refined_count, sizeof(int), cudaMemcpyDeviceToHost));

    return time_of_epipolar_shift_refine_and_merge;
}

float Stereo_Matches_GPU_Pipeline::apply_second_NCC_filter() {

    int num_candidates = h_match_count_merged_after_refine;

    //> Sanity check
    if (num_candidates <= 0) {
        LOG_ERROR("Something's wrong here: no matches to apply the second NCC filter");
        return 0.0f;
    }

    //> Keep the previous matches and count by assigning the pointers to the "previous" ones
    d_prev_merged_matches = d_merged_refined_matches;
    d_prev_merged_count = d_merged_refined_count;

    //> Reset the merged refined matches and count
    d_merged_refined_matches = nullptr;
    d_merged_refined_count = nullptr;

    const float elapsed = apply_second_NCC_filter_on_merged_edges_pipeline(
        device_id,
        h_match_count_passing_NCC_filter,
        right_image_texture_.texObj,
        d_left_patches,
        d_prev_merged_matches,
        num_candidates,
        d_merged_refined_matches,
        d_merged_refined_count,
        d_ncc_scores_passing_second_NCC_filter,
        start,
        stop
    );

    if (d_prev_merged_matches != nullptr) {
        cudacheck(cudaFree(d_prev_merged_matches));
        d_prev_merged_matches = nullptr;
    }
    if (d_prev_merged_count != nullptr) {
        cudacheck(cudaFree(d_prev_merged_count));
        d_prev_merged_count = nullptr;
    }

    return elapsed;
}

float Stereo_Matches_GPU_Pipeline::apply_best_second_ncc_per_left_after_second_NCC()
{
    const int n = h_match_count_passing_NCC_filter;
    if (n <= 0 || d_merged_refined_matches == nullptr || d_merged_refined_count == nullptr
        || d_ncc_scores_passing_second_NCC_filter == nullptr) {
        LOG_ERROR("apply_best_second_ncc_per_left_after_second_NCC: invalid inputs or zero second-NCC matches");
        return 0.0f;
    }

    return apply_best_second_ncc_per_left_edge_pipeline(
        device_id,
        h_match_count_passing_NCC_filter,
        d_merged_refined_matches,
        d_merged_refined_count,
        d_ncc_scores_passing_second_NCC_filter,
        d_left_edges,
        d_left_patches,
        d_left_sift_descriptors,
        d_final_left_patches,
        d_final_right_patches,
        d_final_left_sift_descriptors,
        n,
        start,
        stop
    );

    //> The final stereo matches can be retrieved from d_merged_refined_matches with memcpy, e.g.,
    //  const int n = h_match_count_passing_NCC_filter; 
    //  std::vector<Merged_Refined_Stereo_Match_GPU> gpu_stereo_matches(static_cast<size_t>(n));
    //  cudaMemcpy(gpu_stereo_matches.data(), d_merged_refined_matches, static_cast<size_t>(n) * sizeof(Merged_Refined_Stereo_Match_GPU), cudaMemcpyDeviceToHost) );
}

void Stereo_Matches_GPU_Pipeline::retrieve_stereo_mates( const StereoFrame& frame, std::vector<final_stereo_edge_pair>& out_mates ) const
{
    out_mates.clear();

    if (d_merged_refined_matches == nullptr) {
        LOG_WARNING("retrieve_stereo_mates: no finalized GPU stereo matches available (pipeline not run yet or ownership transferred)");
        return;
    }

    int n = h_match_count_passing_NCC_filter;
    if (d_merged_refined_count != nullptr) {
        cudacheck(cudaMemcpy(&n, d_merged_refined_count, sizeof(int), cudaMemcpyDeviceToHost));
    }
    if (n <= 0) {
        LOG_WARNING("retrieve_stereo_mates: zero finalized GPU stereo matches");
        return;
    }
    std::vector<Merged_Refined_Stereo_Match_GPU> gpu_stereo_matches(static_cast<size_t>(n));
    cudacheck( cudaMemcpy(gpu_stereo_matches.data(), d_merged_refined_matches, static_cast<size_t>(n) * sizeof(Merged_Refined_Stereo_Match_GPU), cudaMemcpyDeviceToHost) );

    out_mates.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Merged_Refined_Stereo_Match_GPU& m = gpu_stereo_matches[static_cast<size_t>(i)];
        if (m.left_edge_idx < 0 || static_cast<size_t>(m.left_edge_idx) >= frame.left_edges.size())
            continue;

        final_stereo_edge_pair mate;
        mate.left_edge  = frame.left_edges[static_cast<size_t>(m.left_edge_idx)];
        //> Construct a synthetic right edge from the GPU-refined position.
        //> frame_source is inherited from the left edge (same stereo pair).
        mate.right_edge = Edge(
            cv::Point2d(static_cast<double>(m.merged_right_x), static_cast<double>(m.merged_right_y)), 
            static_cast<double>(m.merged_right_orientation), 
            false, 
            mate.left_edge.frame_source);
        
            //> Patches, descriptors, and Gamma are left as defaults right now.
        out_mates.push_back(std::move(mate));
    }
}

void Stereo_Matches_GPU_Pipeline::write_finalized_matches_to_file(size_t frame_idx) const
{
    if (d_merged_refined_matches == nullptr || d_merged_refined_count == nullptr) {
        LOG_WARNING("write_finalized_matches_to_file: refined/merge outputs not allocated; skipping");
        return;
    }

    int n = 0;
    cudaError_t err = cudaMemcpy(&n, d_merged_refined_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_ERROR(std::string("write_finalized_matches_to_file: cudaMemcpy count failed: ") + cudaGetErrorString(err));
        return;
    }
    if (n < 0) {
        LOG_ERROR("write_finalized_matches_to_file: invalid negative merged count");
        return;
    }
    if (n == 0) {
        LOG_WARNING("write_finalized_matches_to_file: zero merged matches; skipping file write");
        return;
    }

    std::vector<Merged_Refined_Stereo_Match_GPU> rows(static_cast<size_t>(n));
    err = cudaMemcpy(rows.data(), d_merged_refined_matches, static_cast<size_t>(n) * sizeof(Merged_Refined_Stereo_Match_GPU),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_ERROR(std::string("write_finalized_matches_to_file: cudaMemcpy merged rows failed: ") + cudaGetErrorString(err));
        return;
    }

    const std::string output_dir = (dataset_ != nullptr) ? dataset_->get_output_path() : std::string("output_files");
    const std::string filename = output_dir + "/gpu_finalized_stereo_edge_pairs_frame_" + std::to_string(frame_idx) + ".txt";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        LOG_ERROR("write_finalized_matches_to_file: cannot open " + filename);
        return;
    }
    outfile << "left_x left_y left_orientation merged_right_x merged_right_y merged_right_orientation\n";
    outfile.precision(8);

    for (int i = 0; i < n; ++i) {
        const Merged_Refined_Stereo_Match_GPU& row = rows[static_cast<size_t>(i)];
        const Edge_GPU& le = h_left_edges[row.left_edge_idx];
        outfile << le.location_x << " " << le.location_y << " " << le.orientation << " " << row.merged_right_x << " " << row.merged_right_y << " " << row.merged_right_orientation << "\n";
    }
    outfile.close();
    std::cout << "Wrote " << static_cast<size_t>(n) << " GPU finalized stereo edge pairs to " << filename << std::endl;
}

Stereo_Matches_GPU_Pipeline::~Stereo_Matches_GPU_Pipeline()
{
    release_merged_refined_outputs();

    delete[] h_left_edges;
    delete[] h_right_edges;
    delete[] h_cell_ids;
    delete[] h_matches_out;

    cudacheck_noexcept( cudaFree(d_F) );
    cudacheck_noexcept( cudaFree(d_left_edges) );
    cudacheck_noexcept( cudaFree(d_right_edges) );
    cudacheck_noexcept( cudaFree(d_cell_ids) );
    cudacheck_noexcept( cudaFree(d_edge_indices) );
    cudacheck_noexcept( cudaFree(d_cell_start_idx) );
    cudacheck_noexcept( cudaFree(d_num_of_edges_in_cell) );
    cudacheck_noexcept( cudaFree(d_final_matches_passing_EP_LP_OR_filters) );
    cudacheck_noexcept( cudaFree(d_match_count_passing_EP_LP_OR_filters) );
    cudacheck_noexcept( cudaFree(d_left_patches) );
    if (d_left_sift_descriptors) cudacheck_noexcept( cudaFree(d_left_sift_descriptors) );
    cudacheck_noexcept( cudaFree(d_final_matches_passing_NCC_filter) );
    cudacheck_noexcept( cudaFree(d_final_count_passing_NCC_filter) );
    if (d_ncc_scores_passing_NCC_filter) cudacheck_noexcept( cudaFree(d_ncc_scores_passing_NCC_filter) );
    if (d_final_matches_passing_SIFT_filter) cudacheck_noexcept( cudaFree(d_final_matches_passing_SIFT_filter) );
    if (d_final_count_passing_SIFT_filter) cudacheck_noexcept( cudaFree(d_final_count_passing_SIFT_filter) );
    if (d_sift_scores_passing_SIFT_filter) cudacheck_noexcept( cudaFree(d_sift_scores_passing_SIFT_filter) );
    if (d_ncc_scores_passing_SIFT_filter) cudacheck_noexcept( cudaFree(d_ncc_scores_passing_SIFT_filter) );
    if (d_final_matches_passing_BNB_NCC_filter) cudacheck_noexcept( cudaFree(d_final_matches_passing_BNB_NCC_filter) );
    if (d_final_count_passing_BNB_NCC_filter) cudacheck_noexcept( cudaFree(d_final_count_passing_BNB_NCC_filter) );
    if (d_ncc_scores_passing_BNB_NCC_filter) cudacheck_noexcept( cudaFree(d_ncc_scores_passing_BNB_NCC_filter) );
    if (d_final_matches_passing_BNB_SIFT_filter) cudacheck_noexcept( cudaFree(d_final_matches_passing_BNB_SIFT_filter) );
    if (d_final_count_passing_BNB_SIFT_filter) cudacheck_noexcept( cudaFree(d_final_count_passing_BNB_SIFT_filter) );
    if (d_sift_scores_passing_BNB_SIFT_filter) cudacheck_noexcept( cudaFree(d_sift_scores_passing_BNB_SIFT_filter) );
    if (d_ncc_scores_passing_second_NCC_filter) cudacheck_noexcept( cudaFree(d_ncc_scores_passing_second_NCC_filter) );

    cudacheck_noexcept( cudaFree(d_prev_merged_matches) );
    cudacheck_noexcept( cudaFree(d_prev_merged_count) );

    cudacheck_noexcept(cudaEventDestroy(start));
    cudacheck_noexcept(cudaEventDestroy(stop));
}

namespace {

void write_gpu_stage_match_by_edge_index_file(
    int device_id,
    const Dataset::Ptr &dataset,
    const Edge_GPU *h_left_edges,
    size_t num_left_edges_,
    const Edge_GPU *h_right_edges,
    size_t num_right_edges_,
    const Match_by_Edge_Index *d_matches,
    const int *d_count,
    const std::string &stage_tag,
    size_t frame_idx)
{
#if !WRITE_GPU_STEREO_STAGE_SNAPSHOTS
    (void)device_id;
    (void)dataset;
    (void)h_left_edges;
    (void)num_left_edges_;
    (void)h_right_edges;
    (void)num_right_edges_;
    (void)d_matches;
    (void)d_count;
    (void)stage_tag;
    (void)frame_idx;
    return;
#else
    if (!dataset || d_matches == nullptr || d_count == nullptr)
        return;

    cudacheck(cudaSetDevice(device_id));
    int n = 0;
    cudaError_t err = cudaMemcpy(&n, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_ERROR(std::string("write_gpu_stage: cudaMemcpy count failed (") + stage_tag + "): " + cudaGetErrorString(err));
        return;
    }
    if (n < 0) {
        LOG_ERROR("write_gpu_stage: negative count (" + stage_tag + ")");
        return;
    }

    std::vector<Match_by_Edge_Index> matches(static_cast<size_t>(n));
    if (n > 0) {
        err = cudaMemcpy(matches.data(), d_matches, static_cast<size_t>(n) * sizeof(Match_by_Edge_Index), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            LOG_ERROR(std::string("write_gpu_stage: cudaMemcpy matches failed (") + stage_tag + "): " + cudaGetErrorString(err));
            return;
        }
    }

    const std::string output_dir = dataset->get_output_path();
    const std::string filename = output_dir + "/gpu_stage_" + stage_tag + "_frame_" + std::to_string(frame_idx) + ".txt";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        LOG_ERROR("write_gpu_stage: cannot open " + filename);
        return;
    }
    outfile << "# GPU Match_by_Edge_Index dump at stage \"" << stage_tag << "\".\n";
    outfile << "# Columns: left_idx right_idx left_x left_y left_orientation right_x right_y right_orientation\n";
    outfile.precision(8);
    outfile << std::fixed;

    size_t skipped = 0;
    for (const auto &m : matches) {
        if (m.left_edge_idx < 0 || static_cast<size_t>(m.left_edge_idx) >= num_left_edges_ || m.right_edge_idx < 0
            || static_cast<size_t>(m.right_edge_idx) >= num_right_edges_) {
            ++skipped;
            continue;
        }
        const Edge_GPU &le = h_left_edges[m.left_edge_idx];
        const Edge_GPU &re = h_right_edges[m.right_edge_idx];
        outfile << m.left_edge_idx << " " << m.right_edge_idx << " " << le.location_x << " " << le.location_y << " " << le.orientation << " "
                << re.location_x << " " << re.location_y << " " << re.orientation << "\n";
    }
    outfile.close();
    std::cout << "Wrote " << (matches.size() - skipped) << " GPU pairs (" << stage_tag << ") to " << filename;
    if (skipped > 0)
        std::cout << " (skipped " << skipped << " invalid rows)";
    std::cout << std::endl;
#endif
}

} // namespace

#endif // STEREO_MATCHES_GPU_CPP