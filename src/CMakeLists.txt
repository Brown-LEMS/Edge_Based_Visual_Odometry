set (control_sources
        Dataset.cpp
        Pipeline.cpp
        utility.cpp
        MotionTracker.cpp
        toed/cpu_toed.cpp
        Stereo_Matches.cpp
        Temporal_Matches.cpp
        Stereo_Iterator.cpp
        EdgeClusterer.cpp

        prepare_for_GPU.cpp
        Stereo_Matches_GPU.cpp
        Temporal_Matches_GPU.cpp
)

if(ENABLE_CUDA)
    list(APPEND control_sources
        gpu_kernels/spatial_grid_mapping.cu
        gpu_kernels/temporal_spatial_grid_mapping.cu
        gpu_kernels/epipolar_rasterization.cu
        gpu_kernels/precompute_left_edge_patches_photometry.cu
        gpu_kernels/precompute_left_edge_sift_descriptors.cu
        gpu_kernels/apply_NCC_on_the_fly.cu
        gpu_kernels/apply_SIFT_descriptor_filter.cu
        gpu_kernels/apply_best_nearly_best_filter.cu
        gpu_kernels/combined_edge_epipolar_shift_and_refine_and_merge_pipeline.cu
        gpu_kernels/apply_second_NCC_filter.cu
        gpu_kernels/apply_best_second_ncc_per_left_edge.cu
        gpu_kernels/temporal_candidate_generation.cu
        gpu_kernels/apply_temporal_quad_ncc_filter.cu
        gpu_kernels/apply_temporal_quad_sift_filter.cu
        gpu_kernels/apply_temporal_quad_bnb_filter.cu
        gpu_kernels/temporal_photometric_refine_and_cluster.cu
        gpu_kernels/extract_mate_ordered_left_sift.cu
    )
endif()

add_library( lems_vo_slam ${control_sources} )
include_directories(/gpfs/data/bkimia/opencv-4.x/build_install/include/opencv4)

#> Linking third-party libraries
target_link_libraries(lems_vo_slam
    PUBLIC OpenMP::OpenMP_CXX
    -L/gpfs/data/bkimia/cchien3/yaml-cpp/bin/lib64 yaml-cpp
    ${THIRD_PARTY_LIBS}
)

# Add optional glog and gflags linking if found and enabled
if(USE_GLOGS AND GLOG_FOUND AND GLOG_LIBRARIES)
    target_link_libraries(lems_vo_slam PUBLIC ${GLOG_LIBRARIES})
    if(GLOG_INCLUDE_DIRS)
        target_include_directories(lems_vo_slam PUBLIC ${GLOG_INCLUDE_DIRS})
    endif()
endif()

if(USE_GLOGS AND GFLAGS_FOUND AND GFLAGS_LIBRARIES)
    target_link_libraries(lems_vo_slam PUBLIC ${GFLAGS_LIBRARIES})
    if(GFLAGS_INCLUDE_DIRS)
        target_include_directories(lems_vo_slam PUBLIC ${GFLAGS_INCLUDE_DIRS})
    endif()
endif()