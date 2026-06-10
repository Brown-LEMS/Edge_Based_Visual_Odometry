//> Needed for printf inside cudacheck macro (host and nvcc compilation)
#include <cstdio>
#include <stdexcept>
#include <string>
#include "definitions.h"

//> Macro definitions for GPU settings
#define DEVICE_ID (0)

#define COPY_DATA_FROM_HOST_TO_DEVICE_FOR_TESTING (false)
#define SHOW_TIMING_BREAKDOWN (true)

#define NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING (256)
#define GRID_CELL_SIZE (16)
#define FACTOR_OF_NUM_OF_LEFT_EDGES_FOR_STEREO_EDGE_MATCHES_AFTER_EPIPOLAR_RASTERIZATION (100)

//> Temporal GPU pipeline constants
//> Max candidate buffer = num_kf_mates * this factor (spatial-grid intersection can yield many candidates)
#define FACTOR_OF_NUM_OF_KF_MATES_FOR_TEMPORAL_CANDIDATES (500)

#define WRITE_GPU_TEMPORAL_STAGE_SNAPSHOTS (false)

//> Macros related to NCC filter (must mirror Utility::get_patch_on_one_edge_side + get_Orthogonal_Shifted_Points)
//> PATCH_RADIUS: half-width in pixels; CPU loops i,j in [-half..half] with half = floor(PATCH_SIZE/2) == PATCH_SIZE/2 for odd PATCH_SIZE
#define PATCH_RADIUS                    (PATCH_SIZE / 2)
#define TOTAL_NUM_OF_PATCH_PIXELS       (PATCH_SIZE * PATCH_SIZE)
//> Orthogonal patch-center offset; CPU uses ORTHOGONAL_SHIFT_MAG in Utility::get_Orthogonal_Shifted_Points(edge)
#define OFFSET_DIST                     (ORTHOGONAL_SHIFT_MAG)

//> GPU SIFT-style descriptor parameters
#define SIFT_ORTHOGONAL_SHIFT (8.0)
#define SIFT_DESCRIPTOR_GRID_SIZE (4)
#define SIFT_DESCRIPTOR_ORIENTATION_BINS (8)
#define SIFT_DESCRIPTOR_DIM (SIFT_DESCRIPTOR_GRID_SIZE * SIFT_DESCRIPTOR_GRID_SIZE * SIFT_DESCRIPTOR_ORIENTATION_BINS)
#define SIFT_DESCRIPTOR_SAMPLE_WIDTH (16)
#define SIFT_DESCRIPTOR_SCALE (512.0)

//> GPU epipolar shift and refine parameters
#define GPU_EP_REFINE_MAX_ITER (20)
#define GPU_EP_REFINE_TOL (1e-3f)
#define GPU_EP_REFINE_HUBER_DELTA (3.0f)
#define GPU_MERGE_MAX_SEGMENT_SIZE (384)

//> Temporal photometric refinement parameters (2D Gauss-Newton)
//> Reuses GPU_EP_REFINE_MAX_ITER, GPU_EP_REFINE_TOL, GPU_EP_REFINE_HUBER_DELTA from stereo.
//> A candidate is invalid if its final RMS > outlier threshold.
#define GPU_TEMPORAL_REFINE_OUTLIER_THRESH (GPU_EP_REFINE_HUBER_DELTA * 2.0f)

//> Temporal edge matching GPU settings
#define NCC_THRESH_TEMPORAL  (0.8)
#define SIFT_THRESH_TEMPORAL (200.0)
#define BNB_SIFT_TEMPORAL    (0.8)
#define BNB_NCC_TEMPORAL     (0.9)

//> CUDA error check (logs only; for destructors and other noexcept cleanup)
#define cudacheck_noexcept( a )  do { \
    cudaError_t e = (a); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "\033[1;31mError in %s:%d %s (code %d)\033[0m\n", \
                __func__, __LINE__, cudaGetErrorString(e), static_cast<int>(e)); \
    } \
} while(0)

//> CUDA error check (throws so failures cannot cascade into misleading follow-on errors)
#define cudacheck( a )  do { \
    cudaError_t e = (a); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "\033[1;31mError in %s:%d %s (code %d)\033[0m\n", \
                __func__, __LINE__, cudaGetErrorString(e), static_cast<int>(e)); \
        throw std::runtime_error( \
            std::string("CUDA error in ") + __func__ + ": " + cudaGetErrorString(e)); \
    } \
} while(0)

//> Safe free (noexcept: used during teardown)
#define safe_free( ptr ) do { if (ptr) { cudacheck_noexcept(cudaFree(ptr)); ptr = nullptr; } } while(0)
