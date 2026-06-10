#ifndef PREPARE_FOR_GPU_HPP
#define PREPARE_FOR_GPU_HPP

#include <cmath>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <Eigen/Core>

#include "Stereo_Iterator.h"
#include "gpu_settings.h"
#include "definitions.h"
#include "gpu_kernels.h"
#include "Dataset.h"

class Prepare_For_GPU_Pipeline {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Prepare_For_GPU_Pipeline> Ptr;

    Prepare_For_GPU_Pipeline(Dataset::Ptr dataset, const StereoFrame& current_frame, int device_id);
    ~Prepare_For_GPU_Pipeline();

    //> pointers prepared for the GPU pipeline
    float h_F_row_major_[9] = {};
    float* h_F = h_F_row_major_;

    int device_id;
    int img_left_height, img_right_height;
    int img_left_width, img_right_width;

    float* h_left_img = nullptr;
    float* h_right_img = nullptr;

    //> Images cached in the texture memory
    CUDA_Texture_Wrapper left_image_texture_ = {nullptr, 0};
    CUDA_Texture_Wrapper right_image_texture_ = {nullptr, 0};

    void allocate_image_textures();
    void release_image_textures();
    bool b_image_textures_ready_ = false;

private:
    //> Pointers to the host memory


    Dataset::Ptr dataset_;
    Eigen::Matrix3f h_F_;
    
};

namespace {
    //> Picks the GPU-side image source: undistorted whenever it's available, else the raw
    //> distorted image as a graceful fallback. Used in the constructor's initializer list
    //> (for width/height) and in the constructor body (for the float buffer). Keeping a
    //> single chooser avoids drifting between the two.
    const cv::Mat& pick_image_for_gpu(const cv::Mat& undistorted, const cv::Mat& raw)
    {
        if (!undistorted.empty())
            return undistorted;
        return raw;
    }
    
    int mat_rows_prefer_undistorted(const cv::Mat& undistorted, const cv::Mat& raw)
    {
        const cv::Mat& chosen = pick_image_for_gpu(undistorted, raw);
        return chosen.empty() ? 0 : chosen.rows;
    }
    
    int mat_cols_prefer_undistorted(const cv::Mat& undistorted, const cv::Mat& raw)
    {
        const cv::Mat& chosen = pick_image_for_gpu(undistorted, raw);
        return chosen.empty() ? 0 : chosen.cols;
    }
} // namespace

#endif
