#ifndef PREPARE_FOR_GPU_CPP
#define PREPARE_FOR_GPU_CPP

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
#include "prepare_for_GPU.h"
#include "gpu_kernels.h"
#include "Dataset.h"

Prepare_For_GPU_Pipeline::Prepare_For_GPU_Pipeline(Dataset::Ptr dataset, const StereoFrame& current_frame, int device_id)
    : device_id(device_id)
    ,img_left_height(mat_rows_prefer_undistorted(current_frame.left_image_undistorted, current_frame.left_image))
    ,img_right_height(mat_rows_prefer_undistorted(current_frame.right_image_undistorted, current_frame.right_image))
    ,img_left_width(mat_cols_prefer_undistorted(current_frame.left_image_undistorted, current_frame.left_image))
    ,img_right_width(mat_cols_prefer_undistorted(current_frame.right_image_undistorted, current_frame.right_image))
    ,dataset_(std::move(dataset))
{
    //> Fundamental matrix (left to right) is casted to float matrix.
    //> Eigen stores Matrix3f column-major by default, but kernels index h_F as row-major:
    //> [0..2] = first row, [3..5] = second row, [6..8] = third row.
    h_F_ = dataset_->get_fund_mat_21().cast<float>();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            h_F_row_major_[r * 3 + c] = h_F_(r, c);
        }
    }
    h_F = h_F_row_major_;

    //> Flatten images as float. Note that the data type needs to be CV_32F
    //> Prefer the undistorted image (the same image edges are detected on and CPU SIFT reads),
    //> If the undistorted image is empty, fall back to the raw distorted image
    //> The same chain feeds img_*_width/height above, so the buffer dimensions stay in sync with the source.
    const cv::Mat& left_src  = pick_image_for_gpu( current_frame.left_image_undistorted, current_frame.left_image );
    const cv::Mat& right_src = pick_image_for_gpu( current_frame.right_image_undistorted, current_frame.right_image );
    cv::Mat left_float;
    cv::Mat right_float;
    left_src.convertTo(left_float, CV_32F);
    right_src.convertTo(right_float, CV_32F);
    if (!left_float.isContinuous())
        left_float = left_float.clone();
    if (!right_float.isContinuous())
        right_float = right_float.clone();

    h_left_img = new float[img_left_width * img_left_height];
    h_right_img = new float[img_right_width * img_right_height];
    std::memcpy(h_left_img, left_float.ptr<float>(), static_cast<size_t>(img_left_width) * static_cast<size_t>(img_left_height) * sizeof(float));
    std::memcpy(h_right_img, right_float.ptr<float>(), static_cast<size_t>(img_right_width) * static_cast<size_t>(img_right_height) * sizeof(float));

    //> Bind a valid CUDA device before any allocations (required on multi-GPU systems and some job schedulers)
    //> Discard any sticky CUDA error from earlier API use in this process (e.g. OpenCV built with CUDA).
    (void)cudaGetLastError();

    int driver_ver = 0;
    (void)cudaDriverGetVersion(&driver_ver);
    int runtime_ver = 0;
    (void)cudaRuntimeGetVersion(&runtime_ver);

    int n_cuda_devices = 0;
    cudaError_t err_count = cudaGetDeviceCount(&n_cuda_devices);
    if (err_count != cudaSuccess) {
        const char* cvd = ::getenv("CUDA_VISIBLE_DEVICES");
        std::string cvd_str = cvd ? std::string(" CUDA_VISIBLE_DEVICES=\"") + cvd + "\"." : "";
        throw std::runtime_error(
            std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err_count) + " (error code " + std::to_string(static_cast<int>(err_count)) + "). " +
            "cudaDriverGetVersion=" + std::to_string(driver_ver) + ", cudaRuntimeGetVersion=" + std::to_string(runtime_ver) + "." + cvd_str +
            " Error 999 (cudaErrorUnknown) is commonly a **NVIDIA driver vs CUDA toolkit mismatch**: the top of nvidia-smi shows the **maximum CUDA version the driver supports**; " +
            "this binary uses libcudart 12.x (see runtime number). Rebuild and run with a **CUDA toolkit module that is <= that driver max**, "
            "or use a node with a newer driver. Check `ldd` on your binary to see which libcudart is loaded.");
    }
    if (n_cuda_devices == 0) {
        throw std::runtime_error(
            "cudaGetDeviceCount returned success but reported 0 CUDA devices (runtime version " + std::to_string(runtime_ver) + "). " + 
            "If nvidia-smi lists a GPU, common causes are: (i) running outside your GPU Slurm allocation in a shell that cannot " + 
            "access /dev/nvidia*; (ii) CUDA_VISIBLE_DEVICES empty or invalid; (iii) cluster cgroup denies GPU to this process—launch " + 
            "main_edge_matching from the same interactive GPU session where nvidia-smi works.");
    }
    if (device_id < 0 || device_id >= n_cuda_devices) {
        throw std::runtime_error(
            "CUDA: invalid device_id " + std::to_string(device_id) + " (visible devices: 0-" +
            std::to_string(n_cuda_devices - 1) + "). Adjust DEVICE_ID or CUDA_VISIBLE_DEVICES.");
    }
    cudacheck(cudaSetDevice(device_id));

    //> Upload image data once per frame and keep texture objects alive for all image-sampling GPU stages
    //> This avoids re-uploading the same undistorted images in each pipeline function.
    allocate_image_textures();
}

void Prepare_For_GPU_Pipeline::allocate_image_textures()
{
    if (b_image_textures_ready_)
        return;

    cudacheck(cudaSetDevice(device_id));
    left_image_texture_  = create_CUDA_texture_object_for_img(h_left_img,  img_left_width,  img_left_height);
    right_image_texture_ = create_CUDA_texture_object_for_img(h_right_img, img_right_width, img_right_height);
    b_image_textures_ready_ = true;
}

void Prepare_For_GPU_Pipeline::release_image_textures()
{
    if (left_image_texture_.texObj != 0) {
        cudacheck_noexcept(cudaDestroyTextureObject(left_image_texture_.texObj));
        left_image_texture_.texObj = 0;
    }
    if (left_image_texture_.array != nullptr) {
        cudacheck_noexcept(cudaFreeArray(left_image_texture_.array));
        left_image_texture_.array = nullptr;
    }
    if (right_image_texture_.texObj != 0) {
        cudacheck_noexcept(cudaDestroyTextureObject(right_image_texture_.texObj));
        right_image_texture_.texObj = 0;
    }
    if (right_image_texture_.array != nullptr) {
        cudacheck_noexcept(cudaFreeArray(right_image_texture_.array));
        right_image_texture_.array = nullptr;
    }
    b_image_textures_ready_ = false;
}

Prepare_For_GPU_Pipeline::~Prepare_For_GPU_Pipeline()
{
    //> h_F aliases fixed storage h_F_row_major_; no need to free it.
    release_image_textures();

    delete[] h_left_img;
    delete[] h_right_img;
}

#endif