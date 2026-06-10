#ifndef GN_REFINEMENT_2D_FREE_CUH
#define GN_REFINEMENT_2D_FREE_CUH

#include <cuda_runtime.h>
#include <math.h>

#include "gpu_settings.h"

//> Gauss-Newton 2D photometric refinement for one (KF edge, CF edge) pair
//  when constructing temporal quad candidates 
__device__ static void GN_refinement_2D_free(
    const float* __restrict__ Lc_plus,   //> mean-centered KF patch (+)
    const float* __restrict__ Lc_minus,  //> mean-centered KF patch (-)
    float                     kf_x,
    float                     kf_y,
    float                     kf_orientation,
    float                     cf_x,
    float                     cf_y,
    cudaTextureObject_t       cf_tex,
    int img_w, int img_h,
    float& dx_out, float& dy_out, float& rms_out, bool& valid_out)
{
    const float cos_L      = cosf(kf_orientation);
    const float sin_L      = sinf(kf_orientation);
    const float side_shift = static_cast<float>(PATCH_SIZE) * 0.5f + 1.0f;

    // Initial displacement: CF should be at (kf_pos - d), so d_init = kf_pos - cf_pos
    float dx = kf_x - cf_x;
    float dy = kf_y - cf_y;
    float last_rms = 1e6f;

    float Ip[TOTAL_NUM_OF_PATCH_PIXELS];
    float Im[TOTAL_NUM_OF_PATCH_PIXELS];

    for (int iter = 0; iter < GPU_EP_REFINE_MAX_ITER; ++iter) {
        const float cf_sample_x = kf_x - dx;
        const float cf_sample_y = kf_y - dy;

        //> Patch center positions (same sign convention as CPU: n = (-sin, cos))
        const float cRp_x = cf_sample_x - sin_L * side_shift;
        const float cRp_y = cf_sample_y + cos_L * side_shift;
        const float cRm_x = cf_sample_x + sin_L * side_shift;
        const float cRm_y = cf_sample_y - cos_L * side_shift;

        sample_rot_patch_I(cf_tex, cRp_x, cRp_y, cos_L, sin_L, img_w, img_h, Ip);
        sample_rot_patch_I(cf_tex, cRm_x, cRm_y, cos_L, sin_L, img_w, img_h, Im);

        const float mean_Rp = mean_patch(Ip);
        const float mean_Rm = mean_patch(Im);

        //> Accumulate 2×2 Hessian H and 2D gradient vector b (Huber-weighted).
        //> Jacobian J = [gx, gy] because r = L - (R(cf_pos) - meanR)
        //> and cf_pos = kf_pos - d, so d(R)/d(dx) = -gx → d(r)/d(dx) = gx.
        double H00 = 1e-6, H01 = 0.0, H11 = 1e-6;
        double b0 = 0.0, b1 = 0.0, cost = 0.0;

        for (int pass = 0; pass < 2; ++pass) {
            const float* Lc = (pass == 0) ? Lc_plus  : Lc_minus;
            const float* Ir = (pass == 0) ? Ip       : Im;
            const float  mR = (pass == 0) ? mean_Rp  : mean_Rm;
            const float  cx = (pass == 0) ? cRp_x    : cRm_x;
            const float  cy = (pass == 0) ? cRp_y    : cRm_y;

            int kk = 0;
            for (int vv = -PATCH_RADIUS; vv <= PATCH_RADIUS; ++vv) {
                for (int uu = -PATCH_RADIUS; uu <= PATCH_RADIUS; ++uu, ++kk) {
                    const float sx = cx + static_cast<float>(uu) * cos_L
                                       - static_cast<float>(vv) * sin_L;
                    const float sy = cy + static_cast<float>(uu) * sin_L
                                       + static_cast<float>(vv) * cos_L;
                    float gx = 0.0f, gy = 0.0f;
                    sample_grad_I(cf_tex, sx, sy, img_w, img_h, gx, gy);

                    const float r    = Lc[kk] - (Ir[kk] - mR);
                    const float absr = fabsf(r);
                    const float w    = (absr < GPU_EP_REFINE_HUBER_DELTA)
                                        ? 1.0f : GPU_EP_REFINE_HUBER_DELTA / absr;
                    H00  += w * static_cast<double>(gx) * static_cast<double>(gx);
                    H01  += w * static_cast<double>(gx) * static_cast<double>(gy);
                    H11  += w * static_cast<double>(gy) * static_cast<double>(gy);
                    b0   += w * static_cast<double>(gx) * static_cast<double>(r);
                    b1   += w * static_cast<double>(gy) * static_cast<double>(r);
                    cost += w * static_cast<double>(r)  * static_cast<double>(r);
                }
            }
        }

        //> Solve 2×2 system via Cramer's rule.
        const double det = H00 * H11 - H01 * H01;
        if (fabs(det) < 1e-8) break;

        const float ddx = -static_cast<float>((H11 * b0 - H01 * b1) / det);
        const float ddy = -static_cast<float>((-H01 * b0 + H00 * b1) / det);
        dx += ddx;
        dy += ddy;

        last_rms = sqrtf(static_cast<float>(cost)
                         / static_cast<float>(TOTAL_NUM_OF_PATCH_PIXELS * 2));

        if (sqrtf(ddx * ddx + ddy * ddy) < GPU_EP_REFINE_TOL
            || iter == GPU_EP_REFINE_MAX_ITER - 1)
            break;
    }

    dx_out    = dx;
    dy_out    = dy;
    rms_out   = last_rms;
    valid_out = (last_rms < GPU_TEMPORAL_REFINE_OUTLIER_THRESH);
}

#endif