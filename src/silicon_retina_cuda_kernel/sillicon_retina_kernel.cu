#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "enums.h"

#define CUDA_CHECK(call)                                                          \
    do                                                                            \
    {                                                                             \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess)                                                   \
        {                                                                         \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl;                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// add constants for better maintainability

__constant__ float BLUR_kernel_3x3[9] = {
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
    1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f};

// improved blur function with bounds checking and shared memory optimization
__device__ float apply_blur_avg(int kernel_size, int W, int H,
                                const float *input,
                                int center_idx, int curr_t_offset)
{
    // calculate the 2D coordaintes from linear index
    int y = center_idx / W;
    int x = center_idx % W;

    float output = 0.0f;
    float cnt = 0.0f;

    int ks = kernel_size / 2;
    // unrolled loop for 3 x 3 kernel (most common case)
    if (kernel_size == 3)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                int ny = y + dy;
                int nx = x + dx;

                // bounds check
                if (ny >= 0 && ny < H && nx >= 0 && nx < W)
                {
                    int neighbor_idx = ny * W + nx;
                    output += input[curr_t_offset + neighbor_idx];
                    cnt += 1.0f;
                }
            }
        }
    }
    else
    {
        // General case for other kernel sizes
        for (int dy = -ks; dy <= ks; ++dy)
        {
            for (int dx = -ks; dx <= ks; ++dx)
            {
                int ny = y + dy;
                int nx = x + dx;

                if (ny >= 0 && ny < H && nx >= 0 && nx < W)
                {
                    int neighbor_idx = ny * W + nx;
                    output += input[curr_t_offset + neighbor_idx];
                    cnt += 1.0f;
                }
            }
        }
    }

    return (cnt > 0.0f) ? output / cnt : 0.0f;
}

__device__ float apply_blur(BLUR_MODE mode,
                            int kernel_size, int W, int H,
                            const float *input,
                            int center_idx, int curr_t_offset)
{
    if (mode == BLUR_MODE::BLUR_AVG)
    {
        return apply_blur_avg(kernel_size, W, H, input, center_idx, curr_t_offset);
    }
    // TODO

    return apply_blur_avg(kernel_size, W, H, input, center_idx, curr_t_offset);
}

__forceinline__ __device__ float apply_low_pass_filter1(float incoming_value, float previous_value, float alpha)
{ // fmaf(x, y, z) -> z + x * y
    return fmaf(alpha, incoming_value - previous_value, previous_value);
}

__device__ float apply_low_pass_filter(LPF_MODE mode,
                                       float incoming_value,
                                       float previous_value,
                                       float alpha)
{
    if (mode == LPF_MODE::LPF_1)
    {
        return apply_low_pass_filter1(incoming_value, previous_value, alpha);
    }

    return apply_low_pass_filter1(incoming_value, previous_value, alpha);
}

__forceinline__ __device__ float apply_high_pass_filter1(float incoming_value, float previous_value)
{
    return incoming_value - previous_value;
}

__device__ float apply_high_pass_filter(HPF_MODE mode,
                                        float incoming_value,
                                        float previous_value)
{
    if (mode == HPF_MODE::HPF_1)
    {
        return apply_high_pass_filter1(incoming_value, previous_value);
    }

    return apply_high_pass_filter1(incoming_value, previous_value);
}

__forceinline__ __device__ float euler_method(float previous_value,
                                              float p1,
                                              float I_OPL,
                                              float g_A)
{
    // fmaf(x, y, z) -> z + x * y
    return fmaf(p1, fmaf(-g_A, previous_value, I_OPL), previous_value);
    // return previous_value + p1 * (I_OPL - g_A * previous_value);
}

__forceinline__ __device__ float apply_N(float incoming, float bias)
{
    return fmaxf(incoming, 0.0f) + bias;
}

__forceinline__ __device__ float compute_next_LIF1(float previous_LIF,
                                                   float p2,
                                                   float p3,
                                                   float I_gang,
                                                   float U_rest)
{
    // fmaf(x,y, z) -> z + x * y
    return fmaf(-p3, fabsf(previous_LIF - U_rest), fmaf(p2, I_gang, previous_LIF));
    // return previous_LIF + p2 * I_gang - p3 * abs(previous_LIF - U_rest);
}

__device__ float compute_next_LIF(LIF_INTEGRATION_MODE mode,
                                  float previous_LIF,
                                  float p2,
                                  float p3,
                                  float I_gang,
                                  float U_rest)
{
    if (mode == LIF_INTEGRATION_MODE::LIF_INTEGRATION_1)
    {
        return compute_next_LIF1(previous_LIF, p2, p3, I_gang, U_rest);
    }

    return compute_next_LIF1(previous_LIF, p2, p3, I_gang, U_rest);
}

__forceinline__ __device__ char fire_and_reset(LIF_RESET_MODE mode,
                                               float &V_LIF,
                                               float U_th,
                                               float U_reset)
{
    if (mode == LIF_RESET_MODE::LIF_SUBTRACT_RESET)
    {
        if (V_LIF >= U_th)
        {
            V_LIF -= U_reset;
            return 1; // mark as fired
        }
        return 0; // mark as not fired
    }

    return 0;
}

__global__ void sillicon_retina_kernel(
    const float *__restrict__ imgs,
    float *LPF1,
    float *HPF1,
    float *BLURRED2,
    float *LPF2,
    float *I_OPL,
    float *V_BIP,
    float *Q_V_BIP,
    float *V_BIP_LPF,
    float *HPF2,
    float *I_Gang,
    float *V_LIF,
    unsigned char *fired,

    int T, int H, int W,
    float alpha1,
    float alpha2,
    float V_BIP_LPF_alpha,
    float lambda_OPL,
    float G_A_zero,
    float lambda_A,
    float p1,
    float I_bias,
    float p2,
    float p3,
    float U_rest,
    float U_th,
    float U_reset)
{
    // linear index
    const int center_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // check that thread is not out of valid range
    if (center_idx >= H * W)
        return;

    LPF1[center_idx] = imgs[center_idx];
    HPF1[center_idx] = imgs[center_idx];
    BLURRED2[center_idx] = imgs[center_idx];
    LPF2[center_idx] = imgs[center_idx];
    I_OPL[center_idx] = imgs[center_idx];
    V_BIP[center_idx] = 0.0f;
    Q_V_BIP[center_idx] = 0.0f;
    V_BIP_LPF[center_idx] = 0.0f;
    HPF2[center_idx] = 0.0f;
    I_Gang[center_idx] = 0.0f;
    V_LIF[center_idx] = 0.0f;

    for (int t = 1; t < T; t++)
    {
        const int prev_t_offset = (t - 1) * H * W;
        const int curr_t_offset = t * H * W;

        const int prev_t_idx = prev_t_offset + center_idx;
        const int curr_t_idx = curr_t_offset + center_idx;

        // Stage 1 : low-pass filter
        LPF1[curr_t_idx] = apply_low_pass_filter(LPF_MODE::LPF_1,
                                                 imgs[curr_t_idx],
                                                 LPF1[prev_t_idx],
                                                 alpha1);

        // Stage 2 : high-pass filter
        HPF1[curr_t_idx] = apply_high_pass_filter(HPF_MODE::HPF_1,
                                                  LPF1[curr_t_idx],
                                                  LPF1[prev_t_idx]);

        // stage 3 : second low pass filter
        LPF2[curr_t_idx] = apply_low_pass_filter(LPF_MODE::LPF_1,
                                                 HPF1[curr_t_idx],
                                                 LPF2[prev_t_idx],
                                                 alpha2);

        // stage 4 : blur operation
        BLURRED2[curr_t_idx] = apply_blur(BLUR_MODE::BLUR_AVG,
                                          3, W, H, LPF2, center_idx, curr_t_offset);

        // Stage 5 : Compute current from OPL (outer plexiform layer)
        I_OPL[curr_t_idx] = lambda_OPL * (BLURRED2[curr_t_idx] - LPF2[curr_t_idx]);

        // Stage 6: Compute and clamp the Q function
        Q_V_BIP[curr_t_idx] = G_A_zero + lambda_A * fabsf(V_BIP[prev_t_idx]);
        Q_V_BIP[curr_t_idx] = fmaxf(0.0f, fminf(Q_V_BIP[curr_t_idx], 1.0f));

        // Stage 7: V_BIP_LPF computatio
        V_BIP_LPF[curr_t_idx] = apply_low_pass_filter(LPF_MODE::LPF_1,
                                                      Q_V_BIP[curr_t_idx],
                                                      V_BIP_LPF[prev_t_idx],
                                                      V_BIP_LPF_alpha);

        // Stage 8: g_A computation
        float g_A = apply_blur(BLUR_MODE::BLUR_AVG, 3, W, H, V_BIP_LPF, center_idx, curr_t_offset);

        // Stage 9: V_BIP update using Euler method
        V_BIP[curr_t_idx] = euler_method(V_BIP[prev_t_idx], p1, I_OPL[curr_t_idx], g_A);

        // Stage 10: HPF2 computation
        HPF2[curr_t_idx] = apply_high_pass_filter(HPF_MODE::HPF_1,
                                                  V_BIP[curr_t_idx],
                                                  V_BIP[prev_t_idx]);

        // Stage 11: I_Gang computation
        I_Gang[curr_t_idx] = apply_N(HPF2[curr_t_idx], I_bias);

        // Stage 12: LIF membrane potential update
        V_LIF[curr_t_idx] = compute_next_LIF(LIF_INTEGRATION_MODE::LIF_INTEGRATION_1,
                                             V_LIF[prev_t_idx],
                                             p2, p3,
                                             I_Gang[curr_t_idx],
                                             U_rest);

        // Stage 13: Fire and reset
        fired[curr_t_idx] = fire_and_reset(LIF_RESET_MODE::LIF_SUBTRACT_RESET,
                                           V_LIF[curr_t_idx],
                                           U_th,
                                           U_reset);
    }
}

void run_silicon_retina_model(
    const float *d_imgs,
    float *d_LPF1,
    float *d_HPF1,
    float *d_BLURRED2,
    float *d_LPF2,
    float *d_I_OPL,
    float *d_V_BIP,
    float *d_Q_V_BIP,
    float *d_V_BIP_LPF,
    float *d_HPF2,
    float *d_I_Gang,
    float *d_V_LIF,
    unsigned char *d_fired,

    int T, int H, int W,
    float alpha1,
    float alpha2,
    float V_BIP_LPF_alpha,
    float lambda_OPL,
    float G_A_zero,
    float lambda_A,
    float p1,
    float I_bias,
    float p2,
    float p3,
    float U_rest,
    float U_th,
    float U_reset)
{
    // calculate the optimal block size
    int device;

    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // use occupancy calculator for optimal block size
    int blockSize;
    int minGridSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, sillicon_retina_kernel, 0, 0));

    // ensure block size is reasoanble
    blockSize = (blockSize / 32) * 32;

    if (blockSize == 0)
        blockSize = 256;

    int gridSize = (H * W + blockSize - 1) / blockSize;

    sillicon_retina_kernel<<<gridSize, blockSize>>>(
        d_imgs,
        d_LPF1,
        d_HPF1,
        d_BLURRED2,
        d_LPF2,
        d_I_OPL,
        d_V_BIP,
        d_Q_V_BIP,
        d_V_BIP_LPF,
        d_HPF2,
        d_I_Gang,
        d_V_LIF,
        d_fired,

        T, H, W,
        alpha1,
        alpha2,
        V_BIP_LPF_alpha,
        lambda_OPL,
        G_A_zero,
        lambda_A,
        p1,
        I_bias,
        p2,
        p3,
        U_rest,
        U_th,
        U_reset);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


__global__ void sillicon_retina_kernel_fired_only(const float *__restrict__ imgs,
                                                  float *LPF1,
                                                  float *HPF1,
                                                  float *BLURRED2,
                                                  float *LPF2,
                                                  float *I_OPL,
                                                  float *V_BIP,
                                                  float *Q_V_BIP,
                                                  float *V_BIP_LPF,
                                                  float *HPF2,
                                                  float *I_Gang,
                                                  float *V_LIF,
                                                  unsigned char *fired,

                                                  int T, int H, int W,
                                                  float alpha1,
                                                  float alpha2,
                                                  float V_BIP_LPF_alpha,
                                                  float lambda_OPL,
                                                  float G_A_zero,
                                                  float lambda_A,
                                                  float p1,
                                                  float I_bias,
                                                  float p2,
                                                  float p3,
                                                  float U_rest,
                                                  float U_th,
                                                  float U_reset)
{
    // linear index
    const int center_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // check that thread is not out of valid range
    if (center_idx >= H * W)
        return;

    LPF1[center_idx] = 0.0f;
    HPF1[center_idx] = 0.0f;
    BLURRED2[center_idx] = 0.0f;
    LPF2[center_idx] = 0.0f;
    I_OPL[center_idx] = 0.0f;
    V_BIP[center_idx] = 0.0f;
    Q_V_BIP[center_idx] = 0.0f;
    V_BIP_LPF[center_idx] = 0.0f;
    HPF2[center_idx] = 0.0f;
    I_Gang[center_idx] = 0.0f;
    V_LIF[center_idx] = 0.0f;

    for (int t = 1; t < T; ++t)
    {
        const int current_double_buffer_idx = t % 2;
        const int previous_double_buffer_idx = (t - 1) % 2;

        const int curr_buff_t_offset = current_double_buffer_idx * H * W;
        const int prev_buff_t_offset = previous_double_buffer_idx * H * W;

        const int curr_buff_t_idx = center_idx + curr_buff_t_offset;
        const int prev_buff_t_idx = center_idx + prev_buff_t_offset;

        const int curr_t_idx = t * H * W + center_idx;

        // Stage 1 : low-pass filter
        LPF1[curr_buff_t_idx] = apply_low_pass_filter(LPF_MODE::LPF_1,
                                                      imgs[curr_t_idx],
                                                      LPF1[prev_buff_t_idx],
                                                      alpha1);

        // Stage 2 : high-pass filter
        HPF1[curr_buff_t_idx] = apply_high_pass_filter(HPF_MODE::HPF_1,
                                                       LPF1[curr_buff_t_idx],
                                                       LPF1[prev_buff_t_idx]);

        // stage 3 : second low pass filter
        LPF2[curr_buff_t_idx] = apply_low_pass_filter(LPF_MODE::LPF_1,
                                                      HPF1[curr_buff_t_idx],
                                                      LPF2[prev_buff_t_idx],
                                                      alpha2);

        // stage 4 : blur operation
        BLURRED2[curr_buff_t_idx] = apply_blur(BLUR_MODE::BLUR_AVG,
                                               3, W, H, LPF2, center_idx, curr_buff_t_offset);

        // Stage 5 : Compute current from OPL (outer plexiform layer)
        I_OPL[curr_buff_t_idx] = lambda_OPL * (BLURRED2[curr_buff_t_idx] - LPF2[curr_buff_t_idx]);

        // Stage 6: Compute and clamp the Q function
        Q_V_BIP[curr_buff_t_idx] = G_A_zero + lambda_A * fabsf(V_BIP[prev_buff_t_idx]);
        Q_V_BIP[curr_buff_t_idx] = fmaxf(0.0f, fminf(Q_V_BIP[curr_buff_t_idx], 1.0f));

        // Stage 7: V_BIP_LPF computatio
        V_BIP_LPF[curr_buff_t_idx] = apply_low_pass_filter(LPF_MODE::LPF_1,
                                                           Q_V_BIP[curr_buff_t_idx],
                                                           V_BIP_LPF[prev_buff_t_idx],
                                                           V_BIP_LPF_alpha);

        // Stage 8: g_A computation
        float g_A = apply_blur(BLUR_MODE::BLUR_AVG, 3, W, H, V_BIP_LPF, center_idx, curr_buff_t_offset);

        // Stage 9: V_BIP update using Euler method
        V_BIP[curr_buff_t_idx] = euler_method(V_BIP[prev_buff_t_idx], p1, I_OPL[curr_buff_t_idx], g_A);

        // Stage 10: HPF2 computation
        HPF2[curr_buff_t_idx] = apply_high_pass_filter(HPF_MODE::HPF_1,
                                                       V_BIP[curr_buff_t_idx],
                                                       V_BIP[prev_buff_t_idx]);

        // Stage 11: I_Gang computation
        I_Gang[curr_buff_t_idx] = apply_N(HPF2[curr_buff_t_idx], I_bias);

        // Stage 12: LIF membrane potential update
        V_LIF[curr_buff_t_idx] = compute_next_LIF(LIF_INTEGRATION_MODE::LIF_INTEGRATION_1,
                                                  V_LIF[prev_buff_t_idx],
                                                  p2, p3,
                                                  I_Gang[curr_buff_t_idx],
                                                  U_rest);

        // Stage 13: Fire and reset
        fired[curr_t_idx] = fire_and_reset(LIF_RESET_MODE::LIF_SUBTRACT_RESET,
                                                V_LIF[curr_buff_t_idx],
                                                U_th,
                                                U_reset);
    }
}

void run_silicon_retina_model_fired_only(
    const float *d_imgs,
    float *d_LPF1,
    float *d_HPF1,
    float *d_BLURRED2,
    float *d_LPF2,
    float *d_I_OPL,
    float *d_V_BIP,
    float *d_Q_V_BIP,
    float *d_V_BIP_LPF,
    float *d_HPF2,
    float *d_I_Gang,
    float *d_V_LIF,
    unsigned char *d_fired,

    int T, int H, int W,
    float alpha1,
    float alpha2,
    float V_BIP_LPF_alpha,
    float lambda_OPL,
    float G_A_zero,
    float lambda_A,
    float p1,
    float I_bias,
    float p2,
    float p3,
    float U_rest,
    float U_th,
    float U_reset)
{
    // launch the kernel
    int blockSize = 256;
    int gridSize = (H * W + blockSize - 1) / blockSize;

    sillicon_retina_kernel_fired_only<<<gridSize, blockSize>>>(
        d_imgs,
        d_LPF1,
        d_HPF1,
        d_BLURRED2,
        d_LPF2,
        d_I_OPL,
        d_V_BIP,
        d_Q_V_BIP,
        d_V_BIP_LPF,
        d_HPF2,
        d_I_Gang,
        d_V_LIF,
        d_fired,

        T, H, W,
        alpha1,
        alpha2,
        V_BIP_LPF_alpha,
        lambda_OPL,
        G_A_zero,
        lambda_A,
        p1,
        I_bias,
        p2,
        p3,
        U_rest,
        U_th,
        U_reset);
}

__global__ void silicon_retina_kernel_online(const float *__restrict__ imgs,
                                             float *LPF1,
                                             float *HPF1,
                                             float *BLURRED2,
                                             float *LPF2,
                                             float *I_OPL,
                                             float *V_BIP,
                                             float *Q_V_BIP,
                                             float *V_BIP_LPF,
                                             float *HPF2,
                                             float *I_Gang,
                                             float *V_LIF,
                                             unsigned char *fired,

                                             int curr_t, int H, int W,
                                             float alpha1,
                                             float alpha2,
                                             float V_BIP_LPF_alpha,
                                             float lambda_OPL,
                                             float G_A_zero,
                                             float lambda_A,
                                             float p1,
                                             float I_bias,
                                             float p2,
                                             float p3,
                                             float U_rest,
                                             float U_th,
                                             float U_reset)
{
    // linear index
    const int center_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // check that thread is not out of valid range
    if (center_idx >= H * W)
        return;

    if (curr_t <= 0)
    {
        // initialize the buffers
        LPF1[center_idx] = 0.0f;
        HPF1[center_idx] = 0.0f;
        BLURRED2[center_idx] = 0.0f;
        LPF2[center_idx] = 0.0f;
        I_OPL[center_idx] = 0.0f;
        V_BIP[center_idx] = 0.0f;
        Q_V_BIP[center_idx] = 0.0f;
        V_BIP_LPF[center_idx] = 0.0f;
        HPF2[center_idx] = 0.0f;
        I_Gang[center_idx] = 0.0f;
        V_LIF[center_idx] = 0.0f;

        return;
    }

    const int current_double_buffer_idx = curr_t % 2;
    const int previous_double_buffer_idx = (curr_t - 1) % 2;

    const int curr_buff_t_offset = current_double_buffer_idx * H * W;
    const int prev_buff_t_offset = previous_double_buffer_idx * H * W;

    const int curr_buff_t_idx = center_idx + curr_buff_t_offset;
    const int prev_buff_t_idx = center_idx + prev_buff_t_offset;

    // Stage 1 : low-pass filter
    LPF1[curr_buff_t_idx] = apply_low_pass_filter(LPF_MODE::LPF_1,
                                                  imgs[center_idx],
                                                  LPF1[prev_buff_t_idx],
                                                  alpha1);

    // Stage 2 : high-pass filter
    HPF1[curr_buff_t_idx] = apply_high_pass_filter(HPF_MODE::HPF_1,
                                                   LPF1[curr_buff_t_idx],
                                                   LPF1[prev_buff_t_idx]);

    // stage 3 : second low pass filter
    LPF2[curr_buff_t_idx] = apply_low_pass_filter(LPF_MODE::LPF_1,
                                                  HPF1[curr_buff_t_idx],
                                                  LPF2[prev_buff_t_idx],
                                                  alpha2);

    // stage 4 : blur operation
    BLURRED2[curr_buff_t_idx] = apply_blur(BLUR_MODE::BLUR_AVG,
                                           3, W, H, LPF2, center_idx, curr_buff_t_offset);

    // Stage 5 : Compute current from OPL (outer plexiform layer)
    I_OPL[curr_buff_t_idx] = lambda_OPL * (BLURRED2[curr_buff_t_idx] - LPF2[curr_buff_t_idx]);

    // Stage 6: Compute and clamp the Q function
    Q_V_BIP[curr_buff_t_idx] = G_A_zero + lambda_A * fabsf(V_BIP[prev_buff_t_idx]);
    Q_V_BIP[curr_buff_t_idx] = fmaxf(0.0f, fminf(Q_V_BIP[curr_buff_t_idx], 1.0f));

    // Stage 7: V_BIP_LPF computatio
    V_BIP_LPF[curr_buff_t_idx] = apply_low_pass_filter(LPF_MODE::LPF_1,
                                                       Q_V_BIP[curr_buff_t_idx],
                                                       V_BIP_LPF[prev_buff_t_idx],
                                                       V_BIP_LPF_alpha);

    // Stage 8: g_A computation
    float g_A = apply_blur(BLUR_MODE::BLUR_AVG, 3, W, H, V_BIP_LPF, center_idx, curr_buff_t_offset);

    // Stage 9: V_BIP update using Euler method
    V_BIP[curr_buff_t_idx] = euler_method(V_BIP[prev_buff_t_idx], p1, I_OPL[curr_buff_t_idx], g_A);

    // Stage 10: HPF2 computation
    HPF2[curr_buff_t_idx] = apply_high_pass_filter(HPF_MODE::HPF_1,
                                                   V_BIP[curr_buff_t_idx],
                                                   V_BIP[prev_buff_t_idx]);

    // Stage 11: I_Gang computation
    I_Gang[curr_buff_t_idx] = apply_N(HPF2[curr_buff_t_idx], I_bias);

    // Stage 12: LIF membrane potential update
    V_LIF[curr_buff_t_idx] = compute_next_LIF(LIF_INTEGRATION_MODE::LIF_INTEGRATION_1,
                                              V_LIF[prev_buff_t_idx],
                                              p2, p3,
                                              I_Gang[curr_buff_t_idx],
                                              U_rest);

    // Stage 13: Fire and reset
    fired[curr_buff_t_idx] = fire_and_reset(LIF_RESET_MODE::LIF_SUBTRACT_RESET,
                                            V_LIF[curr_buff_t_idx],
                                            U_th,
                                            U_reset);
}

void run_silicon_retina_model_online(
    const float *d_imgs,
    float *d_LPF1,
    float *d_HPF1,
    float *d_BLURRED2,
    float *d_LPF2,
    float *d_I_OPL,
    float *d_V_BIP,
    float *d_Q_V_BIP,
    float *d_V_BIP_LPF,
    float *d_HPF2,
    float *d_I_Gang,
    float *d_V_LIF,
    unsigned char *d_fired,

    int curr_t, int H, int W,
    float alpha1,
    float alpha2,
    float V_BIP_LPF_alpha,
    float lambda_OPL,
    float G_A_zero,
    float lambda_A,
    float p1,
    float I_bias,
    float p2,
    float p3,
    float U_rest,
    float U_th,
    float U_reset)
{
    // launch the kernel
    int blockSize = 256;
    int gridSize = (H * W + blockSize - 1) / blockSize;

    silicon_retina_kernel_online<<<gridSize, blockSize>>>(
        d_imgs,
        d_LPF1,
        d_HPF1,
        d_BLURRED2,
        d_LPF2,
        d_I_OPL,
        d_V_BIP,
        d_Q_V_BIP,
        d_V_BIP_LPF,
        d_HPF2,
        d_I_Gang,
        d_V_LIF,
        d_fired,

        curr_t, H, W,
        alpha1,
        alpha2,
        V_BIP_LPF_alpha,
        lambda_OPL,
        G_A_zero,
        lambda_A,
        p1,
        I_bias,
        p2,
        p3,
        U_rest,
        U_th,
        U_reset);
}