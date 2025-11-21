#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <vector>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "retina_params.h"

#define CHECK_KERNELS(x) TORCH_CHECK((x), #x " kernels were not initialized !")
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)       \
    do                       \
    {                        \
        CHECK_CUDA(x);       \
        CHECK_CONTIGUOUS(x); \
    } while (0)
#define CHECK_DEVICE(x, y) TORCH_CHECK(x.device() == y.device(), #x " and " #y " must be on the same device")

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

// Max supported kernel size
#define MAX_KERNEL_SIZE 15
#define MAX_KERNEL_ELEMENTS (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE)

// Constant memory for kernel weights - fast broadcast reads
__constant__ float d_kernel1[MAX_KERNEL_ELEMENTS];
__constant__ float d_kernel2[MAX_KERNEL_ELEMENTS];

// device pointers for texture objects
cudaTextureObject_t tex_LPF2 = 0;
cudaTextureObject_t tex_V_BIP_LPF = 0;

static bool kernels_initialized = false;

__device__ __forceinline__ float fclampf(float x, float min, float max)
{
    return fmaxf(min, fminf(max, x));
}

// Generate Gaussian kernel on device
__device__ void generate_gaussian_kernel_dev(float *kernel, int size, float sigma)
{
    int half = size / 2;
    float sum = 0.0f;

    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            int dx = x - half;
            int dy = y - half;
            float value = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            kernel[y * size + x] = value;
            sum += value;
        }
    }

    // Normalize
    for (int i = 0; i < size * size; ++i)
    {
        kernel[i] /= sum;
    }
}

// 2D convolution with constant memory kernel
__device__ float apply_convolution(const float *__restrict__ input,
                                   const float *kernel,
                                   int kernel_size,
                                   int W, int H,
                                   int center_idx,
                                   int curr_t_offset)
{
    const int y = center_idx / W;
    const int x = center_idx % W;
    const int ks = kernel_size / 2;

    float output = 0.0f;

    if (kernel_size == 3)
    {
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy)
        {
            const int ny = y + dy;
            if (ny >= 0 && ny < H)
            {
#pragma unroll
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const int nx = x + dx;
                    if (nx >= 0 && nx < W)
                    {
                        const int kid = (dy + 1) * 3 + (dx + 1);
                        const int neighbor_idx = ny * W + nx;
                        output += input[curr_t_offset + neighbor_idx] * kernel[kid];
                    }
                }
            }
        }
    }
    else
    {
        for (int dy = -ks; dy <= ks; ++dy)
        {
            const int ny = y + dy;
            if (ny >= 0 && ny < H)
            {
                for (int dx = -ks; dx <= ks; ++dx)
                {
                    const int nx = x + dx;
                    if (nx >= 0 && nx < W)
                    {
                        const int kid = (dy + ks) * kernel_size + (dx + ks);
                        const int neighbor_idx = ny * W + nx;
                        output += input[curr_t_offset + neighbor_idx] * kernel[kid];
                    }
                }
            }
        }
    }

    return fclampf(output, 0.0f, 1.0f);
}

// Texture-based convolution (more efficient for larger images)
__device__ float apply_convolution_texture(cudaTextureObject_t tex,
                                           const float *kernel,
                                           int kernel_size,
                                           int W, int H,
                                           int center_idx,
                                           int slice_offset)
{
    const int y = center_idx / W;
    const int x = center_idx % W;
    const int ks = kernel_size / 2;

    float output = 0.0f;

    if (kernel_size == 3)
    {
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy)
        {
#pragma unroll
            for (int dx = -1; dx <= 1; ++dx)
            {
                const int kid = (dy + 1) * 3 + (dx + 1);
                // Texture coordinates (normalized to [0, 1])
                float tx = (x + dx + 0.5f) / W;
                float ty = (y + dy + 0.5f) / H;
                output += tex2D<float>(tex, tx, ty) * kernel[kid];
            }
        }
    }
    else
    {
        for (int dy = -ks; dy <= ks; ++dy)
        {
            for (int dx = -ks; dx <= ks; ++dx)
            {
                const int kid = (dy + ks) * kernel_size + (dx + ks);
                float tx = (x + dx + 0.5f) / W;
                float ty = (y + dy + 0.5f) / H;
                output += tex2D<float>(tex, tx, ty) * kernel[kid];
            }
        }
    }

    return fclampf(output, 0.0f, 1.0f);
}

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

    // TODO : saturating ? don't think it's necessary
    output = fclampf(output, 0.0f, 1.0f);
    return (cnt > 0.0f) ? output / cnt : 0.0f;
}

// Unified blur interface
__device__ float apply_blur(BLUR_MODE mode,
                            KERNEL_TYPE kernel_type,
                            const float *kernel,
                            int kernel_size,
                            int W, int H,
                            const float *input,
                            int center_idx,
                            int curr_t_offset,
                            cudaTextureObject_t tex = 0)
{
    if (mode == BLUR_MODE::BLUR_NONE)
    {
        return input[curr_t_offset + center_idx];
    }

    if (mode == BLUR_MODE::BLUR_AVG && kernel_type == KERNEL_TYPE::BOX)
    {
        return apply_blur_avg(kernel_size, W, H, input, center_idx, curr_t_offset);
    }

    // Use texture if available and beneficial (larger images)
    if (tex != 0 && (W * H) > 65536)
    {
        return apply_convolution_texture(tex, kernel, kernel_size, W, H,
                                         center_idx, curr_t_offset);
    }

    return apply_convolution(input, kernel, kernel_size, W, H,
                             center_idx, curr_t_offset);
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

__forceinline__ __device__ float apply_high_pass_filter(float incoming_value,
                                                        float previous_value,
                                                        int curr,
                                                        int prev,
                                                        float *HPF,
                                                        float *HPF_mem,
                                                        HPF_MODE HPF_mode,
                                                        float HPF_param)
{
    if (HPF_mode == HPF_MODE::MODE1)
    {
        HPF_mem[curr] = HPF_mem[prev] + HPF_param * (incoming_value - HPF_mem[prev]);
        return incoming_value - HPF_mem[curr];
    }
    else if (HPF_mode == HPF_MODE::MODE2)
    {
        return incoming_value - previous_value + (1 - HPF_param) * HPF[prev];
    }
    return incoming_value - previous_value;
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
    if (V_LIF >= U_th)
    {
        V_LIF -= U_reset;
        return 1; // mark as fired
    }
    return 0; // mark as not fired
}

/********************************************************************
 * kernel A
 *******************************************************************/
__global__ void k_stageA(const float *__restrict__ imgs,
                         float *__restrict__ LPF1,
                         float *__restrict__ HPF1_mem,
                         float *__restrict__ HPF1,
                         float *__restrict__ LPF2,
                         int lin_t_offset, int prev_offset, int curr_offset,
                         int H, int W,
                         float alpha1, float alpha2,
                         HPF_MODE HPF1_mode,
                         float HPF1_param)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = H * W;
    if (idx >= HW)
        return;

    int lin_idx = lin_t_offset + idx;
    int prev = prev_offset + idx;
    int curr = curr_offset + idx;

    float lpf1 = apply_low_pass_filter(LPF_MODE::LPF_1,
                                       imgs[lin_idx],
                                       LPF1[prev],
                                       alpha1);
    LPF1[curr] = lpf1;

    float hpf1 = apply_high_pass_filter(lpf1, LPF1[prev],
                                        curr, prev,
                                        HPF1, HPF1_mem,
                                        HPF1_mode, HPF1_param);

    HPF1[curr] = hpf1;

    LPF2[curr] = apply_low_pass_filter(LPF_MODE::LPF_1,
                                       hpf1,
                                       LPF2[prev],
                                       alpha2);
}

/********************************************************************
 * kernel B – first spatial blur (LPF2 → BLURRED2) + Q & V_BIP_LPF
 *******************************************************************/
__global__ void k_stageB(const float *__restrict__ LPF2,
                         float *__restrict__ BLURRED2,
                         float *__restrict__ I_OPL,
                         float *__restrict__ Q,
                         float *__restrict__ V_BIP_LPF,
                         const float *__restrict__ V_BIP,
                         int lin_t_offset, int prev_offset, int curr_offset,
                         int H, int W,
                         float lambda_OPL,
                         float G_A_zero, float lambda_A,
                         float V_BIP_LPF_alpha,
                         BLUR_MODE blur1_mode,
                         KERNEL_TYPE kernel1_type,
                         int kernel1_size,
                         bool turn_off_feedback,
                         cudaTextureObject_t tex = 0)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = H * W;
    if (idx >= HW)
        return;

    int prev = prev_offset + idx;
    int curr = curr_offset + idx;

    BLURRED2[curr] = apply_blur(blur1_mode, kernel1_type, d_kernel1,
                                kernel1_size, W, H,
                                LPF2, idx, curr - idx, tex);

    I_OPL[curr] = lambda_OPL * (BLURRED2[curr] - LPF2[curr]);

    if (turn_off_feedback)
    {
        Q[curr] = 0.0f;
        V_BIP_LPF[curr] = 0.0f;
        return;
    }

    Q[curr] = fmaxf(0.0f, fminf(G_A_zero + lambda_A * fabsf(V_BIP[prev]), 1.0f));

    V_BIP_LPF[curr] = apply_low_pass_filter(LPF_MODE::LPF_1,
                                            Q[curr],
                                            V_BIP_LPF[prev],
                                            V_BIP_LPF_alpha);
}

/********************************************************************
 * kernel C – second spatial blur with configurable kernel
 *******************************************************************/
__global__ void k_stageC(const float *__restrict__ V_BIP_LPF,
                         float *__restrict__ V_BIP,
                         float *__restrict__ HPF2_mem,
                         float *__restrict__ HPF2,
                         float *__restrict__ I_Gang_on,
                         float *__restrict__ I_Gang_off,
                         float *__restrict__ V_LIF_on,
                         float *__restrict__ V_LIF_off,
                         unsigned char *__restrict__ fired_on,
                         unsigned char *__restrict__ fired_off,
                         const float *__restrict__ I_OPL,
                         int lin_t_offset,
                         int prev_offset,
                         int curr_offset,
                         int H, int W,
                         float p1,
                         float I_bias_on,
                         float I_bias_off,
                         float p2, float p3,
                         float U_rest_on, float U_rest_off,
                         float U_th_on, float U_th_off,
                         float U_reset_on, float U_reset_off,
                         HPF_MODE HPF2_mode,
                         float HPF2_param,
                         BLUR_MODE blur2_mode,
                         KERNEL_TYPE kernel2_type,
                         int kernel2_size,
                         bool turn_off_feedback,
                         cudaTextureObject_t tex = 0)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = H * W;
    if (idx >= HW)
        return;

    int prev = prev_offset + idx;
    int curr = curr_offset + idx;
    int lin_idx = lin_t_offset + idx;

    if (!turn_off_feedback)
    {
        float g_A = apply_blur(blur2_mode, kernel2_type, d_kernel2,
                               kernel2_size, W, H,
                               V_BIP_LPF, idx, curr - idx, tex);

        V_BIP[curr] = euler_method(V_BIP[prev], p1, I_OPL[curr], g_A);

        HPF2[curr] = apply_high_pass_filter(V_BIP[curr], V_BIP[prev],
                                            curr, prev, HPF2, HPF2_mem,
                                            HPF2_mode, HPF2_param);
    }
    else
    {
        V_BIP[curr] = I_OPL[curr];
        HPF2[curr] = I_OPL[curr];
    }

    I_Gang_on[curr] = fmaxf(HPF2[curr], 0.0f) + I_bias_on;
    I_Gang_off[curr] = fmaxf(-HPF2[curr], 0.0f) + I_bias_off;

    V_LIF_on[curr] = compute_next_LIF(LIF_INTEGRATION_MODE::LIF_INTEGRATION_1,
                                      V_LIF_on[prev], p2, p3,
                                      I_Gang_on[curr], U_rest_on);

    V_LIF_off[curr] = compute_next_LIF(LIF_INTEGRATION_MODE::LIF_INTEGRATION_1,
                                       V_LIF_off[prev], p2, p3,
                                       I_Gang_off[curr], U_rest_off);

    fired_on[lin_idx] = fire_and_reset(LIF_RESET_MODE::LIF_SUBTRACT_RESET,
                                       V_LIF_on[curr], U_th_on, U_reset_on);

    fired_off[lin_idx] = fire_and_reset(LIF_RESET_MODE::LIF_SUBTRACT_RESET,
                                        V_LIF_off[curr], U_th_off, U_reset_off);
}

// Helper to initialize kernels on device
void initialize_kernels(SiliconRetinaParams &config)
{
    kernels_initialized = true;
    // Initialize kernel 1
    std::vector<float> kernel1_host;

    switch (config.kernel1_type)
    {
    case KERNEL_TYPE::BOX:
    {
        int size = config.kernel1_size * config.kernel1_size;
        kernel1_host.resize(size, 1.0f / size);
        break;
    }
    case KERNEL_TYPE::GAUSSIAN:
        kernel1_host = generate_gaussian_kernel(config.kernel1_size, config.kernel1_sigma);
        break;
    case KERNEL_TYPE::CUSTOM:
        kernel1_host = config.custom_kernel1;
        break;
    default:
        kernel1_host.resize(9, 1.0f / 9.0f); // Default 3x3 box
    }

    // Initialize kernel 2
    std::vector<float> kernel2_host;

    switch (config.kernel2_type)
    {
    case KERNEL_TYPE::BOX:
    {
        int size = config.kernel2_size * config.kernel2_size;
        kernel2_host.resize(size, 1.0f / size);
        break;
    }
    case KERNEL_TYPE::GAUSSIAN:
        kernel2_host = generate_gaussian_kernel(config.kernel2_size, config.kernel2_sigma);
        break;
    case KERNEL_TYPE::CUSTOM:
        kernel2_host = config.custom_kernel2;
        break;
    default:
        kernel2_host.resize(9, 1.0f / 9.0f);
    }

    // Copy to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel1, kernel1_host.data(),
                                  kernel1_host.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel2, kernel2_host.data(),
                                  kernel2_host.size() * sizeof(float)));
}

void run_silicon_retina_model(
    const float *d_imgs,
    float *d_LPF1,
    float *d_HPF1_mem,
    float *d_HPF1,
    float *d_BLURRED2,
    float *d_LPF2,
    float *d_I_OPL,
    float *d_V_BIP,
    float *d_Q_V_BIP,
    float *d_V_BIP_LPF,
    float *d_HPF2_mem,
    float *d_HPF2,
    float *d_I_Gang_on,
    float *d_I_Gang_off,
    float *d_V_LIF_on,
    float *d_V_LIF_off,
    unsigned char *d_fired_on,
    unsigned char *d_fired_off,

    int T,
    SiliconRetinaParams &retina_config)
{
    const int HW = retina_config.H * retina_config.W;

    CHECK_KERNELS(kernels_initialized);

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int block, minGrid;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &block, k_stageA));
    block = (block / 32) * 32;
    if (block == 0)
        block = 256;
    int grid = (HW + block - 1) / block;

    int prev_offset = 0;
    int curr_offset = HW;
    int lin_t_offset = HW;

    for (int t = 1; t < T; ++t)
    {

        prev_offset = (t - 1) * HW;
        curr_offset = t * HW;
        lin_t_offset = t * HW;

        /* ---------- kernel A ---------- */
        k_stageA<<<grid, block>>>(d_imgs,
                                  d_LPF1, d_HPF1_mem, d_HPF1, d_LPF2,
                                  lin_t_offset, prev_offset, curr_offset,
                                  retina_config.H, retina_config.W,
                                  retina_config.alpha1, retina_config.alpha2,
                                  retina_config.HPF1_mode,
                                  retina_config.HPF1_param);

        /* ---------- kernel B ---------- */
        k_stageB<<<grid, block>>>(d_LPF2, d_BLURRED2, d_I_OPL,
                                  d_Q_V_BIP, d_V_BIP_LPF, d_V_BIP,
                                  lin_t_offset, prev_offset, curr_offset,
                                  retina_config.H, retina_config.W,
                                  retina_config.lambda_OPL, retina_config.G_A_zero,
                                  retina_config.lambda_A, retina_config.V_BIP_LPF_alpha,
                                  retina_config.blur1_mode,
                                  retina_config.kernel1_type,
                                  retina_config.kernel1_size,
                                  retina_config.turn_off_feedback);

        /* ---------- kernel C ---------- */
        k_stageC<<<grid, block>>>(d_V_BIP_LPF, d_V_BIP, d_HPF2_mem, d_HPF2,
                                  d_I_Gang_on, d_I_Gang_off,
                                  d_V_LIF_on, d_V_LIF_off,
                                  d_fired_on, d_fired_off,
                                  d_I_OPL,
                                  lin_t_offset, prev_offset, curr_offset,
                                  retina_config.H, retina_config.W,
                                  retina_config.p1,
                                  retina_config.I_bias_on, retina_config.I_bias_off,
                                  retina_config.p2, retina_config.p3,
                                  retina_config.U_rest_on, retina_config.U_rest_off,
                                  retina_config.U_th_on, retina_config.U_th_off,
                                  retina_config.U_reset_on, retina_config.U_reset_off,
                                  retina_config.HPF2_mode, retina_config.HPF2_param,
                                  retina_config.blur2_mode,
                                  retina_config.kernel2_type,
                                  retina_config.kernel2_size,
                                  retina_config.turn_off_feedback);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<torch::Tensor> forward_silicon_retina_verbose(
    const torch::Tensor &d_images,
    torch::Tensor &d_LPF1,
    torch::Tensor &d_HPF1_mem,
    torch::Tensor &d_HPF1,
    torch::Tensor &d_BLURRED2,
    torch::Tensor &d_LPF2,
    torch::Tensor &d_I_OPL,
    torch::Tensor &d_V_BIP,
    torch::Tensor &d_Q_V_BIP,
    torch::Tensor &d_V_BIP_LPF,
    torch::Tensor &d_HPF2_mem,
    torch::Tensor &d_HPF2,

    torch::Tensor &d_I_Gang_on,
    torch::Tensor &d_I_Gang_off,

    torch::Tensor &d_V_LIF_on,
    torch::Tensor &d_V_LIF_off,

    torch::Tensor &d_fired_on,
    torch::Tensor &d_fired_off,

    int T,
    SiliconRetinaParams &retina_config)
{
    CHECK_INPUT(d_images);
    CHECK_INPUT(d_LPF1);
    CHECK_INPUT(d_HPF1);
    CHECK_INPUT(d_HPF1_mem);
    CHECK_INPUT(d_BLURRED2);
    CHECK_INPUT(d_LPF2);
    CHECK_INPUT(d_I_OPL);
    CHECK_INPUT(d_V_BIP);
    CHECK_INPUT(d_Q_V_BIP);
    CHECK_INPUT(d_V_BIP_LPF);
    CHECK_INPUT(d_HPF2);
    CHECK_INPUT(d_HPF2_mem);
    CHECK_INPUT(d_I_Gang_on);
    CHECK_INPUT(d_I_Gang_off);

    CHECK_INPUT(d_V_LIF_on);
    CHECK_INPUT(d_V_LIF_off);

    CHECK_INPUT(d_fired_on);
    CHECK_INPUT(d_fired_off);

    CHECK_DEVICE(d_images, d_LPF1);
    CHECK_DEVICE(d_images, d_HPF1_mem);
    CHECK_DEVICE(d_images, d_HPF1);
    CHECK_DEVICE(d_images, d_BLURRED2);
    CHECK_DEVICE(d_images, d_LPF2);
    CHECK_DEVICE(d_images, d_I_OPL);
    CHECK_DEVICE(d_images, d_V_BIP);
    CHECK_DEVICE(d_images, d_Q_V_BIP);
    CHECK_DEVICE(d_images, d_V_BIP_LPF);
    CHECK_DEVICE(d_images, d_HPF2_mem);
    CHECK_DEVICE(d_images, d_HPF2);
    CHECK_DEVICE(d_images, d_I_Gang_on);
    CHECK_DEVICE(d_images, d_I_Gang_off);

    CHECK_DEVICE(d_images, d_V_LIF_on);
    CHECK_DEVICE(d_images, d_V_LIF_off);

    CHECK_DEVICE(d_images, d_fired_on);
    CHECK_DEVICE(d_images, d_fired_off);

    run_silicon_retina_model(d_images.data<float>(),
                             d_LPF1.data<float>(),
                             d_HPF1_mem.data<float>(),
                             d_HPF1.data<float>(),
                             d_BLURRED2.data<float>(),
                             d_LPF2.data<float>(),
                             d_I_OPL.data<float>(),
                             d_V_BIP.data<float>(),
                             d_Q_V_BIP.data<float>(),
                             d_V_BIP_LPF.data<float>(),
                             d_HPF2_mem.data<float>(),
                             d_HPF2.data<float>(),
                             d_I_Gang_on.data<float>(),
                             d_I_Gang_off.data<float>(),
                             d_V_LIF_on.data<float>(),
                             d_V_LIF_off.data<float>(),
                             d_fired_on.data<unsigned char>(),
                             d_fired_off.data<unsigned char>(),
                             T,
                             retina_config);
    return {d_fired_on, d_fired_off};
}

void run_silicon_retina_model_fired_only(
    const float *d_imgs,
    float *d_LPF1,
    float *d_HPF1_mem,
    float *d_HPF1,
    float *d_BLURRED2,
    float *d_LPF2,
    float *d_I_OPL,
    float *d_V_BIP,
    float *d_Q_V_BIP,
    float *d_V_BIP_LPF,
    float *d_HPF2_mem,
    float *d_HPF2,
    float *d_I_Gang_on,
    float *d_I_Gang_off,
    float *d_V_LIF_on,
    float *d_V_LIF_off,
    unsigned char *d_fired_on,
    unsigned char *d_fired_off,

    int T,
    SiliconRetinaParams &retina_config)
{

    CHECK_KERNELS(kernels_initialized);

    const int HW = retina_config.H * retina_config.W;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int block, minGrid;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &block, k_stageA));
    block = (block / 32) * 32;
    if (block == 0)
        block = 256;
    int grid = (HW + block - 1) / block;

    int prev_offset = 0;
    int curr_offset = HW;
    int lin_t_offset = HW;

    for (int t = 1; t < T; ++t)
    {

        const int current_double_buffer_idx = t % 2;
        const int previous_double_buffer_idx = (t - 1) % 2;

        curr_offset = current_double_buffer_idx * HW;
        prev_offset = previous_double_buffer_idx * HW;
        lin_t_offset = t * HW;

        /* ---------- kernel A ---------- */
        k_stageA<<<grid, block>>>(d_imgs,
                                  d_LPF1, d_HPF1_mem, d_HPF1, d_LPF2,
                                  lin_t_offset, prev_offset, curr_offset,
                                  retina_config.H, retina_config.W,
                                  retina_config.alpha1, retina_config.alpha2,
                                  retina_config.HPF1_mode,
                                  retina_config.HPF1_param);

        /* ---------- kernel B ---------- */
        k_stageB<<<grid, block>>>(d_LPF2, d_BLURRED2, d_I_OPL,
                                  d_Q_V_BIP, d_V_BIP_LPF, d_V_BIP,
                                  lin_t_offset, prev_offset, curr_offset,
                                  retina_config.H, retina_config.W,
                                  retina_config.lambda_OPL, retina_config.G_A_zero,
                                  retina_config.lambda_A, retina_config.V_BIP_LPF_alpha,
                                  retina_config.blur1_mode,
                                  retina_config.kernel1_type,
                                  retina_config.kernel1_size,
                                  retina_config.turn_off_feedback);

        /* ---------- kernel C ---------- */
        k_stageC<<<grid, block>>>(d_V_BIP_LPF, d_V_BIP, d_HPF2_mem, d_HPF2,
                                  d_I_Gang_on, d_I_Gang_off,
                                  d_V_LIF_on, d_V_LIF_off,
                                  d_fired_on, d_fired_off,
                                  d_I_OPL,
                                  lin_t_offset, prev_offset, curr_offset,
                                  retina_config.H, retina_config.W,
                                  retina_config.p1,
                                  retina_config.I_bias_on, retina_config.I_bias_off,
                                  retina_config.p2, retina_config.p3,
                                  retina_config.U_rest_on, retina_config.U_rest_off,
                                  retina_config.U_th_on, retina_config.U_th_off,
                                  retina_config.U_reset_on, retina_config.U_reset_off,
                                  retina_config.HPF2_mode, retina_config.HPF2_param,
                                  retina_config.blur2_mode,
                                  retina_config.kernel2_type,
                                  retina_config.kernel2_size,
                                  retina_config.turn_off_feedback);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<torch::Tensor> forward_silicon_retina(
    const torch::Tensor &d_images,
    torch::Tensor &d_LPF1,
    torch::Tensor &d_HPF1_mem,
    torch::Tensor &d_HPF1,
    torch::Tensor &d_BLURRED2,
    torch::Tensor &d_LPF2,
    torch::Tensor &d_I_OPL,
    torch::Tensor &d_V_BIP,
    torch::Tensor &d_Q_V_BIP,
    torch::Tensor &d_V_BIP_LPF,
    torch::Tensor &d_HPF2_mem,
    torch::Tensor &d_HPF2,

    torch::Tensor &d_I_Gang_on,
    torch::Tensor &d_I_Gang_off,

    torch::Tensor &d_V_LIF_on,
    torch::Tensor &d_V_LIF_off,

    torch::Tensor &d_fired_on,
    torch::Tensor &d_fired_off,

    int T,
    SiliconRetinaParams &retina_config)
{
    CHECK_INPUT(d_images);
    CHECK_INPUT(d_LPF1);
    CHECK_INPUT(d_HPF1);
    CHECK_INPUT(d_HPF1_mem);
    CHECK_INPUT(d_BLURRED2);
    CHECK_INPUT(d_LPF2);
    CHECK_INPUT(d_I_OPL);
    CHECK_INPUT(d_V_BIP);
    CHECK_INPUT(d_Q_V_BIP);
    CHECK_INPUT(d_V_BIP_LPF);
    CHECK_INPUT(d_HPF2);
    CHECK_INPUT(d_HPF2_mem);
    CHECK_INPUT(d_I_Gang_on);
    CHECK_INPUT(d_I_Gang_off);

    CHECK_INPUT(d_V_LIF_on);
    CHECK_INPUT(d_V_LIF_off);

    CHECK_INPUT(d_fired_on);
    CHECK_INPUT(d_fired_off);

    CHECK_DEVICE(d_images, d_LPF1);
    CHECK_DEVICE(d_images, d_HPF1_mem);
    CHECK_DEVICE(d_images, d_HPF1);
    CHECK_DEVICE(d_images, d_BLURRED2);
    CHECK_DEVICE(d_images, d_LPF2);
    CHECK_DEVICE(d_images, d_I_OPL);
    CHECK_DEVICE(d_images, d_V_BIP);
    CHECK_DEVICE(d_images, d_Q_V_BIP);
    CHECK_DEVICE(d_images, d_V_BIP_LPF);
    CHECK_DEVICE(d_images, d_HPF2_mem);
    CHECK_DEVICE(d_images, d_HPF2);
    CHECK_DEVICE(d_images, d_I_Gang_on);
    CHECK_DEVICE(d_images, d_I_Gang_off);

    CHECK_DEVICE(d_images, d_V_LIF_on);
    CHECK_DEVICE(d_images, d_V_LIF_off);

    CHECK_DEVICE(d_images, d_fired_on);
    CHECK_DEVICE(d_images, d_fired_off);

    run_silicon_retina_model_fired_only(d_images.data<float>(),
                                        d_LPF1.data<float>(),
                                        d_HPF1_mem.data<float>(),
                                        d_HPF1.data<float>(),
                                        d_BLURRED2.data<float>(),
                                        d_LPF2.data<float>(),
                                        d_I_OPL.data<float>(),
                                        d_V_BIP.data<float>(),
                                        d_Q_V_BIP.data<float>(),
                                        d_V_BIP_LPF.data<float>(),
                                        d_HPF2_mem.data<float>(),
                                        d_HPF2.data<float>(),
                                        d_I_Gang_on.data<float>(),
                                        d_I_Gang_off.data<float>(),
                                        d_V_LIF_on.data<float>(),
                                        d_V_LIF_off.data<float>(),
                                        d_fired_on.data<unsigned char>(),
                                        d_fired_off.data<unsigned char>(),

                                        T, retina_config);

    return {d_fired_on, d_fired_off};
}

void run_silicon_retina_model_online(
    const float *d_imgs,
    float *d_LPF1,
    float *d_HPF1_mem,
    float *d_HPF1,
    float *d_BLURRED2,
    float *d_LPF2,
    float *d_I_OPL,
    float *d_V_BIP,
    float *d_Q_V_BIP,
    float *d_V_BIP_LPF,
    float *d_HPF2_mem,
    float *d_HPF2,
    float *d_I_Gang_on,
    float *d_I_Gang_off,
    float *d_V_LIF_on,
    float *d_V_LIF_off,
    unsigned char *d_fired_on,
    unsigned char *d_fired_off,

    int curr_t,
    SiliconRetinaParams &retina_config)
{

    CHECK_KERNELS(kernels_initialized);

    const int HW = retina_config.H * retina_config.W;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int block, minGrid;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrid, &block, k_stageA));
    block = (block / 32) * 32;
    if (block == 0)
        block = 256;
    int grid = (HW + block - 1) / block;

    int prev_offset = 0;
    int curr_offset = HW;
    int lin_t_offset = HW;

    const int current_double_buffer_idx = curr_t % 2;
    const int previous_double_buffer_idx = (curr_t - 1) % 2;

    curr_offset = current_double_buffer_idx * HW;
    prev_offset = previous_double_buffer_idx * HW;
    lin_t_offset = 0;

    /* ---------- kernel A ---------- */
    k_stageA<<<grid, block>>>(d_imgs,
                              d_LPF1,
                              d_HPF1_mem,
                              d_HPF1, d_LPF2,
                              lin_t_offset, prev_offset, curr_offset,
                              retina_config.H, retina_config.W,
                              retina_config.alpha1, retina_config.alpha2,
                              retina_config.HPF1_mode, retina_config.HPF1_param);

    /* ---------- kernel B ---------- */
    k_stageB<<<grid, block>>>(d_LPF2, d_BLURRED2, d_I_OPL,
                              d_Q_V_BIP, d_V_BIP_LPF, d_V_BIP,
                              lin_t_offset, prev_offset, curr_offset,
                              retina_config.H, retina_config.W,
                              retina_config.lambda_OPL, retina_config.G_A_zero,
                              retina_config.lambda_A, retina_config.V_BIP_LPF_alpha,
                              retina_config.blur1_mode,
                              retina_config.kernel1_type,
                              retina_config.kernel1_size,
                              retina_config.turn_off_feedback);

    /* ---------- kernel C ---------- */
    k_stageC<<<grid, block>>>(d_V_BIP_LPF, d_V_BIP, d_HPF2_mem, d_HPF2,
                              d_I_Gang_on, d_I_Gang_off,
                              d_V_LIF_on, d_V_LIF_off,
                              d_fired_on, d_fired_off,
                              d_I_OPL,
                              lin_t_offset, prev_offset, curr_offset,
                              retina_config.H, retina_config.W,
                              retina_config.p1,
                              retina_config.I_bias_on, retina_config.I_bias_off,
                              retina_config.p2, retina_config.p3,
                              retina_config.U_rest_on, retina_config.U_rest_off,
                              retina_config.U_th_on, retina_config.U_th_off,
                              retina_config.U_reset_on, retina_config.U_reset_off,
                              retina_config.HPF2_mode, retina_config.HPF2_param,
                              retina_config.blur2_mode,
                              retina_config.kernel2_type,
                              retina_config.kernel2_size,
                              retina_config.turn_off_feedback);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<torch::Tensor> forward_silicon_retina_online(
    const torch::Tensor &d_images,
    torch::Tensor &d_LPF1,
    torch::Tensor &d_HPF1_mem,
    torch::Tensor &d_HPF1,
    torch::Tensor &d_BLURRED2,
    torch::Tensor &d_LPF2,
    torch::Tensor &d_I_OPL,
    torch::Tensor &d_V_BIP,
    torch::Tensor &d_Q_V_BIP,
    torch::Tensor &d_V_BIP_LPF,
    torch::Tensor &d_HPF2_mem,
    torch::Tensor &d_HPF2,

    torch::Tensor &d_I_Gang_on,
    torch::Tensor &d_I_Gang_off,

    torch::Tensor &d_V_LIF_on,
    torch::Tensor &d_V_LIF_off,

    torch::Tensor &d_fired_on,
    torch::Tensor &d_fired_off,

    int curr_t,
    SiliconRetinaParams &retina_config)
{
    CHECK_INPUT(d_images);
    CHECK_INPUT(d_LPF1);
    CHECK_INPUT(d_HPF1);
    CHECK_INPUT(d_BLURRED2);
    CHECK_INPUT(d_LPF2);
    CHECK_INPUT(d_I_OPL);
    CHECK_INPUT(d_V_BIP);
    CHECK_INPUT(d_Q_V_BIP);
    CHECK_INPUT(d_V_BIP_LPF);
    CHECK_INPUT(d_HPF2);
    CHECK_INPUT(d_I_Gang_on);
    CHECK_INPUT(d_I_Gang_off);

    CHECK_INPUT(d_V_LIF_on);
    CHECK_INPUT(d_V_LIF_off);

    CHECK_INPUT(d_fired_on);
    CHECK_INPUT(d_fired_off);

    CHECK_DEVICE(d_images, d_LPF1);
    CHECK_DEVICE(d_images, d_HPF1);
    CHECK_DEVICE(d_images, d_BLURRED2);
    CHECK_DEVICE(d_images, d_LPF2);
    CHECK_DEVICE(d_images, d_I_OPL);
    CHECK_DEVICE(d_images, d_V_BIP);
    CHECK_DEVICE(d_images, d_Q_V_BIP);
    CHECK_DEVICE(d_images, d_V_BIP_LPF);
    CHECK_DEVICE(d_images, d_HPF2);
    CHECK_DEVICE(d_images, d_I_Gang_on);
    CHECK_DEVICE(d_images, d_I_Gang_off);

    CHECK_DEVICE(d_images, d_V_LIF_on);
    CHECK_DEVICE(d_images, d_V_LIF_off);

    CHECK_DEVICE(d_images, d_fired_on);
    CHECK_DEVICE(d_images, d_fired_off);

    run_silicon_retina_model_online(d_images.data<float>(),
                                    d_LPF1.data<float>(),
                                    d_HPF1.data<float>(),
                                    d_HPF1_mem.data<float>(),
                                    d_BLURRED2.data<float>(),
                                    d_LPF2.data<float>(),
                                    d_I_OPL.data<float>(),
                                    d_V_BIP.data<float>(),
                                    d_Q_V_BIP.data<float>(),
                                    d_V_BIP_LPF.data<float>(),
                                    d_HPF2_mem.data<float>(),
                                    d_HPF2.data<float>(),
                                    d_I_Gang_on.data<float>(),
                                    d_I_Gang_off.data<float>(),
                                    d_V_LIF_on.data<float>(),
                                    d_V_LIF_off.data<float>(),
                                    d_fired_on.data<unsigned char>(),
                                    d_fired_off.data<unsigned char>(),

                                    curr_t, retina_config);

    return {d_fired_on, d_fired_off};
}