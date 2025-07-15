#include <cuda_runtime.h>
#include <iostream>
#include <vector>

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
    char  *fired,

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

        LPF1[curr_t_idx] = LPF1[prev_t_idx] + alpha1 * (imgs[curr_t_idx] - LPF1[prev_t_idx]);
        HPF1[curr_t_idx] = LPF1[curr_t_idx] - LPF1[prev_t_idx];
        LPF2[curr_t_idx] = LPF2[prev_t_idx] + alpha2 * (HPF1[curr_t_idx] - LPF2[prev_t_idx]);
        BLURRED2[curr_t_idx] = 0.0f;
        float cnt = 0;

        for (int i = -1; i <= 1; ++i)
        {
            for (int j = -1; j <= 1; ++j)
            {
                int neighbor_idx = center_idx + i * W + j;

                if (neighbor_idx >= 0 && neighbor_idx < H * W)
                {
                    BLURRED2[curr_t_idx] += LPF2[curr_t_offset + neighbor_idx];
                    cnt += 1;
                }
            }
        }
        if (cnt > 0)
            BLURRED2[curr_t_idx] /= cnt;

        I_OPL[curr_t_idx] = lambda_OPL * (BLURRED2[curr_t_idx] - LPF2[curr_t_idx]);

        // compute the Q function
        Q_V_BIP[curr_t_idx] = G_A_zero + lambda_A * V_BIP[prev_t_idx] * V_BIP[prev_t_idx];

        if (Q_V_BIP[curr_t_idx] >= 1.0f)
        {
            Q_V_BIP[curr_t_idx] = 1.0f;
        }
        else if (Q_V_BIP[curr_t_idx] < 0.0f)
        {
            Q_V_BIP[curr_t_idx] = 0.0f;
        }

        // compute g_A
        V_BIP_LPF[curr_t_idx] = V_BIP[prev_t_idx] + V_BIP_LPF_alpha * (Q_V_BIP[prev_t_idx] - V_BIP_LPF[prev_t_idx]);

        float g_A = 0.0f;
        cnt = 0;
        for (int i = -1; i <= 1; ++i)
        {
            for (int j = -1; j <= 1; ++j)
            {
                int neighbor_idx = center_idx + i * W + j;
                if (neighbor_idx >= 0 && neighbor_idx < H * W)
                {
                    g_A += V_BIP_LPF[curr_t_offset + neighbor_idx];
                    cnt++;
                }
            }
        }

        if (cnt > 0)
            g_A /= cnt;

        // compute the V_bip using the euler method
        V_BIP[curr_t_idx] = V_BIP[prev_t_idx] + p1 * (I_OPL[curr_t_idx] - g_A * V_BIP[prev_t_idx]);

        // compute the HPF of the V_BIP
        HPF2[curr_t_idx] = V_BIP[curr_t_idx] - V_BIP[prev_t_idx];

        // compute the I_gang
        I_Gang[curr_t_idx] = max(HPF2[curr_t_idx], 0.0f) + I_bias;

        // compute the membranes

        V_LIF[curr_t_idx] = V_LIF[prev_t_idx] + p2 * I_Gang[curr_t_idx] - abs(V_LIF[prev_t_idx] - U_rest) * p3;

        if (V_LIF[curr_t_idx] >= U_th)
        {
            V_LIF[curr_t_idx] -= U_reset;
            fired[curr_t_idx] = 1; // mark as fired
        }
        else if (V_LIF[curr_t_idx] < U_rest)
        {
            V_LIF[curr_t_idx] = U_rest;
            fired[curr_t_idx] = 0; // mark as not fired
        }
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
    char  *d_fired,

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
    unsigned threads = 256;

    dim3 blocks((T * H * W + threads - 1) / threads);

    sillicon_retina_kernel<<<blocks, threads>>>(
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

/*
int main() {
    // 1. Allocate host buffers
    int T = 10, H = 64, W = 64;
    std::vector<float> imgs(T * H * W, 1.0f);  // Dummy data
    std::vector<float> LPF1(T * H * W, 0.0f);
    std::vector<float> HPF1(T * H * W, 0.0f);
    std::vector<float> BLURRED2(T * H * W, 0.0f);
    std::vector<float> LPF2(T * H * W, 0.0f);
    std::vector<float> I_OPL(T * H * W, 0.0f);

    // 2. Allocate device buffers
    float* d_imgs;
    float* d_LPF1;
    float* d_HPF1;
    float* d_BLURRED2;
    float* d_LPF2;
    float* d_I_OPL;

    float alpha1 = 0.5f;
    float alpha2 = 0.5f;

    cudaMalloc(&d_imgs, imgs.size() * sizeof(float));
    cudaMalloc(&d_LPF1, LPF1.size() * sizeof(float));
    cudaMalloc(&d_HPF1, HPF1.size() * sizeof(float));
    cudaMalloc(&d_BLURRED2, BLURRED2.size() * sizeof(float));
    cudaMalloc(&d_LPF2, LPF2.size() * sizeof(float));
    cudaMalloc(&d_I_OPL, I_OPL.size() * sizeof(float));


    // 3. Copy data to device
    cudaMemcpy(d_imgs, imgs.data(), imgs.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_LPF1, LPF1.data(), LPF1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_HPF1, HPF1.data(), HPF1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BLURRED2, BLURRED2.data(), BLURRED2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_LPF2, LPF2.data(), LPF2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_I_OPL, I_OPL.data(), I_OPL.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Launch kernel
    unsigned threads = 256;
    dim3 blocks((H * W + threads - 1) / threads, 1);
    run_silicon_retina_model<<<blocks, threads>>>(
        d_imgs,
        d_LPF1,
        d_HPF1,
        d_BLURRED2,
        d_LPF2,
        d_I_OPL,
        T,
        H,
        W,
        alpha1,
        alpha2
    );

    cudaDeviceSynchronize();

    // 5. Copy results back
    cudaMemcpy(LPF1.data(), d_LPF1, LPF1.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(HPF1.data(), d_HPF1, HPF1.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(BLURRED2.data(), d_BLURRED2, BLURRED2.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(LPF2.data(), d_LPF2, LPF2.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(I_OPL.data(), d_I_OPL, I_OPL.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Print a few results

    // 7. Free
    cudaFree(d_imgs);
    cudaFree(d_LPF1);
    cudaFree(d_HPF1);
    cudaFree(d_BLURRED2);
    cudaFree(d_LPF2);
    cudaFree(d_I_OPL);

    return 0;
}
*/