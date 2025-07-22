#ifndef SILICON_RETINA_KERNEL_H
#define SILICON_RETINA_KERNEL_H
#include <cuda_runtime.h>
#include "enums.h"


__global__ void run_silicon_retina_model(
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
    unsigned char  *d_fired,

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
    float U_reset);

__global__ void run_silicon_retina_model_fired_only(
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
    unsigned char  *d_fired,

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
    float U_reset);

    
#endif // SILICON_RETINA_KERNEL_H