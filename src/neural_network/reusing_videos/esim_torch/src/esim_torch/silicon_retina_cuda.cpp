
#include <torch/extension.h>
#include <vector>

torch::Tensor forward_silicon_retina_verbose(
    const torch::Tensor &d_images,
    torch::Tensor &d_LPF1,
    torch::Tensor &d_HPF1,
    torch::Tensor &d_BLURRED2,
    torch::Tensor &d_LPF2,
    torch::Tensor &d_I_OPL,
    torch::Tensor &d_V_BIP,
    torch::Tensor &d_Q_V_BIP,
    torch::Tensor &d_V_BIP_LPF,
    torch::Tensor &d_HPF2,
    torch::Tensor &d_I_Gang_on,
    torch::Tensor &d_I_Gang_off,
    torch::Tensor &d_V_LIF_on,
    torch::Tensor &d_V_LIF_off,
    torch::Tensor &d_fired_on,
    torch::Tensor &d_fired_off,

    int T, int H, int W,
    float alpha1,
    float alpha2,
    float V_BIP_LPF_alpha,
    float lambda_OPL,
    float G_A_zero,
    float lambda_A,
    float p1,
    float I_bias_on,
    float I_bias_off,
    float p2,
    float p3,
    float U_rest_on,
    float U_rest_off,

    float U_th_on,
    float U_th_off,

    float U_reset_on,
    float U_reset_off);

torch::Tensor forward_silicon_retina(
    const torch::Tensor &d_images,
    torch::Tensor &d_LPF1,
    torch::Tensor &d_HPF1,
    torch::Tensor &d_BLURRED2,
    torch::Tensor &d_LPF2,
    torch::Tensor &d_I_OPL,
    torch::Tensor &d_V_BIP,
    torch::Tensor &d_Q_V_BIP,
    torch::Tensor &d_V_BIP_LPF,
    torch::Tensor &d_HPF2,
    torch::Tensor &d_I_Gang_on,
    torch::Tensor &d_I_Gang_off,
    torch::Tensor &d_V_LIF_on,
    torch::Tensor &d_V_LIF_off,
    torch::Tensor &d_fired_on,
    torch::Tensor &d_fired_off,

    int T, int H, int W,
    float alpha1,
    float alpha2,
    float V_BIP_LPF_alpha,
    float lambda_OPL,
    float G_A_zero,
    float lambda_A,
    float p1,
    float I_bias_on,
    float I_bias_off,
    float p2,
    float p3,
    float U_rest_on,
    float U_rest_off,

    float U_th_on,
    float U_th_off,

    float U_reset_on,
    float U_reset_off);

torch::Tensor forward_silicon_retina_online(
    const torch::Tensor &d_images,
    torch::Tensor &d_LPF1,
    torch::Tensor &d_HPF1,
    torch::Tensor &d_BLURRED2,
    torch::Tensor &d_LPF2,
    torch::Tensor &d_I_OPL,
    torch::Tensor &d_V_BIP,
    torch::Tensor &d_Q_V_BIP,
    torch::Tensor &d_V_BIP_LPF,
    torch::Tensor &d_HPF2,
    torch::Tensor &d_I_Gang_on,
    torch::Tensor &d_I_Gang_off,
    torch::Tensor &d_V_LIF_on,
    torch::Tensor &d_V_LIF_off,
    torch::Tensor &d_fired_on,
    torch::Tensor &d_fired_off,

    int T, int H, int W,
    float alpha1,
    float alpha2,
    float V_BIP_LPF_alpha,
    float lambda_OPL,
    float G_A_zero,
    float lambda_A,
    float p1,
    float I_bias_on,
    float I_bias_off,
    float p2,
    float p3,
    float U_rest_on,
    float U_rest_off,

    float U_th_on,
    float U_th_off,

    float U_reset_on,
    float U_reset_off);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward_silicon_retina_verbose", &forward_silicon_retina_verbose, "run silicon retina model and save the outputs for every layer (most useful for debugging!)");
    m.def("forward_silicon_retina", &forward_silicon_retina, "run silicon retina model and save only the fired events");
    m.def("forward_silicon_retina_online", &forward_silicon_retina_online, "run silicon retina model online and save only the fired events, however this is done frame by frame (useful for streaming data)");
}