
#include <torch/extension.h>
#include <vector>

torch::Tensor forward_silicon_retina_verbose(
    const torch::Tensor& d_images,
    torch::Tensor& d_LPF1,
    torch::Tensor& d_HPF1,
    torch::Tensor& d_BLURRED2,
    torch::Tensor& d_LPF2,
    torch::Tensor& d_I_OPL,
    torch::Tensor& d_V_BIP,
    torch::Tensor& d_Q_V_BIP,
    torch::Tensor& d_V_BIP_LPF,
    torch::Tensor& d_HPF2,
    torch::Tensor& d_I_Gang,
    torch::Tensor& d_V_LIF,
    torch::Tensor& d_fired,

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
    


/*
std::vector<torch::Tensor> esim_forward(
    const torch::Tensor& images,
    const torch::Tensor& timestamps,
    const torch::Tensor& init_reference_values,
    const torch::Tensor& reference_values_over_time,
    const torch::Tensor& offsets,
    torch::Tensor& events,
    torch::Tensor& timestamps_last_event,
    float contrast_threshold_negative,
    float contrast_threshold_positive,
    int64_t refractory_period);

std::vector<torch::Tensor> esim_forward_count_events(
    const torch::Tensor& images,
    const torch::Tensor& timestamps,
    const torch::Tensor& init_reference_values,
    torch::Tensor& reference_values_over_time
    torch::Tensor& event_counts,
    torch::Tensor& timestamps_last_event,
    float contrast_threshold_negative,
    float contrast_threshold_positive);
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_silicon_retina_verbose", &forward_silicon_retina_verbose, "run silicon retina model and save the outputs for every layer (most useful for debugging!)");
    m.def("forward_silicon_retina", &forward_silicon_retina, "run silicon retina model and save only the fired events");
    m.def("forward_silicon_retina_online", &forward_silicon_retina_online, "run silicon retina model online and save only the fired events, however this is done frame by frame (useful for streaming data)");
}