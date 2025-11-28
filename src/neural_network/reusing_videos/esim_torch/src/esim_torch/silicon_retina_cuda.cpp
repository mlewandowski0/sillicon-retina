#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "retina_params.h"

namespace py = pybind11;

void initialize_kernels(SiliconRetinaParams &config);

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
    SiliconRetinaParams& retina_config);

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
    SiliconRetinaParams &retina_config);


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
    SiliconRetinaParams &retina_config);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Silicon Retina Model with Configurable Convolution Kernels";
    
    // Q_FEEDBACK_MODE enum
    py::enum_<Q_FEEDBACK_MODE>(m, "Q_FEEDBACK_MODE")
        .value("ABS", Q_FEEDBACK_MODE::ABS, "Absolute value feedback")
        .value("QUADRATIC", Q_FEEDBACK_MODE::QUADRATIC, "Quadratic feedback")
        .export_values(); 

    // HPF_MODE enum
    py::enum_<HPF_MODE>(m, "HPF_MODE")
        .value("SUB", HPF_MODE::SUB, "Subtraction mode")
        .value("MODE1", HPF_MODE::MODE1, "High-pass filter mode 1")
        .value("MODE2", HPF_MODE::MODE2, "High-pass filter mode 2")
        .export_values();

    // BLUR_MODE enum
    py::enum_<BLUR_MODE>(m, "BLUR_MODE")
        .value("BLUR_AVG", BLUR_MODE::BLUR_AVG, "Average/box blur")
        .value("BLUR_MAX", BLUR_MODE::BLUR_MAX, "Maximum pooling")
        .value("BLUR_NONE", BLUR_MODE::BLUR_NONE, "No blurring")
        .export_values();
    
    // KERNEL_TYPE enum
    py::enum_<KERNEL_TYPE>(m, "KERNEL_TYPE")
        .value("BOX", KERNEL_TYPE::BOX, "Uniform box filter")
        .value("GAUSSIAN", KERNEL_TYPE::GAUSSIAN, "Gaussian blur")
        .value("LAPLACIAN", KERNEL_TYPE::LAPLACIAN, "Laplacian (second derivative)")
        .value("CUSTOM", KERNEL_TYPE::CUSTOM, "Custom user-provided kernel")
        .export_values();
    
    // SiliconRetinaParams class 
    py::class_<SiliconRetinaParams>(m, "SiliconRetinaParams")
        .def(py::init<
            int, int,
            float, float, float, float, float, float,
            float, float, float, float, float,
            float, float, float, float,
            float, float,
            HPF_MODE, float,
            HPF_MODE, float,
            BLUR_MODE, BLUR_MODE,
            Q_FEEDBACK_MODE,
            bool, bool>(),
            py::arg("H") = 256,
            py::arg("W") = 256,
            py::arg("alpha1") = 0.5f,
            py::arg("alpha2") = 0.5f,
            py::arg("V_BIP_LPF_alpha") = 0.5f,
            py::arg("lambda_OPL") = 3.0f,
            py::arg("G_A_zero") = 0.0f,
            py::arg("lambda_A") = 0.5f,
            py::arg("p1") = 0.25f,
            py::arg("I_bias_on") = 0.01f,
            py::arg("I_bias_off") = 0.01f,
            py::arg("p2") = 1.0f,
            py::arg("p3") = 0.2f,
            py::arg("U_rest_on") = -0.02f,
            py::arg("U_rest_off") = -0.02f,
            py::arg("U_th_on") = 0.05f,
            py::arg("U_th_off") = 0.05f,
            py::arg("U_reset_on") = 0.0f,
            py::arg("U_reset_off") = 0.0f,
            py::arg("HPF1_mode") = HPF_MODE::SUB,
            py::arg("HPF1_param") = 0.5f,
            py::arg("HPF2_mode") = HPF_MODE::SUB,
            py::arg("HPF2_param") = 0.5f,
            py::arg("blur1_mode") = BLUR_MODE::BLUR_AVG,
            py::arg("blur2_mode") = BLUR_MODE::BLUR_AVG,
            py::arg("q_feedback_mode") = Q_FEEDBACK_MODE::ABS,
            py::arg("apply_log_to_input") = false,
            py::arg("turn_off_feedback") = false
        )
        // Sizes
        .def_readwrite("H", &SiliconRetinaParams::H)
        .def_readwrite("W", &SiliconRetinaParams::W)

        // Filters and model parameters
        .def_readwrite("alpha1", &SiliconRetinaParams::alpha1)
        .def_readwrite("alpha2", &SiliconRetinaParams::alpha2)
        .def_readwrite("V_BIP_LPF_alpha", &SiliconRetinaParams::V_BIP_LPF_alpha)
        .def_readwrite("lambda_OPL", &SiliconRetinaParams::lambda_OPL)
        .def_readwrite("G_A_zero", &SiliconRetinaParams::G_A_zero)
        .def_readwrite("lambda_A", &SiliconRetinaParams::lambda_A)
        .def_readwrite("p1", &SiliconRetinaParams::p1)
        .def_readwrite("I_bias_on", &SiliconRetinaParams::I_bias_on)
        .def_readwrite("I_bias_off", &SiliconRetinaParams::I_bias_off)
        .def_readwrite("p2", &SiliconRetinaParams::p2)
        .def_readwrite("p3", &SiliconRetinaParams::p3)
        .def_readwrite("U_rest_on", &SiliconRetinaParams::U_rest_on)
        .def_readwrite("U_rest_off", &SiliconRetinaParams::U_rest_off)
        .def_readwrite("U_th_on", &SiliconRetinaParams::U_th_on)
        .def_readwrite("U_th_off", &SiliconRetinaParams::U_th_off)
        .def_readwrite("U_reset_on", &SiliconRetinaParams::U_reset_on)
        .def_readwrite("U_reset_off", &SiliconRetinaParams::U_reset_off)

        // HPF modes and params
        .def_readwrite("HPF1_mode", &SiliconRetinaParams::HPF1_mode)
        .def_readwrite("HPF1_param", &SiliconRetinaParams::HPF1_param)
        .def_readwrite("HPF2_mode", &SiliconRetinaParams::HPF2_mode)
        .def_readwrite("HPF2_param", &SiliconRetinaParams::HPF2_param)

        // Q feedback mode
        .def_readwrite("q_feedback_mode", &SiliconRetinaParams::q_feedback_mode)

        // Blur modes
        .def_readwrite("blur1_mode", &SiliconRetinaParams::blur1_mode)
        .def_readwrite("blur2_mode", &SiliconRetinaParams::blur2_mode)
        
        // Kernel configuration (NEW)
        .def_readwrite("kernel1_type", &SiliconRetinaParams::kernel1_type, 
                      "Type of convolution kernel for first blur stage")
        .def_readwrite("kernel2_type", &SiliconRetinaParams::kernel2_type,
                      "Type of convolution kernel for second blur stage")
        .def_readwrite("kernel1_size", &SiliconRetinaParams::kernel1_size,
                      "Size of first kernel (must be odd: 3, 5, 7, etc.)")
        .def_readwrite("kernel2_size", &SiliconRetinaParams::kernel2_size,
                      "Size of second kernel (must be odd: 3, 5, 7, etc.)")
        .def_readwrite("kernel1_sigma", &SiliconRetinaParams::kernel1_sigma,
                      "Sigma for Gaussian kernel 1")
        .def_readwrite("kernel2_sigma", &SiliconRetinaParams::kernel2_sigma,
                      "Sigma for Gaussian kernel 2")
        .def_readwrite("custom_kernel1", &SiliconRetinaParams::custom_kernel1,
                      "Custom kernel weights for kernel 1 (1D flattened array)")
        .def_readwrite("custom_kernel2", &SiliconRetinaParams::custom_kernel2,
                      "Custom kernel weights for kernel 2 (1D flattened array)")
        
        // Other params
        .def_readwrite("apply_log_to_input", &SiliconRetinaParams::apply_log_to_input)
        .def_readwrite("turn_off_feedback", &SiliconRetinaParams::turn_off_feedback)
        
        // String representation
        .def("__repr__", &SiliconRetinaParams::repr);

    // Main processing function
    m.def("forward_silicon_retina", 
          &forward_silicon_retina, 
          "Run silicon retina model with configurable convolution kernels",
          py::arg("d_images"),
          py::arg("d_LPF1"),
          py::arg("d_HPF1_mem"),
          py::arg("d_HPF1"),
          py::arg("d_BLURRED2"),
          py::arg("d_LPF2"),
          py::arg("d_I_OPL"),
          py::arg("d_V_BIP"),
          py::arg("d_Q_V_BIP"),
          py::arg("d_V_BIP_LPF"),
          py::arg("d_HPF2_mem"),
          py::arg("d_HPF2"),
          py::arg("d_I_Gang_on"),
          py::arg("d_I_Gang_off"),
          py::arg("d_V_LIF_on"),
          py::arg("d_V_LIF_off"),
          py::arg("d_fired_on"),
          py::arg("d_fired_off"),
          py::arg("T"),
          py::arg("retina_config"));

        // Main processing function
    m.def("forward_silicon_retina_online", 
          &forward_silicon_retina_online, 
          "Run silicon retina model with configurable convolution kernels for online model",
          py::arg("d_images"),
          py::arg("d_LPF1"),
          py::arg("d_HPF1_mem"),
          py::arg("d_HPF1"),
          py::arg("d_BLURRED2"),
          py::arg("d_LPF2"),
          py::arg("d_I_OPL"),
          py::arg("d_V_BIP"),
          py::arg("d_Q_V_BIP"),
          py::arg("d_V_BIP_LPF"),
          py::arg("d_HPF2_mem"),
          py::arg("d_HPF2"),
          py::arg("d_I_Gang_on"),
          py::arg("d_I_Gang_off"),
          py::arg("d_V_LIF_on"),
          py::arg("d_V_LIF_off"),
          py::arg("d_fired_on"),
          py::arg("d_fired_off"),
          py::arg("curr_t"),
          py::arg("retina_config"));

    m.def("forward_silicon_retina_verbose", 
          &forward_silicon_retina_verbose, 
          "Run silicon retina model with configurable convolution kernels for online model",
          py::arg("d_images"),
          py::arg("d_LPF1"),
          py::arg("d_HPF1_mem"),
          py::arg("d_HPF1"),
          py::arg("d_BLURRED2"),
          py::arg("d_LPF2"),
          py::arg("d_I_OPL"),
          py::arg("d_V_BIP"),
          py::arg("d_Q_V_BIP"),
          py::arg("d_V_BIP_LPF"),
          py::arg("d_HPF2_mem"),
          py::arg("d_HPF2"),
          py::arg("d_I_Gang_on"),
          py::arg("d_I_Gang_off"),
          py::arg("d_V_LIF_on"),
          py::arg("d_V_LIF_off"),
          py::arg("d_fired_on"),
          py::arg("d_fired_off"),
          py::arg("T"),
          py::arg("retina_config"));


    m.def("initialize_kernels",
          &initialize_kernels,
          "Initialize convolution kernels based on the configuration in SiliconRetinaParams",
          py::arg("retina_config"));
}