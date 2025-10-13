
#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include "retina_params.h"

namespace py = pybind11;


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
    SiliconRetinaParams& retina_config);


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
    SiliconRetinaParams& retina_config);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::enum_<HPF_MODE>(m, "HPF_MODE")
        .value("SUB", HPF_MODE::SUB)
        .value("MODE1", HPF_MODE::MODE1)
        .value("MODE2", HPF_MODE::MODE2)
        .export_values();

    py::enum_<BLUR_MODE>(m, "BLUR_MODE")
        .value("BLUR_AVG", BLUR_MODE::BLUR_AVG)
        .value("BLUR_MAX", BLUR_MODE::BLUR_MAX)
        .value("BLUR_NONE", BLUR_MODE::BLUR_NONE)
        .export_values();
    
    // Params class 
    py::class_<SiliconRetinaParams>(m, "SiliconRetinaParams")
        .def(py::init<
        int,int,
        float,float,float,float,float,float,
        float,float,float,float,float,
        float,float,float,float,
        float,float,
        HPF_MODE,float,
        HPF_MODE,float,
        BLUR_MODE,BLUR_MODE,
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
        py::arg("apply_log_to_input") = false,
        py::arg("turn_off_feedback") = false
        )
        // sizes
        .def_readwrite("H", &SiliconRetinaParams::H)
        .def_readwrite("W", &SiliconRetinaParams::W)

        // filters and model parameters
        .def_readwrite("alpha1", &SiliconRetinaParams::alpha1)
        .def_readwrite("alpha2", &SiliconRetinaParams::alpha2)
        .def_readwrite("V_BIP_LPF_alpha", &SiliconRetinaParams::V_BIP_LPF_alpha)
        .def_readwrite("lambda_OPL", &SiliconRetinaParams::lambda_OPL)
        .def_readwrite("G_A_zero", &SiliconRetinaParams::G_A_zero)
        .def_readwrite("p1", &SiliconRetinaParams::p1)
        .def_readwrite("I_bias_on", &SiliconRetinaParams::I_bias_on)
        .def_readwrite("I_bias_off", &SiliconRetinaParams::I_bias_off)
        .def_readwrite("p2", &SiliconRetinaParams::p2)
        .def_readwrite("p3", &SiliconRetinaParams::p3)
        .def_readwrite("U_rest_on", &SiliconRetinaParams::U_rest_on)
        .def_readwrite("U_rest_off", &SiliconRetinaParams::U_rest_off)
        .def_readwrite("U_reset_on", &SiliconRetinaParams::U_reset_on)
        .def_readwrite("U_reset_off", &SiliconRetinaParams::U_reset_off)

        // HPF modes and params
        .def_readwrite("HPF1_mode", &SiliconRetinaParams::HPF1_mode)
        .def_readwrite("HPF1_param", &SiliconRetinaParams::HPF1_param)
        .def_readwrite("HPF2_mode", &SiliconRetinaParams::HPF2_mode)
        .def_readwrite("HPF2_param", &SiliconRetinaParams::HPF2_param)

        .def_readwrite("blur1_mode", &SiliconRetinaParams::blur1_mode)
        .def_readwrite("blur2_mode", &SiliconRetinaParams::blur2_mode)
        .def_readwrite("apply_log_to_input", &SiliconRetinaParams::apply_log_to_input)
        .def_readwrite("turn_off_feedback", &SiliconRetinaParams::turn_off_feedback)
        .def("__repr__", &SiliconRetinaParams::repr);

    m.def("forward_silicon_retina_verbose", &forward_silicon_retina_verbose, "run silicon retina model and save the outputs for every layer (most useful for debugging!)");
    m.def("forward_silicon_retina", &forward_silicon_retina, "run silicon retina model and save only the fired events");
    m.def("forward_silicon_retina_online", &forward_silicon_retina_online, "run silicon retina model online and save only the fired events, however this is done frame by frame (useful for streaming data)");
}