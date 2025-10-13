#pragma once
#include <cstdint>

enum class HPF_MODE 
{
    SUB   = 0,
    MODE1 = 1,
    MODE2 = 2
};

enum class BLUR_MODE 
{
    BLUR_AVG  = 0,
    BLUR_MAX  = 1,
    BLUR_NONE = 2   
};

enum LPF_MODE
{
    LPF_1
};

enum LIF_INTEGRATION_MODE
{
    LIF_INTEGRATION_1
};

enum LIF_RESET_MODE
{
    LIF_SUBTRACT_RESET
};

struct SiliconRetinaParams 
{
    // sizes 
    int H = 256;
    int W = 256;
    
    // filters and model parameters 
    float alpha1            =  0.5f;
    float alpha2            =  0.5f;
    float V_BIP_LPF_alpha   =  0.5f;
    float lambda_OPL        =  3.0f;
    float G_A_zero          =  0.0f;
    float lambda_A          =  0.5f;
    float p1                =  0.25f;
    float I_bias_on         =  0.01f;
    float I_bias_off        =  0.01f;
    float p2                =  1.0f;
    float p3                =  0.2f;

    float U_rest_on         = -0.02f;
    float U_rest_off        = -0.02f;

    float U_th_on           =  0.05f;
    float U_th_off          =  0.05f;
    
    float U_reset_on        = 0.0f;
    float U_reset_off       = 0.0f;

    HPF_MODE HPF1_mode      = HPF_MODE::SUB;
    float HPF1_param        = 0.5f;

    HPF_MODE HPF2_mode      = HPF_MODE::SUB;
    float HPF2_param        = 0.5f;

    bool apply_log_to_input = false;
    bool turn_off_feedback  = false;

    BLUR_MODE blur1_mode = BLUR_MODE::BLUR_AVG;
    BLUR_MODE blur2_mode = BLUR_MODE::BLUR_AVG;

    SiliconRetinaParams(int H = 256,
                        int W = 256,
                        float alpha1 = 0.5f,
                        float alpha2 = 0.5f,
                        float V_BIP_LPF_alpha = 0.5f,
                        float lambda_OPL = 3.0f,
                        float G_A_zero = 0.0f,
                        float lambda_A = 0.5f,
                        float p1 = 0.25f,
                        float I_bias_on = 0.01f,
                        float I_bias_off = 0.01f,
                        float p2 = 1.0f,
                        float p3 = 0.2f,
                        float U_rest_on = -0.02f,
                        float U_rest_off = -0.02f,
                        float U_th_on = 0.05f,
                        float U_th_off = 0.05f,
                        float U_reset_on = 0.0f,
                        float U_reset_off = 0.0f,
                        
                        HPF_MODE HPF1_mode = HPF_MODE::SUB,
                        float HPF1_param = 0.5f,
                        HPF_MODE HPF2_mode= HPF_MODE::SUB,
                        float HPF2_param=0.5f,
                        
                        BLUR_MODE blur1_mode = BLUR_MODE::BLUR_AVG,
                        BLUR_MODE blur2_mode = BLUR_MODE::BLUR_AVG,

                        bool apply_log_to_input=false,
                        bool turn_off_feedback=false)
    {
        this->H = H;
        this->W = W;
        this->alpha1 = alpha1;
        this->alpha2 = alpha2;
        this->V_BIP_LPF_alpha = V_BIP_LPF_alpha;
        this->lambda_OPL = lambda_OPL;
        this->G_A_zero = G_A_zero;
        this->lambda_A = lambda_A;
        this->p1 = p1;
        this->I_bias_on = I_bias_on;
        this->I_bias_off = I_bias_off;
        this->p2 = p2;
        this->p3 = p3;
        this->U_rest_on = U_rest_on;
        this->U_rest_off = U_rest_off;
        this->U_th_on = U_th_on;
        this->U_th_off = U_th_off;
        this->U_reset_on = U_reset_on;
        this->U_reset_off = U_reset_off;
        this->HPF1_mode = HPF1_mode;
        this->HPF1_param = HPF1_param;
        this->HPF2_mode = HPF2_mode;
        this->HPF2_param = HPF2_param;
        this->apply_log_to_input = apply_log_to_input;
        this->blur1_mode = blur1_mode;
        this->blur2_mode = blur2_mode;
        this->turn_off_feedback = turn_off_feedback;
    }


    // Nice to have for debugging
    std::string repr() const {
        char buf[2048];
        std::snprintf(buf, sizeof(buf),
            "SiliconRetinaParams(H=%d, W=%d, alpha1=%.6g, alpha2=%.6g, "
            "V_BIP_LPF_alpha=%.6g, lambda_OPL=%.6g, G_A_zero=%.6g, lambda_A=%.6g, "
            "p1=%.6g, I_bias_on=%.6g, I_bias_off=%.6g, p2=%.6g, p3=%.6g, "
            "U_rest_on=%.6g, U_rest_off=%.6g, U_th_on=%.6g, U_th_off=%.6g, "
            "U_reset_on=%.6g, U_reset_off=%.6g, HPF1_mode=%d, HPF1_param=%.6g, "
            "HPF2_mode=%d, HPF2_param=%.6g, BLUR_MODE1=%d, BLUR_MODE2=%d, apply_log_to_input=%d, turn_off_feedback=%d)",
            H, W, alpha1, alpha2, V_BIP_LPF_alpha, lambda_OPL, G_A_zero, lambda_A,
            p1, I_bias_on, I_bias_off, p2, p3, U_rest_on, U_rest_off, U_th_on, U_th_off,
            U_reset_on, U_reset_off, (int)HPF1_mode, HPF1_param, (int)HPF2_mode, HPF2_param, (int)blur1_mode, (int)blur2_mode, (int)apply_log_to_input, (int) turn_off_feedback);
        return std::string(buf);
    }
};