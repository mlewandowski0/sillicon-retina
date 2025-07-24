import torch
from . import silicon_retina_cuda
from dataclasses import dataclass

class SiliconRetinaConfig:
    alpha1 : float = 0.5
    alpha2 : float = 0.5
    V_BIP_LPF_alpha : float = 0.5
    lambda_OPL : float = 3.0
    G_A_zero : float = 0.5
    lambda_A : float = 0.5
    p1 : float = 0.25
    I_bias : float = 0.01
    p2 : float = 1.0
    p3 : float = 0.2
    U_rest : float = -0.02
    U_th : float = 0.05
    U_reset : float = 0.0


class SiliconRetinaVerboseFloatCUDA(torch.nn.Module):
    def __init__(self, silicon_retina_config: SiliconRetinaConfig, width, height, T):
        super(SiliconRetinaVerboseFloatCUDA, self).__init__()
        self.config = silicon_retina_config
        self.width = width
        self.height = height
        self.T = T
        
        self.register_buffer("LPF1", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF1", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("BLURRED2", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("LPF2", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_OPL", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_BIP", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_BIP_LPF", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("Q_V_BIP", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_Gang", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_LIF", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("fired", torch.zeros((self.T, self.height, self.width), dtype=torch.uint8, device="cuda:0"))


    def forward(self, imgs):
        assert imgs.shape[0] == self.T, "Input tensor must have the first dimension equal to T"
        assert imgs.shape[1] == self.height, "Input tensor must have the second dimension equal to height"
        assert imgs.shape[2] == self.width, "Input tensor must have the third dimension equal to width"
        assert imgs.dtype == torch.float32, "Input tensor must be of type float32"

        self.LPF1.zero_()
        self.HPF1.zero_()
        self.BLURRED2.zero_()
        self.LPF2.zero_()
        self.HPF2.zero_()
        self.I_OPL.zero_()
        self.V_BIP.zero_()
        self.Q_V_BIP.zero_()
        self.V_BIP_LPF.zero_()
        self.HPF2.zero_()
        self.I_Gang.zero_()
        self.V_LIF.zero_()
        self.fired.zero_()

        return silicon_retina_cuda.forward_silicon_retina_verbose(imgs, 
                                                                  self.LPF1, 
                                                                  self.HPF1, 
                                                                  self.BLURRED2, 
                                                                  self.LPF2, 
                                                                  self.I_OPL,
                                                                  self.V_BIP,
                                                                  self.Q_V_BIP,
                                                                  self.V_BIP_LPF,
                                                                  self.HPF2, 
                                                                  self.I_Gang, 
                                                                  self.V_LIF, 
                                                                  self.fired,
                                                                  self.T, self.height, self.width,
                                                                  self.config.alpha1,
                                                                  self.config.alpha2,
                                                                  self.config.V_BIP_LPF_alpha,
                                                                  self.config.lambda_OPL,
                                                                  self.config.G_A_zero,
                                                                  self.config.lambda_A,
                                                                  self.config.p1,
                                                                  self.config.I_bias,
                                                                  self.config.p2,
                                                                  self.config.p3,
                                                                  self.config.U_rest,
                                                                  self.config.U_th,
                                                                  self.config.U_reset)
            


#class silicon_retina_cuda(torch.nn.Module):
#    def __init__(self, ):
#        super(silicon_retina_double_buffered_cuda, self).__init__()
#        self.se = se_cuda.silicon_retina_double_buffered_cuda()
#
#    def forward(self, input_tensor):
#        return self.se(input_tensor)
#
#    def extra_repr(self):
#        return 'silicon_retina_double_buffered_cuda'