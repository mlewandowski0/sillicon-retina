import torch
from . import silicon_retina_cuda
from dataclasses import dataclass, asdict, fields

@dataclass
class SiliconRetinaConfig:
    alpha1 : float = 0.5
    alpha2 : float = 0.5
    V_BIP_LPF_alpha : float = 0.5
    lambda_OPL : float = 3.0
    G_A_zero : float = 0.5
    lambda_A : float = 0.5
    p1 : float = 0.25
    I_bias_on : float = 0.01
    I_bias_off : float = 0.01
    p2 : float = 1.0
    p3 : float = 0.2
    U_rest_on : float = -0.02
    U_rest_off : float = -0.02

    U_th_on : float = 0.05
    U_th_off : float = 0.05

    U_reset_on : float = 0.0
    U_reset_off : float = 0.0
    
    HPF1_mode : int = 0
    HPF1_param : float = 0.5
    
    HPF2_mode : int = 0 
    HPF2_param : float = 0.5

    apply_log_to_input : bool = False
    
    # niceties
    def to_dict(self) -> dict:
        return asdict(self)       # handles nested dataclasses too

    @classmethod
    def from_dict(cls, d: dict) -> "SiliconRetinaConfig":
        # ignore unknown keys so older code can read newer JSONs
        names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in names}
        return cls(**filtered)
    

def preprocess_input(img, config):
    if config.apply_log_to_input:
        img = torch.log(255.0 * img + 1e-3) / 5.54126393731521
    return torch.clamp(img, 0., 1.0)


class SiliconRetinaVerboseFloatCUDA(torch.nn.Module):
    def __init__(self, silicon_retina_config: SiliconRetinaConfig, T, height, width):
        super(SiliconRetinaVerboseFloatCUDA, self).__init__()
        self.config = silicon_retina_config
        self.width = width
        self.height = height
        self.T = T
        
        self.register_buffer("LPF1", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF1_mem", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF1", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("BLURRED2", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("LPF2", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_OPL", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_BIP", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("Q_V_BIP", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_BIP_LPF", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2_mem", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        
        self.register_buffer("I_Gang_on", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_Gang_off", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))

        self.register_buffer("V_LIF_on", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_LIF_off", torch.zeros((self.T, self.height, self.width), dtype=torch.float32, device="cuda:0"))

        self.register_buffer("fired_on", torch.zeros((self.T, self.height, self.width), dtype=torch.uint8, device="cuda:0"))
        self.register_buffer("fired_off", torch.zeros((self.T, self.height, self.width), dtype=torch.uint8, device="cuda:0"))


    def forward(self, imgs):
        assert imgs.shape[0] == self.T, "Input tensor must have the first dimension equal to T"
        assert imgs.shape[1] == self.height, "Input tensor must have the second dimension equal to height"
        assert imgs.shape[2] == self.width, "Input tensor must have the third dimension equal to width"
        assert imgs.dtype == torch.float32, "Input tensor must be of type float32"

        imgs = preprocess_input(imgs, config=self.config)

        self.LPF1.zero_()
        self.HPF1.zero_()
        self.BLURRED2.zero_()
        self.LPF2.zero_()
        self.I_OPL.zero_()
        self.V_BIP.zero_()
        self.Q_V_BIP.zero_()
        self.V_BIP_LPF.zero_()
        self.HPF2.zero_()
        self.I_Gang_on.zero_()
        self.I_Gang_off.zero_()
        
        self.V_LIF_on.zero_()
        self.V_LIF_off.zero_()

        self.fired_on.zero_()
        self.fired_off.zero_()

        return silicon_retina_cuda.forward_silicon_retina_verbose(imgs, 
                                                                  self.LPF1, 
                                                                  self.HPF1_mem,
                                                                  self.HPF1, 
                                                                  self.BLURRED2, 
                                                                  self.LPF2, 
                                                                  self.I_OPL,
                                                                  self.V_BIP,
                                                                  self.Q_V_BIP,
                                                                  self.V_BIP_LPF,
                                                                  self.HPF2_mem,
                                                                  self.HPF2, 
                                                                  self.I_Gang_on, 
                                                                  self.I_Gang_off, 
                                                                  self.V_LIF_on, 
                                                                  self.V_LIF_off, 
                                                                  self.fired_on,
                                                                  self.fired_off,
                                                                  self.T, self.height, self.width,
                                                                  self.config.alpha1,
                                                                  self.config.alpha2,
                                                                  self.config.V_BIP_LPF_alpha,
                                                                  self.config.lambda_OPL,
                                                                  self.config.G_A_zero,
                                                                  self.config.lambda_A,
                                                                  self.config.p1,
                                                                  self.config.I_bias_on,
                                                                  self.config.I_bias_off,
                                                                  self.config.p2,
                                                                  self.config.p3,
                                                                  self.config.U_rest_on,
                                                                  self.config.U_rest_off,
                                                                  self.config.U_th_on,
                                                                  self.config.U_th_off,
                                                                  self.config.U_reset_on,
                                                                  self.config.U_reset_off,
                                                                  self.config.HPF1_mode,
                                                                  self.config.HPF1_param,
                                                                  self.config.HPF2_mode,
                                                                  self.config.HPF2_param
                                                                  )


class SiliconRetinaFloatCUDAOnline(torch.nn.Module):
    def __init__(self, silicon_retina_config: SiliconRetinaConfig, height, width):
        super(SiliconRetinaFloatCUDAOnline, self).__init__()
        self.config = silicon_retina_config
        self.width = width
        self.height = height

        self.register_buffer("LPF1",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF1_mem", 
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF1",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("BLURRED2",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("LPF2",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_OPL",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_BIP",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("Q_V_BIP",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_BIP_LPF",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2_mem", 
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))

        self.register_buffer("I_Gang_on",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_Gang_off",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))

        self.register_buffer("V_LIF_on",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_LIF_off",
                             torch.zeros((2, self.height, self.width), dtype=torch.float32, device="cuda:0"))

        self.register_buffer("fired_on",
                             torch.zeros((2, self.height, self.width), dtype=torch.uint8, device="cuda:0"))
        self.register_buffer("fired_off",
                             torch.zeros((2, self.height, self.width), dtype=torch.uint8, device="cuda:0"))

    def reset(self):
        self.LPF1.zero_()
        self.HPF1.zero_()
        self.BLURRED2.zero_()
        self.LPF2.zero_()
        self.I_OPL.zero_()
        self.V_BIP.zero_()
        self.Q_V_BIP.zero_()
        self.V_BIP_LPF.zero_()
        self.HPF2.zero_()
        self.I_Gang_on.zero_()
        self.I_Gang_off.zero_()

        self.V_LIF_on.zero_()
        self.V_LIF_off.zero_()

        self.fired_on.zero_()
        self.fired_off.zero_()

    def forward(self, img, t):
        assert img.shape[0] == 1, "Input tensor must have the first dimension equal to 1"
        assert img.shape[1] == self.height, "Input tensor must have the second dimension equal to height"
        assert img.shape[2] == self.width, "Input tensor must have the third dimension equal to width"
        assert img.dtype == torch.float32, f"Input tensor must be of type float32, was {img.dtype}"

        img = preprocess_input(img, config=self.config)

        return silicon_retina_cuda.forward_silicon_retina_online(img,
                                                                  self.LPF1,
                                                                  self.HPF1_mem,
                                                                  self.HPF1,
                                                                  self.BLURRED2,
                                                                  self.LPF2,
                                                                  self.I_OPL,
                                                                  self.V_BIP,
                                                                  self.Q_V_BIP,
                                                                  self.V_BIP_LPF,
                                                                  self.HPF2_mem,
                                                                  self.HPF2,
                                                                  self.I_Gang_on,
                                                                  self.I_Gang_off,
                                                                  self.V_LIF_on,
                                                                  self.V_LIF_off,
                                                                  self.fired_on,
                                                                  self.fired_off,
                                                                  t, self.height, self.width,
                                                                  self.config.alpha1,
                                                                  self.config.alpha2,
                                                                  self.config.V_BIP_LPF_alpha,
                                                                  self.config.lambda_OPL,
                                                                  self.config.G_A_zero,
                                                                  self.config.lambda_A,
                                                                  self.config.p1,
                                                                  self.config.I_bias_on,
                                                                  self.config.I_bias_off,
                                                                  self.config.p2,
                                                                  self.config.p3,
                                                                  self.config.U_rest_on,
                                                                  self.config.U_rest_off,
                                                                  self.config.U_th_on,
                                                                  self.config.U_th_off,
                                                                  self.config.U_reset_on,
                                                                  self.config.U_reset_off,
                                                                  self.config.HPF1_mode,
                                                                  self.config.HPF1_param,
                                                                  self.config.HPF2_mode,
                                                                  self.config.HPF2_param)

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