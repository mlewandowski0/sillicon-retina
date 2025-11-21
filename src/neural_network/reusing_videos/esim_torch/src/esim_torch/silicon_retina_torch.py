import torch
from . import silicon_retina_cuda
from dataclasses import dataclass, asdict, fields
from typing import List, Optional

@dataclass
class SiliconRetinaConfig:
    # Image dimensions
    H: int = 256
    W: int = 256
    
    # Temporal filters
    alpha1: float = 0.5
    alpha2: float = 0.5
    V_BIP_LPF_alpha: float = 0.5
    
    # Retina model parameters
    lambda_OPL: float = 3.0
    G_A_zero: float = 0.5
    lambda_A: float = 0.5
    p1: float = 0.25
    I_bias_on: float = 0.01
    I_bias_off: float = 0.01
    p2: float = 1.0
    p3: float = 0.2
    
    # LIF parameters
    U_rest_on: float = -0.02
    U_rest_off: float = -0.02
    U_th_on: float = 0.05
    U_th_off: float = 0.05
    U_reset_on: float = 0.0
    U_reset_off: float = 0.0
    
    # High-pass filter modes
    HPF1_mode: int = 0  # silicon_retina_cuda.HPF_MODE.SUB
    HPF1_param: float = 0.5
    HPF2_mode: int = 0  # silicon_retina_cuda.HPF_MODE.SUB
    HPF2_param: float = 0.5
    
    # Blur modes
    blur1_mode: int = 0  # silicon_retina_cuda.BLUR_MODE.BLUR_AVG
    blur2_mode: int = 0  # silicon_retina_cuda.BLUR_MODE.BLUR_AVG
    
    # NEW: Kernel configuration
    kernel1_type: int = 0  # silicon_retina_cuda.KERNEL_TYPE.BOX
    kernel2_type: int = 0  # silicon_retina_cuda.KERNEL_TYPE.BOX
    kernel1_size: int = 3
    kernel2_size: int = 3
    kernel1_sigma: float = 1.0
    kernel2_sigma: float = 1.0
    custom_kernel1: Optional[List[float]] = None
    custom_kernel2: Optional[List[float]] = None
    
    # Other options
    apply_log_to_input: bool = False
    turn_off_feedback: bool = False
    
    def __post_init__(self):
        """Initialize custom kernel lists if None"""
        if self.custom_kernel1 is None:
            self.custom_kernel1 = []
        if self.custom_kernel2 is None:
            self.custom_kernel2 = []
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "SiliconRetinaConfig":
        """Create config from dictionary, ignoring unknown keys"""
        names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in names}
        
        # Convert enum values if they're already enum objects
        if 'HPF1_mode' in filtered and hasattr(filtered['HPF1_mode'], 'value'):
            filtered['HPF1_mode'] = filtered['HPF1_mode'].value
        if 'HPF2_mode' in filtered and hasattr(filtered['HPF2_mode'], 'value'):
            filtered['HPF2_mode'] = filtered['HPF2_mode'].value
        if 'blur1_mode' in filtered and hasattr(filtered['blur1_mode'], 'value'):
            filtered['blur1_mode'] = filtered['blur1_mode'].value
        if 'blur2_mode' in filtered and hasattr(filtered['blur2_mode'], 'value'):
            filtered['blur2_mode'] = filtered['blur2_mode'].value
        if 'kernel1_type' in filtered and hasattr(filtered['kernel1_type'], 'value'):
            filtered['kernel1_type'] = filtered['kernel1_type'].value
        if 'kernel2_type' in filtered and hasattr(filtered['kernel2_type'], 'value'):
            filtered['kernel2_type'] = filtered['kernel2_type'].value
        
        return cls(**filtered)
    
    def to_cpp_struct(self) -> silicon_retina_cuda.SiliconRetinaParams:
        """Convert to C++ parameter struct"""
        params = silicon_retina_cuda.SiliconRetinaParams(
            H=self.H,
            W=self.W,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            V_BIP_LPF_alpha=self.V_BIP_LPF_alpha,
            lambda_OPL=self.lambda_OPL,
            G_A_zero=self.G_A_zero,
            lambda_A=self.lambda_A,
            p1=self.p1,
            I_bias_on=self.I_bias_on,
            I_bias_off=self.I_bias_off,
            p2=self.p2,
            p3=self.p3,
            U_rest_on=self.U_rest_on,
            U_rest_off=self.U_rest_off,
            U_th_on=self.U_th_on,
            U_th_off=self.U_th_off,
            U_reset_on=self.U_reset_on,
            U_reset_off=self.U_reset_off,
            HPF1_mode=silicon_retina_cuda.HPF_MODE(self.HPF1_mode),
            HPF1_param=self.HPF1_param,
            HPF2_mode=silicon_retina_cuda.HPF_MODE(self.HPF2_mode),
            HPF2_param=self.HPF2_param,
            blur1_mode=silicon_retina_cuda.BLUR_MODE(self.blur1_mode),
            blur2_mode=silicon_retina_cuda.BLUR_MODE(self.blur2_mode),
            apply_log_to_input=self.apply_log_to_input,
            turn_off_feedback=self.turn_off_feedback
        )
        
        # Set kernel configuration
        params.kernel1_type = silicon_retina_cuda.KERNEL_TYPE(self.kernel1_type)
        params.kernel2_type = silicon_retina_cuda.KERNEL_TYPE(self.kernel2_type)
        
        params.kernel1_size = int(self.kernel1_size)
        params.kernel2_size = int(self.kernel2_size)
        params.kernel1_sigma = float(self.kernel1_sigma)
        params.kernel2_sigma = float(self.kernel2_sigma)
        
        # Set custom kernels if provided
        if self.custom_kernel1:
            params.custom_kernel1 = self.custom_kernel1
        if self.custom_kernel2:
            params.custom_kernel2 = self.custom_kernel2
        
        return params
    
    @classmethod
    def create_gaussian_config(cls, kernel_size: int = 5, sigma: float = 1.5, **kwargs):
        """Convenience method to create config with Gaussian kernels"""
        config = cls(**kwargs)
        config.kernel1_type = silicon_retina_cuda.KERNEL_TYPE.GAUSSIAN.value
        config.kernel2_type = silicon_retina_cuda.KERNEL_TYPE.GAUSSIAN.value
        config.kernel1_size = kernel_size
        config.kernel2_size = kernel_size
        config.kernel1_sigma = sigma
        config.kernel2_sigma = sigma
        return config
    
    @classmethod
    def create_box_config(cls, kernel_size: int = 3, **kwargs):
        """Convenience method to create config with box kernels"""
        config = cls(**kwargs)
        config.kernel1_type = silicon_retina_cuda.KERNEL_TYPE.BOX.value
        config.kernel2_type = silicon_retina_cuda.KERNEL_TYPE.BOX.value
        config.kernel1_size = kernel_size
        config.kernel2_size = kernel_size
        return config


def preprocess_input(img, config):
    """Preprocess input image"""
    if config.apply_log_to_input:
        img = torch.log(255.0 * img + 1e-3) / 5.54126393731521
    return torch.clamp(img, 0.0, 1.0)


class SiliconRetinaVerboseFloatCUDA(torch.nn.Module):
    """
    Silicon Retina model that stores all intermediate layer outputs.
    Useful for debugging and visualization.
    
    Note: Uses more memory as all T timesteps are stored.
    """
    
    def __init__(self, silicon_retina_config: SiliconRetinaConfig, T: int):
        super(SiliconRetinaVerboseFloatCUDA, self).__init__()
        self.width = silicon_retina_config.W
        self.height = silicon_retina_config.H
        self.config = silicon_retina_config
        self.T = T
        
        # Register all intermediate buffers
        self.register_buffer("LPF1", torch.zeros((self.T, self.height, self.width), 
                                                  dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF1_mem", torch.zeros((self.T, self.height, self.width), 
                                                      dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF1", torch.zeros((self.T, self.height, self.width), 
                                                  dtype=torch.float32, device="cuda:0"))
        self.register_buffer("BLURRED2", torch.zeros((self.T, self.height, self.width), 
                                                      dtype=torch.float32, device="cuda:0"))
        self.register_buffer("LPF2", torch.zeros((self.T, self.height, self.width), 
                                                  dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_OPL", torch.zeros((self.T, self.height, self.width), 
                                                   dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_BIP", torch.zeros((self.T, self.height, self.width), 
                                                   dtype=torch.float32, device="cuda:0"))
        self.register_buffer("Q_V_BIP", torch.zeros((self.T, self.height, self.width), 
                                                     dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_BIP_LPF", torch.zeros((self.T, self.height, self.width), 
                                                       dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2_mem", torch.zeros((self.T, self.height, self.width), 
                                                      dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2", torch.zeros((self.T, self.height, self.width), 
                                                  dtype=torch.float32, device="cuda:0"))
        
        self.register_buffer("I_Gang_on", torch.zeros((self.T, self.height, self.width), 
                                                       dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_Gang_off", torch.zeros((self.T, self.height, self.width), 
                                                        dtype=torch.float32, device="cuda:0"))
        
        self.register_buffer("V_LIF_on", torch.zeros((self.T, self.height, self.width), 
                                                      dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_LIF_off", torch.zeros((self.T, self.height, self.width), 
                                                       dtype=torch.float32, device="cuda:0"))
        
        self.register_buffer("fired_on", torch.zeros((self.T, self.height, self.width), 
                                                      dtype=torch.uint8, device="cuda:0"))
        self.register_buffer("fired_off", torch.zeros((self.T, self.height, self.width), 
                                                       dtype=torch.uint8, device="cuda:0"))
        
        silicon_retina_cuda.initialize_kernels(self.config.to_cpp_struct())
        
    
    def forward(self, imgs):
        """
        Process T frames through the retina model.
        
        Args:
            imgs: Input tensor of shape (T, H, W) with dtype float32
            
        Returns:
            Tuple of (fired_on, fired_off) tensors with spike events
        """
        assert imgs.shape[0] == self.T, f"Input tensor must have T={self.T} frames, got {imgs.shape[0]}"
        assert imgs.shape[1] == self.height, f"Input height must be {self.height}, got {imgs.shape[1]}"
        assert imgs.shape[2] == self.width, f"Input width must be {self.width}, got {imgs.shape[2]}"
        assert imgs.dtype == torch.float32, f"Input must be float32, got {imgs.dtype}"
        
        # Preprocess input
        imgs = preprocess_input(imgs, config=self.config)
        
        # Zero all buffers
        self.LPF1.zero_()
        self.HPF1_mem.zero_()
        self.HPF1.zero_()
        self.BLURRED2.zero_()
        self.LPF2.zero_()
        self.I_OPL.zero_()
        self.V_BIP.zero_()
        self.Q_V_BIP.zero_()
        self.V_BIP_LPF.zero_()
        self.HPF2_mem.zero_()
        self.HPF2.zero_()
        self.I_Gang_on.zero_()
        self.I_Gang_off.zero_()
        self.V_LIF_on.zero_()
        self.V_LIF_off.zero_()
        self.fired_on.zero_()
        self.fired_off.zero_()
        
        # Call CUDA kernel
        return silicon_retina_cuda.forward_silicon_retina_verbose(
            imgs,
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
            self.T,
            self.config.to_cpp_struct()
        )


class SiliconRetinaFloatCUDA(torch.nn.Module):
    """
    Memory-efficient Silicon Retina model using double buffering.
    Only outputs spike events, not intermediate layers.
    
    Recommended for production use.
    """
    
    def __init__(self, silicon_retina_config: SiliconRetinaConfig, T: int):
        super(SiliconRetinaFloatCUDA, self).__init__()
        self.width = silicon_retina_config.W
        self.height = silicon_retina_config.H
        self.config = silicon_retina_config
        self.T = T
        
        # Double-buffered state (only 2 time steps)
        self.register_buffer("LPF1", torch.zeros((2, self.height, self.width), 
                                                  dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF1_mem", torch.zeros((2, self.height, self.width), 
                                                      dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF1", torch.zeros((2, self.height, self.width), 
                                                  dtype=torch.float32, device="cuda:0"))
        self.register_buffer("BLURRED2", torch.zeros((2, self.height, self.width), 
                                                      dtype=torch.float32, device="cuda:0"))
        self.register_buffer("LPF2", torch.zeros((2, self.height, self.width), 
                                                  dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_OPL", torch.zeros((2, self.height, self.width), 
                                                   dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_BIP", torch.zeros((2, self.height, self.width), 
                                                   dtype=torch.float32, device="cuda:0"))
        self.register_buffer("Q_V_BIP", torch.zeros((2, self.height, self.width), 
                                                     dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_BIP_LPF", torch.zeros((2, self.height, self.width), 
                                                       dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2_mem", torch.zeros((2, self.height, self.width), 
                                                      dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2", torch.zeros((2, self.height, self.width), 
                                                  dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_Gang_on", torch.zeros((2, self.height, self.width), 
                                                       dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_Gang_off", torch.zeros((2, self.height, self.width), 
                                                        dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_LIF_on", torch.zeros((2, self.height, self.width), 
                                                      dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_LIF_off", torch.zeros((2, self.height, self.width), 
                                                       dtype=torch.float32, device="cuda:0"))
        
        # Output (full time dimension)
        self.register_buffer("fired_on", torch.zeros((self.T, self.height, self.width), 
                                                      dtype=torch.uint8, device="cuda:0"))
        self.register_buffer("fired_off", torch.zeros((self.T, self.height, self.width), 
                                                       dtype=torch.uint8, device="cuda:0"))

        silicon_retina_cuda.initialize_kernels(self.config.to_cpp_struct())
    
    def forward(self, imgs):
        """
        Process T frames through the retina model (memory efficient).
        
        Args:
            imgs: Input tensor of shape (T, H, W) with dtype float32
            
        Returns:
            Tuple of (fired_on, fired_off) tensors with spike events
        """
        assert imgs.shape[0] == self.T, f"Input tensor must have T={self.T} frames, got {imgs.shape[0]}"
        assert imgs.shape[1] == self.height, f"Input height must be {self.height}, got {imgs.shape[1]}"
        assert imgs.shape[2] == self.width, f"Input width must be {self.width}, got {imgs.shape[2]}"
        assert imgs.dtype == torch.float32, f"Input must be float32, got {imgs.dtype}"
        
        # Preprocess input
        imgs = preprocess_input(imgs, config=self.config)
        
        # Zero all buffers
        self.LPF1.zero_()
        self.HPF1_mem.zero_()
        self.HPF1.zero_()
        self.BLURRED2.zero_()
        self.LPF2.zero_()
        self.I_OPL.zero_()
        self.V_BIP.zero_()
        self.Q_V_BIP.zero_()
        self.V_BIP_LPF.zero_()
        self.HPF2_mem.zero_()
        self.HPF2.zero_()
        self.I_Gang_on.zero_()
        self.I_Gang_off.zero_()
        self.V_LIF_on.zero_()
        self.V_LIF_off.zero_()
        self.fired_on.zero_()
        self.fired_off.zero_()
        
        # Call CUDA kernel
        return silicon_retina_cuda.forward_silicon_retina(
            imgs,
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
            self.T,
            self.config.to_cpp_struct()
        )


class SiliconRetinaFloatCUDAOnline(torch.nn.Module):
    """
    Online/streaming version of Silicon Retina model.
    Processes one frame at a time, maintaining internal state.
    
    Use this for real-time processing or when T is very large.
    """
    
    def __init__(self, silicon_retina_config: SiliconRetinaConfig):
        super(SiliconRetinaFloatCUDAOnline, self).__init__()
        self.width = silicon_retina_config.W
        self.height = silicon_retina_config.H
        self.config = silicon_retina_config
        self.current_t = 0
        
        # Double-buffered state
        self.register_buffer("LPF1", torch.zeros((2, self.height, self.width), 
                                                  dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF1_mem", torch.zeros((2, self.height, self.width), 
                                                      dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF1", torch.zeros((2, self.height, self.width), 
                                                  dtype=torch.float32, device="cuda:0"))
        self.register_buffer("BLURRED2", torch.zeros((2, self.height, self.width), 
                                                      dtype=torch.float32, device="cuda:0"))
        self.register_buffer("LPF2", torch.zeros((2, self.height, self.width), 
                                                  dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_OPL", torch.zeros((2, self.height, self.width), 
                                                   dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_BIP", torch.zeros((2, self.height, self.width), 
                                                   dtype=torch.float32, device="cuda:0"))
        self.register_buffer("Q_V_BIP", torch.zeros((2, self.height, self.width), 
                                                     dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_BIP_LPF", torch.zeros((2, self.height, self.width), 
                                                       dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2_mem", torch.zeros((2, self.height, self.width), 
                                                      dtype=torch.float32, device="cuda:0"))
        self.register_buffer("HPF2", torch.zeros((2, self.height, self.width), 
                                                  dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_Gang_on", torch.zeros((2, self.height, self.width), 
                                                       dtype=torch.float32, device="cuda:0"))
        self.register_buffer("I_Gang_off", torch.zeros((2, self.height, self.width), 
                                                        dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_LIF_on", torch.zeros((2, self.height, self.width), 
                                                      dtype=torch.float32, device="cuda:0"))
        self.register_buffer("V_LIF_off", torch.zeros((2, self.height, self.width), 
                                                       dtype=torch.float32, device="cuda:0"))
        
        # Single frame output
        self.register_buffer("fired_on", torch.zeros((1, self.height, self.width), 
                                                      dtype=torch.uint8, device="cuda:0"))
        self.register_buffer("fired_off", torch.zeros((1, self.height, self.width), 
                                                       dtype=torch.uint8, device="cuda:0"))
        
        silicon_retina_cuda.initialize_kernels(self.config.to_cpp_struct())
    
    def reset(self):
        """Reset internal state to process a new sequence"""
        self.current_t = 0
        self.LPF1.zero_()
        self.HPF1_mem.zero_()
        self.HPF1.zero_()
        self.BLURRED2.zero_()
        self.LPF2.zero_()
        self.I_OPL.zero_()
        self.V_BIP.zero_()
        self.Q_V_BIP.zero_()
        self.V_BIP_LPF.zero_()
        self.HPF2_mem.zero_()
        self.HPF2.zero_()
        self.I_Gang_on.zero_()
        self.I_Gang_off.zero_()
        self.V_LIF_on.zero_()
        self.V_LIF_off.zero_()
        self.fired_on.zero_()
        self.fired_off.zero_()
    
    def forward(self, img, t=None):
        """
        Process a single frame.
        
        Args:
            img: Input tensor of shape (1, H, W) with dtype float32
            t: Optional timestep (if None, uses internal counter)
            
        Returns:
            Tuple of (fired_on, fired_off) tensors with spike events
        """
        assert img.shape[0] == 1, f"Input must have batch size 1, got {img.shape[0]}"
        assert img.shape[1] == self.height, f"Input height must be {self.height}, got {img.shape[1]}"
        assert img.shape[2] == self.width, f"Input width must be {self.width}, got {img.shape[2]}"
        assert img.dtype == torch.float32, f"Input must be float32, got {img.dtype}"
        
        # Use provided t or increment internal counter
        if t is None:
            t = self.current_t
            self.current_t += 1
        
        # Preprocess input
        img = preprocess_input(img, config=self.config)
        
        # Call CUDA kernel
        return silicon_retina_cuda.forward_silicon_retina_online(
            img,
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
            t,
            self.config.to_cpp_struct()
        )