import numpy as np
from scipy.signal import convolve2d
import cv2

def scamp5_quantize(alpha, bits):
    return alpha

def integer_to_fixed(value, total_bits, frac_bits):
    integer_bits = total_bits - frac_bits
    ret_val = 0

    start = 2 ** (integer_bits-1)
    i = 1 << (total_bits - 1)

    while i:
        if i & value:
            ret_val += start

        i >>= 1
        start /= 2

    return ret_val

def apply_lowpass1(incoming, LPF_map, a):
    if LPF_map is None:
        LPF_map = np.zeros_like(incoming)

    LPF_map_next = LPF_map + a * (incoming - LPF_map)
    return LPF_map_next

def apply_lowpass2(incoming, LPF_map, a):
    if LPF_map is None:
        LPF_map = np.zeros_like(incoming)

    LPF_map_next = incoming - a * (incoming - LPF_map)
    return LPF_map_next

def apply_highpass1(incoming, map, a):
    if map is None:
        map = np.zeros_like(incoming)

    next_map = apply_lowpass1(incoming, map, a)
    return incoming - next_map, next_map

def apply_highpass2(incoming, map, a):
    if map is None:
        map = np.zeros_like(incoming)

    next_map = apply_lowpass2(incoming, map, a)
    return incoming - next_map, next_map

def apply_convolution(image, kernel):
    """Fast 2D convolution using SciPy (same as OpenCV filter2D)."""
    return convolve2d(image, kernel, mode='same', boundary='symm')


def compute_I_opl(X, a1, a2, a3, K1, K2, map2, apply_lpf1,
                  E_tauC, E_tauS, enable_blur_1=True):

    if enable_blur_1:
        G_C = apply_convolution(X, K1)
    else:
        G_C = X.copy()

    if apply_lpf1:
        E_tauC = apply_lowpass1(G_C, E_tauC, a1)
        T_wr, map2 = apply_highpass1(E_tauC, map2, a2)
    else:
        E_tauC = apply_lowpass2(G_C, E_tauC, a1)
        T_wr, map2 = apply_highpass2(E_tauC, map2, a2)

    G_s = apply_convolution(T_wr, K2)

    if apply_lpf1:
        E_tauS = apply_lowpass1(G_s, E_tauS, a3)
    else:
        E_tauS = apply_lowpass2(G_s, E_tauS, a3)

    S = E_tauS
    C = T_wr
    I_opl = C - S
    return I_opl, E_tauC, map2, E_tauS, G_C, T_wr, G_s


# compute the bipolar-level contrast gain control
"""
    C \frac{V_{bip}}{dt} = I_{opl} - g_{A} V_{bip} (7)
    g_{A} = G_{A} * E_{A} * Q(V_{bip})             (8)
    Q(V_{bip}) = g_{A}^{0} + \lambda_{A} V_{bip}^2 (9)

    this will be done using euler method

    V_{bip}(t+\delta t) = V_{bip} + \delta t \frac{V_{bip}}{dt}
"""

def gaussian_kernel(kernel_size, sigma=1.0):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    # Calculate the Gaussian kernel
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))

    # Normalize the kernel so that the sum is 1
    kernel = kernel / np.sum(kernel)
    return kernel

def contrast_gain_control(V_bip, I_opl, K3, alpha, a,
                          g_0, lambda_A, map, quant_bits, apply_lpf1):
    global Q, g_A, g_A_blurred

    # Compute the Q(V_{bip})
    # where alpha = \frac{\delta t}{C}
    Q = np.clip(g_0 + lambda_A * np.abs(V_bip), -128, 127)

    # Compute E_A
    if apply_lpf1:
        map = apply_lowpass1(Q, map, a)
    else:
        map = apply_lowpass2(Q, map, a)

    # compute the g_{A}
    g_A_blurred = apply_convolution(map, K3)

    g_A = np.clip(g_A_blurred / 128, 0, 1)
    g_A = scamp5_quantize(g_A, bits=quant_bits)

    new_V_bip = V_bip + alpha * (I_opl - g_A * V_bip)

    return new_V_bip, map, Q, g_A


"""
* 
*		The computation here looks like following 
*		the input is I_{gang}
* 
*		du/dt = 1/tau ( R * I_{gang}(t)  - |u(t) - u_{rest}| )
* 
*		and is being solved by euler method 
* 
*		u(t + \delta t) = u(t) + \delta t / tau (R * I_{gang}(t) - |u(t) - u_{rest}|)
* 
*		where the computation will be done as following 
* 
*		u(t) + p_{2} * (p_{1} * I_{gang}(t) - |u(t) - u_{rest}|)
*
*		where p_{2} and p_{1} will represent the 5 bit fixed point values in a for b_1.b_2 b_3 b_4 b_5
"""


def excitatory_current_upon_ganglion_cell(Vbip, I_bias,
                                          Vm, p1, p2,
                                          a5, map,
                                          Vrest,
                                          threshold,
                                          apply_lpf1,
                                          sub):
    if apply_lpf1:
        hpf_out, map = apply_highpass1(Vbip, map, a5)
    else:
        hpf_out, map = apply_highpass2(Vbip, map, a5)

    Igang = hpf_out.copy()
    Igang[Igang < 0] = 0
    Igang += I_bias

    Vm = Vm + p2 * (Igang - p1 * np.abs(Vm - Vrest))

    events = Vm > threshold
    Vm[events] -= sub

    return Vm, events, Igang, map, hpf_out
