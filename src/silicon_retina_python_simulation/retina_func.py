import cv2
from scipy.signal import convolve2d
import argparse
import numpy as np

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

def format_integer_to_binary(x, width):
    return f"{x:0{width}b}"

def apply_lowpass(incoming, LPF_map, a, b):
    if LPF_map is None:
        LPF_map = np.zeros_like(incoming)

    LPF_map_next = a * LPF_map + b * incoming
    return LPF_map_next

def apply_highpass(incoming, map, a, b):
    if map is None:
        map = np.zeros_like(incoming)

    next_map = a * map + b * incoming
    return incoming - next_map, next_map


def apply_convolution(image, kernel):
    """Fast 2D convolution using SciPy (same as OpenCV filter2D)."""
    return convolve2d(image, kernel, mode='same', boundary='symm')


def compute_I_opl(X, a1, a2, a3, b1, b2, b3, K1, K2,
                  map1, map2, map3):
    G_C = apply_convolution(X, K1)
    E_tauC = apply_lowpass(G_C, map1, a1, b1)
    T_wr, map2 = apply_highpass(E_tauC, map2, a2, b2)
    G_s = apply_convolution(T_wr, K2)
    E_tauS = apply_lowpass(G_s, map3, a3, b3)

    S = E_tauS
    C = T_wr
    I_opl = C - S
    return I_opl, E_tauC, map2, E_tauS


# compute the bipolar-level contrast gain control
"""
    C \frac{V_{bip}}{dt} = I_{opl} - g_{A} V_{bip} (7)
    g_{A} = G_{A} * E_{A} * Q(V_{bip})             (8)
    Q(V_{bip}) = g_{A}^{0} + \lambda_{A} V_{bip}^2 (9)

    this will be done using euler method

    V_{bip}(t+\delta t) = V_{bip} + \delta t \frac{V_{bip}}{dt}
"""


def contrast_gain_control(V_bip, I_opl, K3, alpha, a, b,
                          g_A, lambda_A, map, quant_bits):
    # Compute the Q(V_{bip})
    # where alpha = \frac{\delta t}{C}
    Q = g_A + lambda_A * V_bip * V_bip

    # Compute E_A
    map = apply_lowpass(Q, map, a, b)

    # compute the g_{A}
    g_A = apply_convolution(map, K3)

    g_A = np.clip(g_A, 0, 1)
    g_A = scamp5_quantize(g_A, bits=quant_bits)

    new_V_bip = V_bip + alpha * (I_opl - g_A * V_bip)

    return new_V_bip, map


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


def excitatory_current_upon_ganglion_cell(Vbip, N_bias,
                                          Vm, alpha1,
                                          alpha2, Vrest,
                                          threshold):
    Igang = Vbip.copy()
    Igang[Igang < 0] = 0
    Igang += N_bias

    Vm = Vm + alpha2 * (Igang - alpha1 * (Vm - Vrest))

    events = Vm > threshold

    return Vm, events



if __name__ == "__main__":
    def do_experiment(total_bits, fractional_bits):
        print("-" * 64)
        print(f"total_bits {total_bits} fractional_bits {fractional_bits}")
        for i in range(1 << total_bits):
            print(f"{format_integer_to_binary(i, width=total_bits)} -> {integer_to_fixed(i, total_bits=total_bits, frac_bits=fractional_bits)}")
        print("-" * 64)


    do_experiment(total_bits=5, fractional_bits=4)
    do_experiment(total_bits=6, fractional_bits=4)
    do_experiment(total_bits=5, fractional_bits=5)
