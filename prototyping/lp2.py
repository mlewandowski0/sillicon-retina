import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

###############################################################################
# Helper functions
###############################################################################

def iir_order1(current_input, prev_output, b0, a1):
    """
    Replaces the missing iir_order3 from your MATLAB code.
    Based on the usage pattern, it appears to be a first-order IIR:
        y[n] = b0 * current_input + a1 * y[n-1]
    """
    return b0 * current_input + a1 * prev_output

def convolve(frame, kernel):
    """
    Convolve 2D array 'frame' with 2D array 'kernel' using OpenCV's filter2D.
    Since your kernels are symmetric, correlation == convolution for these.
    By default, this uses BORDER_CONSTANT to emulate zero-padding like MATLAB's conv2(...,'same').
    """
    # Convert frame and kernel to float32 if needed:
    frame_f32  = frame.astype(np.float32)
    kernel_f32 = kernel.astype(np.float32)
    
    # ddepth = -1 means "same type as source."
    filtered = cv2.filter2D(frame_f32, ddepth=-1, kernel=kernel_f32, borderType=cv2.BORDER_CONSTANT)
    return filtered

###############################################################################
# Main code
###############################################################################

def retina_simulation():
    #---------------------------------------------------------------------------
    # 1) LOAD DATA (sinusoidal grating or chirp); replicate:
    #    load chirp -> chirp_in
    #    Then do: sinewave3D = 255 * permute(chirp_in, [2, 3, 1])
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    # 3) Define all parameters
    #---------------------------------------------------------------------------
    # Outer Plexiform Layer (OPL) parameters
    lambda_opl = 1.0
    relative_weight = 0.5
    sigma_center = 0.05
    tau_center   = 0.01
    sigma_surround = 0.15
    tau_surround   = 0.01
    undershoot_tau_center = 0.01
    steps = 0.001  # time step

    # For the center temporal filter
    norm1 = 2.0 * tau_center / steps
    norm1_inv = 1.0 / norm1
    ac_1 = -np.exp(-steps / tau_center)
    bc_0 = 1.0 - (-np.exp(-steps / tau_center))  # i.e. 1 - ac_1

    # For the high-pass filter
    norm2 = 2.0 * (undershoot_tau_center / steps)
    norm2_inv = 1.0 / norm2
    a1_1 = -np.exp(-steps / undershoot_tau_center)
    b1_0 = 1.0 - a1_1
    omega_c = 0.5 / norm2

    # For the surround temporal filter
    norm3 = 2.0 * (tau_surround / steps)
    norm3_inv = 1.0 / norm3
    as_1 = -np.exp(-steps / tau_surround)
    bs_0 = 1.0 - (-np.exp(-steps / tau_surround)) # 1 - as_1
    w_s = norm3_inv * relative_weight
    
    w, h = 640, 480

    # OPL state
    x1 = np.zeros((w, h), dtype=np.float32)  # for the high-pass center filter
    y  = np.zeros((w, h), dtype=np.float32)  # used right after x1 * b1_0
    y1 = np.zeros((w, h), dtype=np.float32)  # center low-pass
    y2 = np.zeros((w, h), dtype=np.float32)  # surround low-pass

    # Bipolar layer
    inputNernst_inhibition = 0.0
    g_leak = 50.0      # bipolar leak
    input_amp = 50.0   # linear input amplification
    lambda_amp = 0.0   # adaptation gain
    adaptation_tau = 0.005
    a_0 = 1.0
    a_1 = -(np.exp(-steps / adaptation_tau))
    b_0 = 1.0 - a_1
    preceding_V_bip       = np.zeros((w, h), dtype=np.float32)
    preceding_inhibition  = np.ones((w, h), dtype=np.float32)
    preceding_attenuation = np.ones((w, h), dtype=np.float32)
    lambda_amp_b_0 = lambda_amp * b_0

    # IPL/Ganglion parameters
    refr_mu    = 0.003
    refr_sigma = 0.001
    noise_sigma = 0.1
    g_L = 50.0
    tau = 0.001
    mem_n = noise_sigma * np.sqrt(g_L / tau)

    initialize = 1000.0 * (refr_mu + refr_sigma * np.ones((w, h), dtype=np.float32))
    zeros1     = np.zeros((w, h), dtype=np.float32)
    V = 0.5 + 0.2 * np.zeros((w, h), dtype=np.float32)
    refr  = 1000.0 * (refr_mu + refr_sigma * np.ones((w, h), dtype=np.float32))
    all_spikes = []

    # Nonlinearity parameters
    lambda_G = 100.0
    i_0 = 37.0
    v_0 = 0.0
    sign = 1.0
    transient_tau = 0.02
    norm4 = 2.0 * (transient_tau / steps)
    omega_g = 0.72 / norm4
    transient = np.zeros((w, h), dtype=np.float32)
    x2 = np.zeros((w, h), dtype=np.float32)
    a2_0 = 1.0
    a2_1 = -np.exp(-steps / transient_tau)
    b2_0 = 1.0 - a2_1

    i_0_i = i_0 * i_0

    #---------------------------------------------------------------------------
    # Example small 3x3, 7x7 kernels from your code
    #---------------------------------------------------------------------------
    kernel_center = np.array([
        [0.011660098, 0.08615712,  0.011660098],
        [0.08615712,  0.63661975,  0.08615712 ],
        [0.011660098, 0.08615712,  0.011660098]
    ], dtype=np.float32)

    kernel_surround = np.array([
        [0.0116601,  0.08615712, 0.0116601 ],
        [0.08615712, 0.63661975, 0.08615712],
        [0.0116601,  0.08615712, 0.0116601 ]
    ], dtype=np.float32)

    kernel_bipolar = np.array([
        [1.96519161240319e-05,   0.000239409349497270, 0.00107295826497866, 0.00176900911404382, 0.00107295826497866, 0.000239409349497270, 1.96519161240319e-05],
        [0.000239409349497270,   0.00291660295438644,  0.0130713075831894,  0.0215509428482683,  0.0130713075831894,  0.00291660295438644,  0.000239409349497270],
        [0.00107295826497866,    0.0130713075831894,   0.0585815363306070,  0.0965846250185641,  0.0585815363306070,  0.0130713075831894,   0.00107295826497866],
        [0.00176900911404382,    0.0215509428482683,   0.0965846250185641,  0.159241125690702,   0.0965846250185641,  0.0215509428482683,   0.00176900911404382],
        [0.00107295826497866,    0.0130713075831894,   0.0585815363306070,  0.0965846250185641,  0.0585815363306070,  0.0130713075831894,   0.00107295826497866],
        [0.000239409349497270,   0.00291660295438644,  0.0130713075831894,  0.0215509428482683,  0.0130713075831894,  0.00291660295438644,  0.000239409349497270],
        [1.96519161240319e-05,   0.000239409349497270, 0.00107295826497866, 0.00176900911404382, 0.00107295826497866, 0.000239409349497270, 1.96519161240319e-05]
    ], dtype=np.float32)


    cap = cv2.VideoCapture(0)  # Replace with video path or camera index
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    #---------------------------------------------------------------------------
    # 4) Main loop over time frames
    #---------------------------------------------------------------------------
    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for simplicity (optional, can work with color too)
        frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY).T


        # Center: convolve with kernel_center, then time low-pass
        center_G = lambda_opl * convolve(frame, kernel_center)
        y1 = iir_order1(center_G, y1, bc_0, ac_1)  # first-order IIR
        E_center = y1 * norm1_inv

        # High-pass filter on center
        y = (x1 * b1_0) + (y * a1_1)  # or (y = b1_0*x1 + a1_1*y)
        x1 = E_center
        T_center = E_center - omega_c * y

        # Surround: convolve T_center with kernel_surround, then low-pass
        surround_G = convolve(T_center, kernel_surround)
        y2 = iir_order1(surround_G, y2, bs_0, as_1)
        surround_signal = w_s * y2

        # Center minus surround
        IOPL = T_center - surround_signal

        cv2.imshow("IOPL", cv2.normalize(IOPL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        #-------------------------- Bipolar cells -------------------------------
        total_conductance = g_leak + preceding_inhibition
        attenuation_map = 1.0 + (-steps * total_conductance)

        E_infinity = (input_amp * IOPL) / total_conductance
        V_bip = ((preceding_V_bip - E_infinity) * attenuation_map) + E_infinity

        # A small non-linear feedback
        inhibition = (lambda_amp_b_0 * (preceding_V_bip * preceding_V_bip) -
                      (preceding_inhibition * a_1)) / a_0
        
        # Spatial convolution for the inhibition
        inhibition = convolve(inhibition, kernel_bipolar)

        # Update states
        preceding_V_bip       = V_bip
        preceding_attenuation = attenuation_map
        preceding_inhibition  = inhibition

        # Store Bipolar (or OPL) result in OPL_filter
        #cv2.imshow("IOPL", cv2.normalize(IOPL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

        #------------------ IPL & Ganglion cell spiking ------------------------
        # A transient filter step
        transient = (x2 * b2_0) + (transient * a2_1)
        x2 = V_bip

        # High-pass
        x_ipl = sign * (V_bip - omega_g * transient)

        # Static nonlinearity
        r =  lambda_G * (x_ipl - v_0)

        # Two cases:
        # if x > v0 -> i_0 + r
        # else       -> i_0_i / (i_0 - r)
        IGang = np.where(r > 0, (i_0 + r), (i_0_i / (i_0 - r)))

        # Membrane & spiking
        # For truly random noise each timestep: noise = np.random.randn(w,h)*someSigma
        V = V + (IGang - g_L * V + mem_n) * tau

        # Refractory updates
        refr = refr - 1.0
        # Force V=0 if still in refractory
        V[refr >= 0.5] = 0.0
        # Spikes
        spikes = (V > 1.0)
        # Reset refractory time where spiked
        refr_noise = initialize
        refr[spikes] = refr_noise[spikes]
        # Force negative refr to 0
        refr[refr < 0.0] = 0.0
        # Reset potential to 0 where spiked
        V[spikes] = 0.0

        # Collect spike frames (boolean arrays) if desired
        # all_spikes.append(spikes.astype(np.uint8))  # or float32
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


retina_simulation()
