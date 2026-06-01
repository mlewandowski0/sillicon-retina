/*****************************************************************************
 * Silicon Retina on SCAMP-5 Pixel Processor Array
 *----------------------------------------------------------------------------
 * Companion code to: "Programmable Silicon Retina on Pixel Processor Array"
 *                    Lewandowski et al., CVPR Workshop.
 *
 * Implements the Wohrer & Kornprobst retina model (OPL -> bipolar contrast
 * gain control -> LIF ganglion) using a 2x2 superpixel layout. Each
 * superpixel uses one PE for compute (PE / "PROC") and three for extended
 * storage (STO1, STO2, STO3), giving effectively 24 AREGs per logical pixel.
 *
 *****************************************************************************
 *
 * SUPERPIXEL PHYSICAL LAYOUT
 *
 *     +--------+--------+
 *     |        |        |
 *     |  PE    |  STO1  |     PE   = top-left      (compute & image capture)
 *     | (PROC) |        |     STO1 = top-right     (East  of PE)
 *     |        |        |     STO2 = bottom-left   (South of PE)
 *     +--------+--------+     STO3 = bottom-right  (diagonal, no direct
 *     |        |        |                          neighbor link to PE)
 *     |  STO2  |  STO3  |
 *     |        |        |     Effective resolution: 128 x 128 logical pixels.
 *     |        |        |
 *     +--------+--------+
 *
 * NEWS routing implied by this layout:
 *     PE   <-> STO1  via West/East   (PE writes NEWS; STO1 reads XW)
 *     PE   <-> STO2  via North/South (PE writes NEWS; STO2 reads XN)
 *     STO3 cannot bus directly to/from PE; it would need a relay.
 *
 *****************************************************************************
 *
 * AREG MEMORY MAP  (analog registers; 6 per cell: A,B,C,D,E,F)
 *
 *  PE  (PROC)        scratch / working - clobbered every stage
 *      A,B,C,D       working values (image, OPL signal, V_LIF, ...)
 *      E,F           diva() temporaries and constant loads via scamp5_load_in
 *
 *  STO1   (East of PE)    persistent state, all reads/writes via PE.<->E bus
 *      A             [transient] blur replica of PE.A during gauss()
 *                    (set by replicate_A / fill_A)
 *      B             PREV_FRAME for the 1st HPF (PE.A from previous frame)
 *                    --> read+written by stage_hpf1()
 *      C             LPF1 state (filtered HPF output)
 *                    --> read+written by stage_lpf1()
 *      D             LPF2 state (filtered surround signal E_S)
 *                    --> read+written by stage_lpf2()
 *      E             V_BIP  (bipolar membrane voltage)
 *                    macro BIPOLAR_REGISTER_STO1 = E
 *                    --> read+written by stage_gain_control()
 *      F             (free)
 *
 *  STO2   (South of PE)   persistent state, all reads/writes via PE.<->S bus
 *      A             [transient] blur replica of PE.A during gauss()
 *      B             E_A LPF state (gain-feedback LPF on Q(V_BIP))
 *                    macro BIPOLAR_LPF_REGISTER_STO2 = B
 *                    --> read+written by stage_gain_control()
 *      C             PREV_FRAME for the 2nd HPF (the previous I_OPL frame)
 *                    macro PREV_FRAME_STO2 = C
 *                    --> read+written by stage_hpf2()
 *      D             pos  V_LIF membrane voltage (ganglion LIF state)
 *                    macro MEMBRANE_STO2 = D
 *                    --> read+written by stage_lif_pos()
 *      E             neg V_LIF membrane voltage (ganglion LIF state)
 *                    macro MEMBRANE_STO2 = D
 *                    --> read+written by stage_lif_pos()
 *      F             (free)
 *
 *  STO3   (diagonal)      currently only used as a blur replica target
 *      A,F           [transient] blur replicas of PE.A / PE.F
 *      
 *      B,C,D,E       (free)
 *
 *  Persistent AREG usage : 7 of 18 STO slots (~39%).
 *  AREG free for future  : STO1.F, STO2.E, STO2.F, STO3.{B,C,D,E}.
 *
 *****************************************************************************
 *
 * DREG MAP  (1-bit; 7 per cell: S0..S6, plus FLAG and RM/RF for digital masks)
 *
 *  S0   DREG_PE       superpixel-type mask: 1 only at PE cells
 *  S1   DREG_STO1     superpixel-type mask: 1 only at STO1 cells
 *  S2   DREG_STO2     superpixel-type mask: 1 only at STO2 cells
 *  S3   DREG_STO3     superpixel-type mask: 1 only at STO3 cells
 *
 *  S4   multiplexed scratch (alive only briefly inside one stage at a time):
 *         - I_OPL_SMALLER_THAN_THRESHOLD   (inside stage_amplify_opl)
 *         - N_NEGATIVE                     (inside stage_nonlinearity_N_pos)
 *         - VBIP_GREATER_THAN_64           (inside stage_gain_control)
 *         - FIRING                         (inside stage_lif_pos)
 *
 *  S5   multiplexed scratch:
 *         - VBIP_GREATER_THAN_32           (inside stage_gain_control)
 *         - FIRING_AND_IS_PE               (inside stage_lif_pos; also held until
 *                                           the display block reads it)
 *
 *  S6   VBIP_GREATER_THAN_16               (inside stage_gain_control only)
 *
 *  IMPORTANT: S4/S5/S6 aliases share the same physical DREG. They never
 *  overlap in time within a frame because stage_gain_control fully consumes
 *  the Q-bits before stage_lif_pos starts. Be careful when adding new stages.
 *
 *****************************************************************************
 *
 * PIPELINE (matches Algorithm 1 in the paper; HPF runs before LPF inside the
 * OPL stage thanks to LTI commutation, which is mathematically equivalent)
 *
 *   stage_capture            -- scamp5_get_image(A,F)
 *   stage_blur_input         -- G_C  (optional)
 *   stage_hpf1               -- T_C step (uses STO1.B as prev frame)
 *   stage_lpf1               -- E_C-ish (uses STO1.C as LPF1 state;
 *                                       leaves PE.C cached as T_C)
 *   stage_blur_surround      -- G_S (also runs fill_A internally)
 *   stage_lpf2               -- E_S  (uses STO1.D as LPF2 state)
 *   stage_compute_opl        -- PE.A := T_C - E_S    (sign-corrected)
 *   stage_amplify_opl        -- optional thresholded amplification
 *   stage_gain_control       -- bipolar contrast gain control (off by default
 *                               on hardware: noise accumulation is brutal)
 *   stage_hpf2               -- T_2 HPF (uses STO2.C as prev I_OPL)
 *   stage_nonlinearity_N_pos     -- max(0,x) + I_bias
 *   stage_lif_pos                -- LIF on STO2.D; FIRING_AND_IS_PE = spike map
 *
 * Each stage is gated by its own GUI toggle. Disable stages from the GUI to
 * leave PE.A holding an intermediate value, then enable `display_A` to probe.
 *
 *****************************************************************************
 *
 * SCAMP INSTRUCTION GOTCHAS (from tutorial AREG constraint table)
 *
 *   mov(X,Y)        requires  X != Y
 *   add(X,Y,Z)      requires  Y != Z
 *   sub(X,Y,Z)      requires  X != Z
 *   neg(X,Y)        requires  X != Y
 *   abs(X,Y)        requires  X != Y       <-- abs(F,F) is NOT valid
 *   divq(X,Y)       requires  X != Y
 *   div(X,Y,Z)      requires  X != Y != Z
 *   diva(X,Y,Z)     requires  X != Y != Z  (most accurate divide-by-two)
 *   bus(X,Y)        requires  X != Y       (note: bus INVERTS: X = -Y + offset
 *                                          so two bus operations compose into
 *                                          a copy with offset cancellation)
 *
 *****************************************************************************/

#include <scamp5.hpp>

using namespace SCAMP5_PE;


// ===========================================================================
// COMPILE-TIME ALIASES
// ===========================================================================

// --- Superpixel-type DREG masks (loaded fresh every frame) -----------------
#define DREG_PE   S0    // mask: 1 only at PE   cells (top-left of superpixel)
#define DREG_STO1 S1    // mask: 1 only at STO1 cells (East  of PE)
#define DREG_STO2 S2    // mask: 1 only at STO2 cells (South of PE)
#define DREG_STO3 S3    // mask: 1 only at STO3 cells (diagonal)

// --- Multiplexed scratch DREG aliases --------------------------------------
// Same physical bits, different roles per stage. Lifetimes do NOT overlap
// within a frame (see header). Adding new stages? Re-check the lifetime
// table before reusing S4-S6.
#define I_OPL_SMALLER_THAN_THRESHOLD S4   // stage_amplify_opl
#define N_NEGATIVE                   S4   // stage_nonlinearity_N_pos
#define FIRING                       S4   // stage_lif_pos (raw threshold flag)
#define FIRING_AND_IS_PE             S5   // stage_lif_pos (spike map at PE cells)
                                          //   -- displayed as the event output
#define FIRING_NEG_AND_IS_PE         S6   // stage lif_neg ( spike map at PE cells)

#define VBIP_GREATER_THAN_64 S4   // stage_gain_control (bit-plane for x1/2)
#define VBIP_GREATER_THAN_32 S5   // stage_gain_control (bit-plane for x1/4)
#define VBIP_GREATER_THAN_16 S6   // stage_gain_control (bit-plane for x1/8)
// (Naming uses the paper's threshold convention; actual compare constants
//  inside stage_gain_control are 32/16/8 - half-scale of V_BIP range.)

// --- Persistent AREG slot aliases (in STO cells, across frames) ------------
// Where each piece of persistent state lives. STO1.B/C/D and STO2 slots not
// listed here are referenced inline by register letter; consult the memory
// map in the file header.
#define BIPOLAR_REGISTER_STO1     E   // STO1.E - V_BIP  (bipolar membrane)
#define BIPOLAR_LPF_REGISTER_STO2 B   // STO2.B - E_A LPF state in gain control
#define PREV_FRAME_STO2           C   // STO2.C - previous I_OPL for 2nd HPF
#define MEMBRANE_STO2             D   // STO2.D - V_LIF pos (LIF membrane voltage)
#define MEMBRANE_NEG_STO2         E   // STO2.E - V_LIF neg (LIF membrane voltage) 

#define BIPOLAR_CURRENT_INPUT       A // PE.A stores the bipolar current input 
#define BIPOLAR_CURRENT_INPUT_SAVED A // STO2.A stores the I_{BIP} which is needed for 
                                      // computation of negative stage of LIF


// --- Numeric constants -----------------------------------------------------
#define BLUR_ITERATIONS         1   // gauss() iterations for OPL spatial blurs
#define BLUR_ITERATIONS_BIPOLAR 1   // gauss() iterations for the gain-control blur

// Numeric register IDs used by the debug probe_for() helper (PE.A..F).
#define A_REG_NUM 0
#define B_REG_NUM 1
#define C_REG_NUM 2
#define D_REG_NUM 3
#define E_REG_NUM 4
#define F_REG_NUM 5


// ===========================================================================
// TYPES
// ===========================================================================

// Fixed-point alpha as a binary fraction with `no_bits` bits:
//     alpha = b_1 * 1/2 + b_2 * 1/4 + ... + b_N * 1/2^N      (b_0 implicit 0)
//     alpha = value / 2^no_bits
// Range: [0, 1 - 1/2^no_bits) -- strictly < 1, stable for an IIR LPF.
struct low_pass_filter
{
    int no_bits;
    int value;
};


// ===========================================================================
// TUNABLE PARAMETERS (filter coefficients + scalars)
// ===========================================================================
// These have GUI sliders attached in setup_gui_controls() below.

low_pass_filter LPF1        = {5, 0b10000};   // 1st temporal LPF in OPL
low_pass_filter LPF2        = {5, 0b10000};   // 2nd temporal LPF in OPL (surround)
low_pass_filter LPF3        = {4, 0b1111 };   // E_A LPF inside gain control
low_pass_filter lambda_A    = {5, 0b10000};   // weight on |V_BIP| in Q()
low_pass_filter alpha_fixed = {5, 0b11000};   // Euler step for V_BIP update
low_pass_filter p1          = {5, 0b11000};   // I_gang gain in LIF
low_pass_filter p2          = {5, 0b11000};   // leak  gain in LIF

int g_A                = 0;     // baseline g_A_0 added inside Q(V_BIP)
int amplify_iters      = 0;     // optional |V_BIP| pre-amplification iterations
int amplify_opl        = 0;     // optional post-OPL amplification iterations
int amplify_opl_thresh = 5;     // threshold used by stage_amplify_opl

int u_rest   = -30;
int U_thresh =  30;
int I_bias   =   0;


// ===========================================================================
// GUI-TOGGLEABLE PIPELINE FLAGS
// ===========================================================================
// Default 1 = stage runs every frame. Default 0 = stage disabled (off button
// on the GUI). Toggling these is the primary way to probe intermediate state.

int apply_blur_1                   = 0;   // optional input blur (G_C)
int apply_1st_hpf                  = 1;
int apply_1st_lpf                  = 1;
int apply_2nd_blur                 = 1;
int apply_2nd_lpf                  = 1;
int apply_3rd_lpf                  = 1;   // reserved (no pipeline stage uses it yet)
int compute_opl                    = 1;
int compute_approximation          = 0;   // gain control - off (noisy on real HW)
int compute_blur_for_approximation = 1;
int add_leftovers_to_feedback      = 1;
int compute_feedback_phase2        = 1;
int apply_2nd_hpf                  = 1;
int apply_N_func                   = 1;
int compute_LIF                    = 0;

int display_A   = 0;   // show PE.A (after the pipeline finishes)
int display_evn = 0;   // show FIRING_AND_IS_PE (the spike map)


// ===========================================================================
// FRAME STATE
// ===========================================================================
bool first_pass = true;          // true on the very first frame loop iter
vs_stopwatch frame_timer;


// ===========================================================================
// SUPERPIXEL MASK LOADERS
// ===========================================================================
// Each function writes one of the four superpixel-type masks into a DREG so
// that exactly one of the four cells in every 2x2 superpixel is 1 and the
// other three are 0.
//
// Called once per frame from load_superpixel_masks(). After that every
// WHERE(DREG_*) restricts the next operation to that cell type.
//
// The _into_S0 variants are alternative loaders that put a non-PE mask into
// S0 instead - they exist for historical reasons and are not used by the
// current pipeline.
// ---------------------------------------------------------------------------
const inline void load_is_PE_into_S0()
{
    scamp5_load_pattern(S0, 0b11111110, 0b11111111, 0b11111110, 0b11111110);
}

const inline void load_is_ST1_into_S0()
{
    scamp5_load_pattern(S0, 0b11111110, 0b11111110, 0b11111110, 0b11111110);
}

const inline void load_is_ST1_into_S1()
{
    scamp5_load_pattern(S1, 0b11111110, 0b11111110, 0b11111110, 0b11111110);
}

const inline void load_is_ST2_into_S0()
{
    scamp5_load_pattern(S0, 0b11111111, 0b11111111, 0b11111110, 0b11111110);
}

const inline void load_is_ST2_into_S2()
{
    scamp5_load_pattern(S2, 0b11111111, 0b11111111, 0b11111110, 0b11111110);
}

const inline void load_is_ST3_into_S0()
{
    scamp5_load_pattern(S0, 0b11111111, 0b11111110, 0b11111110, 0b11111110);
}

const inline void load_is_ST3_into_S3()
{
    scamp5_load_pattern(S3, 0b11111111, 0b11111110, 0b11111110, 0b11111110);
}

// Convenience: load all four superpixel-type masks in one call.
// Use at the top of every frame loop iteration (cheap, defends against
// long-term DREG decay).
inline void load_superpixel_masks()
{
    load_is_PE_into_S0();
    load_is_ST1_into_S1();
    load_is_ST2_into_S2();
    load_is_ST3_into_S3();
}


// ===========================================================================
// SUPERPIXEL REPLICATION HELPERS
// ===========================================================================
// Replicate PE.A (or PE.F) into the A (or F) register of all three STO cells
// in the same superpixel. REQUIRED before any gauss(X, X, ...) call, because
// the resistive grid that implements the spatial blur reads the same
// register plane across every physical cell - stale STO values would corrupt
// the blur.
//
// Routing for replicate_A:
//   PE.NEWS   = PE.A   (broadcast)
//   STO1.A    = STO1.XW = PE.NEWS                    (STO1 is East of PE)
//   STO2.A    = STO2.XN = PE.NEWS                    (STO2 is South of PE)
//   STO2.NEWS = STO2.A (broadcast - relay step)
//   STO3.A    = STO3.XW = STO2.NEWS                  (STO3 is East of STO2)
//
// State touched: STO1.A, STO2.A, STO3.A overwritten with PE.A.
//                STO2.NEWS used as a relay.
// WHERE state at exit: ALL() (all cells selected for subsequent ops).
//
// MUST be called inside an active scamp5_kernel_begin/end block - that's
// why we use the scamp5_kernel_api:: namespace.
// ---------------------------------------------------------------------------
const inline void replicate_A()
{
    scamp5_kernel_api::WHERE(DREG_PE);
    scamp5_kernel_api::bus(NEWS, A);

    scamp5_kernel_api::WHERE(DREG_STO1);
    scamp5_kernel_api::bus(A, XW);

    scamp5_kernel_api::WHERE(DREG_STO2);
    scamp5_kernel_api::bus(A, XN);
    scamp5_kernel_api::bus(NEWS, A);   // relay so STO3 (East of STO2) can read it

    scamp5_kernel_api::WHERE(DREG_STO3);
    scamp5_kernel_api::bus(A, XW);

    scamp5_kernel_api::ALL();
}

const inline void replicate_F()
{
    scamp5_kernel_api::WHERE(DREG_PE);
    scamp5_kernel_api::bus(NEWS, F);

    scamp5_kernel_api::WHERE(DREG_STO1);
    scamp5_kernel_api::bus(F, XW);

    scamp5_kernel_api::WHERE(DREG_STO2);
    scamp5_kernel_api::bus(F, XN);
    scamp5_kernel_api::bus(NEWS, F);

    scamp5_kernel_api::WHERE(DREG_STO3);
    scamp5_kernel_api::bus(F, XW);

    scamp5_kernel_api::ALL();
}

// Standalone version of replicate_A wrapped in its own kernel block.
// Use when you want a single self-contained call; use the bare replicate_A
// inside an existing kernel block.
const inline void fill_A()
{
    scamp5_kernel_begin();
    replicate_A();
    scamp5_kernel_end();
}


// ===========================================================================
// FIXED-POINT MULTIPLY-ACCUMULATE (used by the LPF stages in the OPL)
// ===========================================================================
//
// Computes  B := B + alpha * A_in,  destroying PE.A in the process.
//
// alpha is encoded as a binary fraction with no_bits bits:
//     alpha = b_1 * 1/2 + b_2 * 1/4 + ... + b_N * 1/2^N    (b_0 implicit 0)
//     alpha = value / 2^N,   range [0, 1 - 1/2^N).
//
// The CALLER must apply the b_0 = 0 step (an extra `diva(A, E, F)` BEFORE
// invoking this function) so that the loop below starts at the 1/2 weight.
// Without that prefix the effective alpha would have an extra factor of 2.
//
// On entry:  PE.A = multiplicand,  PE.B = accumulator,
//            PE.E and PE.F free (used as diva temporaries),
//            WHERE state = DREG_PE (inherited; loop has no WHERE inside).
// On exit :  PE.B = (old PE.B) + alpha * (old PE.A) ; PE.A destroyed.
// ---------------------------------------------------------------------------
void mac_alpha_A_into_B(low_pass_filter &alpha)
{
    int bit = 1 << (alpha.no_bits - 1);
    while (bit)
    {
        if (bit & alpha.value)
        {
            scamp5_kernel_begin();
            add(B, B, A);     // B += A   (this bit selected, accumulate)
            diva(A, E, F);    // A /= 2   (advance to next bit weight)
            scamp5_kernel_end();
        }
        else
        {
            scamp5_kernel_begin();
            diva(A, E, F);    // A /= 2   (skip accumulation, just halve)
            scamp5_kernel_end();
        }
        bit >>= 1;
    }
}


// ===========================================================================
// PIPELINE STAGES
// ===========================================================================
// Each `stage_*` function is gated by its own GUI flag (e.g. apply_1st_hpf).
// They are called in sequence from the frame loop in main().
//
// CONVENTION:
//   - Each stage manages its own scamp5_kernel_begin/end blocks.
//   - Stages take no parameters; they read GUI flags and AREG/DREG state.
//   - On entry/exit each stage documents which PE registers and STO slots
//     it touches. Read these state tables before reordering or modifying.
//   - WHERE state IS persistent across kernel blocks (and across stages),
//     but every stage that depends on a particular WHERE state explicitly
//     re-asserts it. So stages can be reordered safely (subject to the
//     intended pipeline dependencies described in the file header).
// ===========================================================================

// ---------------------------------------------------------------------------
// stage_capture
//   PE.A = luminance from the photodiode  (PE.F = copy used as backup)
// ---------------------------------------------------------------------------
inline void stage_capture()
{
    scamp5_get_image(A, F);
}

// ---------------------------------------------------------------------------
// stage_blur_input  (optional G_C input blur)
//   In  : PE.A = current input
//   Out : PE.A = blur(PE.A)
//   Why : Spatial smoothing on the raw input before the temporal filters.
//   Note: Off by default (apply_blur_1 = 0) - on real hardware the analog
//         noise tends to outweigh the smoothing benefit at typical SNRs.
// ---------------------------------------------------------------------------
inline void stage_blur_input()
{
    if (!apply_blur_1) return;

    scamp5_kernel_begin();
        replicate_A();                       // ends with ALL()
        SET(R1, R2);
        gauss(A, A, BLUR_ITERATIONS);
    scamp5_kernel_end();
}

// ---------------------------------------------------------------------------
// stage_hpf1   ("1st HPF" - temporal differencing)
//   Computes : PE.A := PE.A - prev_frame    (PE.A from previous call)
//   STATE    : STO1.B  read (prev frame),  written (new prev = current PE.A)
// ---------------------------------------------------------------------------
inline void stage_hpf1()
{
    if (!apply_1st_hpf) return;

    scamp5_kernel_begin();
        WHERE(DREG_STO1);
        bus(NEWS, B);                    // STO1.NEWS = -STO1.B + offset

        WHERE(DREG_PE);
        bus(B, XE);                      // PE.B = STO1.B (previous frame)

        sub(F, A, B);                    // PE.F = current - prev
        mov(B, F);                       // PE.B = diff (PE.A unchanged so far)

        bus(NEWS, A);                    // broadcast original PE.A

        WHERE(DREG_STO1);
        bus(B, XW);                      // STO1.B := current PE.A (new prev)

        WHERE(DREG_PE);
        mov(A, B);                       // PE.A = diff (HPF output)
    scamp5_kernel_end();
}

// ---------------------------------------------------------------------------
// stage_lpf1   ("1st LPF" - IIR low-pass)
//   Update : state := state + alpha * (input - state)
//            (effective alpha = LPF1.value / 2^LPF1.no_bits)
//   STATE  : STO1.C  read+written (LPF1 state)
//   On exit:
//     PE.A = new LPF state           (becomes input to subsequent blur+LPF)
//     PE.C = new LPF state (cached)  (used by stage_compute_opl as "T_C")
//
//   First-pass behaviour: seed the LPF state to the current input.
// ---------------------------------------------------------------------------
inline void stage_lpf1()
{
    if (!apply_1st_lpf) return;

    if (first_pass)
    {
        // Seed: PE.B := PE.A so the running state starts at the input.
        scamp5_kernel_begin();
            WHERE(DREG_PE);
            mov(B, A);
        scamp5_kernel_end();
    }
    else
    {
        // Compute PE.A := (input - state) / 2  with state = STO1.C.
        // The "/2" is the b_0 = 0 step of the alpha encoding (see mac_alpha).
        scamp5_kernel_begin();
            WHERE(DREG_STO1);
            bus(NEWS, C);                // STO1.NEWS = -STO1.C + offset

            WHERE(DREG_PE);
            bus(B, XE);                  // PE.B = STO1.C (prev LPF state)

            sub(A, A, B);                // PE.A = input - state
            diva(A, E, F);               // PE.A /= 2   (alpha b_0 = 0 prefix)
        scamp5_kernel_end();
    }

    // Fixed-point MAC: PE.B := PE.B + (LPF1.value/2^N) * (input - state).
    mac_alpha_A_into_B(LPF1);

    // Persist new LPF state back to STO1.C, and cache it in PE.C so the
    // OPL subtraction can use it as "T_C" after PE.A is overwritten by
    // the surround pipeline.
    scamp5_kernel_begin();
        bus(NEWS, B);

        WHERE(DREG_STO1);
        bus(C, XW);                      // STO1.C := PE.B (new state)

        WHERE(DREG_PE);
        mov(A, B);                       // PE.A = new state
        mov(C, B);                       // PE.C = new state (cached as T_C)
    scamp5_kernel_end();
}

// ---------------------------------------------------------------------------
// stage_blur_surround   (G_S - spatial blur on the LPF1 output)
//   In  : PE.A = LPF1 output (T_C-like)
//   Out : PE.A = blurred LPF1 output (E_S input source)
//   Note: fill_A() is ALWAYS called (independent of apply_2nd_blur) because
//         the next stage may also need PE.A replicated across the superpixel.
// ---------------------------------------------------------------------------
inline void stage_blur_surround()
{
    fill_A();    // always replicate PE.A across the superpixel

    if (!apply_2nd_blur) return;

    scamp5_kernel_begin();
        ALL();
        SET(R1, R2);
        gauss(A, A, BLUR_ITERATIONS);
    scamp5_kernel_end();
}

// ---------------------------------------------------------------------------
// stage_lpf2   ("2nd LPF" - IIR low-pass on the surround signal)
//   Same shape as stage_lpf1 but the persistent state lives in STO1.D.
//   STATE  : STO1.D read+written.
//   On exit: PE.A = E_S (the surround signal).
//   PE.C is NOT modified - it still holds the cached T_C from stage_lpf1.
// ---------------------------------------------------------------------------
inline void stage_lpf2()
{
    if (!apply_2nd_lpf) return;

    if (first_pass)
    {
        scamp5_kernel_begin();
            WHERE(DREG_PE);
            mov(B, A);                   // seed state = current input
        scamp5_kernel_end();
    }
    else
    {
        scamp5_kernel_begin();
            WHERE(DREG_STO1);
            bus(NEWS, D);                // STO1.NEWS = -STO1.D + offset

            WHERE(DREG_PE);
            bus(B, XE);                  // PE.B = STO1.D (prev state)

            sub(A, A, B);                // PE.A = input - state
            diva(A, E, F);               // alpha b_0 = 0 prefix
        scamp5_kernel_end();
    }

    mac_alpha_A_into_B(LPF2);

    scamp5_kernel_begin();
        bus(NEWS, B);

        WHERE(DREG_STO1);
        bus(D, XW);                      // STO1.D := PE.B (new state)

        WHERE(DREG_PE);
        mov(A, B);                       // PE.A = E_S (surround signal)
    scamp5_kernel_end();
}

// ---------------------------------------------------------------------------
// stage_compute_opl   (I_OPL = T_C - E_S, sign-corrected)
//   On entry: PE.A = E_S (LPF2 output), PE.C = T_C (cached by stage_lpf1).
//   We want  PE.A := T_C - E_S = PE.C - PE.A.
//
//   SCAMP constraint: sub(X,Y,Z) requires X != Z. So we cannot write
//   sub(A, C, A) directly. Workaround:
//       sub(A, A, C);   // PE.A = E_S - T_C    (wrong sign)
//       neg(C, A);      // PE.C = T_C - E_S    (correct sign; clobbers PE.C)
//       mov(A, C);      // PE.A = I_OPL
//   PE.C is no longer needed after this stage, so reusing it as scratch
//   is safe.
// ---------------------------------------------------------------------------
inline void stage_compute_opl()
{
    if (!compute_opl) return;

    scamp5_kernel_begin();
        WHERE(DREG_PE);
        sub(A, A, C);     // PE.A = E_S - T_C     (wrong sign)
        neg(C, A);        // PE.C = T_C - E_S     (correct sign)
        mov(A, C);        // PE.A = I_OPL
    scamp5_kernel_end();
}

// ---------------------------------------------------------------------------
// stage_amplify_opl   (optional thresholded post-OPL amplification)
//   Debug/tuning knob, off by default.
//
//   For pixels with |PE.A| > amplify_opl_thresh, double PE.A `amplify_opl`
//   times. For pixels at or below the threshold, force PE.A := 0.
//   This emphasises strong OPL signals while suppressing noise.
//
//   STATE TOUCHED: PE.A (modified), PE.B and PE.F (scratch),
//                  DREG S4 (I_OPL_SMALLER_THAN_THRESHOLD).
// ---------------------------------------------------------------------------
inline void stage_amplify_opl()
{
    if (amplify_opl <= 0) return;

    // PE.B := |PE.A|
    scamp5_kernel_begin();
        abs(F, A);
        WHERE(DREG_PE);
            mov(B, F);
    scamp5_kernel_end();

    // PE.B := |PE.A| - threshold; record which PEs are BELOW threshold.
    scamp5_load_in(F, amplify_opl_thresh);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
            sub(B, B, F);

        where(B);                                                   // FLAG = (B > 0)
        NOT(I_OPL_SMALLER_THAN_THRESHOLD, FLAG);                    // S4 = (B <= 0)
        AND(I_OPL_SMALLER_THAN_THRESHOLD,
            I_OPL_SMALLER_THAN_THRESHOLD, DREG_PE);                 // S4 &= PE-only

        WHERE(DREG_PE);
    scamp5_kernel_end();

    // Double PE.A `amplify_opl` times in the PEs above threshold.
    for (int i = 0; i < amplify_opl; ++i)
    {
        scamp5_kernel_begin();
            mov(F, A);
            add(A, A, F);
        scamp5_kernel_end();
    }

    // Zero out the below-threshold PEs.
    scamp5_load_in(F, 0);
    scamp5_kernel_begin();
        WHERE(I_OPL_SMALLER_THAN_THRESHOLD);
            mov(A, F);
    scamp5_kernel_end();
}

// ---------------------------------------------------------------------------
// stage_gain_control  (Contrast Gain Control, the "bipolar" / feedback layer)
//
//   One Euler step of:
//       V_BIP <- V_BIP + alpha * (I_OPL - g_A * V_BIP)
//   with g_A ~ blur( LPF3( g_A_0 + lambda_A * |V_BIP| ) ).
//
//   PERSISTENT STATE TOUCHED:
//     STO1.E  (V_BIP)             : read at start, rewritten at the very end
//     STO2.B  (E_A LPF state)     : read mid-function, rewritten mid-function
//
//   SCRATCH DREGS:
//     S4 (VBIP_GREATER_THAN_64)   bit-plane for x1/2 weight in Q decomposition
//     S5 (VBIP_GREATER_THAN_32)   bit-plane for x1/4 weight
//     S6 (VBIP_GREATER_THAN_16)   bit-plane for x1/8 weight
//
//   PE REGISTER LIFETIMES (also restated at each section boundary below):
//     PE.A : holds I_OPL on entry; later becomes I_OPL - g_A*V_BIP;
//            finally V_BIP_new (written back to STO1.E)
//     PE.B : accumulator; ends each section holding the latest derived signal
//     PE.C : intermediate Q-function value; later scratch
//     PE.D : V_BIP (loaded at the top; preserved until the very end)
//     PE.E,F : diva temporaries and scamp5_load_in targets
//
//   On entry:  PE.A = I_OPL
//   On exit :  PE.A = V_BIP_new   (and STO1.E updated)
//              PE.B,C,D destroyed
// ---------------------------------------------------------------------------
void stage_gain_control()
{
    if (!compute_approximation) return;

    // ---- Section 1: zero accumulator PE.B ---------------------------------
    // PE.B will accumulate (g_A_0 + lambda_A * |V_BIP|) in stages.
    scamp5_load_in(F, 0);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        mov(B, F);
    scamp5_kernel_end();

    // ---- Section 2: load V_BIP from STO1.E into PE.D, compute |V_BIP| ----
    // After this block:
    //   PE.D = V_BIP   (preserved through most of the function)
    //   PE.C = |V_BIP|
    //   PE.F = |V_BIP|  (was just used as a copy register)
    scamp5_kernel_begin();
        WHERE(DREG_STO1);
        bus(NEWS, BIPOLAR_REGISTER_STO1);   // STO1.NEWS = -STO1.E + offset

        WHERE(DREG_PE);
        bus(D, XE);                         // PE.D = STO1.E = V_BIP
        mov(F, D);                          // PE.F = V_BIP

        abs(F, F);                          // PE.F = |V_BIP|
        WHERE(DREG_PE);
        mov(C, F);                          // PE.C = |V_BIP|
    scamp5_kernel_end();

    // ---- Section 3: optional |V_BIP| pre-amplification ------------------
    // PE.C := PE.C * 2^amplify_iters
    for (int i = 0; i < amplify_iters; ++i)
    {
        scamp5_kernel_begin();
            WHERE(DREG_PE);
            mov(F, C);
            add(C, C, F);                   // PE.C = 2 * PE.C
        scamp5_kernel_end();
    }

    // ---- Section 4: PE.B := lambda_A * |V_BIP|  (fixed-point MAC) -------
    // Same MAC pattern as mac_alpha_A_into_B but with multiplicand in C.
    // After this loop:
    //   PE.B = lambda_A * |V_BIP|
    //   PE.C = |V_BIP| / 2^N        (mostly garbage, overwritten next)
    {
        int bit = 1 << (lambda_A.no_bits - 1);
        while (bit)
        {
            if (bit & lambda_A.value)
            {
                scamp5_kernel_begin();
                add(B, B, C);
                diva(C, E, F);
                scamp5_kernel_end();
            }
            else
            {
                scamp5_kernel_begin();
                diva(C, E, F);
                scamp5_kernel_end();
            }
            bit >>= 1;
        }
    }

    // ---- Section 5: PE.C := g_A_0 + lambda_A * |V_BIP|  ( == Q(V_BIP) ) -
    scamp5_load_in(F, g_A);                 // F = g_A_0 in every cell
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        add(C, B, F);                       // PE.C = Q(V_BIP)
    scamp5_kernel_end();

    // State now:
    //   PE.A : I_OPL                              (untouched since entry)
    //   PE.B : lambda_A * |V_BIP|
    //   PE.C : g_A_0 + lambda_A * |V_BIP|   == Q(V_BIP)
    //   PE.D : V_BIP                              (untouched since section 2)

    // ---- Section 6: LPF3 step on Q  (the E_A LPF state in STO2.B) -------
    // E_A := E_A + (LPF3/2) * (Q - E_A)
    //   First-pass : E_A := Q   (initialize state)
    //   Otherwise  : PE.B = previous E_A loaded from STO2.B
    if (first_pass)
    {
        scamp5_kernel_begin();
            WHERE(DREG_PE);
            mov(B, C);                      // PE.B := Q (initialize state)
        scamp5_kernel_end();
    }
    else
    {
        scamp5_kernel_begin();
            WHERE(DREG_STO2);
            bus(NEWS, BIPOLAR_LPF_REGISTER_STO2);   // STO2.NEWS = -STO2.B + offset

            WHERE(DREG_PE);
            bus(B, XS);                              // PE.B = STO2.B = prev E_A
        scamp5_kernel_end();
    }

    // PE.C := (Q - E_A) / 2  -- the alpha b_0 = 0 prefix.
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        sub(C, C, B);                       // PE.C = Q - E_A
        diva(C, E, F);                      // PE.C /= 2
    scamp5_kernel_end();

    // PE.B += LPF3 * PE.C   (MAC with multiplicand in C, accumulator in B)
    {
        int bit = 1 << (LPF3.no_bits - 1);
        while (bit)
        {
            if (bit & LPF3.value)
            {
                scamp5_kernel_begin();
                add(B, B, C);
                diva(C, E, F);
                scamp5_kernel_end();
            }
            else
            {
                scamp5_kernel_begin();
                diva(C, E, F);
                scamp5_kernel_end();
            }
            bit >>= 1;
        }
    }

    // ---- Section 7: persist new E_A state back to STO2.B ----------------
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        bus(NEWS, B);                       // PE.NEWS = -PE.B + offset

        WHERE(DREG_STO2);
        bus(BIPOLAR_LPF_REGISTER_STO2, XN); // STO2.B := PE.B
    scamp5_kernel_end();

    // ---- Section 8 (optional): spatially blur PE.B (approximates G_A *) -
    // Replicate PE.B across the superpixel via PE.F so gauss(F,F,...) works
    // on a coherent plane, then copy the blurred result back to PE.B.
    if (compute_blur_for_approximation)
    {
        scamp5_kernel_begin();
            WHERE(DREG_PE);
            mov(F, B);                      // PE.F = E_A
            replicate_F();                  // STO*.F := PE.F (for gauss)

            SET(R1, R2);
            gauss(F, F, BLUR_ITERATIONS_BIPOLAR);
        scamp5_kernel_end();

        scamp5_kernel_begin();
            mov(B, F);                      // PE.B = blurred E_A = g_A (approx)
        scamp5_kernel_end();
    }

    // ---- Section 9: 3-bit ADC of PE.B  (decompose into Q_BIT_* planes) ---
    // At this point:
    //   PE.A : I_OPL                              (since entry)
    //   PE.B : g_A approximation = G_A * E_A * Q(V_BIP)
    //   PE.C : garbage
    //   PE.D : V_BIP                              (since section 2)
    //
    // We decompose PE.B into three 1-bit planes via "subtract-and-test":
    //   bit_half     (S4) := (B > 32) then B -= 32
    //   bit_quarter  (S5) := (B > 16) then B -= 16
    //   bit_eighth   (S6) := (B >  8) then B -=  8
    //
    // The compare constants are HALF the values suggested by the GT_64/32/16
    // names because PE.B is at half-scale of V_BIP's range here. Decoded
    // weights in section 10 are 1/2, 1/4, 1/8.
    scamp5_load_in(F, 32);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        sub(C, B, F);
        where(C);                                       // FLAG = (B > 32)
        MOV(VBIP_GREATER_THAN_64, FLAG);
        AND(VBIP_GREATER_THAN_64, VBIP_GREATER_THAN_64, DREG_PE);

        WHERE(VBIP_GREATER_THAN_64);
        sub(B, B, F);
    scamp5_kernel_end();

    scamp5_load_in(F, 16);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        sub(C, B, F);
        where(C);                                       // FLAG = (B > 16)
        MOV(VBIP_GREATER_THAN_32, FLAG);
        AND(VBIP_GREATER_THAN_32, VBIP_GREATER_THAN_32, DREG_PE);

        WHERE(VBIP_GREATER_THAN_32);
        sub(B, B, F);
    scamp5_kernel_end();

    scamp5_load_in(F, 8);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        sub(C, B, F);
        where(C);                                       // FLAG = (B > 8)
        MOV(VBIP_GREATER_THAN_16, FLAG);
        AND(VBIP_GREATER_THAN_16, VBIP_GREATER_THAN_16, DREG_PE);

        WHERE(VBIP_GREATER_THAN_16);
        sub(B, B, F);
    scamp5_kernel_end();

    // ---- Section 10: PE.B := g_A * V_BIP  (using the 3 bit-planes) -------
    // Build B := sum_i (bit_i * V_BIP / 2^i) using PE.C as the running
    // halved-V_BIP.
    //
    // Two variants depending on the GUI toggle:
    //   add_leftovers_to_feedback = 1 -> keep B's residue (slightly noisier
    //                                   but closer to a higher-bit approx)
    //   add_leftovers_to_feedback = 0 -> zero B first (clean reconstruction)
    if (add_leftovers_to_feedback)
    {
        scamp5_kernel_begin();
            WHERE(DREG_PE);
            mov(C, D);                   // PE.C = V_BIP

            diva(C, E, F);               // PE.C = V_BIP / 2
            WHERE(VBIP_GREATER_THAN_64);
            add(B, B, C);

            WHERE(DREG_PE);
            diva(C, E, F);               // PE.C = V_BIP / 4
            WHERE(VBIP_GREATER_THAN_32);
            add(B, B, C);

            WHERE(DREG_PE);
            diva(C, E, F);               // PE.C = V_BIP / 8
            WHERE(VBIP_GREATER_THAN_16);
            add(B, B, C);
        scamp5_kernel_end();
    }
    else
    {
        scamp5_load_in(F, 0);
        scamp5_kernel_begin();
            WHERE(DREG_PE);
            mov(B, F);                   // PE.B = 0

            mov(C, D);                   // PE.C = V_BIP
            diva(C, E, F);               // PE.C = V_BIP / 2
            WHERE(VBIP_GREATER_THAN_64);
            add(B, B, C);

            WHERE(DREG_PE);
            diva(C, E, F);               // PE.C = V_BIP / 4
            WHERE(VBIP_GREATER_THAN_32);
            add(B, B, C);

            WHERE(DREG_PE);
            diva(C, E, F);               // PE.C = V_BIP / 8
            WHERE(VBIP_GREATER_THAN_16);
            add(B, B, C);
        scamp5_kernel_end();
    }

    // ---- Section 11: Euler step  V_BIP <- V_BIP + alpha * (I_OPL - g_A*V_BIP)
    //   A := I_OPL - g_A * V_BIP
    //   B := alpha_fixed * A   (MAC, with the b_0 = 0 diva done implicitly by
    //                          the b_0=0 encoding; here we don't need the
    //                          stability prefix because alpha_fixed is a gain
    //                          coefficient written by the user, not an LPF.
    //                          The MAC loop below is a direct multiplication.)
    //   A := V_BIP + B   (V_BIP currently in PE.D)
    //   STO1.E := A      (persist new V_BIP)
    if (compute_feedback_phase2)
    {
        scamp5_kernel_begin();
            WHERE(DREG_PE);
            sub(A, A, B);                   // PE.A = I_OPL - g_A * V_BIP
        scamp5_kernel_end();

        // PE.B := 0
        scamp5_load_in(F, 0);
        scamp5_kernel_begin();
            WHERE(DREG_PE);
            mov(B, F);
        scamp5_kernel_end();

        // PE.B += alpha_fixed * PE.A   (this MAC uses A as multiplicand,
        // matching the convention of mac_alpha_A_into_B).
        mac_alpha_A_into_B(alpha_fixed);

        // Persist new V_BIP back to STO1.E.
        //
        // NOTE (likely latent bug in the original code, preserved here):
        //   The line `add(A, A, D)` adds the (mostly-zero, repeatedly halved)
        //   PE.A residue to PE.D = V_BIP, so PE.A ends up approximately equal
        //   to V_BIP rather than the intended V_BIP + alpha*(I_OPL - g_A*V_BIP).
        //   The intended computation appears to want `add(A, B, D)` since PE.B
        //   holds the scaled increment after mac_alpha_A_into_B(alpha_fixed).
        //   This path is gated by compute_approximation (default OFF on real
        //   HW because of noise), so the bug rarely matters in practice; not
        //   changing it during this refactor to avoid altering behavior.
        scamp5_kernel_begin();
            WHERE(DREG_PE);
                add(A, A, D);
                bus(NEWS, A);

            WHERE(DREG_STO1);
                bus(BIPOLAR_REGISTER_STO1, XW);   // STO1.E := PE.A
        scamp5_kernel_end();
    }
}

// ---------------------------------------------------------------------------
// stage_hpf2   ("2nd HPF" - T_2 temporal differencing on the I_OPL signal)
//   Mirror of stage_hpf1 but uses STO2.C (south neighbor) as the prev frame
//   instead of STO1.B.
//   STATE  : STO2.C  read (prev I_OPL),  written (new prev = current PE.A)
//   On exit: PE.A = current - previous
// ---------------------------------------------------------------------------
inline void stage_hpf2()
{
    if (!apply_2nd_hpf) return;

    scamp5_kernel_begin();
        WHERE(DREG_STO2);
        bus(NEWS, PREV_FRAME_STO2);          // STO2.NEWS = -STO2.C + offset

        WHERE(DREG_PE);
        bus(B, XS);                          // PE.B = STO2.C (prev I_OPL)

        sub(F, A, B);                        // PE.F = current - prev
        mov(B, F);                           // PE.B = diff

        bus(NEWS, A);                        // broadcast original PE.A

        WHERE(DREG_STO2);
        bus(PREV_FRAME_STO2, XN);            // STO2.C := current PE.A

        WHERE(DREG_PE);
        mov(A, B);                           // PE.A = HPF output
    scamp5_kernel_end();
}

// The input to the LIF will be stored in the PE.A up to this point
// We need to copy this input and we store it in the STO2.E : 
// we will require it for the the negative part of the computation 
//
//  Basically the plan of action : 
//  1. stage 1 -> 2 -> end result stored in PE.A 
//  2. move PE.A to STO2.E
//  3. compute the positive part of the LIF and generate events etc. 
//  4. after that is done, return back the STO2.E to PE.A 
//  5. invert it
//  6. pretty much compute the same pipeline. 


inline void store_I_bip()
{
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        bus(NEWS, BIPOLAR_CURRENT_INPUT);     // PE.NEWS = PE.A (I_bip)

        WHERE(DREG_STO2);
        bus(BIPOLAR_CURRENT_INPUT_SAVED, XN); // STO2.E := PE.A
    scamp5_kernel_end();
}

// ---------------------------------------------------------------------------
// stage_nonlinearity_N_pos   (N(x) = max(0, x) + I_bias)
//   Half-wave rectification followed by a bias add.
//
//   STATE TOUCHED: PE.A (modified), PE.F (scratch),
//                  DREG S4 (N_NEGATIVE - the rectification mask).
// ---------------------------------------------------------------------------
inline void stage_nonlinearity_N_pos()
{
    if (!apply_N_func) return;

    scamp5_load_in(F, 0);

    scamp5_kernel_begin();
        where(A);                            // FLAG = (PE.A > 0)
        NOT(N_NEGATIVE, FLAG);               // N_NEGATIVE = (PE.A <= 0)
        AND(N_NEGATIVE, N_NEGATIVE, DREG_PE);// limit to PE cells

        WHERE(N_NEGATIVE);
        mov(A, F);                           // PE.A = 0 where non-positive
    scamp5_kernel_end();

    scamp5_load_in(F, I_bias);

    scamp5_kernel_begin();
        WHERE(DREG_PE);
        add(A, A, F);                        // PE.A += I_bias
    scamp5_kernel_end();
}

// ---------------------------------------------------------------------------
// stage_lif_pos   (Leaky Integrate & Fire ganglion neuron)
//
//   One Euler step:
//       V_LIF <- V_LIF + p1 * I_gang - p2 * |V_LIF - u_rest|
//       spike := (V_LIF > U_thresh)
//       if spike: V_LIF -= U_thresh   (subtract-threshold reset)
//
//   STATE TOUCHED:
//     STO2.D (V_LIF / MEMBRANE_STO2)  : read at start, rewritten at end
//     PE.A,B,C,D,E,F                  : all clobbered as scratch
//     DREG S4 (FIRING)                : raw spike flag (transient)
//     DREG S5 (FIRING_AND_IS_PE)      : spike map restricted to PE cells;
//                                        held until the display block reads it
//
//   On entry: PE.A = I_gang (output of stage_nonlinearity_N_pos).
//   On exit : PE.A = post-reset V_LIF ; STO2.D updated; S5 = spike map.
// ---------------------------------------------------------------------------
void stage_lif_pos()
{
    if (!compute_LIF) return;

    // ---- PE.C := p1 * PE.A  (MAC with multiplicand in A, accumulator in C)
    // No b_0 = 0 prefix here: this is a direct gain multiplication, not an
    // IIR LPF. p1 is supplied with the desired weight directly.
    scamp5_load_in(F, 0);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        mov(C, F);                           // PE.C = 0 (accumulator)
    scamp5_kernel_end();

    {
        int bit = 1 << (p1.no_bits - 1);
        while (bit)
        {
            if (bit & p1.value)
            {
                scamp5_kernel_begin();
                add(C, C, A);
                diva(A, E, F);
                scamp5_kernel_end();
            }
            else
            {
                scamp5_kernel_begin();
                diva(A, E, F);
                scamp5_kernel_end();
            }
            bit >>= 1;
        }
    }

    scamp5_kernel_begin();
        mov(A, C);                           // PE.A = p1 * I_gang
    scamp5_kernel_end();

    // ---- PE.B := V_LIF  (load from STO2.D via south neighbor) ----------
    scamp5_kernel_begin();
        WHERE(DREG_STO2);
        bus(NEWS, MEMBRANE_STO2);            // STO2.NEWS = -STO2.D + offset

        WHERE(DREG_PE);
        bus(B, XS);                          // PE.B = STO2.D = V_LIF
    scamp5_kernel_end();

    // ---- PE.C := |V_LIF - u_rest|  (uses PE.F as temp) -----------------
    scamp5_load_in(F, u_rest);
    scamp5_kernel_begin();
        sub(C, B, F);                        // PE.C = V_LIF - u_rest
        mov(F, C);
        abs(F, F);                           // PE.F = |V_LIF - u_rest|

        WHERE(DREG_PE);
        mov(C, F);                           // PE.C = |V_LIF - u_rest|
    scamp5_kernel_end();

    // ---- PE.D := p2 * |V_LIF - u_rest|  (MAC: multiplicand C, accumulator D)
    scamp5_load_in(F, 0);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        mov(D, F);                           // PE.D = 0
    scamp5_kernel_end();

    {
        int bit = 1 << (p2.no_bits - 1);
        while (bit)
        {
            if (bit & p2.value)
            {
                scamp5_kernel_begin();
                add(D, D, C);
                diva(C, E, F);
                scamp5_kernel_end();
            }
            else
            {
                scamp5_kernel_begin();
                diva(C, E, F);
                scamp5_kernel_end();
            }
            bit >>= 1;
        }
    }

    // ---- Combine: V_LIF_new = V_LIF + p1*I_gang - p2 * |V_LIF - u_rest| --
    // PE.A : p1 * I_gang
    // PE.B : V_LIF
    // PE.D : p2 * |V_LIF - u_rest|
    scamp5_kernel_begin();
        add(A, A, B);                        // PE.A = p1*I_gang + V_LIF
        sub(A, A, D);                        // PE.A = V_LIF_new
    scamp5_kernel_end();

    // ---- Threshold + subtract-threshold reset --------------------------
    scamp5_load_in(F, U_thresh);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        sub(B, A, F);                                  // PE.B = V_LIF_new - threshold

        where(B);                                      // FLAG = (PE.B > 0)
        MOV(FIRING, FLAG);                             // S4 = raw spike flag
        AND(FIRING_AND_IS_PE, FIRING, DREG_PE);        // S5 = spike map at PEs

        WHERE(FIRING_AND_IS_PE);
        mov(A, B);                                     // V_LIF -= threshold on spike
    scamp5_kernel_end();

    // ---- Persist new V_LIF back to STO2.D ------------------------------
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        bus(NEWS, A);                                  // PE.NEWS = -PE.A + offset

        WHERE(DREG_STO2);
        bus(MEMBRANE_STO2, XN);                        // STO2.D := PE.A
    scamp5_kernel_end();
}


inline void restore_I_bip_and_prepare()
{
    scamp5_kernel_begin();
        WHERE(DREG_STO2);
        bus(NEWS, BIPOLAR_CURRENT_INPUT_SAVED);     // PE.NEWS = PE.A (I_bip)

        WHERE(DREG_PE);
        bus(BIPOLAR_CURRENT_INPUT, XS); // STO2.E := PE.A

        // now that we have the current back ,we have to invert it
        // and save it to to A 
        neg(C, BIPOLAR_CURRENT_INPUT);
        mov(BIPOLAR_CURRENT_INPUT, C);
    scamp5_kernel_end();
}


// ---------------------------------------------------------------------------
// stage_lif_pos   (Leaky Integrate & Fire ganglion neuron)
//
//   One Euler step:
//       V_LIF <- V_LIF + p1 * I_gang - p2 * |V_LIF - u_rest|
//       spike := (V_LIF > U_thresh)
//       if spike: V_LIF -= U_thresh   (subtract-threshold reset)
//
//   STATE TOUCHED:
//     STO2.D (V_LIF / MEMBRANE_STO2)  : read at start, rewritten at end
//     PE.A,B,C,D,E,F                  : all clobbered as scratch
//     DREG S4 (FIRING)                : raw spike flag (transient)
//     DREG S5 (FIRING_AND_IS_PE)      : spike map restricted to PE cells;
//                                        held until the display block reads it
//
//   On entry: PE.A = I_gang (output of stage_nonlinearity_N_pos).
//   On exit : PE.A = post-reset V_LIF ; STO2.D updated; S5 = spike map.
// ---------------------------------------------------------------------------
void stage_lif_pos()
{
    if (!compute_LIF) return;

    // ---- PE.C := p1 * PE.A  (MAC with multiplicand in A, accumulator in C)
    // No b_0 = 0 prefix here: this is a direct gain multiplication, not an
    // IIR LPF. p1 is supplied with the desired weight directly.
    scamp5_load_in(F, 0);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        mov(C, F);                           // PE.C = 0 (accumulator)
    scamp5_kernel_end();

    {
        int bit = 1 << (p1.no_bits - 1);
        while (bit)
        {
            if (bit & p1.value)
            {
                scamp5_kernel_begin();
                add(C, C, A);
                diva(A, E, F);
                scamp5_kernel_end();
            }
            else
            {
                scamp5_kernel_begin();
                diva(A, E, F);
                scamp5_kernel_end();
            }
            bit >>= 1;
        }
    }

    scamp5_kernel_begin();
        mov(A, C);                           // PE.A = p1 * I_gang
    scamp5_kernel_end();

    // ---- PE.B := V_LIF  (load from STO2.D via south neighbor) ----------
    scamp5_kernel_begin();
        WHERE(DREG_STO2);
        bus(NEWS, MEMBRANE_STO2);            // STO2.NEWS = -STO2.D + offset

        WHERE(DREG_PE);
        bus(B, XS);                          // PE.B = STO2.D = V_LIF
    scamp5_kernel_end();

    // ---- PE.C := |V_LIF - u_rest|  (uses PE.F as temp) -----------------
    scamp5_load_in(F, u_rest);
    scamp5_kernel_begin();
        sub(C, B, F);                        // PE.C = V_LIF - u_rest
        mov(F, C);
        abs(F, F);                           // PE.F = |V_LIF - u_rest|

        WHERE(DREG_PE);
        mov(C, F);                           // PE.C = |V_LIF - u_rest|
    scamp5_kernel_end();

    // ---- PE.D := p2 * |V_LIF - u_rest|  (MAC: multiplicand C, accumulator D)
    scamp5_load_in(F, 0);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        mov(D, F);                           // PE.D = 0
    scamp5_kernel_end();

    {
        int bit = 1 << (p2.no_bits - 1);
        while (bit)
        {
            if (bit & p2.value)
            {
                scamp5_kernel_begin();
                add(D, D, C);
                diva(C, E, F);
                scamp5_kernel_end();
            }
            else
            {
                scamp5_kernel_begin();
                diva(C, E, F);
                scamp5_kernel_end();
            }
            bit >>= 1;
        }
    }

    // ---- Combine: V_LIF_new = V_LIF + p1*I_gang - p2 * |V_LIF - u_rest| --
    // PE.A : p1 * I_gang
    // PE.B : V_LIF
    // PE.D : p2 * |V_LIF - u_rest|
    scamp5_kernel_begin();
        add(A, A, B);                        // PE.A = p1*I_gang + V_LIF
        sub(A, A, D);                        // PE.A = V_LIF_new
    scamp5_kernel_end();

    // ---- Threshold + subtract-threshold reset --------------------------
    scamp5_load_in(F, U_thresh);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        sub(B, A, F);                                  // PE.B = V_LIF_new - threshold

        where(B);                                      // FLAG = (PE.B > 0)
        MOV(FIRING, FLAG);                             // S4 = raw spike flag
        AND(FIRING_AND_IS_PE, FIRING, DREG_PE);        // S5 = spike map at PEs

        WHERE(FIRING_AND_IS_PE);
        mov(A, B);                                     // V_LIF -= threshold on spike
    scamp5_kernel_end();

    // ---- Persist new V_LIF back to STO2.D ------------------------------
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        bus(NEWS, A);                                  // PE.NEWS = -PE.A + offset

        WHERE(DREG_STO2);
        bus(MEMBRANE_STO2, XN);                        // STO2.D := PE.A
    scamp5_kernel_end();
}


// ---------------------------------------------------------------------------
// stage_nonlinearity_N_neg   (N(x) = max(0, x) + I_bias)
//   Half-wave rectification followed by a bias add.
//
//   STATE TOUCHED: PE.A (modified), PE.F (scratch),
//                  DREG S4 (N_NEGATIVE - the rectification mask).
// ---------------------------------------------------------------------------
inline void stage_nonlinearity_N_neg()
{
    if (!apply_N_func) return;

    scamp5_load_in(F, 0);

    scamp5_kernel_begin();
        where(A);                            // FLAG = (PE.A > 0)
        NOT(N_NEGATIVE, FLAG);               // N_NEGATIVE = (PE.A <= 0)
        AND(N_NEGATIVE, N_NEGATIVE, DREG_PE);// limit to PE cells

        WHERE(N_NEGATIVE);
        mov(A, F);                           // PE.A = 0 where non-positive
    scamp5_kernel_end();

    scamp5_load_in(F, I_bias);

    scamp5_kernel_begin();
        WHERE(DREG_PE);
        add(A, A, F);                        // PE.A += I_bias
    scamp5_kernel_end();
}


// ---------------------------------------------------------------------------
// stage_lif_neg   (Leaky Integrate & Fire ganglion neuron)
//
//   One Euler step:
//       V_LIF <- V_LIF + p1 * I_gang - p2 * |V_LIF - u_rest|
//       spike := (V_LIF > U_thresh)
//       if spike: V_LIF -= U_thresh   (subtract-threshold reset)
//
//   STATE TOUCHED:
//     STO2.D (V_LIF / MEMBRANE_STO2)  : read at start, rewritten at end
//     PE.A,B,C,D,E,F                  : all clobbered as scratch
//     DREG S4 (FIRING)                : raw spike flag (transient)
//     DREG S5 (FIRING_AND_IS_PE)      : spike map restricted to PE cells;
//                                        held until the display block reads it
//
//   On entry: PE.A = I_gang (output of stage_nonlinearity_N_pos).
//   On exit : PE.A = post-reset V_LIF ; STO2.D updated; S5 = spike map.
// ---------------------------------------------------------------------------
void stage_lif_neg()
{
    if (!compute_LIF) return;

    // ---- PE.C := p1 * PE.A  (MAC with multiplicand in A, accumulator in C)
    // No b_0 = 0 prefix here: this is a direct gain multiplication, not an
    // IIR LPF. p1 is supplied with the desired weight directly.
    scamp5_load_in(F, 0);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        mov(C, F);                           // PE.C = 0 (accumulator)
    scamp5_kernel_end();

    {
        int bit = 1 << (p1.no_bits - 1);
        while (bit)
        {
            if (bit & p1.value)
            {
                scamp5_kernel_begin();
                add(C, C, A);
                diva(A, E, F);
                scamp5_kernel_end();
            }
            else
            {
                scamp5_kernel_begin();
                diva(A, E, F);
                scamp5_kernel_end();
            }
            bit >>= 1;
        }
    }

    scamp5_kernel_begin();
        mov(A, C);                           // PE.A = p1 * I_gang
    scamp5_kernel_end();

    // ---- PE.B := V_LIF  (load from STO2.D via south neighbor) ----------
    scamp5_kernel_begin();
        WHERE(DREG_STO2);
        bus(NEWS, MEMBRANE_STO2);            // STO2.NEWS = -STO2.D + offset

        WHERE(DREG_PE);
        bus(B, XS);                          // PE.B = STO2.D = V_LIF
    scamp5_kernel_end();

    // ---- PE.C := |V_LIF - u_rest|  (uses PE.F as temp) -----------------
    scamp5_load_in(F, u_rest);
    scamp5_kernel_begin();
        sub(C, B, F);                        // PE.C = V_LIF - u_rest
        mov(F, C);
        abs(F, F);                           // PE.F = |V_LIF - u_rest|

        WHERE(DREG_PE);
        mov(C, F);                           // PE.C = |V_LIF - u_rest|
    scamp5_kernel_end();

    // ---- PE.D := p2 * |V_LIF - u_rest|  (MAC: multiplicand C, accumulator D)
    scamp5_load_in(F, 0);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        mov(D, F);                           // PE.D = 0
    scamp5_kernel_end();

    {
        int bit = 1 << (p2.no_bits - 1);
        while (bit)
        {
            if (bit & p2.value)
            {
                scamp5_kernel_begin();
                add(D, D, C);
                diva(C, E, F);
                scamp5_kernel_end();
            }
            else
            {
                scamp5_kernel_begin();
                diva(C, E, F);
                scamp5_kernel_end();
            }
            bit >>= 1;
        }
    }

    // ---- Combine: V_LIF_new = V_LIF + p1*I_gang - p2 * |V_LIF - u_rest| --
    // PE.A : p1 * I_gang
    // PE.B : V_LIF
    // PE.D : p2 * |V_LIF - u_rest|
    scamp5_kernel_begin();
        add(A, A, B);                        // PE.A = p1*I_gang + V_LIF
        sub(A, A, D);                        // PE.A = V_LIF_new
    scamp5_kernel_end();

    // ---- Threshold + subtract-threshold reset --------------------------
    scamp5_load_in(F, U_thresh);
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        sub(B, A, F);                                       // PE.B = V_LIF_new - threshold

        where(B);                                           // FLAG = (PE.B > 0)
        MOV(FIRING, FLAG);                                  // S4 = raw spike flag
        AND(FIRING_NEG_AND_IS_PE, FIRING, DREG_PE);         // S6 = spike map at PEs

        WHERE(FIRING_NEG_AND_IS_PE);
        mov(A, B);                                          // V_LIF -= threshold on spike
    scamp5_kernel_end();

    // ---- Persist new V_LIF back to STO2.D ------------------------------
    scamp5_kernel_begin();
        WHERE(DREG_PE);
        bus(NEWS, A);                                  // PE.NEWS = -PE.A + offset

        WHERE(DREG_STO2);
        bus(MEMBRANE_NEG_STO2, XN);                    // STO2.E := PE.A
    scamp5_kernel_end();
}

inline void move_LIF_for_displays()
{
    scamp5_kernel_begin();
        WHERE(DREG_PE);
            // move the OFF events to B 
            mov(B, A);

        // restore the map from ON to A        
        WHERE(DREG_STO2);
            bus(NEWS, MEMBRANE_STO2);   // take the membrane from STO2.D

        WHERE(DREG_PE);
            bus(A, XS);                 // move it to PE.A
    scamp5_kernel_end();

}



// ===========================================================================
// DEBUG HELPER
// ===========================================================================
// probe_for(reg_num, display_handle): copy the contents of PE.<reg> across
// the whole superpixel via PE.F and output it on the given display widget.
// Use this from inside the frame loop to peek at intermediate AREG values.
//
// The function destroys PE.F and the F register of all STO cells (because
// of replicate_F). Keep that in mind if used inside a pipeline stage.
// ---------------------------------------------------------------------------
void probe_for(int reg, int display)
{
    if (reg == A_REG_NUM)
    {
        scamp5_kernel_begin();
            mov(F, A);
            replicate_F();
        scamp5_kernel_end();
        scamp5_output_image(F, display);
    }
    else if (reg == B_REG_NUM)
    {
        scamp5_kernel_begin();
            mov(F, B);
            replicate_F();
        scamp5_kernel_end();
        scamp5_output_image(F, display);
    }
    else if (reg == C_REG_NUM)
    {
        scamp5_kernel_begin();
            mov(F, C);
            replicate_F();
        scamp5_kernel_end();
        scamp5_output_image(F, display);
    }
    else if (reg == D_REG_NUM)
    {
        scamp5_kernel_begin();
            mov(F, D);
            replicate_F();
        scamp5_kernel_end();
        scamp5_output_image(F, display);
    }
    else if (reg == F_REG_NUM)
    {
        // WARNING: mov(F, F) violates the AREG constraint (mov(X,Y) requires
        // X != Y per the SCAMP tutorial table). Kept verbatim from the
        // original code; this branch is dead (probe_for() is not called from
        // anywhere) so it does no harm at runtime. If you call probe_for with
        // F_REG_NUM, remove the mov(F, F) line first.
        scamp5_kernel_begin();
            mov(F, F);
            replicate_F();
        scamp5_kernel_end();
        scamp5_output_image(F, display);
    }
}


// ===========================================================================
// GUI SETUP
// ===========================================================================
// All vs_gui_add_slider() / vs_gui_add_display() calls live here. Returns
// the display handles in out-parameters so main() can use them in the
// display block.
//
// The display widgets are returned as a struct of vs_handle values; the
// sliders bind directly to the global tuning variables defined above.
// ---------------------------------------------------------------------------
struct gui_displays
{
    vs_handle display_00;
    vs_handle display_01;
    vs_handle display_10;
    vs_handle display_11;
    vs_handle display_12;
};

inline gui_displays setup_gui_controls()
{
    vs_gui_set_info(VS_M0_PROJECT_INFO_STRING);

    gui_displays dpy;
    dpy.display_00 = vs_gui_add_display("display_00", 0, 0);
    dpy.display_01 = vs_gui_add_display("display_01", 0, 1);
    dpy.display_10 = vs_gui_add_display("display_10", 1, 0);
    dpy.display_11 = vs_gui_add_display("display_11", 1, 1);
    dpy.display_12 = vs_gui_add_display("display_12", 1, 2);

    // -- pipeline-stage toggles --
    vs_gui_add_slider("apply_blur_1",                   0,   1, apply_blur_1,                   &apply_blur_1);
    vs_gui_add_slider("apply_1st_hpf",                  0,   1, apply_1st_hpf,                  &apply_1st_hpf);
    vs_gui_add_slider("apply_1st_lpf",                  0,   1, apply_1st_lpf,                  &apply_1st_lpf);
    vs_gui_add_slider("apply_2nd_blur",                 0,   1, apply_2nd_blur,                 &apply_2nd_blur);
    vs_gui_add_slider("apply_2nd_lpf",                  0,   1, apply_2nd_lpf,                  &apply_2nd_lpf);
    vs_gui_add_slider("apply_3rd_lpf",                  0,   1, apply_3rd_lpf,                  &apply_3rd_lpf);
    vs_gui_add_slider("compute_opl",                    0,   1, compute_opl,                    &compute_opl);
    vs_gui_add_slider("compute approximation",          0,   1, compute_approximation,          &compute_approximation);
    vs_gui_add_slider("compute blur for approximation", 0,   1, compute_blur_for_approximation, &compute_blur_for_approximation);
    vs_gui_add_slider("add leftovers to feedback",      0,   1, add_leftovers_to_feedback,      &add_leftovers_to_feedback);
    vs_gui_add_slider("compute feedback phase2",        0,   1, compute_feedback_phase2,        &compute_feedback_phase2);
    vs_gui_add_slider("apply HPF apply_hpf2",           0,   1, apply_2nd_hpf,                  &apply_2nd_hpf);
    vs_gui_add_slider("apply N",                        0,   1, apply_N_func,                   &apply_N_func);
    vs_gui_add_slider("amplify_opl",                    0,   4, amplify_opl,                    &amplify_opl);
    vs_gui_add_slider("amplify_opl_threshold",          0, 100, amplify_opl_thresh,             &amplify_opl_thresh);
    vs_gui_add_slider("compute_LIF",                    0,   1, compute_LIF,                    &compute_LIF);

    // -- scalar parameters --
    vs_gui_add_slider("I_bias",   -128, 127, I_bias,   &I_bias);
    vs_gui_add_slider("u_rest",   -128, 127, u_rest,   &u_rest);
    vs_gui_add_slider("U_thresh", -128, 127, U_thresh, &U_thresh);

    // -- display selectors --
    vs_gui_add_slider("display A",      0, 1, display_A,   &display_A);
    vs_gui_add_slider("display events", 0, 1, display_evn, &display_evn);

    // -- |V_BIP| amplification and gain offset --
    vs_gui_add_slider("amplify", 0, 4, amplify_iters, &amplify_iters);
    vs_gui_add_slider("g_{A}^{0", -128, 127, g_A, &g_A);

    // -- filter coefficient sliders (use lambda callbacks to update values) --
    auto slider_vxc = vs_gui_add_slider("vxc: ", 0, 4095, 2900);
    vs_on_gui_update(slider_vxc, [&](int32_t v){ vs_scamp5_configure_voltage(3, v); });

    auto sl_lpf1     = vs_gui_add_slider("lpf1.value: ",        1, 31, LPF1.value);
    vs_on_gui_update(sl_lpf1,     [&](int32_t v){ LPF1.value        = v; });

    auto sl_lpf2     = vs_gui_add_slider("lpf2.value: ",        1, 31, LPF2.value);
    vs_on_gui_update(sl_lpf2,     [&](int32_t v){ LPF2.value        = v; });

    auto sl_lambda_A = vs_gui_add_slider("lambda_A.value: ",    1, 31, lambda_A.value);
    vs_on_gui_update(sl_lambda_A, [&](int32_t v){ lambda_A.value    = v; });

    auto sl_lpf3     = vs_gui_add_slider("lpf3.value: ",        1, 16, LPF3.value);
    vs_on_gui_update(sl_lpf3,     [&](int32_t v){ LPF3.value        = v; });

    auto sl_alpha    = vs_gui_add_slider("alpha_fixed.value: ", 1, 31, alpha_fixed.value);
    vs_on_gui_update(sl_alpha,    [&](int32_t v){ alpha_fixed.value = v; });

    auto sl_p1       = vs_gui_add_slider("p1.value: ",          1, 31, p1.value);
    vs_on_gui_update(sl_p1,       [&](int32_t v){ p1.value          = v; });

    auto sl_p2       = vs_gui_add_slider("p2.value: ",          1, 31, p2.value);
    vs_on_gui_update(sl_p2,       [&](int32_t v){ p2.value          = v; });

    return dpy;
}


// ===========================================================================
// ARITHMETIC INITIALIZATION
// ===========================================================================
// Zero out every analog register at startup. The PE's AREGs decay over time
// when not refreshed, so initialising to a known value avoids the first few
// frames carrying garbage from power-on.
// ---------------------------------------------------------------------------
inline void init_aregs_to_zero()
{
    scamp5_kernel_begin();
        ALL();
        scamp5_load_in(A, 0);
        scamp5_load_in(B, 0);
        scamp5_load_in(C, 0);
        scamp5_load_in(D, 0);
        scamp5_load_in(E, 0);
        scamp5_load_in(F, 0);
    scamp5_kernel_end();
}


// ===========================================================================
// DISPLAY OUTPUT
// ===========================================================================
// Display block. NOTE: by the time this runs, PE.A has been written and
// rewritten by many stages. There is no persistent slot for the original
// "input" or "I_OPL" - those values are gone. To probe a specific
// intermediate state, disable later stages from the GUI so PE.A is left
// holding the value of interest, then enable display_A.
//
// FIRING_AND_IS_PE is a DREG (1-bit per cell); scamp5_output_image accepts
// both AREG and DREG inputs.
// ---------------------------------------------------------------------------
inline void output_displays(const gui_displays& dpy)
{
    if (!vs_gui_is_on()) return;

    if (display_A)
    {
        // fill_A replicates PE.A across the superpixel so the output image
        // shows the same value at all 4 cells in each 2x2 block (otherwise
        // STO[1-3].A would alternate visible garbage with PE.A's value).
        fill_A();
        scamp5_output_image(A, dpy.display_00);
    }

    if (display_evn)
        scamp5_output_image(FIRING_AND_IS_PE, dpy.display_10);

    // The other display widgets (display_01, _11, _12) are left blank and
    // available for ad-hoc debugging. Wire up probe_for(reg, display_*) here
    // when you want to inspect a specific AREG.
}

// ===========================================================================
// FRAME STATISTICS
// ===========================================================================
// Print the per-frame timing to the host text panel.
// ---------------------------------------------------------------------------
inline void print_frame_stats(int frame_time_us)
{
    int max_possible_fps = frame_time_us > 0 ? (1000000 / frame_time_us) : 0;
    vs_post_text("!clear");
    vs_post_text("frame time %d microseconds, potential FPS ~%d \n",
                 frame_time_us, max_possible_fps);
}


// ===========================================================================
// MAIN
// ===========================================================================
int main()
{
    // -- Initialization --
    vs_init();
    vs_on_shutdown([&]() { vs_post_text("M0 shutdown\n"); });

    gui_displays dpy = setup_gui_controls();

    init_aregs_to_zero();

    // -- Frame loop --
    //
    // Each iteration:
    //   1. Refresh superpixel masks (cheap, defends against DREG decay).
    //   2. Capture a new image into PE.A.
    //   3. Run the pipeline stages in order; any stage gated off by its GUI
    //      flag becomes a no-op, leaving PE.A in whatever state the previous
    //      stage left it. That is how you probe intermediate values.
    //   4. After first_pass, clear the flag so seeded LPFs use real history.
    //   5. Push the relevant display(s) and stats to the host.
    while (1)
    {
        frame_timer.reset();
        vs_frame_loop_control();

        load_superpixel_masks();
        stage_capture();

        // OPL (outer plexiform layer)
        stage_blur_input();          // optional G_C
        stage_hpf1();                // T_C-step (temporal differencing)
        stage_lpf1();                // E_C-step (IIR LPF, caches PE.C = T_C)
        stage_blur_surround();       // G_S (also always replicates PE.A)
        stage_lpf2();                // E_S (IIR LPF on surround)
        stage_compute_opl();         // PE.A := T_C - E_S  (sign-corrected)
        stage_amplify_opl();         // optional thresholded amplification

        // Bipolar contrast gain control (off by default - noisy on HW)
        stage_gain_control();

        // Ganglion encoding
        stage_hpf2();                // T_2 HPF on the I_OPL stream


        store_I_bip();

        stage_nonlinearity_N_pos();      // max(0, x) + I_bias
        stage_lif_pos();                 // LIF integrator + spike threshold

        restore_I_bip_and_prepare();

        stage_nonlinearity_N_neg();
        stage_lif_neg();



        first_pass = false;

        output_displays(dpy);
        print_frame_stats(frame_timer.get_usec());
    }

    return 0;
}
