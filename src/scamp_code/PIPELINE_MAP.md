# SCAMP-5 Silicon Retina — Memory Map & Pipeline Timeline

Free-to-edit visual aid. Source: [`src/scamp_code/src/scamp5_main.cpp`](src/scamp5_main.cpp).
Render in monospace (your IDE or a `.txt`-style preview).

---

## 1. Superpixel layout (one 2x2 block of physical PEs)

```
              column 0          column 1
            ┌────────────┬────────────┐
   row 0    │            │            │
            │    PE      │    STO1    │     PE   = compute + image capture
            │  (compute) │  (East)    │     STO1 = East  of PE   (XW/XE bus)
            │            │            │     STO2 = South of PE   (XN/XS bus)
            ├────────────┼────────────┤     STO3 = diagonal — no direct bus
   row 1    │            │            │            link to PE; needs relay
            │    STO2    │    STO3    │
            │  (South)   │ (diagonal) │     Logical resolution: 128 x 128
            │            │            │     (256x256 physical / 2x2 block)
            └────────────┴────────────┘
```

---

## 2. Persistent state — where each map lives

Persistent across frames. Each STO cell has 6 AREGs (A..F); the table below
shows which slots carry pipeline state and which stage owns each slot.

```
        ╭────╮                          ╭────╮
        │ PE │   scratch only           │STO1│   East of PE
        ╰────╯   (all of A..F clobbered ╰────╯   ┌──────────────────────────────┐
                  every frame)                   │ A : [transient] blur replica │
                                                 │ B : PREV_FRAME for hpf1      │
                                                 │ C : LPF1 state               │
                                                 │ D : LPF2 state               │
                                                 │ E : V_BIP (membrane voltage) │
                                                 │ F : ── free ──               │
                                                 └──────────────────────────────┘
        ╭────╮                          ╭────╮
        │STO2│   South of PE            │STO3│   diagonal
        ╰────╯   ┌──────────────────────╮       ┌──────────────────────────────┐
                 │ A : [transient]      │       │ A : [transient] blur replica │
                 │     blur replica;    │       │ F : [transient] blur replica │
                 │     ALSO holds the   │       │ B,C,D,E : ── ALL FREE ──     │
                 │     signed I_bip     │       └──────────────────────────────┘
                 │     between LIF_pos  │
                 │     and LIF_neg.     │
                 │ B : E_A LPF state    │
                 │ C : PREV_FRAME for   │
                 │     hpf2 (prev OPL)  │
                 │ D : ON  V_LIF        │
                 │ E : OFF V_LIF (NEW)  │
                 │ F : ── free ──       │
                 └──────────────────────┘

  Persistent slots used : 8 of 18 (44%)
  Free for future state : STO1.F, STO2.F, STO3.{B,C,D,E}  (5 slots)
```

---

## 3. DREG map (1-bit registers, 7 per cell)

```
  ┌──────┬─────────────────────────┬─────────────────────────────────────────────┐
  │ DREG │ Role                    │ Notes                                       │
  ├──────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ S0   │ DREG_PE       mask      │ ┐                                           │
  │ S1   │ DREG_STO1     mask      │ │ Superpixel-type masks, reloaded fresh     │
  │ S2   │ DREG_STO2     mask      │ │ each frame from load_superpixel_masks()   │
  │ S3   │ DREG_STO3     mask      │ ┘                                           │
  ├──────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ S4   │ multiplexed scratch     │ amplify_opl    : I_OPL_SMALLER_THAN_THRESH  │
  │      │ (clobbered every stage  │ nonlinearity_N : N_NEGATIVE                 │
  │      │  that needs it; never   │ gain_control   : VBIP_GREATER_THAN_64 (1/2) │
  │      │  carried across stages) │ lif_pos / neg  : FIRING (raw spike flag)    │
  ├──────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ S5   │ scratch + HELD output   │ gain_control   : VBIP_GREATER_THAN_32 (1/4) │
  │      │                         │ lif_pos        : FIRING_AND_IS_PE  ★ ON     │
  │      │                         │                  ★ = ON spike map, held     │
  │      │                         │                      until display reads it │
  ├──────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ S6   │ scratch + HELD output   │ gain_control   : VBIP_GREATER_THAN_16 (1/8) │
  │      │                         │ lif_neg        : FIRING_NEG_AND_IS_PE ★ OFF │
  │      │                         │                  ★ = OFF spike map, held    │
  │      │                         │                      until display reads it │
  └──────┴─────────────────────────┴─────────────────────────────────────────────┘

  WARNING: S4/S5/S6 lifetimes are non-overlapping by design. The Q-bits
  inside gain_control are fully consumed BEFORE lif_pos starts. lif_pos's
  FIRING goes into S4 (transient) and FIRING_AND_IS_PE goes into S5 (held).
  lif_neg reuses S4 for its FIRING but writes its result into S6, leaving
  S5 untouched so it can still be read at display time. If you add a new
  stage, make sure none of S5/S6's "held" values are still alive at the
  point you intend to reuse them.
```

---

## 4. Pipeline timeline — one frame, top to bottom

```
═════════════════════════════════════════════════════════════════════════════════════════════════════
 t  Stage                  PE.A holds…                          STO slot touched        DREG clobber
─── ────────────────────── ─────────────────────────────────── ──────────────────────── ─────────────
 0  capture                raw luminance                        ──                      S0..S3 (load)
 1  blur_input             blurred raw           (optional)     STO1.A, STO2.A, STO3.A
 2  hpf1                   current − prev_frame                 STO1.B   read & write
 3  lpf1                   LPF1 output                          STO1.C   read & write
                           also: PE.C = T_C cached
 4  blur_surround          blurred LPF1 (G_S)                   STO1.A, STO2.A, STO3.A
                                                                 (always replicates;
                                                                  gauss runs only if
                                                                  apply_2nd_blur=1)
 5  lpf2                   LPF2 output  ( = E_S surround )      STO1.D   read & write
 6  compute_opl            I_OPL = T_C − E_S                    ──
                           (sign-corrected)
 7  amplify_opl   (opt.)   amplified |I_OPL| above threshold    ──                      S4
 8  gain_control  (opt.)   V_BIP_new                            STO1.E   read & write   S4, S5, S6
                                                                STO2.B   read & write   (Q bit-planes)
 9  hpf2                   T_2 output (current − prev OPL)      STO2.C   read & write

 ── ON / OFF channel split (the post-HPF2 signal is the signed I_bip) ──

10  store_I_bip            same (I_bip)                         STO2.A   write only
                                                                  (holds signed I_bip)
11  nonlinearity_N_pos     max(0, +I_bip) + I_bias              ──                      S4
12  lif_pos                ON  V_LIF (post-reset)               STO2.D   read & write   S4 = FIRING
                                                                                         S5 = ON spike map ★
13  restore_I_bip_+_negate −I_bip                                STO2.A   read only
                                                                  (clobbers PE.A,C)
14  nonlinearity_N_neg     max(0, −I_bip) + I_bias              ──                      S4
15  lif_neg                OFF V_LIF (post-reset)               STO2.E   read & write   S4 = FIRING
                                                                                         S6 = OFF spike map ★
═════════════════════════════════════════════════════════════════════════════════════════════════════

★ = held through the end of the frame so the display block can output it.

Stages are gated by GUI sliders; disable later stages to leave PE.A
holding an earlier intermediate value. The 4-panel display block at the
bottom of the frame shows the ON/OFF membranes and spike maps.
```

---

## 5. Display layout (after each frame)

```
  ┌─────────────────────────────┬─────────────────────────────┐
  │   ON  V_LIF                 │   OFF V_LIF                 │
  │   (loads STO2.D into PE.A   │   (loads STO2.E into PE.A   │
  │    and replicates it)       │    and replicates it)       │
  │   GUI: display_v_lif_on     │   GUI: display_v_lif_off    │
  ├─────────────────────────────┼─────────────────────────────┤
  │   ON  events                │   OFF events                │
  │   outputs DREG S5           │   outputs DREG S6           │
  │   (FIRING_AND_IS_PE)        │   (FIRING_NEG_AND_IS_PE)    │
  │   GUI: display_evn_on       │   GUI: display_evn_off      │
  └─────────────────────────────┴─────────────────────────────┘
```

---

## 6. PE register lifetimes during one frame

Useful when reasoning about *scratch* usage inside a stage. `R`/`W`/`Sc`/`.`
markers show whether each PE register is read, written, scratched (clobbered
mid-stage but result is discarded), or untouched.

```
                            PE register
                            A     B    C    D    E    F
 ──────────────────────── ──── ──── ──── ──── ──── ────
  0  capture                W    .    .    .    .    W
  1  blur_input             RW   .    .    .    .    .
  2  hpf1                   RW   W    .    .    .    Sc
  3  lpf1                   RW   W    W    .    Sc   Sc
  4  blur_surround          RW   .    .    .    .    .
  5  lpf2                   RW   W    .    .    Sc   Sc
  6  compute_opl            RW   .    Sc   .    .    Sc
  7  amplify_opl            RW   Sc   .    .    .    Sc
  8  gain_control           RW   Sc   Sc   Sc   Sc   Sc
  9  hpf2                   RW   W    .    .    .    Sc
 10  store_I_bip            R    .    .    .    .    .
 11  nonlinearity_N_pos     RW   .    .    .    .    Sc
 12  lif_pos                RW   Sc   Sc   Sc   Sc   Sc
 13  restore + negate       RW   .    Sc   .    .    .
 14  nonlinearity_N_neg     RW   .    .    .    .    Sc
 15  lif_neg                RW   Sc   Sc   Sc   Sc   Sc
 16  (display block)        Sc   .    .    .    .    Sc       ← only if a V_LIF
                                                                 display flag is on

  PE.A : carries the main signal through every stage (the pipeline trunk).
         After lif_neg it holds the post-reset OFF V_LIF; the display block
         then loads ON/OFF V_LIF from STO2.D/E one at a time.
  PE.B : scratch — usually the accumulator in MAC loops, or a neighbor load.
  PE.C : scratch — also the "T_C cache" between lpf1 and compute_opl, and
         the negation temporary inside restore_I_bip_and_prepare.
  PE.D : scratch — V_BIP holding register inside gain_control.
  PE.E,F: diva() temporaries; F also used as scamp5_load_in target.
```

---

## 7. Quick-reference cheat-sheet

```
  Want to see…                       Toggle off after stage…       PE.A then holds
  ────────────────────────────────── ─────────────────────────── ─────────────────────
  raw input                           none → use probe              luminance
  blurred input (G_C)                 apply_blur_1                  blurred input
  HPF1 output                         apply_1st_hpf                 current − prev
  LPF1 output (T_C)                   apply_1st_lpf                 T_C
  blurred T_C                         apply_2nd_blur                G_S * T_C
  E_S surround                        apply_2nd_lpf                 E_S
  I_OPL                               compute_opl                   T_C − E_S
  V_BIP_new                           compute_approximation         V_BIP_new
  T_2 (signed I_bip)                  apply_2nd_hpf                 current − prev I_OPL
  ON  I_gang                          apply_N_func     (pos branch) max(0, +T_2) + I_bias
  ON  V_LIF (post reset)              compute_LIF      (pos branch) V_LIF_pos
  OFF V_LIF (post reset)              compute_LIF      (neg branch) V_LIF_neg
  ON  spikes only                     display_evn_on=1              (see S5 DREG)
  OFF spikes only                     display_evn_off=1             (see S6 DREG)
```

---

## 8. Neighbor routing reminders

```
   bus(NEWS, X) on PE      ─── PE.NEWS = -PE.X + offset (inverting!)
   bus(X, XE)   on STO1    ─── STO1.X  = -STO1.east.NEWS + offset
                                       = -PE.NEWS + offset
                                       = PE.X        (double-bus = copy)

   PE  ↔  STO1   :  PE writes NEWS,  STO1 reads XW
                    STO1 writes NEWS, PE reads XE
   PE  ↔  STO2   :  PE writes NEWS,  STO2 reads XN
                    STO2 writes NEWS, PE reads XS
   PE  ↔  STO3   :  no direct edge  — must relay via STO1 or STO2
```

---

## 9. Editing notes

- Boxes use Unicode box-drawing characters (`─│┌┐└┘├┤┬┴┼╔╗╚╝═║╠╣╦╩╬`); your
  editor should render them fine in monospace.
- If you add or remove a pipeline stage, update **§4** (timeline), **§6**
  (PE register lifetimes), and the *Notes* column in **§3** (DREG map).
- Persistent-slot table in **§2** is the canonical "memory map" — keep it in
  sync with the macros in [`scamp5_main.cpp`](src/scamp5_main.cpp) (especially
  `BIPOLAR_REGISTER_STO1`, `BIPOLAR_LPF_REGISTER_STO2`, `PREV_FRAME_STO2`,
  `MEMBRANE_STO2`, `MEMBRANE_NEG_STO2`).
- The 4-panel display layout in **§5** mirrors `output_displays()` in
  [scamp5_main.cpp](src/scamp5_main.cpp); GUI panel names and the slider
  flags in `setup_gui_controls()` should stay aligned with this section.
