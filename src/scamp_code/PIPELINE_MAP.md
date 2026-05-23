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
        ╰────╯   ┌────────────────────╮ ╰────╯   ┌──────────────────────────────┐
                 │ A : [transient]    │          │ A : [transient] blur replica │
                 │ B : E_A LPF state  │          │ F : [transient] blur replica │
                 │ C : PREV_FRAME for │          │ B,C,D,E : ── ALL FREE ──     │
                 │     hpf2 (prev OPL)│          └──────────────────────────────┘
                 │ D : V_LIF membrane │
                 │ E : ── free ──     │
                 │ F : ── free ──     │
                 └────────────────────┘

  Persistent slots used : 7 of 18 (39%)
  Free for future state : STO1.F, STO2.E, STO2.F, STO3.{B,C,D,E}  (5 slots)
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
  │      │                         │ nonlinearity_N : N_NEGATIVE                 │
  │      │                         │ gain_control   : VBIP_GREATER_THAN_64 (1/2) │
  │      │                         │ lif            : FIRING (raw spike flag)    │
  ├──────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ S5   │ multiplexed scratch     │ gain_control   : VBIP_GREATER_THAN_32 (1/4) │
  │      │                         │ lif            : FIRING_AND_IS_PE  ★        │
  │      │                         │                  ★ = the spike-map output   │
  │      │                         │                      displayed to the host  │
  ├──────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ S6   │ scratch (1 stage only)  │ gain_control   : VBIP_GREATER_THAN_16 (1/8) │
  └──────┴─────────────────────────┴─────────────────────────────────────────────┘

  WARNING: S4/S5 lifetimes are non-overlapping by design. If you add a new
  stage that needs DREGs, double-check none of the existing slots are still
  alive at that point. The Q-bits inside gain_control are fully consumed
  before lif runs, so S4-S6 are free to be reused by lif.
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
10  nonlinearity_N         max(0, x) + I_bias  = I_gang         ──                      S4
11  lif                    V_LIF (post-reset)                   STO2.D   read & write   S4 = FIRING
                                                                                         S5 = spike map ★
═════════════════════════════════════════════════════════════════════════════════════════════════════

★ = held through the end of the frame so the display block can output it.

Stages are gated by GUI sliders; disable later stages to leave PE.A
holding an earlier intermediate value, then enable `display_A` to probe.
```

---

## 5. PE register lifetimes during one frame

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
 10  nonlinearity_N         RW   .    .    .    .    Sc
 11  lif                    RW   Sc   Sc   Sc   Sc   Sc

  PE.A : carries the main signal through every stage (the pipeline trunk).
  PE.B : scratch — usually the accumulator in MAC loops, or a neighbor load.
  PE.C : scratch — also the "T_C cache" between lpf1 and compute_opl.
  PE.D : scratch — V_BIP holding register inside gain_control.
  PE.E,F: diva() temporaries; F also used as scamp5_load_in target.
```

---

## 6. Quick-reference cheat-sheet

```
  Want to see…                       Toggle off after stage…   PE.A then holds
  ────────────────────────────────── ─────────────────────── ─────────────────────
  raw input                           none → use probe       luminance
  blurred input (G_C)                 apply_blur_1            blurred input
  HPF1 output                         apply_1st_hpf           current − prev
  LPF1 output (T_C)                   apply_1st_lpf           T_C
  blurred T_C                         apply_2nd_blur          G_S * T_C
  E_S surround                        apply_2nd_lpf           E_S
  I_OPL                               compute_opl             T_C − E_S
  V_BIP_new                           compute_approximation   V_BIP_new
  T_2                                 apply_2nd_hpf           current − prev I_OPL
  I_gang                              apply_N_func            max(0, T_2) + I_bias
  V_LIF (post reset)                  compute_LIF             V_LIF
  spikes only                         display_evn=1           (see S5 DREG)
```

---

## 7. Neighbor routing reminders

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

## 8. Editing notes

- Boxes use Unicode box-drawing characters (`─│┌┐└┘├┤┬┴┼╔╗╚╝═║╠╣╦╩╬`); your
  editor should render them fine in monospace.
- If you add or remove a pipeline stage, update **§4** (timeline), **§5**
  (PE register lifetimes), and the *Notes* column in **§3** (DREG map).
- Persistent-slot table in **§2** is the canonical "memory map" — keep it in
  sync with the macros in [`scamp5_main.cpp`](src/scamp5_main.cpp) (especially
  `BIPOLAR_REGISTER_STO1`, `BIPOLAR_LPF_REGISTER_STO2`, `PREV_FRAME_STO2`,
  `MEMBRANE_STO2`).
