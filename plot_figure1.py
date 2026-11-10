#!/usr/bin/env python
# coding: utf-8
"""
Replication Script for Figure 1
Reference: "Reputation dynamics in concurrent games" (Nagafuji, Akiyama, and Sawa, 2026)

Figure 1 Description:
  - Parameter space: (b1, b2) in [0, 10] x [0, 10] with c1 = c2 = 1 and mu = 0.01.
  - Red region: Sustainable Six (S6) configurations are evolutionarily stable and maximize
    the expected payoff under the OR fusion rule (pi_S6_OR >= pi_L8_AND and S6 is ESS under OR).
  - Blue region: Leading Eight (L8) configurations are evolutionarily stable and maximize
    the expected payoff under the AND fusion rule, but S6 is not ESS under OR.
  - White region: Only unconditional defection (ALLD) configurations are evolutionarily stable.

To ensure computational tractability across a fine grid (e.g., 101 x 101 points),
this script checks evolutionary stability against all 255 alternative mutant strategies
for the analytical payoff-maximizing candidates identified in the paper:
  - L8 (under AND fusion): All 8 Leading Eight configurations (Ohtsuki & Iwasa 2004, 2006)
  - S6 (under OR fusion) : All 6 Sustainable Six configurations (Section 3.1.2 & Appendix B)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from multiprocessing import Pool
import multiprocessing as mp

# ============================================================
# Model Baseline Definitions
# ============================================================

def build_behavior_tables():
    acts = list(product([1, 0], repeat=4))
    Cbits = np.array(acts, dtype=np.float64)  # (16, 4)

    c11, c10, c01, c00 = Cbits[:, 0], Cbits[:, 1], Cbits[:, 2], Cbits[:, 3]
    u = c10 - c00
    v = c01 - c00
    w = c11 - c10 - c01 + c00
    coef = np.stack([c00, u, v, w], axis=1)

    def bits_string(idx):
        b = Cbits[idx].astype(int)
        return "".join("C" if x == 1 else "D" for x in b)

    s_bits = [bits_string(i) for i in range(16)]
    return Cbits, coef, s_bits

Cbits16, coef16, s_bits16 = build_behavior_tables()

s1_idx = np.repeat(np.arange(16, dtype=np.int32), 16)
s2_idx = np.tile(np.arange(16, dtype=np.int32), 16)

D1_INDEX_FROM_D8 = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
D2_INDEX_FROM_D8 = np.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int32)

def str_to_norm_vec(d_str: str) -> np.ndarray:
    return np.array([int(c) for c in d_str], dtype=np.int8)

def precompute_D4_for_norm(norm_vec: np.ndarray, mu_e: float, mu_a: float) -> np.ndarray:
    dC = norm_vec[[0, 2, 4, 6]].astype(np.float64)
    dD = norm_vec[[1, 3, 5, 7]].astype(np.float64)
    intended = Cbits16  # (16, 4)
    phi = (
        (1.0 - mu_e) * (intended * dC + (1.0 - intended) * dD)
        + mu_e * (intended * dD + (1.0 - intended) * dC)
    )
    return (1.0 - 2.0 * mu_a) * phi + mu_a  # (16, 4)

def fuse_to_D8(D1_4: np.ndarray, D2_4: np.ndarray, fusion_mode: str) -> np.ndarray:
    D1_8 = D1_4[..., D1_INDEX_FROM_D8]
    D2_8 = D2_4[..., D2_INDEX_FROM_D8]
    if fusion_mode == "AND":
        return D1_8 * D2_8
    elif fusion_mode == "OR":
        return 1.0 - (1.0 - D1_8) * (1.0 - D2_8)
    raise ValueError("Invalid fusion_mode")

def cubic_coefficients_from_D8(D8: np.ndarray):
    D111, D110, D101, D100, D011, D010, D001, D000 = D8
    A = D111 - D110 - D101 + D100 - D011 + D010 + D001 - D000
    B = D110 + D101 - 2.0 * D100 + D011 - 2.0 * D010 - 2.0 * D001 + 3.0 * D000
    C = D100 + D010 + D001 - 3.0 * D000 - 1.0
    D = D000
    return A, B, C, D

def real_roots_in_unit_interval(coeffs):
    coeffs = np.array(coeffs, dtype=np.float64)
    nz = np.where(np.abs(coeffs) > 1e-14)[0]
    if len(nz) == 0:
        return np.array([0.5], dtype=np.float64)
    roots = np.roots(coeffs[nz[0]:])
    real = roots[np.abs(roots.imag) < 1e-8].real
    real = real[(real >= -1e-9) & (real <= 1.0 + 1e-9)]
    return np.clip(real, 0.0, 1.0)

def mutant_h_from_D8(D8_mut: np.ndarray, h_res: float) -> np.ndarray:
    D = np.asarray(D8_mut, dtype=np.float64)
    h = h_res
    oh = 1.0 - h
    E_good = h*h*D[..., 0] + h*oh*D[..., 1] + oh*h*D[..., 2] + oh*oh*D[..., 3]
    F_bad  = h*h*D[..., 4] + h*oh*D[..., 5] + oh*h*D[..., 6] + oh*oh*D[..., 7]
    denom = 1.0 - E_good + F_bad
    out = np.empty_like(denom, dtype=np.float64)
    mask = np.abs(denom) < 1e-15
    out[mask] = np.where(F_bad[mask] > 0.5, 1.0, 0.0)
    out[~mask] = F_bad[~mask] / denom[~mask]
    return np.clip(out, 0.0, 1.0)

def y_scalar_vec(row_idx: int, h_row: float, h_col_vec: np.ndarray) -> np.ndarray:
    c00, u, v, w = coef16[row_idx]
    return c00 + h_row * u + h_col_vec * v + (h_row * h_col_vec) * w

def y_vec_scalar(row_idx_vec: np.ndarray, h_row_vec: np.ndarray, h_col: float) -> np.ndarray:
    c00 = coef16[row_idx_vec, 0]
    u = coef16[row_idx_vec, 1]
    v = coef16[row_idx_vec, 2]
    w = coef16[row_idx_vec, 3]
    return c00 + h_row_vec * u + h_col * v + (h_row_vec * h_col) * w

def apply_action_error(Y, mu_e: float):
    return mu_e + (1.0 - 2.0 * mu_e) * Y

# ============================================================
# Configurations Definition (L8 & S6)
# ============================================================

# All 8 Leading Eight configurations (under AND fusion)
L8_CONFIGS = [
    ("10111010", "CDCC"),
    ("10011010", "CDCC"),
    ("10111011", "CDCD"),
    ("10111001", "CDCD"),
    ("10011011", "CDCD"),
    ("10011001", "CDCD"),
    ("10111000", "CDCD"),
    ("10011000", "CDCD"),
]
l8_pairs_idx = [s_bits16.index(strat) * 16 + s_bits16.index(strat) for _, strat in L8_CONFIGS]

# All 6 Sustainable Six configurations (under OR fusion)
S6_CONFIGS = [
    ("10110010", "CDDC"),  # S6-1
    ("10010010", "CDDC"),  # S6-2
    ("10110011", "CDDD"),  # S6-3
    ("10110001", "CDDD"),  # S6-4
    ("10010011", "CDDD"),  # S6-5
    ("10010001", "CDDD"),  # S6-6
]
s6_pairs_idx = [s_bits16.index(strat) * 16 + s_bits16.index(strat) for _, strat in S6_CONFIGS]

# ============================================================
# Fast ESS Evaluator for Fixed (Norm, Strategy Pair, Fusion)
# ============================================================

class ConfigEvaluator:
    def __init__(self, norm_str: str, pair_idx: int, fusion_mode: str, mu_e: float, mu_a: float):
        self.norm_str = norm_str
        self.pair_idx = pair_idx
        self.fusion_mode = fusion_mode
        self.mu_e = mu_e
        self.mu_a = mu_a

        norm_vec = str_to_norm_vec(norm_str)
        D4 = precompute_D4_for_norm(norm_vec, mu_e, mu_a)
        D1_dual = D4[s1_idx]
        D2_dual = D4[s2_idx]
        self.D8_dual = fuse_to_D8(D1_dual, D2_dual, fusion_mode)  # (256, 8)

        self.D8_res = self.D8_dual[pair_idx]
        self.s1_r = int(s1_idx[pair_idx])
        self.s2_r = int(s2_idx[pair_idx])

        coeffs = cubic_coefficients_from_D8(self.D8_res)
        roots = real_roots_in_unit_interval(coeffs)
        self.h_res = float(np.max(roots)) if len(roots) > 0 else 1.0

        # Precompute mutant reputations
        self.h_mut_vec = mutant_h_from_D8(self.D8_dual, self.h_res)

        # Precompute intended cooperation probabilities
        Y1_ss = float(coef16[self.s1_r, 0] + self.h_res * coef16[self.s1_r, 1] + self.h_res * coef16[self.s1_r, 2] + self.h_res * self.h_res * coef16[self.s1_r, 3])
        Y2_ss = float(coef16[self.s2_r, 0] + self.h_res * coef16[self.s2_r, 1] + self.h_res * coef16[self.s2_r, 2] + self.h_res * self.h_res * coef16[self.s2_r, 3])
        self.Y1_tilde_ss = apply_action_error(Y1_ss, mu_e)
        self.Y2_tilde_ss = apply_action_error(Y2_ss, mu_e)

        Y1_res_mut = y_scalar_vec(self.s1_r, self.h_res, self.h_mut_vec)
        Y2_res_mut = y_scalar_vec(self.s2_r, self.h_res, self.h_mut_vec)
        Y1_mut_res = y_vec_scalar(s1_idx, self.h_mut_vec, self.h_res)
        Y2_mut_res = y_vec_scalar(s2_idx, self.h_mut_vec, self.h_res)

        self.Y1_tilde_res_mut = apply_action_error(Y1_res_mut, mu_e)
        self.Y2_tilde_res_mut = apply_action_error(Y2_res_mut, mu_e)
        self.Y1_tilde_mut_res = apply_action_error(Y1_mut_res, mu_e)
        self.Y2_tilde_mut_res = apply_action_error(Y2_mut_res, mu_e)

    def check_ess(self, b1: float, c1: float, b2: float, c2: float, eps: float = 1e-8):
        pi_res = (b1 - c1) * self.Y1_tilde_ss + (b2 - c2) * self.Y2_tilde_ss
        pi_mut = (
            b1 * self.Y1_tilde_res_mut - c1 * self.Y1_tilde_mut_res
            + b2 * self.Y2_tilde_res_mut - c2 * self.Y2_tilde_mut_res
        )
        pi_mut[self.pair_idx] = -np.inf
        max_pi_mut = np.max(pi_mut)
        is_ess = (pi_res > max_pi_mut + eps)
        return is_ess, pi_res

# ============================================================
# Worker for Grid Evaluation
# ============================================================

eval_L8_AND_list = []
eval_S6_OR_list = []

def init_evaluators(mu_e, mu_a):
    global eval_L8_AND_list, eval_S6_OR_list
    eval_L8_AND_list = [
        ConfigEvaluator(norm, pair, "AND", mu_e, mu_a)
        for (norm, _), pair in zip(L8_CONFIGS, l8_pairs_idx)
    ]
    eval_S6_OR_list = [
        ConfigEvaluator(norm, pair, "OR", mu_e, mu_a)
        for (norm, _), pair in zip(S6_CONFIGS, s6_pairs_idx)
    ]

def evaluate_point(args):
    b1, b2, c1, c2, eps = args

    # 1. Evaluate S6 configurations under OR fusion
    s6_is_ess = False
    max_pi_s6 = -np.inf
    for s6_eval in eval_S6_OR_list:
        is_ess, pi = s6_eval.check_ess(b1, c1, b2, c2, eps)
        if is_ess:
            s6_is_ess = True
            if pi > max_pi_s6:
                max_pi_s6 = pi

    # 2. Evaluate all 8 L8 configurations under AND fusion
    l8_is_ess = False
    max_pi_l8 = -np.inf
    for l8_eval in eval_L8_AND_list:
        is_ess, pi = l8_eval.check_ess(b1, c1, b2, c2, eps)
        if is_ess:
            l8_is_ess = True
            if pi > max_pi_l8:
                max_pi_l8 = pi

    # Classification logic matching Figure 1:
    # 0 = White (ALLD only), 1 = Blue (L8 AND is ESS), 2 = Red (S6 OR is ESS & payoff-maximizing)
    if s6_is_ess and (max_pi_s6 >= max_pi_l8 or not l8_is_ess):
        return 2  # Red
    elif l8_is_ess:
        return 1  # Blue
    else:
        return 0  # White

# ============================================================
# Main Plotting Routine
# ============================================================

def generate_figure1(
    grid_size: int = 101,
    mu: float = 0.01,
    c1: float = 1.0,
    c2: float = 1.0,
    output_filename: str = "Figure1_reproduced.png",
    nproc: int = 4
):
    print(f"Generating Figure 1 on {grid_size}x{grid_size} grid (mu={mu}, c1={c1}, c2={c2})...")
    print(f"Checking all {len(L8_CONFIGS)} L8 configs (AND) and all {len(S6_CONFIGS)} S6 configs (OR)...")
    start_time = time.time()

    b_vals = np.linspace(0.0, 10.0, grid_size)
    B1, B2 = np.meshgrid(b_vals, b_vals)
    points = [(float(b1), float(b2), c1, c2, 1e-8) for b1, b2 in zip(B1.ravel(), B2.ravel())]

    with Pool(processes=nproc, initializer=init_evaluators, initargs=(mu, mu)) as pool:
        results = pool.map(evaluate_point, points)

    Z = np.array(results).reshape(grid_size, grid_size)
    elapsed = time.time() - start_time
    print(f"Grid evaluation finished in {elapsed:.1f}s.")

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cmap = plt.matplotlib.colors.ListedColormap(["#ffffff", "#2b5c8f", "#d93829"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(
        Z,
        origin="lower",
        extent=[0, 10, 0, 10],
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest"
    )

    ax.set_xlabel(r"$b^1$", fontsize=13)
    ax.set_ylabel(r"$b^2$", fontsize=13)
    ax.set_xticks(range(0, 11, 1))
    ax.set_yticks(range(0, 11, 1))
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved Figure 1 to {output_filename}")

if __name__ == "__main__":
    mp.freeze_support()
    generate_figure1(grid_size=101, nproc=4)
