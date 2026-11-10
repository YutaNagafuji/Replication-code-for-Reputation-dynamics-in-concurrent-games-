#!/usr/bin/env python
# coding: utf-8
"""
Evolutionary Stability Analysis for Concurrent Games (8-State Formulation)
Reference: "Reputation dynamics in concurrent games" (Nagafuji, Akiyama, and Sawa, 2026)

Key Model Formulations Implemented:
  - Binary reputation dynamics with 8-state fused evaluation D_{h, h1, h2} (Eq. 13-15)
  - Fusion rules: AND (f_A), OR (f_O), and Probabilistic (f_Q)
  - Stationary resident reputation H*(s) solved via cubic polynomial roots (Eq. 16)
    * Note on multiple roots: when multiple real roots exist in [0, 1],
      the branch maximizing resident expected payoff pi(s|s) is tracked.
  - Mutant stationary reputation H*(s') computed as F / (1 - E + F)
  - Execution errors incorporated via affine transformation (Eq. 20):
      \tilde{Y}_i = mu_e + (1 - 2*mu_e) * Y_i
  - Expected payoffs pi(s|s) and pi(s'|s) calculated using realized cooperation (Eq. 21-22)
  - Evolutionary stability condition (Eq. 23): pi(s|s) > max_{s' != s} pi(s'|s)
  - Filtering based on Cooperative and Semi-cooperative configurations (Definition 1)
"""

import csv
import time
import numpy as np
import pandas as pd
from itertools import product
from multiprocessing import Pool
import multiprocessing as mp


# ============================================================
# Utility Functions
# ============================================================

def progress_bar(x: int, total: int, length: int = 28) -> str:
    filled = int(length * x / total)
    return "█" * filled + "-" * (length - filled)


def fmt_float(x: float) -> str:
    s = f"{x:.10g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def default_output_name(
    b1: float, c1: float, b2: float, c2: float,
    mu_e: float, mu_a: float, eps: float,
    fusion_mode: str, Q_list=None, prefix="ESS_8state"
) -> str:
    fusion_mode = fusion_mode.upper()
    q_part = ""
    if fusion_mode == "Q":
        if Q_list is None or len(Q_list) == 0:
            q_part = "_Qnone"
        else:
            q_part = f"_Q{fmt_float(min(Q_list))}to{fmt_float(max(Q_list))}n{len(Q_list)}"
    return (
        f"{prefix}_{fusion_mode}_"
        f"b1_{fmt_float(b1)}_c1_{fmt_float(c1)}_"
        f"b2_{fmt_float(b2)}_c2_{fmt_float(c2)}_"
        f"mue_{fmt_float(mu_e)}_mua_{fmt_float(mu_a)}_"
        f"eps_{fmt_float(eps)}"
        f"{q_part}.csv"
    )


# ============================================================
# Action Strategies (16 per game, 256 strategy pairs)
# Order: (11, 10, 01, 00)
# ============================================================

def build_behavior_tables():
    acts = list(product([1, 0], repeat=4))
    Cbits = np.array(acts, dtype=np.float64)  # shape (16, 4)

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

# Dual strategy indexing (256 pairs): (s1_idx, s2_idx)
s1_idx = np.repeat(np.arange(16, dtype=np.int32), 16)
s2_idx = np.tile(np.arange(16, dtype=np.int32), 16)


# ============================================================
# Social Norms (256 norms)
# Order: d11C, d11D, d10C, d10D, d01C, d01D, d00C, d00D
# ============================================================

d_bits256 = np.array(list(product([0, 1], repeat=8)), dtype=np.int8)
d_bits_str256 = ["".join(str(x) for x in row.tolist()) for row in d_bits256]


# ============================================================
# State Mapping Indices: 4-state Game-specific to 8-state Fused
# D8 order: 111, 110, 101, 100, 011, 010, 001, 000
# D4 order: 11, 10, 01, 00
# ============================================================

D1_INDEX_FROM_D8 = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
D2_INDEX_FROM_D8 = np.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int32)


# ============================================================
# Game-Specific Transition Probabilities (Eq. 11 & 12)
# ============================================================

def precompute_D4(mu_e: float, mu_a: float) -> np.ndarray:
    """Computes single-game reputation transition probabilities D_{h, h_opp}^{s, i} (Eq. 11, 12)."""
    dC = d_bits256[:, [0, 2, 4, 6]].astype(np.float64)  # (256, 4)
    dD = d_bits256[:, [1, 3, 5, 7]].astype(np.float64)  # (256, 4)

    intended = Cbits16[None, :, :]  # (1, 16, 4)
    dC3 = dC[:, None, :]
    dD3 = dD[:, None, :]

    phi = (
        (1.0 - mu_e) * (intended * dC3 + (1.0 - intended) * dD3)
        + mu_e * (intended * dD3 + (1.0 - intended) * dC3)
    )
    D4 = (1.0 - 2.0 * mu_a) * phi + mu_a
    return D4  # shape: (256 norms, 16 strategies, 4 states)


# ============================================================
# Reputation Fusion (Eq. 13, 14, 15)
# ============================================================

def fuse_to_D8(D1_4: np.ndarray, D2_4: np.ndarray, fusion_mode: str, Q: float = None) -> np.ndarray:
    """Aggregates game-specific reputations into a unified 8-state reputation."""
    fusion_mode = fusion_mode.upper()

    D1_8 = D1_4[..., D1_INDEX_FROM_D8]
    D2_8 = D2_4[..., D2_INDEX_FROM_D8]

    if fusion_mode == "Q":
        if Q is None:
            raise ValueError("Q must be specified when fusion_mode='Q'.")
        # Eq. 15: (1 - Q) * D1 + Q * D2
        return (1.0 - Q) * D1_8 + Q * D2_8

    if fusion_mode == "AND":
        # Eq. 13: D1 * D2
        return D1_8 * D2_8

    if fusion_mode == "OR":
        # Eq. 14: 1 - (1 - D1) * (1 - D2)
        return 1.0 - (1.0 - D1_8) * (1.0 - D2_8)

    raise ValueError("fusion_mode must be 'AND', 'OR', or 'Q'.")


# ============================================================
# Resident Stationary Distribution H*(s) (Eq. 16)
# ============================================================

def cubic_coefficients_from_D8(D8: np.ndarray):
    D111, D110, D101, D100, D011, D010, D001, D000 = D8
    A = D111 - D110 - D101 + D100 - D011 + D010 + D001 - D000
    B = D110 + D101 - 2.0 * D100 + D011 - 2.0 * D010 - 2.0 * D001 + 3.0 * D000
    C = D100 + D010 + D001 - 3.0 * D000 - 1.0
    D = D000
    return A, B, C, D


def real_roots_in_unit_interval(coeffs, imag_tol: float = 1e-8, interval_tol: float = 1e-9) -> np.ndarray:
    coeffs = np.array(coeffs, dtype=np.float64)
    nz = np.where(np.abs(coeffs) > 1e-14)[0]
    if len(nz) == 0:
        return np.array([0.5], dtype=np.float64)
    coeffs = coeffs[nz[0]:]

    roots = np.roots(coeffs)
    real = roots[np.abs(roots.imag) < imag_tol].real
    real = real[(real >= -interval_tol) & (real <= 1.0 + interval_tol)]
    real = np.clip(real, 0.0, 1.0)

    if len(real) == 0:
        return np.array([], dtype=np.float64)

    real = np.sort(real)
    unique = []
    for x in real:
        if not unique or abs(x - unique[-1]) > 1e-7:
            unique.append(float(x))
    return np.array(unique, dtype=np.float64)


def resident_roots_from_D8(D8: np.ndarray) -> np.ndarray:
    coeffs = cubic_coefficients_from_D8(D8)
    return real_roots_in_unit_interval(coeffs)


# ============================================================
# Mutant Stationary Distribution H*(s')
# ============================================================

def mutant_h_from_D8(D8_mut: np.ndarray, h_res: float) -> np.ndarray:
    D = np.asarray(D8_mut, dtype=np.float64)
    D111, D110, D101, D100 = D[..., 0], D[..., 1], D[..., 2], D[..., 3]
    D011, D010, D001, D000 = D[..., 4], D[..., 5], D[..., 6], D[..., 7]

    h = h_res
    one_minus_h = 1.0 - h

    E_good = (
        h * h * D111
        + h * one_minus_h * D110
        + one_minus_h * h * D101
        + one_minus_h * one_minus_h * D100
    )

    F_bad = (
        h * h * D011
        + h * one_minus_h * D010
        + one_minus_h * h * D001
        + one_minus_h * one_minus_h * D000
    )

    denom = 1.0 - E_good + F_bad
    out = np.empty_like(denom, dtype=np.float64)
    mask = np.abs(denom) < 1e-15
    out[mask] = np.where(F_bad[mask] > 0.5, 1.0, 0.0)
    out[~mask] = F_bad[~mask] / denom[~mask]
    return np.clip(out, 0.0, 1.0)


# ============================================================
# Cooperation Probabilities (Eq. 17-20)
# ============================================================

def y_row_scalar_col_vec(row_idx: int, h_row: float, h_col_vec: np.ndarray) -> np.ndarray:
    c00, u, v, w = coef16[row_idx]
    return c00 + h_row * u + h_col_vec * v + (h_row * h_col_vec) * w


def y_row_vec_col_scalar(row_idx_vec: np.ndarray, h_row_vec: np.ndarray, h_col: float) -> np.ndarray:
    c00 = coef16[row_idx_vec, 0]
    u = coef16[row_idx_vec, 1]
    v = coef16[row_idx_vec, 2]
    w = coef16[row_idx_vec, 3]
    return c00 + h_row_vec * u + h_col * v + (h_row_vec * h_col) * w


def y_row_vec_col_vec(row_idx_vec: np.ndarray, h_row_vec: np.ndarray, h_col_vec: np.ndarray) -> np.ndarray:
    c00 = coef16[row_idx_vec, 0]
    u = coef16[row_idx_vec, 1]
    v = coef16[row_idx_vec, 2]
    w = coef16[row_idx_vec, 3]
    return c00 + h_row_vec * u + h_col_vec * v + (h_row_vec * h_col_vec) * w


def apply_action_error(intended_Y, mu_e: float):
    """
    Realized probability of cooperation \tilde{Y}_i (Eq. 20):
        \tilde{Y}_i = mu_e + (1 - 2*mu_e) * Y_i
    """
    return mu_e + (1.0 - 2.0 * mu_e) * intended_Y


# ============================================================
# Resident Steady-State Selection (Handling Multiple Roots)
# ============================================================

def choose_resident_h_by_max_pi(D8_res: np.ndarray, s1_r: int, s2_r: int,
                                b1: float, c1: float, b2: float, c2: float,
                                mu_e: float):
    """
    Evaluates real roots in [0, 1] of the stationary characteristic polynomial (Eq. 16).
    If multiple roots exist (e.g. multistability), selects the branch that maximizes pi(s|s).
    Expected payoffs strictly follow Eq. 20 and Eq. 21 using realized cooperation.
    """
    roots = resident_roots_from_D8(D8_res)

    if len(roots) == 0:
        grid = np.linspace(0.0, 1.0, 1001)
        A, B, C, D = cubic_coefficients_from_D8(D8_res)
        vals = np.abs(A * grid**3 + B * grid**2 + C * grid + D)
        roots = np.array([float(grid[np.argmin(vals)])])

    best_h = None
    best_pi = -np.inf
    best_pi_1 = None
    best_pi_2 = None

    for h in roots:
        # Intended cooperation Y_i(s, s) (Eq. 17)
        Y1_ss = y_row_vec_col_vec(np.array([s1_r]), np.array([h]), np.array([h]))[0]
        Y2_ss = y_row_vec_col_vec(np.array([s2_r]), np.array([h]), np.array([h]))[0]

        # Realized cooperation \tilde{Y}_i(s, s) (Eq. 20)
        Y1_tilde_ss = apply_action_error(Y1_ss, mu_e)
        Y2_tilde_ss = apply_action_error(Y2_ss, mu_e)

        # Expected resident payoff pi(s|s) (Eq. 21)
        pi_1 = (b1 - c1) * Y1_tilde_ss
        pi_2 = (b2 - c2) * Y2_tilde_ss
        pi_tot = pi_1 + pi_2

        if pi_tot > best_pi + 1e-12:
            best_h = float(h)
            best_pi = float(pi_tot)
            best_pi_1 = float(pi_1)
            best_pi_2 = float(pi_2)

    return best_h, best_pi, best_pi_1, best_pi_2, roots


# ============================================================
# Parallel Worker for a Single Social Norm
# ============================================================

def worker_one_norm(args):
    (d_idx, D4_d_16x4, b1, c1, b2, c2, mu_e, eps, fusion_mode, Q_values) = args

    out_rows = []
    norm_d_str = d_bits_str256[d_idx]
    fusion_mode = fusion_mode.upper()
    d11_C_is_1 = (norm_d_str[0] == "1")

    # Baseline theoretical benchmark for fully cooperative state
    denom_relative = (b1 - c1) + (b2 - c2)

    D1_dual = D4_d_16x4[s1_idx]  # (256, 4)
    D2_dual = D4_d_16x4[s2_idx]  # (256, 4)

    for Q in Q_values:
        D8_dual = fuse_to_D8(D1_dual, D2_dual, fusion_mode=fusion_mode, Q=Q)  # (256, 8)

        for r in range(256):
            s1_r = int(s1_idx[r])
            s2_r = int(s2_idx[r])
            s1_str = s_bits16[s1_r]
            s2_str = s_bits16[s2_r]

            # Resident stationary reputation and payoff
            h_res, pi_res, pi_1_res, pi_2_res, h_roots = choose_resident_h_by_max_pi(
                D8_dual[r], s1_r, s2_r, b1, c1, b2, c2, mu_e
            )

            # Mutant stationary reputations H*(s')
            h_mut_vec = mutant_h_from_D8(D8_dual, h_res)  # (256,)

            # Intended cooperation probabilities:
            # Resident -> Mutant (Eq. 18): resident evaluates mutant
            Y1_res_mut = y_row_scalar_col_vec(s1_r, h_res, h_mut_vec)
            Y2_res_mut = y_row_scalar_col_vec(s2_r, h_res, h_mut_vec)

            # Mutant -> Resident (Eq. 19): mutant evaluates resident
            Y1_mut_res = y_row_vec_col_scalar(s1_idx, h_mut_vec, h_res)
            Y2_mut_res = y_row_vec_col_scalar(s2_idx, h_mut_vec, h_res)

            # Realized cooperation probabilities taking execution errors into account (Eq. 20)
            Y1_tilde_res_mut = apply_action_error(Y1_res_mut, mu_e)
            Y2_tilde_res_mut = apply_action_error(Y2_res_mut, mu_e)
            Y1_tilde_mut_res = apply_action_error(Y1_mut_res, mu_e)
            Y2_tilde_mut_res = apply_action_error(Y2_mut_res, mu_e)

            # Expected mutant payoff pi(s'|s) (Eq. 22)
            pi_mut_vec = (
                b1 * Y1_tilde_res_mut - c1 * Y1_tilde_mut_res
                + b2 * Y2_tilde_res_mut - c2 * Y2_tilde_mut_res
            )

            # Exclude self-comparison
            pi_mut_vec[r] = -np.inf
            pi_mut_max = float(np.max(pi_mut_vec))
            argmax_mut = int(np.argmax(pi_mut_vec))
            stability_margin = float(pi_res - pi_mut_max)

            # Evolutionary Stability condition (Definition 2, Eq. 23)
            if pi_res > pi_mut_max + eps:
                rel = (pi_res / denom_relative) if denom_relative != 0 else np.nan

                # Classification based on Definition 1
                is_coop = d11_C_is_1 and (s1_str[0] == "C") and (s2_str[0] == "C")
                is_semi_coop = d11_C_is_1 and ((s1_str[0] == "C") or (s2_str[0] == "C"))

                out_rows.append({
                    "fusion_rule": fusion_mode,
                    "Q": "" if fusion_mode != "Q" else Q,
                    "norm_d": norm_d_str,
                    "strategy_s1": s1_str,
                    "strategy_s2": s2_str,
                    "is_cooperative": is_coop,
                    "is_semi_cooperative": is_semi_coop,
                    "payoff_resident": pi_res,
                    "payoff_resident_game1": pi_1_res,
                    "payoff_resident_game2": pi_2_res,
                    "payoff_resident_relative": rel,
                    "H_star_resident": h_res,
                    "H_roots_all": ";".join(f"{x:.12g}" for x in h_roots),
                    "num_H_roots": len(h_roots),
                    "payoff_mutant_max": pi_mut_max,
                    "stability_margin": stability_margin,
                    "worst_mutant_s1": s_bits16[int(s1_idx[argmax_mut])],
                    "worst_mutant_s2": s_bits16[int(s2_idx[argmax_mut])],
                    "H_star_worst_mutant": float(h_mut_vec[argmax_mut]),
                })

    return out_rows


# ============================================================
# Main ESS Enumeration Routine
# ============================================================

def enumerate_all_dual_ESS_8state(
    b1: float, c1: float, b2: float, c2: float,
    mu_e: float, mu_a: float,
    fusion_mode: str = "AND",
    Q_list=None,
    eps: float = 1e-8,
    output_csv: str = None,
    nproc: int = 3,
    progress_every: int = 8,
) -> str:
    start_time = time.time()
    fusion_mode = fusion_mode.upper()

    if fusion_mode == "Q":
        if Q_list is None:
            Q_list = [round(i * 0.01, 2) for i in range(101)]
        Q_values = list(Q_list)
    else:
        Q_values = [None]

    if output_csv is None:
        output_csv = default_output_name(b1, c1, b2, c2, mu_e, mu_a, eps, fusion_mode, Q_values)

    nproc = max(1, int(nproc))

    print("\n=======================================================")
    print(" Concurrent Games ESS Enumeration (8-State Formulation)")
    print("=======================================================")
    print(f"Fusion Rule    : {fusion_mode}")
    if fusion_mode == "Q":
        print(f"Probabilistic Q: game1 prob = 1-Q, game2 prob = Q")
        print(f"Q values count : {len(Q_values)}")
    print(f"Payoff Game 1  : b1 = {b1}, c1 = {c1}")
    print(f"Payoff Game 2  : b2 = {b2}, c2 = {c2}")
    print(f"Error Rates    : mu_e = {mu_e}, mu_a = {mu_a}")
    print(f"Strictness eps : {eps}")
    print(f"Output File    : {output_csv}")
    print("-------------------------------------------------------")

    print("[Preprocessing] Precomputing D4 single-game transition matrix...")
    D4 = precompute_D4(mu_e, mu_a)

    tasks = [
        (d_idx, D4[d_idx], b1, c1, b2, c2, mu_e, eps, fusion_mode, Q_values)
        for d_idx in range(256)
    ]
    total = len(tasks)

    fieldnames = [
        "fusion_rule", "Q",
        "norm_d", "strategy_s1", "strategy_s2",
        "is_cooperative", "is_semi_cooperative",
        "payoff_resident", "payoff_resident_game1", "payoff_resident_game2",
        "payoff_resident_relative",
        "H_star_resident", "H_roots_all", "num_H_roots",
        "payoff_mutant_max", "stability_margin",
        "worst_mutant_s1", "worst_mutant_s2", "H_star_worst_mutant",
    ]

    print(f"[Execution] Starting pool with {nproc} worker processes...")

    rows_written = 0
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if nproc == 1:
            iterator = map(worker_one_norm, tasks)
            for k, rows in enumerate(iterator, 1):
                for row in rows:
                    writer.writerow(row)
                rows_written += len(rows)

                if (k % progress_every) == 0 or k == total:
                    elapsed = time.time() - start_time
                    progress = k / total
                    eta = (elapsed / progress - elapsed) if progress > 0 else 0
                    print(
                        f"\r[{progress_bar(k, total)}] {progress*100:5.1f}% | "
                        f"Elapsed: {elapsed/60:4.1f}m | ETA: {eta/60:4.1f}m | ESCs found: {rows_written}",
                        end="", flush=True
                    )
        else:
            with Pool(processes=nproc) as pool:
                for k, rows in enumerate(pool.imap_unordered(worker_one_norm, tasks), 1):
                    for row in rows:
                        writer.writerow(row)
                    rows_written += len(rows)

                    if (k % progress_every) == 0 or k == total:
                        elapsed = time.time() - start_time
                        progress = k / total
                        eta = (elapsed / progress - elapsed) if progress > 0 else 0
                        print(
                            f"\r[{progress_bar(k, total)}] {progress*100:5.1f}% | "
                            f"Elapsed: {elapsed/60:4.1f}m | ETA: {eta/60:4.1f}m | ESCs found: {rows_written}",
                            end="", flush=True
                        )

    print(f"\n[Completed] Raw ESC search finished -> {output_csv} (Total records: {rows_written})")
    return output_csv


# ============================================================
# Post-Processing: Filtering Based on Definition 1
# ============================================================

def filter_ESS_configurations(
    input_csv: str,
    filter_mode: str = "cooperative",
) -> str:
    """
    Filters and sorts the enumerated evolutionarily stable configurations.

    Parameters:
      filter_mode:
        - 'cooperative'     : Keeps configurations where d_{11}^C = 1 and s_{11}^1 = s_{11}^2 = C (Definition 1).
        - 'semi_cooperative': Keeps configurations where d_{11}^C = 1 and s_{11}^i = C for at least one game (Definition 1).
        - 'all'             : Retains all ESCs without behavioral filtering.
    """
    print(f"\n[Filtering] Processing results from {input_csv} with filter_mode='{filter_mode}'...")
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    numeric_cols = [
        "payoff_resident", "payoff_resident_game1", "payoff_resident_game2",
        "payoff_resident_relative", "H_star_resident", "num_H_roots",
        "payoff_mutant_max", "stability_margin", "H_star_worst_mutant",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Q" in df.columns:
        df["Q"] = pd.to_numeric(df["Q"], errors="coerce")

    df["norm_d"] = df["norm_d"].astype(str).str.zfill(8)

    # Boolean flags
    df["is_cooperative"] = df["is_cooperative"].astype(str).str.lower() == "true"
    df["is_semi_cooperative"] = df["is_semi_cooperative"].astype(str).str.lower() == "true"

    if filter_mode == "cooperative":
        df = df.loc[df["is_cooperative"]]
    elif filter_mode == "semi_cooperative":
        df = df.loc[df["is_semi_cooperative"]]
    elif filter_mode == "all":
        pass
    else:
        raise ValueError("filter_mode must be 'cooperative', 'semi_cooperative', or 'all'.")

    # Sort primarily by fusion rule, Q (if present), and descending resident payoff
    sort_cols = ["fusion_rule"]
    if df["Q"].notna().any():
        sort_cols.append("Q")
    sort_cols.append("payoff_resident")

    ascending = [True] * len(sort_cols)
    ascending[-1] = False
    df = df.sort_values(by=sort_cols, ascending=ascending).drop_duplicates()

    suffix = f"_{filter_mode}.csv"
    out_csv = input_csv.replace(".csv", suffix)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[Filtering] Saved filtered records -> {out_csv} ({len(df)} configurations)")
    return out_csv


# ============================================================
# Entry Point
# ============================================================

def main():
    # ---- Model & Payoff Parameters ----
    # Fusion rule: "AND", "OR", or "Q"
    fusion_mode = "AND"

    # Payoffs for Game 1 and Game 2 (Table 1)
    b1, c1 = 2.1, 1.0
    b2, c2 = 2.1, 1.0

    # Noise parameters (Section 2.4): mu_e = mu_a = mu
    mu_e, mu_a = 0.01, 0.01
    eps = 1e-8

    # Q values: Used only when fusion_mode="Q" (Eq. 7, Section 3.2)
    Q_list = [round(i * 0.01, 2) for i in range(101)]

    # Parallelism
    nproc = 3

    # Output file
    output_csv = default_output_name(
        b1, c1, b2, c2, mu_e, mu_a, eps,
        fusion_mode=fusion_mode,
        Q_list=Q_list,
        prefix="ESS_8state"
    )

    # 1. Enumerate all ESCs
    raw_csv = enumerate_all_dual_ESS_8state(
        b1=b1, c1=c1,
        b2=b2, c2=c2,
        mu_e=mu_e, mu_a=mu_a,
        fusion_mode=fusion_mode,
        Q_list=Q_list,
        eps=eps,
        output_csv=output_csv,
        nproc=nproc,
        progress_every=8,
    )

    # 2. Filter configurations strictly according to Definition 1
    filtered_csv = filter_ESS_configurations(
        raw_csv,
        filter_mode="cooperative",
    )

    print("\nSummary of output files:")
    print("  Full ESC records    :", raw_csv)
    print("  Cooperative records :", filtered_csv)


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
