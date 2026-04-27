"""Single-config and full-config-list runners for the fragmentation_ngram
experiment (paper Theorem 3.1).

For a fixed (k, M) pair, `run_single_config` runs n_matrices independent
trials. Each trial:
  1. Samples a random k-th order Markov transition matrix over V_y = 2**M.
  2. Generates a Y sequence of length n_sequence.
  3. Fragments Y into binary tokens X.
  4. Fits an n-gram predictor on Y (context K).
  5. Fits an n-gram predictor on X (context K*M).
  6. Computes the theoretical fragmentation penalty MI from the empirical
     marginal over (K+1)-symbol windows.

The empirical penalty loss_X_ngram_per_Y - loss_Y_ngram should match the
theoretical phase_MI + oracle_residual decomposition -- this is the core
validation of Theorem 3.1.

`run_from_config` is the top-level orchestrator: loops over all (k, M) pairs
listed in cfg["source"]["configurations"], saves the figure / CSV / raw JSON,
and runs the validation checks.
"""

import json
import os
import time

import numpy as np

from core.markov import (
    fragment,
    generate_markov_sequence,
    sample_transition_matrix,
)
from core.ngram import ngram_cross_entropy_bits
from core.theory import (
    H_Y_theoretical,
    build_joint_table,
    compute_losses_from_table,
    estimate_marginal_Y,
)

from .plot import plot_fig1


def run_single_config(k, M, cfg):
    """Run the full Theorem 3.1 experiment for a single (k, M) configuration.

    Parameters
    ----------
    k : int
        Markov order of the source on Y.
    M : int
        Fragmentation factor; each Y symbol maps to M binary tokens.
    cfg : dict
        Parsed configuration (see configs/fragmentation_ngram/base.yaml).

    Returns
    -------
    dict with keys:
        k, M, K, V_y, trials (list of trial dicts), mean (dict), std (dict).
    """
    # K resolves to, in priority order:
    #   1. cfg["ngram"]["K"]            -- explicit value
    #   2. k + cfg["ngram"]["K_offset"] -- relative to the Markov order (default 0)
    if cfg["ngram"].get("K") is not None:
        K = int(cfg["ngram"]["K"])
    else:
        K = k + int(cfg["ngram"].get("K_offset", 0))
    V_x = int(cfg["source"]["V_x"])
    V_y = V_x ** M

    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    n_matrices = int(cfg["sampling"]["n_matrices"])
    alpha = float(cfg["sampling"]["alpha"])
    n_sequence = int(cfg["sampling"]["n_sequence"])
    smoothing_alpha = float(cfg["ngram"]["smoothing_alpha"])

    trials = []
    for _trial_idx in range(n_matrices):
        P = sample_transition_matrix(V_y, k, rng, alpha=alpha)
        H_Y_th = H_Y_theoretical(P, V_y, k)

        Y = generate_markov_sequence(P, V_y, k, n_sequence, rng)
        X = fragment(Y, M)

        loss_Y_ngram = ngram_cross_entropy_bits(
            Y, context_len=K, vocab_size=V_y, smoothing_alpha=smoothing_alpha
        )

        loss_X_ngram = ngram_cross_entropy_bits(
            X, context_len=K * M, vocab_size=V_x, smoothing_alpha=smoothing_alpha
        )
        loss_X_ngram_per_Y = loss_X_ngram * M

        marginal = estimate_marginal_Y(Y, K, V_y)
        table = build_joint_table(marginal, V_y, K, M)
        theory_out = compute_losses_from_table(table, M)
        th_L_X_stat_per_Y = theory_out["L_X_stat"] * M
        th_L_X_orc_per_Y = theory_out["L_X_orc"] * M
        # th_penalty_per_Y is the phase-MI component of the fragmentation
        # penalty: bits lost because the predictor is blind to theta.
        th_penalty_per_Y = theory_out["MI"] * M
        # oracle_residual_per_Y is the second component: bits lost because the
        # K*M-bit context at phase theta > 0 is misaligned with full Y symbols.
        oracle_residual_per_Y = th_L_X_orc_per_Y - H_Y_th
        # Total theoretical penalty: the sum of both components equals
        # L_X_stat*M - H_Y_th, which is what the empirical gap converges to.
        th_total_penalty_per_Y = th_penalty_per_Y + oracle_residual_per_Y

        emp_penalty_per_Y = loss_X_ngram_per_Y - loss_Y_ngram

        trials.append(
            {
                "H_Y_th": float(H_Y_th),
                "loss_Y_ngram": float(loss_Y_ngram),
                "loss_X_ngram_per_Y": float(loss_X_ngram_per_Y),
                "th_L_X_stat_per_Y": float(th_L_X_stat_per_Y),
                "th_L_X_orc_per_Y": float(th_L_X_orc_per_Y),
                "th_penalty_per_Y": float(th_penalty_per_Y),
                "oracle_residual_per_Y": float(oracle_residual_per_Y),
                "th_total_penalty_per_Y": float(th_total_penalty_per_Y),
                "emp_penalty_per_Y": float(emp_penalty_per_Y),
            }
        )

    keys = list(trials[0].keys())
    mean = {key: float(np.mean([t[key] for t in trials])) for key in keys}
    std = {key: float(np.std([t[key] for t in trials], ddof=0)) for key in keys}

    return {
        "k": k,
        "M": M,
        "K": K,
        "V_y": V_y,
        "trials": trials,
        "mean": mean,
        "std": std,
    }


def print_summary_table(all_results):
    """Two-part summary.

    Part A: convergence of each n-gram estimator to its theoretical target.
    Part B: decomposition of the empirical gap into phase MI + oracle residual,
            vs the empirical loss_X - loss_Y.
    """
    print("=" * 90)
    print("A. Convergence of empirical estimators")
    print("=" * 90)
    head_a = (
        f"{'k':>2} {'M':>2} | "
        f"{'loss_Y':>8} {'H_Y_th':>8} {'dY':>8} | "
        f"{'loss_X/Y':>9} {'L_Xstat*M':>10} {'dX':>8}"
    )
    print(head_a)
    print("-" * len(head_a))
    for r in all_results:
        m = r["mean"]
        dY = m["loss_Y_ngram"] - m["H_Y_th"]
        dX = m["loss_X_ngram_per_Y"] - m["th_L_X_stat_per_Y"]
        print(
            f"{r['k']:>2} {r['M']:>2} | "
            f"{m['loss_Y_ngram']:>8.4f} {m['H_Y_th']:>8.4f} {dY:>+8.4f} | "
            f"{m['loss_X_ngram_per_Y']:>9.4f} {m['th_L_X_stat_per_Y']:>10.4f} {dX:>+8.4f}"
        )

    print()
    print("=" * 90)
    print("B. Decomposition of the empirical gap  (emp  vs  phase_MI + orc_res)")
    print("=" * 90)
    head_b = (
        f"{'k':>2} {'M':>2} | "
        f"{'phase_MI':>9} {'orc_res':>9} {'th_total':>9} | "
        f"{'emp':>9} | {'rel_err_%':>10}"
    )
    print(head_b)
    print("-" * len(head_b))
    for r in all_results:
        m = r["mean"]
        phase_MI = m["th_penalty_per_Y"]
        orc_res = m["oracle_residual_per_Y"]
        th_total = m["th_total_penalty_per_Y"]
        emp = m["emp_penalty_per_Y"]
        rel = abs(emp - th_total) / emp * 100.0 if abs(emp) > 1e-12 else float("nan")
        print(
            f"{r['k']:>2} {r['M']:>2} | "
            f"{phase_MI:>9.4f} {orc_res:>9.4f} {th_total:>9.4f} | "
            f"{emp:>9.4f} | {rel:>10.3f}"
        )


def print_checks(all_results):
    """Verify the theoretical decomposition matches the empirical gap.

    Checks (per configuration):
      1. phase_MI > 0 and oracle_residual > 0 (both components are real).
      2. |emp - (phase_MI + oracle_residual)| / emp < 0.01 (decomposition holds).
      3. |loss_Y_ngram - H_Y_th| < 0.01 (Y n-gram reaches the true conditional entropy).
    """
    print("\nTheorem 3.1 validation checks:")
    all_pass = True
    for r in all_results:
        m = r["mean"]
        label = f"k={r['k']},M={r['M']}"
        phase_MI = m["th_penalty_per_Y"]
        orc_res = m["oracle_residual_per_Y"]
        th_total = m["th_total_penalty_per_Y"]
        emp = m["emp_penalty_per_Y"]
        rel_err = abs(emp - th_total) / emp if abs(emp) > 1e-12 else float("inf")
        ngram_Y_err = abs(m["loss_Y_ngram"] - m["H_Y_th"])

        c_MI_pos = phase_MI > 0
        # Oracle residual is theoretically >= 0 (conditioning inequality);
        # allow a small negative tolerance for finite-sample noise in the
        # empirical marginal that feeds the joint table.
        c_res_nonneg = orc_res > -0.01
        c_emp_pos = emp > 0
        c_decomp = rel_err < 0.01
        c_ngram_Y = ngram_Y_err < 0.01

        ok = c_MI_pos and c_res_nonneg and c_emp_pos and c_decomp and c_ngram_Y
        all_pass = all_pass and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {label:<10} "
            f"phase_MI={phase_MI:.4f} orc_res={orc_res:.4f} "
            f"th_total={th_total:.4f} emp={emp:.4f} rel_err={rel_err*100:.3f}%"
        )
        if not c_MI_pos:
            print("    - phase_MI not > 0")
        if not c_res_nonneg:
            print(f"    - oracle_residual = {orc_res:.4f} < -0.01")
        if not c_emp_pos:
            print("    - emp_penalty not > 0")
        if not c_decomp:
            print(
                f"    - decomposition mismatch: |emp - th_total| / emp "
                f"= {rel_err*100:.3f}% >= 1%"
            )
        if not c_ngram_Y:
            print(
                f"    - loss_Y_ngram {m['loss_Y_ngram']:.4f} differs from "
                f"H_Y_th {m['H_Y_th']:.4f} by {ngram_Y_err:.4f}"
            )

    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


def run_from_config(cfg, tag, out_dir, verbose=True):
    """Run all (k, M) configurations, print tables, save figure/CSV/JSON, run checks.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config.
    tag : str
        Run tag used as a filename stem (e.g. "fragmentation_ngram_base").
    out_dir : str
        Experiment output root. Subdirs `figures/` and `raw/` are created
        underneath if they don't exist.
    verbose : bool
        If True, print summary tables and a Theorem 3.1 validation check.

    Returns
    -------
    (all_results, fig_paths, raw_path)
    """
    fig_dir = os.path.join(out_dir, "figures")
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    t_start = time.time()
    np.random.seed(int(cfg["seed"]))

    all_results = []
    for k, M in cfg["source"]["configurations"]:
        if verbose:
            print(f"Running k={k}, M={M} ...", flush=True)
        t0 = time.time()
        result = run_single_config(int(k), int(M), cfg)
        dt = time.time() - t0
        if verbose:
            print(f"  K={result['K']}, done in {dt:.1f}s", flush=True)
        all_results.append(result)

    if verbose:
        print()
        print_summary_table(all_results)

    fig_stem = f"fig1_{tag}"
    paths = plot_fig1(all_results, fig_dir, file_stem=fig_stem)
    raw_path = os.path.join(raw_dir, f"{tag}_results.json")
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\nFigure saved: {paths['pdf']}")
        print(f"              {paths['png']}")
        print(f"CSV saved:    {paths['csv']}")
        print(f"Raw results:  {raw_path}")
        print_checks(all_results)
        print(f"\nTotal runtime: {time.time() - t_start:.1f}s")

    return all_results, paths, raw_path
