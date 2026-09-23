"""
marginal_gain_suite.py — Stage 3b exhaustive N×N marginal-gain analysis
------------------------------------------------------------------------
Runs marginal-gain circuit construction for every ordered pair (H_A, H_B)
from a completed run_suite.py suite, producing N×N result matrices.

The model is loaded ONCE and reused across all pairs — the total compute is
O(N² × |candidates|) forward passes with a single model load, comparable to
run_suite.py's total compute since marginal-gain operates on already-pruned
circuits (≤200 edges) rather than the full 1358-edge graph.

Conditions with zero edges (vary_length_doc_desc) are skipped as both H_A
and H_B since they contribute nothing to either role.

Usage:
  python marginal_gain_suite.py runs/SUITE_<timestamp>_docstring.json
  python marginal_gain_suite.py runs/SUITE_<timestamp>_docstring.json \\
      --threshold 0.067 --device cpu --seed 0

Outputs (all named after the input suite):
  marginal_gain_suite_<suite_stem>.json        — full per-pair results
  marginal_gain_delta_f1_<suite_stem>.json     — N×N ΔF1 matrix
  marginal_gain_delta_recall_<suite_stem>.json — N×N ΔRecall matrix
  marginal_gain_mode_a_<suite_stem>.json       — N×N Mode A KL improvement matrix
"""

import argparse
import gc
import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import torch

from marginal_gain import (
    load_edge_set,
    mode_b,
    setup_evaluator,
    evaluate_mode_a,
    run_marginal_gain,
)
from acdc.docstring.utils import get_docstring_subgraph_true_edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run marginal-gain on all ordered pairs from a run_suite.py suite."
    )
    parser.add_argument("suite_json",
        help="Path to SUITE_*.json produced by run_suite.py")
    parser.add_argument("--threshold", type=float, default=0.067,
        help="Marginal-gain KL improvement threshold (default: 0.067)")
    parser.add_argument("--device", default="cpu",
        help="Device for model forward passes (default: cpu)")
    parser.add_argument("--seed", type=int, default=0,
        help="Which seed to use per condition (default: 0)")
    parser.add_argument("--out-dir", default="analysis/stage3b_marginal_gain",
        help="Directory for output files (default: analysis/stage3b_marginal_gain/)")
    parser.add_argument("--runs-root", default=None,
        help="Override run_dir base path (for suites generated on a different machine)")
    args = parser.parse_args()

    suite_path = Path(args.suite_json)
    suite_stem = suite_path.stem
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load suite ---
    data    = json.loads(suite_path.read_text(encoding="utf-8"))
    rows    = data["runs"] if isinstance(data, dict) else data
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if len(rows) != len(ok_rows):
        print(f"Note: {len(rows) - len(ok_rows)} error row(s) excluded.")

    def resolve_run_dir(run_dir_str: str) -> Path:
        p = Path(run_dir_str)
        if args.runs_root:
            parts = p.parts
            try:
                idx = list(parts).index("runs")
                return Path(args.runs_root) / Path(*parts[idx:])
            except ValueError:
                return Path(args.runs_root) / Path(p.name)
        return p

    by_condition: Dict[str, list] = defaultdict(list)
    for row in ok_rows:
        by_condition[row["condition"]].append(row)

    def pick_row(cond: str) -> Optional[dict]:
        cond_rows = by_condition.get(cond)
        if not cond_rows:
            return None
        by_seed = {r["seed"]: r for r in cond_rows}
        row = by_seed.get(args.seed) or sorted(cond_rows, key=lambda r: r["seed"])[0]
        if row["seed"] != args.seed:
            print(f"  Warning: seed {args.seed} not found for '{cond}', using seed {row['seed']}.")
        return row

    # --- Load all edge sets; skip zero-edge conditions ---
    print("\nLoading edge sets...")
    all_conditions = sorted(by_condition.keys())
    edge_sets: Dict[str, Set[str]] = {}
    for cond in all_conditions:
        row = pick_row(cond)
        if row is None:
            continue
        es = load_edge_set(resolve_run_dir(row["run_dir"]))
        if len(es) == 0:
            print(f"  Skipping '{cond}' — zero edges.")
            continue
        edge_sets[cond] = es
        print(f"  {cond}: {len(es)} edges")

    conditions = sorted(edge_sets.keys())
    n = len(conditions)
    print(f"\n{n} conditions with edges: {conditions}")

    # --- Ground truth ---
    gt: Set[str] = {repr(k) for k in get_docstring_subgraph_true_edges()}
    print(f"Ground truth: {len(gt)} edges")

    # --- Single-pass Mode B baselines ---
    single: Dict[str, dict] = {c: mode_b(edge_sets[c], gt) for c in conditions}
    print("\n=== Single-pass Mode B baselines ===")
    print(f"{'condition':38s} {'edges':>6} {'P':>7} {'R':>7} {'F1':>7}")
    for c in conditions:
        s = single[c]
        print(f"{c:38s} {s['edges']:>6} {s['precision']:>7.4f} {s['recall']:>7.4f} {s['f1']:>7.4f}")

    # Helper used by both naive-union and marginal-gain sections
    col_w = 10
    def print_matrix(title, mat):
        print(f"\n=== {title} ===")
        print(f"{'':38s}" + "".join(f"{c[:col_w]:>{col_w}s}" for c in conditions))
        for i, c in enumerate(conditions):
            cells = "".join(
                f"{mat[i,j]:>{col_w}.4f}" if not np.isnan(mat[i,j]) else f"{'—':>{col_w}s}"
                for j in range(n)
            )
            print(f"{c:38s}{cells}")

    def save_matrix(name, mat):
        p = out_dir / f"{name}_{suite_stem}.json"
        p.write_text(json.dumps({"conditions": conditions, "matrix": mat.tolist()}, indent=2), encoding="utf-8")
        print(f"{name} saved: {p}")

    # --- One-time Mode A setup ---
    print(f"\nSetting up Mode A evaluator (device={args.device}, threshold={args.threshold})...")
    exp, model, clean_ds, metric, edge_key_to_edge = setup_evaluator(args.device)

    # Mode A KL for each single-pass circuit
    kl_single: Dict[str, float] = {}
    print("\nMode A KL — single-pass circuits:")
    for c in conditions:
        kl = evaluate_mode_a(edge_sets[c], model, clean_ds, metric, edge_key_to_edge)
        kl_single[c] = kl
        print(f"  {c:38s}  KL={kl:.6f}")

    # --- N×N marginal-gain loop ---
    # --- Naive union N×N (no model needed) ---
    print("\n=== Naive union ΔRecall matrix (pre-computed, no model) ===")
    naive_delta_recall = np.zeros((n, n))
    naive_delta_f1     = np.zeros((n, n))
    naive_pairs        = {}
    for i, ca in enumerate(conditions):
        for j, cb in enumerate(conditions):
            if i == j:
                continue
            mb_union = mode_b(edge_sets[ca] | edge_sets[cb], gt)
            naive_delta_recall[i, j] = round(mb_union["recall"] - single[ca]["recall"], 4)
            naive_delta_f1[i, j]     = round(mb_union["f1"]     - single[ca]["f1"],     4)
            naive_pairs[f"{ca}|{cb}"] = mb_union
    print_matrix("Naive union ΔRecall (row=H_A, col=H_B)", naive_delta_recall)

    save_matrix("two_pass_delta_recall", naive_delta_recall)
    save_matrix("two_pass_delta_f1",     naive_delta_f1)
    (out_dir / f"two_pass_pairs_{suite_stem}.json").write_text(
        json.dumps(naive_pairs, indent=2), encoding="utf-8"
    )
    print(f"Naive union pairs saved: {out_dir / f'two_pass_pairs_{suite_stem}.json'}")

    # --- Marginal-gain N×N loop ---
    pair_results = {}
    delta_f1      = np.full((n, n), np.nan)
    delta_recall  = np.full((n, n), np.nan)
    mode_a_improv = np.full((n, n), np.nan)

    total_pairs = n * (n - 1)
    pair_num    = 0

    for i, cond_a in enumerate(conditions):
        for j, cond_b in enumerate(conditions):
            if i == j:
                continue

            pair_num += 1
            pair_key = f"{cond_a}|{cond_b}"
            print(f"\n[{pair_num}/{total_pairs}] H_A={cond_a}  H_B={cond_b}")

            result = {
                "condition_a": cond_a,
                "condition_b": cond_b,
                "single_a":    single[cond_a],
                "single_b":    single[cond_b],
                "status":      "error",
            }

            try:
                h_a = edge_sets[cond_a]
                h_b = edge_sets[cond_b]

                final_circuit, mg_details = run_marginal_gain(
                    h_a=h_a,
                    h_b=h_b,
                    model=model,
                    clean_ds=clean_ds,
                    metric=metric,
                    edge_key_to_edge=edge_key_to_edge,
                    tau_mg=args.threshold,
                )

                mb   = mode_b(final_circuit, gt)
                kl   = evaluate_mode_a(final_circuit, model, clean_ds, metric, edge_key_to_edge)

                result["marginal_gain"] = {
                    **mb,
                    "mode_a_kl":             round(kl, 8),
                    "mode_a_kl_baseline_ha": round(kl_single[cond_a], 8),
                    "mode_a_kl_improvement": round(kl_single[cond_a] - kl, 8),
                }
                result["algorithm_details"] = mg_details
                result["status"] = "ok"

                delta_f1[i, j]      = mb["f1"]     - single[cond_a]["f1"]
                delta_recall[i, j]  = mb["recall"] - single[cond_a]["recall"]
                mode_a_improv[i, j] = kl_single[cond_a] - kl

                print(f"  -> Mode A KL: {kl:.6f}  (improvement={kl_single[cond_a]-kl:.6f})")
                print(f"  -> Mode B: edges={mb['edges']} P={mb['precision']:.4f} "
                      f"R={mb['recall']:.4f} F1={mb['f1']:.4f}  "
                      f"ΔF1={mb['f1']-single[cond_a]['f1']:+.4f}")

            except Exception as e:
                result["error"]     = repr(e)
                result["traceback"] = traceback.format_exc()
                print(f"  -> FAILED: {e}", flush=True)
                traceback.print_exc()

            pair_results[pair_key] = result

    # --- Print and save marginal-gain summary matrices ---
    print_matrix(f"ΔF1 matrix (marginal-gain vs H_A single-pass, threshold={args.threshold})", delta_f1)
    print_matrix("ΔRecall matrix (marginal-gain)", delta_recall)
    print_matrix("Mode A KL improvement matrix (row=H_A, col=H_B)", mode_a_improv)

    suite_out = {
        "suite":      str(suite_path),
        "threshold":  args.threshold,
        "device":     args.device,
        "seed":       args.seed,
        "conditions": conditions,
        "pairs":      pair_results,
    }
    suite_out_path = out_dir / f"marginal_gain_suite_{suite_stem}.json"
    suite_out_path.write_text(json.dumps(suite_out, indent=2), encoding="utf-8")
    print(f"\nFull results saved:    {suite_out_path}")

    save_matrix("marginal_gain_delta_f1",     delta_f1)
    save_matrix("marginal_gain_delta_recall", delta_recall)
    save_matrix("marginal_gain_mode_a",       mode_a_improv)

    # --- Heatmap ---
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        LABEL_SHORT = {
            "random_random":                   "rand_rand",
            "random_doc":                      "rand_doc",
            "random_def":                      "rand_def",
            "random_answer":                   "rand_ans",
            "random_def_doc":                  "rand_def_doc",
            "random_answer_doc":               "rand_ans_doc",
            "vary_length_doc_desc_random_doc": "vl_rand_doc",
            "zero_ablation":                   "zero_abl",
        }
        labels = [LABEL_SHORT.get(c, c) for c in conditions]

        fig, ax = plt.subplots(figsize=(max(8, n + 3), max(7, n + 1)))
        sns.heatmap(
            delta_f1,
            xticklabels=labels,
            yticklabels=labels,
            annot=True, fmt=".3f", annot_kws={"size": 10},
            cmap="RdYlGn", center=0.0,
            linewidths=0.5, ax=ax,
        )
        ax.set_title(
            f"ΔF1: marginal-gain vs H_A single-pass (τ={args.threshold}, row=H_A, col=H_B)",
            pad=14, fontsize=12,
        )
        ax.set_xlabel("H_B (candidate pool)", fontsize=11)
        ax.set_ylabel("H_A (base circuit)", fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        plt.tight_layout()

        heatmap_path = out_dir / f"marginal_gain_delta_f1_heatmap_{suite_stem}.png"
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        print(f"Heatmap saved:         {heatmap_path}")

    except ImportError as e:
        print(f"Heatmap skipped (missing dependency: {e}).")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
