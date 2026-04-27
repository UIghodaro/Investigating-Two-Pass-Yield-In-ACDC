"""
eval_union_suite.py — Mode A evaluation for all pairwise naive unions
---------------------------------------------------------------------
Loads a SUITE JSON, evaluates every N×N union circuit (H_A ∪ H_B) under
one or more ablation modes, and writes KL + ΔKL matrices alongside full
per-pair Mode B stats.

The model is loaded ONCE and reused across all pairs and ablation modes.
Conditions with zero edges or non-ok status are automatically skipped.

Outputs (per corruption mode, in --out-dir):
  union_mode_a_{ablation}_{suite_stem}.json
    - kl_matrix:       N×N raw KL values
    - delta_kl_matrix: N×N ΔKL vs single-pass H_A (positive = improvement)
    - single_pass_kls: per-condition single-pass KL (diagonal reference)
    - pairs:           full per-pair Mode A + Mode B stats

Usage:
  python eval_union_suite.py runs/SUITE_*.json \\
      --corruption random_random \\
      --corruption zero \\
      --device cpu \\
      --out-dir analysis/stage3a_union_mode_a

  Defaults to both random_random and zero ablation if --corruption is omitted.
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Optional, Set

import torch

from eval_circuit_efficiency import TASK_REGISTRY, setup_evaluator, load_from_pkl, mode_b
from marginal_gain import _set_circuit, _eval_kl


# ---------------------------------------------------------------------------
# Suite loading
# ---------------------------------------------------------------------------

def load_conditions(suite_path: Path, runs_root: Optional[Path]) -> Dict[str, Path]:
    """Return {condition_name: pkl_path} for all valid runs in the suite."""
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    conditions: Dict[str, Path] = {}
    for run in suite["runs"]:
        if run.get("status") != "ok":
            continue
        if run.get("edges_count", 0) == 0:
            print(f"  Skipping '{run['condition']}' — zero edges.")
            continue
        cond = run["condition"]
        run_dir = Path(run["run_dir"])
        if runs_root is not None:
            run_dir = runs_root / run_dir.name
        pkl = run_dir / "another_final_edges.pkl"
        if not pkl.exists():
            print(f"  Warning: pkl not found for '{cond}': {pkl} — skipping.")
            continue
        conditions[cond] = pkl
    return conditions


# ---------------------------------------------------------------------------
# Per-corruption evaluation
# ---------------------------------------------------------------------------

def evaluate_corruption(
    corruption: str,
    task,
    edge_sets: Dict[str, Set[str]],
    cond_names: list,
    suite_stem: str,
    out_dir: Path,
) -> dict:
    """Run N×N union eval for one ablation mode. Returns the result dict."""

    # Update corrupted_ds on the shared task object.
    if corruption == "zero":
        task.corrupted_ds = None
    else:
        from acdc.docstring.utils import get_all_docstring_things
        things = get_all_docstring_things(
            num_examples=50,
            seq_len=41,
            device=task.model.cfg.device,
            metric_name="kl_div",
            dataset_version=corruption,
        )
        task.corrupted_ds = things.validation_patch_data

    # Reset any hooks from a previous corruption pass.
    task.model.reset_hooks()
    edge_key_to_edge, ablation_label = setup_evaluator(task)

    # Full-model and empty-circuit reference KLs.
    with torch.no_grad():
        kl_full = task.metric(task.model(task.clean_ds)).item()

    _set_circuit(edge_key_to_edge, set())
    kl_empty = _eval_kl(task.model, task.clean_ds, task.metric)

    print(f"  Full model KL : {kl_full:.6f}")
    print(f"  Empty circuit : {kl_empty:.6f}")

    # Single-pass KLs (diagonal reference).
    single_kls: Dict[str, float] = {}
    for name in cond_names:
        _set_circuit(edge_key_to_edge, edge_sets[name])
        single_kls[name] = _eval_kl(task.model, task.clean_ds, task.metric)

    # N×N union evaluation.
    col_w = 13
    kl_matrix:       Dict[str, Dict[str, float]] = {}
    delta_kl_matrix: Dict[str, Dict[str, float]] = {}
    pair_results:    Dict[str, dict] = {}

    for a_name in cond_names:
        kl_matrix[a_name]       = {}
        delta_kl_matrix[a_name] = {}
        for b_name in cond_names:
            union = edge_sets[a_name] | edge_sets[b_name]
            _set_circuit(edge_key_to_edge, union)
            kl = _eval_kl(task.model, task.clean_ds, task.metric)

            recovery  = (kl_empty - kl) / kl_empty if kl_empty > 0 else 0.0
            delta_kl  = single_kls[a_name] - kl   # + = improves over H_A alone

            mb = mode_b(union, task.gt_edges) if task.gt_edges is not None else {}

            kl_matrix[a_name][b_name]       = round(kl, 6)
            delta_kl_matrix[a_name][b_name] = round(delta_kl, 6)
            pair_results[f"{a_name}+{b_name}"] = {
                "edges":           len(union),
                "kl":              round(kl, 6),
                "recovery":        round(recovery, 4),
                "delta_kl_vs_ha":  round(delta_kl, 6),
                "mode_b":          mb,
            }

    # Print KL matrix.
    print(f"\n  KL matrix ({ablation_label} ablation):")
    header = f"  {'H_A / H_B':22s}" + "".join(f"{n[:col_w-1]:>{col_w}}" for n in cond_names)
    print(header)
    for a_name in cond_names:
        row = f"  {a_name:22s}" + "".join(
            f"{kl_matrix[a_name][b]:>{col_w}.4f}" for b in cond_names
        )
        print(row)

    # Print ΔKL matrix.
    print(f"\n  ΔKL vs single-pass H_A ({ablation_label}, + = improvement):")
    print(header)
    for a_name in cond_names:
        row = f"  {a_name:22s}" + "".join(
            f"{delta_kl_matrix[a_name][b]:>{col_w}.4f}" for b in cond_names
        )
        print(row)

    # Save.
    out = {
        "suite_stem":        suite_stem,
        "corruption":        corruption,
        "ablation_type":     ablation_label,
        "kl_full_model":     round(kl_full, 6),
        "kl_empty_circuit":  round(kl_empty, 6),
        "conditions":        cond_names,
        "single_pass_kls":   {k: round(v, 6) for k, v in single_kls.items()},
        "kl_matrix":         kl_matrix,
        "delta_kl_matrix":   delta_kl_matrix,
        "pairs":             pair_results,
    }
    out_path = out_dir / f"union_mode_a_{ablation_label}_{suite_stem}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mode A eval for all N×N pairwise naive unions from a suite."
    )
    parser.add_argument("suite_json",
        help="Path to SUITE_*.json produced by run_suite.py.")
    parser.add_argument("--task", default="docstring",
        choices=list(TASK_REGISTRY.keys()))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--corruption", action="append", default=[],
        metavar="DATASET_VERSION",
        help="Ablation mode: 'zero' or any dataset_version (e.g. 'random_random'). "
             "Repeatable. Defaults to both 'random_random' and 'zero' if omitted.")
    parser.add_argument("--runs-root", default=None,
        help="Override run directory root (useful if suite was generated elsewhere).")
    parser.add_argument("--out-dir", default="analysis/stage3a_union_mode_a",
        help="Directory for output JSON files.")
    args = parser.parse_args()

    if not args.corruption:
        args.corruption = ["random_random", "zero"]

    suite_path = Path(args.suite_json)
    suite_stem = suite_path.stem
    runs_root  = Path(args.runs_root) if args.runs_root else None
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load conditions.
    print(f"\nLoading suite: {suite_path.name}")
    conditions = load_conditions(suite_path, runs_root)
    if len(conditions) < 2:
        print(f"Need at least 2 valid conditions, found {len(conditions)}. Exiting.")
        return
    cond_names = list(conditions.keys())
    print(f"  {len(cond_names)} conditions: {cond_names}")

    # Load all edge sets once.
    print("\nLoading edge sets...")
    edge_sets: Dict[str, Set[str]] = {}
    for name, pkl_path in conditions.items():
        edge_sets[name] = load_from_pkl(pkl_path)
        print(f"  {name}: {len(edge_sets[name])} edges")

    # Load model once using the first corruption for the initial task setup.
    print(f"\nLoading task '{args.task}' on '{args.device}'...")
    task = TASK_REGISTRY[args.task](args.device, args.corruption[0])

    # Evaluate each ablation mode.
    for corruption in args.corruption:
        print(f"\n{'='*64}")
        print(f"Ablation: {corruption}")
        print(f"{'='*64}")
        try:
            evaluate_corruption(
                corruption, task, edge_sets, cond_names, suite_stem, out_dir
            )
        finally:
            task.model.reset_hooks()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
