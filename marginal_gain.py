"""
marginal_gain.py — Stage 3b: Marginal-Gain Circuit Construction
----------------------------------------------------------------
Given two ACDC circuits H_A and H_B (from a completed run_suite.py suite),
constructs a refined two-pass circuit by starting from H_A and adding edges
from H_B \ H_A only if they causally improve KL divergence on clean data
above a threshold tau_mg.

Each candidate edge is tested independently: the circuit H_A ∪ {e} is
evaluated against H_A alone. Edges are not accumulated between tests, so
results do not depend on evaluation order.

Ablation method for absent edges is controlled by --corruption:
  --corruption zero          (default) zero ablation — all-zeros corrupted cache.
  --corruption random_random (or any dataset_version) — resample ablation, absent
                             edges replaced with activations from that corruption.

See CLAUDE.md and marginal_gain_explainer.md for rationale.

Usage:
  python marginal_gain.py runs/SUITE_<timestamp>_docstring.json \\
      --condition-a random_random --condition-b zero_ablation

  python marginal_gain.py runs/SUITE_<timestamp>_docstring.json \\
      --condition-a random_random --condition-b zero_ablation \\
      --corruption zero --threshold 0.1 --device cpu --out-dir results/

Outputs (per pair):
  - Console: baseline stats, per-candidate results, final circuit stats
  - marginal_gain_<cond_a>_<cond_b>_<suite_stem>.json — full results
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Set, Dict, Tuple, Optional
import torch

from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.TLACDCEdge import EdgeType
from acdc.global_cache import GlobalCache
from acdc.docstring.utils import get_all_docstring_things, get_docstring_subgraph_true_edges


# ---------------------------------------------------------------------------
# Edge loading  (same normalisation as two_pass.py)
# ---------------------------------------------------------------------------

def load_edge_set(run_dir: Path) -> Set[str]:
    """Return surviving edge keys as repr strings from another_final_edges.pkl.

    Keys are normalised to repr((cn, ci.hashable_tuple, pn, pi.hashable_tuple))
    so they match the format used by get_docstring_subgraph_true_edges().
    """
    p = run_dir / "another_final_edges.pkl"
    if not p.exists():
        print(f"  Warning: pkl not found: {p}")
        return set()
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        result = set()
        for k, _s in obj:
            cn, ci, pn, pi = k
            result.add(repr((cn, ci.hashable_tuple, pn, pi.hashable_tuple)))
        return result
    except Exception as e:
        print(f"  Warning: could not load {p}: {e}")
        return set()


# ---------------------------------------------------------------------------
# Mode B evaluation  (same as two_pass.py — no model needed)
# ---------------------------------------------------------------------------

def mode_b(circuit: Set[str], gt: Set[str]) -> dict:
    """Compute precision/recall/F1 against ground truth."""
    tp = len(circuit & gt)
    fp = len(circuit - gt)
    fn = len(gt - circuit)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "edges":     len(circuit),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Mode A evaluation infrastructure  (model + cache + hooks, once only)
# ---------------------------------------------------------------------------

def setup_evaluator(device: str = "cpu", corruption: str = "zero"):
    """One-time setup for Mode A evaluation.

    Loads the model, builds the corrupted cache, and registers sender +
    receiver hooks on the model. Hooks are registered ONCE; between
    evaluations only edge.present booleans are flipped.

    Args:
        device     — torch device string
        corruption — 'zero' for zero ablation (all-zeros cache), or any
                     dataset_version string (e.g. 'random_random') for
                     resample ablation (absent edges ← activations from
                     that corruption's inputs).

    Returns:
        exp              — TLACDCExperiment shell (for corr + hooks)
        model            — HookedTransformer with hooks registered
        clean_ds         — clean input tokens [50, 41]
        metric           — KL divergence metric (callable, logits → scalar)
        edge_key_to_edge — dict: normalised repr key → Edge object
    """
    # For resample ablation load the named corruption to get patch_data.
    # For zero ablation any dataset_version works (only clean tokens used).
    dataset_version = "random_random" if corruption == "zero" else corruption
    things = get_all_docstring_things(
        num_examples=50,
        seq_len=41,
        device=device,
        metric_name="kl_div",
        dataset_version=dataset_version,
    )
    model     = things.tl_model
    clean_ds  = things.validation_data
    metric    = things.validation_metric

    # --- Init experiment shell (early_exit=True stops before heavy init) ---
    # After early_exit, exp has: model, corr, zero_ablation, verbose, hook_verbose.
    # It does NOT have: ds, global_cache, online_cache_cpu, corrupted_cache_cpu.
    exp = TLACDCExperiment(
        model=model,
        ds=clean_ds,
        ref_ds=None,
        threshold=0.0,          # unused — we never call step()
        metric=metric,
        early_exit=True,
        zero_ablation=True,
        verbose=False,
        hook_verbose=False,
    )

    # --- Gap 2: set attributes left unset by early_exit ---
    exp.ds                = clean_ds
    exp.online_cache_cpu  = True
    exp.corrupted_cache_cpu = True
    exp.global_cache      = GlobalCache(device=("cpu", "cpu"))

    # --- Gap 3: build corrupted cache ---
    with torch.no_grad():
        _, clean_cache_obj = model.run_with_cache(clean_ds)

    if corruption == "zero":
        # Zero ablation: absent edges replaced with zeros.
        for k, v in clean_cache_obj.cache_dict.items():
            if k in exp.corr.graph:
                exp.global_cache.corrupted_cache[k] = torch.zeros_like(v)
        ablation_label = "zero"
    else:
        # Resample ablation: absent edges replaced with activations from
        # the named corruption's inputs (Conmy-comparable).
        with torch.no_grad():
            _, corrupt_cache_obj = model.run_with_cache(things.validation_patch_data)
        for k, v in corrupt_cache_obj.cache_dict.items():
            if k in exp.corr.graph:
                exp.global_cache.corrupted_cache[k] = v
        ablation_label = f"resample({corruption})"

    print(f"  Ablation: {ablation_label}")

    # --- Gap 4: register both sender and receiver hooks upfront ---
    exp.setup_model_hooks(
        add_sender_hooks=True,
        add_receiver_hooks=True,
        doing_acdc_runs=False,   # suppresses deprecation warning
    )

    # --- Build lookup: normalised repr key → Edge object ---
    edge_key_to_edge: Dict[str, object] = {}
    for (cn, ci, pn, pi), edge_obj in exp.corr.all_edges().items():
        if edge_obj.edge_type != EdgeType.PLACEHOLDER:
            key = repr((cn, ci.hashable_tuple, pn, pi.hashable_tuple))
            edge_key_to_edge[key] = edge_obj

    print(f"Evaluator ready: {len(edge_key_to_edge)} non-placeholder edges in corr.")
    return exp, model, clean_ds, metric, edge_key_to_edge


def _set_circuit(edge_key_to_edge: dict, circuit: Set[str]) -> None:
    """Mark exactly the edges in circuit as present; all others absent."""
    for edge_obj in edge_key_to_edge.values():
        edge_obj.present = False
    for key in circuit:
        if key in edge_key_to_edge:
            edge_key_to_edge[key].present = True


def _eval_kl(model, clean_ds: torch.Tensor, metric) -> float:
    """One forward pass through the hooked model; return scalar KL."""
    with torch.no_grad():
        logits = model(clean_ds)
    return metric(logits).item()


# ---------------------------------------------------------------------------
# Marginal-gain algorithm
# ---------------------------------------------------------------------------

def evaluate_mode_a(circuit: Set[str], model, clean_ds: torch.Tensor, metric, edge_key_to_edge: dict) -> float:
    """Evaluate a static circuit under Mode A (zero ablation). Returns scalar KL."""
    _set_circuit(edge_key_to_edge, circuit)
    return _eval_kl(model, clean_ds, metric)


def run_marginal_gain(
    h_a: Set[str],
    h_b: Set[str],
    model,
    clean_ds: torch.Tensor,
    metric,
    edge_key_to_edge: dict,
    tau_mg: float,
) -> Tuple[Set[str], dict]:
    """
    Marginal-gain circuit construction.

    Each candidate edge e in H_B \\ H_A is tested independently:
    evaluates H_A ∪ {e} vs H_A alone. Edge is kept iff
    KL(H_A) - KL(H_A ∪ {e}) > tau_mg.

    Returns:
        final_circuit — H_A ∪ {kept edges}
        details       — per-candidate log dict for JSON output
    """
    candidates = sorted(h_b - h_a)  # sorted for determinism

    # --- Baseline: H_A alone ---
    _set_circuit(edge_key_to_edge, h_a)
    baseline_kl = _eval_kl(model, clean_ds, metric)
    print(f"  Baseline KL (H_A alone, {len(h_a)} edges): {baseline_kl:.6f}")
    print(f"  Candidates from H_B \\ H_A: {len(candidates)}")

    kept   = []
    tested = []

    for i, edge_key in enumerate(candidates):
        if edge_key not in edge_key_to_edge:
            # Edge from pkl is not in the current corr graph — skip
            tested.append({"edge": edge_key, "kl": None, "improvement": None, "kept": False, "reason": "not_in_corr"})
            continue

        # Test H_A ∪ {e} independently against the fixed H_A baseline.
        # Always reset after the test so kept edges do not accumulate and
        # give subsequent candidates a free pass on the improvement check.
        edge_key_to_edge[edge_key].present = True
        kl = _eval_kl(model, clean_ds, metric)
        improvement = baseline_kl - kl

        # Always reset — independent evaluation
        edge_key_to_edge[edge_key].present = False

        if improvement > tau_mg:
            kept.append(edge_key)
            print(f"  [{i+1:3d}/{len(candidates)}] KEEP  improvement={improvement:.6f}  {edge_key[:70]}")

        tested.append({
            "edge":        edge_key,
            "kl":          round(kl, 8),
            "improvement": round(improvement, 8),
            "kept":        improvement > tau_mg,
        })

    print(f"  Kept {len(kept)}/{len(candidates)} candidate edges.")

    # Always restore the circuit to exactly H_A ∪ kept (in case of floating state)
    final_circuit = h_a | set(kept)
    _set_circuit(edge_key_to_edge, final_circuit)

    details = {
        "baseline_kl":        round(baseline_kl, 8),
        "tau_mg":             tau_mg,
        "candidates_total":   len(candidates),
        "candidates_kept":    len(kept),
        "per_candidate":      tested,
    }
    return final_circuit, details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3b: marginal-gain two-pass circuit construction."
    )
    parser.add_argument("suite_json",
        help="Path to SUITE_*.json produced by run_suite.py")
    parser.add_argument("--condition-a", required=True,
        help="Base circuit condition (H_A), e.g. random_random")
    parser.add_argument("--condition-b", required=True,
        help="Candidate pool condition (H_B), e.g. zero_ablation")
    parser.add_argument("--seed", type=int, default=0,
        help="Which seed to use (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.067,
        help="Marginal-gain threshold (default: 0.067). An edge from H_B is "
             "kept iff KL(H_A) - KL(H_A ∪ {e}) > threshold.")
    parser.add_argument("--device", default="cpu",
        help="Device for model forward passes (default: cpu)")
    parser.add_argument("--corruption", default="zero",
        help="Ablation for absent edges. 'zero' = zero ablation (default). "
             "Any dataset_version string (e.g. 'random_random') = resample "
             "ablation using that corruption's activations.")
    parser.add_argument("--out-dir", default="analysis/stage3b_marginal_gain",
        help="Directory for output files (default: analysis/stage3b_marginal_gain/)")
    parser.add_argument("--runs-root", default=None,
        help="Override run_dir base path (for suites generated on a different machine)")
    args = parser.parse_args()

    suite_path = Path(args.suite_json)
    suite_stem = suite_path.stem
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load suite and resolve run dirs ---
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    rows = data["runs"] if isinstance(data, dict) else data
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
            print(f"Error: condition '{cond}' not found in suite.")
            return None
        by_seed = {r["seed"]: r for r in cond_rows}
        row = by_seed.get(args.seed) or sorted(cond_rows, key=lambda r: r["seed"])[0]
        if row["seed"] != args.seed:
            print(f"Warning: seed {args.seed} not found for '{cond}', using seed {row['seed']}.")
        return row

    row_a = pick_row(args.condition_a)
    row_b = pick_row(args.condition_b)
    if row_a is None or row_b is None:
        return

    # --- Load edge sets ---
    print(f"\nLoading edge sets...")
    h_a = load_edge_set(resolve_run_dir(row_a["run_dir"]))
    h_b = load_edge_set(resolve_run_dir(row_b["run_dir"]))
    print(f"  H_A ({args.condition_a}): {len(h_a)} edges")
    print(f"  H_B ({args.condition_b}): {len(h_b)} edges")
    print(f"  H_B \\ H_A (candidates): {len(h_b - h_a)} edges")
    print(f"  H_A ∩ H_B (shared):     {len(h_a & h_b)} edges")

    # --- Ground truth ---
    gt: Set[str] = {repr(k) for k in get_docstring_subgraph_true_edges()}
    print(f"  Ground truth: {len(gt)} edges")

    # --- Baselines (no model needed) ---
    single_a    = mode_b(h_a, gt)
    single_b    = mode_b(h_b, gt)
    naive_union = mode_b(h_a | h_b, gt)

    print(f"\n--- Baselines ---")
    print(f"  H_A ({args.condition_a}):  "
          f"edges={single_a['edges']} P={single_a['precision']:.4f} "
          f"R={single_a['recall']:.4f} F1={single_a['f1']:.4f}")
    print(f"  H_B ({args.condition_b}):  "
          f"edges={single_b['edges']} P={single_b['precision']:.4f} "
          f"R={single_b['recall']:.4f} F1={single_b['f1']:.4f}")
    print(f"  Naive union:    "
          f"edges={naive_union['edges']} P={naive_union['precision']:.4f} "
          f"R={naive_union['recall']:.4f} F1={naive_union['f1']:.4f}")

    # Build output path and skeleton early so the finally block can always
    # write whatever we managed to produce, even on a crash.
    cond_a_short = args.condition_a.replace("_", "-")
    cond_b_short = args.condition_b.replace("_", "-")
    corruption_tag = "" if args.corruption == "zero" else f"_resample-{args.corruption.replace('_', '-')}"
    out_path = out_dir / f"marginal_gain_{cond_a_short}_{cond_b_short}{corruption_tag}_{suite_stem}.json"

    out: dict = {
        "suite":       str(suite_path),
        "condition_a": args.condition_a,
        "condition_b": args.condition_b,
        "seed":        args.seed,
        "threshold":   args.threshold,
        "corruption":  args.corruption,
        "device":      args.device,
        "status":      "error",   # overwritten to "ok" on success
        "single_a":    single_a,
        "single_b":    single_b,
        "naive_union": naive_union,
    }

    try:
        # --- One-time Mode A setup ---
        print(f"\nSetting up Mode A evaluator (device={args.device}, corruption={args.corruption})...")
        exp, model, clean_ds, metric, edge_key_to_edge = setup_evaluator(args.device, args.corruption)

        # Mode A baseline: KL of H_A alone
        _set_circuit(edge_key_to_edge, h_a)
        kl_h_a = _eval_kl(model, clean_ds, metric)
        print(f"  Mode A KL — H_A alone: {kl_h_a:.6f}")
        out["mode_a_kl_baseline_ha"] = round(kl_h_a, 8)

        # --- Run marginal gain ---
        print(f"\n--- Marginal-gain (threshold={args.threshold}) ---")
        final_circuit, mg_details = run_marginal_gain(
            h_a=h_a,
            h_b=h_b,
            model=model,
            clean_ds=clean_ds,
            metric=metric,
            edge_key_to_edge=edge_key_to_edge,
            tau_mg=args.threshold,
        )
        out["algorithm_details"] = mg_details

        # --- Evaluate final circuit ---
        mb_final = mode_b(final_circuit, gt)

        _set_circuit(edge_key_to_edge, final_circuit)
        kl_final = _eval_kl(model, clean_ds, metric)

        print(f"\n--- Final marginal-gain circuit ---")
        print(f"  Mode A KL: {kl_final:.6f}  (H_A baseline: {kl_h_a:.6f}, "
              f"improvement={kl_h_a - kl_final:.6f})")
        print(f"  Mode B: edges={mb_final['edges']} "
              f"P={mb_final['precision']:.4f} R={mb_final['recall']:.4f} F1={mb_final['f1']:.4f} "
              f"(TP={mb_final['tp']} FP={mb_final['fp']} FN={mb_final['fn']})")
        print(f"  ΔF1 vs H_A:          {mb_final['f1'] - single_a['f1']:+.4f}")
        print(f"  ΔF1 vs naive union:  {mb_final['f1'] - naive_union['f1']:+.4f}")

        out["marginal_gain"] = {
            **mb_final,
            "mode_a_kl":             round(kl_final, 8),
            "mode_a_kl_baseline_ha": round(kl_h_a, 8),
            "mode_a_kl_improvement": round(kl_h_a - kl_final, 8),
        }
        out["status"] = "ok"

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        out["error"]     = repr(e)
        out["traceback"] = tb
        print(f"\n[marginal_gain] FAILED: {e}", flush=True)
        print(tb, flush=True)

    finally:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nResults saved: {out_path}  (status: {out['status']})")


if __name__ == "__main__":
    main()
