"""
eval_circuit_efficiency.py — Mode A circuit efficiency comparison
-----------------------------------------------------------------
Usage:
  python eval_circuit_efficiency.py --task docstring --device cpu \\
      --corruption random_random \\
      --pkl rr runs/<run_dir>/another_final_edges.pkl \\
      --mbg mbg09 analysis/stage3b_marginal_gain/mbg_....json \\
            runs/<rr_run_dir>/another_final_edges.pkl \\
      --out analysis/stage3b_efficiency/resample_comparison.json
"""

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import torch

from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.TLACDCEdge import EdgeType
from acdc.global_cache import GlobalCache


# ---------------------------------------------------------------------------
# Mode A evaluation helpers
# ---------------------------------------------------------------------------

def _set_circuit(edge_key_to_edge: dict, circuit: set) -> None:
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
# Task data container
# ---------------------------------------------------------------------------

@dataclass
class TaskThings:
    model:        object
    clean_ds:     torch.Tensor
    metric:       object                   # callable: logits -> scalar KL
    gt_edges:     Optional[Set[str]]       # repr-normalised edge keys, or None
    corrupted_ds: Optional[torch.Tensor] = field(default=None)
    # corrupted_ds is None  → zero ablation
    # corrupted_ds is set   → resample ablation using those activations


# ---------------------------------------------------------------------------
# Task loaders
# ---------------------------------------------------------------------------

def _load_docstring(device: str, corruption: str = "zero") -> TaskThings:
    from acdc.docstring.utils import get_all_docstring_things, get_docstring_subgraph_true_edges

    # For resample ablation we load the corrupted dataset via dataset_version.
    # For zero ablation any version works (we only need model + clean tokens).
    dataset_version = "random_random" if corruption == "zero" else corruption

    things = get_all_docstring_things(
        num_examples=50,
        seq_len=41,
        device=device,
        metric_name="kl_div",
        dataset_version=dataset_version,
    )
    corrupted_ds = None if corruption == "zero" else things.validation_patch_data
    gt_edges = {repr(k) for k in get_docstring_subgraph_true_edges()}
    return TaskThings(
        model=things.tl_model,
        clean_ds=things.validation_data,
        corrupted_ds=corrupted_ds,
        metric=things.validation_metric,
        gt_edges=gt_edges,
    )


# Registry — add new tasks here.
# Each entry is a callable (device, corruption) -> TaskThings.
TASK_REGISTRY = {
    "docstring": _load_docstring,
}


# ---------------------------------------------------------------------------
# Evaluator setup  (zero or resample ablation)
# ---------------------------------------------------------------------------

def setup_evaluator(task: TaskThings) -> Tuple[Dict[str, object], str]:
    """Register ACDC hooks and fill the corrupted cache.

    Returns (edge_key_to_edge, ablation_label).
    ablation_label is a short string for display / JSON output.
    """
    model    = task.model
    clean_ds = task.clean_ds
    metric   = task.metric

    exp = TLACDCExperiment(
        model=model,
        ds=clean_ds,
        ref_ds=None,
        threshold=0.0,
        metric=metric,
        early_exit=True,
        zero_ablation=True,
        verbose=False,
        hook_verbose=False,
    )
    exp.ds                  = clean_ds
    exp.online_cache_cpu    = True
    exp.corrupted_cache_cpu = True
    exp.global_cache        = GlobalCache(device=("cpu", "cpu"))

    # Build the corrupted cache — either zeros or resampled activations.
    with torch.no_grad():
        _, clean_cache_obj = model.run_with_cache(clean_ds)

    if task.corrupted_ds is None:
        # --- Zero ablation ---
        for k, v in clean_cache_obj.cache_dict.items():
            if k in exp.corr.graph:
                exp.global_cache.corrupted_cache[k] = torch.zeros_like(v)
        ablation_label = "zero"
    else:
        # --- Resample ablation (Conmy-comparable) ---
        with torch.no_grad():
            _, corrupted_cache_obj = model.run_with_cache(task.corrupted_ds)
        for k, v in corrupted_cache_obj.cache_dict.items():
            if k in exp.corr.graph:
                exp.global_cache.corrupted_cache[k] = v
        ablation_label = "resample"

    exp.setup_model_hooks(
        add_sender_hooks=True,
        add_receiver_hooks=True,
        doing_acdc_runs=False,
    )

    edge_key_to_edge: Dict[str, object] = {}
    for (cn, ci, pn, pi), edge_obj in exp.corr.all_edges().items():
        if edge_obj.edge_type != EdgeType.PLACEHOLDER:
            key = repr((cn, ci.hashable_tuple, pn, pi.hashable_tuple))
            edge_key_to_edge[key] = edge_obj

    print(f"Evaluator ready ({len(edge_key_to_edge)} non-placeholder edges, "
          f"{ablation_label} ablation).")
    return edge_key_to_edge, ablation_label


# ---------------------------------------------------------------------------
# Circuit loaders
# ---------------------------------------------------------------------------

def load_from_pkl(path: Path) -> Set[str]:
    """Load edge set from another_final_edges.pkl."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    result = set()
    for k, _s in obj:
        cn, ci, pn, pi = k
        result.add(repr((cn, ci.hashable_tuple, pn, pi.hashable_tuple)))
    return result


def load_from_mbg_json(json_path: Path, ha_pkl_path: Path) -> Set[str]:
    """Load final marginal bundle gain circuit: H_A ∪ accepted_edges."""
    h_a = load_from_pkl(ha_pkl_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    accepted = set(data.get("algorithm_details", {}).get("accepted_edges", []))
    return h_a | accepted


# ---------------------------------------------------------------------------
# Mode B helper
# ---------------------------------------------------------------------------

def mode_b(circuit: Set[str], gt: Set[str]) -> dict:
    tp = len(circuit & gt)
    fp = len(circuit - gt)
    fn = len(gt - circuit)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"edges": len(circuit), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare circuit efficiency (Mode A KL) across multiple circuits."
    )
    parser.add_argument("--task", default="docstring",
        choices=list(TASK_REGISTRY.keys()),
        help="Task to evaluate (default: docstring)")
    parser.add_argument("--device", default="cpu",
        help="Device for model forward passes (default: cpu)")
    parser.add_argument("--corruption", default="zero",
        help="Ablation type for absent edges. 'zero' = zero ablation (default). "
             "Any dataset_version string (e.g. 'random_random') = resample ablation "
             "using that corruption's activations. Resample is directly comparable "
             "to Conmy (2023) Table 4.")
    parser.add_argument("--no-gt", action="store_true",
        help="Omit the task ground-truth circuit from evaluation")
    parser.add_argument("--pkl", nargs=2, metavar=("NAME", "PATH"), action="append",
        default=[],
        help="Add a circuit from another_final_edges.pkl. Repeatable.")
    parser.add_argument("--union", nargs="+", metavar="ARG",
        action="append", default=[],
        help="Add a named circuit from the set-union of one or more pkl files. "
             "Usage: NAME PKL_A [PKL_B ...]. Repeatable.")
    parser.add_argument("--mbg", nargs="+", metavar="ARG",
        action="append", default=[],
        help="Add a marginal-bundle-gain circuit. "
             "Usage: NAME MBG_JSON HA_PKL [HB_PKL]. Repeatable.")
    parser.add_argument("--out", default=None,
        help="Optional path to save results as JSON.")
    args = parser.parse_args()

    if args.task not in TASK_REGISTRY:
        print(f"Unknown task '{args.task}'. Available: {list(TASK_REGISTRY.keys())}")
        return

    # --- Load task ---
    print(f"\nLoading task '{args.task}' on '{args.device}' "
          f"(corruption='{args.corruption}')...")
    task = TASK_REGISTRY[args.task](args.device, args.corruption)

    # --- Build circuit list ---
    circuits: Dict[str, Set[str]] = {}

    if not args.no_gt and task.gt_edges is not None:
        circuits["gt"] = task.gt_edges
        print(f"  gt: {len(task.gt_edges)} edges (ground truth)")

    for name, path_str in args.pkl:
        path = Path(path_str)
        if not path.exists():
            print(f"  Warning: pkl not found: {path} — skipping '{name}'")
            continue
        circuits[name] = load_from_pkl(path)
        print(f"  {name}: {len(circuits[name])} edges (from pkl)")

    for union_args in args.union:
        if len(union_args) < 2:
            print(f"  Warning: --union needs NAME PKL_A [PKL_B ...] — got {union_args}, skipping")
            continue
        name = union_args[0]
        circuit: Set[str] = set()
        n_loaded = 0
        for p_str in union_args[1:]:
            p = Path(p_str)
            if not p.exists():
                print(f"  Warning: pkl not found: {p} — skipping in union '{name}'")
                continue
            circuit |= load_from_pkl(p)
            n_loaded += 1
        circuits[name] = circuit
        print(f"  {name}: {len(circuit)} edges (union of {n_loaded} pkl(s))")

    for mbg_args in args.mbg:
        if len(mbg_args) < 3:
            print(f"  Warning: --mbg needs NAME MBG_JSON HA_PKL — got {mbg_args}, skipping")
            continue
        name     = mbg_args[0]
        json_str = mbg_args[1]
        ha_str   = mbg_args[2]
        hb_str   = mbg_args[3] if len(mbg_args) >= 4 else None

        json_path = Path(json_str)
        ha_path   = Path(ha_str)
        if not json_path.exists():
            print(f"  Warning: MBG json not found: {json_path} — skipping '{name}'")
            continue
        if not ha_path.exists():
            print(f"  Warning: H_A pkl not found: {ha_path} — skipping '{name}'")
            continue

        h_a = load_from_pkl(ha_path)
        circuits[f"{name}_ha"] = h_a
        print(f"  {name}_ha: {len(h_a)} edges (H_A pkl)")

        if hb_str is not None:
            hb_path = Path(hb_str)
            if hb_path.exists():
                h_b = load_from_pkl(hb_path)
                circuits[f"{name}_hb"] = h_b
                print(f"  {name}_hb: {len(h_b)} edges (H_B pkl)")
            else:
                print(f"  Warning: H_B pkl not found: {hb_path} — skipping '{name}_hb'")

        circuits[name] = load_from_mbg_json(json_path, ha_path)
        print(f"  {name}: {len(circuits[name])} edges (H_A ∪ accepted bundles)")

    if not circuits:
        print("No circuits to evaluate. Use --pkl, --mbg, or leave --no-gt off.")
        return

    # --- Full model KL (before ACDC hooks are registered) ---
    with torch.no_grad():
        logits_full = task.model(task.clean_ds)
    kl_full = task.metric(logits_full).item()
    print(f"\n  Full model KL (no hooks): {kl_full:.6f}  [expect ≈ 0]")

    # --- Setup evaluator ---
    print()
    edge_key_to_edge, ablation_label = setup_evaluator(task)

    kl_col = f"KL ({ablation_label} abl)"

    # --- Empty circuit baseline ---
    # All edges absent: every receiver activation replaced by the ablated value.
    # Under zero ablation  → all zeros  → deterministic floor.
    # Under resample ablation → all corrupted activations → ACDC starting metric.
    _set_circuit(edge_key_to_edge, set())
    kl_empty = _eval_kl(task.model, task.clean_ds, task.metric)
    print(f"  Empty circuit KL (all edges absent, {ablation_label}): {kl_empty:.6f}")

    # --- Evaluate each circuit ---
    results: Dict[str, dict] = {}
    kl_field = f"kl_{ablation_label}_ablation"

    print(f"\n{'Circuit':30s} {'Edges':>6} {kl_col:>16} {'Recovery':>9} "
          f"{'TP':>4} {'FP':>5} {'FN':>4} {'F1':>7}")
    print("-" * 95)

    for name, circuit in circuits.items():
        _set_circuit(edge_key_to_edge, circuit)
        kl = _eval_kl(task.model, task.clean_ds, task.metric)

        # Recovery: fraction of the range [KL_empty → 0] recovered.
        # Higher = closer to full-model behaviour.
        recovery = (kl_empty - kl) / kl_empty if kl_empty > 0 else 0.0

        mb = mode_b(circuit, task.gt_edges) if task.gt_edges is not None else {}
        results[name] = {
            "edges":    len(circuit),
            kl_field:   round(kl, 6),
            "recovery": round(recovery, 4),
            "mode_b":   mb,
        }

        mb_str = (f"{mb['tp']:>4} {mb['fp']:>5} {mb['fn']:>4} {mb['f1']:>7.4f}"
                  if mb else "   —     —    —       —")
        print(f"  {name:28s} {len(circuit):>6} {kl:>16.6f} {recovery:>9.4f} {mb_str}")

    print("-" * 95)
    print(f"  {'full model':28s} {'all':>6} {kl_full:>16.6f} {'1.0000':>9}")

    # --- Marginal efficiency ---
    pareto = sorted(results.items(), key=lambda x: x[1]["edges"])
    print(f"\nMarginal efficiency (size-sorted, ΔKL-improvement / Δedges):")
    print(f"  {'Circuit':28s} {'Edges':>6} {'KL':>12} {'Recovery':>9} {'ΔKL/Δedge':>12}")
    prev_edges, prev_kl = 0, kl_empty
    for name, r in pareto:
        d_edges = r["edges"] - prev_edges
        d_kl    = prev_kl - r[kl_field]
        marg    = d_kl / d_edges if d_edges > 0 else float("nan")
        m_str   = f"{marg:>12.4f}" if marg == marg else f"{'—':>12}"
        print(f"  {name:28s} {r['edges']:>6} {r[kl_field]:>12.6f} "
              f"{r['recovery']:>9.4f} {m_str}")
        prev_edges, prev_kl = r["edges"], r[kl_field]
    d_e = len(edge_key_to_edge) - prev_edges
    d_k = prev_kl - kl_full
    m_f = d_k / d_e if d_e > 0 else float("nan")
    print(f"  {'full model':28s} {'all':>6} {kl_full:>12.6f} {'1.0000':>9} {m_f:>12.4f}")

    # --- Save ---
    out_data = {
        "task":           args.task,
        "device":         args.device,
        "corruption":     args.corruption,
        "ablation_type":  ablation_label,
        "kl_full_model":  round(kl_full, 6),
        "kl_empty_circuit": round(kl_empty, 6),
        "circuits":       results,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"\nResults saved: {out_path}")
    else:
        print(f"\n(Use --out PATH to save results as JSON)")

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
