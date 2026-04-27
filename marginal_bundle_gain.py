"""
marginal_bundle_gain.py — Stage 3b: Marginal Bundle Gain Circuit Construction
------------------------------------------------------------------------------
Given H_A (random_random) and H_B (zero_ablation), groups the candidate edges
from H_B \\ H_A into head-level bundles and tests each bundle as a unit.

This fixes the path-connectivity failure of the original marginal_gain.py:
head 1.2's Q/K sub-circuit is completely absent from random_random (zero edges
in any role), so individual-edge testing always shows 0.0 KL improvement.
Testing the full bundle adds all of head 1.2 at once, giving it the downstream
path it needs to affect the output.

Ablation method: random_random's ACTUAL corrupted activations (not zero
ablation). This asks "is this bundle task-specifically non-random in the
context of H_A?" rather than "does this bundle compute anything at all?".

Algorithm:
  1. Group H_B \\ H_A edges by head (L, N). Non-head edges form single-edge bundles.
  2. For each bundle B: evaluate circuit = H_A ∪ B. Accept if
     KL(H_A) - KL(H_A ∪ B) > tau_mg.
  3. Optional refinement: within each accepted bundle, test edge-by-edge removal.
     Prune edge e if removing it from (H_A ∪ accepted_bundle) does not worsen
     KL by more than tau_mg.
  4. Final circuit = H_A ∪ {all accepted and refined bundle edges}.

Usage:
  python marginal_bundle_gain.py runs/SUITE_<timestamp>_docstring.json \\
      --condition-a random_random --condition-b zero_ablation

  python marginal_bundle_gain.py runs/SUITE_<timestamp>_docstring.json \\
      --condition-a random_random --condition-b zero_ablation \\
      --threshold 0.10 --device cpu --out-dir runs/ --no-refinement

Outputs:
  - Console: bundle-by-bundle results, final circuit stats
  - marginal_bundle_gain_<cond_a>_<cond_b>_<suite_stem>.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union

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
        import pickle as _pickle
        with open(p, "rb") as f:
            obj = _pickle.load(f)
        result = set()
        for k, _s in obj:
            cn, ci, pn, pi = k
            result.add(repr((cn, ci.hashable_tuple, pn, pi.hashable_tuple)))
        return result
    except Exception as e:
        print(f"  Warning: could not load {p}: {e}")
        return set()


# ---------------------------------------------------------------------------
# Mode B evaluation  (no model needed)
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
# Mode A evaluation helpers
# ---------------------------------------------------------------------------

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
# Bundle assignment
# ---------------------------------------------------------------------------

BundleKey = Union[Tuple[int, int], str]  # (layer, head) or "misc:<edge_repr>"


def _head_bundle_keys(cn: str, ci_tuple: tuple, pn: str, pi_tuple: tuple) -> list:
    """Return all head bundles (layer, head) this edge belongs to.

    An edge belongs to head (L, N)'s bundle if:
      - The receiver (cn, ci_tuple) is one of:
          blocks.L.hook_{q,k,v}_input  with head index N
          blocks.L.attn.hook_{q,k,v}   with head index N
      - OR the sender (pn, pi_tuple) is:
          blocks.L.attn.hook_result     with head index N

    An edge can belong to TWO bundles simultaneously: e.g.
      (blocks.2.hook_k_input[2,X], ..., blocks.1.attn.hook_result[1,2], ...)
    belongs to both (2, X) as an incoming edge AND (1, 2) as an outgoing edge.
    Without dual membership, bundle (1,2) has no downstream path and the
    connectivity failure persists.
    """
    result = []

    # Receiver side — edge is incoming to head (layer, head)
    for pattern in (
        r'blocks\.(\d+)\.hook_([qkv])_input',
        r'blocks\.(\d+)\.attn\.hook_([qkv])$',
    ):
        m = re.match(pattern, cn)
        if m:
            layer = int(m.group(1))
            head = next((x for x in reversed(ci_tuple) if x is not None), None)
            if head is not None:
                result.append((layer, head))
            break  # cn can only match one receiver pattern

    # Sender side — edge is outgoing from head (layer, head)
    m = re.match(r'blocks\.(\d+)\.attn\.hook_result', pn)
    if m:
        layer = int(m.group(1))
        head = next((x for x in reversed(pi_tuple) if x is not None), None)
        if head is not None:
            sender_bundle = (layer, head)
            if sender_bundle not in result:
                result.append(sender_bundle)

    return result


def assign_bundles(candidates: Set[str]) -> Dict[BundleKey, Set[str]]:
    """Group candidate edge keys by head bundle.

    Head edges → one or more (layer, head) buckets (dual membership allowed).
    Non-head edges → individual "misc:<key>" buckets (tested one at a time).
    """
    bundles: Dict[BundleKey, Set[str]] = defaultdict(set)
    for key in candidates:
        cn, ci_tuple, pn, pi_tuple = eval(key)  # noqa: S307
        bkeys = _head_bundle_keys(cn, ci_tuple, pn, pi_tuple)
        if bkeys:
            for bk in bkeys:
                bundles[bk].add(key)
        else:
            bundles[f"misc:{key}"].add(key)
    return dict(bundles)


# ---------------------------------------------------------------------------
# Evaluator setup
# ---------------------------------------------------------------------------

def setup_bundle_evaluator(device: str, corruption_version: str = "random_random",
                           zero_ablation_cache: bool = False):
    """One-time setup for Mode A evaluation.

    By default (zero_ablation_cache=False), runs the model on
    corruption_version's corrupted inputs to build the cache. This asks
    "is this bundle discriminatively non-random in the context of H_A?"

    With zero_ablation_cache=True, the cache is zeroed instead. This asks
    "does this bundle contribute anything at all above zero, in the context
    of H_A?" — more permissive, but the marginal framing (H_A as base)
    may still filter FPs that H_A already accounts for.

    Returns:
        exp, model, clean_ds, metric, edge_key_to_edge
    """
    things = get_all_docstring_things(
        num_examples=50,
        seq_len=41,
        device=device,
        metric_name="kl_div",
        dataset_version=corruption_version,
    )
    model        = things.tl_model
    clean_ds     = things.validation_data
    corrupted_ds = things.validation_patch_data
    metric       = things.validation_metric

    # early_exit=True stops before setup_corrupted_cache and the ref_ds assertion
    exp = TLACDCExperiment(
        model=model,
        ds=clean_ds,
        ref_ds=None,
        threshold=0.0,
        metric=metric,
        early_exit=True,
        zero_ablation=False,
        verbose=False,
        hook_verbose=False,
    )

    exp.ds                  = clean_ds
    exp.online_cache_cpu    = True
    exp.corrupted_cache_cpu = True
    exp.global_cache        = GlobalCache(device=("cpu", "cpu"))

    # Build corrupted cache
    with torch.no_grad():
        _, cache_obj = model.run_with_cache(corrupted_ds)
    for k, v in cache_obj.cache_dict.items():
        if k in exp.corr.graph:
            if zero_ablation_cache:
                exp.global_cache.corrupted_cache[k] = torch.zeros_like(v)
            else:
                exp.global_cache.corrupted_cache[k] = v.clone()

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

    cache_label = "zero ablation" if zero_ablation_cache else f"'{corruption_version}' corrupted"
    print(f"Bundle evaluator ready: {len(edge_key_to_edge)} non-placeholder edges, "
          f"cache={cache_label}.")
    return exp, model, clean_ds, metric, edge_key_to_edge


# ---------------------------------------------------------------------------
# Marginal bundle gain algorithm
# ---------------------------------------------------------------------------

def run_marginal_bundle_gain(
    h_a: Set[str],
    h_b: Set[str],
    model,
    clean_ds: torch.Tensor,
    metric,
    edge_key_to_edge: dict,
    tau: float,
    do_refinement: bool = True,
) -> Tuple[Set[str], dict, float]:
    """
    Marginal bundle gain circuit construction.

    Bundles the candidate edges (H_B \\ H_A) by attention head and tests
    each bundle as a unit against the fixed H_A baseline.

    Returns:
        final_circuit  — H_A ∪ {accepted, refined bundle edges}
        details        — full per-bundle log for JSON output
        final_kl       — Mode A KL of the final circuit
    """
    candidates = h_b - h_a
    bundles = assign_bundles(candidates)

    # Baseline: H_A alone
    _set_circuit(edge_key_to_edge, h_a)
    baseline_kl = _eval_kl(model, clean_ds, metric)
    print(f"  Baseline KL (H_A alone, {len(h_a)} edges): {baseline_kl:.6f}")
    print(f"  Candidate edges: {len(candidates)} across {len(bundles)} bundles")

    accepted_edges: Set[str] = set()
    bundle_results = []

    for bundle_key in sorted(bundles, key=str):
        bundle_edges = bundles[bundle_key]
        is_head_bundle = isinstance(bundle_key, tuple)

        # Edges in bundle that are actually in the correspondence graph
        known = {e for e in bundle_edges if e in edge_key_to_edge}
        unknown = bundle_edges - known
        if unknown:
            print(f"  [{bundle_key}] {len(unknown)} edge(s) not in corr graph — skipped within bundle.")

        if not known:
            bundle_results.append({
                "bundle": str(bundle_key),
                "edges": sorted(bundle_edges),
                "known_edges": 0,
                "kl": None,
                "improvement": None,
                "accepted": False,
                "reason": "no_known_edges",
            })
            continue

        # Test H_A ∪ bundle (known edges only)
        test_circuit = h_a | known
        _set_circuit(edge_key_to_edge, test_circuit)
        kl_with = _eval_kl(model, clean_ds, metric)
        improvement = baseline_kl - kl_with
        accepted = improvement > tau

        label = "ACCEPT" if accepted else "reject"
        print(f"  [{label}] bundle={bundle_key}  edges={len(known)}  "
              f"improvement={improvement:.6f}  (kl={kl_with:.6f})")

        result: dict = {
            "bundle": str(bundle_key),
            "is_head_bundle": is_head_bundle,
            "edges": sorted(bundle_edges),
            "known_edges": len(known),
            "kl_with_bundle": round(kl_with, 8),
            "improvement": round(improvement, 8),
            "accepted": accepted,
        }

        if accepted:
            kept = set(known)

            # Optional refinement: prune edges whose removal costs < tau
            if do_refinement and len(known) > 1:
                pruned: Set[str] = set()
                for edge in sorted(known):
                    candidate_kept = kept - {edge}
                    _set_circuit(edge_key_to_edge, h_a | candidate_kept)
                    kl_without = _eval_kl(model, clean_ds, metric)
                    cost = kl_without - kl_with  # how much does removing this edge hurt?
                    if cost <= tau:
                        pruned.add(edge)
                        print(f"    prune edge (cost={cost:.6f} ≤ τ): {edge[:70]}")
                    else:
                        print(f"    keep  edge (cost={cost:.6f} > τ): {edge[:70]}")
                kept -= pruned
                result["refinement"] = {
                    "pruned_count": len(pruned),
                    "kept_count": len(kept),
                    "pruned": sorted(pruned),
                    "kept_after_refinement": sorted(kept),
                }

            accepted_edges |= kept

        bundle_results.append(result)

    # Restore final circuit and measure Mode A KL
    final_circuit = h_a | accepted_edges
    _set_circuit(edge_key_to_edge, final_circuit)
    final_kl = _eval_kl(model, clean_ds, metric)
    print(f"  Bundles accepted: {sum(1 for r in bundle_results if r['accepted'])}/{len(bundle_results)}")
    print(f"  Edges added to H_A: {len(accepted_edges)}")

    details = {
        "baseline_kl":       round(baseline_kl, 8),
        "tau":               tau,
        "do_refinement":     do_refinement,
        "bundles_total":     len(bundle_results),
        "bundles_accepted":  sum(1 for r in bundle_results if r["accepted"]),
        "edges_added":       len(accepted_edges),
        "accepted_edges":    sorted(accepted_edges),
        "per_bundle":        bundle_results,
    }
    return final_circuit, details, round(final_kl, 8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3b: marginal bundle gain — head-level bundle testing."
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
        help="Bundle acceptance threshold (default: 0.067). A bundle is accepted "
             "iff KL(H_A) - KL(H_A ∪ bundle) > threshold. Also used as the "
             "refinement threshold for pruning individual edges within bundles.")
    parser.add_argument("--no-refinement", action="store_true",
        help="Skip the within-bundle edge refinement step.")
    parser.add_argument("--zero-ablation-cache", action="store_true",
        help="Use zero ablation for absent edges instead of condition-a's corrupted "
             "activations. Asks 'does this bundle contribute anything at all in the "
             "context of H_A?' rather than 'is it discriminatively non-random?'. "
             "More permissive; may require a stricter --threshold.")
    parser.add_argument("--device", default="cpu",
        help="Device for model forward passes (default: cpu)")
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
    print(f"  H_B \\ H_A (candidates):  {len(h_b - h_a)} edges")
    print(f"  H_A ∩ H_B (shared):      {len(h_a & h_b)} edges")

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

    # --- Bundle summary ---
    bundles_preview = assign_bundles(h_b - h_a)
    head_bundles = {k: v for k, v in bundles_preview.items() if isinstance(k, tuple)}
    misc_bundles = {k: v for k, v in bundles_preview.items() if isinstance(k, str)}
    print(f"\n  Bundle breakdown: {len(head_bundles)} head bundle(s), "
          f"{len(misc_bundles)} misc edge(s)")
    for bk in sorted(head_bundles, key=str):
        edges_in_bundle = head_bundles[bk]
        gt_in_bundle = gt & edges_in_bundle
        in_ha = h_a & edges_in_bundle
        print(f"    head {bk}: {len(edges_in_bundle)} edges  "
              f"({len(gt_in_bundle)} GT, {len(in_ha)} already in H_A)")

    # Build output path early so finally block can always write something
    cond_a_short = args.condition_a.replace("_", "-")
    cond_b_short = args.condition_b.replace("_", "-")
    cache_suffix = "_zeroabl" if args.zero_ablation_cache else ""
    out_path = out_dir / f"marginal_bundle_gain_{cond_a_short}_{cond_b_short}{cache_suffix}_{suite_stem}.json"

    out: dict = {
        "suite":        str(suite_path),
        "condition_a":  args.condition_a,
        "condition_b":  args.condition_b,
        "seed":         args.seed,
        "threshold":    args.threshold,
        "do_refinement": not args.no_refinement,
        "zero_ablation_cache": args.zero_ablation_cache,
        "device":       args.device,
        "status":       "error",
        "single_a":     single_a,
        "single_b":     single_b,
        "naive_union":  naive_union,
    }

    try:
        # --- One-time Mode A setup with random_random's corrupted cache ---
        cache_desc = "zero ablation" if args.zero_ablation_cache else f"'{args.condition_a}' corrupted"
        print(f"\nSetting up bundle evaluator "
              f"(device={args.device}, cache={cache_desc})...")
        exp, model, clean_ds, metric, edge_key_to_edge = setup_bundle_evaluator(
            device=args.device,
            corruption_version=args.condition_a,
            zero_ablation_cache=args.zero_ablation_cache,
        )

        # Mode A baseline: KL of H_A alone
        _set_circuit(edge_key_to_edge, h_a)
        kl_h_a = _eval_kl(model, clean_ds, metric)
        print(f"  Mode A KL — H_A alone: {kl_h_a:.6f}")
        out["mode_a_kl_baseline_ha"] = round(kl_h_a, 8)

        # --- Run marginal bundle gain ---
        print(f"\n--- Marginal bundle gain (threshold={args.threshold}, "
              f"refinement={not args.no_refinement}) ---")
        final_circuit, details, kl_final = run_marginal_bundle_gain(
            h_a=h_a,
            h_b=h_b,
            model=model,
            clean_ds=clean_ds,
            metric=metric,
            edge_key_to_edge=edge_key_to_edge,
            tau=args.threshold,
            do_refinement=not args.no_refinement,
        )
        out["algorithm_details"] = details

        # --- Evaluate final circuit ---
        mb_final = mode_b(final_circuit, gt)

        print(f"\n--- Final marginal-bundle-gain circuit ---")
        print(f"  Mode A KL: {kl_final:.6f}  (H_A baseline: {kl_h_a:.6f}, "
              f"improvement={kl_h_a - kl_final:.6f})")
        print(f"  Mode B: edges={mb_final['edges']} "
              f"P={mb_final['precision']:.4f} R={mb_final['recall']:.4f} "
              f"F1={mb_final['f1']:.4f} "
              f"(TP={mb_final['tp']} FP={mb_final['fp']} FN={mb_final['fn']})")
        print(f"  ΔF1 vs H_A:         {mb_final['f1'] - single_a['f1']:+.4f}")
        print(f"  ΔF1 vs naive union: {mb_final['f1'] - naive_union['f1']:+.4f}")
        print(f"  ΔRecall vs H_A:     {mb_final['recall'] - single_a['recall']:+.4f}")

        out["marginal_bundle_gain"] = {
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
        print(f"\n[marginal_bundle_gain] FAILED: {e}", flush=True)
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
