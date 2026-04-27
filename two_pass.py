"""
two_pass.py — Stage 3 analysis
-------------------------------
Given a completed run_suite.py summary JSON, evaluates all pairwise two-pass
union circuits H_A ∪ H_B using Mode B (precision/recall/F1 against the 37-edge
docstring ground truth).

Outputs:
  - Console: single-pass table, union stats per unordered pair, ΔRecall matrix
  - two_pass_pairs_<suite_stem>.json   — Mode B stats for every unordered pair union
  - two_pass_delta_recall_<suite_stem>.json — N×N ΔRecall matrix (row=first pass, col=second pass)

Usage:
  python two_pass.py runs/SUITE_<timestamp>_docstring.json
  python two_pass.py runs/SUITE_<timestamp>_docstring.json --out-dir results/
  python two_pass.py runs/SUITE_<timestamp>_docstring.json --seed 1
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np

from acdc.docstring.utils import get_docstring_subgraph_true_edges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_edge_set(run_dir: Path) -> Set[str]:
    """Return surviving edge keys as repr strings from another_final_edges.pkl.

    save_edges() stores keys as (str, TorchIndex, str, TorchIndex).
    get_docstring_subgraph_true_edges() stores keys as (str, tuple, str, tuple)
    using .hashable_tuple.  We normalise here so both sides use the same format.
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


def mode_b(circuit: Set[str], gt: Set[str]) -> dict:
    """Compute Mode B stats for a circuit edge set against ground truth."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-pass union Mode B analysis from a run_suite.py JSON."
    )
    parser.add_argument("suite_json", help="Path to SUITE_*.json produced by run_suite.py")
    parser.add_argument(
        "--out-dir", default="analysis/stage3a_naive_union",
        help="Directory for output files (default: analysis/stage3a_naive_union/)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Which seed to use for each condition (default: 0; falls back to first available)",
    )
    parser.add_argument(
        "--runs-root", default=None,
        help="Override the root directory of run_dirs stored in the suite JSON. "
             "Useful when the suite was generated on a different machine or path. "
             "The run folder name is preserved; only the base path is replaced. "
             "Example: --runs-root /path/to/repo",
    )
    args = parser.parse_args()

    suite_path = Path(args.suite_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_stem = suite_path.stem  # e.g. SUITE_20260312_192142_docstring

    # Ground truth edge reprs
    gt: Set[str] = {repr(k) for k in get_docstring_subgraph_true_edges()}
    print(f"Ground truth: {len(gt)} edges")

    # Load suite
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    rows = data["runs"] if isinstance(data, dict) else data
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    skipped = len(rows) - len(ok_rows)
    if skipped:
        print(f"Note: {skipped} error row(s) excluded.")

    # Group by condition; pick requested seed (fall back to first available)
    by_condition: Dict[str, list] = defaultdict(list)
    for row in ok_rows:
        by_condition[row["condition"]].append(row)

    selected: Dict[str, dict] = {}
    for cond, cond_rows in by_condition.items():
        by_seed = {r["seed"]: r for r in cond_rows}
        row = by_seed.get(args.seed) or sorted(cond_rows, key=lambda r: r["seed"])[0]
        if row["seed"] != args.seed:
            print(f"  Warning: seed {args.seed} not found for '{cond}', using seed {row['seed']}")
        selected[cond] = row

    conditions = sorted(selected.keys())
    n = len(conditions)

    # Load edge sets — optionally remap base path if suite was generated elsewhere.
    # Preserves the path from "runs/" onwards so the subdirectory structure is kept.
    def resolve_run_dir(run_dir_str: str) -> Path:
        p = Path(run_dir_str)
        if args.runs_root:
            parts = p.parts
            try:
                runs_idx = list(parts).index("runs")
                relative = Path(*parts[runs_idx:])
            except ValueError:
                relative = Path(p.name)
            return Path(args.runs_root) / relative
        return p

    edge_sets: Dict[str, Set[str]] = {
        cond: load_edge_set(resolve_run_dir(selected[cond]["run_dir"]))
        for cond in conditions
    }

    # ------------------------------------------------------------------
    # Single-pass Mode B
    # ------------------------------------------------------------------
    single: Dict[str, dict] = {cond: mode_b(edge_sets[cond], gt) for cond in conditions}

    print("\n=== Single-pass Mode B (seed {}) ===".format(args.seed))
    print(f"{'condition':38s} {'edges':>6} {'P':>7} {'R':>7} {'F1':>7}")
    for cond in conditions:
        s = single[cond]
        print(f"{cond:38s} {s['edges']:>6} {s['precision']:>7.4f} {s['recall']:>7.4f} {s['f1']:>7.4f}")

    # ------------------------------------------------------------------
    # Pairwise union stats (unordered pairs)
    # ------------------------------------------------------------------
    pairs: Dict[Tuple[str, str], dict] = {}
    for i, a in enumerate(conditions):
        for j, b in enumerate(conditions):
            if j <= i:
                continue
            union = edge_sets[a] | edge_sets[b]
            pairs[(a, b)] = mode_b(union, gt)

    print("\n=== Two-pass union Mode B (unordered pairs) ===")
    print(f"{'A':28s} {'B':28s} {'edges':>6} {'P':>7} {'R':>7} {'F1':>7}")
    for (a, b), stats in sorted(pairs.items(), key=lambda x: -x[1]["f1"]):
        print(
            f"{a:28s} {b:28s} {stats['edges']:>6} "
            f"{stats['precision']:>7.4f} {stats['recall']:>7.4f} {stats['f1']:>7.4f}"
        )

    pairs_out = {
        f"{a}|{b}": stats for (a, b), stats in pairs.items()
    }
    pairs_path = out_dir / f"two_pass_pairs_{suite_stem}.json"
    pairs_path.write_text(json.dumps(pairs_out, indent=2), encoding="utf-8")
    print(f"\nPair stats saved: {pairs_path}")

    # ------------------------------------------------------------------
    # ΔRecall matrix (ordered: row = first pass A, col = second pass B)
    # cell[i,j] = Recall(H_A ∪ H_B) - Recall(H_A)
    # ------------------------------------------------------------------
    delta = np.zeros((n, n))
    for i, a in enumerate(conditions):
        for j, b in enumerate(conditions):
            if i == j:
                delta[i, j] = 0.0
                continue
            union = edge_sets[a] | edge_sets[b]
            union_recall = mode_b(union, gt)["recall"]
            delta[i, j] = round(union_recall - single[a]["recall"], 4)

    col_w = 9
    print("\n=== ΔRecall matrix (row = first pass, col = second pass) ===")
    print(f"{'':38s}" + "".join(f"{c[:col_w]:>{col_w}s}" for c in conditions))
    for i, a in enumerate(conditions):
        cells = "".join(f"{delta[i, j]:>{col_w}.4f}" for j in range(n))
        print(f"{a:38s}{cells}")

    delta_path = out_dir / f"two_pass_delta_recall_{suite_stem}.json"
    delta_path.write_text(
        json.dumps({"conditions": conditions, "matrix": delta.tolist()}, indent=2),
        encoding="utf-8",
    )
    print(f"\nΔRecall matrix saved: {delta_path}")


if __name__ == "__main__":
    main()
