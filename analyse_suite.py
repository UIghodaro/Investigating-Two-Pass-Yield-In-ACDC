
"""
Stage /Research Question 2 - analyse_suite.py
----------------
Read a completed run_suite.py summary JSON and produce:
  - Yield table  (edges_count, precision, recall, F1 averaged across seeds)
  - Pairwise Jaccard matrix (average across matching seeds)
  - jaccard_matrix.json and yield_table.json in the output directory
  - jaccard_heatmap.png (requires matplotlib + seaborn)

Usage:
  python analyse_suite.py runs/SUITE_<timestamp>_docstring.json
  python analyse_suite.py runs/SUITE_<timestamp>_docstring.json --out-dir results/
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import numpy as np


# ---------------------------------------------------------------------------
# Edge loading
# ---------------------------------------------------------------------------

def load_edge_set(run_dir: Path) -> Set[str]:
    """Return the set of edge keys from another_final_edges.pkl as repr strings."""
    p = run_dir / "another_final_edges.pkl"
    if not p.exists():
        return set()
    try:
        obj = pickle.load(open(p, "rb"))
        return {repr(k) for k, s in obj if s is not None}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Jaccard
# ---------------------------------------------------------------------------

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse a run_suite.py output: Jaccard matrix + yield table."
    )
    parser.add_argument("suite_json", help="Path to SUITE_*.json produced by run_suite.py")
    parser.add_argument(
        "--out-dir", default="analysis",
        help="Directory for output files (default: analysis/). Jaccard outputs go to "
             "<out-dir>/stage2_jaccard/, yield table to <out-dir>/stage1_yield/.",
    )
    args = parser.parse_args()

    suite_path = Path(args.suite_json)
    suite_stem = suite_path.stem  # e.g. "SUITE_20260324_041413_docstring"
    out_dir = Path(args.out_dir)
    jaccard_dir = out_dir / "stage2_jaccard"
    yield_dir   = out_dir / "stage1_yield"
    jaccard_dir.mkdir(parents=True, exist_ok=True)
    yield_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(suite_path.read_text(encoding="utf-8"))
    # Support both old format (bare list) and new format ({"env": ..., "runs": [...]})
    rows = data["runs"] if isinstance(data, dict) else data

    # Skip error rows
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if len(ok_rows) < len(rows):
        print(f"Note: {len(rows) - len(ok_rows)} error row(s) excluded from analysis.")

    # Group by condition
    by_condition: Dict[str, list] = defaultdict(list)
    for row in ok_rows:
        by_condition[row["condition"]].append(row)

    conditions = sorted(by_condition.keys())
    n = len(conditions)

    if n == 0:
        print("No valid conditions found in suite JSON.")
        return

    # Load edge sets: condition -> seed -> set[str]
    edge_sets: Dict[str, Dict[int, Set[str]]] = {}
    for cond in conditions:
        edge_sets[cond] = {}
        for row in by_condition[cond]:
            edge_sets[cond][row["seed"]] = load_edge_set(Path(row["run_dir"]))

    # ------------------------------------------------------------------
    # Jaccard matrix  (average over matching seeds)
    # ------------------------------------------------------------------
    matrix = np.full((n, n), np.nan)
    for i, cond_a in enumerate(conditions):
        for j, cond_b in enumerate(conditions):
            common = sorted(set(edge_sets[cond_a]) & set(edge_sets[cond_b]))
            if not common:
                continue
            vals = [jaccard(edge_sets[cond_a][s], edge_sets[cond_b][s]) for s in common]
            matrix[i, j] = sum(vals) / len(vals)

    # Print matrix
    col_w = 10
    print("\n=== Jaccard similarity matrix (avg across seeds) ===")
    print(f"{'':38s}" + "".join(f"{c[:col_w]:>{col_w}s}" for c in conditions))
    for i, cond_a in enumerate(conditions):
        cells = "".join(
            f"{matrix[i, j]:>{col_w}.3f}" if not np.isnan(matrix[i, j]) else f"{'N/A':>{col_w}s}"
            for j in range(n)
        )
        print(f"{cond_a:38s}{cells}")

    matrix_path = jaccard_dir / f"jaccard_matrix_{suite_stem}.json"
    matrix_path.write_text(
        json.dumps({"conditions": conditions, "matrix": matrix.tolist()}, indent=2),
        encoding="utf-8",
    )
    print(f"\nJaccard matrix saved: {matrix_path}")

    # ------------------------------------------------------------------
    # Yield table  (avg across seeds)
    # ------------------------------------------------------------------
    yield_fields = ["edges_count", "precision", "recall", "f1"]
    print("\n=== Yield table (avg across seeds) ===")
    print(f"{'condition':38s}" + "".join(f"{f:>12s}" for f in yield_fields))

    yield_data = {}
    for cond in conditions:
        cond_rows = by_condition[cond]
        avgs = {}
        for field in yield_fields:
            vals = [r[field] for r in cond_rows if r.get(field) is not None]
            avgs[field] = round(sum(vals) / len(vals), 4) if vals else None
        yield_data[cond] = avgs
        cells = "".join(
            f"{avgs[f]:>12.4f}" if avgs[f] is not None else f"{'N/A':>12s}"
            for f in yield_fields
        )
        print(f"{cond:38s}{cells}")

    yield_path = yield_dir / f"yield_table_{suite_stem}.json"
    yield_path.write_text(json.dumps(yield_data, indent=2), encoding="utf-8")
    print(f"\nYield table saved:   {yield_path}")

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Exclude conditions with no edges — their Jaccard(∅, X)=0 is not
        # meaningful complementarity and misleads the colour scale.
        plot_conditions = [
            c for c in conditions
            if any(len(s) > 0 for s in edge_sets[c].values())
        ]
        plot_idx = [conditions.index(c) for c in plot_conditions]
        plot_matrix = matrix[np.ix_(plot_idx, plot_idx)]

        LABEL_SHORT = {
            "random_random":                    "rand_rand",
            "random_doc":                       "rand_doc",
            "random_def":                       "rand_def",
            "random_answer":                    "rand_ans",
            "random_def_doc":                   "rand_def_doc",
            "random_answer_doc":                "rand_ans_doc",
            "vary_length_doc_desc":             "vl_doc_desc",
            "vary_length_doc_desc_random_doc":  "vl_rand_doc",
            "zero_ablation":                    "zero_abl",
        }
        plot_labels = [LABEL_SHORT.get(c, c) for c in plot_conditions]

        m = len(plot_conditions)
        fig, ax = plt.subplots(figsize=(max(8, m + 3), max(7, m + 1)))
        sns.heatmap(
            plot_matrix,
            xticklabels=plot_labels,
            yticklabels=plot_labels,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 11},
            cmap="Blues_r",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Pairwise Jaccard similarity (darker = more complementary)", pad=14, fontsize=13)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
        plt.tight_layout()

        heatmap_path = jaccard_dir / f"jaccard_heatmap_{suite_stem}.png"
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        print(f"Heatmap saved:       {heatmap_path}")

    except ImportError as e:
        print(f"Heatmap skipped (missing dependency: {e}). Install matplotlib and seaborn to enable.")


if __name__ == "__main__":
    main()
