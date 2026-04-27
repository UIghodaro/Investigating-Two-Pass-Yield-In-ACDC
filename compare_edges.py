"""
compare_edges.py — Pairwise edge comparison between two ACDC runs
-----------------------------------------------------------------
Loads two another_final_edges.pkl files and reports: edge counts,
overlap, Jaccard similarity, and the top-N edges unique to each.

Usage:
  python compare_edges.py runs/<run_a>/another_final_edges.pkl \\
                          runs/<run_b>/another_final_edges.pkl
  python compare_edges.py <base.pkl> <other.pkl> --top 20
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

EdgeKey = Tuple[Any, Any, Any, Any]
Edge = Tuple[EdgeKey, float]


def load_edges(path: Path) -> List[Edge]:
    """Load and validate an edge pkl file; skip None-scored placeholder edges."""
    skipped = 0
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise TypeError(f"{path} expected list, got {type(obj)}")
    out: List[Edge] = []
    for item in obj:
        if not (isinstance(item, tuple) and len(item) == 2):
            raise TypeError(f"{path} expected (key, score) tuple, got {item!r}")
        key, score = item
        if score is None:
            skipped += 1
            continue
        out.append((key, float(score)))
    if skipped:
        print(f"{path}: skipped {skipped} placeholder edge(s) with score=None")
    return out


def key_to_str(k: EdgeKey) -> str:
    return repr(k)


def as_dict(edges: Iterable[Edge]) -> Dict[str, float]:
    """Convert edge list to {str_key: score} dict for set-based comparison."""
    return {key_to_str(k): s for k, s in edges}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser(description="Compare two ACDC edge.pkl outputs.")
    parser.add_argument("base", type=str, help="Baseline edges.pkl")
    parser.add_argument("other", type=str, help="Other edges.pkl (pass1/pass2)")
    parser.add_argument("--top", type=int, default=10, help="Show top-N new edges by score.")
    args = parser.parse_args()

    base_edges = load_edges(Path(args.base))
    other_edges = load_edges(Path(args.other))

    base_d = as_dict(base_edges)
    other_d = as_dict(other_edges)

    base_set = set(base_d.keys())
    other_set = set(other_d.keys())

    inter = base_set & other_set
    only_other = other_set - base_set
    only_base = base_set - other_set

    print("Base edges:", len(base_set))
    print("Other edges:", len(other_set))
    print("Overlap:", len(inter))
    print("Only in other:", len(only_other))
    print("Only in base:", len(only_base))
    print("Jaccard:", round(jaccard(base_set, other_set), 4))

    if only_other:
        ranked = sorted(only_other, key=lambda k: other_d[k], reverse=True)
        print(f"\nTop {min(args.top, len(ranked))} new edges in OTHER:")
        for k in ranked[: args.top]:
            
            print(f"  score={other_d[k]:.6f} edge={k}")

    if only_base:
        ranked = sorted(only_base, key=lambda k: base_d[k], reverse=True)
        print(f"\nTop {min(args.top, len(ranked))} edges missing from OTHER (present in BASE):")
        for k in ranked[: args.top]:
            print(f"  score={base_d[k]:.6f} edge={k}")


if __name__ == "__main__":
    main()
