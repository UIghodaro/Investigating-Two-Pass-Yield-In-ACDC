"""
visualise_circuits.py  —  publication-quality circuit diagrams
---------------------------------------------------------------
Generates PNG + SVG circuit diagrams for any set of circuits.

GT is always loaded as the reference for vs-GT diagrams and its own
standalone diagram is always produced.  All other circuits are specified
on the command line:

  --conditions NAME [NAME …]
        Resolve circuits by condition name from a suite JSON.
        e.g. --conditions random_doc random_random zero_ablation

  --suite-json PATH
        Suite JSON to resolve --conditions from.
        Default: runs/SUITE_20260324_041413_docstring.json  (τ=0.10)

  --pkl NAME PATH
        Load an arbitrary pkl directly (repeatable).
        e.g. --pkl mbg_random analysis/stage3b.../marginal_bundle_gain.pkl

  --mbg NAME JSON HA_PKL
        Load an mbg circuit: H_A ∪ accepted bundle edges (repeatable).
        e.g. --mbg mbg_random analysis/.../mbg.json runs/.../another_final_edges.pkl

  --runs-root PATH
        Remap run_dir paths in the suite JSON (useful if suite was generated
        on a different machine or repo location).

  --out-dir PATH
        Output root.  Default: images/circuits/

Two detail levels per circuit:
  collapsed  — head-level only (a0.5, a1.2, …)
  full       — ACDC edge-level with separate q/k/v input nodes

Two diagram types per circuit:
  standalone — plain, single colour scheme
  vs_gt      — TP (green) / FP (red) / FN (grey dashed) against GT

Example:
  "C:/Users/osiam/miniconda3/envs/acdc_NEW/python.exe" visualise_circuits.py \\
      --conditions random_doc random_random
"""

import argparse
import ast
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Set

import pygraphviz as pgv

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO             = Path(__file__).parent
DEFAULT_OUT_DIR  = REPO / "images" / "circuits"
DEFAULT_SUITE    = REPO / "runs" / "SUITE_20260324_041413_docstring.json"

# ---------------------------------------------------------------------------
# Edge loading  (normalisation matches eval_circuit_efficiency.py exactly)
# ---------------------------------------------------------------------------

def load_from_pkl(path: Path) -> Set[str]:
    """Load present edges from another_final_edges.pkl as repr-normalised strings."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    result = set()
    for k, _s in obj:
        cn, ci, pn, pi = k
        result.add(repr((cn, ci.hashable_tuple, pn, pi.hashable_tuple)))
    return result


def load_from_mbg_json(json_path: Path, ha_pkl_path: Path) -> Set[str]:
    """Load mbg_Random final circuit: H_A ∪ accepted bundle edges."""
    h_a = load_from_pkl(ha_pkl_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    accepted = set(data.get("algorithm_details", {}).get("accepted_edges", []))
    return h_a | accepted


def load_gt() -> Set[str]:
    """Load H&J ground-truth edges as repr-normalised strings."""
    from acdc.docstring.utils import get_docstring_subgraph_true_edges
    return {repr(k) for k in get_docstring_subgraph_true_edges()}


# ---------------------------------------------------------------------------
# Node labelling
# ---------------------------------------------------------------------------

def node_label(hook_name: str, idx: tuple, collapsed: bool) -> str:
    """
    Convert (hook_name, index_tuple) → short display label.

    collapsed=False (full detail):
      blocks.0.hook_resid_pre  (None,)          → 'embed'
      blocks.1.hook_q_input    (None,None,2)    → 'a1.2_q'
      blocks.1.hook_k_input    (None,None,2)    → 'a1.2_k'
      blocks.1.hook_v_input    (None,None,2)    → 'a1.2_v'
      blocks.1.attn.hook_result (None,None,2)   → 'a1.2'
      blocks.3.hook_resid_post (None,)           → 'output'

    collapsed=True (head level):
      All head-related nodes collapse to 'aL.N'.
      Intra-head self-loops (e.g. a1.2_k → a1.2 → a1.2) are filtered
      upstream by the src != dst check in build_graph.
    """
    if "hook_resid_pre" in hook_name:
        return "embed"
    if "pos_embed" in hook_name:
        return "pos"
    if "hook_embed" in hook_name:
        return "embed"

    m = re.search(r"blocks\.(\d+)", hook_name)
    if not m:
        return hook_name   # fallback — should not occur in docstring task

    layer = int(m.group(1))

    if "hook_resid_post" in hook_name:
        return "output"

    # All remaining nodes are head-specific; extract head index from idx tuple.
    head = next((i for i in idx if i is not None), "?")

    if collapsed:
        return f"a{layer}.{head}"

    # Full QKV detail
    if "hook_q_input" in hook_name:
        return f"a{layer}.{head}_q"
    if "hook_k_input" in hook_name:
        return f"a{layer}.{head}_k"
    if "hook_v_input" in hook_name:
        return f"a{layer}.{head}_v"
    # hook_q/k/v (projections) and hook_result all map to the head result node.
    return f"a{layer}.{head}"


def node_rank(label: str, collapsed: bool) -> int:
    """
    Assign an integer rank so dot places nodes in the right horizontal row.

    collapsed:
      embed=0, a0.*=1, a1.*=2, a2.*=3, a3.*=4, output=100

    full:
      embed=0
      QKV inputs for layer L  → rank 2L+1  (1, 3, 5, 7)
      Head results  for layer L → rank 2L+2  (2, 4, 6, 8)
      output=100
    """
    if label in ("embed", "pos"):
        return 0
    if label == "output":
        return 100

    m = re.match(r"a(\d+)\.", label)
    if not m:
        return 50

    layer = int(m.group(1))

    if collapsed:
        return layer + 1

    is_qkv_input = label.endswith(("_q", "_k", "_v"))
    if is_qkv_input:
        return layer * 2 + 1
    return layer * 2 + 2


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

# Layer fill colours (L0–L3) — chosen to be distinct and print-safe
_LAYER_FILL = ["#AED6F1", "#A9DFBF", "#FAD7A0", "#D7BDE2"]
_EMBED_FILL  = "#F9E79F"   # yellow
_OUTPUT_FILL = "#F5CBA7"   # salmon
_QKV_FILL    = "#FAFAFA"   # near-white for QKV input nodes (subordinate role)

# Edge colours for TP/FP/FN comparison
TP_COLOR   = "#1A7A3C"   # dark green  — True  Positive
FP_COLOR   = "#C0392B"   # dark red    — False Positive
FN_COLOR   = "#95A5A6"   # grey        — False Negative (dashed)
PLAIN_COLOR = "#2C3E50"  # dark slate  — standalone diagrams


def fill_color(label: str) -> str:
    if label in ("embed", "pos"):
        return _EMBED_FILL
    if label == "output":
        return _OUTPUT_FILL
    m = re.match(r"a(\d+)\.", label)
    if not m:
        return "#FFFFFF"
    layer = int(m.group(1))
    if label.endswith(("_q", "_k", "_v")):
        return _QKV_FILL
    return _LAYER_FILL[layer % 4]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(
    circuit: Set[str],
    collapsed: bool,
    title: str = "",
    gt: Optional[Set[str]] = None,
) -> pgv.AGraph:
    """
    Build a pygraphviz AGraph from a set of edge repr strings.

    If gt is provided, edges are coloured as TP / FP / FN relative to GT.
    Otherwise all edges are drawn in PLAIN_COLOR.
    """
    g = pgv.AGraph(directed=True, strict=False)
    g.graph_attr.update(
        rankdir="TB",
        bgcolor="white",
        splines="true",
        overlap="false",
        fontname="Helvetica",
        fontsize="13",
        label=title,
        labelloc="t",
        pad="0.5",
        nodesep="0.45",
        ranksep="0.7",
    )
    g.node_attr.update(
        shape="box",
        style="filled,rounded",
        fontname="Helvetica",
        fontsize="10",
        margin="0.15,0.08",
        color="#555555",
    )
    g.edge_attr.update(
        fontname="Helvetica",
        fontsize="8",
        arrowsize="0.7",
    )

    # ------------------------------------------------------------------
    # Parse edge repr strings → (src_label, dst_label)
    # Edge format: repr((child_name, child_idx, parent_name, parent_idx))
    # Graphviz edge direction: parent (sender) → child (receiver)
    # ------------------------------------------------------------------
    def labels(e: str):
        cn, ci, pn, pi = ast.literal_eval(e)
        src = node_label(pn, pi, collapsed)   # parent  = sender
        dst = node_label(cn, ci, collapsed)   # child   = receiver
        return src, dst

    circ_pairs: set = set()
    for e in circuit:
        s, d = labels(e)
        if s != d:   # filter intra-head self-loops (arise in collapsed mode)
            circ_pairs.add((s, d))

    gt_pairs: set = set()
    if gt is not None:
        for e in gt:
            s, d = labels(e)
            if s != d:
                gt_pairs.add((s, d))

    # ------------------------------------------------------------------
    # Collect all node labels that appear in at least one edge
    # ------------------------------------------------------------------
    all_nodes: set = set()
    for s, d in circ_pairs:
        all_nodes.update([s, d])
    for s, d in gt_pairs:
        all_nodes.update([s, d])

    # Add nodes
    for lbl in all_nodes:
        g.add_node(lbl, fillcolor=fill_color(lbl))

    # Group nodes into rank=same subgraphs for layered layout
    rank_map: Dict[int, list] = {}
    for lbl in all_nodes:
        r = node_rank(lbl, collapsed)
        rank_map.setdefault(r, []).append(lbl)
    for r in sorted(rank_map):
        g.add_subgraph(rank_map[r], rank="same")

    # ------------------------------------------------------------------
    # Add edges
    # ------------------------------------------------------------------
    seen: set = set()

    def add_edge(s: str, d: str, color: str, style: str, pw: float) -> None:
        key = (s, d, color)
        if key in seen:
            return
        seen.add(key)
        g.add_edge(s, d, color=color, style=style, penwidth=str(pw))

    if gt is None:
        for s, d in circ_pairs:
            add_edge(s, d, PLAIN_COLOR, "solid", 1.8)
    else:
        for s, d in circ_pairs:
            if (s, d) in gt_pairs:
                add_edge(s, d, TP_COLOR, "solid", 2.4)   # TP — bold green
            else:
                add_edge(s, d, FP_COLOR, "solid", 1.8)   # FP — red
        for s, d in gt_pairs:
            if (s, d) not in circ_pairs:
                add_edge(s, d, FN_COLOR, "dashed", 1.4)  # FN — grey dashed

    return g


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(g: pgv.AGraph, stem: Path) -> None:
    """Write stem.gv, stem.png (200 DPI), stem.svg."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    g.write(str(stem) + ".gv")
    g.draw(path=str(stem) + ".png", prog="dot", args="-Gdpi=200")
    g.draw(path=str(stem) + ".svg", prog="dot")
    print(f"    {stem.name}.{{gv,png,svg}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality circuit diagrams.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--conditions", nargs="+", metavar="NAME", default=[],
        help="Condition names to resolve from --suite-json (e.g. random_doc random_random).",
    )
    parser.add_argument(
        "--suite-json", metavar="PATH", default=str(DEFAULT_SUITE),
        help=f"Suite JSON to resolve --conditions from.  Default: {DEFAULT_SUITE.name}",
    )
    parser.add_argument(
        "--pkl", nargs=2, action="append", metavar=("NAME", "PATH"), default=[],
        help="Load an arbitrary pkl directly.  Repeatable.",
    )
    parser.add_argument(
        "--mbg", nargs=3, action="append", metavar=("NAME", "JSON", "HA_PKL"), default=[],
        help="Load an mbg circuit (H_A ∪ accepted bundles).  Repeatable.",
    )
    parser.add_argument(
        "--runs-root", metavar="PATH", default=None,
        help="Remap run_dir paths in the suite JSON to a different root directory.",
    )
    parser.add_argument(
        "--out-dir", metavar="PATH", default=str(DEFAULT_OUT_DIR),
        help=f"Output root directory.  Default: {DEFAULT_OUT_DIR}",
    )
    args = parser.parse_args()

    if not args.conditions and not args.pkl and not args.mbg:
        parser.error("Specify at least one circuit via --conditions, --pkl, or --mbg.")

    out_dir = Path(args.out_dir)

    # ------------------------------------------------------------------
    # Always load GT
    # ------------------------------------------------------------------
    print("Loading edge sets…")
    gt = load_gt()
    print(f"  gt : {len(gt)} edges")

    # ------------------------------------------------------------------
    # Resolve --conditions from suite JSON
    # ------------------------------------------------------------------
    circuits: Dict[str, tuple] = {}   # name -> (edge_set, title)

    if args.conditions:
        suite_path = Path(args.suite_json)
        if not suite_path.exists():
            parser.error(f"Suite JSON not found: {suite_path}")
        suite_data = json.loads(suite_path.read_text(encoding="utf-8"))
        # Build condition -> run entry map (first match wins)
        run_map = {}
        for entry in suite_data["runs"]:
            cond = entry["condition"]
            if cond not in run_map:
                run_map[cond] = entry

        for cond in args.conditions:
            if cond not in run_map:
                parser.error(
                    f"Condition '{cond}' not found in {suite_path.name}. "
                    f"Available: {sorted(run_map)}"
                )
            entry = run_map[cond]
            run_dir = Path(entry["run_dir"])
            if args.runs_root:
                run_dir = Path(args.runs_root) / run_dir.name
            pkl_path = run_dir / "another_final_edges.pkl"
            if not pkl_path.exists():
                parser.error(f"pkl not found: {pkl_path}")
            edge_set = load_from_pkl(pkl_path)
            n = entry["edges_count"]
            tau = entry["threshold"]
            circuits[cond] = (edge_set, f"{cond} ({n} edges, τ={tau})")
            print(f"  {cond} : {len(edge_set)} edges")

    # ------------------------------------------------------------------
    # Explicit --pkl entries
    # ------------------------------------------------------------------
    for name, path in args.pkl:
        p = Path(path)
        if not p.exists():
            parser.error(f"pkl not found: {p}")
        edge_set = load_from_pkl(p)
        circuits[name] = (edge_set, f"{name} ({len(edge_set)} edges)")
        print(f"  {name} : {len(edge_set)} edges  (explicit pkl)")

    # ------------------------------------------------------------------
    # Explicit --mbg entries
    # ------------------------------------------------------------------
    for name, json_path, ha_pkl in args.mbg:
        jp, hp = Path(json_path), Path(ha_pkl)
        if not jp.exists():
            parser.error(f"mbg JSON not found: {jp}")
        if not hp.exists():
            parser.error(f"H_A pkl not found: {hp}")
        edge_set = load_from_mbg_json(jp, hp)
        circuits[name] = (edge_set, f"{name} ({len(edge_set)} edges)")
        print(f"  {name} : {len(edge_set)} edges  (mbg)")

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    for detail in ("collapsed", "full"):
        col = (detail == "collapsed")
        print(f"\n-- {detail} --")

        # GT standalone (always)
        g = build_graph(gt, collapsed=col, title="Ground truth (37 edges)")
        render(g, out_dir / detail / "gt")

        for name, (circ, title) in circuits.items():
            # Standalone
            g = build_graph(circ, collapsed=col, title=title)
            render(g, out_dir / detail / name)

            # vs GT
            g2 = build_graph(
                circ, collapsed=col,
                title=f"{title}  |  green=TP  red=FP  grey dashed=FN",
                gt=gt,
            )
            render(g2, out_dir / detail / f"{name}_vs_gt")

    print("\nDone. Files written to:", out_dir)


if __name__ == "__main__":
    main()
