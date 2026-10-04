# Corruption Strategies for Activation Patching in ACDC

CS344 Undergraduate Dissertation — University of Warwick, 2026  
**Student:** u5504941  
**Supervisor:** Professor Ranko Lazic

Built on the [Conmy et al. (2023) ACDC codebase](https://github.com/ArthurConmy/Automatic-Circuit-Discovery).

---

## Overview

This project systematically benchmarks nine corruption strategies applied to Automatic Circuit Discovery (ACDC) on the Python Docstring Completion task, evaluating circuit quality and whether a two-pass procedure combining two corruptions outperforms any single-pass run.

**Three research questions were addressed:**
- RQ1: Which corruption strategies produce the highest-yield circuits?
- RQ2: Do high-yield corruptions recover distinct or overlapping circuits?
- RQ3: Can a two-pass procedure outperform single-pass at comparable sparsity?

**Main Result:**
1. The method of corruption used by Conmy et al., complete token randomisation (random ablation), proved to be the most effective single pass corruption by a substantial difference.
2. As a result of the destructive nature of random ablation, all corruption strategies (except for zero ablation) effectively recovered a subset of the circuit gained from random ablation
3. It was found that an approach of using a random ablation circuit as a filter for an edge pool derived using zero ablation yields a circuit with empirically higher quality than single-pass random ablation, as well as outperforming the Ground-Truth - potentially due to finding edges which were previously unspecified in the codebase.    

---

## Environment Setup

**Requirements:** Windows 10/11, Miniconda virtual environment (downloadable from the anaconda website),  ~8GB disk space (~6GB is used for env downloads)


```bash
# environment.yml contains the necessary dependencies and pip components for running the suite
conda env create -f environment.yml -n acdc
conda activate acdc
```

> **Model download:** The first ACDC run downloads the `attn-only-4l` model weights (~50 MB) from HuggingFace automatically via TransformerLens. An internet connection is required for this step; subsequent runs use the cached copy.

---

## Reproducing Results

All authoritative results used were ran on a suite using threshold τ = 0.10 with cpu as the device
### Stage 0 — Run the corruption suite

Runs all 9 conditions (8 corruptions + zero ablation) over ACDC in one go and retrieves circuits. Saves results to `runs/`:

```bash
# Run the following in terminal - if there are errors then please check that the env has been correctly set up
python run_suite.py --task docstring --device cpu --threshold 0.10 --seeds 0
```

Output: `runs/SUITE_<timestamp>_docstring.json` plus per-run subdirectories with edge pkl files, stdout logs, and Mode B stats.

> **Runtime:** Each of the 9 ACDC conditions takes approximately 20–40 minutes on CPU. 4–6 hours were used for the full suite to complete. The suite runs sequentially with a 30-second pause between conditions to release OS resources.

> Multiple seeds produce identical results — the Docstring dataset generator uses a fixed internal seed independent of `--seed`... so no use fiddling with it

### Stage 1 — Yield table + Jaccard / complementarity analysis

Computes Mode B stats (precision, recall, F1 vs ground truth) for all recovered circuits, alongside pairwise Jaccard similarity:

```bash
python analyse_suite.py --suite-json runs/SUITE_<timestamp>_docstring.json
```

Output: Mode B yield table — saved to `analysis/stage1_yield/`; N×N Jaccard matrix JSON and heatmap PNG — saved to `analysis/stage2_jaccard/`.

### Stage 2 — Naive union (Mode B)

Tests all pairwise naive unions of circuits from the suite under Mode B (F1 vs ground truth):

```bash
python two_pass.py --suite-json runs/SUITE_<timestamp>_docstring.json
```

Output: pairwise union stats and ΔRecall matrix — saved to `analysis/stage3a_naive_union/`.

### Stage 3 — Marginal Bundle Gain

Runs the MBG filtered two-pass procedure using two specified conditions from the suite:

```bash
python marginal_bundle_gain.py \
    --suite-json runs/SUITE_<timestamp>_docstring.json \
    --condition-a random_random \
    --condition-b zero_ablation \
    --threshold 0.10 \
    --device cpu
```

Output: per-bundle KL results and final MBG circuit JSON — saved to `analysis/stage3b_marginal_gain/`.

### Mode A evaluation (KL divergence)

Evaluates circuits under resample ablation (Conmy-comparable, primary) or zero ablation (secondary).
Each circuit is specified as `--pkl NAME PATH`. `--mbg NAME MBG_JSON HA_PKL` loads a Marginal Bundle Gain circuit.
Pre-computed results are in `analysis/` — re-running is only necessary to verify from scratch.

```bash
# Resample ablation (primary, Conmy-comparable)
python eval_circuit_efficiency.py \
    --task docstring --device cpu --corruption random_random \
    --pkl random_random runs/SUITE_<timestamp>_docstring/<run_dir>/another_final_edges.pkl \
    --pkl zero_ablation runs/SUITE_<timestamp>_docstring/<run_dir>/another_final_edges.pkl \
    --mbg mbg_random analysis/stage3b_marginal_gain/marginal_bundle_gain_random-random_zero-ablation_<suite_stem>.json \
          runs/SUITE_<timestamp>_docstring/<rr_run_dir>/another_final_edges.pkl \
    --out analysis/mode_a_resample.json

# Zero ablation (secondary)
python eval_circuit_efficiency.py \
    --task docstring --device cpu --corruption zero \
    --pkl random_random runs/SUITE_<timestamp>_docstring/<run_dir>/another_final_edges.pkl \
    --out analysis/mode_a_zero.json
```

### Circuit visualisation

Runs pygraphviz to generate circuit diagrams:

```bash
python visualise_circuits.py \
    --suite-json runs/SUITE_<timestamp>_docstring.json \
    --conditions random_random random_doc random_def random_answer \
                 random_def_doc random_answer_doc \
                 vary_length_doc_desc_random_doc zero_ablation
```

Output: PNG/SVG circuit diagrams — saved to `images/circuits/`.

---

## Key Files

| File | Purpose |
|---|---|
| `acdc/main.py` | ACDC entry point; `--dataset-version` arg added |
| `acdc/TLACDCExperiment.py` | Core experiment class; `--corrupted-batch-size` added |
| `run_suite.py` | Stage 0 — automated multi-condition runner |
| `analyse_suite.py` | Stage 1 — Mode B yield table + Jaccard matrix + heatmap |
| `two_pass.py` | Stage 2 — naive union Mode B evaluation |
| `marginal_bundle_gain.py` | Stage 3 — Marginal Bundle Gain filtered two-pass |
| `eval_circuit_efficiency.py` | Mode A KL evaluation under resample or zero ablation |
| `visualise_circuits.py` | Publication-quality circuit diagrams |
| `compare_edges.py` | Utility — pairwise Jaccard + unique edge inspection between two pkl files |
| `runs/` | Timestamped run outputs (edge pkls, stdout, metadata, Mode B stats) |
| `analysis/` | Computed results: yield tables, Jaccard matrices, efficiency JSONs |

---

## Modifications to Upstream Codebase

The following changes were made to the original Conmy (2023) ACDC rep alongside the additional python files:

- As of the most recent commit, all google colab integration has been removed (including use of plotly, colab, etc... this may be re-implemented soon, though my additions do not at all interact with google colab)
- `--dataset-version` flag in `acdc/main.py` to expose the full corruption family
- `--corrupted-batch-size` batching in `TLACDCExperiment.py` for memory efficiency
- Mode B ground truth evaluation (recall bug fix) at end of `acdc/main.py`
- 7 stability fixes for torch 2.5.1 + Windows (torchtyping import guard, corrupted cache OOM fix, deprecated `cache_all` replaced with `run_with_cache`, PNG rendering guard, explicit VRAM release on exit)
