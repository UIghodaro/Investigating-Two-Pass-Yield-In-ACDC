"""
Stage 0 - run_suite.py
----------------------
Automate the full ACDC corruption suite, running all conditions sequentially
and collecting results into a timestamped summary JSON.

Usage:
  python run_suite.py --task docstring --device cpu --threshold 0.10 --seeds 0
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, cwd=None, stdout_path=None):
    print("\n>>>", " ".join(cmd), "\n")
    
    if stdout_path is None:
        subprocess.run(cmd, cwd=cwd, check=True)
        return

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "w", encoding="utf-8") as f:
        # expandable_segments=True tells PyTorch's CUDA allocator to grow existing
        # memory segments rather than always requesting new fixed-size blocks.
        # This prevents OOM errors caused by allocator fragmentation on small GPUs
        # (observed on RTX 3050 4GB when the unembed logit tensor can't be placed
        # in a contiguous block despite sufficient total free VRAM).
        # PyTorch's own OOM error message suggested this setting as a remedy.
        env = os.environ.copy()
        if "--device" in cmd and cmd[cmd.index("--device") + 1] == "cuda":
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        try:
            for line in p.stdout:
                sys.stdout.write(line)
                f.write(line)
        finally:
            # Explicitly close the pipe handle in the finally block so it is
            # released immediately rather than waiting for CPython's GC.
            # On Windows, GC timing is non-deterministic and pipe handles left
            # open across multiple subprocess runs caused progressive resource
            # exhaustion: run 2 crashed mid-init, run 3 crashed earlier, run 4
            # crashed during Python imports before any ACDC code ran.
            p.stdout.close()
            rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)

def python_env_info():
    info = {
        "python_exe": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    # torch/cuda info if available
    try:
        import torch
        info.update({
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "torch_cuda_available": torch.cuda.is_available(),
        })
    except Exception as e:
        info["torch_import_error"] = repr(e)
    return info

def load_edges_count(run_dir: Path):
    """Count edges in the final circuit (another_final_edges.pkl). (Doesn't fail the run if missing.)"""
    import pickle
    p = run_dir / "another_final_edges.pkl"
    if not p.exists():
        return None
    try:
        obj = pickle.load(open(p, "rb"))
        # another_final_edges.pkl is a list of (edge_key, score) tuples for the completed circuit
        return len(obj)
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Stage 0: automated multi-condition ACDC corruption suite runner."
    )
    parser.add_argument("--task", default="docstring")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--corrupted-batch-size", type=int, default=0,
        help="Examples per batch for corrupted cache build. Lower = less GPU memory.")
    # Seeds do not alter results for the docstring task — internal seed is fixed.
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds, e.g. 0,1")
    parser.add_argument("--metric", default=None)
    parser.add_argument("--extra", default="",
        help="Extra flags passed through verbatim to acdc.main.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_root = repo_root / "runs"
    runs_root.mkdir(exist_ok=True)


    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Docstring corruption variants (dataset_version keys from docstring_induction_prompt_generator).
    # Comment out individual entries to run a subset.
    DOCSTRING_VERSIONS = [
        "random_random",               # default; both def and doc args fully randomised
        "random_doc",                  # doc arg names replaced; def intact
        "random_def",                  # def arg names replaced; doc intact
        "random_answer",               # only the target answer arg in def replaced
        "random_def_doc",              # random_def + random_doc combined
        "random_answer_doc",           # random_answer + random_doc combined
        "vary_length_doc_desc",        # arg names preserved; description words redistributed
        "vary_length_doc_desc_random_doc",  # description redistributed + random_doc
    ]

    # If a condition doesn't work, comment it out and rerun.
    conditions = (
        [(v, ["--dataset-version", v]) for v in DOCSTRING_VERSIONS]
        + [("zero_ablation", ["--zero-ablation"])]
    )

    summary_rows = []

    # Concurrently run corruption strategies under different seeds
    for seed in seeds:
        for (cond_name, cond_flags) in conditions:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{ts}_{args.task}_{cond_name}_t{args.threshold}_s{seed}_{args.device}"
            run_dir = runs_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            # Write metadata
            config = {
                "task": args.task,
                "device": args.device,
                "threshold": args.threshold,
                "corrupted_batch_size": args.corrupted_batch_size,
                "seed": seed,
                "metric": args.metric,
                "cond_name": cond_name,
                "cond_flags": cond_flags,
                "extra": args.extra,
            }
            (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

            # Build ACDC.main command for console
            cmd = [
                sys.executable, "-u", "-m", "acdc.main",
                "--task", args.task,
                "--threshold", str(args.threshold),
                "--device", args.device,
                "--corrupted-batch-size", str(args.corrupted_batch_size),
                "--seed", str(seed),
            ]

            if args.metric:
                cmd += ["--metric", args.metric]

            cmd += cond_flags

            # Pass-through extra flags (string)
            if args.extra.strip():
                cmd += args.extra.strip().split()

            # Remove stale root-level artefacts so a crashed run doesn't inherit the previous run's files
            for fname in ["edges.pkl", "another_final_edges.pkl", "mode_b_stats.json"]:
                (repo_root / fname).unlink(missing_ok=True)

            # Run and log; retry once on crash (0xC0000005 access violations are
            # non-deterministic on Windows + torch CPU and usually clear on a second attempt)
            stdout_path = run_dir / "stdout.txt"
            for attempt in range(2):
                try:
                    run(cmd, cwd=repo_root, stdout_path=stdout_path)
                    break  # success
                except Exception as e:
                    try:
                        lines = stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        tail = "\n".join(lines[-30:])
                    except Exception:
                        tail = "(stdout unavailable)"
                    if attempt == 0:
                        print(f"\n[run_suite] RUN FAILED (attempt 1), retrying: {run_name} "
                              f"(exit code {e.returncode if hasattr(e, 'returncode') else '?'})",
                              flush=True)
                        time.sleep(10)
                    else:
                        print(f"\n[run_suite] RUN FAILED (attempt 2), giving up: {run_name} "
                              f"(exit code {e.returncode if hasattr(e, 'returncode') else '?'})",
                              flush=True)
                        (run_dir / "error.txt").write_text(tail, encoding="utf-8")

            # Copy expected artefacts into run_dir if they were created elsewhere
            # If ACDC already writes into a run folder, skip this section later.
            for fname in ["edges.pkl", "another_final_edges.pkl", "mode_b_stats.json"]:
                src = repo_root / fname
                if src.exists():
                    dst = run_dir / fname
                    try:
                        dst.write_bytes(src.read_bytes())
                    except Exception:
                        pass

            edge_count = load_edges_count(run_dir)

            mode_b_stats = {}
            mode_b_path = run_dir / "mode_b_stats.json"
            if mode_b_path.exists():
                try:
                    mode_b_stats = json.loads(mode_b_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            summary_rows.append({
                "run_dir": str(run_dir),
                "task": args.task,
                "condition": cond_name,
                "seed": seed,
                "threshold": args.threshold,
                "device": args.device,
                "corrupted_batch_size": args.corrupted_batch_size,
                "edges_count": edge_count,
                "status": "ok" if not (run_dir / "error.txt").exists() else "error",
                **mode_b_stats,
            })

    # Write suite summary
    summary_path = runs_root / f"SUITE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.task}.json"
    suite_summary = {"env": python_env_info(), "runs": summary_rows}
    summary_path.write_text(json.dumps(suite_summary, indent=2), encoding="utf-8")
    print("\nSuite summary:", summary_path)

if __name__ == "__main__":
    main()
