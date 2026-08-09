"""
Sweep launcher for the full-fine-tuning LR grid across datasets/optimizers.
Run this from Full_Fine_Tuning/ (one level above Experiments/):

    python sweep.py --datasets no_robots astrophysics                  # auto-detect ALL free GPUs
    python sweep.py --datasets no_robots astrophysics --n_gpus 4       # auto-detect, use up to 4 free GPUs
    python sweep.py --datasets no_robots --optimizers adam --n_gpus 1
    python sweep.py --datasets no_robots astrophysics --n_gpus 4 --dry_run

Each (dataset, optimizer, lr) combination is one
`python -m Experiments.run ...` subprocess. Jobs are pulled from a shared
queue by however many GPU workers were resolved, each pinned to one
physical GPU via CUDA_VISIBLE_DEVICES for the duration of its job, so no two
jobs ever land on the same GPU at once.
Per-run stdout/stderr goes to sweep_logs/<run_name>.log so a failed run
doesn't get lost in interleaved terminal output.

Anything after --extra_args is passed straight through to Experiments.run, e.g.:
    python sweep.py --datasets no_robots --n_gpus 4 --extra_args --batch_size 8 --n_train 3200

GPU SELECTION (on a shared node, which physical GPUs are free changes over
time -- other people's jobs come and go):
  - Default (no --gpu_ids, no --n_gpus): auto-detects every GPU currently
    under the memory threshold via `nvidia-smi` and uses all of them.
  - --n_gpus N (no --gpu_ids): auto-detects free GPUs the same way, then
    uses the first N of them. Does NOT assume ids 0..N-1 are free.
  - --gpu_ids: pins to exactly the physical GPU ids given, no auto-detection
    at all -- use this if you want to force specific GPUs regardless of what
    nvidia-smi reports free, e.g.:
        python sweep.py --datasets no_robots --gpu_ids 3 4 5
"""

import argparse
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ADAM_LRS = [5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5, 3.5e-5, 4e-5, 5e-5, 7e-5]
MUON_LRS = [1e-5, 2e-5, 5e-5, 7e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4, 7.5e-4, 1e-3]

OPTIMIZER_LRS = {
    "adam": ADAM_LRS,
    "muon": MUON_LRS,
}

# A GPU with less than this much memory used is considered "free". 1000 MiB
# gives headroom above driver/idle overhead (e.g. the 0-3MiB seen on truly
# idle GPUs) without being so low that a lightly-used GPU slips through.
DEFAULT_FREE_MEM_THRESHOLD_MIB = 1000


def get_free_gpu_ids(threshold_mib=DEFAULT_FREE_MEM_THRESHOLD_MIB):
    """Queries nvidia-smi for GPUs with less than threshold_mib memory used.
    Returns their physical indices, in ascending order. Falls back to [0]
    with a warning if nvidia-smi can't be queried at all (e.g. no GPU driver
    visible in this environment)."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        print(f"WARNING: could not query nvidia-smi ({e}); falling back to GPU 0 only. "
              f"Use --gpu_ids to specify manually if this is wrong.")
        return [0]

    free_ids = []
    for line in output.strip().splitlines():
        idx_str, mem_str = (x.strip() for x in line.split(","))
        if int(mem_str) < threshold_mib:
            free_ids.append(int(idx_str))
    return free_ids


def resolve_gpu_ids(args):
    """Priority: explicit --gpu_ids > auto-detected free GPUs (capped at
    --n_gpus if given) > error if nothing free."""
    if args.gpu_ids:
        return args.gpu_ids

    free_ids = get_free_gpu_ids(args.gpu_mem_threshold)
    if not free_ids:
        raise SystemExit(
            "No free GPUs detected via nvidia-smi (all above the memory "
            f"threshold of {args.gpu_mem_threshold} MiB). Check `nvidia-smi` "
            "manually, or pass --gpu_ids to force specific GPUs anyway."
        )

    if args.n_gpus is not None:
        gpu_ids = free_ids[: args.n_gpus]
        if len(gpu_ids) < args.n_gpus:
            print(f"WARNING: requested --n_gpus {args.n_gpus} but only "
                  f"{len(free_ids)} free GPU(s) detected ({free_ids}); using {gpu_ids}.")
    else:
        gpu_ids = free_ids

    print(f"Auto-detected free GPUs: {free_ids} -> using {gpu_ids}")
    return gpu_ids


def build_jobs(datasets, optimizers):
    jobs = []
    for dataset in datasets:
        for optimizer in optimizers:
            for lr in OPTIMIZER_LRS[optimizer]:
                jobs.append({"dataset": dataset, "optimizer": optimizer, "lr": lr})
    return jobs


def run_job(job, gpu_id, extra_args, log_dir, dry_run):
    dataset, optimizer, lr = job["dataset"], job["optimizer"], job["lr"]
    run_name = f"{dataset}_{optimizer}_lr{lr:.1e}"
    log_path = os.path.join(log_dir, f"{run_name}.log")

    cmd = [
        sys.executable, "-m", "Experiments.run",
        "--dataset", dataset,
        "--optimizer", optimizer,
        "--lr", str(lr),
    ] + extra_args

    if dry_run:
        print(f"[gpu {gpu_id}] DRY RUN: {' '.join(cmd)}")
        return run_name, 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[gpu {gpu_id}] START {run_name}")
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[gpu {gpu_id}] {status}: {run_name} (log: {log_path})")
    return run_name, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True,
                         help="Dataset keys from DATASET_CONFIGS in data_utils.py, e.g. no_robots astrophysics")
    parser.add_argument("--optimizers", nargs="+", default=["adam", "muon"], choices=["adam", "muon"])
    parser.add_argument("--n_gpus", type=int, default=None,
                         help="How many auto-detected free GPUs to use. Omit to use ALL free GPUs found. "
                              "Ignored if --gpu_ids is given.")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None,
                         help="Explicit physical GPU ids to use, e.g. --gpu_ids 3 4 5. "
                              "Overrides auto-detection entirely -- use this to force specific "
                              "GPUs regardless of what nvidia-smi currently reports as free.")
    parser.add_argument("--gpu_mem_threshold", type=int, default=DEFAULT_FREE_MEM_THRESHOLD_MIB,
                         help=f"MiB of memory used below which a GPU is considered free for "
                              f"auto-detection (default {DEFAULT_FREE_MEM_THRESHOLD_MIB}). Ignored if --gpu_ids is given.")
    parser.add_argument("--log_dir", type=str, default="sweep_logs")
    parser.add_argument("--dry_run", action="store_true", help="Print the commands without running them")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[],
                         help="Everything after this flag is forwarded to run.py as-is")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    jobs = build_jobs(args.datasets, args.optimizers)
    gpu_ids = resolve_gpu_ids(args)
    print(f"Queued {len(jobs)} runs ({args.datasets} x {args.optimizers}) across GPUs {gpu_ids}")

    # A queue of free GPU ids: each worker claims one before running a job and
    # returns it after, so exactly len(gpu_ids) jobs are ever in flight at once
    # and two jobs never share a physical GPU.
    gpu_queue = queue.Queue()
    for gid in gpu_ids:
        gpu_queue.put(gid)

    def worker(job):
        gpu_id = gpu_queue.get()
        try:
            return run_job(job, gpu_id, args.extra_args, args.log_dir, args.dry_run)
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        results = list(executor.map(worker, jobs))

    failed = [r for r in results if r[1] != 0]
    print(f"\n=== SWEEP DONE: {len(results) - len(failed)}/{len(results)} succeeded ===")
    if failed:
        print("Failed runs:")
        for name, code in failed:
            print(f"  {name} (exit {code})")


if __name__ == "__main__":
    main()