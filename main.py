"""
Sweep launcher for the full-fine-tuning LR grid across datasets/optimizers.
Run this from Full_Fine_Tuning_4datasets/:

    python main.py --datasets no_robots --n_gpus 6
    python main.py --experiment aux_bs_scaling --datasets no_robots --n_gpus 6
    python main.py --experiment aux_fine --datasets no_robots --n_gpus 6
    python main.py --experiment bs_transfer_fixed --datasets no_robots --n_gpus 6
    python main.py --experiment full_bs_sweep --datasets no_robots --n_gpus 6
    python main.py --datasets no_robots --dry_run

--experiment selects which job grid gets built:
  standard          -- original dataset x optimizer x lr (x batch_size for muon) grid
  aux_bs_scaling    -- muon_lr fixed at its own best-per-batch-size value; aux_lr swept
                       around a sqrt-scaling prediction, per batch size
  aux_fine          -- fine-grained aux_lr sweep at the single reference point
                       (batch_size=4, muon_lr=3e-4)
  bs_transfer_fixed -- batch_size swept with BOTH muon_lr and aux_lr pinned at the
                       single reference values, no retuning at all -- baseline for
                       how much retuning actually buys you
  full_bs_sweep     -- MUON ONLY. effective batch size 4..2048 (micro-batch fixed at 4,
                       reached purely via gradient_accumulation_steps) x 7-point muon_lr
                       grid x 7-point aux_lr grid. n_train is held fixed (whatever
                       Experiments.run's default is) across every batch size -- this is
                       the CIFAR-style fixed-seen-datapoints design (per Jonas's question
                       relayed by Antonio): step count varies inversely with batch size,
                       examples seen does not. Adam is NOT included in this experiment --
                       it gets its own separate sweep across the same batch range later.

GPU SELECTION:
  - Default: auto-detects every GPU under the memory threshold.
  - --n_gpus N: auto-detects, uses the first N free ones.
  - --gpu_ids: pins to exactly the ids given, no auto-detection.

NOTE: GPU auto-detection runs ONCE at sweep start -- if a GPU that was free
becomes busy partway through, jobs already queued to that worker slot will
still be dispatched there and may OOM.
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

MUON_BATCH_SIZES = [4, 8, 12, 16, 20]
ADAM_FIXED_BATCH_SIZE = 4

REFERENCE_MUON_LR = 3e-4
REFERENCE_AUX_LR = 3.5e-5

BEST_MUON_LR_PER_BATCH = {4: 3e-4, 8: 5e-4, 12: 5e-4, 16: 7.5e-4, 20: 7.5e-4}

AUX_LR_GRID_PER_BATCH = {
    4:  [1.5e-5, 2e-5, 2.5e-5, 3.5e-5, 5e-5],
    8:  [2.5e-5, 3.5e-5, 5e-5, 7e-5, 1e-4],
    12: [3.5e-5, 5e-5, 7e-5, 1e-4, 1.5e-4],
    16: [3.5e-5, 5e-5, 7e-5, 1e-4, 1.5e-4],
    20: [3.5e-5, 5e-5, 7.5e-5, 1e-4, 1.5e-4, 2e-4],
}

AUX_LR_FINE_GRID = [1.5e-5, 2e-5, 2.5e-5, 3e-5, 3.5e-5, 4e-5, 4.5e-5, 5e-5, 6e-5]

# full_bs_sweep -- CIFAR-style fixed-seen-datapoints design (Jonas's question).
# MUON ONLY. Adam gets its own separate sweep later, across the same batch
# range, as its own experiment -- not included here. Micro-batch fixed at
# FULL_BS_SWEEP_MICRO_BATCH; effective batch reached purely via
# grad_accum_steps, so no new memory risk at any target.
FULL_BS_SWEEP_EFFECTIVE_BATCHES = [4, 8, 16, 20, 32, 64, 128, 256, 512, 1024, 2048]
FULL_BS_SWEEP_MICRO_BATCH = 4
FULL_BS_SWEEP_MUON_LRS = [1e-4, 2e-4, 3e-4, 5e-4, 7.5e-4, 1.5e-3, 3e-3]
FULL_BS_SWEEP_AUX_LRS = [1e-5, 2e-5, 3.5e-5, 7e-5, 1.5e-4, 3e-4, 6e-4]

# Kept for a later, separate Adam-only sweep across the same batch range --
# not used in build_jobs_full_bs_sweep below.
FULL_BS_SWEEP_ADAM_LRS = [5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4]

DEFAULT_FREE_MEM_THRESHOLD_MIB = 1000


def get_free_gpu_ids(threshold_mib=DEFAULT_FREE_MEM_THRESHOLD_MIB):
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


def build_jobs_standard(datasets, optimizers):
    jobs = []
    for dataset in datasets:
        if "adam" in optimizers:
            for lr in OPTIMIZER_LRS["adam"]:
                jobs.append({"dataset": dataset, "optimizer": "adam", "lr": lr,
                             "batch_size": ADAM_FIXED_BATCH_SIZE, "aux_lr": None})
        if "muon" in optimizers:
            for lr in OPTIMIZER_LRS["muon"]:
                for bs in MUON_BATCH_SIZES:
                    jobs.append({"dataset": dataset, "optimizer": "muon", "lr": lr,
                                 "batch_size": bs, "aux_lr": None})
    return jobs


def build_jobs_aux_bs_scaling(datasets):
    jobs = []
    for dataset in datasets:
        for bs, muon_lr in BEST_MUON_LR_PER_BATCH.items():
            for aux_lr in AUX_LR_GRID_PER_BATCH[bs]:
                jobs.append({"dataset": dataset, "optimizer": "muon", "lr": muon_lr,
                             "batch_size": bs, "aux_lr": aux_lr})
    return jobs


def build_jobs_aux_fine(datasets):
    jobs = []
    for dataset in datasets:
        for aux_lr in AUX_LR_FINE_GRID:
            jobs.append({"dataset": dataset, "optimizer": "muon", "lr": REFERENCE_MUON_LR,
                         "batch_size": 4, "aux_lr": aux_lr})
    return jobs


def build_jobs_bs_transfer_fixed(datasets):
    jobs = []
    for dataset in datasets:
        for bs in MUON_BATCH_SIZES:
            jobs.append({"dataset": dataset, "optimizer": "muon", "lr": REFERENCE_MUON_LR,
                         "batch_size": bs, "aux_lr": REFERENCE_AUX_LR})
    return jobs


def build_jobs_full_bs_sweep(datasets):
    """CIFAR-style fixed-seen-datapoints sweep across effective batch 4..2048,
    MUON ONLY. Micro-batch fixed at FULL_BS_SWEEP_MICRO_BATCH; effective batch
    reached purely via grad_accum_steps. n_train is NOT set here -- left at
    Experiments.run's own default (6400) for every job, so every point shares
    the same example budget (CIFAR's "fixed seen datapoints" design).

    Full 3-way grid: 11 batch sizes x 7 muon_lr x 7 aux_lr = 539 jobs.
    """
    jobs = []
    for dataset in datasets:
        for eff_batch in FULL_BS_SWEEP_EFFECTIVE_BATCHES:
            grad_accum = eff_batch // FULL_BS_SWEEP_MICRO_BATCH
            for muon_lr in FULL_BS_SWEEP_MUON_LRS:
                for aux_lr in FULL_BS_SWEEP_AUX_LRS:
                    jobs.append({
                        "dataset": dataset, "optimizer": "muon", "lr": muon_lr,
                        "batch_size": FULL_BS_SWEEP_MICRO_BATCH,
                        "grad_accum_steps": grad_accum, "aux_lr": aux_lr,
                    })
    return jobs


EXPERIMENT_BUILDERS = {
    "aux_bs_scaling": build_jobs_aux_bs_scaling,
    "aux_fine": build_jobs_aux_fine,
    "bs_transfer_fixed": build_jobs_bs_transfer_fixed,
    "full_bs_sweep": build_jobs_full_bs_sweep,
}


def run_job(job, gpu_id, extra_args, log_dir, dry_run):
    dataset, optimizer, lr, batch_size = job["dataset"], job["optimizer"], job["lr"], job["batch_size"]
    aux_lr = job.get("aux_lr")
    grad_accum = job.get("grad_accum_steps")  # None => Experiments.run's own default (8) applies

    effective_batch = batch_size * (grad_accum if grad_accum is not None else 8)
    tag = (f"{dataset}_{optimizer}_lr{lr:.1e}_micro{batch_size}"
           f"_accum{grad_accum if grad_accum is not None else 8}_eff{effective_batch}")
    if aux_lr is not None:
        tag += f"_aux{aux_lr:.1e}"
    log_path = os.path.join(log_dir, f"{tag}.log")

    cmd = [
        sys.executable, "-m", "Experiments.run",
        "--dataset", dataset,
        "--optimizer", optimizer,
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--gradient_checkpointing",
    ]
    if grad_accum is not None:
        cmd += ["--gradient_accumulation_steps", str(grad_accum)]
    if aux_lr is not None:
        cmd += ["--adamw_lr", str(aux_lr)]
    cmd += extra_args

    if dry_run:
        print(f"[gpu {gpu_id}] DRY RUN: {' '.join(cmd)}")
        return tag, 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[gpu {gpu_id}] START {tag}")
    with open(log_path, "w") as f:
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[gpu {gpu_id}] {status}: {tag} (log: {log_path})")
    return tag, result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="standard",
                         choices=["standard"] + list(EXPERIMENT_BUILDERS.keys()))
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--optimizers", nargs="+", default=["adam", "muon"], choices=["adam", "muon"],
                         help="Only used for --experiment standard.")
    parser.add_argument("--n_gpus", type=int, default=None)
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None)
    parser.add_argument("--gpu_mem_threshold", type=int, default=DEFAULT_FREE_MEM_THRESHOLD_MIB)
    parser.add_argument("--log_dir", type=str, default="sweep_logs")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    if args.experiment == "standard":
        jobs = build_jobs_standard(args.datasets, args.optimizers)
    else:
        jobs = EXPERIMENT_BUILDERS[args.experiment](args.datasets)

    gpu_ids = resolve_gpu_ids(args)
    print(f"Experiment: {args.experiment}")
    print(f"Queued {len(jobs)} runs across GPUs {gpu_ids}")

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
    print(f"\n=== SWEEP DONE ({args.experiment}): {len(results) - len(failed)}/{len(results)} succeeded ===")
    if failed:
        print("Failed runs:")
        for name, code in failed:
            print(f"  {name} (exit {code})")


if __name__ == "__main__":
    main()