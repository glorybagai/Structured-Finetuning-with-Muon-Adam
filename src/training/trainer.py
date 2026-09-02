"""
Training loop. Called from Experiments/run.py as:

    train(
        model=..., tokenizer=..., optimizer=..., train_ds=..., test_ds=...,
        device=..., dataset_name=..., optimizer_name=..., lr=...,
        batch_size=..., aux_lr=..., gradient_accumulation_steps=..., eval_every_n_steps=...,
    )

train_ds/test_ds come from build_datasets() in data_loader.py: each example
is {"input_ids": [...], "labels": [...], "attention_mask": [...]}, variable
length (only truncated, not padded) -- so this file's collate_fn does the
padding dynamically per batch.

Results and logs are split two ways:

  - results/{dataset_name}_results.csv -- one row per eval (train + eval),
    plus a final row at end of training. One CSV per dataset (not shared
    across no_robots / arxiv_astro_abstracts) so downstream analysis
    doesn't need to filter by dataset_name first. Rows are further keyed by
    optimizer_name, lr, batch_size, AND aux_lr, since a single dataset's
    CSV now accumulates runs that vary all four (e.g. the batch-size sweep
    and the aux_lr sweeps) -- without these as their own columns, rows from
    different configs at the same (optimizer, lr) would be indistinguishable
    from each other. aux_lr is None/blank for adam runs (adam has no aux
    parameter group -- it's a single optimizer over everything).

  - logs/{dataset_name}_{optimizer_name}_lr{lr}.log -- a dedicated,
    human-readable training curve for that single run (same lines that get
    printed to stdout), separate from sweep_logs/ (which captures raw
    subprocess stdout/stderr for the whole run including model-loading
    noise, warnings, tracebacks, etc.). This file is just the clean
    step-by-step train/eval loss history for that one config.
"""
import csv
import os
import time

import torch
from torch.utils.data import DataLoader

# Project root = three levels up from this file (src/training/trainer.py -> root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

RESULTS_FIELDS = [
    "optimizer_name", "lr", "batch_size", "aux_lr", "step", "split",
    "loss", "perplexity", "elapsed_seconds",
]


def get_results_csv_path(dataset_name):
    """One results CSV per dataset, e.g. results/no_robots_results.csv."""
    return os.path.join(RESULTS_DIR, f"{dataset_name}_results.csv")


def get_run_log_path(dataset_name, optimizer_name, lr):
    """One log file per run, e.g. logs/no_robots_adam_lr2.5e-05.log."""
    return os.path.join(LOGS_DIR, f"{dataset_name}_{optimizer_name}_lr{lr:.1e}.log")


def make_collate_fn(pad_token_id):
    """Dynamically pads input_ids/attention_mask/labels to the longest
    sequence in each batch. Labels are padded with -100 so padding never
    contributes to the loss; input_ids are padded with pad_token_id."""

    def collate_fn(batch):
        max_len = max(len(ex["input_ids"]) for ex in batch)

        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


def append_result(dataset_name: str, row: dict):
    """Appends one row to results/{dataset_name}_results.csv, writing the
    header if the file doesn't exist yet."""
    results_csv = get_results_csv_path(dataset_name)
    file_exists = os.path.isfile(results_csv)
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_run_log(log_path: str, line: str):
    """Appends one line to this run's dedicated log file."""
    with open(log_path, "a") as f:
        f.write(line + "\n")


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Runs a full pass over test_ds and returns mean loss."""
    model.eval()
    total_loss, total_batches = 0.0, 0
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def train(
    model,
    tokenizer,
    optimizer,
    train_ds,
    test_ds,
    device,
    dataset_name,
    optimizer_name,
    lr,
    batch_size=4,
    aux_lr=None,
    gradient_accumulation_steps=8,
    eval_every_n_steps=25,
):
    """Standard causal-LM fine-tuning loop with gradient accumulation and
    periodic evaluation. Works for both chat-format (masked prompt) and
    plain-text (unmasked) examples from data_loader.py -- the loss is just
    whatever HF's model.forward(labels=...) computes, so no special-casing
    is needed here between the no_robots and arxiv_astro_abstracts runs."""

    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    run_log_path = get_run_log_path(dataset_name, optimizer_name, lr)
    append_run_log(
        run_log_path,
        f"=== {dataset_name} / {optimizer_name} / lr={lr:.1e} / batch_size={batch_size} "
        f"/ aux_lr={aux_lr if aux_lr is not None else 'n/a'} ===",
    )

    model.train()
    start_time = time.time()

    step = 0
    optimizer.zero_grad(set_to_none=True)

    running_loss, running_count = 0.0, 0

    for micro_step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        # Defense in depth: build_datasets() now filters out fully-masked
        # examples (the root cause of the ultrachat_200k NaN bug), so this
        # should rarely trigger. Kept as a safety net for any future dataset
        # that slips a degenerate batch through -- e.g. every example in a
        # batch has zero unmasked label tokens, which makes cross-entropy's
        # mean-over-nothing produce NaN. Without this guard, backward() on a
        # NaN loss corrupts every model weight it touches (irrecoverably,
        # since it silently propagates), not just the logged average.
        if not torch.isfinite(outputs.loss):
            print(f"WARNING: non-finite loss ({outputs.loss.item()}) at micro_step "
                  f"{micro_step}, skipping this batch (not backpropagated).")
            continue

        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        running_loss += outputs.loss.item()
        running_count += 1

        if (micro_step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % eval_every_n_steps == 0:
                train_loss = running_loss / max(running_count, 1)
                eval_loss = evaluate(model, test_loader, device)
                elapsed = time.time() - start_time

                line = (f"[step {step}] train_loss={train_loss:.4f} "
                        f"eval_loss={eval_loss:.4f} elapsed={elapsed:.1f}s")
                print(line)
                append_run_log(run_log_path, line)

                append_result(dataset_name, {
                    "optimizer_name": optimizer_name,
                    "lr": lr, "batch_size": batch_size, "aux_lr": aux_lr,
                    "step": step, "split": "train",
                    "loss": train_loss, "perplexity": float(torch.exp(torch.tensor(train_loss))),
                    "elapsed_seconds": elapsed,
                })
                append_result(dataset_name, {
                    "optimizer_name": optimizer_name,
                    "lr": lr, "batch_size": batch_size, "aux_lr": aux_lr,
                    "step": step, "split": "eval",
                    "loss": eval_loss, "perplexity": float(torch.exp(torch.tensor(eval_loss))),
                    "elapsed_seconds": elapsed,
                })

                running_loss, running_count = 0.0, 0

    # Final eval + log at the end of training, even if the last partial
    # accumulation window didn't line up with eval_every_n_steps.
    final_eval_loss = evaluate(model, test_loader, device)
    elapsed = time.time() - start_time
    final_line = f"[final] step={step} eval_loss={final_eval_loss:.4f} elapsed={elapsed:.1f}s"
    print(final_line)
    append_run_log(run_log_path, final_line)

    append_result(dataset_name, {
        "optimizer_name": optimizer_name,
        "lr": lr, "batch_size": batch_size, "aux_lr": aux_lr,
        "step": step, "split": "final_eval",
        "loss": final_eval_loss, "perplexity": float(torch.exp(torch.tensor(final_eval_loss))),
        "elapsed_seconds": elapsed,
    })