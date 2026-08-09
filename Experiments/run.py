"""
Entry point. Lives inside the Experiments package now, so it's launched as:

    python -m Experiments.run --dataset no_robots --optimizer adam --lr 2.5e-5

from the Full_Fine_Tuning/ directory (one level above Experiments/), not run
directly with `python run.py` from inside Experiments/ -- the relative
imports below only resolve when it's invoked as a module within the package.

To add a new dataset, add one entry to DATASET_CONFIGS in data_utils.py --
nothing else in this file needs to change.
"""

import argparse
from src.Model.Qwen_loader import load_model_and_tokenizer
from src.Data.data_loader import build_datasets, DATASET_CONFIGS
from src.Optimizers.adam import build_adam_optimizer
from src.Optimizers.muon import build_muon_optimizer
from src.training.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--optimizer", type=str, required=True, choices=["adam", "muon"])
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_every_n_steps", type=int, default=25)
    parser.add_argument("--n_train", type=int, default=6400)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.95, help="Muon momentum (ignored for adam)")
    parser.add_argument("--adjust_lr_fn", type=str, default=None, choices=[None, "original", "match_rms_adamw"],
                         help="Muon LR adjustment (torch.optim.Muon arg, ignored for adam)")
    parser.add_argument("--adamw_lr", type=float, default=None,
                         help="LR for the AdamW fallback params under Muon (embeddings/head/norms). Defaults to --lr.")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model_name)

    train_ds, test_ds = build_datasets(
        args.dataset, tokenizer,
        n_train=args.n_train, n_test=args.n_test, max_length=args.max_length,
    )

    if args.optimizer == "adam":
        optimizer = build_adam_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = build_muon_optimizer(
            model, lr=args.lr, weight_decay=args.weight_decay,
            momentum=args.momentum, adjust_lr_fn=args.adjust_lr_fn,
            adamw_lr=args.adamw_lr,
        )

    train(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_ds=train_ds,
        test_ds=test_ds,
        device=device,
        dataset_name=args.dataset,
        optimizer_name=args.optimizer,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_every_n_steps=args.eval_every_n_steps,
    )


if __name__ == "__main__":
    main()