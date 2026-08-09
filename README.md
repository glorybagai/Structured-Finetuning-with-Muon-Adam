📘 README — Modular Finetuning Pipeline for Qwen (Adam + Muon)
🔥 Overview
This repository contains a modular, experiment‑driven finetuning pipeline for Qwen models, supporting:

Full finetuning (no LoRA)

Multiple datasets (plug‑and‑play)

Multiple optimizers:

AdamW (baseline)

Muon (first‑principles implementation with Newton–Schulz orthogonalization)

Reproducible experiments via CLI

Clean separation between reusable core code (src/) and experiment logic (experiments/)

The goal is to make it easy to run controlled comparisons between AdamW vs Muon across any dataset, without rewriting scripts.

📁 Project Structure
Code
full_finetuning/
│
├── src/                         # Core reusable code (never experiment-specific)
│   ├── model/                   # Qwen loading, tokenizer, training config
│   ├── training/                # Training loop, evaluation loop
│   ├── optimizers/              # AdamW + Muon implementations
│   ├── data/                    # Dataset loaders, chat formatting, preprocessing
│   └── utils/                   # Logging, metrics, checkpoint helpers
│
├── experiments/
│   └── run.py                   # CLI entry point for all experiments
│
├── .venv/                       # Virtual environment (local)
├── pyproject.toml               # Dependencies
└── README.md                    # This file
This structure ensures maximum modularity:
all reusable logic lives in src/, while run.py orchestrates experiments.

🚀 Running Experiments
You run experiments through the unified CLI:

AdamW
Code
python experiments/run.py --dataset no_robots --optimizer adam --lr 2.5e-5
Muon
Code
python experiments/run.py --dataset no_robots --optimizer muon --lr 0.02
Common flags
Code
--batch_size 4
--gradient_accumulation_steps 8
--eval_every_n_steps 25
--n_train 6400
--n_test 100
--max_length 1024
--momentum 0.95
--weight_decay 0.01
--adamw_lr <float>   # fallback LR for embeddings/head/norms under Muon
📦 Adding a New Dataset
All dataset definitions live in:

Code
src/data/dataset_loader.py
Add a new entry to DATASET_CONFIGS:

python
DATASET_CONFIGS = {
    "no_robots": {
        "hf_name": "HuggingFaceH4/no_robots",
        "train_split": "train",
        "test_split": "test"
    },
    "my_new_dataset": {
        "hf_name": "myorg/mydataset",
        "train_split": "train_sft",
        "test_split": "test_sft"
    }
}
Then run:

Code
python experiments/run.py --dataset my_new_dataset --optimizer adam --lr 2e-5
No other files need to change.

🧠 Muon Optimizer (First‑Principles Implementation)
This repository includes a full Muon implementation based on Keller Jordan’s reference design:

Momentum update

Quintic Newton–Schulz orthogonalization

Aspect‑ratio scaling

Parameter routing:

Muon → hidden layer matrices (ndim ≥ 2, excluding embeddings/head)

AdamW → embeddings, LM head, norms, biases

This is not PyTorch’s built‑in torch.optim.Muon.
It is a research‑grade implementation designed for experimentation.

Located in:

Code
src/optimizers/muon.py
Muon + AdamW are wrapped into a single CombinedOptimizer so the training loop does not need special handling.

🏋️ Training Loop
The training loop is fully generic and lives in:

Code
src/training/trainer.py
Features:

Gradient accumulation

Periodic evaluation

CSV logging per run

Final results appended to results.csv

Works identically for AdamW and Muon

🧪 Evaluation
Evaluation uses:

Negative log-likelihood (NLL)

Average validation loss

Token-level masking for chat-format datasets

Located in:

Code
src/training/evaluator.py
🧩 Model Loading
Qwen model + tokenizer loading lives in:

Code
src/model/qwen_loader.py
Includes:

bf16 loading

left-padding tokenizer

use_cache=False for training

optional gradient checkpointing

📊 Logging & Results
Each run produces:

Code
logs/<dataset>_<optimizer>_train_log_lr*.csv
logs/<dataset>_<optimizer>_val_log_lr*.csv
<dataset>_<optimizer>_result_lr*.txt
results.csv   # aggregated results across all runs
🛠 Dependencies
Install via:

Code
pip install -r requirements.txt
or using pyproject.toml with Poetry/Pip:

Code
pip install .
🧭 Goals of This Pipeline
This project is designed for:

Research on optimizer behavior

Comparing Muon vs AdamW under controlled conditions

Dataset-agnostic finetuning

Reproducible experiments

Modular code organization

Easy extension to new datasets or optimizers

📬 Contact
If you want help extending this pipeline (datasets, distributed Muon, LoRA, RMS diagnostics), feel free to ask.


They're two different training objectives, which is why they need different tokenization logic:

**no_robots → instruction fine-tuning (SFT)**
Each example is a `messages` list (user turn + assistant turn). The task is: given a prompt, learn to produce the assistant's response. That's why the prompt tokens get masked with `-100` — the model isn't penalized for not "predicting" the user's question, it's only trained to predict the response. This is what your LoRA-vs-Muon/AdamW sweeps on Qwen3-4B are built around: teaching the model chat behavior.

**arxiv_astro_abstracts → continued pretraining (CPT) / domain adaptation**
Each example is just a raw abstract, no conversational structure. The task is plain next-token prediction over astro text — same objective as pretraining itself, just on a narrower domain. There's no "prompt" to mask because there's no question being answered; the whole sequence is both input and target, shifted by one token as usual. The goal isn't to teach new behavior, it's to shift the model's internal representations/vocabulary toward astrophysics content (this is essentially what AstroLLaMA's original abstracts-based training did).

So concretely:
| | no_robots | arxiv_astro_abstracts |
|---|---|---|
| Objective | learn to answer, given a prompt | learn to predict astro text generally |
| Label masking | prompt = -100, response = real labels | no masking, everything is a label |
| What it changes in the model | instruction-following/chat behavior | domain knowledge/language distribution |
| Typical scale | thousands of examples is plenty | usually much larger, since it's shifting a distribution not teaching a skill |

If your actual goal is a chat model that's *also* good at astro, the usual recipe is CPT on the abstracts first, then SFT on no_robots (or a similar instruction set) afterward — CPT alone will make the model better at generating astro-sounding text but won't teach it to follow instructions.