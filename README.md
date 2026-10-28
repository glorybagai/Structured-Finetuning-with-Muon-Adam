# Modular Fine-Tuning Pipeline for Qwen: AdamW vs. Muon

A modular, experiment-driven pipeline for full-parameter fine-tuning and controlled optimizer comparisons on Qwen language models.

**Focus:** Efficient LLM Training · Optimizer Research · Full Fine-Tuning · Continued Pretraining · Supervised Fine-Tuning

## Overview

This repository provides a modular pipeline for training Qwen language models using AdamW and Muon. It is designed to support controlled experiments across multiple datasets and training objectives without requiring changes to the core training infrastructure.

The pipeline separates reusable components, including model loading, data processing, optimization, training, and evaluation, from experiment configuration and orchestration.

Key features include:

- **Full-parameter fine-tuning:** Updates model parameters without LoRA or other parameter-efficient fine-tuning adapters.
- **Multiple training objectives:** Supports supervised fine-tuning (SFT) and continued pretraining (CPT) through objective-specific data processing.
- **Multiple optimizers:** Compares AdamW with a first-principles Muon implementation.
- **Dataset-agnostic design:** Adds datasets through configuration rather than rewriting the training pipeline.
- **Reproducible experiments:** Uses a unified command-line interface, configurable hyperparameters, and per-run logging.
- **Extensible architecture:** Keeps reusable training logic separate from experiment-specific code.

The primary research objective is to investigate how optimizer choice and training configuration affect language-model fine-tuning performance.

## Table of Contents

- [Architecture](#architecture)
- [Supported Training Objectives](#supported-training-objectives)
- [Optimizers](#optimizers)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running Experiments](#running-experiments)
- [Adding a Dataset](#adding-a-dataset)
- [Training and Evaluation](#training-and-evaluation)
- [Logging and Results](#logging-and-results)
- [Dependencies](#dependencies)
- [Research Directions](#research-directions)

## Architecture

The pipeline follows a modular design with three main layers:

1. **Core components (`src/`)** implement reusable functionality for model loading, data processing, optimization, training, and evaluation.
2. **Experiment orchestration (`experiments/`)** exposes a unified CLI for configuring and launching runs.
3. **Results and logs** record training progress, evaluation metrics, and aggregated experiment outcomes.

This separation makes it possible to compare optimizers and datasets under a consistent training and evaluation framework.

## Supported Training Objectives

The pipeline supports two distinct language-model training objectives. Each requires a different data representation, target construction strategy, and loss-masking policy.

### 1. Supervised Fine-Tuning (SFT)

**Example dataset:** [`HuggingFaceH4/no_robots`](https://huggingface.co/datasets/HuggingFaceH4/no_robots)

SFT trains a model to generate appropriate responses to user instructions. Each example contains conversational messages, typically including a user prompt and an assistant response.

The conversation is formatted using the model's chat template and tokenized into a single sequence. Prompt tokens are masked in the labels so that the loss is calculated on the intended assistant-response tokens rather than the user prompt.

Conceptually:

```text
Input:
User: Explain gradient descent.
Assistant: Gradient descent minimizes a loss function by ...

Labels:
User prompt      -> -100 (ignored by the loss)
Assistant reply  -> token IDs (included in the loss)
```

The `-100` label is the standard ignore index used by PyTorch's cross-entropy loss.

**Objective:** Improve instruction-following and response generation.

**Typical use case:** Comparing AdamW and Muon on instruction-tuning workloads, including optimizer and learning-rate sweeps.

### 2. Continued Pretraining (CPT)

**Example dataset:** [`arxiv_astro_abstracts`](https://huggingface.co/datasets) (configure the actual Hugging Face dataset identifier in `dataset_loader.py`).

CPT adapts a pretrained language model to a particular domain using raw text rather than conversational examples.

For example, an astrophysics abstract is tokenized as a continuous text sequence. The model is trained using the standard causal language-modeling objective, predicting each next token from the preceding context.

```text
Input:
We investigate the formation of galaxies ...

Training target:
Predict the next token at each valid sequence position.
```

Unlike the SFT pipeline, CPT does not mask a user prompt because the entire text sequence supplies the language-modeling targets, subject to standard padding and causal-shift handling.

**Objective:** Adapt the model's representations and token-prediction distribution to domain-specific text.

**Typical use case:** Investigating optimizer behavior during domain adaptation, such as continued training on astrophysics literature.

### SFT vs. CPT

| Property | SFT | CPT |
|---|---|---|
| Example dataset | `no_robots` | `arxiv_astro_abstracts` |
| Data format | Conversational messages | Raw text |
| Training objective | Instruction-following | Next-token prediction |
| Prompt masking | Mask prompt tokens with `-100` | No conversational prompt masking |
| Loss targets | Assistant-response tokens | All valid text tokens |
| Primary purpose | Improve response behavior | Adapt to a domain's text distribution |

The objective-specific preprocessing is handled by the data pipeline, while the training loop and optimizer interface remain shared.

**Combining the objectives:** A model intended to both acquire domain-specific knowledge and follow instructions can be trained using CPT followed by SFT. This is a possible extension of the pipeline, not a claim that every experiment in this repository uses both stages.

## Optimizers

The repository supports two optimization methods through a common training interface.

### AdamW

AdamW serves as the baseline optimizer. It uses adaptive first- and second-moment estimates with decoupled weight decay.

Example configuration:

```bash
python experiments/run.py \
    --dataset no_robots \
    --optimizer adam \
    --lr 2.5e-5
```

### Muon

The repository includes a first-principles Muon implementation based on Keller Jordan's reference design.

Its main components include:

- Momentum-based updates.
- Quintic Newton–Schulz orthogonalization.
- Aspect-ratio scaling.
- Parameter-specific optimizer routing.

Muon is applied to eligible hidden-layer matrix parameters, while AdamW handles parameter groups such as embeddings, the language-model head, normalization parameters, and biases.

This hybrid arrangement is implemented through a `CombinedOptimizer`, allowing the training loop to use a common optimizer interface.

**Implementation note:** This is a custom research implementation, not PyTorch's built-in `torch.optim.Muon`.

Implementation: `src/optimizers/muon.py`

## Repository Structure

```text
full_finetuning/
├── src/
│   ├── model/
│   │   ├── qwen_loader.py
│   │   └── ...
│   ├── training/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── ...
│   ├── optimizers/
│   │   ├── muon.py
│   │   └── ...
│   ├── data/
│   │   ├── dataset_loader.py
│   │   └── ...
│   └── utils/
├── experiments/
│   └── run.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

The tree above illustrates the intended organization. Update individual filenames to match the actual repository.

- `src/model/`: Model and tokenizer loading, including training configuration.
- `src/training/`: Training loop, evaluation loop, and loss handling.
- `src/optimizers/`: AdamW and Muon implementations.
- `src/data/`: Dataset loading, preprocessing, chat formatting, and objective-specific labels.
- `src/utils/`: Logging, metrics, and checkpoint utilities.
- `experiments/run.py`: Unified experiment entry point.

Local virtual environments, datasets, checkpoints, and generated logs should generally remain outside version control.

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd full_finetuning
```

### 2. Install dependencies

Using the project's dependency file:

```bash
pip install -r requirements.txt
```

Alternatively, if the project is configured for installation through `pyproject.toml`:

```bash
pip install .
```

Use the dependency-management approach supported by the current project configuration.

### 3. Configure the environment

Before running experiments, verify:

- Python and library versions.
- Access to the selected pretrained Qwen checkpoint.
- Dataset availability and access permissions.
- GPU memory and precision requirements.
- Output directories and experiment configuration.

The pipeline's actual supported hardware and software requirements should be documented once verified.

## Running Experiments

All experiments are launched through `experiments/run.py`.

### AdamW baseline

```bash
python experiments/run.py \
    --dataset no_robots \
    --optimizer adam \
    --lr 2.5e-5
```

### Muon

```bash
python experiments/run.py \
    --dataset no_robots \
    --optimizer muon \
    --lr 0.02
```

The learning rates shown are example configurations, not universal recommendations. AdamW and Muon may require different learning-rate ranges.

### Common configuration options

The following flags are available in the current experiment interface, subject to the CLI implementation:

| Argument | Example | Purpose |
|---|---:|---|
| `--batch_size` | `4` | Per-device batch size |
| `--gradient_accumulation_steps` | `8` | Number of microbatches accumulated before an optimizer update |
| `--eval_every_n_steps` | `25` | Evaluation frequency |
| `--n_train` | `6400` | Number of training examples selected |
| `--n_test` | `100` | Number of evaluation examples selected |
| `--max_length` | `1024` | Maximum sequence length |
| `--momentum` | `0.95` | Muon momentum coefficient |
| `--weight_decay` | `0.01` | Weight-decay coefficient |
| `--adamw_lr` | Configurable | AdamW learning rate for parameter groups handled by AdamW under Muon |

Confirm the exact flag names and defaults using `python experiments/run.py --help`.

## Adding a Dataset

Dataset definitions are maintained in:

`src/data/dataset_loader.py`

To add a dataset, register its Hugging Face identifier and the appropriate split names in `DATASET_CONFIGS`.

For example:

```python
DATASET_CONFIGS = {
    "no_robots": {
        "hf_name": "HuggingFaceH4/no_robots",
        "train_split": "train",
        "test_split": "test",
    },
    "my_new_dataset": {
        "hf_name": "myorg/mydataset",
        "train_split": "train_sft",
        "test_split": "test_sft",
    },
}
```

The actual configuration must match the dataset's published identifier, available splits, and schema.

The data-processing pipeline must also select the correct training objective:

- **SFT:** Parse conversational examples, apply the chat template, and mask non-target tokens.
- **CPT:** Tokenize raw text and construct causal language-modeling labels.

Once the dataset is registered and its schema is supported, launch it through the same CLI:

```bash
python experiments/run.py \
    --dataset my_new_dataset \
    --optimizer adam \
    --lr 2e-5
```

A new dataset may require a preprocessing adapter if its schema differs from existing datasets, but the reusable training loop should not need to be rewritten.

## Training and Evaluation

### Training

The generic training loop is implemented in `src/training/trainer.py`.

Features include:

- Gradient accumulation.
- Periodic evaluation.
- Per-run CSV logging.
- Aggregated results.
- A shared execution path for supported optimizers.

### Evaluation

Evaluation is implemented in `src/training/evaluator.py`.

The primary metric is negative log-likelihood (NLL), reported through average evaluation loss.

For SFT, evaluation should respect the assistant-response masking policy so that prompt tokens do not contribute to the loss. For CPT, evaluation should measure next-token prediction over valid text tokens, excluding padding and other ignored positions.

Lower NLL indicates better next-token predictive performance on the evaluated data. Comparisons are meaningful when the evaluation data, tokenization, masking, and aggregation procedures are consistent.

## Logging and Results

Each experiment records training and evaluation outputs. The current pipeline is configured to produce files such as:

```text
logs/
├── <run>_train_log_lr<value>.csv
├── val_log_lr<value>.csv
└── <run>_result_lr<value>.txt

results.csv
```

- **Training logs:** Record training metrics over optimization steps.
- **Validation logs:** Record periodic evaluation metrics.
- **Run summaries:** Store final per-run results.
- **Aggregated results:** Collect experiment outcomes across runs.

Actual filenames and naming conventions may vary with the current logging implementation.

## Research Directions

This pipeline is intended to support research on:

- Optimizer comparisons under controlled training conditions.
- Muon versus AdamW across training objectives.
- Learning-rate and optimizer hyperparameter sensitivity.
- Full-parameter fine-tuning of language models.
- Domain adaptation through continued pretraining.
- The interaction between data processing, optimization, and evaluation.
- Extensions to additional optimizers, datasets, and training configurations.

## Acknowledgements

The Muon implementation is based on the reference design by Keller Jordan. Qwen models and datasets are provided by their respective contributors on Hugging Face.


