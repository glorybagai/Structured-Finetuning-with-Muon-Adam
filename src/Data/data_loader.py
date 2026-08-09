"""
Chat-format + plain-text + QA-pair dataset loading with prompt-masked tokenization.

Same formatting logic as the original no_robots script, parameterized so a
new dataset is one entry in DATASET_CONFIGS. Supports three formats:

  - "chat":       instruction-tuning datasets with a `messages` field
                  (e.g. no_robots, ultrachat_200k). Prompt tokens masked
                  with -100, model trained only on the response.
  - "plain_text": continued-pretraining / domain-adaptation datasets that
                  are just raw text (e.g. arxiv abstracts). No prompt to
                  mask, the whole sequence is the label.
  - "qa_pairs":   instruction-tuning datasets with separate prompt/response
                  columns instead of a chat-template-ready `messages` list
                  (e.g. MetaMathQA's `query`/`response`). Same masking
                  idea as "chat", just without a chat template — the raw
                  prompt string is masked, the raw response string is the
                  training target.
"""
from datasets import load_dataset

# Add new datasets here. Each entry needs the HF hub id, the split names it
# uses (some datasets call it "train_sft"/"test_sft" instead of "train"/
# "test"; set test_split=None if the dataset only ships one split and a
# test set should be carved out of train instead), and a "format" telling
# build_datasets which tokenization path to use.
#   - "plain_text" entries need "text_field"
#   - "qa_pairs" entries need "prompt_field" and "response_field"
DATASET_CONFIGS = {
    "no_robots": dict(
        hf_name="HuggingFaceH4/no_robots",
        train_split="train",
        test_split="test",
        format="chat",
    ),
    "arxiv_astro_abstracts": dict(
        hf_name="UniverseTBD/arxiv-astro-abstracts-all",
        train_split="train",
        test_split="test",  # confirmed: real splits are train / test / valid
        format="plain_text",
        text_field="text",
        gated=True,
    ),
    "ultrachat_200k": dict(
        hf_name="HuggingFaceH4/ultrachat_200k",
        train_split="train_sft",
        test_split="test_sft",
        format="chat",
    ),
    "metamathqa": dict(
        hf_name="meta-math/MetaMathQA",
        train_split="train",
        test_split=None,  # only one split on the hub — carve test out of train
        format="qa_pairs",
        prompt_field="query",
        response_field="response",
    ),
}


def format_chat_example(example, tokenizer, max_length=1024):
    """Applies chat template, masks the prompt tokens with -100, keeps the response as labels."""
    messages = example["messages"]
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(
        full_text, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]
    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        print("WARNING: full_ids does not start with prompt_ids — tokenization may not be prefix-stable")
    labels = full_ids.copy()
    for i in range(len(prompt_ids)):
        labels[i] = -100
    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def format_plain_text_example(example, tokenizer, max_length=1024, text_field="text"):
    """Tokenizes raw text for causal LM training. No prompt to mask — whole
    sequence is the label, same as standard continued-pretraining setups."""
    text = example[text_field]
    # EOS at the end so the model learns where one abstract ends and the
    # next document would begin, matching standard CPT/packing conventions.
    if tokenizer.eos_token:
        text = text + tokenizer.eos_token
    full_ids = tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]
    return {
        "input_ids": full_ids,
        "labels": full_ids.copy(),
        "attention_mask": [1] * len(full_ids),
    }


def format_qa_pairs_example(example, tokenizer, max_length=1024, prompt_field="query", response_field="response"):
    """Masks the prompt, trains on the response — same idea as
    format_chat_example, but for datasets with plain prompt/response
    columns instead of a chat-template-ready `messages` list."""
    prompt_text = example[prompt_field]
    response_text = example[response_field]

    # Separator mirrors how a chat template would delimit turns; keeps
    # prompt_ids a clean prefix of full_ids so the masking boundary lines up.
    full_text = prompt_text + "\n" + response_text
    if tokenizer.eos_token:
        full_text = full_text + tokenizer.eos_token

    full_ids = tokenizer(
        full_text, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]
    prompt_ids = tokenizer(
        prompt_text + "\n", add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]

    if full_ids[: len(prompt_ids)] != prompt_ids:
        print("WARNING: full_ids does not start with prompt_ids — tokenization may not be prefix-stable")

    labels = full_ids.copy()
    for i in range(len(prompt_ids)):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def format_example(example, tokenizer, dataset_format, max_length=1024, cfg=None):
    """Dispatches to the right formatter based on dataset_format. cfg is the
    dataset's DATASET_CONFIGS entry, used to pull format-specific field names."""
    cfg = cfg or {}
    if dataset_format == "chat":
        return format_chat_example(example, tokenizer, max_length)
    elif dataset_format == "plain_text":
        return format_plain_text_example(example, tokenizer, max_length, cfg.get("text_field", "text"))
    elif dataset_format == "qa_pairs":
        return format_qa_pairs_example(
            example, tokenizer, max_length,
            cfg.get("prompt_field", "query"), cfg.get("response_field", "response"),
        )
    else:
        raise ValueError(f"Unknown format '{dataset_format}'. Use 'chat', 'plain_text', or 'qa_pairs'.")


def has_valid_labels(example):
    """True if at least one label token is unmasked (i.e. there's something
    to actually compute loss on). False means the whole example got masked
    away — this happens when a "chat"/"qa_pairs" example's prompt portion
    alone reaches max_length, leaving zero response tokens. Dropping these
    prevents fully-masked micro-batches, which produce a NaN loss (mean over
    zero unmasked tokens) that silently poisons the logged train_loss."""
    return any(label != -100 for label in example["labels"])


def build_datasets(dataset_key, tokenizer, n_train=6400, n_test=100, max_length=1024):
    """Looks up dataset_key in DATASET_CONFIGS, loads + formats train/test splits."""
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Add it to DATASET_CONFIGS in data_utils.py.")
    cfg = DATASET_CONFIGS[dataset_key]

    if cfg.get("gated"):
        # load_dataset will raise a clear error itself if not authenticated,
        # but this makes the requirement explicit rather than a surprise
        # 401 deep in a training run. Needs `huggingface-cli login` first,
        # or pass token=... through to load_dataset below.
        print(f"NOTE: '{dataset_key}' is a gated dataset — make sure you've "
              f"accepted its terms on the HF hub and are logged in (or pass a token).")

    dataset = load_dataset(cfg["hf_name"])
    dataset_format = cfg["format"]

    if cfg["test_split"] is None:
        # No separate test split on the hub (e.g. MetaMathQA) — carve one
        # out of train instead. Fixed seed so the split is reproducible
        # across runs/optimizers/LRs in the sweep.
        full_train = dataset[cfg["train_split"]]
        n_test_capped = min(n_test, len(full_train) - 1)
        split = full_train.train_test_split(test_size=n_test_capped, seed=42)
        train_split, test_split = split["train"], split["test"]
    else:
        train_split = dataset[cfg["train_split"]]
        test_split = dataset[cfg["test_split"]]

    train_ds = train_split.select(range(min(n_train, len(train_split)))).map(
        lambda ex: format_example(ex, tokenizer, dataset_format, max_length, cfg),
        remove_columns=train_split.column_names,
    )
    test_ds = test_split.select(range(min(n_test, len(test_split)))).map(
        lambda ex: format_example(ex, tokenizer, dataset_format, max_length, cfg),
        remove_columns=test_split.column_names,
    )

    # Drop fully-masked examples (see has_valid_labels docstring). Only
    # "chat"/"qa_pairs" formats can produce these — "plain_text" never masks
    # anything — but the filter is cheap and safe to run on every format.
    n_train_before, n_test_before = len(train_ds), len(test_ds)
    train_ds = train_ds.filter(has_valid_labels)
    test_ds = test_ds.filter(has_valid_labels)
    n_train_dropped = n_train_before - len(train_ds)
    n_test_dropped = n_test_before - len(test_ds)
    if n_train_dropped or n_test_dropped:
        print(f"'{dataset_key}': dropped {n_train_dropped}/{n_train_before} train and "
              f"{n_test_dropped}/{n_test_before} test examples with no unmasked "
              f"label tokens (prompt alone reached max_length={max_length}).")

    return train_ds, test_ds