"""
Model + tokenizer loading, shared across all runs regardless of dataset or optimizer.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_name, gradient_checkpointing: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,  # instead of torch_dtype=...
    )

    # needed for training + gradient checkpointing
    model.config.use_cache = False

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # often recommended so gradients flow correctly
        model.enable_input_require_grads()

    model.to(device)
    print("Model loaded.")

    return model, tokenizer, device
