"""
Plain torch.optim.AdamW, wrapped so run.py can build either optimizer
through the same call signature: build_optimizer(model, lr, weight_decay).
"""

import torch


def build_adam_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.999)):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)