"""
Muon optimizer — uses torch.optim.Muon directly. This landed natively in
PyTorch 2.9 (Oct 2025): https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
No custom Newton-Schulz/optimizer code here, just the import and the usual
param-group setup around it.

Requires torch >= 2.9. Check with `python -c "import torch; print(torch.__version__)"`
on seal-node1 — if it's older, `pip install --upgrade torch` (or update your venv)
before this will import.

torch.optim.Muon only accepts strictly 2D parameters (p.ndim == 2) — that's a
hidden requirement of the native implementation, not just a design choice, so
convolutional or other >2D weights won't work with it as-is. Embeddings, the
LM head, norm weights, and biases still need a separate optimizer, so
build_muon_optimizer() splits params and routes the non-Muon group to AdamW,
returning both wrapped in something with a single .step()/.zero_grad(), so
train.py doesn't need to know two optimizers are involved.
"""

import torch

try:
    from torch.optim import Muon
except ImportError as e:
    raise ImportError(
        "torch.optim.Muon is not available — it requires torch >= 2.9. "
        "Check your version with `python -c \"import torch; print(torch.__version__)\"` "
        "and upgrade if needed."
    ) from e


class CombinedOptimizer:
    """Wraps several torch optimizers behind one .step()/.zero_grad() interface."""

    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups


def build_muon_optimizer(model, lr, weight_decay=0.1, momentum=0.95, nesterov=True,
                          ns_steps=5, adjust_lr_fn=None,
                          adamw_lr=3.5e-5, adamw_betas=(0.9, 0.999), adamw_weight_decay=0.01):
    """
    Splits model.named_parameters() into:
      - muon_params:  strictly 2D, excluding embedding / lm_head weights
      - adamw_params: everything else (embeddings, lm_head, norms, biases)
    and returns a CombinedOptimizer running torch.optim.Muon on the first
    group and torch.optim.AdamW on the second.

    adjust_lr_fn: None/"original" (Keller Jordan's default LR scaling) or
    "match_rms_adamw" (the RMS-matching variant from "Muon is Scalable for
    LLM Training", which lets Muon reuse an AdamW-tuned LR directly — worth
    trying given your Muon-vs-AdamW LR grid mismatch on the no_robots runs).

    adamw_lr defaults to the same lr if not given, but you'll usually want to
    decouple it since Muon and AdamW operate at different LR scales.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name and "lm_head" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    muon_opt = Muon(
        muon_params, lr=lr, weight_decay=weight_decay, momentum=momentum,
        nesterov=nesterov, ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn,
    )
    adamw_opt = torch.optim.AdamW(
        adamw_params, lr=adamw_lr if adamw_lr is not None else lr,
        betas=adamw_betas, weight_decay=adamw_weight_decay,
    )

    return CombinedOptimizer([muon_opt, adamw_opt])