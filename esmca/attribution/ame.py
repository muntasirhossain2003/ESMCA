"""Attribution Map Extractor (AME) -- paper Sec 3.2.2.

Uses Integrated Gradients (Sundararajan et al., 2017), attributing the
model's (frozen-backbone-path) output logits back to input token
embeddings with a zero/padding baseline:

    phi_i(x) = (x_i - x'_i) * integral_0^1 dF(x' + a(x - x')) / dx_i da

Per Fig. 1, the AME reads directly off the frozen backbone (no task
adapters engaged) -- `ESMCAModel.forward_frozen`. A per-task attribution
prototype is the mean attribution vector over `n_samples` validation
examples (Sec 3.2.2), and only this compact vector is kept: no raw
training data is retained.
"""
from __future__ import annotations

from typing import Optional

import torch
from captum.attr import LayerIntegratedGradients
from torch.utils.data import DataLoader

from esmca.models.backbone import NUM_CHOICES
from esmca.models.esmca_model import ESMCAModel


class AttributionMapExtractor:
    def __init__(self, model: ESMCAModel, n_steps: int = 32):
        self.model = model
        self.n_steps = n_steps
        self._ig = LayerIntegratedGradients(self._forward_func, model.backbone.embeddings)

    def _forward_func(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model.forward_frozen(input_ids, attention_mask)

    def _to_device(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        device = next(self.model.parameters()).device
        return tuple(t.to(device) for t in tensors)

    def attribute(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns phi(x): one attribution scalar per choice-token, shape [B, num_choices * L]."""
        input_ids, attention_mask = self._to_device(input_ids, attention_mask)
        if target is not None:
            (target,) = self._to_device(target)
        self.model.eval()
        if target is None:
            with torch.no_grad():
                logits = self.model.forward_frozen(input_ids, attention_mask)
                target = logits.argmax(dim=-1)
        pad_id = self.model.tokenizer.pad_token_id or 0
        baseline_ids = torch.full_like(input_ids, pad_id)
        with torch.no_grad():
            attributions = self._ig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=(attention_mask,),
                target=target,
                n_steps=self.n_steps,
            )
        # attributions match input shape [B, num_choices, L]; flatten the
        # choice/seq dims into a single per-example attribution vector.
        return attributions.reshape(input_ids.shape[0], -1).detach()

    def attribute_differentiable(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Grad-times-input attribution that stays part of the autograd graph
        (unlike `attribute`, which uses multi-step IG and detaches its
        output). Used only for the consolidation loss L_attrib (Eq. 4),
        where we need gradients to flow back into the *current* task's LoRA
        adapter and the shared head -- whatever adapter is currently active
        stays active here (this deliberately does not force the frozen-only
        path that `attribute` uses for routing/prototypes)."""
        input_ids, attention_mask, target = self._to_device(input_ids, attention_mask, target)
        with torch.enable_grad():
            embeds = self.model.backbone.embeddings(input_ids)  # [B, num_choices, L, H]
            embeds.requires_grad_(True)
            batch_size = input_ids.shape[0]
            flat_embeds = embeds.reshape(batch_size * NUM_CHOICES, *embeds.shape[2:])
            flat_mask = attention_mask.reshape(batch_size * NUM_CHOICES, -1)
            cls_hidden = self.model.backbone(
                inputs_embeds=flat_embeds, attention_mask=flat_mask
            ).last_hidden_state[:, 0]
            cls_hidden = cls_hidden.to(self.model.head.dense.weight.dtype)
            scores = self.model.head(cls_hidden).view(batch_size, NUM_CHOICES)  # [B, num_choices]
            selected = scores.gather(1, target.unsqueeze(1)).squeeze(1).sum()
            (grads,) = torch.autograd.grad(selected, embeds, create_graph=True)
        # Match the [B, num_choices * L] vector shape returned by `attribute`.
        return (embeds * grads).sum(dim=-1).reshape(batch_size, -1)

    def compute_prototype(self, loader: DataLoader, n_samples: int = 256) -> torch.Tensor:
        """Average attribution over up to `n_samples` examples -> Phi_t."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        collected = []
        seen = 0
        for batch in loader:
            if seen >= n_samples:
                break
            # Process one sample at a time to avoid OOM during IG
            for i in range(batch["input_ids"].shape[0]):
                if seen >= n_samples:
                    break
                single_ids = batch["input_ids"][i:i+1]
                single_mask = batch["attention_mask"][i:i+1]
                single_labels = batch["labels"][i:i+1]
                attrs = self.attribute(single_ids, single_mask, target=single_labels)
                collected.append(attrs)
                seen += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        all_attrs = torch.cat(collected, dim=0)[:n_samples]
        return all_attrs.mean(dim=0)
