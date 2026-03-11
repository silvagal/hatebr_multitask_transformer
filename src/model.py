from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class ModelOutputs:
    bin_logits: Optional[torch.Tensor] = None
    level_logits: Optional[torch.Tensor] = None
    target_logits: Optional[torch.Tensor] = None


def _cls_pooler(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0, :]


class MultiHeadModel(nn.Module):
    def __init__(self, model_name: str, hidden_size: int = 768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if hidden_size is None:
            hidden_size = self.encoder.config.hidden_size
        self.head_bin = nn.Linear(hidden_size, 2)
        self.head_level = nn.Linear(hidden_size, 4)
        self.head_target = nn.Linear(hidden_size, 9)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutputs:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _cls_pooler(outputs.last_hidden_state)
        return ModelOutputs(
            bin_logits=self.head_bin(pooled),
            level_logits=self.head_level(pooled),
            target_logits=self.head_target(pooled),
        )


class SingleTaskModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _cls_pooler(outputs.last_hidden_state)
        return self.classifier(pooled)


def get_single_task_output(task: str, logits: torch.Tensor) -> ModelOutputs:
    if task == "bin":
        return ModelOutputs(bin_logits=logits)
    if task == "level":
        return ModelOutputs(level_logits=logits)
    if task == "target":
        return ModelOutputs(target_logits=logits)
    raise ValueError(f"Unknown task: {task}")


def get_single_task_num_labels(task: str) -> int:
    if task == "bin":
        return 2
    if task == "level":
        return 4
    if task == "target":
        return 9
    raise ValueError(f"Unknown task: {task}")


def build_losses() -> Tuple[nn.Module, nn.Module, nn.Module]:
    loss_bin = nn.CrossEntropyLoss()
    loss_level = nn.CrossEntropyLoss()
    loss_target = nn.BCEWithLogitsLoss()
    return loss_bin, loss_level, loss_target
