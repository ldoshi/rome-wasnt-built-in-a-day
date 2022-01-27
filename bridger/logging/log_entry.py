"""Definitions of log entries stored to ObjectLogManager."""

import dataclasses

TRAINING_BATCH_LOG_ENTRY = "training_batch"

@dataclasses.dataclass
class TrainingBatch():
    batch_idx: int
    indices: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    successes: torch.Tensor
    weights: torch.Tensor
    loss: torch.Tensor
