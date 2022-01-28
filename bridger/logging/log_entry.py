"""Definitions of log entries stored to ObjectLogManager."""

import dataclasses
import torch

TRAINING_BATCH_LOG_ENTRY = "training_batch"


@dataclasses.dataclass
class TrainingBatchLogEntry:
    """The contents of a training batch from a training step.

    A training batch is extracted from the replay buffer at each
    training step. The batch_idx identifies which training step this
    TrainingBatchLogEntry corresponds to.

    """
    batch_idx: int
    indices: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    successes: torch.Tensor
    weights: torch.Tensor
    loss: torch.Tensor
