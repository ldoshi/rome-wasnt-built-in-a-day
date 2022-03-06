"""Definitions of log entries stored to ObjectLogManager."""

import dataclasses
import numpy as np
import torch
from typing import Any, List

TRAINING_BATCH_LOG_ENTRY = "training_batch"
STATE_NORMALIZED_LOG_ENTRY = "state_normalized"

TRAINING_HISTORY_VISIT_LOG_ENTRY = "training_history_visit"

@dataclasses.dataclass
class TrainingHistoryVisitLogEntry:
    """An entry representing a single state visit."""
    batch_idx: int
    state_id: int

@dataclasses.dataclass
class TrainingBatchLogEntry:
    """The contents of a training batch from a training step.

    A training batch is extracted from the replay buffer at each
    training step. The batch_idx identifies which training step this
    TrainingBatchLogEntry corresponds to.

    """

    batch_idx: int
    indices: torch.Tensor
    state_ids: List[int]
    actions: torch.Tensor
    next_state_ids: List[int]
    rewards: torch.Tensor
    successes: torch.Tensor
    weights: torch.Tensor
    loss: torch.Tensor


@dataclasses.dataclass
class NormalizedLogEntry:
    """A pairing of a normalized object with its unique id.

    The unique id will be used to identify and join this object with
    other log entries. The type of object for a particular instance of
    a log file can be determined by either checking the initialization
    of the correponding LoggerAndNormalizer and dynamically checking
    the type of object at read time.

    """

    id: int
    object: Any

