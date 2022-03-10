"""Definitions of log entries stored to ObjectLogManager."""

import dataclasses
import numpy as np
import torch
from typing import Any, List

TRAINING_BATCH_LOG_ENTRY = "training_batch"
STATE_NORMALIZED_LOG_ENTRY = "state_normalized"

TRAINING_HISTORY_VISIT_LOG_ENTRY = "training_history_visit"
TRAINING_HISTORY_TD_ERROR_LOG_ENTRY = "training_history_td_error"
TRAINING_HISTORY_Q_VALUE_LOG_ENTRY = "training_history_q_value"

@dataclasses.dataclass
class TrainingHistoryTDErrorLogEntry:
    """The TD Error for a (state, action) pair from a training step.

    A training batch is extracted from the replay buffer at each
    training step. The batch_idx identifies which training step this
    TrainingHistoryTDErrorLogEntry corresponds to.

    """

    batch_idx: int
    state_id: int
    action: int
    td_error: float


@dataclasses.dataclass
class TrainingHistoryQValueLogEntry:
    """The Q values for the main and target networks.

    Each entry represents the Q value and Q target value for the
    (state, action) pair calculated during the batch_idx-th training
    step (before any network updates in that same batch_idx-th step).

    """

    batch_idx: int
    state_id: int
    action: int
    q_value: float
    q_target_value: float


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


@dataclasses.dataclass
class OccurrenceLogEntry:
    """An entry noting the occurrence of an object.

    The type of object for a particular instance of a log file can be
    determined by either checking the initialization of the
    correponding OccurrenceLogger and dynamically checking the type of
    object at read time.

    """

    batch_idx: int
    object: Any
