"""Definitions of log entries stored to ObjectLogManager."""

import dataclasses
import numpy as np
import torch

TRAINING_BATCH_LOG_ENTRY = "training_batch"
STATE_NORMALIZED_LOG_ENTRY = "state_normalized"


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

@dataclasses.dataclass
class NormalizedLogEntry:
    """A pairing of a normalized object with its unique id.    

    The unique id will be used to identify and join this object with
    other log entries. The type of object for a particular instance of
    a log file can be determined by either checking the initialization
    of the correponding LoggerAndNormalizer and dynamically checking
    the type of object at read time.

    """

    object: Any
    id: int


    
