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
class StateNormalizedLogEntry:
    """A pairing of a state with its unique id.

    Normalized log entries are expected to always have two fields: 
      object: <type being logged in a normalized way>
      id:int
    
    The id is initialized to -1 and will be set by the LoggerAndNormalizer.

    """

    object: np.ndarry
    id: int = -1
    
