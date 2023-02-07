"""Helper functions for tests."""

import os
import pathlib
import shutil


from bridger.logging_utils import object_logging
from bridger import builder_trainer

from typing import List, Optional

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

TMP_DIR = "tmp/nested_tmp"
_OBJECT_LOGGING_DIR = "tmp_object_logging_dir"


def object_logging_dir():
    return os.path.join(TMP_DIR, _OBJECT_LOGGING_DIR)


def create_temp_dir():
    path = pathlib.Path(TMP_DIR)
    path.mkdir(parents=True, exist_ok=True)


def delete_temp_dir():
    path = pathlib.Path(TMP_DIR)
    shutil.rmtree(path.parts[0], ignore_errors=True)


def get_model(
    object_log_manager: object_logging.ObjectLogManager,
    debug: bool = False,
    debug_action_inversion_checker: bool = False,
    debug_td_error: bool = False,
    env_width=3,
    max_episode_length=1,
    batch_size=5,
    initial_memories_count=1000,
    q=builder_trainer.Q_CNN,
    tabular_q_initialization_brick_count=3,
) -> builder_trainer.BridgeBuilderModel:
    return builder_trainer.BridgeBuilderModel(
        object_log_manager,
        env_width=env_width,
        env_force_standard_config=True,
        seed=12345,
        max_episode_length=max_episode_length,
        val_batch_size=1,
        batch_size=batch_size,
        object_logging_dir=_OBJECT_LOGGING_DIR,
        initial_memories_count=initial_memories_count,
        debug=debug,
        debug_action_inversion_checker=debug_action_inversion_checker,
        debug_td_error=debug_td_error,
        q=q,
        tabular_q_initialization_brick_count=tabular_q_initialization_brick_count,
    )


def get_trainer(
    max_steps: int = 1, callbacks: Optional[List[Callback]] = None
) -> Trainer:
    return Trainer(
        val_check_interval=1,
        # The validation batch size can be adjusted via a config, but
        # we only need a single batch.
        limit_val_batches=1,
        max_steps=max_steps,
        callbacks=callbacks,
    )
