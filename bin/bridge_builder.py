#!/usr/bin/env python

# Current usage notes:

# If this is called with command line arg "interactive-mode" set to True, then
# Trainer will intermittently enter an IPython shell, allowing you to inspect
# model state at your leisure. This shell can be exited by calling one of three
# model commands:
# 1. return_to_training will disable interactive mode and complete the requested
#    training without additional IPython breakpoints.
# 2. follow_policy will run the minimum of a requested number of steps or through
#    the end of a requested number of episodes before returning to the IPython shell
# 3. take_action will take the requested action, potentially multiple times, before
#    returning to the IPython shell

import datetime
import os

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from bridger import builder
from bridger import builder_trainer
from bridger.callbacks import DemoCallback
from pathlib import Path
from bridger.logging_utils import object_logging


def _get_full_experiment_name(experiment_name: str) -> str:
    """Returns the full experiment name.

    The user-provided experiment name is appended to a prefix of the
    current datetime.

    Args:
      experiment_name: The user-provided experiment name.

    Returns:
      The full experiment name to use as a unique label for this run.

    """
    return (
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"{'_' if experiment_name else ''}{experiment_name}"
    )


_DEMO_CALLBACK_FREQUENCY = 100
MAX_DEMO_EPISODE_LENGTH = 50


def run():
    # TODO(arvind): split the args into those relevant for the LightningModule
    #               and those relevant for the Trainer/Callbacks
    parser = builder_trainer.get_hyperparam_parser()
    hparams = parser.parse_args()

    full_experiment_name = _get_full_experiment_name(hparams.experiment_name)
    with object_logging.ObjectLogManager(
        hparams.object_logging_base_dir, full_experiment_name, create_experiment_dir=hparams.debug
    ) as object_log_manager:
        if hparams.load_checkpoint_path:
            # TODO(arvind): Decide on and implement the functionality we'd like to
            #               support in terms of loading weights from a checkpoint
            #               while using newly specified config hyperparameters
            model = builder_trainer.BridgeBuilderModel.load_from_checkpoint(
                hparams.load_checkpoint_path, object_log_manager, hparams=hparams
            )
        else:
            model = builder_trainer.BridgeBuilderModel(
                object_log_manager, hparams=hparams
            )

        callbacks = [
            # Only retains checkpoint with minimum monitored quantity seen so far.
            # By default, just saves the (temporally) last checkpoint. This should
            # eventually be done based on monitoring a well defined validation
            # metric that doesn't depend on the most recent batch of memories
            ModelCheckpoint(
                monitor=None,  # Should show a quantity, e.g. "train_loss"
                every_n_train_steps=hparams.checkpoint_interval,
            ),
            EarlyStopping(
                monitor="val_reward",
                patience=hparams.early_stopping_patience,
                mode="max",
                strict=True,
                check_on_train_epoch_end=False,
            ),
        ]
        if hparams.debug:
            callbacks += [
                DemoCallback(
                    steps_per_update=_DEMO_CALLBACK_FREQUENCY,
                    max_episode_length=hparams.max_episode_length,
                ),
            ]

        trainer = Trainer(
            gradient_clip_val=hparams.gradient_clip_val,
            check_val_every_n_epoch=hparams.val_check_interval,
            # The validation batch size can be adjusted via a config, but
            # we only need a single batch.
            limit_val_batches=1,
            logger=TensorBoardLogger(
                save_dir=hparams.checkpoint_model_dir,
                name=full_experiment_name,
                version="",
            ),
            limit_train_batches=1,
            max_epochs=hparams.max_training_batches,
            reload_dataloaders_every_n_epochs=1,
            callbacks=callbacks,
        )

        trainer.fit(model)

        # TODO(lyric): Temporary code until we decide how to track and
        # store evaluation results.
        build_count = 1
        # TODO(lyric): Choose and persist a set seed value used for
        # evaluations to ensure they are repeatable and comparable.
        seed = 123456
        evaluation_env = builder_trainer.make_env(
            name=hparams.env_name,
            width=hparams.env_width,
            force_standard_config=hparams.env_force_standard_config,
            seed=seed,
        )

        demo_episode_length = MAX_DEMO_EPISODE_LENGTH
        # Longer episodes than the tabular q table size will fail.
        if hparams.q == "tabular":
            demo_episode_length = min(
                hparams.tabular_q_initialization_brick_count, demo_episode_length
            )

        build_evaluator = builder.BuildEvaluator(
            env=evaluation_env,
            policy=model.trained_policy,
            build_count=build_count,
            episode_length=demo_episode_length,
        )
        build_evaluator.print_report()


if __name__ == "__main__":
    run()
