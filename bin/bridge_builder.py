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

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

from bridger import builder
from bridger import builder_trainer
from bridger.callbacks import DemoCallback
from pathlib import Path
from bridger.logging_utils import object_logging

_DEMO_CALLBACK_FREQUENCY = 100
MAX_DEMO_EPISODE_LENGTH = 50


def run():
    # TODO(arvind): split the args into those relevant for the LightningModule
    #               and those relevant for the Trainer/Callbacks
    parser = builder_trainer.get_hyperparam_parser()
    hparams = parser.parse_args()
    with object_logging.ObjectLogManager(
        hparams.object_logging_dir
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
            val_check_interval=hparams.val_check_interval,
            # The validation batch size can be adjusted via a config, but
            # we only need a single batch.
            limit_val_batches=1,
            default_root_dir=hparams.checkpoint_model_dir,
            max_steps=hparams.max_training_batches,
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
        build_evaluator = builder.BuildEvaluator(
            env=evaluation_env,
            policy=model.trained_policy,
            build_count=build_count,
            episode_length=MAX_DEMO_EPISODE_LENGTH,
        )
        build_evaluator.print_report()


if __name__ == "__main__":
    run()
