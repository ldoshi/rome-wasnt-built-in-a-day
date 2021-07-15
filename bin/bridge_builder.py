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
# 3. take_action will take the requested action, potentially mutliple times, before
#    returning to the IPython shell

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from bridger import builder
from bridger.callbacks import DemoCallback, HistoryCallback


def test():
    MAX_STEPS = 100
    MAX_DEMO_EPISODE_LENGTH = 50
    # TODO(arvind): split the args into those relevant for the LightningModule
    #               and those relevant for the Trainer/Callbacks
    parser = builder.get_hyperparam_parser()
    hparams = parser.parse_args()
    # hparams.debug = True
    model = builder.BridgeBuilder(hparams)

    callbacks = [
        # Only retains checkpoint with minimum monitored quantity seen so far.
        # By default, just saves the (temporally) last checkpoint. This should
        # eventually be done based on monitoring a well defined validation
        # metric that doesn't depend on the most recent batch of memories
        ModelCheckpoint(
            monitor=None,  # Should show a quantity, e.g. "train_loss"
            every_n_train_steps=hparams.checkpoint_interval,
        ),
    ]
    if hparams.debug:
        callbacks += [
            HistoryCallback(
                steps_per_update=MAX_STEPS,
            ),
            DemoCallback(
                steps_per_update=MAX_STEPS,
                max_episode_length=MAX_DEMO_EPISODE_LENGTH,
            ),
        ]
    # TODO: After validation logic has been added to BridgeBuilder,
    # 1. Make val_check_interval below a settable parameter with reasonable default
    # 2. Update callback variable above to reflect the validation logic and pass it
    #    to Trainer init below
    trainer = Trainer(
        gradient_clip_val=hparams.gradient_clip_val,
        val_check_interval=int(1e6),
        default_root_dir=hparams.checkpoint_model_dir,
        max_steps=hparams.max_training_batches,
        callbacks=callbacks,
    )

    trainer.fit(model)


if __name__ == "__main__":
    test()
