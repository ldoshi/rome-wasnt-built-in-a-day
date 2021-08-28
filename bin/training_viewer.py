#!/usr/bin/env python

"""Display training and debugging information. 

The training viewer reads TrainingHistory from the directory indicated
by the flag `training_history_dir` and displays it via the
TrainingPanel.
"""

import time

from bridger import builder_trainer
from bridger.training_history import TrainingHistory
from bridger.training_panel import TrainingPanel


def view_training():
    parser = builder_trainer.get_hyperparam_parser()
    hparams = parser.parse_args()
    env = builder_trainer.make_env(hparams)

    panel = TrainingPanel(
        states_n=10,
        state_width=env.shape[1],
        state_height=env.shape[0],
        actions_n=env.nA,
    )

    history = TrainingHistory(deserialization_dir=hparams.training_history_dir)

    # A fancier version of this loop could use watchdog to monitor for
    # new files.
    while True:
        new_data = history.deserialize_latest()
        if new_data:
            panel.update_panel(history.get_history_by_visit_count())

        time.sleep(30)


if __name__ == "__main__":
    view_training()
