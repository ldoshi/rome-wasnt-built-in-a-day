#!/usr/bin/env python

"""Display training and debugging information. 

The training viewer reads training history files from the directory
indicated by the flag `object_logging_dir` and displays it via the
TrainingPanel.

"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from typing import Tuple

from bridger import builder_trainer
from bridger.logging import log_entry
from bridger.logging import object_log_readers

def equally_spaced_indices(length, n):
    return np.round(np.linspace(0, length - 1, n)).astype(int)

class TrainingPanel:
    def __init__(self, states_n: int , training_history_database: object_log_readers.TrainingHistoryDatabase):
        """Initializes the viewer plots
        
        Args:
          states_n: The max number of states for which to display debug information.
          training_history_database: The accessor to the logs that back the debug 
            data displayed in the viewer.

        """
        
        self._states_n = states_n
        self._training_history_database = training_history_database
        self._max_points_per_plot = 50
        self._plot_width = 4
        self._plot_count = 3

        width = 1 + self._plot_width * self._plot_count
        gs_args = {
            "width_ratios": [1, self._plot_width, self._plot_width, self._plot_width],
            "hspace": 0.5,
        }

        self._fig, self._axs = plt.subplots(
            self._states_n,
            4,
            gridspec_kw=gs_args,
            figsize=([2 * width, 2 * self._states_n]),
        )

        plt.ion()
        plt.show()

    def _state_plots_init(self):
        for i in range(self._states_n):
            ax = self._axs[0, i]
            # Major ticks
            ax.set_xticks(np.arange(0, 10, 1))
            ax.set_yticks(np.arange(0, 10, 1))
            # Labels for major ticks
            ax.set_xticklabels(np.arange(1, 11, 1))
            ax.set_yticklabels(np.arange(1, 11, 1))
            # Minor ticks
            ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)

            ax.grid(which="minor")

    def _render_states(self, states: pd.DataFrame):
        for i in range(min(len(states), self._states_n)):
            ax = self._axs[i, 0]
            ax.cla()
            ax.matshow(states.iloc[i].state, cmap="binary")
            ax.set_title(f"Visits: {states.iloc[i].visit_count}")

    def _render_series(self, state_training_history, column_index, method, title):
        for i in range(min(len(state_training_history), self._states_n)):
            ax = self._axs[i, column_index]
            ax.cla()
            history = state_training_history[i]
            for a in range(self._training_history_database.actions_n):
                xs, ys = getattr(history, method)(a)
                if len(xs) > self._max_points_per_plot:
                    subsampled_indices_for_plotting = equally_spaced_indices(
                        len(xs), self._max_points_per_plot
                    )
                    xs = [xs[index] for index in subsampled_indices_for_plotting]
                    ys = [ys[index] for index in subsampled_indices_for_plotting]
                ax.plot(xs, ys, label=a)

            ax.set_title(title)
            ax.legend()

    def update_panel(self, states: pd.DataFrame) -> None:
        """Updates the viewer to show debug information for the provided states.
        
        states: The states dataframe is expected to have 3 columns:
          state_id, state, and visit_count.

        """

        self._render_states(states)
#        self._render_series(state_training_history, 1, "get_q_values", "Q")
#        self._render_series(
 #           state_training_history, 2, "get_q_target_values", "Q Target"
 #       )
 #       self._render_series(state_training_history, 3, "get_td_errors", "TD Error")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    
def view_training():
    parser = builder_trainer.get_hyperparam_parser()
    hparams = parser.parse_args()

    training_history_database = object_log_readers.TrainingHistoryDatabase(hparams.object_logging_dir)

    states_n = 10
    panel = TrainingPanel(        states_n=states_n,        training_history_database=training_history_database    )

    most_visited_states = training_history_database.get_states_by_visit_count(n=states_n)
    panel.update_panel(most_visited_states)

    plt.pause(1000000)


if __name__ == "__main__":
    view_training()
