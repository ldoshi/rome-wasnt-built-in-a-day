import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from training_history import TrainingHistory
from training_history_test import build_test_history

import numpy as np


def equally_spaced_indices(length, n):
    return np.round(np.linspace(0, length - 1, n)).astype(int)


class TrainingPanel:
    def __init__(self, states_n, state_width, state_height, actions_n):
        self._max_points_per_plot = 50
        self._states_n = states_n
        self._state_width = state_width
        self._state_height = state_height
        self._actions_n = actions_n
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

        plt.show(block=False)

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

    def _render_states(self, state_training_history):
        for i in range(min(len(state_training_history), self._states_n)):
            ax = self._axs[i, 0]
            ax.cla()
            ax.matshow(state_training_history[i].state, cmap="binary")
            ax.set_title("Visits: %d" % state_training_history[i].visit_count)

    def _render_series(self, state_training_history, column_index, method, title):
        for i in range(min(len(state_training_history), self._states_n)):
            ax = self._axs[i, column_index]
            ax.cla()
            history = state_training_history[i]
            for a in range(self._actions_n):
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

    def update_panel(self, state_training_history):
        self._render_states(state_training_history)
        self._render_series(state_training_history, 1, "get_q_values", "Q")
        self._render_series(
            state_training_history, 2, "get_q_target_values", "Q Target"
        )
        self._render_series(state_training_history, 3, "get_td_errors", "TD Error")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


if __name__ == "__main__":
    history = build_test_history()
    training_panel = TrainingPanel(3, 2, 2, 3)

    training_panel.update_panel(history.get_history_by_visit_count())
