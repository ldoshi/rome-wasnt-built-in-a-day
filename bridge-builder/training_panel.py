import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from training_history import TrainingHistory
from training_history_test import build_test_history

class TrainingPanel:
    def __init__(self, states_n, state_width, state_height, actions_n):
        self._states_n = states_n
        self._state_width = state_width
        self._state_height = state_height
        self._actions_n = actions_n
        self._plot_width = 4
        self._plot_count = 3
        width = 1 + self._plot_width * self._plot_count
        gs_args = {"width_ratios" : [1, self._plot_width, self._plot_width, self._plot_width],
                   "hspace" : .5}

        self._fig, self._axs = plt.subplots(self._states_n, 4, gridspec_kw=gs_args, figsize=([2 * width, 2 * self._states_n]))

        self._titles = ["State", "Q", "Q Target", "TD Target Delta"]
        for i in range(self._states_n):
            for j,title in enumerate(self._titles):
                self._axs[i,j].set_title(title)

        plt.show(block=False)

    def _state_plots_init(self):
        for i in range(self._states_n):
            ax = self._axs[0,i]
            # Major ticks
            ax.set_xticks(np.arange(0, 10, 1))
            ax.set_yticks(np.arange(0, 10, 1))
            # Labels for major ticks
            ax.set_xticklabels(np.arange(1, 11, 1))
            ax.set_yticklabels(np.arange(1, 11, 1))
            # Minor ticks
            ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 10, 1), minor=True)

            ax.grid(which='minor')
        
    def _render_states(self, state_training_history):
        for i in range(min(len(state_training_history), self._states_n)):
            ax = self._axs[i,0]
            ax.cla()
            ax.matshow(state_training_history[i].state, cmap='hot')

    def _render_series(self, state_training_history, column_index, method):
        for i in range(min(len(state_training_history), self._states_n)):
            ax = self._axs[i, column_index]
            ax.cla()
            history = state_training_history[i]
            for a in range(self._actions_n):
                xs, ys = getattr(history, method)(a)
                ax.plot(xs, ys, label=a)

            ax.legend()
        
    def update_panel(self, state_training_history):
        self._render_states(state_training_history)
        self._render_series(state_training_history, 1, "get_q_values")
        self._render_series(state_training_history, 2, "get_q_target_values")
        self._render_series(state_training_history, 3, "get_td_target_deltas")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        
if __name__ == '__main__':
    history = build_test_history()
    training_panel = TrainingPanel(3, 2, 2, 3)

    training_panel.update_panel(history.get_history_by_visit_count())
