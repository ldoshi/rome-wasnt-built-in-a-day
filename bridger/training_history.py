import numpy as np

from collections import defaultdict


class StateTrainingHistoryDataSeries:
    def __init__(self):
        self.epochs = []
        self.data = []


class StateTrainingHistory:
    def __init__(self, state):
        self._state = state
        self._state_complexity = int(np.sum(self._state))
        self._visit_count = 0
        self._q_epochs = []
        self._q_history = defaultdict(list)
        self._q_target_history = defaultdict(list)
        self._td_errors = defaultdict(StateTrainingHistoryDataSeries)

    def add_q_values(self, epoch, q_values, q_target_values):
        assert not self._q_history or len(self._q_history) == len(q_values)
        assert not self._q_target_history or len(self._q_target_history) == len(
            q_target_values
        )

        self._q_epochs.append(epoch)
        for a, q_value in enumerate(q_values):
            self._q_history[a].append(q_value)
        for a, q_target_value in enumerate(q_target_values):
            self._q_target_history[a].append(q_target_value)

    def add_td_error(self, action, epoch, td_error):
        series = self._td_errors[action]
        series.epochs.append(epoch)
        series.data.append(td_error)

    def increment_visit_count(self):
        self._visit_count += 1

    def get_td_errors(self, action):
        if action not in self._td_errors:
            return [], []

        series = self._td_errors[action]
        return series.epochs, series.data

    def get_q_values(self, action):
        return self._q_epochs, self._q_history[action]

    def get_q_target_values(self, action):
        return self._q_epochs, self._q_target_history[action]

    @property
    def state(self):
        return self._state

    @property
    def visit_count(self):
        return self._visit_count

    @property
    def state_complexity(self):
        # Currently implemented as the number of bricks
        return self._state_complexity


class TrainingHistory:
    def __init__(self, state_hash=str):
        self._training_history = {}
        self._state_hash = state_hash

    def _add_if_missing(self, state):
        key = self._state_hash(state) if self._state_hash else state
        if key not in self._training_history:
            self._training_history[key] = StateTrainingHistory(state)
        return key

    def add_q_values(self, epoch, state, q_values, q_target_values):
        self._training_history[self._add_if_missing(state)].add_q_values(
            epoch, q_values, q_target_values
        )

    def add_td_error(self, epoch, state, action, td_error):
        self._training_history[self._add_if_missing(state)].add_td_error(
            action, epoch, td_error
        )

    def increment_visit_count(self, state):
        self._training_history[self._add_if_missing(state)].increment_visit_count()

    # These are sorted in descending order.
    def get_history_by_visit_count(self, n=None):
        return [
            v
            for k, v in sorted(
                self._training_history.items(),
                key=lambda item: item[1].visit_count,
                reverse=True,
            )
        ][:n]

    # These are sorted in ascending order.
    def get_history_by_complexity(self, n=None):
        return [
            v
            for k, v in sorted(
                self._training_history.items(),
                key=lambda item: item[1].state_complexity,
            )
        ][:n]
