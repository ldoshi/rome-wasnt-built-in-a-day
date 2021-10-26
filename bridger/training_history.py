import copy
import numpy as np
import pathlib
import pickle
import time

from collections import defaultdict

_TRAINING_HISTORY_PREFIX = "training_history_entry_{}"


class StateTrainingHistoryDataSeries:
    def __init__(self):
        self.epochs = []
        self.data = []


class StateTrainingHistory:
    def __init__(self, state):
        self._state = copy.deepcopy(state)
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
            assert len(self._q_epochs) == len(self._q_history[a]), q_values
        for a, q_target_value in enumerate(q_target_values):
            self._q_target_history[a].append(q_target_value)
            assert len(self._q_epochs) == len(self._q_target_history[a]), q_target_values

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
    def __init__(
        self,
        serialization_dir=None,
        deserialization_dir=None,
        state_hash=lambda state: str(np.array(state)),
    ):
        self._training_history = {}
        self._state_hash = state_hash
        self._serialization_dir = serialization_dir
        if serialization_dir:
            path = pathlib.Path(self._serialization_dir)
            path.mkdir(parents=True, exist_ok=True)
            for filepath in path.iterdir():
                filepath.unlink()

        self._deserialization_dir = deserialization_dir
        self._most_recently_deserialized = None

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

    # TODO(lyric): Consider making the serialization an async background task.
    #
    # TODO(lyric): The current serialization rewrites full state each
    # time, which can grow the file quickly. This makes the write load
    # closer to n^2 rather than n. The next improvement will be to
    # clear history after each serialization and then reassemble the
    # compontents during deserialize.
    def serialize(self):
        assert self._serialization_dir, "serialization_dir must be provided"
        id = int(time.time() * 1e6)
        path = pathlib.Path(
            self._serialization_dir, _TRAINING_HISTORY_PREFIX.format(id)
        )
        with path.open(mode="wb") as f:
            pickle.dump(self._training_history, f)

    def deserialize_latest(self):
        """Deserializes the latest file available in the _serialization_dir.

        Returns false if a new file was not detected and the content
        of training history is unchanged.

        """
        # TODO(lyric): This current loads the most recent file. If we
        # instead serialize in chunks and reassemble during
        # deserialization, this code will need to be adjusted.
        files = [x for x in pathlib.Path(self._deserialization_dir).iterdir()]
        if not files:
            return False

        filepath = max(files)
        if filepath == self._most_recently_deserialized:
            return False

        try:
            with filepath.open(mode="rb") as f:
                self._training_history = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            # We likely started reading the file before it was fully
            # written. Try again next time.
            return False

        self._most_recently_deserialized = filepath
        return True

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
