# Training scenarios used for debugging.

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from bridger import training_panel
from bridger.builder import BridgeBuilder

# Creates the smallest world where a bridge can be built. Cycles
# through a series of actions to build simple episodes into the
# training set. Inspecting the TrainingPanel should show that the
# value function reflects the experiences encountered.


def tiny_world():

    MAX_EPISODE_LENGTH = 5

    model = BridgeBuilder.instantiate(
        env_height=3,
        env_width=4,
        env_vary_heights=False,
        debug=True,
        max_episode_length=MAX_EPISODE_LENGTH,
    )

    panel = training_panel.TrainingPanel(
        states_n=10,
        state_width=model.hparams.env_width,
        state_height=model.hparams.env_height,
        actions_n=model.env.nA,
    )

    class FixedRotatingPolicy(Callback):
        def __init__(self, max_episode_length):
            self.max_episode_length = max_episode_length
            self.actions = self.action_sequence()

        def action_sequence(self):
            # Exercises various scenarios.
            while True:
                yield from [0] * self.max_episode_length  # Only the left
                yield from [1] * self.max_episode_length  # Only the middle gap
                yield from [2] * self.max_episode_length  # Only the right
                yield from [0] + [1] * self.max_episode_length  # Left then middle
                yield from [2] + [1] * self.max_episode_length  # Right then middle
                yield from [0, 2]  # Left to right
                yield from [2, 0]  # Right to left

        def on_train_start(self, trainer, model):
            model.policy = lambda *args, **kwargs: next(self.actions)

    trainer = Trainer(
        val_check_interval=int(1e6),
        max_steps=10 * (5 * MAX_EPISODE_LENGTH + 6),
        callbacks=[FixedRotatingPolicy(MAX_EPISODE_LENGTH)],
    )
    trainer.fit(model)

    panel.update_panel(model.training_history.get_history_by_visit_count())
    return model, panel


# env and replay_buffer should be treated as read-only.
class DebugUtil:
    def __init__(self, environment_name, env, replay_buffer):
        self._replay_buffer = replay_buffer
        self._debug_env = gym.make(environment_name)
        self._debug_env.setup(
            env.shape[0], env.shape[1], vary_heights=(len(env.height_pairs) > 1)
        )

    # Returns the state following the provided series of actions after a reset().
    def get_state(self, actions=None):
        state = self._debug_env.reset()
        for a in actions:
            state, _, _, _ = self._debug_env.step(a)

        return state

    # Returns entries from replay buffer.
    # Filters on states, actions, and rewards are AND-ed together.
    # Filters within an input, such as actions, are OR-ed together. Provide None to match all.
    def extract_replay_buffer_entries(self, states=None, actions=None, rewards=None):
        out = []
        if None == states == actions == rewards:  # noqa: E711
            return out

        for entry in self._replay_buffer._content:
            if states and entry[0] not in states:
                continue
            elif actions and entry[1] not in actions:
                continue
            elif rewards and entry[3] not in rewards:
                continue
            out.append(entry)

        return out
