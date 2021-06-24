import argparse
import IPython
import gym
import gym_bridges.envs
import numpy as np
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

from bridger import config, policies, qfunctions, replay_buffer, training_history


# TODO(arvind): Encapsulate all optional parts of workflow (e.g. interactive
# mode, debug mode, display mode) as Lightning Callbacks
class BridgeBuilder(pl.LightningModule):
    def __init__(self, hparams):
        """Constructor for the BridgeBuilder Module

        Args:
            hparams: a Namespace object, of the kind returned by an argparse
                     ArgumentParser. For more details, see
                     #get_hyperparam_parser"""

        super(BridgeBuilder, self).__init__()
        #  TODO(arvind) Simplify once you understand hyperparam handling in PL 1.3
        for k, v in hparams.__dict__.items():
            self.hparams[k] = v
        torch.manual_seed(hparams.seed)

        self.env = self.make_env()

        self.replay_buffer = replay_buffer.ReplayBuffer(
            capacity=hparams.capacity,
            alpha=hparams.alpha,
            beta=hparams.beta_training_start,
            batch_size=hparams.batch_size,
        )

        state = self.env.reset()
        self.Q = qfunctions.CNNQ(*state.shape, self.env.nA)
        self.target = qfunctions.CNNQ(*state.shape, self.env.nA)
        self.target.load_state_dict(self.Q.state_dict())
        # TODO(lyric): Consider specifying the policy as a hyperparam
        self.policy = policies.EpsilonGreedyPolicy(self.Q)

        self.epsilon = hparams.epsilon_training_start
        self.memories = self._memory_generator()

        self.next_action = None
        self._breakpoint = {"step": 0, "episode": 0}

        if hparams.debug:
            # TODO(arvind): Move as much of this functionality as possible into
            # the tensorboard logging already being done here.
            self.training_history = training_history.TrainingHistory()

    def make_env(self):
        env = gym.make(
            self.hparams.env_name,
            width=self.hparams.env_width,
            force_standard_config=self.hparams.env_force_standard_config,
        )
        return env

    def on_train_start(self):
        self.make_memories()

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.update_target()
        if self.hparams.debug:
            self.record_q_values(batch_idx)
        self.make_memories()

    def update_target(self):
        params = self.target.state_dict()
        update = self.Q.state_dict()
        for param in params:
            params[param] += self.hparams.tau * (update[param] - params[param])
        self.target.load_state_dict(params)

    def record_q_values(self, training_step):
        visited_state_histories = self.training_history.get_history_by_visit_count(100)
        states = [
            visited_state_history.state
            for visited_state_history in visited_state_histories
        ]
        states_tensor = torch.tensor(states)
        triples = zip(
            states, self.Q(states_tensor).tolist(), self.target(states_tensor).tolist()
        )
        for triple in triples:
            self.training_history.add_q_values(training_step, *triple)

    def make_memories(self):
        with torch.no_grad():
            for i in range(self.hparams.inter_training_steps):
                next(self.memories)

    def _memory_generator(self):
        """A generator that serves up sequential transitions experienced by the
        agent. When an episode ends, a new one starts immediately. Each item
        yielded is a tuple with the following elements (in order):

         episode_idx: starting from 0, incremented every time an episode ends
                      and another begins
         step_idx:    starting from 0, incremented with each transition,
                      irrespective of the episode it is in
         start_state: the state at the beginning of the transition
         action:      the action taken during the transition
         end_state:   the state at the end of the transition
         reward:      the reward gained through the transition
         finished:    whether the transition marked the end of the episode"""

        episode_idx = 0
        total_step_idx = 0
        while True:
            for step_idx in range(self.hparams.max_episode_length):
                self._checkpoint({"episode": episode_idx, "step": total_step_idx})
                start_state, action, end_state, reward, finished = self()
                if self.hparams.debug:
                    self.training_history.increment_visit_count(start_state)
                yield (
                    episode_idx,
                    step_idx,
                    start_state,
                    action,
                    end_state,
                    reward,
                    finished,
                )
                total_step_idx += 1
                if finished:
                    break
            self._update_epsilon()
            self.env.reset()
            episode_idx += 1

    def _checkpoint(self, thresholds):
        """A checkpointer that compares instance-state breakpoints to method
        inputs to determine whether to enter a breakpoint. This only runs while
        interactive mode is enabled.

        Args:
         thresholds: a dict mapping some subset of 'episode' and 'step' to
                     the current corresponding indices (as tracked by
                     `_memory_generator#`)

        When these current-state thresholds reach or exceed corresponding
        values in the instance variable `breakpoint`, a breakpoint is entered
        (via `IPython.embed#`). This breakpoint will reoccur immediately and
        repeatedly, even as the user manually exits the IPython shell, until
        self._breakpoint has been updated"""
        while self.hparams.interactive_mode:
            if all(self._breakpoint[k] > v for k, v in thresholds.items()):
                break  # Don't stop for a breakpoint
            self._breakpoint.update(thresholds)
            self.next_action = None
            IPython.embed()

    def enable_interactive_mode(self):
        self.hparams.interactive_mode = True

    def disable_interactive_mode(self):
        self.hparams.interactive_mode = False

    def take_action(self, action, repetitions=1):
        """Exits the current breakpoint and reenters one only after the input
        `action` has been taken `repetitions` times (or the current episode
        has ended)."""
        self.next_action = action
        # Updates self._breakpoint for use in self._checkpoint#
        self._breakpoint["episode"] += 1  # Run until current episode ends OR
        self._breakpoint["step"] += repetitions  # for `repetitions` steps
        # Exits current breakpoint
        IPython.core.getipython.get_ipython().exiter()

    def follow_policy(self, num_actions=None, num_episodes=1):
        """Exits the current breakpoint and draws actions from the policy.
        Reenters a breakpoint only after either `num_actions` steps have been
        taken or `num_episodes` episodes have newly finished.

        Note: it is expected that only one of `num_actions` and `num_episodes`
              are set. If `num_actions` is set, these actions will be preempted
              by the end of the current episode. If `num_episodes` is set,
              no limit is placed on the total number of actions taken. Finally,
              if neither is set, the policy will be followed until the end of
              the current epsiode."""
        # Updates self._breakpoint for use in self._checkpoint#
        if num_actions is None:
            self._breakpoint["step"] = np.inf  # Run indefinitely until ...
        else:
            assert num_episodes == 1
            self._breakpoint["step"] += num_actions  # Take `num_actions` steps
        # ... `num_episodes` episodes have completed
        self._breakpoint["episode"] += num_episodes
        IPython.core.getipython.get_ipython().exiter()

    def return_to_training(self):
        self.disable_interactive_mode()
        IPython.core.getipython.get_ipython().exiter()

    def forward(self):
        state = self.env.state
        if self.hparams.interactive_mode and self.next_action is not None:
            action = self.next_action
        else:
            action = self.policy(
                torch.as_tensor(state, dtype=torch.float), epsilon=self.epsilon
            )
        result = (state, action, *self.env.step(action)[:3])
        self.replay_buffer.add_new_experience(*result)

        if self.hparams.env_display:
            self.env.render()

        return result

    def _update_epsilon(self):
        if self.hparams.epsilon_decay_rule == "arithmetic":
            self.epsilon -= self.hparams.epsilon_decay_rate
        elif self.hparams.epsilon_decay_rule == "geometric":
            self.epsilon /= self.hparams.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.hparams.epsilon)

    def _update_beta(self):
        if self.hparams.beta_growth_rule == "arithmetic":
            self.replay_buffer.beta += self.hparams.beta_growth_rate
        elif self.hparams.beta_growth_rule == "geometric":
            self.replay_buffer.beta *= self.hparams.beta_growth_rate
        self.replay_buffer.beta = min(
            self.replay_buffer.beta, self.hparams.beta_training_end
        )

    def get_td_error(self, states, actions, next_states, rewards, finished):

        row_idx = torch.arange(actions.shape[0])
        qvals = self.Q(states)[row_idx, actions]
        with torch.no_grad():
            next_actions = self.Q(next_states).argmax(dim=1)
            next_vals = self.target(next_states)[row_idx, next_actions]
            expected_qvals = rewards + (~finished) * self.hparams.gamma * next_vals
        return torch.abs(expected_qvals - qvals)

    def compute_loss(self, td_errors, weights=None):
        if weights is not None:
            td_errors = weights * td_errors
        # TODO(arvind): Change design to clip the gradient rather than the loss
        return (td_errors.clip(max=self.hparams.update_bound) ** 2).mean()

    def training_step(self, batch, batch_idx):
        indices, states, actions, next_states, rewards, finished, weights = batch
        td_errors = self.get_td_error(states, actions, next_states, rewards, finished)
        if self.hparams.debug:
            triples = zip(states.tolist(), actions.tolist(), td_errors.tolist())
            for triple in triples:
                # For debuging only. Averages the td error per (state, action) pair.
                self.training_history.add_td_error(batch_idx, *triple)

        loss = self.compute_loss(td_errors, weights=weights)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Update replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        self._update_beta()

        return loss

    # TODO(arvind): Override hooks to compute non-TD-error metrics for val and test

    def configure_optimizers(self):
        # TODO(arvind): This should work, but should we say Q.parameters(), or
        # is that limiting for the future?
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return DataLoader(self.replay_buffer, batch_size=self.hparams.batch_size)

    # TODO(arvind): Override hooks to load data appropriately for val and test

    @staticmethod
    def get_hyperparam_parser(parser=None):
        return config.get_hyperparam_parser(
            config.bridger_config,
            description="Hyperparameter Parser for the BridgeBuilder Model",
            parser=parser,
        )

    @staticmethod
    def instantiate(**kwargs):
        missing = [key for key in kwargs if key not in config.bridger_config]
        if len(missing) > 0:
            missing = ",".join(missing)
            print(
                "WARNING: The following are not recognized hyperparameters "
                f"for BridgeBuilder: {missing}"
            )
        hparams = dict(**kwargs)
        for key, val in config.bridger_config.items():
            if key not in hparams:
                check = not val.get("required", False)
                assert check, f"Required argument {key} not provided"
                if "default" in val:
                    hparams[key] = val["default"]
        return BridgeBuilder(argparse.Namespace(**hparams))
