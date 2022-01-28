import argparse
import IPython
import gym
import gym_bridges.envs
import numpy as np
from bridger import builder
import pytorch_lightning as pl
import torch
from typing import Union

from torch.utils.data import DataLoader

from bridger import config, policies, qfunctions, replay_buffer, training_history, utils
from bridger.logging import object_logging
from bridger.logging import log_entry


def get_hyperparam_parser(parser=None):
    return config.get_hyperparam_parser(
        config.bridger_config,
        description="Hyperparameter Parser for the BridgeBuilderModel",
        parser=parser,
    )


def make_env(
    name: str,
    width: int,
    force_standard_config: bool,
    seed: Union[int, float, None] = None,
) -> gym.Env:
    env = gym.make(
        name, width=width, force_standard_config=force_standard_config, seed=seed
    )
    return env


class ValidationBuilder(torch.utils.data.IterableDataset):
    """Produces build results using a policy based on the current model.

    The model underpinning the policy refers to the same one being
    trained and will thus evolve over time.

    """

    def __init__(self, env: gym.Env, policy: policies.Policy, episode_length: int):
        self._builder = builder.Builder(env)
        self._policy = policy
        self._episode_length = episode_length

    def __iter__(self):
        """Yields the result of a single building episode each call."""
        while True:
            build_result = self._builder.build(
                self._policy, self._episode_length, render=False
            )
            yield [build_result.success, build_result.reward, build_result.steps]


# TODO(arvind): Encapsulate all optional parts of workflow (e.g. interactive
# mode, debug mode, display mode) as Lightning Callbacks

# pylint: disable=too-many-instance-attributes
class BridgeBuilderModel(pl.LightningModule):
    @utils.validate_input("BridgeBuilderModel", config.bridger_config)
    def __init__(
        self,
        object_log_manager: object_logging.ObjectLogManager,
        hparams=None,
        **kwargs
    ):
        """Constructor for the BridgeBuilderModel Module

        Args:
          object_log_manager: Logger for pickle-able objects.
          hparams: A dictionary or argparse.Namespace object containing
            hyperparameters to be used for initialization

        Keyword Args: a dictionary containing hyperparameters to be used for
            initializing this LightningModule

        Note - if a key is found in both `hparams` and `kwargs`, the value in
            `kwargs` will be used"""

        super().__init__()
        self._object_log_manager = object_log_manager
        if hparams:
            self.save_hyperparameters(hparams)
        if kwargs:
            self.save_hyperparameters(kwargs)

        torch.manual_seed(self.hparams.seed)
        np.random.seed(self.hparams.seed)

        self.env = make_env(
            name=self.hparams.env_name,
            width=self.hparams.env_width,
            force_standard_config=self.hparams.env_force_standard_config,
            seed=torch.rand(1).item(),
        )
        self._validation_env = make_env(
            name=self.hparams.env_name,
            width=self.hparams.env_width,
            force_standard_config=self.hparams.env_force_standard_config,
            seed=torch.rand(1).item(),
        )

        self.replay_buffer = replay_buffer.ReplayBuffer(
            capacity=self.hparams.capacity,
            alpha=self.hparams.alpha,
            beta=self.hparams.beta_training_start,
            batch_size=self.hparams.batch_size,
        )

        self.Q = qfunctions.CNNQ(*self.env.shape, self.env.nA)
        self.target = qfunctions.CNNQ(*self.env.shape, self.env.nA)
        self.target.load_state_dict(self.Q.state_dict())
        # TODO(lyric): Consider specifying the policy as a hyperparam
        self.policy = policies.EpsilonGreedyPolicy(self.Q)
        # At this time, the world is static once the initial
        # conditions are set. The agent is not navigating a dynamic
        # environment.
        self._validation_policy = policies.GreedyPolicy(self.Q)

        self.epsilon = self.hparams.epsilon_training_start
        self.memories = self._memory_generator()

        self.next_action = None
        self.state = self.env.reset()
        self._breakpoint = {"step": 0, "episode": 0}

        if self.hparams.debug:
            # TODO(arvind): Move as much of this functionality as possible into
            # the tensorboard logging already being done here.
            self.training_history = training_history.TrainingHistory(
                serialization_dir=self.hparams.training_history_dir
            )

    @property
    def trained_policy(self):
        return policies.GreedyPolicy(self.Q)

    def on_train_start(self):
        self.make_memories(self.hparams.initial_memories_count)

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

    def make_memories(self, requested_memory_count=None):
        memory_count = (
            requested_memory_count
            if requested_memory_count
            else self.hparams.inter_training_steps
        )
        with torch.no_grad():
            for _ in range(memory_count):
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
         success:    whether the transition marked the end of the episode"""

        episode_idx = 0
        total_step_idx = 0
        while True:
            for step_idx in range(self.hparams.max_episode_length):
                self._checkpoint({"episode": episode_idx, "step": total_step_idx})
                start_state, action, end_state, reward, success = self()
                if self.hparams.debug:
                    self.training_history.increment_visit_count(start_state)
                yield (
                    episode_idx,
                    step_idx,
                    start_state,
                    action,
                    end_state,
                    reward,
                    success,
                )
                total_step_idx += 1
                if success:
                    break
            self._update_epsilon()
            self.state = self.env.reset()
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
            self._breakpoint |= thresholds
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
        taken or `num_episodes` episodes have newly succeeded.

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
        state = self.state
        if self.hparams.interactive_mode and self.next_action is not None:
            action = self.next_action
        else:
            action = self.policy(
                torch.as_tensor(state, dtype=torch.float), epsilon=self.epsilon
            )

        next_state, reward, done, _ = self.env.step(action)
        self.state = next_state
        result = (state, action, next_state, reward, done)
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

    def get_td_error(self, states, actions, next_states, rewards, success):

        row_idx = torch.arange(actions.shape[0])
        qvals = self.Q(states)[row_idx, actions]
        with torch.no_grad():
            next_actions = self.Q(next_states).argmax(dim=1)
            next_vals = self.target(next_states)[row_idx, next_actions]
            expected_qvals = rewards + (~success) * self.hparams.gamma * next_vals
        return expected_qvals - qvals

    def compute_loss(self, td_errors, weights=None):
        if weights is not None:
            td_errors = weights * td_errors
        return (td_errors ** 2).mean()

    def training_step(self, batch, batch_idx):
        indices, states, actions, next_states, rewards, success, weights = batch
        td_errors = self.get_td_error(states, actions, next_states, rewards, success)

        loss = self.compute_loss(td_errors, weights=weights)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        if self.hparams.debug:
            self._object_log_manager.log(
                log_entry.TRAINING_BATCH_LOG_ENTRY,
                log_entry.TrainingBatch(batch_idx, *batch, loss),
            )

            triples = zip(states.tolist(), actions.tolist(), td_errors.tolist())
            for triple in triples:
                # For debuging only. Averages the td error per (state, action) pair.
                self.training_history.add_td_error(batch_idx, *triple)

        # Update replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        self._update_beta()

        return loss

    def validation_step(self, batch, batch_idx):
        success, rewards, steps = batch
        self.log("val_reward", torch.Tensor.float(rewards).mean())

    # TODO(arvind): Override hooks to compute non-TD-error metrics for val and test

    def configure_optimizers(self):
        # TODO(arvind): This should work, but should we say Q.parameters(), or
        # is that limiting for the future?
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return DataLoader(
            self.replay_buffer,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            ValidationBuilder(
                env=self._validation_env,
                policy=self._validation_policy,
                episode_length=self.hparams.max_episode_length,
            ),
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
        )

    # TODO(arvind): Override hooks to load data appropriately for test
