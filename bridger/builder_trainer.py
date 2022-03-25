import argparse
import IPython
import copy
import gym
import numpy as np
from bridger import builder
import pytorch_lightning as pl
import torch
from typing import Union

from torch.utils.data import DataLoader
from typing import Any, Union, Generator, Optional


from bridger import config, policies, qfunctions, replay_buffer, utils
from bridger.logging import object_logging
from bridger.logging import log_entry

# TODO(lyric): Make this is a training config if being able to
# override it proves to be desirable.
_FREQUENTLY_VISITED_STATE_COUNT = 100


def get_hyperparam_parser(parser=None) -> argparse.ArgumentParser:
    """Hyperparameter parser for the BridgeBuilderTrainer Model.

    Reads hyperparameters from bridger.config as well as environment parameters
    for setup. Arguments can also be passed directly to the command-line
    explicitly (i.e. --tau=0.1).

    Reads from the following files:

    Config file:            Content:
    agent.py               hyperparameters
    buffer.py              hyperparameters
    checkpointing.py       checkpoint parameters
    env.py                 Gym environment parameters
    training.py            hyperparameters and environment parameters

    Args:
        parser: an optional argument for parsing the command line. Currently,
        config.get_hyperparam_parser uses an ArgumentParser if None is passed.

    Returns:
        A parser with the loaded config parameters.
    """
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
    """Function that instantiates an instance of the environment with the appropriate arguments.

    Args:
        name: name of environment to construct.
        width: width of the
        bridge_builder environment.
        force_standard_config: whether to only use the standard environment configuration.

    Returns:
        An instantiated gym environment.
    """
    env = gym.make(
        name, width=width, force_standard_config=force_standard_config, seed=seed
    )
    return env


class ValidationBuilder(torch.utils.data.IterableDataset):
    """Produces build results using a policy based on the current model.

    The model underpinning the policy refers to the same one being
    trained and will thus evolve over time."""

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

# TODO(arvind): Redesign the signature checking mechanism. Using
# utils.validate_input# is not robust with changes in Lightning functionality

# pylint: disable=too-many-instance-attributes
class BridgeBuilderModel(pl.LightningModule):
    @utils.validate_input("BridgeBuilderModel", config.bridger_config)
    def __init__(
        self,
        object_log_manager: object_logging.ObjectLogManager,
        *args,
        hparams=None,
        **kwargs
    ):
        """Constructor for the BridgeBuilderModel Module

        Args:
          object_log_manager: Logger for pickle-able objects.
          hparams: Dictionary or argparse.Namespace object containing hyperparameters
            to be used for initialization.

        Keyword Args:
            A dictionary containing hyperparameters to be used for initializing this LightningModule.

        Note - if a key is found in both `hparams` and `kwargs`, the value in
            `kwargs` will be used"""

        super().__init__()
        self._object_log_manager = object_log_manager
        # TODO(Issue#106): Find a more efficient make_hashable_fn than
        # str.
        self._state_logger = object_logging.LoggerAndNormalizer(
            log_filename=log_entry.STATE_NORMALIZED_LOG_ENTRY,
            object_log_manager=self._object_log_manager,
            log_entry_object_class=torch.Tensor,
            make_hashable_fn=str,
        )
        self._state_visit_logger = object_logging.OccurrenceLogger(
            log_filename=log_entry.TRAINING_HISTORY_VISIT_LOG_ENTRY,
            object_log_manager=self._object_log_manager,
            log_entry_object_class=torch.Tensor,
            logger_and_normalizer=self._state_logger,
        )
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

    @property
    def trained_policy(self):
        return policies.GreedyPolicy(self.Q)

    def on_train_start(self):
        """Populates the replay buffer with an initial set of memories before training steps begin."""
        self.make_memories(
            batch_idx=-1, requested_memory_count=self.hparams.initial_memories_count
        )

    def on_train_batch_end(
        self,
        outputs: Union[torch.Tensor, dict[str, Any]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Complete follow-on calculations after the model weight updates made during the training step. Follow-on calculations include updating the target network, making additional memories using the updated model, and additional bookkeeping.

        Args:
            outputs: The output of a training step, type defined in pytorch_lightning/utilities/types.py.
            batch: A group of memories, size determined by `hparams.batch_size`.
            batch_idx: The index of the current batch, which also signifies the current round of model weight updates.
            dataloader_idx: The index of the dataloader.
        """
        self.update_target()

        if self.hparams.debug:
            self._record_q_values_debug_helper(batch_idx)
        self.make_memories(batch_idx)

    def update_target(self) -> None:
        """Updates the target network weights based on the Q network weights.

        The target network is updated using a weighted sum of its current weights and the Q network weights to increase stability in training."""
        params = self.target.state_dict()
        update = self.Q.state_dict()
        for param in params:
            params[param] += self.hparams.tau * (update[param] - params[param])
        self.target.load_state_dict(params)

    def _record_q_values_debug_helper(self, batch_idx: int) -> None:
        """Compute and log q values.

        Args:
           batch_idx: A sequential value identifying which training
             step produced the q values.

        """
        frequently_visted_states = self._state_visit_logger.get_top_n(
            _FREQUENTLY_VISITED_STATE_COUNT
        )

        frequently_visted_states_tensor = torch.stack(frequently_visted_states)
        for state, q_values, q_target_values in zip(
            frequently_visted_states,
            self.Q(frequently_visted_states_tensor).tolist(),
            self.target(frequently_visted_states_tensor).tolist(),
        ):
            state_id = self._state_logger.get_logged_object_id(state)
            for action, (q_value, q_target_value) in enumerate(
                zip(q_values, q_target_values)
            ):
                self._object_log_manager.log(
                    log_entry.TRAINING_HISTORY_Q_VALUE_LOG_ENTRY,
                    log_entry.TrainingHistoryQValueLogEntry(
                        batch_idx=batch_idx,
                        state_id=state_id,
                        action=action,
                        q_value=q_value,
                        q_target_value=q_target_value,
                    ),
                )

    def make_memories(self, batch_idx, requested_memory_count=None):
        """Makes memories according to the requested memory count or default number of steps."""
        memory_count = (
            requested_memory_count
            if requested_memory_count
            else self.hparams.inter_training_steps
        )
        with torch.no_grad():
            for _ in range(memory_count):
                _, _, start_state, _, _, _, _ = next(self.memories)
                if self.hparams.debug:
                    self._state_visit_logger.log_occurrence(
                        batch_idx=batch_idx, object=torch.Tensor(start_state).double()
                    )

    def _memory_generator(
        self,
    ) -> Generator[tuple[int, int, Any, Any, Any, Any, Any], None, None]:
        """A generator that serves up sequential transitions experienced by the
        agent. When an episode ends, a new one starts immediately.

        Returns:
            Generator object yielding tuples with the following values:

            episode_idx: Starting from 0, incremented every time an episode ends
                        and another begins.
            step_idx:    Starting from 0, incremented with each transition,
                        irrespective of the episode it is in.
            start_state: The state at the beginning of the transition.
            action:      The action taken during the transition.
            end_state:   The state at the end of the transition.
            reward:      The reward gained through the transition.
            success:    Whether the transition marked the end of the episode."""

        episode_idx = 0
        total_step_idx = 0
        while True:
            for step_idx in range(self.hparams.max_episode_length):
                self._checkpoint({"episode": episode_idx, "step": total_step_idx})
                start_state, action, end_state, reward, success = self()
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

    def _checkpoint(self, thresholds: dict[str, int]) -> None:
        """A checkpointer that compares instance-state breakpoints to method
        inputs to determine whether to enter a breakpoint and only runs while
        interactive mode is enabled.

        When these current-state thresholds reach or exceed corresponding values
        in the instance variable `breakpoint`, a breakpoint is entered (via
        `IPython.embed#`). This breakpoint will reoccur immediately and
        repeatedly, even as the user manually exits the IPython shell, until
        self._breakpoint has been updated.

        Args:
            thresholds: A dict mapping some subset of 'episode' and 'step' to
            the current corresponding indices (as tracked by `_memory_generator#`).

        """
        while self.hparams.interactive_mode:
            if all(self._breakpoint[k] > v for k, v in thresholds.items()):
                break  # Don't stop for a breakpoint
            self._breakpoint |= thresholds
            self.next_action = None
            IPython.embed()

    def enable_interactive_mode(self) -> None:
        """Enables interactive mode. Sets a flag to enable an interactive shell
        with IPython that can be used as an interpreter during training,
        allowing introspection into the current Namespace. See
        `bridge_builder.py` for an example of usage.

        If this is called with command line arg "interactive-mode" set to True,
        then the Trainer will intermittently enter an IPython shell, allowing
        you to inspect model state at your leisure. This shell can be exited by
        calling one of three model commands:

        1. return_to_training will disable interactive mode and complete the
           requested training without additional IPython breakpoints.

        2. follow_policy will run the minimum of a requested number of steps or
           through the end of a requested number of episodes before returning to
           the IPython shell

        3. take_action will take the requested action, potentially multiple
           times, before returning to the IPython shell.
        """
        self.hparams.interactive_mode = True

    def disable_interactive_mode(self) -> None:
        """Disables interactive mode."""
        self.hparams.interactive_mode = False

    def take_action(self, action: int, repetitions: int = 1):
        """Exits the current breakpoint and reenters one only after the input
        `action` has been taken `repetitions` times (or the current episode has
        ended)."""
        self.next_action = action
        # Updates self._breakpoint for use in self._checkpoint#
        self._breakpoint["episode"] += 1  # Run until current episode ends OR
        self._breakpoint["step"] += repetitions  # for `repetitions` steps
        # Exits current breakpoint
        IPython.core.getipython.get_ipython().exiter()

    def follow_policy(
        self, num_actions: Optional[int] = None, num_episodes: int = 1
    ) -> None:
        """Exits the current breakpoint and draws actions from the policy.
        Reenters a breakpoint only after either `num_actions` steps have been
        taken or `num_episodes` episodes have newly succeeded.

        Note: It is expected that only one of `num_actions` and `num_episodes`
              are set. If `num_actions` is set, these actions will be preempted
              by the end of the current episode. If `num_episodes` is set, no
              limit is placed on the total number of actions taken. Finally, if
              neither is set, the policy will be followed until the end of the
              current episode."""
        # Updates self._breakpoint for use in self._checkpoint#
        if num_actions is None:
            self._breakpoint["step"] = np.inf  # Run indefinitely until ...
        else:
            assert num_episodes == 1
            self._breakpoint["step"] += num_actions  # Take `num_actions` steps
        # ... `num_episodes` episodes have completed
        self._breakpoint["episode"] += num_episodes
        IPython.core.getipython.get_ipython().exiter()

    def return_to_training(self) -> None:
        """Exits the current breakpoint."""
        self.disable_interactive_mode()
        IPython.core.getipython.get_ipython().exiter()

    def forward(self):
        """Forward propagation of the model. If in interactive mode, the user
        can provide the next action with `take_action`. Otherwise, action is
        determined by policy."""
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

    def _update_epsilon(self) -> None:
        """Selects the rule by which epsilon changes in the Epsilon Greedy
        algorithm. See `_memory_generator` for usage."""
        if self.hparams.epsilon_decay_rule == "arithmetic":
            self.epsilon -= self.hparams.epsilon_decay_rate
        elif self.hparams.epsilon_decay_rule == "geometric":
            self.epsilon /= self.hparams.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.hparams.epsilon)

    def _update_beta(self) -> None:
        """Updates beta according to pre-specified hyperparameters.

        Beta is a measure of how much prioritized memories in the replay buffer
        should be preferred. For more information, look up Gibbs sampling or
        importance sampling."""
        if self.hparams.beta_growth_rule == "arithmetic":
            self.replay_buffer.beta += self.hparams.beta_growth_rate
        elif self.hparams.beta_growth_rule == "geometric":
            self.replay_buffer.beta *= self.hparams.beta_growth_rate
        self.replay_buffer.beta = min(
            self.replay_buffer.beta, self.hparams.beta_training_end
        )

    def get_td_error(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        success: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates TD error during training and estimates the state-value
        function of Markov decision process under a policy.

        Args:
            states: The state at the beginning of the transition.
            actions: The action taken during the transition.
            next_states: The state at the end of the transition.
            rewards: The reward gained through the transition.
            success: Whether the transition marked the end of the episode.
        """

        row_idx = torch.arange(actions.shape[0])
        qvals = self.Q(states)[row_idx, actions]
        with torch.no_grad():
            next_actions = self.Q(next_states).argmax(dim=1)
            next_vals = self.target(next_states)[row_idx, next_actions]
            expected_qvals = rewards + (~success) * self.hparams.gamma * next_vals
        return expected_qvals - qvals

    def compute_loss(
        self, td_errors: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes loss using td errors in training.

        Args:
            td_errors: Calculated errors to be back-propagated from time t+1 to time t.
            weights: An optional tensor that weights calculation of td_errors.
        """
        if weights is not None:
            td_errors = weights * td_errors
        return (td_errors ** 2).mean()

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a single training step on the Q network.

        The loss is computed across a batch of memories sampled from the replay buffer. The replay buffer sampling weights are updated based on the TD error from the samples.
        """
        indices, states, actions, next_states, rewards, success, weights = batch
        td_errors = self.get_td_error(states, actions, next_states, rewards, success)

        loss = self.compute_loss(td_errors, weights=weights)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        if self.hparams.debug:
            # TODO(Issue#105): Move more deepcopy calls to logging
            # layer. Check if any deepcopy operations can be avoided.
            indices_copy = copy.deepcopy(indices)
            actions_copy = copy.deepcopy(actions)
            rewards_copy = copy.deepcopy(rewards)
            success_copy = copy.deepcopy(success)
            weights_copy = copy.deepcopy(weights)

            self._object_log_manager.log(
                log_entry.TRAINING_BATCH_LOG_ENTRY,
                log_entry.TrainingBatchLogEntry(
                    batch_idx=batch_idx,
                    indices=indices_copy,
                    state_ids=[
                        self._state_logger.get_logged_object_id(state)
                        for state in states
                    ],
                    actions=actions_copy,
                    next_state_ids=[
                        self._state_logger.get_logged_object_id(next_state)
                        for next_state in next_states
                    ],
                    rewards=rewards_copy,
                    successes=success_copy,
                    weights=weights_copy,
                    loss=loss,
                ),
            )

            for state, action, td_error in zip(
                states, actions.tolist(), td_errors.tolist()
            ):
                self._object_log_manager.log(
                    log_entry.TRAINING_HISTORY_TD_ERROR_LOG_ENTRY,
                    log_entry.TrainingHistoryTDErrorLogEntry(
                        batch_idx=batch_idx,
                        state_id=self._state_logger.get_logged_object_id(state),
                        action=action,
                        td_error=td_error,
                    ),
                )

        # Update replay buffer.
        self.replay_buffer.update_priorities(indices, td_errors)
        self._update_beta()
        return loss

    def validation_step(self, batch, batch_idx):
        success, rewards, steps = batch
        self.log("val_reward", torch.Tensor.float(rewards).mean())

    # TODO(arvind): Override hooks to compute non-TD-error metrics for val and test

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Returns the optimizer to use for training"""
        # TODO(arvind): This should work, but should we say Q.parameters(), or
        # is that limiting for the future?
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        """Samples a batch of memories from the replay buffer for training.

        Used to prevent early bottlenecking in the model at data generation step
        and allows for computations to be split across multiple source files."""
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
