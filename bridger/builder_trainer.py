import argparse
import dataclasses
import IPython
import copy
import gym
import pickle
import numpy as np
from bridger import builder
import lightning
import torch
from typing import Union, Optional, Callable, Hashable

from torch.utils.data import DataLoader
from typing import Any, Union, Generator, Optional


from bridger import (
    config,
    hash_utils,
    policies,
    qfunctions,
    replay_buffer,
    replay_buffer_initializers,
)
from bridger.debug import action_inversion_checker
from bridger.logging_utils import object_logging
from bridger.logging_utils import log_entry

# TODO(lyric): Make this is a training config if being able to
# override it proves to be desirable.
_FREQUENTLY_VISITED_STATE_COUNT = 100

Q_CNN = "cnn"
Q_TABULAR = "tabular"


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
        width: width of the bridge_builder environment.
        force_standard_config: whether to only use the standard environment configuration.

    Returns:
        An instantiated gym environment.
    """
    env = gym.make(
        name,
        width=width,
        force_standard_config=force_standard_config,
        seed=seed,
    )
    return env


def get_actions_for_standard_configuration(
    env_width: int,
) -> list[list[int]]:
    """Produces actions to build a bridge in the standard configuration.

    The implementation presumes an environment of even width where the
    bridge is built up from edge to edge without intermediate
    supports. The actions are initialized based on this presumption.

    Args:
      env_width: The width of the environment.

    Returns:
      The sequence of actions defining the target outcome when starting
      with the standard configuration.

    Raises:
      ValueError: If the env_width is not even.
    """
    if env_width % 2:
        raise ValueError(
            f"The env width ({env_width}) must be even to use the ActionInversionChecker."
        )

    bricks_per_side = int((env_width - 2) / 2)
    actions = [
        list(range(0, bricks_per_side)),
        list(
            range(
                env_width - 2,
                env_width - 2 - bricks_per_side,
                -1,
            )
        ),
    ]
    return actions


class ValidationBuilder(torch.utils.data.IterableDataset):
    """Produces build results using a policy based on the current model.

    The model underpinning the policy refers to the same one being
    trained and will thus evolve over time."""

    def __init__(self, env: gym.Env, policy: policies.Policy, episode_length: int, initial_state: np.ndarray, ):
        self._builder = builder.Builder(env)
        self._policy = policy
        self._episode_length = episode_length
        self._initial_state = initial_state

    def __iter__(self):
        """Yields the result of a single building episode each call."""
        while True:
            build_result = self._builder.build(
                self._policy, self._episode_length, render=False, initial_state=self._initial_state
            )
            yield [build_result.success, build_result.reward, build_result.steps]


class StateActionCache:
    """
    StateActionCache stores state-action pairs as keys for lookup for state, actions, next_state, rewards, and environment completion status for debugging.

    Attributes:
        self.hits: The number of cache hits.
        self.misses: The number of cache misses. TODO(joseph): Double check that the cache does not miss after being generated once with a given build environment.
    """

    def __init__(
        self,
        env: gym.Env,
        make_hashable_fn: Optional[Callable[[Any], Hashable]] = None,
    ):
        self._cache = {}
        self._env: gym.Env = env
        if make_hashable_fn:
            self._make_hashable_fn = make_hashable_fn
        else:
            self._make_hashable_fn = lambda x: x
        self.hits: int = 0
        self.misses: int = 0

    def __len__(self):
        return len(self._cache)

    def _cache_put(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """
        Puts the state-action into the cache along with the associated next_state, reward, and completion state.
        """
        self._cache[(self._make_hashable_fn(state), action)] = (
            state,
            action,
            next_state,
            reward,
            done,
        )

    def cache_get(
        self, state: np.ndarray, action: int
    ) -> tuple[np.ndarray, int, np.ndarray, float, bool]:
        """
        Get the next state, reward, and environment completion status for a given state-action pair. If the cache misses, compute the estimator by stepping through the current env.
        """
        state_representation = self._make_hashable_fn(state)
        if (state_representation, action) in self._cache:
            self.hits += 1
            return self._cache[(state_representation, action)]

        # If the cache misses, then compute the resulting (next_state, reward, done) given the (state, action_pair), add it to the cache, and return the result.
        self.misses += 1

        self._env.reset(state)
        next_state, reward, done, _ = self._env.step(action)
        self._cache_put(state, action, next_state, reward, done)

        return self._cache[(state_representation, action)]


# TODO(arvind): Encapsulate all optional parts of workflow (e.g. interactive
# mode, debug mode, display mode) as Lightning Callbacks

# TODO(arvind): Redesign the signature checking mechanism. Using
# config.validate_input# is not robust with changes in Lightning functionality


# TODO(lyric): Refactor and align with Joseph.
@dataclasses.dataclass(frozen=True)
class SuccessEntry:
    trajectory: tuple[int]
    reward: tuple[float]


# First impl: train until reward is >=. Soften this later, especially
# if we vary envs and also consider a success rate.
class BackwardAlgorithmManager:

    def __init__(
        self,
        success_entries: set[SuccessEntry],
        env: gym.Env,
        policy: policies.Policy,
        episode_length: int,
    ):
        self._builder = builder.Builder(env)
        self._policy = policy
        self._episode_length = episode_length

        self._success_entries = success_entries
        assert (
            len(self._success_entries) == 1
        ), "Started by assuming a single success entry for guidance."

        state = env.reset()
        self._start_states = []

        self._success_entry = next(iter(success_entries))
        for action in self._success_entry.trajectory:
            self._start_states.append(state)
            state, reward, done, _ = env.step(action)

        self._trajectory_index = len(self._start_states) - 1

    def state(self) -> np.ndarray:
        # TODO(lyric): Initial impl: just return the current trajectory index without jitter
        trajectory_index =max(0, self._trajectory_index)
        
        return self._start_states[trajectory_index]

    def move_backward_if_necessary(self) -> bool:
        build_result = self._builder.build(
            policy=self._policy,
            episode_length=self._episode_length,
            render=False,
            initial_state=self.state(),
        )

        if build_result.success and build_result.reward >= sum(
            self._success_entry.reward[self._trajectory_index :]
        ):
            self._trajectory_index -= 1
            return True
        return False


# pylint: disable=too-many-instance-attributes
class BridgeBuilderModel(lightning.LightningModule):
    @config.validate_input("BridgeBuilderModel", config.bridger_config)
    def __init__(
        self,
        object_log_manager: object_logging.ObjectLogManager,
        *args,
        hparams=None,
        **kwargs,
    ):
        """Constructor for the BridgeBuilderModel Module

        Args:
          object_log_manager: Logger for pickle-able objects.
          hparams: Dictionary or argparse.Namespace object containing hyperparameters
            to be used for initialization.

        Keyword Args:
            A dictionary containing hyperparameters to be used for initializing this LightningModule.

        Note - if a key is found in both `hparams` and `kwargs`, the value in
            `kwargs` will be used

        Raises:
          ValueError: If a param value does not match expectations.
        """

        super().__init__()
        self._object_log_manager = object_log_manager
        self._state_logger = object_logging.LoggerAndNormalizer(
            log_filename=log_entry.STATE_NORMALIZED_LOG_ENTRY,
            object_log_manager=self._object_log_manager,
            log_entry_object_class=torch.Tensor,
            make_hashable_fn=hash_utils.hash_tensor,
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
        self._state_action_cache = StateActionCache(
            env=self._validation_env, make_hashable_fn=hash_utils.hash_tensor
        )

        self.replay_buffer = replay_buffer.ReplayBuffer(
            capacity=self.hparams.capacity,
            alpha=self.hparams.alpha,
            beta=self.hparams.beta_training_start,
            batch_size=self.hparams.batch_size,
            debug=self.hparams.debug,
        )

        if self.hparams.q == Q_CNN:
            self.q_manager = qfunctions.CNNQManager(
                *self.env.shape, self.env.nA, self.hparams.tau
            )
        elif self.hparams.q == Q_TABULAR:
            self.q_manager = qfunctions.TabularQManager(
                env=self._validation_env,
                brick_count=self.hparams.tabular_q_initialization_brick_count,
                tau=self.hparams.tau,
            )
        else:
            if self.hparams.q in qfunctions.choices:
                raise ValueError(
                    f"Provided q function not supported in trainer: {self.hparams.q}"
                )
            else:
                raise ValueError(f"Unrecognized q function: {self.hparams.q}")

        # TODO(lyric): Consider specifying the policy as a hyperparam
        self.policy = policies.EpsilonGreedyPolicy(self.q_manager.q)
        # At this time, the world is static once the initial
        # conditions are set. The agent is not navigating a dynamic
        # environment.
        self._validation_policy = policies.GreedyPolicy(self.q_manager.q)

        self.epsilon = self.hparams.epsilon_training_start
        self.memories = self._memory_generator()

        self.next_action = None
        self.state = self.env.reset()
        self._breakpoint = {"step": 0, "episode": 0}

        self._action_inversion_checker = None
        if self.hparams.debug_action_inversion_checker:
            actions = get_actions_for_standard_configuration(self.hparams.env_width)
            self._action_inversion_checker = (
                action_inversion_checker.ActionInversionChecker(
                    env=self._validation_env, actions=actions
                )
            )

        #        with open(self.hparams.go_explore_success_entries_path) as f:
        #            success_entries = pickle.load(f)
        success_entries = {
#            SuccessEntry(trajectory=(0, 2), reward=(-0.1, -0.1))
            SuccessEntry(trajectory=(0, 1, 4, 3), reward=(-0.1, -0.1, -0.1, -0.1))
        }
        self._backward_algorithm_manager = BackwardAlgorithmManager(
            success_entries=success_entries,
            env=self._validation_env,
            policy=self._validation_policy,
            episode_length=self.hparams.max_episode_length,
        )

        self._make_initial_memories()

    @property
    def trained_policy(self):
        return policies.GreedyPolicy(self.q_manager.q)

    def _make_initial_memories(self):
        """Populates the replay buffer with an initial set of memories before training steps begin."""
        if self.hparams.initialize_replay_buffer_strategy is not None:
            replay_buffer_initializers.initialize_replay_buffer(
                strategy=self.hparams.initialize_replay_buffer_strategy,
                replay_buffer_capacity=self.hparams.capacity,
                env=self._validation_env,
                add_new_experience=self.replay_buffer.add_new_experience,
                state_visit_logger=self._state_visit_logger,
                state_logger=self._state_logger,
            )
        else:
            self.make_memories(
                batch_idx=-1, requested_memory_count=self.hparams.initial_memories_count
            )

    def on_train_batch_end(
        self,
        outputs: Union[torch.Tensor, dict[str, Any]],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Complete follow-on calculations after the model weight updates made during the training step. Follow-on calculations include updating the target network, making additional memories using the updated model, and additional bookkeeping.

        Args:
            outputs: The output of a training step, type defined in lightning/utilities/types.py.
            batch: A group of memories, size determined by `hparams.batch_size`.
            batch_idx: The index of the current batch, which also signifies the current round of model weight updates.
        """
        self.q_manager.update_target()

        if self.hparams.debug:
            self._record_q_values_debug_helper()

        moved_back = self._backward_algorithm_manager.move_backward_if_necessary()
        self.log(
            "moved_back", moved_back, on_step=False,on_epoch=True, prog_bar=False,logger=True
        )

        self.make_memories(batch_idx=self.global_step)

    def _record_q_values_debug_helper(self) -> None:
        """Compute and log q values."""
        frequently_visted_states = self._state_visit_logger.get_top_n(
            _FREQUENTLY_VISITED_STATE_COUNT
        )

        frequently_visted_states_tensor = torch.stack(frequently_visted_states)
        for state, q_values, q_target_values in zip(
            frequently_visted_states,
            self.q_manager.q(frequently_visted_states_tensor).tolist(),
            self.q_manager.target(frequently_visted_states_tensor).tolist(),
        ):
            state_id = self._state_logger.get_logged_object_id(state)
            for action, (q_value, q_target_value) in enumerate(
                zip(q_values, q_target_values)
            ):
                self._object_log_manager.log(
                    log_entry.TRAINING_HISTORY_Q_VALUE_LOG_ENTRY,
                    log_entry.TrainingHistoryQValueLogEntry(
                        batch_idx=self.global_step,
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
            if requested_memory_count is not None
            else self.hparams.inter_training_steps
        )
        with torch.no_grad():
            for _ in range(memory_count):
                _, _, start_state, _, _, _, _ = next(self.memories)
                if self.hparams.debug:
                    self._state_visit_logger.log_occurrence(
                        batch_idx=batch_idx, object=start_state
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
#            self.state = self.env.reset(self._backward_algorithm_manager.state())
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
            print(state.is_cuda, "hello lyric")
            action = self.policy(
                state, epsilon=self.epsilon
            )

        next_state, reward, done, _ = self.env.step(action)
        self.state = next_state
        result = (state, action, next_state, reward, done)
        self.replay_buffer.add_new_experience(
            *result,
            (
                self._state_logger.get_logged_object_id(torch.Tensor(state))
                if self.hparams.debug
                else None
            ),
        )

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
        qvals = self.q_manager.q(states)[row_idx, actions]
        with torch.no_grad():
            next_actions = self.q_manager.q(next_states).argmax(dim=1)
            next_vals = self.q_manager.target(next_states)[row_idx, next_actions]
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
        return (td_errors**2).mean()

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a single training step on the Q network.

        The loss is computed across a batch of memories sampled from the replay buffer. The replay buffer sampling weights are updated based on the TD error from the samples.
        """
        indices, states, actions, next_states, rewards, success, weights = batch
        td_errors = self.get_td_error(states, actions, next_states, rewards, success)

        loss = self.compute_loss(td_errors, weights=weights)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "epsilon",
            self.epsilon,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if self.hparams.debug:
            # TODO(Issue#105): Move more deepcopy calls to logging
            # layer. Check if any deepcopy operations can be avoided.
            indices_copy = copy.deepcopy(indices)
            actions_copy = copy.deepcopy(actions)
            rewards_copy = copy.deepcopy(rewards)
            success_copy = copy.deepcopy(success)
            weights_copy = copy.deepcopy(weights)
            replay_buffer_state_counts_copy = sorted(
                [
                    (state_id, count)
                    for state_id, count in self.replay_buffer.state_histogram.items()
                ]
            )

            self._object_log_manager.log(
                log_entry.TRAINING_BATCH_LOG_ENTRY,
                log_entry.TrainingBatchLogEntry(
                    batch_idx=self.global_step,
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
                    replay_buffer_state_counts=replay_buffer_state_counts_copy,
                ),
            )

            if self.hparams.debug_td_error:
                # Log richer representation of td error for testing.
                frequent_states: list[torch.Tensor] = (
                    self._state_visit_logger.get_top_n(_FREQUENTLY_VISITED_STATE_COUNT)
                )
                # Sample all possible actions over the state space.
                actions = range(self.env.nA)

                # TODO(Issue#154): Combine cache entries to make a single batched call to get_td_error before logging.
                for frequent_state in frequent_states:
                    for cache_action in actions:
                        (
                            state,
                            action,
                            next_state,
                            reward,
                            environment_completion_status,
                        ) = self._state_action_cache.cache_get(
                            frequent_state.numpy(), cache_action
                        )

                        self._object_log_manager.log(
                            log_entry.TRAINING_HISTORY_TD_ERROR_LOG_ENTRY,
                            log_entry.TrainingHistoryTDErrorLogEntry(
                                batch_idx=self.global_step,
                                state_id=self._state_logger.get_logged_object_id(
                                    torch.tensor(state)
                                ),
                                action=action,
                                td_error=self.get_td_error(
                                    states=torch.tensor([state]),
                                    actions=torch.tensor([action]),
                                    next_states=torch.tensor([next_state]),
                                    rewards=torch.tensor([reward]),
                                    success=torch.tensor(
                                        [environment_completion_status]
                                    ),
                                ).item(),
                            ),
                        )
            else:
                # Revert to logging td error log entries the original way.
                for state, action, td_error in zip(
                    states, actions.tolist(), td_errors.tolist()
                ):
                    self._object_log_manager.log(
                        log_entry.TRAINING_HISTORY_TD_ERROR_LOG_ENTRY,
                        log_entry.TrainingHistoryTDErrorLogEntry(
                            batch_idx=self.global_step,
                            state_id=self._state_logger.get_logged_object_id(state),
                            action=action,
                            td_error=td_error,
                        ),
                    )

        if self.hparams.debug_action_inversion_checker:
            action_inversion_reports = self._action_inversion_checker.check(
                policy=self._validation_policy
            )
            self.log("action_inversion_incident_rate", len(action_inversion_reports))

            for report in action_inversion_reports:
                self._object_log_manager.log(
                    log_entry.ACTION_INVERSION_REPORT_ENTRY,
                    log_entry.ActionInversionReportEntry(
                        batch_idx=self.global_step,
                        state_id=self._state_logger.get_logged_object_id(report.state),
                        preferred_actions=report.preferred_actions,
                        policy_action=report.policy_action,
                    ),
                )

        # Update replay buffer.
        self.replay_buffer.update_priorities(indices, td_errors)
        self._update_beta()
        self._update_epsilon()
        return loss

    def validation_step(self, batch, batch_idx):
        """Runs a single validation step based on a policy."""
        success, rewards, steps = batch
        self.log("val_success", torch.sum(success) / len(batch))
        self.log("val_reward", torch.Tensor.float(rewards).mean())
        self.log("val_steps", torch.Tensor.float(steps).mean())

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
            # num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            ValidationBuilder(
                env=self._validation_env,
                initial_state=self._backward_algorithm_manager.state(),
                policy=self._validation_policy,
                episode_length=self.hparams.max_episode_length,
            ),
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
        )

    # TODO(arvind): Override hooks to load data appropriately for test
