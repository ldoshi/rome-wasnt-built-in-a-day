import abc

from collections.abc import Hashable
import functools
import gym
import itertools
import multiprocessing
import torch
from torch.distributions.categorical import Categorical
import torch.nn
import torch.nn.functional as F

from bridger import hash_utils

from typing import Any, Callable, Optional


def _update_target(tau: float, q: torch.nn.Module, target: torch.nn.Module) -> None:
    """Updates the target network weights based on the q network weights.

    The target network is updated using a weighted sum of its current
    weights and the q network weights to increase stability in
    training.

    """
    params = target.state_dict()
    update = q.state_dict()
    for param in params:
        params[param] += tau * (update[param] - params[param])

    target.load_state_dict(params)


class QManager(abc.ABC, torch.nn.Module):
    """Base class to interact with a q function and its target.

    The q and its target are coupled to ensure they have the same
    architecture and are updated appropriately.

    If a separate target is not being used, the q and target accessors
    evaluate the same underlying function.

    """

    @abc.abstractmethod
    def update_target(self) -> None:
        """Update the target function from the q function."""
        pass

    @property
    @abc.abstractmethod
    def q(self) -> torch.nn.Module:
        """Accessor to evaluate the q function."""
        pass

    @property
    @abc.abstractmethod
    def target(self) -> torch.nn.Module:
        """Accessor to evaluate the target function."""
        pass


class CNNQManager(QManager):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        num_actions: int,
        tau: Optional[float],
    ):
        """Manager implementing q and the target as CNNQs.

        Args:
          image_height: The env height.
          image_width: The env width.
          num_actions: The number of actions supported in the env.
          tau: The fraction of the step from target to q when the
            target is updated. A value of 1 means q replaces the
            target completely. If None, a separate target network is
            not used.

        """
        super(CNNQManager, self).__init__()

        self._tau = tau
        self._q = CNNQ(
            image_height=image_height, image_width=image_width, num_actions=num_actions
        )

        if self._tau is not None:
            self._target = CNNQ(
                image_height=image_height,
                image_width=image_width,
                num_actions=num_actions,
            )
            self._target.load_state_dict(self._q.state_dict())
        else:
            self._target = self._q

    def update_target(self) -> None:
        if self._tau is None:
            return

        _update_target(self._tau, self._q, self._target)

    @property
    def q(self) -> torch.nn.Module:
        return self._q

    @property
    def target(self) -> torch.nn.Module:
        return self._target

class CNNQ(torch.nn.Module):
    """Base class for CNN Q-function neural network module."""

    def __init__(self, image_height: int, image_width: int, num_actions: int):
        super(CNNQ, self).__init__()
        self.image_height = image_height
        self.image_width = image_width

        paddings = [1, 1]
        strides = [2, 1]
        kernel_sizes = [3, 3]
        channel_nums = [3, 4, 8]

        args_iter = zip(
            channel_nums[:-1], channel_nums[1:], kernel_sizes, strides, paddings
        )

        self.CNN = torch.nn.ModuleList([torch.nn.Conv2d(*args) for args in args_iter])
        H, W = self.image_height, self.image_width
        for padding, kernel_size, stride in zip(paddings, kernel_sizes, strides):
            H = int((H + 2 * padding - kernel_size) / stride) + 1
            W = int((W + 2 * padding - kernel_size) / stride) + 1
        C = channel_nums[-1]
        network_widths = [C * H * W, 64]
        args_iter = zip(network_widths[:-1], network_widths[1:])
        self.network = torch.nn.ModuleList([torch.nn.Linear(*args) for args in args_iter])

        critic_widths = [64, 1]
        args_iter = zip(critic_widths[:-1], critic_widths[1:])
        self.critic = torch.nn.ModuleList([torch.nn.Linear(*args) for args in args_iter])

        actor_widths = [64, num_actions]
        args_iter = zip(actor_widths[:-1], actor_widths[1:])
        self.actor = torch.nn.ModuleList([torch.nn.Linear(*args) for args in args_iter])


    def _forward_network(self, x):
        x = x.reshape(-1, self.image_height, self.image_width)
        x = encode_enum_state_to_channels(x, self.CNN[0].in_channels).float()
        for layer in self.CNN:
            x = torch.relu(layer(x))
        x = self.network[0](x.reshape(x.shape[0], -1))
        for layer in self.network[1:]:
            x = layer(torch.relu(x))
        return x
        

    # def get_value(self, x):
    #     x = self._forward_network(x)
    #     for layer in self.critic:
    #         x = layer(torch.relu(x))
    #     return x

    def get_action_and_value(self, x, action = None):
        x = self._forward_network(x)

        actor = x
        for layer in self.actor:
            actor = layer(torch.relu(actor))
            
        probs = Categorical(logits=actor)
        if action is None:
            action = probs.sample()

        critic = x
        for layer in self.critic:
            critic = layer(torch.relu(critic))
            
        return action, probs.log_prob(action), critic


def encode_enum_state_to_channels(state_tensor: torch.Tensor, num_channels: int):
    """Takes a 3-dim state tensor and returns a one-hot tensor with a new channels
    dimension as the second dimension (batch, channels, height, width)"""
    # Note: if memory-usage problems, consider alternatives to int64 tensor
    x = F.one_hot(state_tensor.long(), num_channels)
    return x.permute(0, 3, 1, 2)


class TabularQManager(QManager):
    def __init__(
        self,
        env: gym.Env,
        brick_count: int,
        tau: Optional[float],
    ):
        """Manager implementing q and the target as TabularQs.

        Args:
          env: A gym for executing population strategies.
          brick_count: The number of bricks to place as a tool for
            enumerating reachable states.
          tau: The fraction of the step from target to q when the
            target is updated. A value of 1 means q replaces the
            target completely. If None, a separate target network is
            not used.

        """
        super(TabularQManager, self).__init__()

        self._tau = tau
        self._q = TabularQ(env=env, brick_count=brick_count)

        if self._tau is not None:
            self._target = TabularQ(env=env, brick_count=brick_count)
            self._target.load_state_dict(self._q.state_dict())
        else:
            self._target = self._q

    def update_target(self) -> None:
        if self._tau is None:
            return

        _update_target(self._tau, self._q, self._target)

    @property
    def q(self) -> torch.nn.Module:
        return self._q

    @property
    def target(self) -> torch.nn.Module:
        return self._target


def _collect_parameters(
    env: gym.Env,
    hash_fn: Callable[[Any], str],
    remaining_brick_count: int,
    initial_actions: tuple[int],
) -> set[str]:
    state_hashes = set()
    for remaining_actions in itertools.product(
        range(env.nA), repeat=remaining_brick_count
    ):
        episode_actions = initial_actions + remaining_actions

        state = env.reset()
        for action in episode_actions:
            next_state, _, done, _ = env.step(action)
            state_hashes.add(hash_fn(next_state))
            if done:
                break
            state = next_state

    return state_hashes


class TabularQ(torch.nn.Module):
    """A Q-function based on a look-up table.

    This implementation is intended for debugging in simpler
    scenarios.  For example, we can eliminate any effects from the
    state function approximator affecting the efficacy of learning.

    Limitations:
      Not only does the tabular method only work if we can reasonably
      enumerate all the states, but also we *must* enumerate any
      states that will be encountered up front due to the requirement
      of registering all potential parameters with the optimizer at
      the beginning.

    """

    def __init__(
        self,
        env: gym.Env,
        brick_count: int,
        hash_fn: Callable[[Any], Hashable] = hash_utils.hash_tensor,
    ):
        """Initializes all the states that this instance can handle.

        Args:
          env: A gym for executing population strategies
          brick_count: The number of bricks to place as a tool for
            enumerating reachable states.

        """

        super(TabularQ, self).__init__()
        self._q = torch.nn.ParameterDict()
        self._hash_fn = hash_fn
        self._state_dimensions = len(env.reset().shape)

        # Assigning a non-trivial chunk of work per
        # _collect_parameters call. Adjust later if needed.
        remaining_brick_count = min(5, brick_count - 1)
        initial_brick_count = brick_count - remaining_brick_count

        _collect_parameters_function = functools.partial(
            _collect_parameters, env, self._internal_hash, remaining_brick_count
        )

        state_hashes = set()
        state_hashes.add(self._internal_hash(env.reset()))

        with multiprocessing.Pool() as pool:
            for hashes in pool.map(
                _collect_parameters_function,
                itertools.product(range(env.nA), repeat=initial_brick_count),
            ):
                state_hashes.update(hashes)

        for state_hash in state_hashes:
            self._q[state_hash] = torch.nn.Parameter(
                torch.rand(env.nA, requires_grad=True)
            )

    def _internal_hash(self, x) -> str:
        """Hashes states per component requirements.

        ParameterDict does not allow non-str as dict keys. The
        self._hash_fn is used first to make the treatment of ndarray
        and tensors consistent.
        """

        return str(self._hash_fn(x))

    def forward(self, x):
        # The tensor must be converted to int to match the state hash
        # keys. Additionally, "." is not allowed in ParameterDict keys
        # so we cannot use float. State cell values are defined as ints
        # anyway.

        if len(x.shape) == self._state_dimensions:
            return torch.clone(self._q[self._internal_hash(x.int())])

        return torch.stack([self._q[self._internal_hash(state.int())] for state in x])


# This architecture has not yet been validated (and is likely poor).
choices = {"default": CNNQ, "cnn": CNNQ, "tabular": TabularQ}
