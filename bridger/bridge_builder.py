# Current usage notes:
#
# This file is currently designed to be used interactively. Just running the file won't do anything interesting.
#
# All constants in TrainingConfig have not yet been debugged or validated (changes likely needed).
# The NeworkQ architecture has not yet been debugged or validated (and is likely poor).
#
# Trainer offers three tiers of functions to use interactively (and interchangably):
# 1. train(...) will run the training for a requested number of episodes, or the one providing in the TrainingConfig.
# 2. follow_policy(...) will run a requested number of steps or through the end of a single episode.
# 3. take_action(...) will take the requested action.
#
# The update_models function may be called manually if desired, and is also called at training_frequency by follow_policy.
#
# Use demo(...) to see how the policy performs.

import gym
import gym_bridges.envs

import logging
import numpy as np
import os
import datetime
import random
import sys
import pdb
import matplotlib
import tensorflow as tf
from tensorflow.keras import layers
from pandas import DataFrame
import time
from collections import deque, namedtuple
from replay_buffer import ReplayBuffer

import policies
import training_history
import training_panel

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def reload_modules():
    from importlib import reload

    reload(training_history)
    reload(training_panel)


class NetworkQ:
    def __init__(self, input_shape, action_space_n, tau):
        self.model = tf.keras.models.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(
                4,
                3,
                strides=(2, 2),
                padding="same",
                activation="relu",
                input_shape=input_shape,
            )
        )
        self.model.add(
            tf.keras.layers.Conv2D(
                8,
                2,
                strides=(1, 1),
                padding="same",
                activation="relu",
                input_shape=input_shape,
            )
        )
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(action_space_n))

        self.model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        self._tau = tau

    @property
    def weights(self):
        return self.model.get_weights()

    def initialize_weights(self, weights):
        self.model.set_weights(weights)

    def set_weights(self, weights):
        current = self.weights
        self.model.set_weights(
            [
                (1 - self._tau) * current[i] + self._tau * weights[i]
                for i in range(len(weights))
            ]
        )

    def update(self, states, q_values, q_loss=None, epochs=1):
        # TODO(lyric): Revisit tensorboard_callback and/or q_loss.
        #      history = self.model.fit(np.expand_dims(states,axis=3), q_values, epochs=epochs, verbose=0, callbacks=[tensorboard_callback])
        history = self.model.fit(
            np.expand_dims(states, axis=3), q_values, epochs=epochs, verbose=0
        )
        if q_loss:
            q_loss(history.history["loss"][-1])

    def predict(self, states):
        return self.model.predict(np.expand_dims(states, axis=3))

    def __call__(self, state):
        return self.predict([state])


# TODO(lyric): Make training history optional in the future to reduce overhead. Consider adding a "debug" config.
class Trainer:
    def __init__(self, env, training_config, summary_writer=None):
        self._env = env
        self._training_state = TrainingState(training_config)
        self._training_history = training_history.TrainingHistory()
        self._replay_buffer = ReplayBuffer(
            capacity=training_config.replay_buffer_capacity,
            alpha=training_config.replay_buffer_alpha,
        )
        self._summary_writer = summary_writer
        self._q = NetworkQ(
            env.reset().shape + (1,),
            training_config.action_space_n,
            training_config.tau,
        )
        self._q_target = NetworkQ(
            env.reset().shape + (1,),
            training_config.action_space_n,
            training_config.tau,
        )
        self._q_target.initialize_weights(self._q.weights)
        # TODO(lyric): Consider moving the policy to training_config if multiple are supported.
        self._policy = policies.EpsilonGreedyPolicy(self._q)

    @property
    def q(self):
        return self._q

    @property
    def training_state(self):
        return self._training_state

    @property
    def training_history(self):
        return self._training_history

    # Train through the requested number of episodes.
    def train(self, number_of_episodes=None):
        if number_of_episodes is None:
            number_of_episodes = self._training_state.training_config.number_of_episodes

        for _ in range(number_of_episodes):
            self.follow_policy()

    # Take the provided action n times.
    def take_action(self, action, n=1, training_frequency=None, verbose=False):
        if training_frequency is None:
            training_frequency = self._training_state.training_config.training_frequency

        self._start_episode_if_necessary()

        for _ in range(n):
            next_state, reward, is_done, _ = self._env.step(action)
            if verbose:
                self._env.render()

            self._replay_buffer.add_new_experience(
                self._training_state.current_state, action, next_state, reward, is_done
            )
            self._training_state.current_state = next_state
            self._training_state.increment_current_episode_step()

            if not is_done:
                self._training_history.increment_visit_count(next_state)

            if self._training_state.current_epoch % training_frequency == 0:
                self.update_models()

            if is_done or not self._training_state.episode_active:
                self._training_state.end_episode()
                break

    # Follow the policy for the requested number of steps. Returns early if the episode ends. Starts a new episode if one is not already active. If steps_n is None, go until the episode ends.
    def follow_policy(self, steps_n=None, training_frequency=None, verbose=False):
        self._start_episode_if_necessary()

        step_counter = 0
        while self._should_take_step(step_counter, steps_n):
            action = self._policy(
                self._training_state.current_state,
                epsilon=self._training_state.current_epsilon,
            )
            self.take_action(
                action, training_frequency=training_frequency, verbose=verbose
            )
            step_counter += 1

    def _should_take_step(self, step_counter, steps_n):
        if not self._training_state.episode_active:
            return False

        if steps_n:
            return step_counter < steps_n

        return True

    def _start_episode_if_necessary(self):
        if self._training_state.episode_active:
            return

        self._training_state.start_episode()
        self._training_state.current_state = self._env.reset()

    def update_models(self):
        (
            indices,
            states,
            actions,
            next_states,
            rewards,
            is_dones,
            weights,
        ) = self._replay_buffer.sample(
            self._training_state.training_config.batch_size,
            self._training_state.current_replay_buffer_beta,
        )

        q_values_next = self._q.predict(next_states)
        next_actions = np.argmax(q_values_next, axis=1)
        td_targets = (
            rewards
            + (1 - is_dones)
            * self._training_state.training_config.gamma
            * self._q_target.predict(next_states)[np.arange(len(states)), next_actions]
        )

        q_values_now = self._q.predict(states)
        td_errors = td_targets - q_values_now[np.arange(len(actions)), actions]
        self._replay_buffer.update_priorities(indices, td_errors)
        self._log_td_error_to_training_history_debug(states, actions, td_errors)

        weighted_td_errors = weights * td_errors
        q_values_now[
            np.arange(len(actions)), actions
        ] += self._training_state.training_config.alpha * np.clip(
            weighted_td_errors,
            -self._training_state.training_config.update_bound,
            self._training_state.training_config.update_bound,
        )

        self._q.update(states, q_values_now)
        self._q_target.set_weights(self._q.weights)

        self._log_q_values_to_training_history_debug()

    # For debuging only. Averages the td error per (state, action) pair.
    def _log_td_error_to_training_history_debug(self, states, actions, td_errors):
        for i in range(len(states)):
            self._training_history.add_td_error(
                states[i], actions[i], self._training_state.current_epoch, td_errors[i]
            )

    # For debuging only. Computes q and q_target values across most visited 100 states and all actions.
    def _log_q_values_to_training_history_debug(self):
        visited_state_histories = self._training_history.get_history_by_visit_count(100)
        states = np.array(
            [
                visited_state_history.state
                for visited_state_history in visited_state_histories
            ]
        )

        q_values = self._q.predict(states)
        q_target_values = self._q_target.predict(states)

        for i in range(len(states)):
            self._training_history.add_q_values(
                states[i],
                self._training_state.current_epoch,
                q_values[i],
                q_target_values[i],
            )


class TrainingConfig:
    # If provided, the epsilon policy should be an array of length number_of_episodes with the epsilon value to use for each episode.
    # At least one of epsilon and epsilon_policy should be provided.
    def __init__(
        self,
        number_of_episodes,
        episode_length,
        training_frequency,
        replay_buffer_capacity,
        replay_buffer_alpha,
        replay_buffer_beta_policy,
        update_bound,
        action_space_n,
        tau,
        batch_size,
        gamma,
        alpha,
        epsilon=None,
        epsilon_policy=None,
    ):
        self.number_of_episodes = number_of_episodes
        self.episode_length = episode_length
        self.training_frequency = training_frequency
        self.replay_buffer_capacity = replay_buffer_capacity
        self.replay_buffer_alpha = replay_buffer_alpha
        self.replay_buffer_beta_policy = replay_buffer_beta_policy
        self.update_bound = update_bound
        self.action_space_n = action_space_n
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.set_epsilon_policy(epsilon_policy)

    def set_epsilon(self, epsilon):
        if epsilon is None:
            assert self._epsilon_policy

        self.epsilon = epsilon

    def set_epsilon_policy(self, epsilon_policy):
        if epsilon_policy is None:
            assert self.epsilon
        else:
            assert len(epsilon_policy) == self.number_of_episodes

        self.epsilon_policy = epsilon_policy


class TrainingState:
    def __init__(self, training_config):
        self.training_config = training_config
        self._current_epoch = 0
        self._current_episode = 0
        self._current_episode_step = self.training_config.episode_length
        self.current_state = None

    @property
    def current_epoch(self):
        return self._current_epoch

    def _increment_current_epoch(self):
        self._current_epoch += 1

    @property
    def current_episode(self):
        return self._current_episode

    def _increment_current_episode(self):
        self._current_episode += 1

    def increment_current_episode_step(self):
        self._current_episode_step += 1
        self._increment_current_epoch()

    # Follow the epsilon_policy, unless epsilon is provided as an override.
    @property
    def current_epsilon(self):
        if self.training_config.epsilon:
            return self.training_config.epsilon

        # Use the last value of the epsilon policy for additional calls.
        epsilon_policy_index = self._current_episode
        if self._current_episode >= len(self.training_config.epsilon_policy):
            epsilon_policy_index = -1

        return self.training_config.epsilon_policy[epsilon_policy_index]

    # Follow the replay_buffer_beta policy.
    @property
    def current_replay_buffer_beta(self):
        # Use the last value of the replay_buffer_beta policy for additional calls.
        replay_buffer_beta_policy_index = self._current_episode
        if self._current_episode >= len(self.training_config.epsilon_policy):
            replay_buffer_beta_policy_index = -1

        return self.training_config.replay_buffer_beta_policy[
            replay_buffer_beta_policy_index
        ]

    @property
    def episode_active(self):
        return self._current_episode_step < self.training_config.episode_length

    def start_episode(self):
        self._current_episode_step = 0

    def end_episode(self):
        self._current_episode_step = self.training_config.episode_length
        self._increment_current_episode()


def demo(env, estimator, episode_length):
    state = env.reset()
    for t in range(episode_length):
        # TODO(lyric): Currently hard-coded a policy to choose best action based on action-value function.
        action = np.argmax(estimator.predict([state]))
        next_state, reward, is_done, _ = env.step(action)
        env.render()
        state = next_state

        if is_done:
            print("finished at %d" % t)
            break


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
        if not states and not actions and not rewards:
            return out

        for entry in self._replay_buffer._content:
            match = True
            if match and states and entry[0] not in states:
                match = False
            if match and actions and entry[1] not in actions:
                match = False
            if match and rewards and entry[3] not in rewards:
                match = False

            if match:
                out.append(entry)

        return out


environment_name = "gym_bridges.envs:Bridges-v0"
env = gym.make(environment_name)
env.setup(4, 6, vary_heights=True)

number_of_episodes = 1000
epsilon_policy = np.linspace(1, 0.05, number_of_episodes)
replay_buffer_beta_policy = np.linspace(0.5, 1, number_of_episodes)

training_config = TrainingConfig(
    number_of_episodes=number_of_episodes,
    episode_length=10,
    training_frequency=1,
    replay_buffer_capacity=10000,
    replay_buffer_alpha=0.5,
    replay_buffer_beta_policy=replay_buffer_beta_policy,
    update_bound=1,
    action_space_n=env.action_space.n,
    tau=0.01,
    batch_size=100,
    gamma=0.99,
    alpha=1,
    epsilon=None,
    epsilon_policy=epsilon_policy,
)

trainer = Trainer(env, training_config)

# Uncomment to view the TrainingPanel.
panel = training_panel.TrainingPanel(
    states_n=20,
    state_width=env.shape[1],
    state_height=env.shape[0],
    actions_n=env.action_space.n,
)

# Uncomment if you want to run training using the training_config as is. It may or may not work out!
for _ in range(int(number_of_episodes / 5)):
    trainer.train(5)

    # Uncomment to update the TrainingPanel.
    panel.update_panel(trainer.training_history.get_history_by_visit_count())

    # Uncomment to run the policy in the environment.
    demo(env, trainer.q, 50)
