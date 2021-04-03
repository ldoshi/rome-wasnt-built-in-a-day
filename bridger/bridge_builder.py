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
