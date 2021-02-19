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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict([observation])
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

class Memory: 
    size = 0
    content = None
    index = 0
    
    def __init__(self, n):                                
        self.size = n              
        self.content = []          
  
    def add(self, s, a, ss, r, d): 
        if len(self.content) == self.size:    
            self.content[self.index] = [s, a, ss, r, d]
            self.index = (self.index + 1) % self.size
        else:                      
            self.content.append([s, a, ss, r, d])
  
    def sample(self, n):           
      samples = random.sample(self.content, min(n, len(self.content)))
  
      states = np.array([i[0] for i in samples]) 
      actions = np.array([i[1] for i in samples])
      next_states = np.array([i[2] for i in samples])
      rewards = np.array([[i[3]] for i in samples])
      is_dones = np.array([[i[4]] for i in samples])
      return states, actions, next_states, rewards, is_dones


class NetworkQ:                           
  model = None                     
  tau = None                    
                                   
  def __init__(self,input_shape, action_space_n, tau):
    
      
    self.model = tf.keras.models.Sequential()
    self.model.add(tf.keras.layers.Conv2D(4,3,strides=(2,2),padding="same",activation="relu",input_shape=input_shape))
    self.model.add(tf.keras.layers.Conv2D(8,2,strides=(1,1),padding="same",activation="relu",input_shape=input_shape))
    self.model.add(tf.keras.layers.Flatten())
    self.model.add(tf.keras.layers.Dense(64, activation='relu'))
    self.model.add(tf.keras.layers.Dense(action_space_n))
  
    self.model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])        
    self.tau = tau           
  
  def get_model(self):             
    return self.model;             
  
  def get_weights(self):           
    return self.model.get_weights()
  
  def initialize_weights(self, weights):      
    self.model.set_weights(weights)
  
  def set_weights(self, weights):  
    current = self.get_weights()   
    self.model.set_weights([current[i] * self.tau + (1 - self.tau) * weights[i] for i in range(len(weights))])             
  
  def update(self, states, q_values, q_loss=None, epochs=1):
      # TODO(lyric): Revisit tensorboard_callback and/or q_loss.
      #      history = self.model.fit(np.expand_dims(states,axis=3), q_values, epochs=epochs, verbose=0, callbacks=[tensorboard_callback])
      history = self.model.fit(np.expand_dims(states,axis=3), q_values, epochs=epochs, verbose=0)
      if q_loss:
        q_loss(history.history['loss'][-1])
  
  def predict(self, states):
    return self.model.predict(np.expand_dims(states, axis=3))
  

# TODO(lyric): Add TrainingHistory and relevant updates (and maybe loss).    
class Trainer:
    def __init__(self, env, training_config, summary_writer=None): 
        self._env = env
        self._training_state = TrainingState(training_config)
        self._memory = Memory(training_config.memory_size)
        self._summary_writer = summary_writer
        self._q = NetworkQ(env.reset().shape + (1,), training_config.action_space_n, training_config.tau)
        self._q_target = NetworkQ(env.reset().shape + (1,), training_config.action_space_n, training_config.tau)
        self._q_target.initialize_weights(self._q.get_weights())
        # TODO(lyric): Consider moving the policy to training_config if multiple are supported.
        self._policy = make_epsilon_greedy_policy(self._q, training_config.action_space_n)

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
    def take_action(self, action, n=1, verbose=False):
        self._start_episode_if_necessary()
        
        for _ in range(n):
            next_state, reward, is_done, _ = self._env.step(action)
            if verbose:
                self._env.render()
            self._memory.add(self._training_state.current_state, action, next_state, reward, is_done)
            self._training_state.current_state = next_state
            self._training_state.increment_current_episode_step()

            if not self._training_state.episode_active or is_done:
                self._training_state.end_episode()
                break

    # Follow the policy for the requested number of steps. Returns early if the episode ends. Starts a new episode if one is not already active. If steps_n is None, go until the episode ends.
    def follow_policy(self, steps_n=None, training_frequency=None,verbose=False):
        if training_frequency is None:
            training_frequency = self._training_state.training_config.training_frequency
        
        self._start_episode_if_necessary()

        step_counter = 0
        while self._should_take_step(step_counter, steps_n):
            action_probabilities = self._policy(self._training_state.current_state, self._training_state.current_epsilon)
            action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
            self.take_action(action,verbose=verbose)
            
            if self._training_state.current_epoch % training_frequency == 0:
                self.update_models()

            step_counter +=1         

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
        states, actions, next_states, rewards, is_dones = self._memory.sample(self._training_state.training_config.batch_size)    
    
        q_values_now = self._q.predict(states)
        td_targets = rewards[:,0] + (1-is_dones[:,0]) * self._training_state.training_config.gamma * self._q_target.predict(next_states).max(axis=1)   

        q_values_now[np.arange(len(actions)),actions] += self._training_state.training_config.alpha * np.clip(td_targets - q_values_now[np.arange(len(actions)),actions], -self._training_state.training_config.update_bound, self._training_state.training_config.update_bound)
    
        self._q.update(states, q_values_now) 
        self._q_target.set_weights(self._q.get_weights())     


class TrainingConfig:
    # If provided, the epsilon policy should be an array of length number_of_episodes with the epsilon value to use for each episode.
    # At least one of epsilon and epsilon_policy should be provided.
    def __init__(self, number_of_episodes, episode_length, training_frequency, memory_size, update_bound, action_space_n, tau, batch_size, gamma, alpha, epsilon=None, epsilon_policy=None):
        self.number_of_episodes = number_of_episodes
        self.episode_length = episode_length
        self.training_frequency = training_frequency
        self.memory_size = memory_size
        self.update_bound = update_bound
        self.action_space_n = action_space_n
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        assert epsilon is not None or epsilon_policy is not None
        self.epsilon = epsilon
        self.set_epsilon_policy(epsilon_policy)

    def set_epsilon(self, epsilon):
        if epsilon is None:
            assert self._epsilon_policy
            
        self.epsilon = epsilon

    def set_epsilon_policy(self, epsilon_policy):
        if epsilon_policy is None:
            assert self._epsilon

        if epsilon_policy is not None:
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

environment_name = "gym_bridges.envs:Bridges-v0"
env = gym.make(environment_name)
env.setup(5,6)

number_of_episodes=100
epsilon_policy = np.linspace(1, .05, number_of_episodes)

training_config = TrainingConfig(
    number_of_episodes=number_of_episodes,
    episode_length=40,
    training_frequency=1,
    memory_size=10000,
    update_bound=50,
    action_space_n=env.action_space.n,
    tau=.5,
    batch_size=100,
    gamma=.99,
    alpha=.5,
    epsilon=None,
    epsilon_policy=epsilon_policy)

trainer = Trainer(env, training_config)

# Uncomment if you want to run training using the training_config as is. It may or may not work out!
# trainer.train()

demo(env, trainer.q, 50)
