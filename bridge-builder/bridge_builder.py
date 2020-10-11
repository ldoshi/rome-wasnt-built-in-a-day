import gym
import gym_bridges.envs

import numpy as np
import os
import datetime
import random
import sys
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.kernel_approximation
import pdb
import matplotlib
import tensorflow as tf
from tensorflow.keras import layers
from pandas import DataFrame
import time                   
from collections import deque, namedtuple                     
                   
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
      dones = np.array([[i[4]] for i in samples])
      return states, actions, next_states, rewards, dones

class Q:                           
  model = None                     
  polyak = None                    
                                   
  def __init__(self,input_shape, action_space, polyak):     
    self.model = tf.keras.models.Sequential()
    self.model.add(tf.keras.layers.Conv2D(4,3,strides=(2,2),padding="valid",activation="relu",input_shape=input_shape))
    self.model.add(tf.keras.layers.Conv2D(8,2,strides=(1,1),padding="valid",activation="relu",input_shape=input_shape))
    self.model.add(tf.keras.layers.Flatten())
    self.model.add(tf.keras.layers.Dense(64, activation='relu'))
    self.model.add(tf.keras.layers.Dense(action_space))
  
    self.model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])        
    self.polyak = polyak           
  
  def get_model(self):             
    return self.model;             
  
  def get_weights(self):           
    return self.model.get_weights()
  
  def initialize_weights(self, weights):      
    self.model.set_weights(weights)
  
  def set_weights(self, weights):  
    current = self.get_weights()   
    self.model.set_weights([current[i] * self.polyak + (1 - self.polyak) * weights[i] for i in range(len(weights))])             
  
  def update(self, states, ys, q_loss=None, epochs=1):                                
      history = self.model.fit(states, ys, epochs=epochs, verbose=0, callbacks=[tensorboard_callback])
      if q_loss:
        q_loss(history.history['loss'][-1])
  
  def predict(self, states):
    return self.model.predict(np.expand_dims(states, axis=3))
  
def update_models(memory, q, q_target, q_loss, update_bound):
    BATCH_SIZE = 1000             
    gamma = .99                   
    states, actions, next_states, rewards, dones = memory.sample(BATCH_SIZE)    

    q_values_now = q.predict(states)
    td_targets = rewards + (1-dones) * gamma * q_target.predict(next_states).max(axis=1)   
#    pdb.set_trace()
    q_values_now[np.arange(len(actions)),actions] += np.clip(td_targets - q_values_now[np.arange(len(actions)),actions], -update_bound, update_bound)
    
    q.update(states, q_values_now, q_loss) 
    q_target.set_weights(q.get_weights())     

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


def train(iterations, episode_length, train_frequency, memory, q, q_target, q_loss,update_bound): 
    current_epoch = 1             
    for iteration in range(iterations):      
        print("ITERATION %d" % iteration)

        policy = make_epsilon_greedy_policy(estimator, env.action_space.n)
        
        state = env.reset()       
        for t in range(episode_length):
            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, reward, done, _ = env.step(action)
            memory.add(state, action, next_state, reward, done)
            print(state)
            print(next_state)
            print(action)
            print(reward)
 
            state = next_state    
 
            if current_epoch % train_frequency == 0:
                update_models(memory, q, q_target, q_loss, update_bound)
                with summary_writer.as_default():
                  tf.summary.scalar('q_loss', q_loss.result(), step=current_epoch)
            if done:              
                print("finished!")
                break             

            current_epoch += 1    
 
            if current_epoch % 1500 == 0:
              demo(env,episode_length)
             
def demo(env, estimator, episode_length):
    state = env.reset()           
    for t in range(episode_length):
        action = np.argmax(estimator.predict([state]))
        next_state, reward, done, _ = env.step(action)
 
        env.render()              
        state = next_state        
 
        if done:                  
            print("finished at %d" % t)
            break

environment_name = "gym_bridges.envs:Bridges-v0"
env = gym.make(environment_name)

STATE_SPACE_N = len(env.reset())  
ACTION_SPACE_N = len(env.action_space.sample())    
UPDATE_BOUND = 50

iterations = 100                
train_frequency = 10             
episode_length = 400              
 
memory = Memory(10000)            
q = Q(input_shape=[STATE_SPACE_N + ACTION_SPACE_N],polyak=.99)
q_target = Q(input_shape=[STATE_SPACE_N + ACTION_SPACE_N],polyak=.99)            
q_target.initialize_weights(q.get_weights()) 
                         
# Tensorboard Related             
log_dir="log/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)    
 
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'log/bridge_builder/' + current_time + '/'
summary_writer = tf.summary.create_file_writer(log_dir)
 
# Define our metrics              
q_loss = tf.keras.metrics.Mean('q_loss', dtype=tf.float32)
 
train(iterations=iterations, episode_length=episode_length, train_frequency=train_frequency, memory=memory, q=q, q_target=q_target, q_loss=q_loss, update_bound=UPDATE_BOUND)


demo(env, 500, pi)




