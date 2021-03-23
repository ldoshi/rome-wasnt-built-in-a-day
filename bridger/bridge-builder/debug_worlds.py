# Training scenarios used for debugging.

import gym
import gym_bridges.envs
import numpy as np

import bridge_builder
import training_panel

environment_name = "gym_bridges.envs:Bridges-v0"

# Creates the smallest world where a bridge can be built. Cycles
# through a series of actions to build simple episodes into the
# training set. Inspecting the TrainingPanel should show that the
# value function reflects the experiences encountered.
def tiny_world():
    env = gym.make(environment_name)
    env.setup(3, 4)

    panel = training_panel.TrainingPanel(
        states_n=10,
        state_width=env.shape[1],
        state_height=env.shape[0],
        actions_n=env.action_space.n,
    )

    number_of_episodes = 100
    epsilon_policy = np.linspace(1, 0.05, number_of_episodes)

    training_config = bridge_builder.TrainingConfig(
        number_of_episodes=number_of_episodes,
        episode_length=5,
        training_frequency=1,
        memory_size=10000,
        update_bound=100,
        action_space_n=env.action_space.n,
        tau=0.01,
        batch_size=100,
        gamma=0.99,
        alpha=1,
        epsilon=None,
        epsilon_policy=epsilon_policy,
    )

    trainer = bridge_builder.Trainer(env, training_config)

    # Exercises various scenarios.
    iterations = 10
    for _ in range(iterations):
        # Only the left.
        for _ in range(training_config.episode_length):
            trainer.take_action(0)

        # Only the gap in the middle.
        for _ in range(training_config.episode_length):
            trainer.take_action(1)

        # Only the right.
        for _ in range(training_config.episode_length):
            trainer.take_action(2)

        # One on the left, then only the middle.
        trainer.take_action(0)
        for _ in range(training_config.episode_length):
            trainer.take_action(1)

        # One on the right, then only the middle.
        trainer.take_action(2)
        for _ in range(training_config.episode_length):
            trainer.take_action(1)

        # Complete a bridge left to right.
        trainer.take_action(0)
        trainer.take_action(2)

        # Complete a bridge right to left.
        trainer.take_action(2)
        trainer.take_action(0)

    panel.update_panel(trainer.training_history.get_history_by_visit_count())
    return env, trainer, panel


env, trainer, panel = tiny_world()
