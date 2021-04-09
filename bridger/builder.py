import argparse
import IPython
import gym
import gym_bridges.envs
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

from bridger import config, policies, qfunctions, replay_buffer


# TODO: Add hooks for TrainingHistory
class BridgeBuilder(pl.LightningModule):
    def __init__(self, hparams):
        """Constructor for the BridgeBuilder Module

        Args:
            hparams: a Namespace object, of the kind returned by an argparse
                     ArgumentParser. For more details, see
                     #get_hyperparam_parser"""

        super(BridgeBuilder, self).__init__()

        self.hparams = hparams
        torch.manual_seed(hparams.seed)

        self.env = gym.make(hparams.env_name)
        self.env.setup(
            hparams.env_height, hparams.env_width, vary_heights=hparams.env_vary_heights
        )

        self.replay_buffer = replay_buffer.ReplayBuffer(
            capacity=hparams.capacity,
            alpha=hparams.alpha,
            beta=hparams.beta_training_start,
            batch_size=hparams.batch_size,
        )

        self.Q = qfunctions.CNNQ(hparams.env_height, hparams.env_width, self.env.nA)
        self.target = qfunctions.CNNQ(
            hparams.env_height, hparams.env_width, self.env.nA
        )
        self.target.load_state_dict(self.Q.state_dict())

        self.policy = policies.EpsilonGreedyPolicy(self.Q)

        self.epsilon = hparams.epsilon_training_start
        self.memories = self.memory_generator()

        self.next_action = None
        self.breakpoint = {"step": 0, "episode": 0}

    def on_train_epoch_start(self):
        self.make_memories()

    def on_train_batch_end(outputs, batch, batch_idx, dataloader_idx):
        params = self.target.state_dict()
        update = self.Q.state_dict()
        for param in params:
            params[param] += self.hparams.tau * (update[param] - params[param])
        self.target.load_state_dict(params)
        self.make_memories()

    def make_memories(self):
        with torch.no_grad():
            for i in range(self.hparams.inter_training_steps):
                next(self.memories)

    def memory_generator(self):
        """A generator that serves up sequential transitions experienced by the
        agent. When an episode ends, a new one starts immediately. Each item
        yielded is a tuple with the following elements (in order):

         episode_idx: starting from 0, incremented every time an episode ends
                      and another begins
         step_idx:    starting from 0, incremented with each step,
                      irrespective of the episode it is in
         start_state: the state at the beginning of the transition
         action:      the action taken during the transition
         end_state:   the state at the end of the transition
         reward:      the reward gained through the transition
         finished:    whether the transition marked the end of the episode"""

        episode_idx = 0
        total_step_idx = 0
        while True:
            for step_idx in range(self.hparams.episode_length):
                self.checkpoint({"episode": episode_idx, "step": total_step_idx})
                memory = self()  # start_state, action, end_state, reward, finished
                yield (episode_idx, step_idx, *memory)
                total_step_idx += 1
                if memory[-1]:
                    break
            self.update_epsilon()
            self.env.reset()
            episode_idx += 1

    def checkpoint(self, thresholds):
        """A checkpointer that compares instance-state breakpoints to method
        inputs to determine whether to enter a breakpoint. This only runs while
        interactive mode is enabled.

        Args:
         thresholds: a dict mapping some subset of 'episode' and 'step' to
                     the current corresponding indices (as tracked by
                     `memory_generator#`)

        When these current-state thresholds reach or exceed corresponding
        values in the instance variable `breakpoint`, a breakpoint is entered
        (via `IPython.embed#`). This breakpoint will reoccur immediately and
        repeatedly, even as the user manually exits the IPython shell, until
        self.breakpoint has been updated"""
        while self.hparams.interactive_mode:
            if all(self.breakpoint[k] > v for k, v in thresholds.items()):
                break  # Don't stop for a checkpoint
            self.breakpoint = thresholds
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
        # Updates self.breakpoint for use in self.checkpoint#
        self.breakpoint["episode"] += 1  # Run until current episode ends OR
        self.breakpoint["step"] += repetitions  # for `repetitions` steps
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
        # Updates self.breakpoint for use in self.checkpoint#
        if num_actions is None:
            self.breakpoint["step"] = np.inf  # Run indefinitely until ...
        else:
            assert num_episodes == 1
            self.breakpoint["step"] += num_actions  # Take `num_actions` steps
        # ... `num_episodes` episodes have completed
        self.breakpoint["episode"] += num_episodes
        IPython.core.getipython.get_ipython().exiter()

    def return_to_training(self):
        self.disable_interactive_mode()
        IPython.core.getipython.get_ipython().exiter()

    def forward(self):
        state = self.env.state
        if self.hparams.interactive_mode and self.next_action is not None:
            action = self.next_action
        else:
            action = self.policy(torch.tensor(state), epsilon=self.epsilon)
        result = (state, action, *self.env.step(action)[:3])
        self.replay_buffer.add_new_experience(*result)
        return result

    def update_epsilon(self):
        if self.hparams.epsilon_decay_rule == "arithmetic":
            self.epsilon -= self.hparams.epsilon_decay_rate
        elif self.hparams.epsilon_decay_rule == "geometric":
            self.epsilon /= self.hparams.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.hparams.epsilon)

    def update_beta(self):
        if self.hparams.beta_growth_rule == "arithmetic":
            self.replay_buffer.beta += self.hparams.beta_growth_rate
        elif self.hparams.beta_growth_rule == "geometric":
            self.replay_buffer.beta *= self.hparams.beta_growth_rate
        self.replay_buffer.beta = min(
            self.replay_buffer.beta, self.hparams.beta_training_end
        )

    def get_td_error(self, states, actions, next_states, rewards, finished):

        row_idx = torch.arange(self.env.nA)
        qvals = self.Q(states)[row_idx, actions]
        with torch.no_grad():
            next_actions = self.Q(next_states).argmax(dim=1)
            next_vals = self.target(next_states)[row_idx, next_actions]
            expected_qvals = rewards + (~finished) * self.hparams.gamma * next_vals
        return torch.abs(expected_qvals - qvals)

    def compute_loss(self, td_errors, weights=None):
        if weights:
            td_errors = weights * td_errors
        # TODO: Change design to clip the gradient rather than the loss
        return (td_errors.clip(max=self.hparams.update_bound) ** 2).mean()

    def training_step(self, batch, batch_idx):
        indices, states, actions, next_states, rewards, finished, weights = batch
        td_errors = self.get_td_error(states, actions, next_states, rewards, finished)
        loss = self.compute_loss(td_errors, weights=weights)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Update replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        self.update_beta()

        return loss

    # TODO: Override hooks to compute non-TD-error metris for val and test

    def configure_optimizers(self):
        # TODO: This should work, but should we say Q.parameters(), or is
        # that limiting for the future?
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.replay_buffer, batch_size=self.hparams.batch_size)

    # TODO: Override hooks to load data appropriately for val and test

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
        return DeepDiagnoser(argparse.Namespace(**hparams))
