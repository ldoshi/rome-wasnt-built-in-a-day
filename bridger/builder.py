import argparse
import gym
import gym_bridges.envs
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

from bridger import config, policies, qfunctions, replay_buffer


# TODO: Add hooks for TrainingHistory
# TODO: Add hooks for stepping through code at different scales
# TODO: Ensure that all states coming out of gym env are immediately converted to torch from numpy
# TODO: Ensure that all state handling in this package is done in torch
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

        self.Q = qfunctions.CNNQ(hparams.env_height, hparams.env_width, env.nA)
        self.target = qfunctions.CNNQ(hparams.env_height, hparams.env_width, env.nA)
        self.target.load_state_dict(self.Q.state_dict())

        self.policy = policies.EpsilonGreedyPolicy(self.Q)

        self.epsilon = hparams.epsilon_training_start
        self.make_memories()

    def make_memories(self):
        self.Q.freeze()
        for i in range(self.hparams.inter_training_episodes):
            for j in range(self.hparams.episode_length):
                next_state, reward, is_done = self()
                if is_done:
                    break
            self.update_epsilon()
            env.reset()
        self.Q.unfreeze()

    def forward(self):
        state, action = env.state, self.policy(env.state, epsilon=self.epsilon)
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
