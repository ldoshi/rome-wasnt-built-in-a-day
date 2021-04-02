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
        self.memGen = self.memory_generator()

        self.next_action = None
        self.checkpoint = {"step": 0, "episode": 0}

    def on_train_batch_start(self):
        with torch.no_grad():
            for i in range(self.hparams.inter_training_steps):
                episode_idx, step_idx, *_ = next(self.memGen)

    def memory_generator(self):
        episode_idx = 0
        total_step_idx = 0
        while True:
            for step_idx in range(self.hparams.episode_length):
                self.checkpoint({"episode": episode_idx, "step": total_step_idx})
                yield (episode_idx, step_idx, *(self()))
                total_step_idx += 1
                if finished:
                    break
            self.update_epsilon()
            self.env.reset()
            episode_idx += 1

    def checkpoint(self, thresholds):
        while self.hparams.interactive_mode:
            if all(self.checkpoint[k] > v for k, v in thresholds.items()):
                break  # Don't stop for a checkpoint
            self.checkpoint = thresholds
            self.next_action = None
            IPython.embed()

    def enable_interactive_mode(self):
        self.hparams.interactive_mode = True

    def disable_interactive_mode(self):
        self.hparams.interactive_mode = False

    def take_action(self, action, repetitions=1):
        self.next_action = action
        self.checkpoint["episode"] += 1
        self.checkpoint["step"] += repetitions
        IPython.core.getipython.get_ipython().exiter()

    def follow_policy(self, num_actions=None, num_episodes=1):
        if num_actions is None:
            self.checkpoint["step"] = np.inf
        else:
            assert num_episodes == 1
            self.checkpoint["step"] += num_actions
        self.checkpoint["episode"] += num_episodes
        IPython.core.getipython.get_ipython().exiter()

    def return_to_training(self):
        self.disable_interactive_mode()
        IPython.core.getipython.get_ipython().exiter()

    def forward(self):
        state, action = self.env.state, self.policy(
            torch.tensor(self.env.state), epsilon=self.epsilon
        )
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
