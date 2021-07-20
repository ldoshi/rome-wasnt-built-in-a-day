import torch

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from bridger import builder
from bridger import training_panel


class DemoCallback(Callback):
    def __init__(self, steps_per_update, max_episode_length):
        self.max_episode_length = max_episode_length
        self.frequency = steps_per_update

    def on_train_batch_end(
        self, trainer, model, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx + 1) % self.frequency == 0:
            with torch.no_grad():
                self.demo(model, self.max_episode_length)

    def demo(self, model, episode_length):
        env = builder.make_env(model.hparams)
        policy = model.policy
        state = env.reset()
        for t in range(episode_length):
            state, reward, is_done, _ = env.step(policy(torch.tensor(state)))
            env.render()
            if is_done:
                print("finished at %d" % t)
                break


class EarlyStoppingCallback(EarlyStopping):
    """A custom early stopping callback that inherits from the default EarlyStopping. We currently use on_train_batch_end as the hook for the callback, as on_validation_end doesn't get called with our current DQN implementation. The original behavior of the inherited EarlyStopping callback runs at the end of every validation epoch, which never triggers since we only have 1 epoch running at any given time. We use iterations/batches to read from the replay buffer, so we callback on_train_batch_end instead. See note from link: https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html"""

    def on_train_batch_end(
        self, trainer, model, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gets called when train batch ends. The current code will error on the first iteration of training without the if statement since the model doesn't log metrics until after the first on_train_batch_end() in the training_step() function. To fix this, we currently skip checking for early stopping in the first step."""
        if model._custom_val_loss == 3:
            pass
        else:
            self._run_early_stopping_check(trainer)


class HistoryCallback(Callback):
    def __init__(self, steps_per_update):
        self.frequency = steps_per_update

    def on_train_batch_end(
        self, trainer, model, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx + 1) % self.frequency == 0:
            model.training_history.serialize()
