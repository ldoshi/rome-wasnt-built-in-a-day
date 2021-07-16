from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

from pytorch_lightning.callbacks import Callback

from bridger import builder
from bridger import training_panel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


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
    def on_validation_end(self, trainer, pl_module):
        pass

    def on_train_batch_end(
        self, trainer, model, outputs, batch, batch_idx, dataloader_idx
    ):
        # The code will error on the first iteration of training since the model doesn't log train_loss until after the first on_train_batch_end in the training_step() function. For the first step, skip checking for early stopping.
        if model.early_stopping_variable == 1:
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
