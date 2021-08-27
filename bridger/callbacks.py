import torch

from pytorch_lightning.callbacks import Callback

from bridger import builder_trainer
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
        env = builder_trainer.make_env(model.hparams)
        policy = model.policy
        state = env.reset()
        for t in range(episode_length):
            state, reward, is_done, _ = env.step(policy(torch.tensor(state)))
            env.render()
            if is_done:
                print("finished at %d" % t)
                break


class HistoryCallback(Callback):
    def __init__(self, steps_per_update):
        self.frequency = steps_per_update

    def on_train_batch_end(
        self, trainer, model, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx + 1) % self.frequency == 0:
            model.training_history.serialize()
