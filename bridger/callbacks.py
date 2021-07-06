import torch

from pytorch_lightning.callbacks import Callback

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
        state = env.state
        for t in range(episode_length):
            state, reward, is_done = env.step(policy(torch.tensor(state)))[:3]
            env.render()
            if is_done:
                print("finished at %d" % t)
                break


class PanelCallback(Callback):
    def __init__(
        self, steps_per_update, states_n, state_width, state_height, actions_n
    ):
        self.panel = training_panel.TrainingPanel(
            states_n,
            state_width,
            state_height,
            actions_n,
        )
        self.frequency = steps_per_update

    def on_train_batch_end(
        self, trainer, model, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx + 1) % self.frequency == 0:
            self.panel.update_panel(model.training_history.get_history_by_visit_count())
