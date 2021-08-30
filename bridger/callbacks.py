import torch

from pytorch_lightning.callbacks import Callback

from bridger import builder_trainer
from bridger import builder
from bridger import training_panel


class DemoCallback(Callback):
    def __init__(self, hparams, steps_per_update, max_episode_length):
        self._max_episode_length = max_episode_length
        self._frequency = steps_per_update
        self._builder = builder.Builder(
            builder_trainer.make_env(
                name=hparams.env_name,
                width=hparams.env_width,
                force_standard_config=hparams.env_force_standard_config,
            )
        )

    def on_train_batch_end(
        self, trainer, model, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx + 1) % self._frequency == 0:
            with torch.no_grad():
                build_result = self._builder.build(
                    model.policy, self._max_episode_length
                )
                if build_result.finished:
                    print(
                        f"Built in {build_result.steps} steps with reward {build_result.reward}."
                    )


class HistoryCallback(Callback):
    def __init__(self, steps_per_update):
        self.frequency = steps_per_update

    def on_train_batch_end(
        self, trainer, model, outputs, batch, batch_idx, dataloader_idx
    ):
        if (batch_idx + 1) % self.frequency == 0:
            model.training_history.serialize()
