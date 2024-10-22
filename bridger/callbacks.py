import torch

from lightning.pytorch import Callback

from bridger import builder_trainer
from bridger import builder


class DemoCallback(Callback):
    def __init__(self, steps_per_update, max_episode_length):
        self._max_episode_length = max_episode_length
        self._frequency = steps_per_update
        self._builder = None

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        return
        if not self._builder:
            # We could also initialize a new Builder each time. This
            # is a slight optimization to save repeated calls to
            # make_env.
            self._builder = builder.Builder(
                builder_trainer.make_env(
                    name=model.hparams.env_name,
                    width=model.hparams.env_width,
                    force_standard_config=model.hparams.env_force_standard_config,
                    seed=torch.rand(1).item(),
                )
            )

        if (model.global_step + 1) % self._frequency == 0:
            with torch.no_grad():
                build_result = self._builder.build(
                    model.policy, self._max_episode_length
                )
                if build_result.success:
                    print(
                        f"Built in {build_result.steps} steps with reward {build_result.reward}."
                    )
