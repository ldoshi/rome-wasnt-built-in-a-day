import torch

from pytorch_lightning.callbacks import Callback

from bridger import builder_trainer
from bridger import builder


class DemoCallback(Callback):
    def __init__(self, steps_per_update, max_episode_length):
        self._max_episode_length = max_episode_length
        self._frequency = steps_per_update
        self._epoch = 0
        self._builder = None
        
    def on_epoch_end(
        self, trainer, model,
#            self, trainer, model, outputs, batch, batch_idx, dataloader_idx
    ):
        self._epoch += 1
        if (self._epoch) % self._frequency > 0:
            return
        env = builder_trainer.make_env(
                    name=model.hparams.env_name,
                     width=model.hparams.env_width,
                    force_standard_config=model.hparams.env_force_standard_config,
            render=True,
                    seed=torch.rand(1).item(),
        )
        with torch.no_grad():
            state, _ = env.reset()
            total_reward = 0
            for i in range(self._max_episode_length):
                # This lint error seems to be a torch+pylint issue in general.
                # pylint: disable=not-callable

                state, reward, success, _, _= env.step(model.policy(torch.tensor(state)))

                total_reward += reward
                if success:
                    break
            print("REWARD: ", total_reward)
            
