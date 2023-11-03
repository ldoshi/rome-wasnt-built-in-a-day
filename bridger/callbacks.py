import torch

from pytorch_lightning.callbacks import Callback

from bridger import builder_trainer
from bridger import builder


class DemoCallback(Callback):
    def __init__(self, steps_per_update, max_episode_length):
        self._max_episode_length = max_episode_length
        self._frequency = steps_per_update
        self._builder = None
        
    def on_train_batch_end(
        self, trainer, model, outputs, batch, batch_idx, dataloader_idx
    ):
        env = builder_trainer.make_env(
                    name=model.hparams.env_name,
                    width=model.hparams.env_width,
                    force_standard_config=model.hparams.env_force_standard_config,
                    seed=torch.rand(1).item(),
        )
        if (batch_idx + 1) % self._frequency == 0:
          with torch.no_grad():
              state = env.reset()
              total_reward = 0
              for i in range(self._max_episode_length):
                  # This lint error seems to be a torch+pylint issue in general.
                  # pylint: disable=not-callable
                  print("yoooo: " , model.policy(torch.tensor(state)))
                  state, reward, success, _ = env.step(model.policy(torch.tensor(state)))

                  total_reward += reward
                  #env.render()
                  if success:
                      break
              print("REWARD: ", total_reward)
            
