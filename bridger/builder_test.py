import sys
import unittest

from bridger import builder
from bridger.callbacks import EarlyStoppingCallback
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback


class CustomCallback(Callback):
    def __init__(self):
        self.count = 0

    def on_train_batch_end(
        self, trainer, model, outputs, batch, batch_idx, dataloader_idx
    ):
        self.count += 1


class TestEarlyCallback(unittest.TestCase):
    """
    Unit-test that checks functionality of EarlyStopping callback using a CustomCallback .
    """

    def test_early_callback(self):
        parser = builder.get_hyperparam_parser()
        hparams = parser.parse_args()
        # Set the default hyperparameter for testing EarlyCallback functionality to a smaller number of iterations.
        hparams.custom_val_loss_threshold = 10
        model = builder.BridgeBuilder(hparams)
        # Save reference to object to get self.count later
        custom_callback = CustomCallback()

        callbacks = [
            custom_callback,
            EarlyStoppingCallback(
                monitor="custom_val_loss",
                min_delta=sys.float_info.epsilon,  # Set to some arbitrarily delta, shouldn't matter how small.
                patience=0,
                verbose=False,
                mode="max",
                strict=True,
            ),
        ]
        trainer = Trainer(
            gradient_clip_val=hparams.gradient_clip_val,
            val_check_interval=int(1e6),
            default_root_dir=hparams.checkpoint_model_dir,
            max_steps=hparams.max_training_batches,
            callbacks=callbacks,
        )

        trainer.fit(model)

        # Test that the number of counter loops matches the dummy threshold.
        self.assertEqual(custom_callback.count, hparams.custom_val_loss_threshold)


if __name__ == "__main__":
    unittest.main()
