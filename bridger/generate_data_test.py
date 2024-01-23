import shutil
import pathlib
import unittest

from bridger.builder_trainer import make_env
from bridger import generate_data

TMP_LOG_DIR = "tmp_log_dir_10_6_8"
_ENV_NAME = "gym_bridges.envs:Bridges-v0"


def create_temp_dir():
    path = pathlib.Path(TMP_LOG_DIR)
    path.mkdir(exist_ok=True)


def delete_temp_dir():
    path = pathlib.Path(TMP_LOG_DIR)
    shutil.rmtree(path, ignore_errors=True)


class TestGenerateData(unittest.TestCase):
    def setUp(self):
        create_temp_dir()

    def test_generate_six_wide(self):
        dataset = generate_data.DatasetGenerator(
            log_filename_directory=TMP_LOG_DIR,
            n_states=None,
            n_bricks=10,
            k=6,
            env=make_env(name=_ENV_NAME, width=8, force_standard_config=True),
            state_fn=generate_data.n_bricks,
        )
        dataset.generate_dataset()
        dataset.finalize()

    def test_generate_random(self):
        dataset = generate_data.DatasetGenerator(
            log_filename_directory=TMP_LOG_DIR,
            n_bricks=12,
            n_states=10000,
            k=6,
            env=make_env(name=_ENV_NAME, width=8, force_standard_config=True),
            state_fn=generate_data.n_states,
        )
        dataset.generate_dataset()
        dataset.finalize()
