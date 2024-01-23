import shutil
import pathlib
import unittest

from bridger.builder_trainer import make_env
from bridger.generate_data import DatasetGenerator

TMP_LOG_DIR = "tmp_log_dir"
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
        dataset = DatasetGenerator(
            log_filename_directory=TMP_LOG_DIR,
            n_bricks=6,
            k=4,
            env=make_env(name=_ENV_NAME, width=6, force_standard_config=True),
        )
        dataset.generate_dataset()
        dataset.finalize()
