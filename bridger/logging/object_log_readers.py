"""Object log consumption tools to support various use-cases.

Supported use-cases range from simple and direct reading of a single
object log to the TrainingHistoryDatabase, which supports aggregate
queries on top of logged data.

"""
from typing import Any, Callable, List, Optional

import collections
import copy
import shutil
import pickle
import os
import pathlib
import torch

from collections.abc import Hashable

from bridger.logging import log_entry


def read_object_log(dirname: str, log_filename: str):
    with open(os.path.join(dirname, log_filename), "rb") as f:
        buffer = None
        while True:
            try:
                buffer = pickle.load(f)

                for element in buffer:
                    yield element

            except EOFError:
                break
