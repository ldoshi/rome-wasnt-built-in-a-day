"""Prints out the contents of a log.

   Usage:
   $ python -m tools.log_printer --log object_logging_dir/state_normalized
"""

import argparse
from bridger.logging_utils import object_log_readers
import os
import torch


def main():
    parser = argparse.ArgumentParser(description="Print out a log file.")
    parser.add_argument(
        "--log",
        help="The filepath to the log file to print out.",
        required=True,
    )

    args = parser.parse_args()
    for entry in object_log_readers.read_object_log(args.log):
        print(entry)


if __name__ == "__main__":
    main()
