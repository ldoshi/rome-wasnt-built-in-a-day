#!/usr/bin/env python

"""Display training batch debugging information.

The training batch analyzer reads training batch data from the
file(s) indicated by the flag `training_batch_history_files` and offers a suite of interactive analysis functions.

This tool is intended to be run with -i.

"""

import argparse
import dataclasses
import pickle
import torch

from tqdm import tqdm

training_batch_histories = []
tbh = training_batch_histories


@dataclasses.dataclass
class TrainingBatch:
    indices: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    successes: torch.Tensor
    weights: torch.Tensor


class TrainingBatchHistory:
    def __init__(self, filename: str):
        self.batches = []
        with open(filename, "rb") as f:
            try:
                while True:
                    raw_batch = pickle.load(f)
                    assert len(raw_batch) == 7, raw_batch
                    self.batches.append(
                        TrainingBatch(
                            indices=raw_batch[0],
                            states=raw_batch[1],
                            actions=raw_batch[2],
                            next_states=raw_batch[3],
                            rewards=raw_batch[4],
                            successes=raw_batch[5],
                            weights=raw_batch[6],
                        )
                    )

            except EOFError:
                pass


def diff_training_batch_histories(
    history_x: TrainingBatchHistory, history_y: TrainingBatchHistory
) -> None:
    """Identifies the first position for which the history_x and history_y differ on at least one field."""
    stop = False
    for step, (x, y) in enumerate(tqdm(zip(history_x.batches, history_y.batches))):
        if torch.any(x.indices != y.indices):
            print(f"Indices mismatch at step {step}")
            print(x.indices)
            print(y.indices)
            stop = True

        if torch.any(x.states != y.states):
            print(f"States mismatch at step {step}")
            print(x.states)
            print(y.states)
            stop = True

        if torch.any(x.actions != y.actions):
            print(f"Actions mismatch at step {step}")
            print(x.actions)
            print(y.actions)
            stop = True

        if torch.any(x.next_states != y.next_states):
            print(f"Next_States mismatch at step {step}")
            print(x.next_states)
            print(y.next_states)
            stop = True

        if torch.any(x.rewards != y.rewards):
            print(f"Rewards mismatch at step {step}")
            print(x.rewards)
            print(y.rewards)
            stop = True

        if torch.any(x.successes != y.successes):
            print(f"Successes mismatch at step {step}")
            print(x.successes)
            print(y.successes)
            stop = True

        if torch.any(x.weights != y.weights):
            print(f"Weights mismatch at step {step}")
            print(x.weights)
            print(y.weights)
            stop = True

        if stop:
            break


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--files", type=str, nargs="+", help="Training batch history files"
    )
    args = parser.parse_args()

    assert len(args.files) == 2
    training_batch_histories += [
        TrainingBatchHistory(filename) for filename in args.files
    ]

    diff_training_batch_histories(*training_batch_histories)


if __name__ == "__main__":
    #    app.run(main)
    main()
