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
    batch_idx: int
    indices: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    rewards: torch.Tensor
    successes: torch.Tensor
    weights: torch.Tensor
    loss: torch.Tensor


class TrainingBatchHistory:
    def __init__(self, filename: str):
        self.batches = []
        with open(filename, "rb") as f:
            try:
                while True:
                    raw_batch = pickle.load(f)
                    assert len(raw_batch) == 9, raw_batch
                    self.batches.append(
                        TrainingBatch(
                            **dict(
                                zip(
                                    [
                                        "loss",
                                        "indices",
                                        "states",
                                        "actions",
                                        "next_states",
                                        "rewards",
                                        "successes",
                                        "weights",
                                        "batch_idx",
                                    ],
                                    raw_batch,
                                )
                            )
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
        for field, container in x.__dataclass_fields__.items():
            typ = container.type
            value_x = getattr(x, field)
            value_y = getattr(y, field)
            try:
                if typ == torch.Tensor:
                    if value_x.type() == "torch.BoolTensor":
                        if torch.all(value_x == value_y):
                            continue
                    elif torch.allclose(value_x, value_y):
                        continue
                elif value_x == value_y:
                    continue

                print(f"Mismatch in {field} at step {step}")
                print(value_x)
                print(value_y)
                stop = True
            except:
                import IPython

                IPython.embed()

        if stop:
            break


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--files", type=str, nargs="+", help="Training batch history files"
    )
    args = parser.parse_args()

    global training_batch_histories

    assert len(args.files) == 2
    training_batch_histories += [
        TrainingBatchHistory(filename) for filename in args.files
    ]

    diff_training_batch_histories(*training_batch_histories)


if __name__ == "__main__":
    #    app.run(main)
    main()
