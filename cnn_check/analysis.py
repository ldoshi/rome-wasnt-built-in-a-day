#!/usr/bin/env python

import collections
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torch.nn
from torch.utils.data import DataLoader
import argparse
from sklearn.model_selection import train_test_split
import time
from torch.utils.tensorboard import SummaryWriter
import os


parser = argparse.ArgumentParser()

parser.add_argument(
    "--label_dir", type=str, default="../bridger/tmp_log_dir_10_6_8/labels"
)
parser.add_argument("--train_test_split_ratio", type=float, default=0.8)
parser.add_argument("--train_validate_split_ratio", type=float, default=0.75)
parser.add_argument("--random_state", default=42)
args = parser.parse_args()

def compute_label_distribution(label_file: str):
    with open(label_file, "rb") as f:
        labels = pickle.load(f)

    labels_train_side, labels_test = train_test_split(
        labels,
        train_size=args.train_test_split_ratio,
        random_state=args.random_state,
        shuffle=True,
    )
    labels_train, labels_validate = train_test_split(
        labels_train_side,
        train_size=args.train_validate_split_ratio,
        random_state=args.random_state,
        shuffle=True,
    )

    return collections.Counter(labels_train), collections.Counter(labels_validate)


def print_distribution(split: str, counter: collections.Counter) -> None:
    print(f"{split}")
    print("Label Distribution (Raw)")
    for k,v in counter.most_common():
        print(f"{k}: {v}")
    print()
    print("Label Distribution (%)")
    for k,v in counter.most_common():
        print(f"{k}: {v/counter.total()*100:.2f}")
    print()
    print()

for label_file_name in os.listdir(args.label_dir):
    label_file = os.path.join(args.label_dir, label_file_name)
    label_distribution_train, label_distribution_validate = compute_label_distribution(label_file)

    print(f"File: {label_file_name}")

    print_distribution("Train", label_distribution_train)
    print_distribution("Validate", label_distribution_validate)


