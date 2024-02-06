#!/usr/bin/env python

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
import sklearn.metrics as metrics
from sklearn.utils import shuffle
from collections import defaultdict, Counter


class CNN(torch.nn.Module):
    """Base class for CNN neural network module."""

    def __init__(self, image_height: int, image_width: int, num_actions: int):
        super(CNN, self).__init__()
        self.image_height = image_height
        self.image_width = image_width

        paddings = args.paddings
        strides = args.strides
        kernel_sizes = args.kernel_sizes
        channel_nums = args.channel_nums

        args_iter = zip(
            channel_nums[:-1], channel_nums[1:], kernel_sizes, strides, paddings
        )

        self.cnn = torch.nn.ModuleList([torch.nn.Conv2d(*args) for args in args_iter])
        H, W = self.image_height, self.image_width
        for padding, kernel_size, stride in zip(paddings, kernel_sizes, strides):
            H = int((H + 2 * padding - kernel_size) / stride) + 1
            W = int((W + 2 * padding - kernel_size) / stride) + 1
        C = channel_nums[-1]

        dense_widths = [C * H * W, 64, output_head_count]

        args_iter = zip(dense_widths[:-1], dense_widths[1:])
        self.dnn = torch.nn.ModuleList([torch.nn.Linear(*args) for args in args_iter])

    def forward(self, x):
        x = x.reshape(-1, self.image_height, self.image_width)
        x = encode_enum_state_to_channels(x, self.cnn[0].in_channels).float()
        for layer in self.cnn:
            x = torch.relu(layer(x))
        x = self.dnn[0](x.reshape(x.shape[0], -1))
        for layer in self.dnn[1:]:
            x = layer(torch.relu(x))
        if args.mode == "binary":
            x = torch.sigmoid(x)
        elif args.mode == "multiclass":
            x = torch.softmax(x, 0)
        return x


def encode_enum_state_to_channels(state_tensor: torch.Tensor, num_channels: int):
    """Takes a 3-dim state tensor and returns a one-hot tensor with a new channels
    dimension as the second dimension (batch, channels, height, width)"""
    # Note: if memory-usage problems, consider alternatives to int64 tensor
    x = F.one_hot(state_tensor.long(), num_channels)
    return x.permute(0, 3, 1, 2)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--experiment-name",
    type=str,
    default=time.strftime("%Y%m%d-%H%M%S"),
)
parser.add_argument("--experiment-name-prefix", type=str, default="")
parser.add_argument("--input_file", type=str, default="inputs/bridges.pkl")
parser.add_argument("--label_file", type=str, default="labels/bridge_height.pkl")
parser.add_argument("--n_fewest_elements", type=int, default=1)
parser.add_argument("--mode", type=str, default="multiclass")
parser.add_argument("--num_classes", type=int, default=7)
parser.add_argument("--train_test_split_ratio", type=float, default=0.8)
parser.add_argument("--train_validate_split_ratio", type=float, default=0.75)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--paddings", nargs=2, type=int, default=[1, 1])
parser.add_argument("--strides", nargs=2, type=int, default=[2, 1])
parser.add_argument("--kernel_sizes", nargs=2, type=int, default=[3, 3])
parser.add_argument("--channel_nums", nargs=3, type=int, default=[3, 4, 8])
parser.add_argument("--random_state", default=42)
parser.add_argument("--rebalance", type=bool, default=False)
parser.add_argument("--generate_confusion_matrix", action="store_false", default=False)

args = parser.parse_args()

if args.mode == "binary":
    loss_fn = torch.nn.BCELoss()
    output_head_count = 1
elif args.mode == "multiclass":
    loss_fn = torch.nn.CrossEntropyLoss()
    output_head_count = args.num_classes


with open(args.input_file, "rb") as f:
    inputs = pickle.load(f)
    inputs = [torch.tensor(x) for x in inputs]

with open(args.label_file, "rb") as f:
    labels = pickle.load(f)
    if args.mode == "binary":
        labels = np.array(labels, dtype=np.float32)
    elif args.mode == "multiclass":
        labels = np.array(labels)

if args.mode == "binary":
    loss_fn = torch.nn.BCELoss()
    output_head_count = 1
elif args.mode == "multiclass":
    loss_fn = torch.nn.CrossEntropyLoss()
    output_head_count = args.num_classes

with open(args.input_file, "rb") as f:
    inputs = pickle.load(f)
    inputs = [torch.tensor(x) for x in inputs]

with open(args.label_file, "rb") as f:
    labels = pickle.load(f)
    if args.mode == "binary":
        labels = np.array(labels, dtype=np.float32)
    elif args.mode == "multiclass":
        labels = np.array(labels)


model = CNN(*inputs[0].shape, inputs[0].shape[1])

# Downsample all classes to the class with the fewest elements.
data = list(zip(inputs, labels))
data = shuffle(data)

if args.rebalance:
    len_fewest_label = Counter(labels).most_common()[-args.n_fewest_elements][1]
    print(f"All classes will be downsampled to at most {len_fewest_label} elements.")

    label_to_state_dict = defaultdict(list)
    for state, label in data:
        if len(label_to_state_dict[label]) < len_fewest_label:
            label_to_state_dict[label].append(state)

    data = []
    for label, state_list in label_to_state_dict.items():
        for state in state_list:
            data.append((state, label))

data_train_side, data_test = train_test_split(
    data,
    train_size=args.train_test_split_ratio,
    random_state=args.random_state,
    shuffle=True,
)
data_train, data_validate = train_test_split(
    data_train_side,
    train_size=args.train_validate_split_ratio,
    random_state=args.random_state,
    shuffle=True,
)

data_loader = DataLoader(data_train, args.batch_size)
validation_data_loader = DataLoader(data_validate, len(data_validate))

print()
print(f"Train Size: {len(data_train)}")
print(f"Valid Size: {len(data_validate)}")
print(f" Test Size: {len(data_test)}")

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def dump_metrics(epoch, prefix, label, output, writer=None, print_metrics=True, verbose=False):
    accuracy = metrics.accuracy_score(label, output)
    precision = 3 # metrics.precision_score(label,output)
    recall = 3 # metrics.recall_score(label, output)
    confusion_matrix =metrics.confusion_matrix(label, output)

    if writer:
        writer.add_scalar(f"{prefix} accuracy", accuracy, epoch)
        writer.add_scalar(f"{prefix} precision", precision, epoch)
        writer.add_scalar(f"{prefix} recall", recall, epoch)

    if print_metrics:
        print(f"Epoch {i} {prefix} Metrics")
        print(f"  accuracy: {accuracy}")
        print(f"  precision: {precision}")
        print(f"  recall: {recall}")
        if verbose:
            print(f"  confusion matrix:\n{confusion_matrix}") 

# Enable Tensorboard writer for logging loss/accuracy. By default, Tensorboard logs are written to the 'runs' folder.
writer = SummaryWriter(log_dir=os.path.join("runs", args.experiment_name))

for i in range(args.epochs):
    model.train()
    output_all = []
    train_label_all = []
    for j, (input, train_label) in enumerate(data_loader):
        # calculate output
        output = model(input)

        # calculate loss
        if args.mode == "binary":
            loss = loss_fn(output, train_label.reshape(-1, 1))
        elif args.mode == "multiclass":
            label_one_hot = F.one_hot(train_label, num_classes=args.num_classes).float()
            loss = loss_fn(output, label_one_hot)

        # if j == 0:
        # print("EPOCH: " , i)
        # print("OUTPUT: ", output)
        # print("LABEL: ", label)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.mode == "binary":
            output = output.round()
        elif args.mode == "multiclass":
            output = output.argmax(axis=1)

        output_all.extend(output)
        train_label_all.extend(train_label)

    if i % 1 == 0:
        print(f"epoch {i}\tloss: {loss}")
        dump_metrics(i, "Train", train_label_all, output_all, writer, print_metrics=True, verbose=args.generate_confusion_matrix)
        writer.add_scalar("Train loss", loss, i)

        model.eval()
        eval_output_all = []
        eval_label_all = []
        for eval_input, eval_label in validation_data_loader:
            eval_output = model(eval_input)
            if args.mode == "binary":
                eval_output = eval_output.round()
            elif args.mode == "multiclass":
                eval_output = eval_output.argmax(axis=1)
            eval_output_all.extend(eval_output)
            eval_label_all.extend(eval_label)

        dump_metrics(i, "Eval", eval_label_all, eval_output_all, writer, print_metrics=True, verbose=args.generate_confusion_matrix)


writer.close()
