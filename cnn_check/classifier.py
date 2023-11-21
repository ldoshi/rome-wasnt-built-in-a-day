import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torch.nn
from torch.utils.data import DataLoader

#input_file = "test_inputs"
#label_file = "test_labels"

input_file = "../bridger/tmp_log_dir/bridges.pkl"
# Binary
#label_file = "../bridger/tmp_log_dir/is_bridge.pkl"
#label_file = "../bridger/tmp_log_dir/is_bridge_and_uses_less_than_k_bricks.pkl"
# Multiclass
label_file = "../bridger/tmp_log_dir/bridge_height.pkl"

# Binary
#loss_fn = torch.nn.BCELoss()
# Multiclass
loss_fn = torch.nn.CrossEntropyLoss()

# ACTIONABLE
num_classes = 7
train_test_split = .8
train_validate_split = .75
batch_size = 20
lr = .001
epochs = 300

with open(input_file, 'rb') as f:
    inputs = pickle.load(f)
    inputs = [torch.tensor(x) for x in inputs]

with open(label_file, 'rb') as f:
    labels = pickle.load(f)
    # Binary
#    labels = np.array(labels, dtype=np.float32)
# Multiclass
    labels = np.array(labels)


class CNN(torch.nn.Module):
    """Base class for CNN neural network module."""

    def __init__(self, image_height: int, image_width: int, num_actions: int):
        super(CNN, self).__init__()
        self.image_height = image_height
        self.image_width = image_width

        paddings = [1, 1]
        strides = [2, 1]
        kernel_sizes = [3, 3]
        channel_nums = [3, 4, 8]

        args_iter = zip(
            channel_nums[:-1], channel_nums[1:], kernel_sizes, strides, paddings
        )

        self.cnn = torch.nn.ModuleList([torch.nn.Conv2d(*args) for args in args_iter])
        H, W = self.image_height, self.image_width
        for padding, kernel_size, stride in zip(paddings, kernel_sizes, strides):
            H = int((H + 2 * padding - kernel_size) / stride) + 1
            W = int((W + 2 * padding - kernel_size) / stride) + 1
        C = channel_nums[-1]

        dense_widths = [C * H * W, 64, num_classes]

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
 # Binary
#        x = torch.sigmoid(x)
 # Multiclass
        x = torch.softmax(x, 0)
        return x


def encode_enum_state_to_channels(state_tensor: torch.Tensor, num_channels: int):
    """Takes a 3-dim state tensor and returns a one-hot tensor with a new channels
    dimension as the second dimension (batch, channels, height, width)"""
    # Note: if memory-usage problems, consider alternatives to int64 tensor
    x = F.one_hot(state_tensor.long(), num_channels)
    return x.permute(0, 3, 1, 2)

model = CNN(*inputs[0].shape, inputs[0].shape[1])

train_validate_count = round(len(inputs)*train_test_split)
data = list(zip(inputs, labels))
data_train_validate = data[:train_validate_count]
data_test = data[train_validate_count:]

train_count = round(len(data_train_validate) * train_validate_split)
data_train = data_train_validate[:train_count]
data_loader = DataLoader(data_train, batch_size)
data_validate = data_train_validate[train_count:]
validation_data_loader = DataLoader(data_validate, len(data_validate))

print()
print(f"Train Size: {len(data_train)}")
print(f"Valid Size: {len(data_validate)}")
print(f" Test Size: {len(data_test)}")

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for i in range(epochs):
    model.train()
    for j,(input,label) in enumerate(data_loader):
          
        #calculate output
        output = model(input)
      
        #calculate loss

        # Binary
#                loss = loss_fn(output, label.reshape(-1,1))

        # Multiclass
        label = F.one_hot(label, num_classes=num_classes).float()
        loss = loss_fn(output, label)



        accuracy = (output.round() == label).float().mean()
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % 20 == 0:
        print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,accuracy))

    model.eval()
    # Binary
#    for eval_input, eval_label in validation_data_loader:
#        eval_output = model(eval_input)
#        eval_accuracy = (eval_output.round() == eval_label).float().mean()
#        print(f"Evaluation accuracy: {eval_accuracy:.2f}")
    # Multiclass
