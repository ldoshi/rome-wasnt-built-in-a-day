import torch
import torch.nn.functional as F


class CNNQ(torch.nn.Module):
    def __init__(self, image_height, image_width, num_actions):
        super(CNNQ, self).__init__()
        self.image_height = image_height
        self.image_width = image_width

        paddings = [1, 1]
        strides = [2, 1]
        kernel_sizes = [3, 3]
        channel_nums = [3, 4, 8]

        args_iter = zip(
            channel_nums[:-1], channel_nums[1:], kernel_sizes, strides, paddings
        )

        self.CNN = torch.nn.ModuleList([torch.nn.Conv2d(*args) for args in args_iter])
        H, W = self.image_height, self.image_width
        for padding, kernel_size, stride in zip(paddings, kernel_sizes, strides):
            H = int((H + 2 * padding - kernel_size) / stride) + 1
            W = int((W + 2 * padding - kernel_size) / stride) + 1
        C = channel_nums[-1]
        dense_widths = [C * H * W, 64, num_actions]
        args_iter = zip(dense_widths[:-1], dense_widths[1:])
        self.DNN = torch.nn.ModuleList([torch.nn.Linear(*args) for args in args_iter])

    def forward(self, x):
        x = x.reshape(-1, self.image_height, self.image_width)
        x = encode_enum_state_to_channels(x, self.CNN[0].in_channels).float()
        for layer in self.CNN:
            x = torch.relu(layer(x))
        x = self.DNN[0](x.reshape(x.shape[0], -1))
        for layer in self.DNN[1:]:
            x = layer(torch.relu(x))
        return x


def encode_enum_state_to_channels(
    state_tensor: torch.Tensor, num_channels: int
):
    """Takes a 3-dim state tensor and returns a one-hot tensor with a new channels
    dimension as the second dimension (batch, channels, height, width)"""
    # Note: if memory-usage problems, consider alternatives to int64 tensor
    x = F.one_hot(state_tensor.long(), num_channels)
    return x.permute(0, 3, 1, 2)

# This architecture has not yet been validated (and is likely poor).
choices = {"default": CNNQ}
