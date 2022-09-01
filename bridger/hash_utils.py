"""A set of hash functions"""

from torch import Tensor
from typing import Union


def hash_tensor(x: Tensor) -> tuple[tuple[int, ...], tuple[Union[int, float], ...]]:
    """A function to hash torch tensors to be used as keys in a dictionary.

    This satisfies the property that two tensors share the same hash value iff
    they are themselves identical. This function is necessary because the
    built-in hash# returns different values for identically-valued but distinct
    tensor objects.

    Args:
        x: the torch tensor object to be hashed

    Returns a unique hash value for the input tensor. This will be a tuple
        `(shape, vals)` where `shape` is itself a tuple representing the shape
        of `x` and `vals` is a flattened tuple containing the elements of `x`"""

    return (tuple(x.shape), tuple(x.reshape(-1).tolist()))
