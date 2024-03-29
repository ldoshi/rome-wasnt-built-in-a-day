import pytest
import torch
import unittest

from time import process_time
from parameterized import parameterized

from bridger import hash_utils


def _check_hash_time(keys, values=None, test_dict=None, hash_func=None):
    """Helper function to test combined explicit/implicit hashing time.

    Args:
        keys: a list of distinct torch Tensors of identical shape
        values: an optional list of values to associate with the respective
            tensors in `keys`.
        test_dict: an optional starting dictionary mapping tensors to values
            that defaults to an empty dictionary.
        hash_func: an optional function to be used for hashing tensors that
            defaults to the identity function

    Returns the time (in seconds) required for the system to populate (if
        `values` is provided) or access keys in (if `values` is not
        provided) the dictionary with tensor keys from `keys` using the hash
        function `hash_func`."""

    if hash_func is None:
        hash_func = lambda x: x
    if test_dict is None:
        test_dict = {}
    start_time = process_time()
    if values is None:
        for key in keys:
            test_dict.get(hash_func(key))
    else:
        for key, value in zip(keys, values):
            test_dict[hash_func(key)] = value
    return process_time() - start_time


class TestHash(unittest.TestCase):
    """Testing Hash Functions."""

    @parameterized.expand(
        [
            ("String Hash", str, 10000),
            ("Tuple Hash", hash_utils.hash_tensor, 10000),
        ]
    )
    def test_hash_validity(self, name, hash_func, repetitions, shape=(6, 7)):
        torch.manual_seed(0)
        hashed_tensors = {}
        for _ in range(repetitions):
            tensor = torch.rand(shape)
            tensor_hash = hash_func(tensor)

            self.assertEqual(
                tensor_hash, hash_func(torch.clone(tensor))
            ), f"Hash {name} failed for identical tensors"

            # Here we ensure that two differently-shaped tensors that are
            # identical when flattened hash to different values

            # Construct a tensor with identical content when flattened
            reshaped_tensor = tensor.reshape(tensor.shape[::-1])
            self.assertTrue(
                (tensor.reshape(-1) == reshaped_tensor.reshape(-1)).all()
            ), "Attempted reshape in test is invalid"

            # Ensure that the differently shaped tensor with identical flattened
            # content hashes to a different value
            self.assertNotEqual(
                tensor_hash, hash_func(reshaped_tensor)
            ), f"Hash {name} failed for reshaped tensors"

            if tensor_hash in hashed_tensors:
                self.assertTrue(
                    (tensor == hashed_tensors[tensor_hash]).all()
                ), f"Hash {name} failed for non-identical tensors"
            else:
                hashed_tensors[tensor_hash] = tensor

    @parameterized.expand(
        [
            ("String Hash", str, 10000),
            ("Tuple Hash", hash_utils.hash_tensor, 10000),
        ]
    )
    @pytest.mark.integtest
    def test_hash_speed(self, name, hash_func, repetitions, shape=(6, 7)):
        keys = [torch.rand(shape) for i in range(repetitions)]
        values = torch.rand(repetitions).tolist()
        test_dict = {}
        assign_time = _check_hash_time(
            keys, values=values, test_dict=test_dict, hash_func=hash_func
        )
        access_time = _check_hash_time(keys, test_dict=test_dict, hash_func=hash_func)
        print(f"{name} Runtime for {repetitions} key-value pairs:")
        print(f"{assign_time} sec for assigning & {access_time} sec for accessing")


if __name__ == "__main__":
    unittest.main()
