import pytest
import torch
import unittest

from time import process_time
from parameterized import parameterized

from bridger.utils import hash_tensor


class TestHash(unittest.TestCase):
    """Testing Hash Functions."""

    @staticmethod
    def _check_time(keys, values=None, test_dict=None, hash_func=None):
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

    @parameterized.expand(
        [
            ("String Hash", str, (6, 7), 10000),
            ("Tuple Hash", hash_tensor, (6, 7), 10000),
        ]
    )
    def test_hash_validity(self, name, hash_func, shape, repetitions):
        torch.manual_seed(0)
        tensors = [torch.rand(shape) for i in range(repetitions)]
        assert all(
            hash_func(x) == hash_func(torch.clone(x)) for x in tensors
        ), f"Hash {name} failed for identical tensors"

        hashed_tensors = {}
        for tensor_hash, tensor in zip(map(hash_func, tensors), tensors):
            if tensor_hash in hashed_tensors:
                assert (
                    tensor == hashed_tensors[tensor_hash]
                ).all(), f"Hash {name} failed for non-identical tensors"
            else:
                hashed_tensors[tensor_hash] = tensor

    @parameterized.expand(
        [
            ("String Hash", str, (6, 7), 10000),
            ("Tuple Hash", hash_tensor, (6, 7), 10000),
        ]
    )
    @pytest.mark.integtest
    def test_hash_speed(self, name, hash_func, shape, repetitions):
        keys = [torch.rand(shape) for i in range(repetitions)]
        values = torch.rand(repetitions).tolist()
        test_dict = {}
        assign_time = TestHash._check_time(
            keys, values=values, test_dict=test_dict, hash_func=hash_func
        )
        access_time = TestHash._check_time(
            keys, test_dict=test_dict, hash_func=hash_func
        )
        print(f"{name} Runtime for {repetitions} key-value pairs:")
        print(f"{assign_time} sec for assigning & {access_time} sec for accessing")


if __name__ == "__main__":
    unittest.main()
