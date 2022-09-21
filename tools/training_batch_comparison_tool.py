import argparse
from bridger.logging import object_log_readers
import os
import torch

NUM_ERROR_THRESHOLD = 5

def compare_training_batches(expected_log: str, test_log: str) -> bool:
    """Compares training batches between two log files.

    Checks that all training batch entries share the same values
    across all attributes.

    Args:
      expected_log: The filepath to the first TrainingBatchLogEntry
        log file.
      test_log: The filepath to the secon TrainingBatchLogEntry log
        file.

    Returns:
      True if all batches matched, false otherwise. When false is
      returned, the errors will be printed to the console.

    """
    batch_entry_error_counter = 0

    for entry_index, (expected_log_batch_entry, test_log_batch_entry) in enumerate(
        zip(
            object_log_readers.read_object_log(expected_log),
            object_log_readers.read_object_log(test_log),
        )
    ):
        for field in expected_log_batch_entry.__dataclass_fields__:
            expected_object_log_value = getattr(expected_log_batch_entry, field)
            test_object_log_value = getattr(test_log_batch_entry, field)
            if isinstance(expected_object_log_value, torch.Tensor):
                if field == "loss":
                    if not torch.allclose(
                        expected_object_log_value,
                        test_object_log_value,
                        atol=1e-4,
                    ):
                        print(
                            f"Batch entry error count: {batch_entry_error_counter}.\n",
                            f"\nFor log batch entry index: {expected_log_batch_entry.batch_idx}, {field} values are not equal: ",
                            f"\nExpected logged training batch value: \n{expected_object_log_value}",
                            f"\nTest logged training batch value: \n{test_object_log_value}",
                        )
                        batch_entry_error_counter += 1
                    continue

                if not torch.equal(expected_object_log_value, test_object_log_value):
                    print(
                        f"Batch entry error count: {batch_entry_error_counter}.",
                        f"\nFor log batch entry index: {expected_log_batch_entry.batch_idx}, {field} values are not equal: ",
                        f"\nExpected logged training batch value: \n{expected_object_log_value}",
                        f"\nTest logged training batch value: \n{test_object_log_value}",
                    )
                    batch_entry_error_counter += 1
                    continue

                continue

            if (
                not isinstance(expected_object_log_value, torch.Tensor)
                and expected_object_log_value != test_object_log_value
            ):
                print(
                    f"Batch entry error count: {batch_entry_error_counter}.",
                    f"\nFor log batch entry index: {expected_log_batch_entry.batch_idx}, {field} values are not equal: ",
                    f"\nExpected logged training batch value: \n{expected_object_log_value}",
                    f"\nTest logged training batch value: \n{test_object_log_value}",
                )
                batch_entry_error_counter += 1

        # Exit early if we hit the maximum number of displayed errors.
        if batch_entry_error_counter >= NUM_ERROR_THRESHOLD:
            return False

    return batch_entry_error_counter == 0 


def main():
    """Checks that two log files of training batch entries share the same values
    across all attributes.
    Example to run:
    $ python -m tools.training_batch_comparison_tool
      --expected_log object_logging_dir/training_batch
      --test_log object_logging_dir_2/training_batch
    """

    parser = argparse.ArgumentParser(
        description="Compare two object log managers for equality."
    )
    parser.add_argument(
        "--expected_log",
        help="The filepath to the first TrainingBatchLogEntry log file.",
        required=True,
    )
    parser.add_argument(
        "--test_log",
        help="The filepath to the second TrainingBatchLogEntry log file.",
        required=True,
    )
    args = parser.parse_args()
    if compare_training_batches(expected_log=args.expected_log, test_log=args.test_log):
        print("Success!")
    else:
        print("Failed!")

if __name__ == "__main__":
    main()
