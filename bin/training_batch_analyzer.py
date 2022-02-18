import argparse
import bridger.logging.object_logging as object_logging
import os
import torch

DISPLAY_NUM_ERROR_MSGS = 10


def main():
    """Checks that two logged training batch entries share the same values
    across all attributes.

    Example to run:

    (compare-log-batch-entries) % python -m bin.training_batch_analyzer
    --path_expected_log_entry tmp_object_logging_dir/training_batch
    --path_test_log_entry tmp_object_logging_dir/training_batch_2

    Batch entry error count: 0 For log batch entry index: 0, loss values are not equal:  Expected logged training batch value: 0.9522438841027203 Test logged training batch value: 0.9522439050239484
    ...
    Batch entry error count: 6 For log batch entry index: 7, loss values are not equal:  Expected logged training batch value: 5.393130202373399 Test logged training batch value: 5.393129525809499
    """

    parser = argparse.ArgumentParser(
        description="Compare two object log managers for equality."
    )
    parser.add_argument(
        "--path_expected_log_entry",
        help="The filepath to the first TrainingBatchLogEntry file.",
        default="",
    )
    parser.add_argument(
        "--path_test_log_entry",
        help="The filepath to the second TrainingBatchLogEntry file.",
        default="",
    )
    args = parser.parse_args()

    expected_basename = os.path.basename(args.path_expected_log_entry)
    expected_dirname = os.path.dirname(args.path_expected_log_entry)
    test_basename = os.path.basename(args.path_test_log_entry)
    test_dirname = os.path.dirname(args.path_test_log_entry)

    batch_entry_error_counter = 0

    for expected_log_batch_entry, test_log_batch_entry in zip(
        object_logging.read_object_log(expected_dirname, expected_basename),
        object_logging.read_object_log(test_dirname, test_basename),
    ):
        for field, container in expected_log_batch_entry.__dataclass_fields__.items():
            # Exit early if we hit the maximum number of displayed errors.
            if batch_entry_error_counter == DISPLAY_NUM_ERROR_MSGS:
                return
            expected_object_log_value = getattr(expected_log_batch_entry, field)
            test_object_log_value = getattr(test_log_batch_entry, field)

            if container.type == torch.Tensor:
                if field == "loss":
                    if not torch.allclose(
                      expected_object_log_value,
                     test_object_log_value,
                      atol=1e-4,
                    ):
                    print(
                        f"Batch entry error count: {batch_entry_error_counter}.",
                        f"For log batch entry index: {expected_log_batch_entry.batch_idx}, {field} values are not equal: ",
                        f"Expected logged training batch value: {expected_object_log_value}",
                        f"Test logged training batch value: {test_object_log_value}",
                    )
                    batch_entry_error_counter += 1
                    continue

                if not torch.equal(expected_object_log_value, test_object_log_value):
                    print(
                        f"Batch entry error count: {batch_entry_error_counter}.",
                        f"For log batch entry index: {expected_log_batch_entry.batch_idx}, {field} values are not equal: ",
                        f"Expected logged training batch value: {expected_object_log_value}",
                        f"Test logged training batch value: {test_object_log_value}",
                    )
                    batch_entry_error_counter += 1

                continue

            if (
                container.type is not torch.Tensor
                and expected_object_log_value != test_object_log_value
            ):
                print(
                    f"Batch entry error count: {batch_entry_error_counter}.",
                    f"For log batch entry index: {expected_log_batch_entry.batch_idx}, {field} values are not equal: ",
                    f"Expected logged training batch value: {expected_object_log_value}",
                    f"Test logged training batch value: {test_object_log_value}",
                )
                batch_entry_error_counter += 1


if __name__ == "__main__":
    main()
