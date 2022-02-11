import argparse
import bridger.logging.object_logging as object_logging
import os
import torch


def main():
    # Checks that two object log batch entry files share the same values across all attributes in the file.
    parser = argparse.ArgumentParser(
        description="Compare two object log managers for equality"
    )
    parser.add_argument(
        "--pathExpectedLogEntry",
        help="The filepath to the first LogBatchEntry file.",
        default="",
    )
    parser.add_argument(
        "--pathTestLogEntry",
        help="The filepath to the second LogBatchEntry file.",
        default="",
    )
    args = parser.parse_args()

    expected_basename, expected_dirname = os.path.basename(
        args.pathExpectedLogEntry
    ), os.path.dirname(args.pathExpectedLogEntry)
    test_basename, test_dirname = os.path.basename(
        args.pathTestLogEntry
    ), os.path.dirname(args.pathTestLogEntry)

    for expected_log_batch_entry, test_log_batch_entry in zip(
        object_logging.read_object_log(expected_dirname, expected_basename),
        object_logging.read_object_log(test_dirname, test_basename),
    ):
        for field, container in expected_log_batch_entry.__dataclass_fields__.items():
            expected_object_log_value = getattr(expected_log_batch_entry, field)
            test_object_log_value = getattr(test_log_batch_entry, field)
            if container.type == torch.Tensor:
                if field == "loss":
                    print(
                        f"{field} values are equal: ",
                        torch.allclose(
                            expected_object_log_value,
                            test_object_log_value,
                            atol=1e-4,
                        ),
                    )
                else:
                    print(
                        f"{field} values are equal: ",
                        torch.equal(expected_object_log_value, test_object_log_value),
                    )
            else:
                print(
                    f"{field} is not a torch.Tensor: ",
                    expected_object_log_value,
                    test_object_log_value,
                )


if __name__ == "__main__":
    main()
