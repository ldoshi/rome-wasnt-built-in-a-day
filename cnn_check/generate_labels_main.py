import argparse
import numpy as np
import pickle
import os


def load_states(states_file: str) -> list[np.ndarray]:
    with open(states_file, "rb") as f:
        return pickle.load(f)


def has_consecutive_bricks_flat_on_top(states: list[np.ndarray]) -> list[np.ndarray]:
    labels = []
    for state in states:
        for i in range(state.shape[0]):
            row_sum = np.sum(state[i])
            if row_sum:
                labels.append(row_sum > 7)
                break
    return labels


def is_one_sided(states: list[np.ndarray]) -> list[np.ndarray]:
    labels = []
    for state in states:
        labels.append(np.sum(state.T[0]) == 1 or np.sum(state.T[-1]) == 1)
    return labels


def has_gap_in_the_middle(states: list[np.ndarray]) -> list[np.ndarray]:
    labels = []
    middle_one = states[0].shape[1] // 2
    middle_two = middle_one - 1
    for state in states:
        labels.append(
            np.sum(state.T[middle_one]) == 0 or np.sum(state.T[middle_two]) == 0
        )
    return labels


def has_small_gap_in_the_middle(states: list[np.ndarray]) -> list[np.ndarray]:
    labels = []
    middle_one = states[0].shape[1] // 2
    middle_two = middle_one - 1
    for state in states:
        labels.append(
            (np.sum(state.T[middle_one]) == 0 or np.sum(state.T[middle_two]) == 0)
            and (np.sum(state.T[middle_one + 1]) > 0)
            and (np.sum(state.T[middle_two - 1]) > 0)
        )
    return labels


def main():
    parser = argparse.ArgumentParser(description="Generate new labels for states.")
    parser.add_argument(
        "--states_path",
        help="A file containing the states to label.",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="The dir to which to save label files.",
        type=str,
    )
    parsed_args = parser.parse_args()

    #   parsed_args.states_path = "/home/lyric/Documents/rome-wasnt-built-in-a-day/data_14_100000_10_10/bridges.pkl"
    #  parsed_args.output_dir = "/home/lyric/Documents/rome-wasnt-built-in-a-day/data_14_100000_10_10/"

    states = load_states(parsed_args.states_path)

    labelers = [
        (has_consecutive_bricks_flat_on_top, "consecutive_bricks_flat_on_top.pkl"),
        (is_one_sided, "one_sided.pkl"),
        (has_gap_in_the_middle, "gap_in_the_middle.pkl"),
        (has_small_gap_in_the_middle, "small_gap_in_the_middle.pkl"),
    ]

    for labeler, filename in labelers:
        print(filename)
        labels = labeler(states)
        with open(os.path.join(parsed_args.output_dir, filename), "wb") as f:
            pickle.dump(labels, f)


if __name__ == "__main__":
    main()


# /home/lyric/Documents/rome-wasnt-built-in-a-day/tmp_log_dir/bridges.pkl
