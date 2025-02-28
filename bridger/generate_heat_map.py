import matplotlib.pyplot as plt
import numpy as np
from go_explore_phase_1 import CacheEntry, StateCache, StateCellManager
import argparse
from bridger.logging_utils.object_log_readers import read_object_log


def plot_2d_histogram(cache_entries: list[CacheEntry]) -> None:
    """
    Plots a 2D histogram based on the state representations from a list of CacheEntry objects.

    This function visualizes the combined influence of state representations and visit frequencies
    from a given list of CacheEntry objects. The resulting heatmap shows how frequently different
    states have been visited and their cumulative influence.

    Args:
        cache_entries (list[CacheEntry]): A list of CacheEntry objects containing state representations.

    Behavior:
        - Extracts the state representation shape to determine the grid size.
        - Computes the cumulative `state_superposition` and `visit_counts`.
        - Flips the resulting matrix vertically for correct orientation.
        - Displays a heatmap using Matplotlib with appropriate axis labels and a color scale.

    Visualization:
        - X-axis: State X coordinates.
        - Y-axis: State Y coordinates.
        - Color intensity: Represents the combined influence of visit frequency and state values.

    Returns:
        None. Displays the generated 2D histogram.

    """
    if not cache_entries:
        print("Error: No cache entries found.")
        return
    grid = cache_entries[0].state_representative.numpy()
    grid_shape = grid.shape
    visit_counts = np.zeros(grid_shape, dtype=float)
    state_superposition = np.zeros(grid_shape, dtype=float)

    for entry in cache_entries:
        state_superposition += entry.state_representative.numpy()
        visit_counts += state_superposition

    plt.figure(figsize=(8, 6))
    plt.imshow(
        np.flipud(state_superposition + visit_counts),
        origin="lower",
        cmap="hot",
        interpolation="nearest",
    )
    plt.colorbar(label="Visit Frequency + State Influence")

    # Set x and y ticks for all values
    x_ticks = np.arange(grid_shape[1])
    y_ticks = np.arange(grid_shape[0])

    plt.xticks(x_ticks)  # Show all x-axis values
    plt.yticks(y_ticks)  # Show all y-axis values

    plt.ticklabel_format(useOffset=False, style="plain")  # Disable scientific notation

    plt.xlabel("State X")
    plt.ylabel("State Y")
    plt.title("2D Histogram with Superimposed State Representation")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pickle_file",
        type=str,
        required=True,
        help="Path to pickle file containing OccurrenceEntry object",
    )
    args = parser.parse_args()
    pickle_file = args.pickle_file

    cache_entries = read_object_log(args.pickle_file)
    first_entry = next(cache_entries, None)

    if first_entry is None:
        print("Error: No cache entries found.")
    else:
        plot_2d_histogram(list(first_entry.object._cache.values()))
