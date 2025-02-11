import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from dataclasses import dataclass
from go_explore_phase_1 import StateCache, StateCellManager, CacheEntry


def plot_2d_histogram(cache_entries):
    # Determine grid size from state representation shape
    print(cache_entries)
    grid_shape = cache_entries[0]
    visit_counts = np.zeros(grid_shape.state_representative_encoded[0], dtype=float)
    state_superposition = np.zeros(
        grid_shape.state_representative_encoded[0], dtype=float
    )

    for entry in cache_entries:
        state_superposition += entry.state_representative.numpy()
        visit_counts += state_superposition

    plt.figure(figsize=(8, 6))
    plt.imshow(
        state_superposition + visit_counts,
        origin="lower",
        cmap="hot",
        interpolation="nearest",
    )
    plt.colorbar(label="Visit Frequency + State Influence")
    plt.xlabel("State X")
    plt.ylabel("State Y")
    plt.title("2D Histogram with Superimposed State Representation")
    plt.show()


# Load CacheEntry objects from a pickle file
def load_cache_entries(pickle_file):
    with open(pickle_file, "rb") as f:
        return pickle.load(f)


# Example usage
pickle_file = "/tmp/state_cache-8.pkl"  # Replace with actual path
cache_data = load_cache_entries(pickle_file)

plot_2d_histogram(list(cache_data._cache.values()))
