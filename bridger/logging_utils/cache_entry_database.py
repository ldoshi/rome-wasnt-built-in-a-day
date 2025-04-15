from bridger.go_explore_phase_1 import CacheEntry
import time
import numpy as np


class CacheEntryDatabase:
    """
    Provides different views into a cache. The cache is a list of CacheEntry objects and can be sorted by trajectory length, steps_since_led_to_something_new, visit count, or sample count.
    """

    def __init__(self, cache_entries: list[CacheEntry]):
        print(f"Initializing CacheEntryDatabase with {len(cache_entries)} entries")
        start_time = time.time()

        self.cache_entries = cache_entries
        end_time = time.time()
        print(
            f"CacheEntryDatabase initialization took {end_time - start_time:.2f} seconds"
        )

    def sort_by_key(self, sort_key: "SortKey"):
        """
        Sorts the cache entries using the provided sort key.
        """
        print(f"Sorting cache entries by {sort_key.key}")
        start_time = time.time()
        try:
            self.cache_entries = sorted(self.cache_entries, key=sort_key)
            end_time = time.time()
            print(f"Sorting completed in {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error during sorting: {e}")
            raise

    def get_top_n_by_sort_key(
        self, sort_key: "SortKey", n: int, ascending: bool = False
    ):
        """
        Returns the top n cache entries along with their metric values.

        Args:
            sort_key: The key to sort by
            n: Number of entries to return
            ascending: If True, sort in ascending order, otherwise descending (default)

        Returns:
            A dictionary containing:
            - states: List of state representations
            - values: List of corresponding metric values
        """
        print(f"Getting top {n} entries by {sort_key.key}")
        start_time = time.time()
        try:
            self.sort_by_key(sort_key)
            if ascending:
                entries = self.cache_entries[:n].copy()
            else:
                entries = self.cache_entries[-n:].copy()
                entries.reverse()

            # Convert values to Python native types, handling both numbers and tuples
            def convert_value(value):
                if isinstance(value, (np.int64, np.int32)):
                    return int(value)
                elif isinstance(value, tuple):
                    return [
                        int(x) if isinstance(x, (np.int64, np.int32)) else x
                        for x in value
                    ]
                return value

            result = {
                "states": [
                    cache_entry.state_representative.tolist() for cache_entry in entries
                ],
                "values": [
                    convert_value(getattr(cache_entry, sort_key.key))
                    for cache_entry in entries
                ],
            }

            end_time = time.time()
            print(f"Retrieved top {n} entries in {end_time - start_time:.2f} seconds")
            return result
        except Exception as e:
            print(f"Error getting top entries: {e}")
            raise


class SortKey:
    def __init__(self, key: str):
        self.key = key

    def __call__(self, cache_entry: CacheEntry):
        try:
            value = getattr(cache_entry, self.key)
            return value
        except Exception as e:
            print(f"Error accessing {self.key} on cache entry: {e}")
            raise


class TrajectorySortKey(SortKey):
    def __init__(self):
        super().__init__("trajectory")


class StepsSinceLedToSomethingNewSortKey(SortKey):
    def __init__(self):
        super().__init__("steps_since_led_to_something_new")


class StepsSinceLedToSomethingNewResetCountSortKey(SortKey):
    def __init__(self):
        super().__init__("steps_since_led_to_something_new_reset_count")


class VisitCountSortKey(SortKey):
    def __init__(self):
        super().__init__("visit_count")


class SampleCountSortKey(SortKey):
    def __init__(self):
        super().__init__("sampled_count")
