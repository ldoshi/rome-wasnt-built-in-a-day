from bridger.go_explore_phase_1 import CacheEntry
import time


class CacheEntryDatabase:
    """
    Provides different views into a cache. The cache is a list of CacheEntry objects and can be sorted by trajectory length, steps_since_led_to_something_new, visit count, or sample count.
    """

    def __init__(self, cache_entries: list[CacheEntry]):
        print(f"Initializing CacheEntryDatabase with {len(cache_entries)} entries")
        start_time = time.time()

        # Debug info about the entries
        if cache_entries:
            print(f"First entry type: {type(cache_entries[0])}")
            print(f"First entry attributes: {dir(cache_entries[0])}")
            try:
                print(
                    f"First entry state shape: {cache_entries[0].state_representative.shape}"
                )
            except Exception as e:
                print(f"Error getting state shape: {e}")

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

    def get_top_n_by_sort_key(self, sort_key: "SortKey", n: int):
        """
        Returns the top n cache entries.
        """
        print(f"Getting top {n} entries by {sort_key.key}")
        start_time = time.time()
        try:
            sorted_entries = self.sort_by_key(sort_key())
            result = sorted_entries[:n]
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
        super().__init__("sample_count")
