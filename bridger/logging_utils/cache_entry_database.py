from bridger.go_explore_phase_1 import CacheEntry


class CacheEntryDatabase:
    """
    Provides different views into a cache. The cache is a list of CacheEntry objects and can be sorted by trajectory length, steps_since_led_to_something_new, visit count, or sample count.
    """

    def __init__(self, cache_entries: list[CacheEntry]):
        self.cache_entries = cache_entries

    def sort_by_key(self, sort_key: "SortKey"):
        """
        Sorts the cache entries using the provided sort key.
        """
        self.cache_entries = sorted(self.cache_entries, key=sort_key)

    def get_top_n_by_sort_key(self, sort_key: "SortKey", n: int):
        """
        Returns the top n cache entries.
        """
        return self.sort_by_key(sort_key)[:n]


class SortKey:
    def __init__(self, key: str):
        self.key = key

    def __call__(self, cache_entry: CacheEntry):
        return getattr(cache_entry, self.key)


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
