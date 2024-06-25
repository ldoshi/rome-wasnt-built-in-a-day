import unittest

from bridger.go_explore_phase_1 import SuccessEntryGenerator
from gym_bridges.envs.bridges_env import BridgesEnv

class TestGoExplorePhaseOne(unittest.TestCase):
    def test_go_explore_phase_one(self):
        width, num_iterations, num_actions = 6, 10, 8
        success_entry_generator_single_process = SuccessEntryGenerator(
            processes=1,
            width=width,
            env=BridgesEnv(width=width, force_standard_config=True),
            num_iterations=num_iterations,
            num_actions=num_actions,
        )
        success_entry_generator_multiple_process = SuccessEntryGenerator(
            processes=2,
            width=width,
            env=BridgesEnv(width=width, force_standard_config=True),
            num_iterations=num_iterations,
            num_actions=num_actions,
        )
        print(success_entry_generator_single_process.success_entries in success_entry_generator_multiple_process.success_entries)

if __name__ == "__main__":
    unittest.main()
