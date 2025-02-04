import unittest
from bridger import config

from bridger.go_explore_phase_1 import SuccessEntryGenerator
from gym_bridges.envs.bridges_env import BridgesEnv


class Hparams:
    def __init__(self):
        self.go_explore_epsilon_1 = 0.001
        self.go_explore_epsilon_2 = 0.00001
        self.go_explore_pa = 0.5
        self.go_explore_wa_sampled = 0.1
        self.go_explore_wa_led_to_something_new = 0
        self.go_explore_wa_times_visited = 0.3
        self.cell_manager = "state_cell_manager"


class TestGoExplorePhaseOne(unittest.TestCase):
    def test_go_explore_phase_one(self):
        width, num_iterations, num_actions = 6, 4, 4

        hparams = Hparams()

        success_entry_generator_single_process = SuccessEntryGenerator(
            processes=1,
            width=width,
            env=BridgesEnv(width=width, force_standard_config=True),
            num_iterations=num_iterations,
            num_actions=num_actions,
            hparams=hparams,
        )
        success_entry_generator_multiple_process = SuccessEntryGenerator(
            processes=2,
            width=width,
            env=BridgesEnv(width=width, force_standard_config=True),
            num_iterations=num_iterations,
            num_actions=num_actions,
            hparams=hparams,
        )

        print(success_entry_generator_single_process.success_entries)
        print("-------------------")
        print(success_entry_generator_multiple_process.success_entries)

        self.assertTrue(
            success_entry_generator_single_process.success_entries
            in success_entry_generator_multiple_process.success_entries
        )


if __name__ == "__main__":
    unittest.main()
