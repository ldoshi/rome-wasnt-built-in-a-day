"""Enables interactive analysis of logged action inversion reports.

   Usage:
   $ python -m tools.action_inversion_analysis_tool
     --action_inversion_log object_logging_dir/action_inversion_report
     --state_normalized_log object_logging_dir/state_normalized

   This will drop into an interactive prompt with the analyzer object
   already defined. From there, call ActionInversionAnalyzer functions
   like plot_reports. Use the following to list public functions:
   [ function for function in dir(analyzer) if not function.startswith('_')]

"""

import argparse
import collections
import dataclasses
import enum
import IPython
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import torch

from typing import Dict, List, Optional

from bridger.logging import object_log_readers
from bridger.logging import log_entry


class PlotReportsDisplayType(enum.IntEnum):
    _order_ = "EMPTY GROUND BRICK PREFERRED_ACTION POLICY_ACTION MATCHING_ACTION"
    EMPTY = 0
    GROUND = 1
    BRICK = 2
    PREFERRED_ACTION = 3
    POLICY_ACTION = 4
    MATCHING_ACTION = 5


_PLOT_REPORTS_WIDTH = 6
_PLOT_REPORTS_COLOR_MAPPING = {
    PlotReportsDisplayType.EMPTY: "white",
    PlotReportsDisplayType.GROUND: "grey",
    PlotReportsDisplayType.BRICK: "black",
    PlotReportsDisplayType.PREFERRED_ACTION: "lightgreen",
    PlotReportsDisplayType.POLICY_ACTION: "red",
    PlotReportsDisplayType.MATCHING_ACTION: "darkgreen",
}


@dataclasses.dataclass
class DivergenceEntry:
    """Describes an incidence of divergence

    Attributes:
      batch_idx: The batch_idx where the divergence occurs.
      convergence_run_length: The number of batches with 0 action
        inversion reports before this diversion occurred.
      divergence_magnitude: The number of action inversion reports
        logged in the moment of divergence.

    """

    batch_idx: int
    convergence_run_length: int
    divergence_magnitude: int


def _load_reports(
    action_inversion_log: str,
) -> Dict[int, List[log_entry.ActionInversionReportEntry]]:
    reports = collections.defaultdict(list)
    for log_entry in object_log_readers.read_object_log(action_inversion_log):
        reports[log_entry.batch_idx].append(log_entry)

    return reports


def _load_states(state_normalized_log: str) -> Dict[int, torch.Tensor]:
    states = {}
    for log_entry in object_log_readers.read_object_log(state_normalized_log):
        states[log_entry.id] = log_entry.object

    return states


def _build_actions_display(
    report: log_entry.ActionInversionReportEntry, env_width: int
) -> np.ndarray:
    """Returns an array with display values corresponding to action designations per position."""
    actions_display = np.zeros(env_width)

    for i in range(env_width):
        if i in report.preferred_actions and i == report.policy_action:
            actions_display[i] = PlotReportsDisplayType.MATCHING_ACTION
            continue

        if i in report.preferred_actions:
            actions_display[i] = PlotReportsDisplayType.PREFERRED_ACTION
            continue

        if i == report.policy_action:
            actions_display[i] = PlotReportsDisplayType.POLICY_ACTION
            continue

    return actions_display


def _remap_display_data(display_data: np.ndarray) -> colors.ListedColormap:
    """Constructs a color map based on content.

     The values in display_data are updated in place to values
     corresponding to indices in the returned colormap to color
     elements according to type. The renumbering is required to abide
     by matshow requirements that each value corresponds to a colormap
     entry without any gaps to achieve our expected coloring.

    Args:
      display_data: The construction data to display with values
        corresponding to element type (ground, bricks, preferred
        actions, etc). The contents is modified in place to have a new
        meaning for display.

    Returns:
      A colormap with values that correspond to values in the updated
      display_data.

    """
    # The indices of color_mapping must correspond to values in
    # display_data. The color name at an index in color_mapping
    # represents the color to use when rendering the corresponding
    # value in display_data.
    color_mapping = []
    display_types = list(PlotReportsDisplayType)
    current_max_index = len(display_types) - 1
    for i, display_type in enumerate(display_types):
        if (display_data == display_type).any():
            # The display_type exists so we add its color in the
            # corresponding position.
            color_mapping.append(_PLOT_REPORTS_COLOR_MAPPING[display_type])
            assert (len(color_mapping) - 1) == display_type
        else:
            # The current display_type does not occur so we swap in
            # the element type with the highest value which is
            # present.
            while current_max_index >= len(color_mapping):
                candidate = display_types[current_max_index]
                current_max_index -= 1

                if (display_data == candidate).any():
                    display_data[(display_data == candidate)] = display_type
                    color_mapping.append(_PLOT_REPORTS_COLOR_MAPPING[candidate])
                    assert (len(color_mapping) - 1) == display_type
                    break

        if i > current_max_index:
            break

    return colors.ListedColormap(color_mapping)


class ActionInversionAnalyzer:
    def __init__(self, action_inversion_log: str, state_normalized_log: str):
        self._reports = _load_reports(action_inversion_log)
        self._states = _load_states(state_normalized_log)
        self._divergences = self._summarize_divergences()

    def _summarize_divergences(self) -> List[DivergenceEntry]:
        """Summarizes incidents of divergence.

        A divergence is defined as an incident of logging at least one
        action inversion report following a period of one or more
        batches during which 0 action inversion reports logged.

        This function assumes ActionInversionReportEntry's are
        iterated in the same order in which they were logged, i.e. in
        increasing order of batch_idx.

        Returns:
          A list of DivergenceEntry to summarize all the divergences.

        """
        # TODO(lyric): Consider a more efficient data structure if
        # we're still using this tool with much larger log files.
        divergences = []
        last_batch_with_reports = 0
        for (batch_idx, reports) in self._reports.items():
            if len(reports):
                convergence_run_length = batch_idx - last_batch_with_reports - 1
                if convergence_run_length:
                    divergences.append(
                        DivergenceEntry(
                            batch_idx=batch_idx,
                            convergence_run_length=convergence_run_length,
                            divergence_magnitude=len(reports),
                        )
                    )
                last_batch_with_reports = batch_idx

        return divergences

    def _get_divergences(
        self, start_batch_idx: Optional[int] = None, end_batch_idx: Optional[int] = None
    ) -> List[DivergenceEntry]:
        """Returns divergences occurring within the provided range.

        Args:
          start_batch_idx: The first batch index to consider when
            searching for divergences. Defaults to first batch.
          end_batch_idx: The final batch index to consider when
            searching for divergences. The end point is
            inclusive. Defaults to final batch.

        Returns:
          A list of DivergenceEntry to summarize all the divergences
          between start_batch_idx and end_batch_idx.

        """
        start = 0
        if start_batch_idx is not None:
            for i, entry in enumerate(self._divergences):
                if start_batch_idx <= entry.batch_idx:
                    break
                start = i + 1

        end = len(self._divergences)
        if end_batch_idx is not None:
            for i in range(len(self._divergences) - 1, -1, -1):
                if self._divergences[i].batch_idx <= end_batch_idx:
                    break
                end = i

        return self._divergences[start:end]

    def plot_incidence_rate(
        self, start_batch_idx: Optional[int] = None, end_batch_idx: Optional[int] = None
    ) -> None:
        """Visualizes the number of action inversion reports per batch.

        Args:
          start_batch_idx: The first batch index to consider when
            searching for divergences. Defaults to first batch.
          end_batch_idx: The final batch index to consider when
            searching for divergences. The end point is
            inclusive. Defaults to final batch.

        """
        xs = []
        ys = []
        for (batch_idx, reports) in self._reports.items():
            if start_batch_idx is not None and batch_idx < start_batch_idx:
                continue
            if end_batch_idx is not None and batch_idx > end_batch_idx:
                break

            xs.append(batch_idx)
            ys.append(len(reports))

        fig, axs = plt.subplots(1, 1)
        axs.bar(xs, ys)
        axs.set_title("Number of Action Inversion Reports Per Batch")
        axs.set_xlabel("Batch")
        axs.set_ylabel("Action Inversion Reports")
        plt.ion()
        plt.show()

    def plot_divergences(
        self, start_batch_idx: Optional[int] = None, end_batch_idx: Optional[int] = None
    ) -> None:
        """Visualizes the incidents of divergence in the provided range.

        A divergence is defined as an incident of logging at least one
        action inversion report following a period of one or more
        batches during which 0 action inversion reports logged.

        Args:
          start_batch_idx: The first batch index to consider when
            searching for divergences. Defaults to first batch.
          end_batch_idx: The final batch index to consider when
            searching for divergences. The end point is
            inclusive. Defaults to final batch.

        """
        divergences = self._get_divergences(
            start_batch_idx=start_batch_idx, end_batch_idx=end_batch_idx
        )
        xs = []
        ys = []
        for divergence in divergences:
            xs.append(divergence.batch_idx)
            ys.append(divergence.divergence_magnitude)

        fig, axs = plt.subplots(1, 1)
        axs.bar(xs, ys)
        axs.set_title("Divergence Magnitudes Across Batches")
        axs.set_xlabel("Batch")
        axs.set_ylabel("Divegence Magnitude (# Reports Logged)")
        plt.ion()
        plt.show()

    def print_divergences(
        self,
        start_batch_idx: Optional[int] = None,
        end_batch_idx: Optional[int] = None,
        n: Optional[int] = None,
        sort_by_convergence_run_length: bool = False,
        sort_by_divergence_magnitude: bool = False,
        return_divergences: bool = False,
    ) -> Optional[List[DivergenceEntry]]:
        """Prints a summary of the divergences.

        A divergence is defined as an incident of logging at least one
        action inversion report following a period of one or more
        batches during which 0 action inversion reports logged.

        Args:
          start_batch_idx: The first batch index to consider when
            searching for divergences. Defaults to first batch.
          end_batch_idx: The final batch index to consider when
            searching for divergences. The end point is
            inclusive. Defaults to final batch.
          n: The max number of divergences matching the provided
            criteria to print out.
          sort_by_convergence_run_length: Sort divergence incidents by
            convergence run length, longest to shortest. The default
            sort order of batch_idx becomes the secondary sort field.
          sort_by_divergence_magnitude: Sort divergence incidents by
            their divergence magnitude, largest to smallest. The
            default sort order of batch_idx becomes the secondary sort
            field. The sort order is (convergence run length,
            divergence magnitude, batch_idx) if both
            sort_by_convergence_run_length and
            sort_by_divergence_magnitude are provided.
          return_divergences: Return the list of divergence entries
            matching the provided criteria if True.

        Returns:
          The list of divergence entries that were printed if
          return_divergences.

        """
        divergences = self._get_divergences(
            start_batch_idx=start_batch_idx, end_batch_idx=end_batch_idx
        )
        count = n if n is not None else len(divergences)

        print("Divergence Summary")
        print(f"Printing {count} of {len(divergences)} entries.")

        if not divergences:
            if return_divergences:
                return []
            return

        if sort_by_convergence_run_length and sort_by_divergence_magnitude:
            sort_key = lambda entry: (
                -entry.convergence_run_length,
                -entry.divergence_magnitude,
                entry.batch_idx,
            )
        elif sort_by_convergence_run_length:
            sort_key = lambda entry: (-entry.convergence_run_length, entry.batch_idx)
        elif sort_by_divergence_magnitude:
            sort_key = lambda entry: (-entry.divergence_magnitude, entry.batch_idx)
        else:
            sort_key = lambda entry: entry.batch_idx

        divergences_sorted = sorted(divergences, key=sort_key)

        # Print all numbers using the width of the widest to ensure
        # consistent columns.
        display_width = max(len(f"{divergences[-1].batch_idx}"), 3)
        print(
            "Column order: batch index (BI), convergence run length "
            "(CRL), divergence magnitude (DM)"
        )
        print()
        print(
            f"{'BI':{display_width}}  "
            f"{'CRL':{display_width}}  "
            f"{'DM':{display_width}}"
        )
        for entry in divergences_sorted[:count]:
            print(
                f"{entry.batch_idx:{display_width}d}  "
                f"{entry.convergence_run_length:{display_width}d}  "
                f"{entry.divergence_magnitude:{display_width}d}"
            )

        if return_divergences:
            return divergences_sorted

    def plot_reports(self, batch_idx: int, width: int = _PLOT_REPORTS_WIDTH) -> None:
        """Visualizes the action inversion reports for a batch.

        Plots a visual representation of the state, the preferred
        actions, and the policy action for each action inversion
        report for a batch. The state is represented with ground as
        grey and bricks as black. The preferred actions are shown as
        light green squares above the state. The policy action will be
        dark green if it aligns with a preferred action and red
        otherwise to indicate a suboptimal policy.

        Args:
          batch_idx: The batch for which to visualize reports.
          width: The max number of states to plot in a row before
            moving to the next row.

        """

        reports = self._reports[batch_idx]
        if not reports:
            print(f"No action inversion reports for batch_idx: {batch_idx}.")
            return

        width = min(width, len(reports))
        height = (len(reports) + width - 1) // width

        fig, axs = plt.subplots(height, width)
        # Make axs a 2D array in all cases for consistent access below.
        if type(axs) != np.ndarray:
            axs = np.array([axs])
        axs = axs.reshape(height, width)

        for i, report in enumerate(reports):
            state = self._states[report.state_id]
            actions_display = _build_actions_display(
                report=report, env_width=state.shape[1]
            )
            state_and_actions_display = np.vstack((actions_display, state))
            cmap = _remap_display_data(display_data=state_and_actions_display)
            x = i % width
            y = i // width
            ax = axs[y, x]
            ax.matshow(state_and_actions_display, cmap=cmap)

        fig.suptitle(f"Action Inversion Reports for Batch {batch_idx}")
        plt.ion()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Load action inversion reports for analysis."
    )
    parser.add_argument(
        "--action_inversion_log",
        help="The filepath to the ActionInversionReportEntry log file.",
        required=True,
    )
    parser.add_argument(
        "--state_normalized_log",
        help="The filepath to the NormalizedLogEntry log file for states.",
        required=True,
    )
    args = parser.parse_args()

    analyzer = ActionInversionAnalyzer(
        action_inversion_log=args.action_inversion_log,
        state_normalized_log=args.state_normalized_log,
    )
    print(
        "\nWelcome to the Action Inversion Analyzer!\n"
        "To render plots, please first run:\n"
        "  %matplotlib\n\n"
        "An instance of ActionInversionAnalyzer called analyzer has been created.\n"
        "Execute the following to list its public functions:\n"
        "  [ function for function in dir(analyzer) if not function.startswith('_')]\n"
    )
    IPython.embed()


if __name__ == "__main__":
    main()
