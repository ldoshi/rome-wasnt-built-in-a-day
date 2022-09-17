import argparse
import collections
import dataclasses
import enum
import IPython
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import torch

from typing import Dict, List, Optional

from bridger.logging import object_log_readers
from bridger.logging import log_entry

class ShowReportsDisplayType(enum.IntEnum):
    _order_ = "EMPTY GROUND BRICK PREFERRED_ACTION POLICY_ACTION MATCHING_ACTION"
    EMPTY = 0
    GROUND = 1
    BRICK = 2
    PREFERRED_ACTION = 3
    POLICY_ACTION = 4
    MATCHING_ACTION = 5

_SHOW_REPORTS_WIDTH = 6
_SHOW_REPORTS_COLOR_MAPPING = {
    ShowReportsDisplayType.EMPTY: 'white',
    ShowReportsDisplayType.GROUND: 'grey',
    ShowReportsDisplayType.BRICK: 'black',
    ShowReportsDisplayType.PREFERRED_ACTION: 'lightgreen',
    ShowReportsDisplayType.POLICY_ACTION: 'red',
    ShowReportsDisplayType.MATCHING_ACTION: 'darkgreen'
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


def _load_reports(action_inversion_log: str) -> Dict[int, log_entry.ActionInversionReportEntry]:
    reports = collections.defaultdict(list)
    for log_entry in object_log_readers.read_object_log(action_inversion_log):
        reports[log_entry.batch_idx].append(log_entry)

    return reports

def _load_states(state_normalized_log: str) -> Dict[int, torch.Tensor]:
    states = {}
    for log_entry in object_log_readers.read_object_log(state_normalized_log):
        states[log_entry.id] = log_entry.object

    return states

def _build_actions_display(report: log_entry.ActionInversionReportEntry, env_width: int) -> np.ndarray:
    actions_display = np.zeros(env_width)

    for i in range(env_width):
        if i in report.preferred_actions and i == report.policy_action:
            actions_display[i] = _SHOW_REPORTS_MATCHING_ACTION
            continue

        if i in report.preferred_actions:
            actions_display[i] = _SHOW_REPORTS_PREFERRED_ACTION
            continue

        if i == report.policy_action:
            actions_display[i] = _SHOW_REPORTS_POLICY_ACTION
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
    display_types = list(ShowReportsDisplayType)
    current_max_index = len(display_types) - 1
    for i, display_type in enumerate(display_types):
        if (display_data == display_type).any():
            # The display_type exists so we add its color in the corresponding position.
            color_mapping.append(_SHOW_REPORTS_COLOR_MAPPING[display_type])
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
                    color_mapping.append(_SHOW_REPORTS_COLOR_MAPPING[candidate])
                    assert (len(color_mapping) - 1) == display_type
                    break
                
        if i > current_max_index:
            break

                
    return colors.ListedColormap(color_mapping)
    

class ActionInversionAnalyzer:
    
    
    def __init__(self, action_inversion_log: str, state_normalized_log: str):
        self._reports = _load_reports(action_inversion_log)
        self._states = _load_states(state_normalized_log)
        

    def summarize_divergences(self, start_batch_idx: Optional[int] = None, end_batch_idx: Optional[int] = None) -> List[DivergenceEntry]:
        """Summarizes incidents of divergence in the provided range.

        A divergence is defined as a period of one or more batches where
        there were 0 action inversion reports logged.

        Args:
          reports: The logged action inversion reports.
          start_batch_idx: The first batch index to consider when
            searching for divergences. Defaults to first batch.
          end_batch_idx: The final batch index to consider when
            searching for divergences. The end point is
            inclusive. Defaults to final batch.

        Returns:
        A list of DivergenceEntry to summarize all the divergences.

        """
        # TODO(lyric): Consider a more efficient data structure if
        # we're still using this tool with much larger log files.
        divergences = []
        convergence_run_length = 0
        for (batch_idx, reports) in self._reports.items():
            if start_batch_idx is not None and batch_idx < start_batch_idx:
                continue
            if end_batch_idx is not None and batch_idx > end_batch_idx:
                break

            print(batch_idx)
            
            if len(reports):
                if convergence_run_length:
                    divergences.append(DivergenceEntry(batch_idx=batch_idx, convergence_run_length=convergence_run_length, divergence_magnitude=len(reports)))
                    convergence_run_length = 0
                
            else:
                convergence_run_length += 1
                
        return divergences

    def show_reports(self, batch_idx: int, width: int = _SHOW_REPORTS_WIDTH):

        reports = self._reports[batch_idx]
        if not reports:
            print(f"No action inversion reports for batch_idx: {batch_idx}.")

        width = min(width, len(reports))
        height = (len(reports) + width - 1) // width
            
        fig, axs = plt.subplots(            height, width                    )
        # If either dimension is 1, the default uses a 1D array
        # instead of a 2D array. Make it 2D here in all cases for
        # consistent access below.
        axs = axs.reshape(height, width)

        for i, report in enumerate(reports):
            state = self._states[report.state_id]
            actions_display = _build_actions_display(report=report, env_width=state.shape[1])
            state_and_actions_display = np.vstack((actions_display, state))
            cmap = _remap_display_data(display_data=state_and_actions_display)
            x = i % width
            y = i // width
            ax = axs[y, x]
            ax.matshow(state_and_actions_display, cmap=cmap)

        plt.show()
    
def main():
    """Checks that two log files of training batch entries share the same values
    across all attributes.
    Example to run:
    $ python -m bin.action_inversion_analysis_tool
      --action_inversion_log object_logging_dir/action_inversion_report
      --state_normalized_log object_logging_dir/state_normalized
    """

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

    analyzer = ActionInversionAnalyzer(action_inversion_log=args.action_inversion_log, state_normalized_log= args.state_normalized_log)
    print(analyzer.summarize_divergences(end_batch_idx=20))
    analyzer.show_reports(5)
    #IPython.embed()

#if __name__ == "__main__":
#    main()

analyzer = ActionInversionAnalyzer(action_inversion_log=action_inversion_log, state_normalized_log= state_normalized_log)
analyzer.show_reports(5)
