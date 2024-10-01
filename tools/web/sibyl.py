import argparse
import flask
import os
import threading
import time
import json

from typing import Any, Optional

from bridger.logging_utils import log_entry
from tools.web import object_log_cache
from tools.web import plot_utils
from collections import Counter

app = flask.Flask(__name__)

_OBJECT_LOG_CACHE = None
_LOG_DIR = None
_SIBYL_TMP_DIR = "sibyl_tmp"

_START_BATCH_IDX_DEFAULT_VALUE = -1
_MAX_POINTS_PER_SERIES_DEFAULT_VALUE = 200
_NUMBER_OF_STATES_DEFAULT_VALUE = 10
_STATE_FILTER_FUNCTION_BODY_DEFAULT_VALUE = """
// Compose the body of a state filter function here. Feel free to delete these comments.
//
// Assume a function with Args:
//   state: A 2D array representing the state. The row index comes first and 0 is the top row.
//
// Returns:
//   True if the state should be displayed here, false otherwise.

return true;
"""

_EXPERIMENT_NAME = "experiment_name"
_START_BATCH_IDX = "start_batch_idx"
_END_BATCH_IDX = "end_batch_idx"
_MAX_POINTS_PER_SERIES = "max_points_per_series"
_NUMBER_OF_STATES = "number_of_states"
_STATE_FILTER_FUNCTION_BODY = "state_filter_function_body"

_BATCH_IDX = "batch_idx"


def _get_string_or_default(name: str, default: Optional[str] = None) -> Optional[str]:
    value = flask.request.args.get(name)
    return value if value is not None else default


def _get_int_or_default(name: str, default: Optional[int] = None) -> Optional[int]:
    value = flask.request.args.get(name)

    try:
        return int(value)
    except:
        return default


def _get_experiment_names() -> list[str]:
    return [
        x
        for x in sorted(os.listdir(_LOG_DIR))
        if x != log_entry.STATE_NORMALIZED_LOG_ENTRY
    ]


@app.route("/training_history_plot_data", methods=["GET"])
def training_history_plot_data():
    """Provides plot data on states and metrics based on filters.

    This endpoint is intended to respond to an AJAX call."""
    start = int(time.time() * 1e3)
    experiment_name = _get_string_or_default(_EXPERIMENT_NAME)
    start_batch_idx = _get_int_or_default(_START_BATCH_IDX)
    end_batch_idx = _get_int_or_default(_END_BATCH_IDX)
    max_points_per_series = _get_int_or_default(_MAX_POINTS_PER_SERIES)
    number_of_states = _get_int_or_default(_NUMBER_OF_STATES)

    training_history_database = _OBJECT_LOG_CACHE.get(
        experiment_name=experiment_name,
        data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
    )

    hparams_data = None
    with open(os.path.join(_LOG_DIR, experiment_name, "hparams.json")) as f:
        hparams_data = json.load(f)

    state_visits = training_history_database.get_states_by_visit_count(
        n=number_of_states,
    )
    plot_data = []

    min_batch_idx = start_batch_idx
    max_batch_idx = end_batch_idx if end_batch_idx is not None else start_batch_idx

    metrics_and_data_fns = [
        ("td_error", "TD Error", training_history_database.get_td_errors, "line"),
        ("q_value", "Q", training_history_database.get_q_values, "line"),
        (
            "q_target_value",
            "Q Target",
            training_history_database.get_q_target_values,
            "line",
        ),
        (
            "batch_action_frequency",
            "Sampled in Batch",
            training_history_database.get_batch_action_frequencies,
            "scatter",
        ),
    ]
    for visit_entry in state_visits:
        state_plot_data = {
            "visit_count": visit_entry.visit_count,
            "state": visit_entry.state.tolist(),
            "state_idx": visit_entry.state_id,
            "metrics": [],
        }

        for metric, metric_display_name, data_fn, plot_type in metrics_and_data_fns:
            series_data = []
            series_labels = []
            for action in range(training_history_database.nA):
                batch_idxs, values = data_fn(
                    state_id=visit_entry.state_id,
                    action=action,
                    start_batch_idx=start_batch_idx,
                    end_batch_idx=end_batch_idx,
                )
                batch_idxs = plot_utils.downsample_list(
                    data=batch_idxs, n=max_points_per_series
                )
                values = plot_utils.downsample_list(
                    data=values, n=max_points_per_series
                )
                series_data.append(
                    [
                        {"x": batch_idx, "y": value}
                        for batch_idx, value in zip(batch_idxs, values)
                    ]
                )
                series_labels.append(str(action))

                # add tests in object_log_readers.
                # fix handling for min and max if batch_idxs is empty.
                if not batch_idxs:
                    continue

                min_batch_idx = (
                    min(min_batch_idx, batch_idxs[0])
                    if min_batch_idx is not None
                    else batch_idxs[0]
                )
                max_batch_idx = (
                    max(max_batch_idx, batch_idxs[-1])
                    if max_batch_idx is not None
                    else batch_idxs[-1]
                )

            state_plot_data["metrics"].append(
                {
                    "metric": metric_display_name,
                    "series_data": series_data,
                    "series_labels": series_labels,
                    "plot_type": plot_type,
                }
            )

        plot_data.append(state_plot_data)

    end = int(time.time() * 1e3)
    print(f"Sibyl training_history_plot_data took {end-start} ms.")
    return {
        "plot_data": plot_data,
        "hparams_data": hparams_data,
        "labels": list(range(min_batch_idx, max_batch_idx + 1)),
    }


@app.route("/replay_buffer_state_counts_plot_data", methods=["GET"])
def replay_buffer_state_counts_plot_data():
    """Provides plot data on states and metrics based on filters.

    This endpoint is intended to respond to an AJAX call."""
    start = int(time.time() * 1e3)
    experiment_name = _get_string_or_default(_EXPERIMENT_NAME)

    training_history_database = _OBJECT_LOG_CACHE.get(
        experiment_name=experiment_name,
        data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
    )

    # State visits are sorted by most visited to least visited.
    replay_buffer_states_by_visit_count = [
        visit_entry.state.tolist()
        for visit_entry in training_history_database.get_states_by_visit_count()
    ]

    (
        _,
        replay_buffer_state_counts_by_batch,
    ) = training_history_database.get_replay_buffer_state_counts()

    # Sum the replay buffer_state_counts.
    total_replay_buffer_state_counts = Counter()
    for batch_replay_buffer_state_counts in replay_buffer_state_counts_by_batch:
        total_replay_buffer_state_counts += Counter(batch_replay_buffer_state_counts)

    end = int(time.time() * 1e3)
    print(f"Sibyl replay buffer state counts took {end-start} ms.")
    return {
        "total_replay_buffer_state_counts": total_replay_buffer_state_counts,
        "replay_buffer_states_by_visit_count": replay_buffer_states_by_visit_count,
    }


@app.route("/action_inversion_plot_data", methods=["GET"])
def action_inversion_plot_data():
    """Provides summary plot data based on filters.

    This endpoint is intended to respond to an AJAX call."""
    start = int(time.time() * 1e3)
    experiment_name = _get_string_or_default(_EXPERIMENT_NAME)
    start_batch_idx = _get_int_or_default(_START_BATCH_IDX)
    end_batch_idx = _get_int_or_default(_END_BATCH_IDX)

    action_inversion_database = _OBJECT_LOG_CACHE.get(
        experiment_name=experiment_name,
        data_key=object_log_cache.ACTION_INVERSION_DATABASE_KEY,
    )

    series_data = []
    series_labels = []
    (
        incidence_rate_batch_idxs,
        incidence_rate_report_counts,
    ) = action_inversion_database.get_incidence_rate(
        start_batch_idx=start_batch_idx, end_batch_idx=end_batch_idx
    )
    series_data.append(
        [
            {"x": batch_idx, "y": report_count}
            for batch_idx, report_count in zip(
                incidence_rate_batch_idxs, incidence_rate_report_counts
            )
        ]
    )
    series_labels.append("Action Inversion Reports")

    min_batch_idx = incidence_rate_batch_idxs[0] if incidence_rate_batch_idxs else 0
    max_batch_idx = incidence_rate_batch_idxs[-1] if incidence_rate_batch_idxs else 0

    divergences = action_inversion_database.get_divergences(
        start_batch_idx=start_batch_idx, end_batch_idx=end_batch_idx
    )
    series_data.append(
        [
            {"x": divergence.batch_idx, "y": divergence.divergence_magnitude}
            for divergence in divergences
        ]
    )
    series_labels.append("Divergence Magnitude")
    if divergences:
        min_batch_idx = min(min_batch_idx, divergences[0].batch_idx)
        max_batch_idx = max(max_batch_idx, divergences[-1].batch_idx)

    end = int(time.time() * 1e3)
    print(f"Sibly action_inversion_plot_data took {end-start} ms.")
    return {
        "title": "Action Inversion Reports and Divergence Magnitudes",
        "series_data": series_data,
        "series_labels": series_labels,
        "labels": list(range(min_batch_idx, max_batch_idx + 1)),
    }


@app.route("/action_inversion_batch_reports", methods=["GET"])
def action_inversion_batch_reports():
    """Provides action inversion reports for the requested batch.

    This endpoint is intended to respond to an AJAX call."""
    start = int(time.time() * 1e3)
    experiment_name = _get_string_or_default(_EXPERIMENT_NAME)
    batch_idx = _get_int_or_default(_BATCH_IDX)
    if batch_idx is None:
        return []

    action_inversion_database = _OBJECT_LOG_CACHE.get(
        experiment_name=experiment_name,
        data_key=object_log_cache.ACTION_INVERSION_DATABASE_KEY,
    )

    reports = action_inversion_database.get_reports(batch_idx=batch_idx)
    results = []
    for report, state in reports:
        results.append(
            {
                "preferred_actions": list(report.preferred_actions),
                "policy_action": report.policy_action,
                "state": state.tolist(),
            }
        )

    end = int(time.time() * 1e3)
    print(f"Sibly action_inversion_plot_data took {end-start} ms.")
    return results


@app.route("/", methods=["GET"])
@app.route("/training_history", methods=["GET"])
def training_history():
    experiment_names = _get_experiment_names()
    selected_experiment_name = _get_string_or_default(
        name=_EXPERIMENT_NAME, default=experiment_names[0]
    )
    start_batch_idx = _get_int_or_default(
        name=_START_BATCH_IDX, default=_START_BATCH_IDX_DEFAULT_VALUE
    )
    end_batch_idx = _get_int_or_default(_END_BATCH_IDX)
    max_points_per_series = _get_int_or_default(
        name=_MAX_POINTS_PER_SERIES, default=_MAX_POINTS_PER_SERIES_DEFAULT_VALUE
    )
    number_of_states = _get_int_or_default(
        name=_NUMBER_OF_STATES, default=_NUMBER_OF_STATES_DEFAULT_VALUE
    )
    state_filter_function_body = _get_string_or_default(
        name=_STATE_FILTER_FUNCTION_BODY,
        default=_STATE_FILTER_FUNCTION_BODY_DEFAULT_VALUE,
    )

    return flask.render_template(
        "training_history.html",
        experiment_names=experiment_names,
        selected_experiment_name=selected_experiment_name,
        start_batch_idx=start_batch_idx,
        end_batch_idx=end_batch_idx,
        max_points_per_series=max_points_per_series,
        number_of_states=number_of_states,
        state_filter_function_body=state_filter_function_body,
    )


@app.route("/replay_buffer_state_counts")
def replay_buffer_state_counts():
    experiment_names = sorted(os.listdir(_LOG_DIR))
    selected_experiment_name = _get_string_or_default(
        name=_EXPERIMENT_NAME, default=experiment_names[0]
    )
    return flask.render_template(
        "replay_buffer_state_counts.html",
        experiment_names=experiment_names,
        selected_experiment_name=selected_experiment_name,
    )


@app.route("/action_inversion")
def action_inversion():
    experiment_names = sorted(os.listdir(_LOG_DIR))
    selected_experiment_name = _get_string_or_default(
        name=_EXPERIMENT_NAME, default=experiment_names[0]
    )
    start_batch_idx = _get_int_or_default(
        name=_START_BATCH_IDX, default=_START_BATCH_IDX_DEFAULT_VALUE
    )
    end_batch_idx = _get_int_or_default(_END_BATCH_IDX)

    return flask.render_template(
        "action_inversion.html",
        experiment_names=experiment_names,
        selected_experiment_name=selected_experiment_name,
        start_batch_idx=start_batch_idx,
        end_batch_idx=end_batch_idx,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Sibyl debugger.")
    parser.add_argument(
        "--log_dir",
        help="The path to the object logging dir.",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    _LOG_DIR = args.log_dir
    _OBJECT_LOG_CACHE = object_log_cache.ObjectLogCache(
        log_dir=_LOG_DIR, temp_dir=_SIBYL_TMP_DIR
    )

    convert_logs_background_thread = threading.Thread(
        target=_OBJECT_LOG_CACHE.convert_logs_to_saved_databases,
        args=(_get_experiment_names(),),
    )
    convert_logs_background_thread.start()

    app.run(host="0.0.0.0", port=5001)
