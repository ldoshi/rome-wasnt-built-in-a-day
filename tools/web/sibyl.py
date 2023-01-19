import argparse
import flask
import os
import time

from typing import Optional

from tools.web import object_log_cache
from tools.web import plot_utils

app = flask.Flask(__name__)

_OBJECT_LOG_CACHE = None
_LOG_DIR = None


def _get_string_or_none(name: str) -> Optional[str]:
    return flask.request.args.get(name)


def _get_int_or_none(name: str) -> Optional[int]:
    value = flask.request.args.get(name)

    try:
        return int(value)
    except:
        return None


@app.route("/training_history_plot_data", methods=["GET"])
def training_history_plot_data():
    """Provides plot data on states and metrics based on filters.

    This endpoint is intended to respond to an AJAX call."""
    start = int(time.time() * 1e3)
    experiment_name = _get_string_or_none("experiment_name")
    start_batch_idx = _get_int_or_none("start_batch_idx")
    end_batch_idx = _get_int_or_none("end_batch_idx")
    max_points_per_series = _get_int_or_none("max_points_per_series")
    number_of_states = _get_int_or_none("number_of_states")

    training_history_database = _OBJECT_LOG_CACHE.get(
        experiment_name=experiment_name,
        data_key=object_log_cache.TRAINING_HISTORY_DATABASE_KEY,
    )

    states = training_history_database.get_states_by_visit_count(
        n=number_of_states,
    )
    plot_data = []

    min_batch_idx = start_batch_idx
    max_batch_idx = end_batch_idx if end_batch_idx is not None else start_batch_idx

    metrics_and_data_fns = [
        ("td_error", "TD Error", training_history_database.get_td_errors),
        ("q_value", "Q", training_history_database.get_q_values),
        ("q_target_value", "Q Target", training_history_database.get_q_target_values),
    ]
    for index, row in states.iterrows():
        state_plot_data = {
            "visit_count": row["visit_count"],
            "state": row["state"].tolist(),
            "metrics": [],
        }

        for metric, metric_display_name, data_fn in metrics_and_data_fns:
            series_data = []
            series_labels = []
            for action in range(training_history_database.nA):
                batch_idxs, values = data_fn(
                    state_id=row["state_id"],
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
                }
            )

        plot_data.append(state_plot_data)

    end = int(time.time() * 1e3)
    print(f"Sibyl training_history_plot_data took {end-start} ms.")
    return {
        "plot_data": plot_data,
        "labels": list(range(min_batch_idx, max_batch_idx + 1)),
    }


@app.route("/action_inversion_plot_data", methods=["GET"])
def action_inversion_plot_data():
    """Provides summary plot data based on filters.

    This endpoint is intended to respond to an AJAX call."""
    start = int(time.time() * 1e3)
    experiment_name = _get_string_or_none("experiment_name")
    start_batch_idx = _get_int_or_none("start_batch_idx")
    end_batch_idx = _get_int_or_none("end_batch_idx")

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
    experiment_name = _get_string_or_none("experiment_name")
    batch_idx = _get_int_or_none("batch_idx")
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


@app.route("/")
@app.route("/training_history")
def training_history():
    experiment_names = sorted(os.listdir(_LOG_DIR))
    return flask.render_template(
        "training_history.html", experiment_names=experiment_names
    )


@app.route("/action_inversion")
def action_inversion():
    experiment_names = sorted(os.listdir(_LOG_DIR))
    return flask.render_template(
        "action_inversion.html", experiment_names=experiment_names
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
    _OBJECT_LOG_CACHE = object_log_cache.ObjectLogCache(log_dir=_LOG_DIR)

    app.run(host="0.0.0.0", port=5001)
