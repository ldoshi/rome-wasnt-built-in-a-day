import argparse
import flask
import time

from typing import Optional

import object_log_cache
import plot_utils

app = flask.Flask(__name__)

_OBJECT_LOG_CACHE = None


def _get_int_or_none(name: str) -> Optional[int]:
    value = flask.request.args.get(name)
    return int(value) if value else None


@app.route("/training_history_plot_data", methods=["GET"])
def training_history_plot_data():
    """Provides plot data on states and metrics based on filters.

    This endpoint is intended to respond to an AJAX call."""
    start = int(time.time() * 1e3)
    start_batch_index = _get_int_or_none("start_batch_index")
    end_batch_index = _get_int_or_none("end_batch_index")
    max_points_per_series = _get_int_or_none("max_points_per_series")
    number_of_states = _get_int_or_none("number_of_states")

    training_history_database = _OBJECT_LOG_CACHE.get(
        object_log_cache.TRAINING_HISTORY_DATABASE_KEY
    )

    states = training_history_database.get_states_by_visit_count(
        n=number_of_states,
        start_batch_index=start_batch_index,
        end_batch_index=end_batch_index,
    )
    plot_data = []

    min_batch_index = start_batch_index
    max_batch_index = end_batch_index

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
            for action in range(training_history_database.actions_n):
                df = data_fn(
                    state_id=row["state_id"],
                    action=action,
                    start_batch_index=start_batch_index,
                    end_batch_index=end_batch_index,
                )
                df = plot_utils.downsample(df=df, n=max_points_per_series)
                series_data.append(
                    [
                        {"x": df_row["batch_idx"], "y": df_row[metric]}
                        for df_index, df_row in df.iterrows()
                    ]
                )
                series_labels.append(str(action))
                min_batch_index = (
                    min(min_batch_index, df["batch_idx"].min())
                    if min_batch_index is not None
                    else df["batch_idx"].min()
                )
                max_batch_index = (
                    max(max_batch_index, df["batch_idx"].max())
                    if max_batch_index is not None
                    else df["batch_idx"].max()
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
    print(f"Sibly training_history_plot_data took {end-start} ms.")
    return {
        "plot_data": plot_data,
        "labels": list(range(min_batch_index, max_batch_index + 1)),
    }

@app.route("/action_inversion_plot_data", methods=["GET"])
def action_inversion_plot_data():
    """Provides summary plot data based on filters.

    This endpoint is intended to respond to an AJAX call."""
    start = int(time.time() * 1e3)
    start_batch_index = _get_int_or_none("start_batch_index")
    end_batch_index = _get_int_or_none("end_batch_index")

    action_inversion_database = _OBJECT_LOG_CACHE.get(
        object_log_cache.ACTION_INVERSION_DATABASE_KEY
    )

    series_data = []
    series_labels = []
    incidence_rate_batch_idxs, incidence_rate_report_counts = action_inversion_database.get_incidence_rate(start_batch_index=start_batch_index, end_batch_index=end_batch_index)
    series_data.append([{"x": batch_idx, "y": report_count} for batch_idx, report_count in zip(incidence_rate_batch_idxs, incidence_rate_report_counts)])
    series_labels.append("Action Inversion Reports")

    min_batch_index = incidence_rate_batch_idxs[0] if incidence_rate_batch_idxs else 0
    max_batch_index = incidence_rate_batch_idxs[-1] if incidence_rate_batch_idxs else 0

    divergences = action_inversion_database.get_divergences(start_batch_index=start_batch_index, end_batch_index=end_batch_index)
    series_data.append([{"x": divergence.batch_idx, "y" : divergence.divergence_magnitude}   for divergence in divergences])
    series_labels.append("Divergence Magnitude")
    if divergences:
        min_batch_index = min(min_batch_index, divergences[0].batch_idx)
        max_batch_index = max(max_batch_index, divergences[-1].batch_idx)

    end = int(time.time() * 1e3)
    print(f"Sibly action_inversion_plot_data took {end-start} ms.")
    return {
        "title": "Action Inversion Reports and Divergence Magnitudes",
        "series_data": series_data,
        "series_labels": series_labels,
        "labels" : list(range(min_batch_index, max_batch_index + 1)),
    }

@app.route("/")
@app.route("/training_history")
def training_history():
    return flask.render_template("training_history.html")

@app.route("/action_inversion")
def action_inversion():
    return flask.render_template("action_inversion.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Sibyl debugger.")
    parser.add_argument(
        "--log_dir",
        help="The path to the object logging dir.",
        required=True,
    )
    args = parser.parse_args()
    _OBJECT_LOG_CACHE = object_log_cache.ObjectLogCache(log_dir=args.log_dir)

    app.run(host="0.0.0.0")
