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

    states_by_state_id = _OBJECT_LOG_CACHE.get(object_log_cache.STATES_BY_STATE_ID_KEY)
    training_history_database = _OBJECT_LOG_CACHE.get(object_log_cache.TRAINING_HISTORY_DATABASE_KEY)

    states = training_history_database.get_states_by_visit_count(n=number_of_states, start_batch_index=start_batch_index, end_batch_index=end_batch_index)
    plot_data = []

    min_batch_index = start_batch_index
    max_batch_index = end_batch_index

    metrics_and_data_fns = [("td_error", "TD Error", training_history_database.get_td_errors),
                            ("q_value", "Q" , training_history_database.get_q_values),
                            ("q_target_value", "Q Target" , training_history_database.get_q_target_values)]
    for index, row in states.iterrows():
        state_plot_data = {'visit_count' : row['visit_count'], 'state' : states_by_state_id[row['state_id']].tolist(), 'metrics' : []}

        for metric, metric_display_name, data_fn in metrics_and_data_fns:
            series_data = []
            series_labels = []
            for action in range(training_history_database.actions_n):
                df = data_fn(state_id=row['state_id'], action=action, start_batch_index=start_batch_index, end_batch_index=end_batch_index)
                df = plot_utils.downsample(df=df, n=max_points_per_series)
                series_data.append([{"x" : df_row["batch_idx"], "y" : df_row[metric]}        for df_index, df_row in df.iterrows()])
                series_labels.append(str(action))
                min_batch_index = min(min_batch_index, df['batch_idx'].min()) if min_batch_index is not None else df['batch_idx'].min()
                max_batch_index = max(max_batch_index, df['batch_idx'].max()) if max_batch_index is not None else df['batch_idx'].max()

            state_plot_data['metrics'].append({
                'metric' : metric_display_name,
                'series_data' : series_data,
                'series_labels' : series_labels
            })
            
        plot_data.append(state_plot_data)    

    end = int(time.time() * 1e3)
    print(f"Sibly training_history_plot_data took {end-start} ms.")
    return {'plot_data' : plot_data, 'labels' : list(range(min_batch_index, max_batch_index + 1))}


@app.route('/training_history')
def training_history():    
    return flask.render_template('training_history.html')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Load Sibyl debugger."
    )
    parser.add_argument(
        "--log_dir",
        help="The path to the object logging dir.",
        required=True,
    )
    args = parser.parse_args()
    _OBJECT_LOG_CACHE = object_log_cache.ObjectLogCache(log_dir=args.log_dir)
    
    app.run(host='0.0.0.0')
