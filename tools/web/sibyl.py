import argparse
import flask

import object_log_cache

app = flask.Flask(__name__)

_OBJECT_LOG_CACHE = None

# Visualize data

@app.route('/')
def index():

    return 'Web App with Python Flask! ' + str(_OBJECT_LOG_CACHE.get(object_log_cache.STATES_BY_STATE_ID_KEY).keys()) + ' ' + str(_OBJECT_LOG_CACHE.key_hit_counts) + ' ' + str(_OBJECT_LOG_CACHE.key_miss_counts)

@app.route("/training_history_plot_data", methods=["GET"])
def training_history_plot_data():
    start_batch_index = flask.request.args.get("start_batch_index")
    end_batch_index = flask.request.args.get("end_batch_index")
    max_points_per_plot = flask.request.args.get("max_points_per_plot")

    print('start ' , start_batch_index)
    print('end ' , end_batch_index)
    print('max ' , max_points_per_plot)
    
    return {'hi' : 1}


@app.route('/training_history')
def training_history():

    states_by_state_id = _OBJECT_LOG_CACHE.get(object_log_cache.STATES_BY_STATE_ID_KEY)
    a = next(iter(states_by_state_id))
    print(states_by_state_id[a].tolist())
    
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
