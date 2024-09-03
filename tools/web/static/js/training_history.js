let _MAX_RENDER_COUNT = 20

let _CHART_OPTIONS_TEMPLATE = {
    scales: {
	x: {
	    title: {
		display: true,
		color: "#FFFFFF",
		font: {
                    size: 12,
		}
            },
	    ticks: {
		color: "#FFFFFF",
		font: {
                    size: 12,
		}
	    }
	},
        y: {
	    ticks: {
		color: "#FFFFFF",
		font: {
                    size: 12,
		}
	    },
        },
    },
    plugins: {
	title: {
	    color: "#FFFFFF",
	    display: true,
	    font: {
                size: 16,
	    }
	},
	legend: {
	    align: 'end',
	    labels: {
		boxWidth: 8,
		color: "#FFFFFF",
	    }
	},
    },
    elements: {
	point: {
	    pointStyle: 'cross',
	    radius: 1,
	    hoverRadius: 10,
	},
	line: {
	    borderWidth: 1,
	}
    },
};

// TODO(lyric): Create a better color list.
let _COLORS = [
    'rgba(255, 0, 0, 1)',
    'rgba(0, 255, 0, 1)',
    'rgba(0, 0, 255, 1)',
    'rgba(255, 255, 0, 1)',
    'rgba(0, 255, 255, 1)',
    'rgba(255, 0, 255, 1)',
    'rgba(127, 0, 0, 1)',
    'rgba(0, 127, 0, 1)',
    'rgba(0, 0, 127, 1)',
    'rgba(127, 127, 0, 1)',
    'rgba(0, 127, 127, 1)',
    'rgba(127, 0, 127, 1)',
]

let _DATA = null;

let _STATE_FILTER = null;

function update_plots() {
    let plot_params = get_plot_params();

    remove_load_error_indicator();
    add_loading_indicator();

    $.get(`${_ROOT_URL}training_history_plot_data`,
	  plot_params,
	  function(data, response) {
	      _DATA = data;
	      render_plots();
	      $("#current-experiment-name").html(plot_params['experiment_name']);
    }).fail(function() {
	add_load_error_indicator();
    }).always(function() {
	remove_loading_indicator();
    });    
}

function update_state_filter() {
    _STATE_FILTER = new Function('state', $("#state-filter-function-body").val());
    render_plots();
}

function get_plot_params() {
    let experiment_name = $("#experiment-name").val();
    let start_batch_idx = $("#start-batch-idx").val();
    let end_batch_idx = $("#end-batch-idx").val();
    let max_points_per_series = $("#max-points-per-series").val();
    let number_of_states = $("#number-of-states").val();

    return {
	"experiment_name" : experiment_name,
	"start_batch_idx" : start_batch_idx,
	"end_batch_idx" : end_batch_idx,
	"max_points_per_series" : max_points_per_series,
	"number_of_states" : number_of_states,
    }
}

function update_url_params() {
    let url_param_updates = get_plot_params();
    url_param_updates["state_filter_function_body"] = $("#state-filter-function-body").val();
    update_url(url_param_updates);
}

function render_plots() {
    update_url_params();
    
    if (_DATA == null) {
	return;
    }
    
    let plot_data = _DATA['plot_data'];
    if (plot_data.length == 0) {
	return;
    }

    let state_index_list = []
    let render_count = Math.min(plot_data.length, _MAX_RENDER_COUNT)

    for (let state_index = 0; state_index < plot_data.length; state_index++) {
	if (_STATE_FILTER(plot_data[state_index]['state'])) {
	    state_index_list.push(state_index);
	}

	if (state_index_list.length == render_count) {
	    break;
	}
    }
    
    create_plot_div_structure(state_index_list.length, plot_data[0]['metrics'].length);

    for (let i = 0; i < state_index_list.length; i++) {
	state_index = state_index_list[i]
	let row_data = plot_data[state_index];
	render_state_plot(i, row_data);
	for (let metric_index = 0; metric_index < row_data['metrics'].length; metric_index++) {
	    let metric_entry = row_data['metrics'][metric_index];
	    render_training_plot(metric_entry['metric'], i, metric_index, _DATA['labels'], metric_entry['series_data'], metric_entry['series_labels'], metric_entry['plot_type']);
	}
    }
}

function create_plot_div_structure(state_count, metric_count) {
    // Creates the plot-related div structure for all the rows of
    // data.

    let plot_divs = "";
    for (let i = 0; i < state_count; i++) {
	plot_divs += `<div class="plots-row plots-row-background-${i % 2}">`;
	plot_divs += `<div id="plot-holder-${i}-state" class="plot-holder-state"></div>`;
	for (let j = 0; j < metric_count; j++) {
	    plot_divs += `<div id="plot-holder-${i}-metric-${j}" class="plot-holder-metric plot-holder-metric-default-width"></div>`;
	}
	plot_divs += '</div>';
    }
    $("#plots-holder").html(plot_divs)
}

function render_training_plot(metric, state_index, metric_index, labels, series_data, series_labels, plot_type) {
    let canvas_id = `plot-canvas-${state_index}-metric-${metric_index}`;
    training_plot_html = `<canvas id="${canvas_id}" class="plot-canvas"></canvas>`;
    $(`#plot-holder-${state_index}-metric-${metric_index}`).html(training_plot_html);
    
    let datasets = [];
    for (let i = 0; i < series_data.length; i++) {
	let dataset = {};
	dataset['data'] = series_data[i];
	dataset['label'] = series_labels[i];
	dataset['borderColor'] = _COLORS[i];
	dataset['backgroundColor'] = _COLORS[i];
	datasets.push(dataset);
    }
    
    chart_options = structuredClone(_CHART_OPTIONS_TEMPLATE);
    chart_options['plugins']['title']['text'] = metric;
    chart_options['scales']['x']['title']['text'] = 'Batch Index';
    new Chart($(`#${canvas_id}`), {
	type: plot_type,
	data: {
            labels: labels,
            datasets: datasets
	},
	options: chart_options,
    });    
}

function render_state_plot(state_index, data) {
    let canvas_id = `state-plot-state-canvas-${state_index}`
    let state_plot_html = `<div class="state-plot-info">Visits: ${data['visit_count']}</div>`;
    state_plot_html += `<div id="state-plot-state-${state_index}" class="state-plot-state"><canvas id="${canvas_id}" class="plot-canvas"></canvas></div>`;
    $(`#plot-holder-${state_index}-state`).html(state_plot_html);

    render_array_2d(data['state'], canvas_id);
}

function zoom_in_charts() {
    $("#zoom-button-full").addClass("button-selected");
    $("#zoom-button-full").removeClass("button-unselected");
    
    $("#zoom-button-default").addClass("button-unselected");
    $("#zoom-button-default").removeClass("button-selected");

    $(".plot-holder-metric").addClass("plot-holder-metric-full-width");
    $(".plot-holder-metric").removeClass("plot-holder-metric-default-width");
}

function zoom_default_charts() {
    $("#zoom-button-default").addClass("button-selected");
    $("#zoom-button-default").removeClass("button-unselected");

    $("#zoom-button-full").addClass("button-unselected");
    $("#zoom-button-full").removeClass("button-selected");

    $(".plot-holder-metric").addClass("plot-holder-metric-default-width");
    $(".plot-holder-metric").removeClass("plot-holder-metric-full-width");
}

