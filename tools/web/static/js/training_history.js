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

let _DATASET_TEMPLATE = {
    borderWidth: 1,
    radius: 1,
    hoverRadius: 2,
    hoverBorderWidth: 2
};

function update_plots() {
    let start_batch_index = $("#start-batch-index").val();
    let end_batch_index = $("#end-batch-index").val();
    let max_points_per_series = $("#max-points-per-series").val();
    let number_of_states = $("#number-of-states").val();

    $.get(`${_ROOT_URL}training_history_plot_data`, { "start_batch_index": start_batch_index, "end_batch_index" : end_batch_index, "max_points_per_series" : max_points_per_series, "number_of_states" : number_of_states}, function(data, response) {
	let plot_data = data['plot_data'];
	if (plot_data.length == 0) {
	    return;
	}

	create_plot_div_structure(plot_data.length, plot_data[0]['metrics'].length);

	for (let state_index = 0; state_index < plot_data.length; state_index++) {
	    let row_data = plot_data[state_index];
	    render_state_plot(state_index, row_data);
	    for (let metric_index = 0; metric_index < row_data['metrics'].length; metric_index++) {
		let metric_entry = row_data['metrics'][metric_index];
		render_training_plot(metric_entry['metric'], state_index, metric_index, data['labels'], metric_entry['series_data'], metric_entry['series_labels']);
	    }
	}
    });    
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

function render_training_plot(metric, state_index, metric_index, labels, series_data, series_labels) {
    let canvas_id = `plot-canvas-${state_index}-metric-${metric_index}`;
    training_plot_html = `<canvas id="${canvas_id}" class="plot-canvas"></canvas>`;
    $(`#plot-holder-${state_index}-metric-${metric_index}`).html(training_plot_html);
    
    let datasets = [];
    for (let i = 0; i < series_data.length; i++) {
	let dataset = structuredClone(_DATASET_TEMPLATE);
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
	type: 'line',
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
