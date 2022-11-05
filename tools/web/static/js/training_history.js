let _CHART_OPTIONS_TEMPLATE = {
    scales: {
        y: {
	    min: 0,
        },
    }
};

//    chart_options['scales']['y']['max'] = 100;
//    chart_options['scales']['y']['title'] = {display: true, text: 'Percent of Day'}

// Add color list.
let _COLORS = [
    'rgba(255, 0, 0, 1)',
    'rgba(0, 255, 0, 1)',
    'rgba(0, 0, 255, 1)',
    'rgba(255, 255, 0, 1)',
    'rgba(0, 255, 255, 1)',
    'rgba(255, 0, 255, 1)',
    ]

let _DATASET_TEMPLATE = {
    borderWidth: 1,
    radius: 1,
    hoverRadius: 2,
    hoverBorderWidth: 2
};


// Set default values?
function update_plots() {

    // get data.
    let start_batch_index = $("#start-batch-index").val();
    let end_batch_index = $("#end-batch-index").val();
    let max_points_per_series = $("#max-points-per-series").val();
    let number_of_states = $("#number-of-states").val();

    $.get(`${_ROOT_URL}training_history_plot_data`, { "start_batch_index": start_batch_index, "end_batch_index" : end_batch_index, "max_points_per_series" : max_points_per_series, "number_of_states" : number_of_states}, function(data, response) {
	let plot_data = data['plot_data'];
	create_plot_div_structure(plot_data.length);
	for (let i = 0; i < plot_data.length; i++) {
	    let row_data = plot_data[i];
	    render_state_plot(i, row_data);
	    render_training_plot("td-error", i, data['labels'], row_data['td_error']['series_data'], row_data['td_error']['series_labels']);
	}

	
    });    
}

function create_plot_div_structure(state_count) {
    // Creates the plot-related div structure for all the rows of
    // data.

    let plot_divs = "";
    for (let i = 0; i < state_count; i++) {
	plot_divs += `<div class="plots-row plots-row-background-${i % 2}">`;
	plot_divs += `<div id="state-plot-holder-${i}" class="state-plot-holder"></div>`;
	plot_divs += `<div id="td-error-plot-holder-${i}" class="training-plot-holder"></div>`;
	plot_divs += `<div id="q-plot-holder-${i}" class="training-plot-holder"></div>`;
	plot_divs += `<div id="q-target-plot-holder-${i}" class="training-plot-holder"></div>`;	
	plot_divs += '</div>';
    }
    $("#plots-holder").html(plot_divs)
}

function render_training_plot(training_identifier, state_index, labels, series_data, series_labels) {
    // Add things like title, axis control. min and max. fix axis label colors. legend formatting.
    // Plot zoom or enlarge panel ideas. how about a button to make charts full width and wrap/stack!.
    let canvas_id = `${training_identifier}-plot-canvas-${state_index}`;
    training_plot_html = `<canvas id="${canvas_id}" class="plot-canvas"></canvas>`;
    $(`#${training_identifier}-plot-holder-${state_index}`).html(training_plot_html);
    
    let datasets = []
    for (let i = 0; i < series_data.length; i++) {
	let dataset = structuredClone(_DATASET_TEMPLATE);
	dataset['data'] = series_data[i];
	dataset['label'] = series_labels[i];
	dataset['borderColor'] = _COLORS[i];
	dataset['backgroundColor'] = _COLORS[i];
	datasets.push(dataset);
    }
    
    chart_options = structuredClone(_CHART_OPTIONS_TEMPLATE);

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
    // Set the info and canvas html structure.
    let canvas_id = `state-plot-state-canvas-${state_index}`
    let state_plot_html = `<div class="state-plot-info">Visits: ${data['visit_count']}</div>`;
    state_plot_html += `<div id="state-plot-state-${state_index}" class="state-plot-state"><canvas id="${canvas_id}" class="plot-canvas"></canvas></div>`;
    $(`#state-plot-holder-${state_index}`).html(state_plot_html);

    render_array_2d(data['state'], canvas_id);
}


