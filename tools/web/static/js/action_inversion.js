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
let _COLORS_BACKGROUND = [
    'rgba(99, 255, 132, 0.6)',
    'rgba(255, 0, 0, 0.6)',
]

let _COLORS_BORDER = [
    'rgba(99, 255, 132, 1)',
    'rgba(255, 0, 0, 1)',
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

    $.get(`${_ROOT_URL}action_inversion_plot_data`, { "start_batch_index": start_batch_index, "end_batch_index" : end_batch_index}, function(data, response) {
	render_action_inversion_plot(data);
    });    
}

function update_batch_reports() {
    let batch_index = $("#view-batch-reports-batch-index").val();
    alert(batch_index);
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

function render_action_inversion_plot(data) {
    let datasets = [];

    $("#plot-holder").html('<canvas id="action-inversion-canvas" class="plot-canvas"></canvas>');

    for (let i = 0; i < data['series_data'].length; i++) {
	let dataset = structuredClone(_DATASET_TEMPLATE);
	dataset['data'] = data['series_data'][i];
	dataset['label'] = data['series_labels'][i];
	dataset['borderColor'] = _COLORS_BORDER[i];
	dataset['backgroundColor'] = _COLORS_BACKGROUND[i];
	datasets.push(dataset);
    }

    chart_options = structuredClone(_CHART_OPTIONS_TEMPLATE);
    chart_options['plugins']['title']['text'] = data['title'];
    chart_options['scales']['x']['title']['text'] = 'Batch Index';
    chart_options['scales']['y']['title']['text'] = 'Number of Reports';
    new Chart($("#action-inversion-canvas"), {
	type: 'bar',
	data: {
            labels: data['labels'],
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
