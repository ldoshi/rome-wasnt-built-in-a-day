let _BATCH_REPORTS_PER_ROW = 6;

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
    hoverBorderWidth: 2
};

function get_plot_params() {
    let experiment_name = $("#experiment-name").val();
    let start_batch_idx = $("#start-batch-idx").val();
    let end_batch_idx = $("#end-batch-idx").val();

    return {
	"experiment_name" : experiment_name,
	"start_batch_idx" : start_batch_idx,
	"end_batch_idx" : end_batch_idx,
    }
}

function update_plots() {
    let plot_params = get_plot_params();

    remove_load_error_indicator();
    add_loading_indicator();

    $.get(`${_ROOT_URL}action_inversion_plot_data`,
	  plot_params,
	  function(data, response) {
	      render_action_inversion_plot(data);
	      update_url_params();
	      $("#current-experiment-name").html(plot_params['experiment_name']);
    }).fail(function() {
	add_load_error_indicator();
    }).always(function() {
	remove_loading_indicator();
    });    
}

function update_url_params() {
  let url_param_updates = get_plot_params();
  update_url(url_param_updates);    
}

function update_batch_reports() {
    let experiment_name = $("#experiment-name").val();
    let batch_idx = $("#view-batch-reports-batch-idx").val();

    remove_load_error_indicator();
    add_loading_indicator();

    $.get(`${_ROOT_URL}action_inversion_batch_reports`,
	  {
	      "experiment_name" : experiment_name,
	      "batch_idx": batch_idx
	  },
	  function(data, response) {
	      create_batch_report_div_structure(data.length);
	      render_batch_reports(data);
    }).fail(function() {
	      add_load_error_indicator();
    }).always(function() {
	remove_loading_indicator();
    });    
}

function create_batch_report_div_structure(count) {
    let divs = "";
    for (let i = 0; i < count; i++) {
	if (i % _BATCH_REPORTS_PER_ROW == 0) {
	    divs += '<div class="batch-reports-row">';
	}

	divs += `<div class="batch-report"><canvas id="batch-report-canvas-${i}" class="plot-canvas"></canvas></div>`
	
	if (i % _BATCH_REPORTS_PER_ROW == (_BATCH_REPORTS_PER_ROW - 1) || i == (count -1)) {
	    divs += '</div>';
	}
    }
    $("#batch-report-holder").html(divs);
}

function render_batch_reports(data) {
    for (let i = 0; i < data.length; i++) {
	report = data[i]
        let canvas_id = `batch-report-canvas-${i}`;
	render_array_2d(report['state'], canvas_id, report['preferred_actions'], [report['policy_action']]);
    }
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
