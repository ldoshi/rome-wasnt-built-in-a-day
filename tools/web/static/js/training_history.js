// Set default values?
function update_plots() {

    // get data.
    let start_batch_index = $("#start-batch-index").val();
    let end_batch_index = $("#end-batch-index").val();
    let max_points_per_plot = $("#max-points-per-plot").val();

    $.get(`${_ROOT_URL}training_history_plot_data`, { "start_batch_index": start_batch_index, "end_batch_index" : end_batch_index, "max_points_per_plot" : max_points_per_plot}, function(data, response) {
	alert(data);
    });    
    
    
    // render it.

//    create_plot_div_structure(3);
  //  render_state_plot(0, null);
  //  render_state_plot(1, null);
   // render_state_plot(2, null);
}

function create_plot_div_structure(state_count) {
    // Creates the plot-related div structure for all the rows of
    // data.

    let plot_divs = "";
    for (let i = 0; i < state_count; i++) {
	plot_divs += '<div class="plots-row">';
	plot_divs += `<div id="state-plot-holder-${i}" class="plot-holder state-plot-holder">`;
	plot_divs += '</div>';
	plot_divs += '</div>';
    }
    $("#plots-holder").html(plot_divs)
}

function render_state_plots(states) {
    
}

function render_state_plot(state_index, data) {
    // Set the info and canvas html structure.
    let canvas_id = `state-plot-state-canvas-${state_index}`
    let state_plot_divs = '<div class="state-plot-info">hello</div>';
    state_plot_divs += `<div id="state-plot-state-${state_index}" class="state-plot-state"><canvas id="${canvas_id}" class="state-plot-state-canvas"></canvas></div>`;
    $(`#state-plot-holder-${state_index}`).html(state_plot_divs);

    data = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 2, 0], [0, 0, 0, 0, 2, 2], [1, 0, 0, 0, 0, 1]];

    render_array_2d(data, canvas_id);
}


