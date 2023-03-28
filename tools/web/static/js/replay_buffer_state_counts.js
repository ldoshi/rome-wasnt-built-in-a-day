let _DATA = null;

let _CHART_OPTIONS_TEMPLATE = {
  type: "bar",
  data: {
    datasets: [
      {
        label: "Replay Buffer State Counts",
        borderWidth: 1,
        barPercentage: 1,
        categoryPercentage: 1,
        borderRadius: 5,
      },
    ],
  },
  options: {
    scales: {
      x: {
        type: "linear",
        offset: true,
        grid: {
          offset: false,
        },
        ticks: {
          stepSize: 1,
        },
        title: {
          display: true,
          text: "state id",
          font: {
            size: 14,
          },
        },
      },
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: "state counts",
          font: {
            size: 14,
          },
        },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {},
      },
    },
  },
};

function update_plots() {
  let experiment_name = $("#experiment-name").val();

  remove_load_error_indicator();
  add_loading_indicator();

  $.get(
    `${_ROOT_URL}replay_buffer_state_counts_plot_data`,
    {
      experiment_name: experiment_name,
    },
    function (data, response) {
      _DATA = data;
      render_plots();
      $("#current-experiment-name").html(experiment_name);
    }).fail(function() {
	add_load_error_indicator();
    }).always(function() {
	remove_loading_indicator();
    });    
}

function render_plots() {
  // Grab the data from the global _DATA object.
  if (_DATA == null) {
    return;
  }

  const total_replay_buffer_state_counts =
    _DATA["total_replay_buffer_state_counts"];

  let data = [];

  for (let [state_id, total_count] of Object.entries(
    total_replay_buffer_state_counts
  )) {
    data.push({ x: state_id, y: total_count });
  }

  const background_color = "rgba(255, 99, 132, 0.6)";
  const border_color = "rgba(255, 99, 132, 1)";

  histogram_html = "<canvas id=histogram></canvas>";
  $("#histogram-holder").html(histogram_html);

  let chart_options = structuredClone(_CHART_OPTIONS_TEMPLATE);
  chart_options["data"]["datasets"][0]["data"] = data;
  chart_options["data"]["datasets"][0]["backgroundColor"] = background_color;
  chart_options["data"]["datasets"][0]["borderColor"] = border_color;
  chart_options["options"]["plugins"]["tooltip"]["callbacks"]["title"] = (
    items
  ) => {
    if (!items.length) {
      return "";
    }
    const item = items[0];
    const state_id = item.parsed.x;
    return `State id: ${state_id}`;
  };
  new Chart($("#histogram"), chart_options);
}
