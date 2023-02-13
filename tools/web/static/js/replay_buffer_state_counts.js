let _MAX_RENDER_COUNT = 20;

let _CHART_OPTIONS_TEMPLATE = {
  scales: {
    x: {
      title: {
        display: true,
        color: "#FFFFFF",
        font: {
          size: 12,
        },
      },
      ticks: {
        color: "#FFFFFF",
        font: {
          size: 12,
        },
      },
    },
    y: {
      ticks: {
        color: "#FFFFFF",
        font: {
          size: 12,
        },
      },
    },
  },
  plugins: {
    title: {
      color: "#FFFFFF",
      display: true,
      font: {
        size: 16,
      },
    },
    legend: {
      align: "end",
      labels: {
        boxWidth: 8,
        color: "#FFFFFF",
      },
    },
  },
  elements: {
    point: {
      pointStyle: "cross",
      radius: 1,
      hoverRadius: 10,
    },
    line: {
      borderWidth: 1,
    },
  },
};

// TODO(lyric): Create a better color list.
let _COLORS = [
  "rgba(255, 0, 0, 1)",
  "rgba(0, 255, 0, 1)",
  "rgba(0, 0, 255, 1)",
  "rgba(255, 255, 0, 1)",
  "rgba(0, 255, 255, 1)",
  "rgba(255, 0, 255, 1)",
  "rgba(127, 0, 0, 1)",
  "rgba(0, 127, 0, 1)",
  "rgba(0, 0, 127, 1)",
  "rgba(127, 127, 0, 1)",
  "rgba(0, 127, 127, 1)",
  "rgba(127, 0, 127, 1)",
];

let _DATA = null;

function update_plots() {
  let current_batch_idx = $("#current-batch-idx").val();

  $.get(
    `${_ROOT_URL}replay_buffer_state_counts_plot_data`,
    {
      current_batch_idx: current_batch_idx,
    },
    function (data, response) {
      _DATA = data;
      render_plots();
      console.log(_DATA);
    }
  );
}

function generate_histogram() {
  let x_vals = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
  let y_vals = [5, 8, 24, 16, 32, 42, 30, 17, 11];
  const data = x_vals.map((k, i) => ({ x: k, y: y_vals[i] }));

  const backgroundColor = Array(x_vals.length).fill("rgba(255, 99, 132, 0.6)");
  const borderColor = Array(x_vals.length).fill("rgba(255, 99, 132, 1)");

  const ctx = document.getElementById("histogram").getContext("2d");
  const histogram = new Chart(ctx, {
    type: "bar",
    data: {
      datasets: [
        {
          label: "Number of instances of states in replay buffer",
          data: data,
          backgroundColor: backgroundColor,
          borderColor: borderColor,
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
          offset: false,
          grid: {
            offset: false,
          },
          ticks: {
            stepSize: 1,
          },
          title: {
            display: true,
            text: "State id",
            font: {
              size: 14,
            },
          },
        },
        y: {
          // beginAtZero: true
          title: {
            display: true,
            text: "Visits",
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
          callbacks: {
            title: (items) => {
              if (!items.length) {
                return "";
              }
              const item = items[0];
              const x = item.parsed.x;
              const min = x - 0.5;
              const max = x + 0.5;
              return `State id: ${min}`;
            },
          },
        },
      },
    },
  });
}

function render_plots() {
  if (_DATA == null) {
    return;
  }

  let plot_data = _DATA["plot_data"][1];
  console.log(plot_data);

  if (plot_data.length == 0) {
    return;
  }

  // create_plot_div_structure(1, 1);

  // for (let i = 0; i < plot_data.length; i++) {
  //   let row_data = plot_data[i];
  //   render_histogram_plot(i, row_data);
  //   for (
  //     let metric_index = 0;
  //     metric_index < row_data["metrics"].length;
  //     metric_index++
  //   ) {
  //     let metric_entry = row_data["metrics"][metric_index];
  //     render_training_plot(
  //       metric_entry["metric"],
  //       i,
  //       metric_index,
  //       _DATA["labels"],
  //       metric_entry["series_data"],
  //       metric_entry["series_labels"]
  //     );
  //   }
  // }
  generate_histogram();
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
    plot_divs += "</div>";
  }
  $("#plots-holder").html(plot_divs);
}

function render_training_plot(
  metric,
  state_index,
  metric_index,
  labels,
  series_data,
  series_labels
) {
  let canvas_id = `plot-canvas-${state_index}-metric-${metric_index}`;
  training_plot_html = `<canvas id="${canvas_id}" class="plot-canvas"></canvas>`;
  $(`#plot-holder-${state_index}-metric-${metric_index}`).html(
    training_plot_html
  );

  let datasets = [];
  for (let i = 0; i < series_data.length; i++) {
    let dataset = {};
    dataset["data"] = series_data[i];
    dataset["label"] = series_labels[i];
    dataset["borderColor"] = _COLORS[i];
    dataset["backgroundColor"] = _COLORS[i];
    datasets.push(dataset);
  }

  chart_options = structuredClone(_CHART_OPTIONS_TEMPLATE);
  chart_options["plugins"]["title"]["text"] = metric;
  chart_options["scales"]["x"]["title"]["text"] = "Batch Index";
  new Chart($(`#${canvas_id}`), {
    type: "line",
    data: {
      labels: labels,
      datasets: datasets,
    },
    options: chart_options,
  });
}

function render_histogram_plot(state_index, data) {
  let canvas_id = `state-plot-state-canvas-${state_index}`;
  let state_plot_html = `<div class="state-plot-info">Visits: ${data["visit_count"]}</div>`;
  state_plot_html += `<div id="state-plot-state-${state_index}" class="state-plot-state"><canvas id="${canvas_id}" class="plot-canvas"></canvas></div>`;
  $(`#plot-holder-${state_index}-state`).html(state_plot_html);

  render_array_2d(data["state"], canvas_id);
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
