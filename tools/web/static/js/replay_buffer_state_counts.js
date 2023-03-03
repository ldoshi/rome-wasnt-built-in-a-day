let _DATA = null;

function update_plots() {
  let experiment_name = $("#experiment-name").val();

  $.get(
    `${_ROOT_URL}replay_buffer_state_counts_plot_data`,
    {
      experiment_name: experiment_name,
    },
    function (data, response) {
      _DATA = data;
      render_plots();
      $("#current-experiment-name").html(experiment_name);
    }
  );
}

function generate_histogram() {
  let x_vals = [];
  let y_vals = [];

  const orig_data = _DATA["plot_data"][1][0];

  for (let i = 0; i < orig_data.length; i++) {
    x_vals.push(orig_data[i][0]);
    y_vals.push(orig_data[i][1]);
  }

  const data = x_vals.map((k, i) => ({ x: k, y: y_vals[i] }));

  const backgroundColor = Array(x_vals.length).fill("rgba(255, 99, 132, 0.6)");
  const borderColor = Array(x_vals.length).fill("rgba(255, 99, 132, 1)");

  // Remove existing canvas histogram element if one exists.
  const existing_ctx = document.getElementById("histogram");
  if (existing_ctx) {
    existing_ctx.remove();
  }

  $(`#histogram-holder`).append(`<canvas id="histogram"></canvas>`);
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
              return `State id: ${x}`;
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

  if (plot_data.length == 0) {
    return;
  }

  generate_histogram();
}
