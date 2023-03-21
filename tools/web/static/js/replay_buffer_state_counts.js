let _DATA = null;

let _CHART_OPTIONS_TEMPLATE = {
  // type: "bar",
  // data: {
  //   datasets: [
  //     {
  //       label: "Replay Buffer State Counts",
  //       borderWidth: 1,
  //       barPercentage: 1,
  //       categoryPercentage: 1,
  //       borderRadius: 5,
  //     },
  //   ],
  // },
  // options: {
  //   scales: {
  //     x: {
  //       type: "linear",
  //       offset: false,
  //       grid: {
  //         offset: false,
  //       },
  //       ticks: {
  //         stepSize: 1,
  //       },
  //       title: {
  //         display: true,
  //         text: "state id",
  //         font: {
  //           size: 14,
  //         },
  //       },
  //     },
  //     y: {
  //       beginAtZero: true,
  //       title: {
  //         display: true,
  //         text: "state counts",
  //         font: {
  //           size: 14,
  //         },
  //       },
  //     },
  //   },
  //   plugins: {
  //     legend: {
  //       display: false,
  //     },
  //     tooltip: {
  //       callbacks: {},
  //     },
  //   },
  // },
};

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

function render_plots() {
  if (_DATA == null) {
    return;
  }

  let plot_data = _DATA["plot_data"][1];

  if (plot_data.length == 0) {
    return;
  }

  let xs = [];
  let ys = [];

  const original_data = _DATA["plot_data"][1][0];

  for (let i = 0; i < original_data.length; i++) {
    xs.push(original_data[i][0]);
    ys.push(original_data[i][1]);
  }

  const data = xs.map((k, i) => ({ x: k, y: ys[i] }));

  const background_color = Array(xs.length).fill("rgba(255, 99, 132, 0.6)");
  const border_color = Array(xs.length).fill("rgba(255, 99, 132, 1)");

  histogram_html = "<canvas id=histogram></canvas>";
  $("#histogram-holder").html(histogram_html);

  console.log("test");
  let chart_options = structuredClone(_CHART_OPTIONS_TEMPLATE);
  // chart_options["data"]["datasets"][0]["data"] = data;
  // chart_options["data"]["datasets"][0]["backgroundColor"] = background_color;
  // chart_options["data"]["datasets"][0]["borderColor"] = border_color;
  console.log(_CHART_OPTIONS_TEMPLATE);
  // chart_options["options"]["plugins"]["tooltip"]["callbacks"]["title"] = (
  //   items
  // ) => {
  //   if (!items.length) {
  //     return "";
  //   }
  //   const item = items[0];
  //   const x = item.parsed.x;
  //   const min = x - 0.5;
  //   const max = x + 0.5;
  //   return `State id: ${min} - ${max}`;
  // };
  new Chart($("#histogram"), chart_options);
}
