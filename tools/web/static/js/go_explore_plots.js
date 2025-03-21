// Chart.js configuration template
const CHART_OPTIONS_TEMPLATE = {
  scales: {
    x: {
      title: {
        display: true,
        color: "#FFFFFF",
        font: { size: 12 },
      },
      ticks: {
        color: "#FFFFFF",
        font: { size: 12 },
      },
    },
    y: {
      ticks: {
        color: "#FFFFFF",
        font: { size: 12 },
      },
    },
  },
  plugins: {
    title: {
      color: "#FFFFFF",
      display: true,
      font: { size: 16 },
    },
    legend: {
      align: "end",
      labels: {
        boxWidth: 8,
        color: "#FFFFFF",
      },
    },
  },
};

// Color palette for different metrics
const COLORS = [
  "rgba(255, 0, 0, 1)",
  "rgba(0, 255, 0, 1)",
  "rgba(0, 0, 255, 1)",
  "rgba(255, 255, 0, 1)",
  "rgba(0, 255, 255, 1)",
  "rgba(255, 0, 255, 1)",
];

let data = null;

// Function to update plots with new data
function updatePlots() {
  const n = document.getElementById("n-states").value || 10;

  // Show loading indicator
  document.getElementById("loading").classList.remove("hidden");

  // Fetch data from endpoint
  fetch(
    `${_ROOT_URL}n_fewest_steps_since_led_to_something_new_go_explore?n=${n}`
  )
    .then((response) => response.json())
    .then((responseData) => {
      data = responseData;
      renderPlots();
    })
    .catch((error) => {
      console.error("Error fetching data:", error);
      document.getElementById("load-error").classList.remove("hidden");
    })
    .finally(() => {
      document.getElementById("loading").classList.add("hidden");
    });
}

// Function to render all plots
function renderPlots() {
  if (!data) return;

  // Create container for plots if it doesn't exist
  let container = document.getElementById("plots-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "plots-container";
    document.body.appendChild(container);
  }

  // Clear existing plots
  container.innerHTML = "";

  // Create plots for each metric
  const metrics = [
    { key: "trajectory_length", label: "Trajectory Length" },
    { key: "steps_since_led_to_something_new", label: "Steps Since New Cell" },
    {
      key: "steps_since_led_to_something_new_reset_count",
      label: "Reset Count",
    },
    { key: "sample_count", label: "Sample Count" },
    { key: "visit_count", label: "Visit Count" },
  ];

  metrics.forEach((metric, index) => {
    const plotDiv = document.createElement("div");
    plotDiv.className = "plot-container";
    container.appendChild(plotDiv);

    const canvas = document.createElement("canvas");
    canvas.id = `plot-${metric.key}`;
    plotDiv.appendChild(canvas);

    const chartOptions = structuredClone(CHART_OPTIONS_TEMPLATE);
    chartOptions.plugins.title.text = metric.label;

    new Chart(canvas, {
      type: "bar",
      data: {
        labels: Array.from(
          { length: data[metric.key].length },
          (_, i) => `State ${i + 1}`
        ),
        datasets: [
          {
            label: metric.label,
            data: data[metric.key],
            backgroundColor: COLORS[index % COLORS.length],
            borderColor: COLORS[index % COLORS.length],
          },
        ],
      },
      options: chartOptions,
    });
  });

  // Create state visualization grid
  const stateGrid = document.createElement("div");
  stateGrid.id = "state-grid";
  container.appendChild(stateGrid);

  // Add state visualizations
  data.states.forEach((state, index) => {
    const stateDiv = document.createElement("div");
    stateDiv.className = "state-container";
    stateGrid.appendChild(stateDiv);

    const stateTitle = document.createElement("h3");
    stateTitle.textContent = `State ${index + 1}`;
    stateDiv.appendChild(stateTitle);

    const stateCanvas = document.createElement("canvas");
    stateCanvas.id = `state-${index}`;
    stateDiv.appendChild(stateCanvas);

    // Render 2D state array
    renderStateGrid(state, stateCanvas);
  });
}

// Function to render a 2D state grid
function renderStateGrid(state, canvas) {
  const ctx = canvas.getContext("2d");
  const cellSize = 20;
  const padding = 2;

  // Set canvas size based on state dimensions
  canvas.width = state[0].length * (cellSize + padding);
  canvas.height = state.length * (cellSize + padding);

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw state grid
  state.forEach((row, y) => {
    row.forEach((value, x) => {
      ctx.fillStyle = value ? "#FFFFFF" : "#000000";
      ctx.fillRect(
        x * (cellSize + padding),
        y * (cellSize + padding),
        cellSize,
        cellSize
      );
    });
  });
}

// Initialize when document is ready
document.addEventListener("DOMContentLoaded", () => {
  // Add controls if they don't exist
  const controls = document.createElement("div");
  controls.className = "controls";
  controls.innerHTML = `
        <div class="control">
            <label for="n-states">Number of States:</label>
            <input type="number" id="n-states" value="10" min="1" max="50">
        </div>
        <button onclick="updatePlots()">Update Plots</button>
    `;
  document.body.insertBefore(controls, document.body.firstChild);

  // Initial plot update
  updatePlots();
});
