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
let currentMetric = "trajectory_length"; // Default metric

// Available metrics with their display names
const METRICS = [
  { key: "trajectory_length", label: "Trajectory Length" },
  { key: "steps_since_led_to_something_new", label: "Steps Since New Cell" },
  { key: "steps_since_led_to_something_new_reset_count", label: "Reset Count" },
  { key: "sampled_count", label: "Sample Count" },
  { key: "visit_count", label: "Visit Count" },
];

// Function to update plots with new data
function updatePlots() {
  const nInput = document.getElementById("n-states");
  const metricSelector = document.getElementById("metric-selector");

  if (!nInput || !metricSelector) {
    console.error("Required DOM elements not found");
    return;
  }

  const n = nInput.value || 10;
  currentMetric = metricSelector.value;

  // Show loading indicator
  document.getElementById("loading").classList.remove("hidden");
  document.getElementById("load-error").classList.add("hidden");

  // Fetch data from endpoint
  fetch(`${_ROOT_URL}go_explore?n=${n}`)
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

  // Get the current metric data
  const metricData = data[currentMetric];
  if (!metricData || !metricData.states || !metricData.values) return;

  // Find the metric label
  const metricLabel =
    METRICS.find((m) => m.key === currentMetric)?.label || currentMetric;

  // Create a title for the current metric
  const metricTitle = document.createElement("h2");
  metricTitle.style.color = "#FFFFFF";
  metricTitle.style.textAlign = "center";
  metricTitle.style.marginBottom = "20px";
  metricTitle.textContent = metricLabel;
  container.appendChild(metricTitle);

  // Create state grid container
  const gridContainer = document.createElement("div");
  gridContainer.className = "state-grid";
  container.appendChild(gridContainer);

  // Create state visualizations
  metricData.states.forEach((state, stateIndex) => {
    const stateContainer = document.createElement("div");
    stateContainer.className = "state-container";

    // Create title with metric value
    const title = document.createElement("h3");
    title.textContent = `Value: ${metricData.values[stateIndex]}`;
    stateContainer.appendChild(title);

    // Create canvas for state visualization
    const canvas = document.createElement("canvas");
    stateContainer.appendChild(canvas);

    // Render state grid
    renderStateGrid(state, canvas);

    gridContainer.appendChild(stateContainer);
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
      // Use different colors for different values
      let color;
      if (value === 0) {
        color = "#000000"; // Black for empty
      } else if (value === 1) {
        color = "#FFFFFF"; // White for walls/obstacles
      } else if (value === 2) {
        color = "#FF0000"; // Red for player/agent
      } else if (value === 3) {
        color = "#00FF00"; // Green for goals
      } else {
        color = "#888888"; // Gray for other values
      }

      ctx.fillStyle = color;
      ctx.fillRect(
        x * (cellSize + padding),
        y * (cellSize + padding),
        cellSize,
        cellSize
      );

      // Add a subtle border around each cell
      ctx.strokeStyle = "#333";
      ctx.strokeRect(
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
  // Create controls if they don't exist
  let controls = document.querySelector(".controls");
  if (!controls) {
    controls = document.createElement("div");
    controls.className = "controls";
    document.body.insertBefore(controls, document.body.firstChild);
  }

  // Create metric selector
  const metricSelector = document.createElement("select");
  metricSelector.id = "metric-selector";
  metricSelector.className = "control";

  // Add metric options
  METRICS.forEach((metric) => {
    const option = document.createElement("option");
    option.value = metric.key;
    option.textContent = metric.label;
    metricSelector.appendChild(option);
  });

  // Add metric selector to controls
  const metricControl = document.createElement("div");
  metricControl.className = "control";
  metricControl.innerHTML = `
    <label for="metric-selector">Metric:</label>
    ${metricSelector.outerHTML}
  `;
  controls.insertBefore(metricControl, controls.firstChild);

  // Add number of states input if it doesn't exist
  if (!document.getElementById("n-states")) {
    const nStatesControl = document.createElement("div");
    nStatesControl.className = "control";
    nStatesControl.innerHTML = `
      <label for="n-states">Number of States:</label>
      <input type="number" id="n-states" value="10" min="1" max="50">
    `;
    controls.appendChild(nStatesControl);
  }

  // Add update button if it doesn't exist
  if (!document.querySelector(".controls button")) {
    const updateButton = document.createElement("button");
    updateButton.textContent = "Update Plots";
    updateButton.onclick = updatePlots;
    controls.appendChild(updateButton);
  }

  // Wait a short moment for DOM to be fully updated before initial plot update
  setTimeout(updatePlots, 100);
});
