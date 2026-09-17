"use strict";
const $ = (id) => document.getElementById(id);
let current = null;
const temperature = (value) => `${value.toFixed(1)}°`;
// Keep the provider's wall-clock labels: never reinterpret naive historical dates
// in the viewer's browser timezone, or silently shift UTC labels.
function formatDate(value, short = false) {
  const [date, time] = value.split("T");
  return `${short ? date.slice(5) : date} ${time.slice(0, 5)}`;
}
function node(name, attrs = {}, text = "") {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) =>
    element.setAttribute(key, value),
  );
  if (text) element.textContent = text;
  return element;
}
function drawChart(data) {
  const svg = $("chart");
  svg.replaceChildren();
  const width = Math.max(440, $("chart-wrap").clientWidth);
  const compact = width < 600;
  const fontSize = compact ? 15 : 11;
  const right = width - 26;
  svg.setAttribute("viewBox", `0 0 ${width} 350`);
  svg.append(
    node(
      "title",
      { id: "chart-title" },
      "Hourly temperature history and forecast",
    ),
  );
  svg.append(
    node(
      "desc",
      { id: "chart-desc" },
      `48 hours of history followed by ${data.horizon} forecast hours. ${Math.round(data.interval_level * 100)} percent uncertainty band. Exact predictions are in the table below.`,
    ),
  );
  const history = data.history,
    predicted = data.forecast;
  const all = [...history, ...data.lower_ci, ...data.upper_ci];
  const low = Math.floor(Math.min(...all) - 2),
    high = Math.ceil(Math.max(...all) + 2);
  const n = history.length + predicted.length;
  const x = (i) => 52 + (i * (right - 52)) / (n - 1),
    y = (v) => 285 - ((v - low) * 235) / (high - low);
  for (let i = 0; i <= 4; i++) {
    const value = low + ((high - low) * i) / 4;
    svg.append(
      node("line", {
        x1: 52,
        x2: right,
        y1: y(value),
        y2: y(value),
        stroke: "#e7ebe3",
      }),
    );
    svg.append(
      node(
        "text",
        {
          x: 38,
          y: y(value) + 4,
          "text-anchor": "end",
          fill: "#73837e",
          "font-size": fontSize,
        },
        `${value.toFixed(0)}°`,
      ),
    );
  }
  const cut = x(history.length - 1);
  svg.append(
    node("rect", {
      x: cut,
      y: 40,
      width: right - cut,
      height: 245,
      fill: "#f5f8ee",
      opacity: 0.7,
    }),
  );
  svg.append(
    node("line", {
      x1: cut,
      x2: cut,
      y1: 40,
      y2: 285,
      stroke: "#a5b5a0",
      "stroke-dasharray": "4 5",
    }),
  );
  svg.append(
    node(
      "text",
      {
        x: Math.min(cut + 8, right - 90),
        y: 28,
        fill: "#668061",
        "font-size": compact ? 13 : 10,
      },
      "FORECAST →",
    ),
  );
  const path = (values, start) =>
    values
      .map(
        (v, i) =>
          `${i ? "L" : "M"}${x(start + i).toFixed(2)},${y(v).toFixed(2)}`,
      )
      .join(" ");
  const band = data.upper_ci
    .map((v, i) => `${x(history.length + i)},${y(v)}`)
    .concat(
      data.lower_ci.map((v, i) => `${x(history.length + i)},${y(v)}`).reverse(),
    )
    .join(" ");
  svg.append(node("polygon", { points: band, fill: "#d5e6b4", opacity: 0.8 }));
  svg.append(
    node("path", {
      d: path(history, 0),
      fill: "none",
      stroke: "#6c8783",
      "stroke-width": 2.5,
      "stroke-linejoin": "round",
    }),
  );
  svg.append(
    node("path", {
      d: path([history.at(-1), ...predicted], history.length - 1),
      fill: "none",
      stroke: "#176958",
      "stroke-width": 3,
      "stroke-linejoin": "round",
    }),
  );
  const times = [...data.history_timestamps, ...data.forecast_timestamps];
  const ticks = compact ? 2 : 4;
  for (let i = 0; i <= ticks; i++) {
    const at = Math.round((i * (n - 1)) / ticks);
    svg.append(
      node(
        "text",
        {
          x: x(at),
          y: 315,
          "text-anchor": i === 0 ? "start" : i === ticks ? "end" : "middle",
          fill: "#73837e",
          "font-size": fontSize,
        },
        formatDate(times[at], true),
      ),
    );
  }
  // Native SVG tooltips expose exact timestamps/values without third-party scripts.
  predicted.forEach((v, i) => {
    const dot = node("circle", {
      cx: x(history.length + i),
      cy: y(v),
      r: 5,
      fill: "#176958",
      "fill-opacity": 0.01,
    });
    dot.append(
      node(
        "title",
        {},
        `${formatDate(data.forecast_timestamps[i])}: ${temperature(v)}C (${temperature(data.lower_ci[i])}–${temperature(data.upper_ci[i])}C)`,
      ),
    );
    svg.append(dot);
  });
}
function render(data) {
  current = data;
  $("output").hidden = false;
  $("empty").hidden = true;
  $("latest").textContent = `${temperature(data.history.at(-1))}C`;
  $("latest-date").textContent = formatDate(data.data_cutoff);
  $("average").textContent =
    `${temperature(data.forecast.reduce((a, b) => a + b, 0) / data.horizon)}C`;
  $("forecast-span").textContent =
    `Next ${data.horizon} ${data.horizon === 1 ? "hour" : "hours"}`;
  $("range").textContent =
    `${temperature(Math.min(...data.forecast))} – ${temperature(Math.max(...data.forecast))}`;
  $("mode-badge").textContent =
    data.mode === "live" ? "RECENT WEATHER" : "HISTORICAL DEMO";
  $("mode-badge").classList.toggle("live", data.mode === "live");
  $("interval-legend").textContent =
    `${Math.round(data.interval_level * 100)}% interval`;
  $("chart-note").textContent =
    `Times: ${data.timezone}. ${data.interval_method}.`;
  $("updated").textContent =
    `Data through ${formatDate(data.data_cutoff)} · ${data.data_source}`;
  $("model-version").textContent =
    `${data.model.toUpperCase()} / ${data.model_version}`;
  $("status").textContent =
    `${data.model.toUpperCase()} · ${data.horizon}-hour forecast · ${data.mode === "live" ? "Recent weather context" : "Historical demonstration, not today's weather"}`;
  $("table-caption").textContent =
    `Hourly temperature (°C) · ${data.timezone} · ${Math.round(data.interval_level * 100)}% interval`;
  $("forecast-table").replaceChildren();
  data.forecast.forEach((value, i) => {
    const row = document.createElement("tr");
    [
      formatDate(data.forecast_timestamps[i]),
      value.toFixed(2),
      data.lower_ci[i].toFixed(2),
      data.upper_ci[i].toFixed(2),
    ].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    $("forecast-table").append(row);
  });
  drawChart(data);
}
async function generate(event) {
  event?.preventDefault();
  const submit = $("submit");
  submit.disabled = true;
  $("forecast-form").setAttribute("aria-busy", "true");
  $("error").hidden = true;
  $("output").hidden = true;
  $("empty").hidden = false;
  current = null;
  $("status").textContent =
    "Generating forecast… The first model load may take a moment.";
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const params = new URLSearchParams({
      model: $("model").value,
      horizon: $("horizon").value,
      use_live_data: String($("mode").value === "live"),
    });
    const response = await fetch(`/forecast?${params}`, {
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok)
      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : "The request could not be completed. Check your forecast settings.",
      );
    render(data);
  } catch (error) {
    $("error").textContent =
      error.name === "AbortError"
        ? "The forecast took too long. Please try again."
        : error.message ||
          "Could not connect. Check that the server is running.";
    $("error").hidden = false;
    $("status").textContent =
      "Forecast unavailable. Adjust the settings or retry.";
    $("empty").hidden = true;
  } finally {
    clearTimeout(timeout);
    submit.disabled = false;
    $("forecast-form").removeAttribute("aria-busy");
  }
}
$("forecast-form").addEventListener("submit", generate);
$("horizon").addEventListener("input", () => {
  $("horizon-value").textContent = `${$("horizon").value} hours`;
});
$("mode").addEventListener("change", () => {
  $("mode-help").textContent =
    $("mode").value === "live"
      ? "Use recent model-derived weather from Open-Meteo, in UTC."
      : "Explore a forecast after the dataset ends in 2017.";
});
$("model").addEventListener("change", () => {
  $("model-help").textContent =
    $("model").value === "tft"
      ? "Learns from a week of temperature history and calendar cycles."
      : "Captures temperature trends and the daily cycle.";
});
$("download").addEventListener("click", () => {
  if (!current) return;
  const rows = [
    [
      "timestamp",
      "forecast_c",
      "lower_c",
      "upper_c",
      "interval_level",
      "model_version",
      "mode",
      "timezone",
    ],
    ...current.forecast.map((v, i) => [
      current.forecast_timestamps[i],
      v,
      current.lower_ci[i],
      current.upper_ci[i],
      current.interval_level,
      current.model_version,
      current.mode,
      current.timezone,
    ]),
  ];
  const csv = rows
    .map((row) =>
      row.map((v) => `"${String(v).replaceAll('"', '""')}"`).join(","),
    )
    .join("\r\n");
  const url = URL.createObjectURL(
    new Blob([csv], { type: "text/csv;charset=utf-8" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `jena-${current.model}-${current.mode}.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
generate();
new ResizeObserver(() => {
  if (current) drawChart(current);
}).observe($("chart-wrap"));
