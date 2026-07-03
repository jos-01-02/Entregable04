const state = {
  report: null,
  chart2: null,
  chart4: null,
};

const DATASET_VARIANTS = [
  ["Monday-WorkingHours.pcap_ISCX.csv"],
  ["Tuesday-WorkingHours.pcap_ISCX.csv"],
  ["Wednesday-workingHours.pcap_ISCX.csv"],
  ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"],
  [
    "Thursday-WorkingHours-Afternoon-Infiltration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
  ],
  ["Friday-WorkingHours-Morning.pcap_ISCX.csv"],
  ["Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"],
  ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"],
];
const LOCAL_DATASET_DIR = "cicids2017_local_data";

const EXCLUDED_COLUMNS = new Set([
  "Timestamp",
  "Flow ID",
  "Source IP",
  "Destination IP",
  "Source Port",
  "Destination Port",
]);

const logEl = document.getElementById("log");
const summaryCardsEl = document.getElementById("summaryCards");
const reportFileEl = document.getElementById("reportFile");
const progressBarEl = document.getElementById("progressBar");
const progressLabelEl = document.getElementById("progressLabel");
const progressPctEl = document.getElementById("progressPct");
const progressMetaEl = document.getElementById("progressMeta");

document.getElementById("downloadBtn").addEventListener("click", runAnalysis);
document.getElementById("demoBtn").addEventListener("click", loadDemo);

function log(msg) {
  const timestamp = new Date().toLocaleTimeString();
  logEl.textContent += `[${timestamp}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function resetLog() {
  logEl.textContent = "";
}

function setProgress(percent, label, meta) {
  const safePercent = Math.max(0, Math.min(100, percent));
  progressBarEl.value = safePercent;
  progressPctEl.textContent = `${safePercent.toFixed(1)}%`;
  if (label) {
    progressLabelEl.textContent = label;
  }
  if (meta) {
    progressMetaEl.textContent = meta;
  }
}

function resetProgress() {
  setProgress(0, "En espera", 'Presiona "Procesar CSV locales" para iniciar.');
}

function addToMapCounter(map, key, amount = 1) {
  map.set(key, (map.get(key) || 0) + amount);
}

function toValidRecord(row) {
  const label = String(row.Label || "").trim();
  const destPort = Number(row["Destination Port"]);
  if (!label || !Number.isFinite(destPort)) {
    return null;
  }
  return { label, destPort: String(destPort) };
}

async function assertFileExists(url) {
  const response = await fetch(url, { method: "HEAD", cache: "no-store" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const size = Number(response.headers.get("content-length") || 0);
  return Number.isFinite(size) ? size : 0;
}

function parseCsvUrlInChunks(url, onRows, onChunkProgress) {
  return new Promise((resolve, reject) => {
    const absoluteUrl = new URL(url, window.location.href).href;
    let parsedRows = 0;

    Papa.parse(absoluteUrl, {
      download: true,
      header: true,
      transformHeader: (header) => String(header || "").trim(),
      skipEmptyLines: true,
      dynamicTyping: false,
      worker: false,
      chunkSize: 1024 * 1024,
      chunk: (results) => {
        const chunkRows = results?.data || [];
        if (chunkRows.length) {
          parsedRows += chunkRows.length;
          onRows(chunkRows);
        }
        if (onChunkProgress) {
          const cursor = Number(results?.meta?.cursor || 0);
          onChunkProgress(cursor);
        }
      },
      error: (error) => reject(error),
      complete: () => resolve(parsedRows),
    });
  });
}

function buildProgressUpdater(passIndex, passCount) {
  return ({ fileIndex, fileCount, fileName, fileProgress }) => {
    const passStart = (passIndex / passCount) * 100;
    const passSpan = 100 / passCount;
    const fileUnit = (fileIndex + fileProgress) / fileCount;
    const overall = passStart + fileUnit * passSpan;
    setProgress(
      overall,
      `Procesando ${passIndex + 1}/${passCount}`,
      `${fileName} · archivo ${fileIndex + 1}/${fileCount}`
    );
  };
}

async function processLocalFilesInPass(passName, onRows, onProgress) {
  const fileCount = DATASET_VARIANTS.length;
  for (let fileIndex = 0; fileIndex < DATASET_VARIANTS.length; fileIndex += 1) {
    const variants = DATASET_VARIANTS[fileIndex];
    const canonical = variants[0];
    let loaded = false;

    for (const variant of variants) {
      const url = `${LOCAL_DATASET_DIR}/${encodeURIComponent(variant)}?t=${Date.now()}`;
      try {
        log(`[${passName}] Leyendo local: ${variant}`);
        const contentLength = await assertFileExists(url);
        onProgress({ fileIndex, fileCount, fileName: variant, fileProgress: 0 });
        const parsed = await parseCsvUrlInChunks(url, onRows, (cursor) => {
          const fileProgress = contentLength > 0
            ? Math.max(0, Math.min(1, cursor / contentLength))
            : 0;
          onProgress({ fileIndex, fileCount, fileName: variant, fileProgress });
        });
        onProgress({ fileIndex, fileCount, fileName: variant, fileProgress: 1 });
        log(`[${passName}] OK ${variant}: ${parsed.toLocaleString()} filas`);
        loaded = true;
        break;
      } catch (error) {
        log(`[${passName}] No disponible ${variant}: ${error.message}`);
      }
    }

    if (!loaded) {
      throw new Error(
        `Falta ${canonical} en ${LOCAL_DATASET_DIR}. Ejecuta: node prepare_dataset.mjs`
      );
    }
  }
}

function renderSummary(agg) {
  const maxClassSupport = Math.max(0, ...agg.classCount.values());

  const cards = [
    { value: agg.validRows.toLocaleString(), label: "Filas validas" },
    { value: agg.classCount.size.toLocaleString(), label: "Clases detectadas" },
    { value: agg.portCount.size.toLocaleString(), label: "Puertos destino" },
    { value: maxClassSupport.toLocaleString(), label: "Clase con mayor soporte" },
  ];

  summaryCardsEl.innerHTML = cards
    .map((card) => `<article class="card"><b>${card.value}</b><span>${card.label}</span></article>`)
    .join("");
}

function getTopPorts(portCount, limit = 20) {
  return [...portCount.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([port]) => String(port));
}

function renderChart1(chart1Data) {
  const classes = chart1Data.classes;
  const topPorts = chart1Data.topPorts;
  const matrix = classes.map(() => topPorts.map(() => 0));

  const classIndex = new Map(classes.map((c, i) => [c, i]));
  const portIndex = new Map(topPorts.map((p, i) => [p, i]));

  for (const [key, count] of chart1Data.classPortCount.entries()) {
    const sep = key.indexOf("||");
    const label = key.slice(0, sep);
    const port = key.slice(sep + 2);
    const c = classIndex.get(label);
    const p = portIndex.get(port);
    if (c !== undefined && p !== undefined) {
      matrix[c][p] = count;
    }
  }

  const z = matrix.map((line) => line.map((v) => Math.log1p(v)));

  Plotly.newPlot(
    "chart1",
    [
      {
        z,
        x: topPorts,
        y: classes,
        type: "heatmap",
        colorscale: "YlOrRd",
      },
    ],
    {
      title: "Chart 1: Heatmap log(1 + frecuencia)",
      margin: { l: 130, r: 20, t: 45, b: 90 },
      xaxis: { title: "Destination Port", tickangle: -35 },
      yaxis: { title: "Label" },
    },
    { responsive: true }
  );
}

function destroyChart(chart) {
  if (chart) {
    chart.destroy();
  }
}

function renderChart2(classCount) {
  const counts = [...classCount.entries()].sort((a, b) => a[1] - b[1]);
  const labels = counts.map(([label]) => label);
  const data = counts.map(([, count]) => count);

  destroyChart(state.chart2);
  state.chart2 = new Chart(document.getElementById("chart2"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Instancias",
          data,
          borderWidth: 1,
          backgroundColor: "rgba(6,95,70,0.75)",
          borderColor: "rgba(6,95,70,1)",
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: {
        title: { display: true, text: "Chart 2: Distribucion de clases" },
      },
      scales: {
        x: {
          type: "logarithmic",
          title: { display: true, text: "Conteo (escala log)" },
        },
      },
    },
  });
}

function extractReportClasses(report) {
  const ignore = new Set(["accuracy", "macro avg", "weighted avg"]);
  return Object.keys(report).filter((k) => typeof report[k] === "object" && !ignore.has(k));
}

function buildEstimatedConfusionMatrix(report) {
  const classes = extractReportClasses(report);
  const n = classes.length;
  const cm = Array.from({ length: n }, () => Array(n).fill(0));

  for (let i = 0; i < n; i += 1) {
    const recall = Number(report[classes[i]].recall || 0);
    cm[i][i] = recall;
    const rest = n > 1 ? (1 - recall) / (n - 1) : 0;
    for (let j = 0; j < n; j += 1) {
      if (j !== i) {
        cm[i][j] = Math.max(rest, 0);
      }
    }
  }

  return { classes, cm };
}

function renderChart3FromReport(report) {
  const { classes, cm } = buildEstimatedConfusionMatrix(report);

  Plotly.newPlot(
    "chart3",
    [
      {
        z: cm,
        x: classes,
        y: classes,
        type: "heatmap",
        colorscale: "Blues",
        zmin: 0,
        zmax: 1,
      },
    ],
    {
      title: "Chart 3: Matriz de confusion normalizada (estimada)",
      margin: { l: 130, r: 20, t: 45, b: 120 },
      xaxis: { title: "Prediccion", tickangle: -35 },
      yaxis: { title: "Valor real" },
    },
    { responsive: true }
  );
}

function renderChart4FromReport(report) {
  const classes = extractReportClasses(report);
  const precision = classes.map((c) => Number(report[c].precision || 0));
  const recall = classes.map((c) => Number(report[c].recall || 0));
  const f1 = classes.map((c) => Number(report[c]["f1-score"] || 0));

  destroyChart(state.chart4);
  state.chart4 = new Chart(document.getElementById("chart4"), {
    type: "bar",
    data: {
      labels: classes,
      datasets: [
        {
          label: "Precision",
          data: precision,
          backgroundColor: "rgba(14,116,144,0.75)",
        },
        {
          label: "Recall",
          data: recall,
          backgroundColor: "rgba(180,83,9,0.75)",
        },
        {
          label: "F1-Score",
          data: f1,
          backgroundColor: "rgba(21,128,61,0.75)",
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: { display: true, text: "Chart 4: Precision, Recall y F1 por clase" },
      },
      scales: {
        x: { ticks: { maxRotation: 40, minRotation: 40 } },
        y: { min: 0, max: 1.05 },
      },
    },
  });
}

async function readReportJson(file) {
  if (!file) {
    return null;
  }
  const text = await file.text();
  return JSON.parse(text);
}

async function runAnalysis() {
  resetLog();
  setProgress(0, "Iniciando", "Preparando lectura de archivos...");
  const agg = {
    totalRows: 0,
    validRows: 0,
    classCount: new Map(),
    portCount: new Map(),
    classPortCountAll: new Map(),
  };

  try {
    log("Procesando CSV locales por streaming...");
    const pass1Progress = buildProgressUpdater(0, 1);
    await processLocalFilesInPass("PASS-1", (chunkRows) => {
      for (const row of chunkRows) {
        agg.totalRows += 1;
        const rec = toValidRecord(row);
        if (!rec) {
          continue;
        }
        agg.validRows += 1;
        addToMapCounter(agg.classPortCountAll, `${rec.label}||${rec.destPort}`);
        addToMapCounter(agg.classCount, rec.label);
        addToMapCounter(agg.portCount, rec.destPort);
      }
    }, pass1Progress);
  } catch (error) {
    setProgress(0, "Error", "No se pudo completar el procesamiento.");
    log(`Error de carga local: ${error.message}`);
    log("Primero ejecuta en terminal: node prepare_dataset.mjs");
    return;
  }

  const topPorts = getTopPorts(agg.portCount, 20);
  const topPortSet = new Set(topPorts);
  const classPortCount = new Map();

  for (const [key, count] of agg.classPortCountAll.entries()) {
    const sep = key.indexOf("||");
    const port = key.slice(sep + 2);
    if (topPortSet.has(port)) {
      classPortCount.set(key, count);
    }
  }

  log(`Total cargado (bruto): ${agg.totalRows.toLocaleString()} filas`);
  log(`Total valido: ${agg.validRows.toLocaleString()} filas`);
  log("Nota: modo streaming sin deduplicacion para evitar OOM en navegador.");

  renderChart1({
    classes: [...agg.classCount.keys()].sort(),
    topPorts,
    classPortCount,
  });
  renderSummary(agg);
  renderChart2(agg.classCount);
  log("Charts 1 y 2 generados.");

  try {
    state.report = await readReportJson(reportFileEl.files[0]);
  } catch (error) {
    log(`No se pudo leer JSON: ${error.message}`);
  }

  if (state.report) {
    renderChart3FromReport(state.report);
    renderChart4FromReport(state.report);
    log("Charts 3 y 4 generados desde classification_report.json.");
  } else {
    Plotly.purge("chart3");
    destroyChart(state.chart4);
    log("Sin JSON: se omitieron charts 3 y 4.");
  }

  setProgress(100, "Completado", "Proceso terminado correctamente.");
  log("Proceso finalizado.");
}

function loadDemo() {
  resetLog();
  setProgress(100, "Demo lista", "Se cargaron datos sinteticos de prueba.");
  const labels = ["BENIGN", "DoS Hulk", "PortScan", "DDoS", "Infiltration"];
  const demoRows = [];

  for (let i = 0; i < 1500; i += 1) {
    const label = labels[Math.floor(Math.random() * labels.length)];
    const ports = [80, 443, 53, 22, 8080, 3389, 445, 21, 123, 25];
    const port = ports[Math.floor(Math.random() * ports.length)];
    demoRows.push({ Label: label, "Destination Port": port });
  }

  const demoReport = {
    BENIGN: { precision: 0.98, recall: 0.99, "f1-score": 0.98, support: 800 },
    "DoS Hulk": { precision: 0.95, recall: 0.93, "f1-score": 0.94, support: 280 },
    PortScan: { precision: 0.89, recall: 0.91, "f1-score": 0.9, support: 220 },
    DDoS: { precision: 0.9, recall: 0.87, "f1-score": 0.88, support: 160 },
    Infiltration: { precision: 0.52, recall: 0.41, "f1-score": 0.46, support: 40 },
    accuracy: 0.93,
    "macro avg": { precision: 0.85, recall: 0.82, "f1-score": 0.83, support: 1500 },
    "weighted avg": { precision: 0.93, recall: 0.93, "f1-score": 0.93, support: 1500 },
  };

  state.report = demoReport;

  const classCount = new Map();
  const portCount = new Map();
  const classPortCount = new Map();
  for (const row of demoRows) {
    const rec = toValidRecord(row);
    if (!rec) {
      continue;
    }
    addToMapCounter(classCount, rec.label);
    addToMapCounter(portCount, rec.destPort);
    addToMapCounter(classPortCount, `${rec.label}||${rec.destPort}`);
  }

  renderSummary({ validRows: demoRows.length, classCount, portCount });
  renderChart1({ classes: [...classCount.keys()].sort(), topPorts: getTopPorts(portCount, 20), classPortCount });
  renderChart2(classCount);
  renderChart3FromReport(demoReport);
  renderChart4FromReport(demoReport);

  log("Demo cargada. Ya puedes validar la pagina con npx serve.");
}

resetProgress();
