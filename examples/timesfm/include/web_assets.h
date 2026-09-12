#pragma once

#include <string>

namespace timesfm {

inline std::string get_index_html() {
    std::string html;

    // Chunk 1: Head & CSS
    html += R"rawliteral(<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Google TimesFM 3.0 — Foundation Forecasting Studio</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1e293b;
      --card-hover: #1e293b;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --primary-soft: rgba(56, 189, 248, 0.12);
      --accent: #f97316;
      --accent-hover: #ea580c;
      --accent-soft: rgba(249, 115, 22, 0.12);
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --success: #10b981;
      --warning: #f59e0b;
      --purple: #a855f7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      min-height: 100vh;
      padding: 24px 24px 48px;
    }
    .container { max-width: 1480px; margin: 0 auto; }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
      gap: 16px;
      flex-wrap: wrap;
    }
    .logo-group h1 {
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #38bdf8 0%, #f97316 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .logo-group p { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
    .header-actions { display: flex; align-items: center; gap: 12px; }
    .nav-tabs {
      display: flex;
      gap: 6px;
      background: #0b1120;
      padding: 4px;
      border-radius: 10px;
      border: 1px solid var(--card-border);
    }
    .nav-tab {
      padding: 8px 18px;
      border-radius: 8px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .nav-tab.active {
      background: var(--primary);
      color: #090d16;
      box-shadow: 0 2px 8px rgba(56, 189, 248, 0.3);
    }
    .btn-help {
      display: flex;
      align-items: center;
      gap: 6px;
      background: #1e293b;
      border: 1px solid #334155;
      color: #cbd5e1;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-help:hover { background: #334155; color: #fff; border-color: var(--primary); }
    .status-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 12px;
      color: var(--success);
      font-weight: 600;
    }
    .status-dot { width: 8px; height: 8px; background: var(--success); border-radius: 50%; box-shadow: 0 0 10px var(--success); }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .grid { display: grid; grid-template-columns: 400px 1fr; gap: 24px; }
    @media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 22px;
      box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.4);
      position: relative;
    }
    .card-title {
      font-size: 15px;
      font-weight: 700;
      margin-bottom: 16px;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .tooltip-icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 16px;
      height: 16px;
      background: #1e293b;
      border: 1px solid #334155;
      color: #94a3b8;
      border-radius: 50%;
      font-size: 11px;
      cursor: help;
      position: relative;
      margin-left: 6px;
      font-weight: bold;
    }
    .tooltip-icon:hover::after {
      content: attr(data-tip);
      position: absolute;
      bottom: 130%;
      left: 50%;
      transform: translateX(-50%);
      background: #0f172a;
      color: #f8fafc;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 12px;
      white-space: normal;
      width: 240px;
      line-height: 1.4;
      box-shadow: 0 6px 18px rgba(0,0,0,0.6);
      border: 1px solid #334155;
      z-index: 100;
      pointer-events: none;
      font-weight: normal;
    }
    .form-group { margin-bottom: 16px; }
    .form-group label {
      display: flex;
      align-items: center;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .preset-badge {
      font-size: 11px;
      color: var(--primary);
      background: var(--primary-soft);
      border: 1px solid rgba(56, 189, 248, 0.2);
      padding: 4px 8px;
      border-radius: 6px;
      margin-top: 6px;
      line-height: 1.4;
    }
    select, input[type="number"], input[type="text"], textarea {
      width: 100%;
      background: #0b1120;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 12px;
      color: var(--text);
      font-size: 13px;
      outline: none;
      transition: border-color 0.2s;
      font-family: inherit;
    }
    textarea { resize: vertical; min-height: 54px; }
    select:focus, input:focus, textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }
    .dropzone {
      border: 2px dashed var(--card-border);
      border-radius: 8px;
      padding: 12px 10px;
      text-align: center;
      cursor: pointer;
      background: rgba(11, 17, 32, 0.4);
      transition: all 0.2s;
    }
    .dropzone:hover { border-color: var(--primary); background: var(--primary-soft); }
    .dropzone p { font-size: 12px; color: var(--text-muted); }
    input[type="file"] { display: none; }
    .slider-container { display: flex; flex-direction: column; gap: 4px; }
    .slider-header { display: flex; justify-content: space-between; font-size: 13px; }
    .slider-val { color: var(--primary); font-weight: 700; }
    input[type="range"] {
      width: 100%;
      height: 6px;
      background: #1e293b;
      border-radius: 4px;
      outline: none;
      accent-color: var(--primary);
      cursor: pointer;
    }
    .checkbox-group { display: flex; flex-direction: column; gap: 9px; margin-top: 6px; }
    .checkbox-label {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      font-size: 13px;
      color: var(--text);
      cursor: pointer;
      line-height: 1.35;
    }
    .checkbox-label input[type="checkbox"] {
      accent-color: var(--accent);
      width: 16px;
      height: 16px;
      margin-top: 1px;
    }
    .btn-primary {
      width: 100%;
      background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
      color: #090d16;
      font-weight: 700;
      font-size: 14px;
      border: none;
      border-radius: 8px;
      padding: 12px;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25);
      transition: all 0.2s;
    }
    .btn-accent {
      width: 100%;
      background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
      color: #ffffff;
      font-weight: 700;
      font-size: 14px;
      border: none;
      border-radius: 8px;
      padding: 12px;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(249, 115, 22, 0.25);
      transition: all 0.2s;
    }
    .btn-primary:hover, .btn-accent:hover { transform: translateY(-1px); opacity: 0.95; }
    .btn-primary:disabled, .btn-accent:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .export-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--card-border);
    }
    .btn-export {
      background: #1e293b;
      border: 1px solid #334155;
      color: #e2e8f0;
      padding: 7px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .btn-export:hover { background: #334155; border-color: var(--primary); color: #fff; }
    .view-switcher {
      display: flex;
      gap: 4px;
      background: #0b1120;
      padding: 3px;
      border-radius: 8px;
      border: 1px solid var(--card-border);
    }
    .view-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .view-btn.active { background: #1e293b; color: var(--primary); }
    .stats-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .stat-card {
      background: #0b1120;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 10px 12px;
    }
    .stat-title {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
    }
    .stat-val { font-size: 17px; font-weight: 700; color: #f8fafc; margin-top: 3px; }
    .kpi-banner {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 14px;
    }
    .kpi-item { display: flex; flex-direction: column; }
    .kpi-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.4px; }
    .kpi-value { font-size: 16px; font-weight: 700; color: var(--text); margin-top: 2px; }
    .chart-box { position: relative; height: 410px; width: 100%; }
    .table-container { margin-top: 16px; overflow-x: auto; max-height: 260px; border-radius: 8px; border: 1px solid var(--card-border); }
    table { width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }
    th { background: #0b1120; color: var(--text-muted); padding: 9px 10px; border-bottom: 1px solid var(--card-border); position: sticky; top: 0; }
    td { padding: 8px 10px; border-bottom: 1px solid #1e293b; color: #e2e8f0; font-family: ui-monospace, monospace; }
    .modal-backdrop {
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      z-index: 1000;
      justify-content: center;
      align-items: center;
      padding: 20px;
    }
    .modal-backdrop.active { display: flex; }
    .modal-card {
      background: #111827;
      border: 1px solid #334155;
      border-radius: 16px;
      max-width: 840px;
      width: 100%;
      max-height: 85vh;
      overflow-y: auto;
      padding: 28px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.8);
      position: relative;
    }
    .modal-close {
      position: absolute;
      top: 20px;
      right: 20px;
      background: #1e293b;
      border: 1px solid #334155;
      color: #94a3b8;
      width: 32px;
      height: 32px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: bold;
    }
    .modal-close:hover { color: #fff; background: #334155; }
    .modal-title { font-size: 20px; font-weight: 700; color: #f8fafc; margin-bottom: 16px; }
    .help-section { margin-bottom: 20px; }
    .help-section h3 { font-size: 15px; color: var(--primary); margin-bottom: 8px; }
    .help-section p { font-size: 13px; color: #cbd5e1; line-height: 1.6; margin-bottom: 8px; }
    .help-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
    @media (max-width: 600px) { .help-grid { grid-template-columns: 1fr; } }
    .help-item { background: #0b1120; border: 1px solid #1e293b; padding: 10px 12px; border-radius: 8px; }
    .help-item strong { color: var(--accent); font-size: 13px; display: block; margin-bottom: 4px; }
    .help-item span { color: #94a3b8; font-size: 12px; line-height: 1.4; display: block; }
    .toast-container { position: fixed; bottom: 24px; right: 24px; z-index: 2000; display: flex; flex-direction: column; gap: 8px; }
    .toast {
      background: #1e293b;
      color: #f8fafc;
      border: 1px solid #334155;
      padding: 10px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 500;
      box-shadow: 0 8px 20px rgba(0,0,0,0.5);
      animation: slideIn 0.25s ease-out;
    }
    .toast.success { border-left: 4px solid var(--success); }
    .toast.info { border-left: 4px solid var(--primary); }
    @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
  </style>
</head>
)rawliteral";

    // Chunk 2: Body Header & Forecast Controls
    html += R"rawliteral(
<body>
<div class="container">
  <div class="header">
    <div class="logo-group">
      <h1>Google TimesFM 3.0 Web Studio</h1>
      <p>Zero-Shot Foundation Forecasting &amp; Probabilistic Backtesting powered by ggmlc</p>
    </div>
    <div class="header-actions">
      <div class="nav-tabs">
        <button class="nav-tab active" onclick="switchTab('forecast')">Forecast Studio</button>
        <button class="nav-tab" onclick="switchTab('backtest')">Backtesting Suite</button>
      </div>
      <button class="btn-help" onclick="toggleHelpModal(true)">
        <span>Help &amp; Guide</span>
        <span style="background: rgba(56,189,248,0.2); color: var(--primary); border-radius: 50%; width: 18px; height: 18px; display: inline-flex; align-items: center; justify-content: center; font-size: 11px;">?</span>
      </button>
      <div class="status-badge">
        <div class="status-dot"></div>
        <span id="backendStatus">Engine Ready</span>
      </div>
    </div>
  </div>

  <!-- Tab 1: Forecast Studio -->
  <div id="tab-forecast" class="tab-content active">
    <div class="grid">
      <div class="card">
        <div class="card-title">Forecast Controls</div>
        
        <div class="form-group">
          <label>Built-in Preset Scenario <span class="tooltip-icon" data-tip="Curated realistic real-world and synthetic datasets.">i</span></label>
          <select id="presetSelect" onchange="onPresetChange()">
            <option value="weekly_retail">Weekly Retail Demand (Day Multipliers)</option>
            <option value="airline_passengers">Airline Passengers Benchmark (1949-1960)</option>
            <option value="sunspots">Monthly Sunspots Solar Activity (180 steps)</option>
            <option value="trend_seasonal">Synthetic Trend + Dual Seasonality</option>
            <option value="spiky_demand">Spiky Inventory Demand (Poisson Shocks)</option>
            <option value="linear_trend">Linear Drift + Gaussian Noise</option>
            <option value="seasonal_sine">Smooth Seasonal Sine Wave</option>
            <option value="random_walk">Financial Random Walk Process</option>
          </select>
          <div id="presetBadge" class="preset-badge">Weekly retail demand with weekly day-of-week multipliers and upward growth.</div>
        </div>

        <div class="form-group">
          <label>Or Paste Freeform Numbers <span class="tooltip-icon" data-tip="Paste comma or whitespace separated numbers.">i</span></label>
          <textarea id="textInput" placeholder="e.g. 12.4, 15.2, 18.0, 21.5, 25.1, 28.3, 31.0, 34.5..." oninput="onTextChange()"></textarea>
        </div>

        <div class="form-group">
          <label>Or Upload CSV File <span class="tooltip-icon" data-tip="Upload custom CSV time series.">i</span></label>
          <div class="dropzone" onclick="document.getElementById('csvInput').click()">
            <p id="fileLabel">Drop CSV or Click to Browse</p>
          </div>
          <input type="file" id="csvInput" accept=".csv" onchange="onCsvUpload(event)">
        </div>

        <div class="form-group">
          <div class="slider-container">
            <div class="slider-header">
              <label>Forecast Horizon <span class="tooltip-icon" data-tip="Number of future time steps to generate autoregressively in 64-step patches.">i</span></label>
              <span id="horizonVal" class="slider-val">128 steps</span>
            </div>
            <input type="range" id="horizonRange" min="16" max="256" step="16" value="128" oninput="document.getElementById('horizonVal').innerText = this.value + ' steps'">
          </div>
        </div>

        <div class="form-group">
          <label>Inference Pipelines &amp; Clamping <span class="tooltip-icon" data-tip="TimesFM mathematical preprocessing and output conditioning pipelines.">i</span></label>
          <div class="checkbox-group">
            <label class="checkbox-label">
              <input type="checkbox" id="revinCheck" checked>
              <div><strong>RevIN Normalization &amp; CPM</strong><div style="font-size: 11px; color: var(--text-muted);">Mean-shift and variance scaling to handle non-stationary distribution shifts.</div></div>
            </label>
            <label class="checkbox-label">
              <input type="checkbox" id="detrendCheck" checked>
              <div><strong>Linear Detrending (R² &gt; 0.5)</strong><div style="font-size: 11px; color: var(--text-muted);">Extracts persistent slope to prevent zero-shot mean-reversion collapse.</div></div>
            </label>
            <label class="checkbox-label">
              <input type="checkbox" id="sortQuantilesCheck" checked>
              <div><strong>Quantile Monotonicity (q10 ≤ ... ≤ q90)</strong><div style="font-size: 11px; color: var(--text-muted);">Re-sorts quantile tracks per step to prevent crossing probability bounds.</div></div>
            </label>
            <label class="checkbox-label">
              <input type="checkbox" id="positiveCheck">
              <div><strong>Non-Negativity Constraint (Clamp ≥ 0)</strong><div style="font-size: 11px; color: var(--text-muted);">Clamps predictions at zero for physical counts, sales, or web traffic.</div></div>
            </label>
            <label class="checkbox-label">
              <input type="checkbox" id="symAvgCheck">
              <div><strong>Symmetric Flip-Invariance Averaging</strong><div style="font-size: 11px; color: var(--text-muted);">Averages forecasts of x and -x to guarantee sign symmetry: (f(x) - f(-x))/2.</div></div>
            </label>
          </div>
        </div>

        <button id="btnForecast" class="btn-primary" onclick="runForecast()">Generate Zero-Shot Forecast</button>
      </div>

      <div class="card">
        <div class="card-title">
          <span>Interactive Probabilistic Forecast</span>
          <div class="view-switcher">
            <button id="viewBtnFan" class="view-btn active" onclick="switchViewMode('fan')">Fan Bands</button>
            <button id="viewBtnAll" class="view-btn" onclick="switchViewMode('all')">All Quantiles</button>
            <button id="viewBtnTrend" class="view-btn" onclick="switchViewMode('trend')">Trend Decomposition</button>
          </div>
        </div>

        <div class="kpi-banner">
          <div class="kpi-item"><span class="kpi-label">Expected Final (q50)</span><span id="kpiFinalVal" class="kpi-value">--</span></div>
          <div class="kpi-item"><span class="kpi-label">Expected Growth</span><span id="kpiGrowthVal" class="kpi-value">-- %</span></div>
          <div class="kpi-item"><span class="kpi-label">80% Uncertainty Band</span><span id="kpiSpreadVal" class="kpi-value">± --</span></div>
          <div class="kpi-item"><span class="kpi-label">Expected Range [q10, q90]</span><span id="kpiRangeVal" class="kpi-value">[--, --]</span></div>
        </div>

        <div class="stats-row">
          <div class="stat-card"><div class="stat-title">Inference Latency</div><div class="stat-val" id="statLatency" style="color: var(--primary);">-- ms</div></div>
          <div class="stat-card"><div class="stat-title">Horizon Length</div><div class="stat-val" id="statHorizon">-- steps</div></div>
          <div class="stat-card"><div class="stat-title">RevIN Mean / Std</div><div class="stat-val" id="statRevIN" style="font-size: 15px;">-- / --</div></div>
          <div class="stat-card"><div class="stat-title">Detrending R²</div><div class="stat-val" id="statTrend">--</div></div>
        </div>

        <div class="chart-box"><canvas id="forecastCanvas"></canvas></div>

        <div class="export-bar">
          <span style="font-size: 12px; color: var(--text-muted); font-weight: 600; margin-right: 4px;">Export Actions:</span>
          <button class="btn-export" onclick="downloadCsv()"><span>📥 Download CSV</span></button>
          <button class="btn-export" onclick="downloadJson()"><span>📦 Download JSON</span></button>
          <button class="btn-export" onclick="copyTableToClipboard()"><span>📋 Copy Table (TSV)</span></button>
          <button class="btn-export" onclick="copyJsonToClipboard()"><span>📋 Copy JSON</span></button>
        </div>
      </div>
    </div>
  </div>
)rawliteral";

    // Chunk 3: Tab 2 Backtest Suite & Modals
    html += R"rawliteral(
  <!-- Tab 2: Backtesting Suite -->
  <div id="tab-backtest" class="tab-content">
    <div class="grid">
      <div class="card">
        <div class="card-title">Backtest Configuration</div>
        <div class="form-group">
          <label>Evaluation Dataset Preset</label>
          <select id="btPresetSelect" onchange="onBtPresetChange()">
            <option value="weekly_retail">Weekly Retail Demand (192 steps)</option>
            <option value="airline_passengers">Airline Passengers Benchmark (144 steps)</option>
            <option value="sunspots">Monthly Sunspots Solar Activity (180 steps)</option>
            <option value="trend_seasonal">Synthetic Trend + Seasonality (192 steps)</option>
          </select>
        </div>

        <div class="form-group">
          <label>Or Upload Multi-Step CSV</label>
          <div class="dropzone" onclick="document.getElementById('btCsvInput').click()"><p id="btFileLabel">Drop CSV or Click to Browse</p></div>
          <input type="file" id="btCsvInput" accept=".csv" onchange="onBtCsvUpload(event)">
        </div>

        <div class="form-group">
          <div class="slider-container">
            <div class="slider-header"><label>Rolling Context Window</label><span id="btContextVal" class="slider-val" style="color: var(--accent);">128 steps</span></div>
            <input type="range" id="btContextRange" min="32" max="256" step="32" value="128" oninput="document.getElementById('btContextVal').innerText = this.value + ' steps'">
          </div>
        </div>

        <div class="form-group">
          <div class="slider-container">
            <div class="slider-header"><label>Evaluation Horizon</label><span id="btHorizonVal" class="slider-val" style="color: var(--accent);">32 steps</span></div>
            <input type="range" id="btHorizonRange" min="16" max="128" step="16" value="32" oninput="document.getElementById('btHorizonVal').innerText = this.value + ' steps'">
          </div>
        </div>

        <div class="form-group">
          <div class="slider-container">
            <div class="slider-header"><label>Rolling Stride</label><span id="btStrideVal" class="slider-val" style="color: var(--accent);">32 steps</span></div>
            <input type="range" id="btStrideRange" min="16" max="128" step="16" value="32" oninput="document.getElementById('btStrideVal').innerText = this.value + ' steps'">
          </div>
        </div>

        <button id="btnBacktest" class="btn-accent" onclick="runBacktest()">Run Rolling Backtest Engine</button>
      </div>

      <div class="card">
        <div class="card-title">Backtesting Benchmark &amp; Calibration Metrics</div>
        
        <div class="stats-row">
          <div class="stat-card">
            <div class="stat-title">Mean Abs Error (MAE) <span class="tooltip-icon" data-tip="Average absolute error in raw data units.">i</span></div>
            <div class="stat-val" id="btStatMAE" style="color: var(--primary);">--</div>
          </div>
          <div class="stat-card">
            <div class="stat-title">Symmetric MAPE (sMAPE) <span class="tooltip-icon" data-tip="Scale-independent percentage error bounded 0-200%.">i</span></div>
            <div class="stat-val" id="btStatSMAPE" style="color: #38bdf8;">-- %</div>
          </div>
          <div class="stat-card">
            <div class="stat-title">Standard MAPE <span class="tooltip-icon" data-tip="Mean absolute percentage error relative to actual values.">i</span></div>
            <div class="stat-val" id="btStatMAPE" style="color: #38bdf8;">-- %</div>
          </div>
          <div class="stat-card">
            <div class="stat-title">Root Mean Sq (RMSE) <span class="tooltip-icon" data-tip="Penalizes large errors more heavily than MAE.">i</span></div>
            <div class="stat-val" id="btStatRMSE">--</div>
          </div>
          <div class="stat-card">
            <div class="stat-title">80% Interval Cov <span class="tooltip-icon" data-tip="Percentage of actual future observations falling inside the q10-q90 interval (ideal: 80%).">i</span></div>
            <div class="stat-val" id="btStatCov" style="color: #a78bfa;">-- %</div>
          </div>
          <div class="stat-card">
            <div class="stat-title">MAE vs Naive Ratio <span class="tooltip-icon" data-tip="Ratio of TimesFM MAE to Naive persistence baseline MAE. Less than 1.0 means TimesFM beats naive persistence.">i</span></div>
            <div class="stat-val" id="btStatNaiveRatio" style="color: var(--success);">--</div>
          </div>
        </div>

        <div class="chart-box"><canvas id="backtestCanvas"></canvas></div>

        <div class="table-container">
          <table id="btTable">
            <thead>
              <tr>
                <th>Window</th>
                <th>Cut-Off</th>
                <th>MAE</th>
                <th>sMAPE (%)</th>
                <th>MAPE (%)</th>
                <th>RMSE</th>
                <th>80% Cov</th>
                <th>40% Cov</th>
                <th>MAE vs Naive</th>
              </tr>
            </thead>
            <tbody id="btTableBody">
              <tr><td colspan="9" style="text-align: center; color: var(--text-muted);">Run rolling backtest to view per-window evaluation metrics</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Help & Knowledge Modal -->
<div id="helpModal" class="modal-backdrop" onclick="onModalBackdropClick(event)">
  <div class="modal-card">
    <button class="modal-close" onclick="toggleHelpModal(false)">✕</button>
    <div class="modal-title">📖 TimesFM 3.0 Knowledge &amp; User Guide</div>

    <div class="help-section">
      <h3>What is Google TimesFM 3.0?</h3>
      <p>TimesFM (Time Series Foundation Model) is Google Research's pre-trained decoder-only transformer designed for zero-shot time series forecasting. It treats time series as sequences of patches (192 input steps mapped to 64 output forecast steps) and outputs multi-step probabilistic quantile trajectories simultaneously without task-specific fine-tuning.</p>
    </div>

    <div class="help-section">
      <h3>Understanding Probabilistic Quantiles (q10 to q90)</h3>
      <p>Traditional forecasting models output only a single point estimate. TimesFM predicts 9 distinct probability quantiles:</p>
      <div class="help-grid">
        <div class="help-item">
          <strong>q50 (Median Forecast)</strong>
          <span>The 50th percentile: 50% probability the actual outcome is above, 50% below. Used as the primary point forecast.</span>
        </div>
        <div class="help-item">
          <strong>80% Confidence Band (q10 to q90)</strong>
          <span>The interval between the 10th and 90th percentile. Under ideal calibration, 80% of actual future values will land in this band.</span>
        </div>
        <div class="help-item">
          <strong>40% Confidence Band (q30 to q70)</strong>
          <span>The tighter interquartile band representing the most probable core trajectory.</span>
        </div>
        <div class="help-item">
          <strong>Uncertainty Expansion</strong>
          <span>Confidence bands naturally widen as the horizon increases, capturing growing entropy over distant futures.</span>
        </div>
      </div>
    </div>

    <div class="help-section">
      <h3>Inference Conditioning Pipelines</h3>
      <div class="help-grid">
        <div class="help-item">
          <strong>RevIN (Reversible Instance Normalization)</strong>
          <span>Standardizes input mean and variance before patching to shield the model from massive scale differences and distribution shift, denormalizing output quantiles afterward.</span>
        </div>
        <div class="help-item">
          <strong>Linear Detrending</strong>
          <span>Fits ordinary least squares slope. If R² &gt; 0.5, subtracts the linear drift before inference and re-applies it to future horizon steps to prevent mean-reversion collapse.</span>
        </div>
        <div class="help-item">
          <strong>Quantile Monotonicity</strong>
          <span>Enforces strict ascending order (q10 ≤ q20 ≤ ... ≤ q90) across each step, eliminating mathematical anomalies where quantile bands cross.</span>
        </div>
        <div class="help-item">
          <strong>Non-Negativity Constraint</strong>
          <span>Clamps negative values to 0 for physical metrics like retail sales, inventory demand, passenger counts, or website visits.</span>
        </div>
      </div>
    </div>

    <div class="help-section">
      <h3>Evaluating Backtest Metrics</h3>
      <div class="help-grid">
        <div class="help-item">
          <strong>MAE &amp; RMSE</strong>
          <span>Mean Absolute Error and Root Mean Squared Error measured in raw data units. Lower is better.</span>
        </div>
        <div class="help-item">
          <strong>sMAPE &amp; MAPE (%)</strong>
          <span>Symmetric and Standard Mean Absolute Percentage Error. Useful for comparing accuracy across different scales.</span>
        </div>
        <div class="help-item">
          <strong>80% / 40% Interval Coverage</strong>
          <span>Fraction of real ground-truth points falling inside the predicted quantile intervals (target: 80% and 40%).</span>
        </div>
        <div class="help-item">
          <strong>MAE vs Naive Ratio</strong>
          <span>Ratio comparing TimesFM error to a persistence baseline (predicting last observed value). A ratio &lt; 1.0 confirms TimesFM beats naive persistence.</span>
        </div>
      </div>
    </div>
  </div>
</div>

<div id="toastContainer" class="toast-container"></div>
)rawliteral";

    // Chunk 4: Client JS Part 1 (Presets, Event Handlers, Chart Inits)
    html += R"rawliteral(
<script>
let currentHistory = [];
let lastForecastResult = null;
let chartInstance = null;
let btChartInstance = null;
let currentViewMode = 'fan';

const PRESET_DESCRIPTIONS = {
  'weekly_retail': 'Weekly retail demand with weekly day-of-week multipliers and upward growth.',
  'airline_passengers': 'Classic Box-Jenkins monthly international airline passengers (1949-1960) with strong trend & annual seasonality.',
  'sunspots': 'Monthly smoothed solar sunspot cycles with ~11-year asymmetric solar maximum activity.',
  'trend_seasonal': 'Synthetic dataset with linear slope combined with fast weekly and slow monthly sinusoidal cycles.',
  'spiky_demand': 'Sparse inventory demand model with background base rate and Poisson shock spikes.',
  'linear_trend': 'Steady positive linear slope with additive Gaussian white noise.',
  'seasonal_sine': 'Smooth pure sine wave with period of 12 steps.',
  'random_walk': 'Financial random walk with cumulative normal innovations.'
};

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerText = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = 'opacity 0.3s, transform 0.3s';
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

function toggleHelpModal(show) {
  const modal = document.getElementById('helpModal');
  if (show) modal.classList.add('active');
  else modal.classList.remove('active');
}

function onModalBackdropClick(e) {
  if (e.target.id === 'helpModal') toggleHelpModal(false);
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') toggleHelpModal(false);
  if ((e.key === '?' || e.key === 'h' || e.key === 'H') && !['input', 'textarea'].includes(document.activeElement.tagName.toLowerCase())) {
    toggleHelpModal(true);
  }
});

function switchTab(tab) {
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  if (tab === 'forecast') {
    document.querySelector('.nav-tab:nth-child(1)').classList.add('active');
    document.getElementById('tab-forecast').classList.add('active');
  } else {
    document.querySelector('.nav-tab:nth-child(2)').classList.add('active');
    document.getElementById('tab-backtest').classList.add('active');
    if (!btChartInstance) initBtChart();
  }
}

function switchViewMode(mode) {
  currentViewMode = mode;
  document.getElementById('viewBtnFan').classList.toggle('active', mode === 'fan');
  document.getElementById('viewBtnAll').classList.toggle('active', mode === 'all');
  document.getElementById('viewBtnTrend').classList.toggle('active', mode === 'trend');
  if (lastForecastResult) {
    renderForecastResults(lastForecastResult);
  } else {
    updatePlotPreview();
  }
}

function initChart() {
  const ctx = document.getElementById('forecastCanvas').getContext('2d');
  chartInstance = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } }
      },
      plugins: {
        legend: { labels: { color: '#cbd5e1', usePointStyle: true, boxWidth: 8 } },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.92)',
          titleColor: '#38bdf8',
          bodyColor: '#f8fafc',
          borderColor: '#334155',
          borderWidth: 1,
          padding: 10
        }
      }
    }
  });
}

function initBtChart() {
  const ctx = document.getElementById('backtestCanvas').getContext('2d');
  btChartInstance = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b' } }
      },
      plugins: {
        legend: { labels: { color: '#cbd5e1', usePointStyle: true, boxWidth: 8 } },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.92)',
          borderColor: '#334155',
          borderWidth: 1
        }
      }
    }
  });
}
)rawliteral";

    // Chunk 5: Client JS Part 2 (Presets loader, CSV parsing, Forecast execution)
    html += R"rawliteral(
function onPresetChange() {
  const p = document.getElementById('presetSelect').value;
  document.getElementById('textInput').value = '';
  document.getElementById('fileLabel').innerText = "Loaded Preset: " + p;
  document.getElementById('presetBadge').innerText = PRESET_DESCRIPTIONS[p] || p;
  loadPreset(p);
  showToast(`Loaded preset: ${p}`, 'info');
}

function onBtPresetChange() {
  const p = document.getElementById('btPresetSelect').value;
  document.getElementById('btFileLabel').innerText = "Loaded Preset: " + p;
  loadPreset(p);
  showToast(`Loaded backtest dataset: ${p}`, 'info');
}

function onTextChange() {
  const txt = document.getElementById('textInput').value.trim();
  if (!txt) return;
  const vals = txt.split(/[\s,;|]+/).map(v => parseFloat(v)).filter(v => !isNaN(v));
  if (vals.length >= 8) {
    currentHistory = vals;
    lastForecastResult = null;
    document.getElementById('fileLabel').innerText = `Pasted Text (${vals.length} observations)`;
    updatePlotPreview();
  }
}

function onCsvUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(evt) {
    parseCsv(evt.target.result, 'fileLabel');
  };
  reader.readAsText(file);
}

function onBtCsvUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(evt) {
    parseCsv(evt.target.result, 'btFileLabel');
  };
  reader.readAsText(file);
}

function parseCsv(content, labelId) {
  const lines = content.split('\n');
  const values = [];
  for (let i = 1; i < lines.length; ++i) {
    const parts = lines[i].split(',');
    if (parts.length > 0) {
      const val = parseFloat(parts[parts.length - 1]);
      if (!isNaN(val)) values.push(val);
    }
  }
  if (values.length >= 8) {
    currentHistory = values;
    lastForecastResult = null;
    document.getElementById(labelId).innerText = `CSV Ingested (${values.length} rows)`;
    updatePlotPreview();
    showToast(`Successfully parsed CSV with ${values.length} rows`, 'success');
  }
}

function loadPreset(name) {
  currentHistory = [];
  lastForecastResult = null;
  let n = 192;
  if (name === "airline_passengers") {
    const air = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432];
    currentHistory = air;
  } else if (name === "weekly_retail") {
    const mults = [0.70, 0.75, 0.80, 0.85, 1.15, 1.45, 1.30];
    for (let t = 0; t < n; ++t) {
      currentHistory.push(Math.max(0, (30.0 + 0.08 * t) * mults[t % 7] + (Math.sin(t * 0.4) * 1.5)));
    }
  } else if (name === "spiky_demand") {
    for (let t = 0; t < n; ++t) {
      let v = 4.0 + 0.5 * Math.sin(2 * Math.PI * t / 16.0);
      if (t % 17 === 0) v += 12.0;
      currentHistory.push(Math.max(0, v));
    }
  } else if (name === "sunspots") {
    for (let t = 0; t < 180; ++t) {
      let cycle = Math.sin(2 * Math.PI * t / 132.0);
      let base = cycle > 0 ? cycle * cycle * 120 : cycle * 20;
      currentHistory.push(Math.max(0, base + Math.sin(t * 1.7) * 8.0 + 10.0));
    }
  } else if (name === "linear_trend") {
    for (let t = 0; t < n; ++t) currentHistory.push(10.0 + 0.25 * t + (Math.sin(t * 0.8) * 1.2));
  } else if (name === "seasonal_sine") {
    for (let t = 0; t < n; ++t) currentHistory.push(50.0 + 15.0 * Math.sin(2 * Math.PI * t / 12.0));
  } else if (name === "random_walk") {
    let cur = 100.0;
    for (let t = 0; t < n; ++t) {
      cur += Math.sin(t * 0.9) * 2.2 + (t % 5 === 0 ? -1.5 : 1.1);
      currentHistory.push(cur);
    }
  } else {
    for (let t = 0; t < n; ++t) {
      currentHistory.push(40.0 + 0.12 * t + 8.0 * Math.sin(2 * Math.PI * t / 12.0));
    }
  }
  updatePlotPreview();
}

function updatePlotPreview() {
  const labels = currentHistory.map((_, i) => 'T-' + (currentHistory.length - 1 - i));
  chartInstance.data.labels = labels;
  chartInstance.data.datasets = [
    { label: 'Historical Context', data: currentHistory, borderColor: '#38bdf8', borderWidth: 2.5, pointRadius: 0 }
  ];
  chartInstance.update();
}

async function runForecast() {
  if (currentHistory.length === 0) loadPreset('weekly_retail');

  const horizon = parseInt(document.getElementById('horizonRange').value);
  const normalize = document.getElementById('revinCheck').checked;
  const detrend = document.getElementById('detrendCheck').checked;
  const sortQuantiles = document.getElementById('sortQuantilesCheck').checked;
  const makePositive = document.getElementById('positiveCheck').checked;
  const symAvg = document.getElementById('symAvgCheck').checked;

  document.getElementById('btnForecast').disabled = true;
  document.getElementById('btnForecast').innerText = "Running Model...";

  try {
    const payload = {
      context: currentHistory,
      horizon: horizon,
      normalize: normalize,
      detrend: detrend,
      sort_quantiles: sortQuantiles,
      make_positive: makePositive,
      use_symmetric_averaging: symAvg
    };

    const res = await fetch('/api/forecast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("HTTP error " + res.status);
    const data = await res.json();
    lastForecastResult = data;

    document.getElementById('statLatency').innerText = data.inference_time_ms.toFixed(1) + " ms";
    document.getElementById('statHorizon').innerText = data.horizon + " steps";
    document.getElementById('statRevIN').innerText = data.revin.mean.toFixed(1) + " / " + data.revin.std.toFixed(1);
    document.getElementById('statTrend').innerText = data.detrending.applied ? ("R²=" + data.detrending.r2.toFixed(2)) : "None";

    const lastHist = currentHistory[currentHistory.length - 1];
    const finalQ50 = data.forecast.q50[data.horizon - 1];
    const finalQ10 = data.forecast.q10[data.horizon - 1];
    const finalQ90 = data.forecast.q90[data.horizon - 1];
    const growth = ((finalQ50 - lastHist) / (Math.abs(lastHist) + 1e-5)) * 100.0;
    const spread = (finalQ90 - finalQ10) / 2.0;

    document.getElementById('kpiFinalVal').innerText = finalQ50.toFixed(2);
    document.getElementById('kpiGrowthVal').innerText = (growth >= 0 ? "+" : "") + growth.toFixed(1) + " %";
    document.getElementById('kpiGrowthVal').style.color = growth >= 0 ? "var(--success)" : "var(--accent)";
    document.getElementById('kpiSpreadVal').innerText = "± " + spread.toFixed(2);
    document.getElementById('kpiRangeVal').innerText = `[${finalQ10.toFixed(1)}, ${finalQ90.toFixed(1)}]`;

    renderForecastResults(data);
    showToast(`Generated ${data.horizon}-step zero-shot forecast in ${data.inference_time_ms.toFixed(1)} ms`, 'success');
  } catch (err) {
    console.error(err);
    alert("Inference request failed: " + err.message);
  } finally {
    document.getElementById('btnForecast').disabled = false;
    document.getElementById('btnForecast').innerText = "Generate Zero-Shot Forecast";
  }
}
)rawliteral";

    // Chunk 6: Client JS Part 3 (Rendering, Exports, Backtesting engine)
    html += R"rawliteral(
function renderForecastResults(data) {
  const totalPoints = currentHistory.length + data.horizon;
  const labels = Array.from({length: totalPoints}, (_, i) => i < currentHistory.length ? 'T-' + (currentHistory.length - 1 - i) : 'T+' + (i - currentHistory.length + 1));

  const histPadded = currentHistory.concat(Array(data.horizon).fill(null));
  const q10 = Array(currentHistory.length).fill(null).concat(data.forecast.q10);
  const q20 = Array(currentHistory.length).fill(null).concat(data.forecast.q20 || data.forecast.q10);
  const q30 = Array(currentHistory.length).fill(null).concat(data.forecast.q30);
  const q40 = Array(currentHistory.length).fill(null).concat(data.forecast.q40 || data.forecast.q30);
  const q50 = Array(currentHistory.length - 1).fill(null).concat([currentHistory[currentHistory.length - 1]], data.forecast.q50);
  const q60 = Array(currentHistory.length).fill(null).concat(data.forecast.q60 || data.forecast.q70);
  const q70 = Array(currentHistory.length).fill(null).concat(data.forecast.q70);
  const q80 = Array(currentHistory.length).fill(null).concat(data.forecast.q80 || data.forecast.q90);
  const q90 = Array(currentHistory.length).fill(null).concat(data.forecast.q90);

  chartInstance.data.labels = labels;

  if (currentViewMode === 'fan') {
    chartInstance.data.datasets = [
      { label: 'Historical Context', data: histPadded, borderColor: '#38bdf8', borderWidth: 2.5, pointRadius: 0 },
      { label: 'Median Forecast (q50)', data: q50, borderColor: '#f97316', borderWidth: 2.8, pointRadius: 0 },
      { label: '80% Band (q10-q90)', data: q90, borderColor: 'transparent', backgroundColor: 'rgba(249, 115, 22, 0.14)', fill: '+3', pointRadius: 0 },
      { label: '40% Band (q30-q70)', data: q70, borderColor: 'transparent', backgroundColor: 'rgba(249, 115, 22, 0.26)', fill: '+1', pointRadius: 0 },
      { label: 'q30', data: q30, borderColor: 'transparent', fill: false, pointRadius: 0 },
      { label: 'q10', data: q10, borderColor: 'transparent', fill: false, pointRadius: 0 }
    ];
  } else if (currentViewMode === 'all') {
    chartInstance.data.datasets = [
      { label: 'Historical Context', data: histPadded, borderColor: '#38bdf8', borderWidth: 2.5, pointRadius: 0 },
      { label: 'q90 (Upper 90%)', data: q90, borderColor: '#ea580c', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 },
      { label: 'q80', data: q80, borderColor: '#f97316', borderDash: [2, 2], borderWidth: 1.2, pointRadius: 0 },
      { label: 'q70', data: q70, borderColor: '#fb923c', borderWidth: 1.5, pointRadius: 0 },
      { label: 'q60', data: q60, borderColor: '#fdba74', borderWidth: 1.2, pointRadius: 0 },
      { label: 'q50 (Median)', data: q50, borderColor: '#ffffff', borderWidth: 3.0, pointRadius: 0 },
      { label: 'q40', data: q40, borderColor: '#fdba74', borderWidth: 1.2, pointRadius: 0 },
      { label: 'q30', data: q30, borderColor: '#fb923c', borderWidth: 1.5, pointRadius: 0 },
      { label: 'q20', data: q20, borderColor: '#f97316', borderDash: [2, 2], borderWidth: 1.2, pointRadius: 0 },
      { label: 'q10 (Lower 10%)', data: q10, borderColor: '#ea580c', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 }
    ];
  } else if (currentViewMode === 'trend') {
    const slope = (data.detrending && data.detrending.slope) ? data.detrending.slope : 0.0;
    const mean = (data.revin && data.revin.mean) ? data.revin.mean : 0.0;
    const trendData = labels.map((_, i) => (mean + slope * (i - currentHistory.length)));
    chartInstance.data.datasets = [
      { label: 'Historical Context', data: histPadded, borderColor: '#38bdf8', borderWidth: 2.5, pointRadius: 0 },
      { label: 'Median Forecast (q50)', data: q50, borderColor: '#f97316', borderWidth: 2.8, pointRadius: 0 },
      { label: 'Extracted Linear Trend', data: trendData, borderColor: '#a855f7', borderDash: [6, 6], borderWidth: 2.0, pointRadius: 0 }
    ];
  }

  chartInstance.update();
}

function downloadCsv() {
  if (!lastForecastResult) {
    showToast("Please generate a forecast first before downloading CSV", "info");
    return;
  }
  const data = lastForecastResult;
  let csv = "Step,TimeIndex,Historical_Context,q10,q20,q30,q40,q50_Median,q60,q70,q80,q90\n";
  const H = data.horizon;
  const ctxLen = currentHistory.length;

  for (let i = 0; i < ctxLen; ++i) {
    csv += `${i + 1},T-${ctxLen - 1 - i},${currentHistory[i].toFixed(6)},,,,,,,,,\n`;
  }
  for (let t = 0; t < H; ++t) {
    const q10 = (data.forecast.q10 ? data.forecast.q10[t] : 0).toFixed(6);
    const q20 = (data.forecast.q20 ? data.forecast.q20[t] : q10).toFixed(6);
    const q30 = (data.forecast.q30 ? data.forecast.q30[t] : q10).toFixed(6);
    const q40 = (data.forecast.q40 ? data.forecast.q40[t] : q30).toFixed(6);
    const q50 = (data.forecast.q50 ? data.forecast.q50[t] : 0).toFixed(6);
    const q60 = (data.forecast.q60 ? data.forecast.q60[t] : q50).toFixed(6);
    const q70 = (data.forecast.q70 ? data.forecast.q70[t] : q50).toFixed(6);
    const q80 = (data.forecast.q80 ? data.forecast.q80[t] : q70).toFixed(6);
    const q90 = (data.forecast.q90 ? data.forecast.q90[t] : q70).toFixed(6);
    csv += `${ctxLen + t + 1},T+${t + 1},,${q10},${q20},${q30},${q40},${q50},${q60},${q70},${q80},${q90}\n`;
  }

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `timesfm_forecast_${H}steps.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast(`Downloaded timesfm_forecast_${H}steps.csv`, 'success');
}

function downloadJson() {
  if (!lastForecastResult) {
    showToast("Please generate a forecast first before downloading JSON", "info");
    return;
  }
  const jsonStr = JSON.stringify(lastForecastResult, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `timesfm_forecast_${lastForecastResult.horizon}steps.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast("Downloaded forecast JSON file", "success");
}

function copyTableToClipboard() {
  if (!lastForecastResult) {
    showToast("Please generate a forecast first", "info");
    return;
  }
  const data = lastForecastResult;
  let tsv = "Step\tTimeIndex\tHistorical\tq10\tq30\tq50 (Median)\tq70\tq90\n";
  const H = data.horizon;
  const ctxLen = currentHistory.length;

  for (let i = 0; i < ctxLen; ++i) {
    tsv += `${i + 1}\tT-${ctxLen - 1 - i}\t${currentHistory[i].toFixed(4)}\t\t\t\t\t\n`;
  }
  for (let t = 0; t < H; ++t) {
    const q10 = data.forecast.q10[t].toFixed(4);
    const q30 = data.forecast.q30[t].toFixed(4);
    const q50 = data.forecast.q50[t].toFixed(4);
    const q70 = data.forecast.q70[t].toFixed(4);
    const q90 = data.forecast.q90[t].toFixed(4);
    tsv += `${ctxLen + t + 1}\tT+${t + 1}\t\t${q10}\t${q30}\t${q50}\t${q70}\t${q90}\n`;
  }

  navigator.clipboard.writeText(tsv).then(() => {
    showToast("Copied TSV forecast table to clipboard (ready for Excel/Sheets)", "success");
  }).catch(() => {
    showToast("Failed to copy table to clipboard", "info");
  });
}

function copyJsonToClipboard() {
  if (!lastForecastResult) {
    showToast("Please generate a forecast first", "info");
    return;
  }
  navigator.clipboard.writeText(JSON.stringify(lastForecastResult, null, 2)).then(() => {
    showToast("Copied JSON forecast to clipboard", "success");
  }).catch(() => {
    showToast("Failed to copy JSON to clipboard", "info");
  });
}

async function runBacktest() {
  if (currentHistory.length === 0) loadPreset('weekly_retail');

  const ctxLen = parseInt(document.getElementById('btContextRange').value);
  const horizon = parseInt(document.getElementById('btHorizonRange').value);
  const stride = parseInt(document.getElementById('btStrideRange').value);

  document.getElementById('btnBacktest').disabled = true;
  document.getElementById('btnBacktest').innerText = "Evaluating Rolling Windows...";

  try {
    const payload = {
      series: currentHistory,
      context_len: ctxLen,
      horizon: horizon,
      stride: stride,
      max_windows: 32,
      normalize: true,
      detrend: true,
      sort_quantiles: true
    };

    const res = await fetch('/api/backtest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("HTTP error " + res.status);
    const data = await res.json();

    document.getElementById('btStatMAE').innerText = data.avg_mae.toFixed(2);
    document.getElementById('btStatSMAPE').innerText = data.avg_smape.toFixed(1) + " %";
    document.getElementById('btStatMAPE').innerText = data.avg_mape.toFixed(1) + " %";
    document.getElementById('btStatRMSE').innerText = data.avg_rmse.toFixed(2);
    document.getElementById('btStatCov').innerText = data.avg_coverage_80.toFixed(1) + " %";
    document.getElementById('btStatNaiveRatio').innerText = data.avg_mae_vs_naive_ratio.toFixed(3) + (data.avg_mae_vs_naive_ratio < 1.0 ? " (Beat)" : "");

    const tbody = document.getElementById('btTableBody');
    tbody.innerHTML = '';
    data.windows.forEach(w => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>W-${w.window_idx + 1}</td><td>T=${w.cut_t}</td><td>${w.mae.toFixed(2)}</td><td>${w.smape.toFixed(1)}%</td><td>${w.mape.toFixed(1)}%</td><td>${w.rmse.toFixed(2)}</td><td>${w.coverage_80.toFixed(1)}%</td><td>${w.coverage_40.toFixed(1)}%</td><td>${w.mae_vs_naive_ratio.toFixed(3)}</td>`;
      tbody.appendChild(tr);
    });

    if (!btChartInstance) initBtChart();
    const labels = currentHistory.map((_, i) => 'T=' + i);
    btChartInstance.data.labels = labels;
    btChartInstance.data.datasets = [
      { label: 'Full Ground Truth Series', data: currentHistory, borderColor: '#94a3b8', borderWidth: 2, pointRadius: 0 }
    ];
    btChartInstance.update();
    showToast(`Completed rolling backtest across ${data.windows.length} windows in ${data.total_time_ms.toFixed(1)} ms`, 'success');
  } catch (err) {
    console.error(err);
    alert("Backtest request failed: " + err.message);
  } finally {
    document.getElementById('btnBacktest').disabled = false;
    document.getElementById('btnBacktest').innerText = "Run Rolling Backtest Engine";
  }
}

window.addEventListener('DOMContentLoaded', () => {
  initChart();
  loadPreset('weekly_retail');
});
</script>
</body>
</html>
)rawliteral";

    return html;
}

} // namespace timesfm
