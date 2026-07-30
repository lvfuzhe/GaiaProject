"use strict";

const POLL_INTERVAL_MS = 1400;
const PLAYER_COLORS = ["#18705a", "#c14c39", "#2d68a7", "#ad751c"];
const TERRAIN_COLORS = [
  "#4d78b8", "#d7a93e", "#56a48e", "#d86b57", "#ba6f47",
  "#8b8f96", "#b7d8e7", "#7356a8", "#5bbf79"
];
const PHASE_LABELS = {
  run_started: "初始化完成",
  self_play_started: "正在生成自博弈",
  self_play_step: "正在搜索动作",
  self_play_completed: "自博弈已完成",
  training_started: "正在更新网络",
  training_update: "正在反向传播",
  arena_started: "正在运行竞技场",
  arena_completed: "竞技场已完成",
  iteration_completed: "迭代已保存",
  run_completed: "训练已完成",
  run_failed: "训练失败"
};

const state = {
  events: [],
  lastSequence: 0,
  runId: null,
  source: "--",
  connected: false,
  live: true,
  polling: false
};

const byId = (id) => document.getElementById(id);
const latest = (type) => [...state.events].reverse().find((event) => event.type === type);
const eventsOf = (type) => state.events.filter((event) => event.type === type);
const payload = (event) => event ? event.payload || {} : {};

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: digits }).format(Number(value));
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "--";
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours > 0
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function formatTime(timestamp) {
  if (!timestamp) return "--";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false
  }).format(new Date(timestamp));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function pollEvents(force = false) {
  if (state.polling || (!state.live && !force)) return;
  state.polling = true;
  try {
    const response = await fetch(`/api/events?after=${state.lastSequence}&limit=5000`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    state.source = data.source || state.source;
    state.connected = true;
    for (const event of data.events || []) {
      if (event.type === "run_started" && state.runId && event.run_id !== state.runId) {
        state.events = [];
      }
      if (event.type === "run_started") state.runId = event.run_id;
      if (!state.runId || event.run_id === state.runId) state.events.push(event);
      state.lastSequence = Math.max(state.lastSequence, Number(event.sequence) || 0);
    }
    if (!state.runId && state.events.length) state.runId = state.events.at(-1).run_id;
  } catch (error) {
    state.connected = false;
  } finally {
    state.polling = false;
    render();
  }
}

function render() {
  renderStatus();
  renderMetrics();
  renderProgress();
  renderLossChart();
  renderIterations();
  renderSelfPlay();
  renderDiagnostics();
  byId("footer-source").textContent = `metrics: ${state.source}`;
  byId("footer-clock").textContent = formatTime(new Date().toISOString());
}

function runState() {
  const event = state.events.at(-1);
  if (!event) return "waiting";
  if (event.type === "run_failed") return "failed";
  if (event.type === "run_completed") return "complete";
  return "running";
}

function renderStatus() {
  const status = runState();
  const last = state.events.at(-1);
  const labels = {
    waiting: "等待数据",
    running: "训练运行中",
    complete: "训练已完成",
    failed: "训练失败"
  };
  byId("run-status").textContent = state.connected ? labels[status] : "监控服务失联";
  byId("phase-label").textContent = last ? (PHASE_LABELS[last.type] || last.type) : "尚未发现训练事件";
  const dot = byId("connection-dot");
  dot.className = `status-dot ${state.connected ? status : "failed"}`;
}

function currentIteration() {
  const event = [...state.events].reverse().find((item) => payload(item).iteration !== undefined);
  return Number(payload(event).iteration || 0);
}

function renderMetrics() {
  const start = latest("run_started");
  const config = payload(start).config || {};
  const iteration = currentIteration();
  const completedGames = eventsOf("self_play_completed");
  const game = completedGames.at(-1);
  const update = latest("training_update") || latest("iteration_completed");
  const updateData = payload(update);
  const replay = Number(updateData.replay_positions || payload(game).replay_positions || 0);
  const positions = completedGames.reduce((total, event) => total + Number(payload(event).positions || 0), 0);
  const recentGames = completedGames.slice(-8);
  const recentPositions = recentGames.reduce((total, event) => total + Number(payload(event).positions || 0), 0);
  const recentDuration = recentGames.reduce((total, event) => total + Number(payload(event).duration_seconds || 0), 0);
  const throughput = recentDuration > 0 ? recentPositions / recentDuration : null;

  byId("metric-iteration").textContent = `${iteration} / ${config.iterations || 0}`;
  byId("metric-iteration-note").textContent = iteration ? `当前第 ${iteration} 轮训练` : "未开始";
  byId("metric-games").textContent = formatNumber(completedGames.length);
  byId("metric-games-note").textContent = `${formatNumber(positions)} 个训练局面`;
  byId("metric-replay").textContent = formatNumber(replay);
  const capacity = Number(config.replay_capacity || 0);
  byId("metric-replay-note").textContent = capacity ? `容量 ${formatNumber(replay / capacity * 100, 1)}%` : "容量 --";
  byId("metric-loss").textContent = update ? formatNumber(updateData.loss, 4) : "--";
  byId("metric-loss-note").textContent = update
    ? `策略 ${formatNumber(updateData.policy_loss, 3)} · 价值 ${formatNumber(updateData.value_loss, 3)}`
    : "等待反向传播";
  byId("metric-throughput").textContent = throughput === null ? "--" : formatNumber(throughput, 1);

  const end = latest("run_completed") || latest("run_failed");
  let elapsed = 0;
  if (start) {
    elapsed = end
      ? (new Date(end.timestamp) - new Date(start.timestamp)) / 1000
      : (Date.now() - new Date(start.timestamp).getTime()) / 1000;
  }
  byId("metric-elapsed").textContent = formatDuration(elapsed);
  byId("metric-device").textContent = `设备 ${payload(start).device || "--"}`;
}

function renderProgress() {
  const start = latest("run_started");
  const config = payload(start).config || {};
  const iterations = Number(config.iterations || 0);
  const iteration = currentIteration();
  const last = state.events.at(-1);
  const data = payload(last);
  let within = 0;
  if (last && last.type.startsWith("self_play")) {
    within = 0.7 * Number(data.game_in_iteration || 0) / Math.max(1, Number(data.games_per_iteration || config.games_per_iteration || 1));
  } else if (last && (last.type === "training_started" || last.type === "training_update")) {
    within = 0.7 + 0.2 * Number(data.update || 0) / Math.max(1, Number(data.updates || config.updates_per_iteration || 1));
  } else if (last && last.type.startsWith("arena")) {
    within = 0.94;
  } else if (last && ["iteration_completed", "run_completed"].includes(last.type)) {
    within = 1;
  }
  const progress = iterations ? Math.min(1, Math.max(0, ((iteration - 1) + within) / iterations)) : 0;
  byId("progress-fill").value = progress * 100;
  byId("progress-percent").textContent = `${formatNumber(progress * 100, 0)}%`;

  const currentType = last ? last.type : "";
  const phases = {
    "phase-selfplay": currentType.startsWith("self_play") ? "active" : (currentType && !currentType.startsWith("self_play") ? "done" : ""),
    "phase-training": currentType.startsWith("training") ? "active" : (["arena_started", "arena_completed", "iteration_completed", "run_completed"].includes(currentType) ? "done" : ""),
    "phase-arena": currentType.startsWith("arena") ? "active" : (["iteration_completed", "run_completed"].includes(currentType) ? "done" : ""),
    "phase-checkpoint": currentType === "iteration_completed" ? "active" : (currentType === "run_completed" ? "done" : "")
  };
  for (const [id, className] of Object.entries(phases)) byId(id).className = className;

  const gameStart = latest("self_play_started");
  const trainUpdate = latest("training_update");
  const arena = latest("arena_completed");
  const checkpoint = latest("iteration_completed");
  byId("phase-selfplay-note").textContent = gameStart
    ? `第 ${payload(gameStart).game_in_iteration} / ${payload(gameStart).games_per_iteration} 局`
    : "等待";
  byId("phase-training-note").textContent = trainUpdate
    ? `更新 ${payload(trainUpdate).update} / ${payload(trainUpdate).updates}`
    : "等待";
  byId("phase-arena-note").textContent = arena
    ? `均值 ${formatNumber(payload(arena).mean_value, 3)}`
    : "等待";
  byId("phase-checkpoint-note").textContent = checkpoint
    ? `迭代 ${payload(checkpoint).iteration} 已保存`
    : "等待";
}

function setupCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

function renderLossChart() {
  const canvas = byId("loss-chart");
  const { context, width, height } = setupCanvas(canvas);
  context.clearRect(0, 0, width, height);
  const updates = eventsOf("training_update").slice(-160);
  byId("loss-empty").hidden = updates.length > 0;
  if (!updates.length) return;

  const series = [
    { key: "loss", color: "#c14c39" },
    { key: "policy_loss", color: "#2d68a7" },
    { key: "value_loss", color: "#18705a" }
  ];
  const all = updates.flatMap((event) => series.map((line) => Number(payload(event)[line.key] || 0)));
  const min = Math.min(...all, 0);
  const max = Math.max(...all, 1);
  const padding = { left: 44, right: 12, top: 12, bottom: 28 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  context.strokeStyle = "#e3e7e3";
  context.fillStyle = "#7a847e";
  context.font = "10px Segoe UI";
  context.lineWidth = 1;
  for (let row = 0; row <= 4; row += 1) {
    const y = padding.top + chartHeight * row / 4;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    const value = max - (max - min) * row / 4;
    context.fillText(value.toFixed(2), 4, y + 3);
  }

  for (const line of series) {
    context.beginPath();
    context.strokeStyle = line.color;
    context.lineWidth = 2;
    updates.forEach((event, index) => {
      const x = padding.left + chartWidth * index / Math.max(1, updates.length - 1);
      const normalized = (Number(payload(event)[line.key]) - min) / Math.max(0.0001, max - min);
      const y = padding.top + chartHeight * (1 - normalized);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  }
  context.fillStyle = "#7a847e";
  context.fillText("早", padding.left, height - 7);
  context.fillText("最近", width - padding.right - 23, height - 7);
}

function renderIterations() {
  const iterations = eventsOf("iteration_completed").slice().reverse();
  byId("iteration-count").textContent = `${iterations.length} 条记录`;
  byId("iteration-table").innerHTML = iterations.length
    ? iterations.map((event) => {
      const item = payload(event);
      return `<tr>
        <td>${formatNumber(item.iteration)}</td>
        <td>${formatNumber(item.new_positions)}</td>
        <td>${formatNumber(item.replay_positions)}</td>
        <td>${formatNumber(item.loss, 4)}</td>
        <td>${formatNumber(item.policy_loss, 3)} / ${formatNumber(item.value_loss, 3)}</td>
        <td>${formatDuration(Number(item.duration_seconds))}</td>
      </tr>`;
    }).join("")
    : '<tr><td colspan="6" class="empty-cell">暂无迭代结果</td></tr>';
}

function latestState() {
  const event = [...state.events].reverse().find((item) => payload(item).state);
  return payload(event).state || null;
}

function renderSelfPlay() {
  const snapshot = latestState();
  renderBoard(snapshot);
  renderPlayers(snapshot);
  renderLatestAction();
  renderGames();
}

function renderBoard(snapshot) {
  const canvas = byId("board-canvas");
  const { context, width, height } = setupCanvas(canvas);
  context.clearRect(0, 0, width, height);
  byId("board-empty").hidden = Boolean(snapshot);
  byId("board-round").textContent = snapshot ? `第 ${snapshot.round} / ${snapshot.max_rounds} 轮` : "第 -- 轮";
  if (!snapshot || !snapshot.planets?.length) return;

  const points = snapshot.planets.map((planet) => ({
    ...planet,
    rawX: Math.sqrt(3) * (planet.q + planet.r / 2),
    rawY: 1.5 * planet.r
  }));
  const minX = Math.min(...points.map((point) => point.rawX));
  const maxX = Math.max(...points.map((point) => point.rawX));
  const minY = Math.min(...points.map((point) => point.rawY));
  const maxY = Math.max(...points.map((point) => point.rawY));
  const scale = Math.min((width - 70) / Math.max(1, maxX - minX), (height - 70) / Math.max(1, maxY - minY));
  const size = Math.max(17, Math.min(31, scale * 0.47));
  const offsetX = (width - (maxX - minX) * scale) / 2 - minX * scale;
  const offsetY = (height - (maxY - minY) * scale) / 2 - minY * scale;

  for (const planet of points) {
    const x = offsetX + planet.rawX * scale;
    const y = offsetY + planet.rawY * scale;
    drawHex(context, x, y, size, TERRAIN_COLORS[planet.terrain] || "#c7cec8");
    context.fillStyle = "rgba(255,255,255,0.86)";
    context.font = "700 9px Segoe UI";
    context.textAlign = "center";
    context.fillText(String(planet.id), x, y + 3);
    if (planet.owner >= 0) {
      context.strokeStyle = PLAYER_COLORS[planet.owner] || "#17211d";
      context.lineWidth = 4;
      context.beginPath();
      context.arc(x, y, size + 4, 0, Math.PI * 2);
      context.stroke();
      drawBuilding(context, x, y, planet.building, PLAYER_COLORS[planet.owner]);
    }
    if (planet.gaiaformer >= 0 && planet.owner < 0) {
      context.strokeStyle = PLAYER_COLORS[planet.gaiaformer] || "#17211d";
      context.lineWidth = 3;
      context.setLineDash([4, 3]);
      context.beginPath();
      context.arc(x, y, size + 4, 0, Math.PI * 2);
      context.stroke();
      context.setLineDash([]);
      context.fillStyle = PLAYER_COLORS[planet.gaiaformer] || "#17211d";
      context.font = "700 11px Segoe UI";
      context.fillText("G", x, y + 4);
    }
    if (planet.federated && planet.owner >= 0) {
      context.strokeStyle = "rgba(23,33,29,0.7)";
      context.lineWidth = 1;
      context.setLineDash([2, 3]);
      context.beginPath();
      context.arc(x, y, size + 9, 0, Math.PI * 2);
      context.stroke();
      context.setLineDash([]);
    }
  }
}

function drawHex(context, x, y, size, fill) {
  context.beginPath();
  for (let side = 0; side < 6; side += 1) {
    const angle = Math.PI / 180 * (60 * side - 30);
    const px = x + size * Math.cos(angle);
    const py = y + size * Math.sin(angle);
    if (side === 0) context.moveTo(px, py); else context.lineTo(px, py);
  }
  context.closePath();
  context.fillStyle = fill;
  context.fill();
  context.strokeStyle = "rgba(23,33,29,0.25)";
  context.lineWidth = 1;
  context.stroke();
}

function drawBuilding(context, x, y, building, color) {
  context.fillStyle = color;
  context.strokeStyle = "#ffffff";
  context.lineWidth = 1.5;
  if (building === "mine") {
    context.beginPath();
    context.arc(x, y, 6, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  } else if (building === "trading_station") {
    context.fillRect(x - 7, y - 7, 14, 14);
    context.strokeRect(x - 7, y - 7, 14, 14);
  } else if (building === "research_lab") {
    context.beginPath();
    context.moveTo(x, y - 9);
    context.lineTo(x + 8, y + 7);
    context.lineTo(x - 8, y + 7);
    context.closePath();
    context.fill();
    context.stroke();
  } else if (building === "planetary_institute") {
    context.beginPath();
    context.moveTo(x, y - 10);
    context.lineTo(x + 10, y);
    context.lineTo(x, y + 10);
    context.lineTo(x - 10, y);
    context.closePath();
    context.fill();
    context.stroke();
  } else if (building === "academy") {
    context.fillRect(x - 9, y - 6, 18, 12);
    context.fillRect(x - 5, y - 10, 10, 20);
    context.strokeRect(x - 9, y - 6, 18, 12);
    context.strokeRect(x - 5, y - 10, 10, 20);
  }
}

function renderPlayers(snapshot) {
  const players = snapshot?.players || [];
  const scores = snapshot?.scores || [];
  byId("active-player-note").textContent = snapshot?.current_player === null || snapshot?.current_player === undefined
    ? "对局已结束"
    : `当前行动 P${snapshot.current_player}`;
  byId("players-table").innerHTML = players.length
    ? players.map((player) => `<tr class="${player.id === snapshot.current_player ? "active-row" : ""}">
        <td><span class="player-label"><i class="player-color p${player.id}"></i>P${player.id}${player.faction ? ` · ${escapeHtml(player.faction)}` : ""}</span></td>
        <td>${formatNumber(scores[player.id], 1)}</td>
        <td>${formatNumber(player.credits)}</td>
        <td>${formatNumber(player.ore)}</td>
        <td>${formatNumber(player.knowledge)}</td>
        <td>${formatNumber(player.qic)}</td>
        <td class="mono">${player.power ? player.power.join(" / ") : "--"}</td>
        <td class="mono">${player.tracks.map((value) => formatNumber(value)).join(" · ")}</td>
        <td>${formatNumber(player.federations)}</td>
        <td>${player.passed ? "已过轮" : "行动中"}</td>
      </tr>`).join("")
    : '<tr><td colspan="10" class="empty-cell">暂无玩家状态</td></tr>';
}

function renderLatestAction() {
  const event = latest("self_play_step");
  const item = payload(event);
  byId("action-player").textContent = event ? `P${item.player}` : "P--";
  byId("action-label").textContent = item.action_label || "等待搜索结果";
  const candidates = item.candidates || [];
  byId("candidate-list").innerHTML = candidates.length
    ? candidates.map((candidate) => `<div class="candidate-row">
        <div>
          <strong>${escapeHtml(candidate.label)}</strong>
          <small>${formatNumber(candidate.visits)} visits</small>
          <progress class="mini-track" max="1" value="${Number(candidate.probability)}"></progress>
        </div>
        <div class="candidate-probability">${formatNumber(Number(candidate.probability) * 100, 1)}%</div>
      </div>`).join("")
    : '<div class="candidate-empty">暂无候选动作</div>';
  byId("root-values").innerHTML = item.root_value
    ? item.root_value.map((value, index) => `<span class="value-pill">P${index} ${formatNumber(value, 2)}</span>`).join("")
    : "--";
}

function renderGames() {
  const games = eventsOf("self_play_completed").slice().reverse();
  byId("games-count").textContent = `${games.length} 局`;
  byId("games-table").innerHTML = games.length
    ? games.slice(0, 20).map((event) => {
      const item = payload(event);
      return `<tr>
        <td>${formatNumber(item.iteration)}</td>
        <td>${formatNumber(item.game_in_iteration)}</td>
        <td>${formatNumber(item.moves)}</td>
        <td>${formatNumber(item.positions)}</td>
        <td class="mono">${(item.scores || []).map((score) => formatNumber(score, 1)).join(" / ")}</td>
        <td>${formatDuration(Number(item.duration_seconds))}</td>
      </tr>`;
    }).join("")
    : '<tr><td colspan="6" class="empty-cell">暂无完成对局</td></tr>';
}

function renderDiagnostics() {
  const start = latest("run_started");
  const config = payload(start).config || {};
  byId("run-id").textContent = state.runId ? `run ${state.runId}` : "run --";
  const preferred = [
    ["players", "玩家"], ["iterations", "迭代"], ["games_per_iteration", "每轮对局"],
    ["updates_per_iteration", "每轮更新"], ["simulations", "搜索模拟"], ["batch_size", "批大小"],
    ["hidden_size", "隐藏层"], ["residual_blocks", "残差块"], ["learning_rate", "学习率"],
    ["replay_capacity", "回放容量"], ["seed", "随机种子"], ["output", "检查点"]
  ];
  byId("config-list").innerHTML = start
    ? preferred.filter(([key]) => config[key] !== undefined).map(([key, label]) =>
      `<div><dt>${label}</dt><dd class="${key === "output" ? "mono" : ""}">${escapeHtml(config[key])}</dd></div>`
    ).join("")
    : "<div><dt>状态</dt><dd>等待训练数据</dd></div>";

  const last = state.events.at(-1);
  const badge = byId("health-badge");
  badge.className = `health-badge ${state.connected ? "connected" : "failed"}`;
  badge.textContent = state.connected ? "已连接" : "失联";
  byId("health-source").textContent = state.source;
  byId("health-updated").textContent = last ? `${formatTime(last.timestamp)} · #${last.sequence}` : "--";
  byId("health-events").textContent = formatNumber(state.events.length);
  byId("health-parameters").textContent = formatNumber(payload(start).model_parameters);
  byId("health-shapes").textContent = start
    ? `${payload(start).observation_size} / ${payload(start).action_size}`
    : "--";

  const events = state.events.slice(-120).reverse();
  byId("event-count").textContent = `${state.events.length} 条`;
  byId("event-list").innerHTML = events.length
    ? events.map((event) => `<div class="event-row">
        <span class="event-time">${formatTime(event.timestamp)}</span>
        <span class="event-type">${escapeHtml(event.type)}</span>
        <span class="event-detail">${escapeHtml(eventSummary(event))}</span>
      </div>`).join("")
    : '<div class="event-empty">暂无事件</div>';
}

function eventSummary(event) {
  const item = payload(event);
  if (event.type === "self_play_step") return `P${item.player} · ${item.action_label}`;
  if (event.type === "self_play_completed") return `第 ${item.game_in_iteration} 局 · ${item.moves} 步 · ${formatNumber(item.duration_seconds, 2)}s`;
  if (event.type === "training_update") return `update ${item.update}/${item.updates} · loss ${formatNumber(item.loss, 4)}`;
  if (event.type === "iteration_completed") return `iteration ${item.iteration} · replay ${formatNumber(item.replay_positions)}`;
  if (event.type === "arena_completed") return `${item.games} games · value ${formatNumber(item.mean_value, 3)}`;
  if (event.type === "run_failed") return `${item.error_type}: ${item.message}`;
  if (event.type === "run_completed") return `${item.total_games} games · ${formatDuration(Number(item.duration_seconds))}`;
  return PHASE_LABELS[event.type] || "";
}

function selectView(name) {
  const selected = ["overview", "selfplay", "diagnostics"].includes(name) ? name : "overview";
  document.querySelectorAll(".view").forEach((view) => {
    const active = view.id === selected;
    view.classList.toggle("active", active);
    view.hidden = !active;
  });
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.view === selected;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  window.location.hash = selected;
  requestAnimationFrame(() => {
    renderLossChart();
    renderBoard(latestState());
  });
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => selectView(tab.dataset.view));
});
byId("live-toggle").addEventListener("change", (event) => {
  state.live = event.target.checked;
  if (state.live) pollEvents(true);
});
byId("refresh-button").addEventListener("click", () => pollEvents(true));
window.addEventListener("resize", () => {
  renderLossChart();
  renderBoard(latestState());
});

selectView(window.location.hash.replace("#", "") || "overview");
pollEvents(true);
setInterval(() => pollEvents(false), POLL_INTERVAL_MS);
setInterval(() => {
  renderStatus();
  renderMetrics();
  byId("footer-clock").textContent = formatTime(new Date().toISOString());
}, 1000);
