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
  polling: false,
  history: {
    index: null,
    runId: null,
    iteration: null,
    game: null,
    trace: null,
    step: 0,
    loading: false,
    playing: false,
    timer: null
  }
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
    let historyChanged = !state.history.index;
    for (const event of data.events || []) {
      if (event.type === "run_started" && state.runId && event.run_id !== state.runId) {
        state.events = [];
      }
      if (event.type === "run_started") state.runId = event.run_id;
      if (!state.runId || event.run_id === state.runId) state.events.push(event);
      state.lastSequence = Math.max(state.lastSequence, Number(event.sequence) || 0);
      if (["run_started", "self_play_completed", "iteration_completed", "run_completed", "run_failed"].includes(event.type)) {
        historyChanged = true;
      }
    }
    if (!state.runId && state.events.length) state.runId = state.events.at(-1).run_id;
    if (historyChanged) await refreshHistoryIndex();
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
  renderHistory();
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

function historyRuns() {
  return state.history.index?.runs || [];
}

function historyRun() {
  return historyRuns().find((run) => run.run_id === state.history.runId) || null;
}

function historyIteration() {
  return historyRun()?.iterations?.find((item) => item.iteration === state.history.iteration) || null;
}

function historyGame() {
  return historyIteration()?.games?.find((item) => item.game === state.history.game) || null;
}

async function refreshHistoryIndex() {
  try {
    const response = await fetch("/api/history", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const index = await response.json();
    state.history.index = index;
    const runs = historyRuns();
    if (!runs.length) {
      state.history.runId = null;
      state.history.iteration = null;
      state.history.game = null;
      state.history.trace = null;
      renderHistory();
      return;
    }
    const preferredRun = runs.find((run) => run.run_id === state.history.runId)
      || runs.find((run) => run.run_id === state.runId)
      || runs.at(-1);
    state.history.runId = preferredRun.run_id;
    const iterations = preferredRun.iterations || [];
    const preferredIteration = iterations.find((item) => item.iteration === state.history.iteration)
      || iterations.at(-1);
    state.history.iteration = preferredIteration?.iteration ?? null;
    const games = preferredIteration?.games || [];
    const preferredGame = games.find((item) => item.game === state.history.game) || games.at(-1);
    state.history.game = preferredGame?.game ?? null;
    renderHistorySelectors();
    if (state.history.runId && state.history.iteration !== null && state.history.game !== null) {
      const current = state.history.trace;
      const sameGame = current
        && current.run_id === state.history.runId
        && current.iteration === state.history.iteration
        && current.game === state.history.game;
      await loadHistoryGame(!sameGame);
    } else {
      state.history.trace = null;
      renderHistory();
    }
  } catch (error) {
    state.history.index = state.history.index || { runs: [] };
    renderHistory();
  }
}

async function loadHistoryGame(goToEnd = true) {
  const { runId, iteration, game } = state.history;
  if (!runId || iteration === null || game === null) {
    state.history.trace = null;
    renderHistory();
    return;
  }
  state.history.loading = true;
  renderHistory();
  try {
    const params = new URLSearchParams({ run_id: runId, iteration: String(iteration), game: String(game) });
    const response = await fetch(`/api/game?${params.toString()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const trace = await response.json();
    if (trace.run_id === state.history.runId
      && trace.iteration === state.history.iteration
      && trace.game === state.history.game) {
      state.history.trace = trace;
      state.history.step = goToEnd ? Math.max(0, trace.steps.length - 1) : Math.min(state.history.step, trace.steps.length - 1);
    }
  } catch (error) {
    state.history.trace = null;
  } finally {
    state.history.loading = false;
    renderHistory();
  }
}

function renderHistorySelectors() {
  const runSelect = byId("history-run-select");
  const iterationSelect = byId("history-iteration-select");
  const gameSelect = byId("history-game-select");
  if (!runSelect || !iterationSelect || !gameSelect) return;
  const runs = historyRuns();
  runSelect.innerHTML = runs.length
    ? runs.map((run) => `<option value="${escapeHtml(run.run_id)}">${escapeHtml(run.ruleset || "unknown")} · ${escapeHtml(run.run_id)} · ${escapeHtml(run.status)}</option>`).join("")
    : '<option value="">暂无运行</option>';
  runSelect.value = state.history.runId || "";
  const iterations = historyRun()?.iterations || [];
  iterationSelect.innerHTML = iterations.length
    ? iterations.map((item) => `<option value="${item.iteration}">第 ${item.iteration} 轮 · ${item.games.length} 局</option>`).join("")
    : '<option value="">暂无迭代</option>';
  iterationSelect.value = state.history.iteration === null ? "" : String(state.history.iteration);
  const games = historyIteration()?.games || [];
  gameSelect.innerHTML = games.length
    ? games.map((item) => {
      const score = item.scores ? ` · ${(item.scores || []).map((value) => formatNumber(value, 1)).join("/")}` : "";
      const coverage = item.moves === null ? "" : ` · ${item.captured_moves}/${item.moves} 步`;
      return `<option value="${item.game}">第 ${item.game} 局${score}${coverage}</option>`;
    }).join("")
    : '<option value="">暂无对局</option>';
  gameSelect.value = state.history.game === null ? "" : String(state.history.game);
  runSelect.disabled = !runs.length;
  iterationSelect.disabled = !iterations.length;
  gameSelect.disabled = !games.length;
}

function auditHistoryTrace(trace) {
  if (!trace) return [];
  const steps = trace.steps || [];
  const summary = trace.summary || {};
  const expected = Number(summary.moves);
  const moves = steps.filter((step) => step.move > 0);
  const states = steps.map((step) => step.state).filter(Boolean);
  const checks = [];
  const complete = Number.isFinite(expected) && trace.trace_complete
    && moves.length === expected;
  checks.push({
    status: complete ? "pass" : "warn",
    title: complete ? "每一步规则状态均已记录" : "规则状态记录不完整",
    detail: Number.isFinite(expected) ? `${trace.captured_moves} / ${expected} 步` : "对局尚未结束或旧日志未记录步数"
  });
  const rounds = states.map((snapshot) => Number(snapshot.round)).filter(Number.isFinite);
  const roundsValid = rounds.every((round, index) => round >= 1 && round <= 6 && (index === 0 || round >= rounds[index - 1]));
  checks.push({
    status: roundsValid && rounds.length ? "pass" : "fail",
    title: roundsValid && rounds.length ? "轮次顺序连续" : "轮次顺序异常",
    detail: rounds.length ? `观测到 ${new Set(rounds).size} 个轮次` : "没有可验证的状态"
  });
  const resourcesValid = states.every((snapshot) => (snapshot.players || []).every((player) => (
    Number(player.credits) >= 0 && Number(player.credits) <= 30
    && Number(player.ore) >= 0 && Number(player.ore) <= 15
    && Number(player.knowledge) >= 0 && Number(player.knowledge) <= 15
    && Number(player.qic) >= 0
    && (!player.power || player.power.every((value) => Number(value) >= 0))
  )));
  checks.push({
    status: resourcesValid ? "pass" : "fail",
    title: resourcesValid ? "资源和能量均在规则上限内" : "资源或能量超出边界",
    detail: resourcesValid ? "信用点 0-30、矿石/知识 0-15" : "请定位到对应行动步"
  });
  const ownersValid = states.every((snapshot) => {
    const count = (snapshot.players || []).length;
    return (snapshot.planets || []).every((planet) => Number(planet.owner) >= -1 && Number(planet.owner) < count);
  });
  checks.push({
    status: ownersValid ? "pass" : "fail",
    title: ownersValid ? "星球所有权状态有效" : "星球所有权状态异常",
    detail: ownersValid ? "每个星球最多归属一个玩家" : "发现未知玩家编号"
  });
  const lastSnapshot = states.at(-1);
  const terminal = Boolean(lastSnapshot?.terminal) && Boolean(summary.moves !== undefined);
  checks.push({
    status: terminal ? "pass" : "warn",
    title: terminal ? "整局已进入终局状态" : "整局尚未完成",
    detail: terminal ? `最终分数 ${(summary.scores || lastSnapshot.scores || []).map((value) => formatNumber(value, 1)).join(" / ")}` : "训练中的对局会随新事件更新"
  });
  const actionMetadata = moves.every((step) => step.action !== null && step.legal_actions !== null);
  checks.push({
    status: actionMetadata ? "pass" : "warn",
    title: actionMetadata ? "动作由规则引擎合法转移" : "旧日志缺少动作元数据",
    detail: actionMetadata ? "每个动作在 apply 前均经过合法动作集合校验" : "重新训练可获得完整动作审计信息"
  });
  return checks;
}

function historyDelta(previous, current, step) {
  if (!current) return "等待状态变化";
  if (!previous) return "初始状态 · 没有前置动作";
  const changes = [];
  const labels = [["credits", "信用点"], ["ore", "矿石"], ["knowledge", "知识"], ["qic", "QIC"], ["vp", "VP"]];
  const player = Number(step.player);
  const before = previous.players?.[player];
  const after = current.players?.[player];
  if (before && after) {
    for (const [key, label] of labels) {
      const delta = Number(after[key]) - Number(before[key]);
      if (delta) changes.push(`${label} ${delta > 0 ? "+" : ""}${delta}`);
    }
    if (before.power && after.power && before.power.join() !== after.power.join()) {
      changes.push(`能量 ${before.power.join("/")} → ${after.power.join("/")}`);
    }
    if (before.tracks && after.tracks && before.tracks.join() !== after.tracks.join()) {
      changes.push(`科研 ${after.tracks.join("·")}`);
    }
  }
  if (Number(previous.round) !== Number(current.round)) changes.push(`进入第 ${current.round} 轮`);
  return changes.length ? changes.join(" · ") : "状态已转移，资源数值无变化";
}

function renderHistory() {
  const content = byId("history-content");
  const empty = byId("history-empty");
  if (!content || !empty) return;
  renderHistorySelectors();
  const trace = state.history.trace;
  const hasTrace = Boolean(trace?.steps?.length);
  empty.hidden = hasTrace || state.history.loading;
  content.hidden = !hasTrace;
  if (!hasTrace) {
    if (state.history.loading) empty.textContent = "正在读取历史快照";
    return;
  }
  const steps = trace.steps;
  state.history.step = Math.max(0, Math.min(state.history.step, steps.length - 1));
  const step = steps[state.history.step];
  const snapshot = step.state;
  drawBoard(byId("history-board-canvas"), snapshot);
  byId("history-board-empty").hidden = Boolean(snapshot);
  byId("history-board-round").textContent = snapshot ? `第 ${snapshot.round} / ${snapshot.max_rounds} 轮` : "第 -- 轮";
  const summary = trace.summary || {};
  byId("history-final-scores").textContent = (summary.scores || snapshot?.scores || []).map((value) => formatNumber(value, 1)).join(" / ") || "--";
  byId("history-trace-coverage").textContent = summary.moves === undefined ? `${trace.captured_moves} 步` : `${trace.captured_moves} / ${summary.moves} 步`;
  byId("history-duration").textContent = summary.duration_seconds === undefined ? "--" : formatDuration(Number(summary.duration_seconds));
  byId("history-ruleset").textContent = snapshot?.ruleset || "--";
  byId("history-action-code").textContent = step.action === null || step.action === undefined ? "--" : String(step.action);
  byId("history-action-label").textContent = step.action_label || "状态快照";
  byId("history-action-player").textContent = step.player === null || step.player === undefined ? "P--" : `P${step.player}`;
  byId("history-step-slider").max = String(Math.max(0, steps.length - 1));
  byId("history-step-slider").value = String(state.history.step);
  byId("history-step-label").textContent = `${step.move} / ${Math.max(0, steps.length - 1)}`;
  const previous = steps[state.history.step - 1]?.state;
  byId("history-delta").textContent = historyDelta(previous, snapshot, step);
  renderPlayerRows("history-players-table", snapshot, "history-active-player");
  const checks = auditHistoryTrace(trace);
  const failed = checks.some((check) => check.status === "fail");
  const warned = checks.some((check) => check.status === "warn");
  const badge = byId("history-audit-badge");
  badge.className = `health-badge ${failed ? "failed" : warned ? "warning" : "connected"}`;
  badge.textContent = failed ? "发现异常" : warned ? "需补录" : "通过";
  byId("history-check-list").innerHTML = checks.map((check) => `<li class="${check.status}"><span>${escapeHtml(check.title)}</span><small>${escapeHtml(check.detail)}</small></li>`).join("");
  byId("history-move-count").textContent = `${Math.max(0, steps.length - 1)} 步`;
  byId("history-action-table").innerHTML = steps.map((item, index) => {
    const itemState = item.state || {};
    const scores = (itemState.scores || []).map((value) => formatNumber(value, 1)).join(" / ");
    return `<tr data-step="${index}" class="${index === state.history.step ? "current-step" : ""}">
      <td>${formatNumber(item.move)}</td>
      <td>${formatNumber(itemState.round)}</td>
      <td>${item.player === null || item.player === undefined ? "--" : `P${item.player}`}</td>
      <td>${escapeHtml(item.action_label || "状态快照")}</td>
      <td class="mono">${item.action === null || item.action === undefined ? "--" : item.action}</td>
      <td class="mono">${scores || "--"}</td>
    </tr>`;
  }).join("");
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
  byId("board-empty").hidden = Boolean(snapshot);
  byId("board-round").textContent = snapshot ? `第 ${snapshot.round} / ${snapshot.max_rounds} 轮` : "第 -- 轮";
  drawBoard(byId("board-canvas"), snapshot);
}

function drawBoard(canvas, snapshot) {
  if (!canvas) return;
  const { context, width, height } = setupCanvas(canvas);
  context.clearRect(0, 0, width, height);
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
  byId("active-player-note").textContent = snapshot?.current_player === null || snapshot?.current_player === undefined
    ? "对局已结束"
    : `当前行动 P${snapshot.current_player}`;
  renderPlayerRows("players-table", snapshot);
}

function renderPlayerRows(tableId, snapshot, noteId = null) {
  const players = snapshot?.players || [];
  const scores = snapshot?.scores || [];
  if (noteId) {
    byId(noteId).textContent = snapshot?.current_player === null || snapshot?.current_player === undefined
      ? "对局已结束"
      : `当前行动 P${snapshot.current_player}`;
  }
  byId(tableId).innerHTML = players.length
    ? players.map((player) => `<tr class="${player.id === snapshot.current_player ? "active-row" : ""}">
        <td><span class="player-label"><i class="player-color p${player.id}"></i>P${player.id}${player.faction ? ` · ${escapeHtml(player.faction)}` : ""}</span></td>
        <td>${formatNumber(scores[player.id], 1)}</td>
        <td>${formatNumber(player.credits)}</td>
        <td>${formatNumber(player.ore)}</td>
        <td>${formatNumber(player.knowledge)}</td>
        <td>${formatNumber(player.qic)}</td>
        <td class="mono">${player.power ? player.power.join(" / ") : "--"}</td>
        <td class="mono">${(player.tracks || []).map((value) => formatNumber(value)).join(" · ") || "--"}</td>
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

function setHistoryStep(step) {
  const trace = state.history.trace;
  if (!trace?.steps?.length) return;
  state.history.step = Math.max(0, Math.min(Number(step), trace.steps.length - 1));
  renderHistory();
}

function stopHistoryPlayback() {
  if (state.history.timer) window.clearInterval(state.history.timer);
  state.history.timer = null;
  state.history.playing = false;
  byId("history-play").textContent = "▶";
  byId("history-play").title = "播放";
  byId("history-play").setAttribute("aria-label", "播放");
}

function toggleHistoryPlayback() {
  if (state.history.playing) {
    stopHistoryPlayback();
    return;
  }
  const trace = state.history.trace;
  if (!trace?.steps?.length) return;
  if (state.history.step >= trace.steps.length - 1) state.history.step = 0;
  state.history.playing = true;
  byId("history-play").textContent = "Ⅱ";
  byId("history-play").title = "暂停";
  byId("history-play").setAttribute("aria-label", "暂停");
  state.history.timer = window.setInterval(() => {
    if (state.history.step >= trace.steps.length - 1) {
      stopHistoryPlayback();
      return;
    }
    state.history.step += 1;
    renderHistory();
  }, 650);
  renderHistory();
}

function selectView(name) {
  const selected = ["overview", "selfplay", "history", "diagnostics"].includes(name) ? name : "overview";
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
    renderHistory();
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
byId("history-run-select").addEventListener("change", async (event) => {
  stopHistoryPlayback();
  state.history.runId = event.target.value || null;
  const iterations = historyRun()?.iterations || [];
  state.history.iteration = iterations.at(-1)?.iteration ?? null;
  state.history.game = iterations.at(-1)?.games?.at(-1)?.game ?? null;
  state.history.trace = null;
  await loadHistoryGame(true);
});
byId("history-iteration-select").addEventListener("change", async (event) => {
  stopHistoryPlayback();
  state.history.iteration = Number(event.target.value);
  const games = historyIteration()?.games || [];
  state.history.game = games.at(-1)?.game ?? null;
  state.history.trace = null;
  await loadHistoryGame(true);
});
byId("history-game-select").addEventListener("change", async (event) => {
  stopHistoryPlayback();
  state.history.game = Number(event.target.value);
  state.history.trace = null;
  await loadHistoryGame(true);
});
byId("history-step-slider").addEventListener("input", (event) => setHistoryStep(event.target.value));
byId("history-previous").addEventListener("click", () => setHistoryStep(state.history.step - 1));
byId("history-next").addEventListener("click", () => setHistoryStep(state.history.step + 1));
byId("history-play").addEventListener("click", toggleHistoryPlayback);
byId("history-action-table").addEventListener("click", (event) => {
  const row = event.target.closest("tr[data-step]");
  if (row) {
    stopHistoryPlayback();
    setHistoryStep(row.dataset.step);
  }
});
window.addEventListener("resize", () => {
  renderLossChart();
  renderBoard(latestState());
  renderHistory();
});

selectView(window.location.hash.replace("#", "") || "overview");
pollEvents(true);
setInterval(() => pollEvents(false), POLL_INTERVAL_MS);
setInterval(() => {
  renderStatus();
  renderMetrics();
  byId("footer-clock").textContent = formatTime(new Date().toISOString());
}, 1000);
