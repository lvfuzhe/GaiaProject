"use strict";

const POLL_INTERVAL_MS = 1400;
const PLAYER_COLORS = ["#18705a", "#c14c39", "#2d68a7", "#ad751c"];
const TERRAIN_COLORS = [
  "#4d78b8", "#d7a93e", "#56a48e", "#d86b57", "#ba6f47",
  "#8b8f96", "#b7d8e7", "#7356a8", "#5bbf79", "#d3b65b"
];
const SECTOR_FILLS = ["#eef4f1", "#f4f0e8", "#eef2f7", "#f5eeee", "#eff3ea"];
const TERRAIN_LABELS = ["地球型", "沙漠", "沼泽", "火山", "氧化", "钛金", "冰冻", "跨维", "盖亚", "失落星球"];
const TRACK_LABELS = {
  terraforming: "地形改造",
  navigation: "航行",
  artificial_intelligence: "人工智能",
  gaia_project: "盖亚计划",
  economy: "经济",
  science: "科学"
};
const TRACK_KEYS = Object.keys(TRACK_LABELS);
const BOOSTER_NAMES = [
  "2 信用点；特殊行动：免费 1 步改造后建矿",
  "充能 2；特殊行动：一次建造或盖亚计划航行距离 +3",
  "1 矿石 + 1 知识",
  "1 矿石 + 2 能量片",
  "2 信用点 + 1 Q.I.C.",
  "1 矿石；过轮时每座矿场 1 VP",
  "1 矿石；过轮时每座贸易站 2 VP",
  "1 知识；过轮时每座研究所 3 VP",
  "充能 4；过轮时每座行星研究院或学院 4 VP",
  "4 信用点；过轮时每颗已殖民盖亚星球 1 VP"
];
const ADVANCED_TECH_NAMES = [
  "特殊行动：1 Q.I.C. + 5 信用点",
  "特殊行动：3 矿石",
  "特殊行动：3 知识",
  "每座矿场 2 VP",
  "每个已殖民星区 1 矿石",
  "每个已殖民星区 2 VP",
  "每颗已殖民盖亚星球 2 VP",
  "每枚联邦板块 5 VP",
  "每座贸易站 4 VP",
  "过轮时每枚联邦板块 3 VP",
  "过轮时每座研究所 3 VP",
  "过轮时每种已殖民星球类型 1 VP",
  "每次推进科研 2 VP",
  "每次建造矿场 3 VP",
  "每次建造贸易站 3 VP"
];
const FEDERATION_NAMES = [
  "6 VP + 2 知识", "7 VP + 2 矿石", "8 VP + 1 Q.I.C.",
  "8 VP + 2 能量片", "7 VP + 6 信用点", "12 VP"
];
const SETUP_LABELS = {
  "terraform-2": "每个地形改造步数 +2 VP",
  "research-2": "每次推进科研 +2 VP",
  "mine-2": "每次建造矿场 +2 VP",
  "federation-5": "每次获得联邦板块 +5 VP",
  "trading-3": "每次建造贸易站 +3 VP",
  "trading-4": "每次建造贸易站 +4 VP",
  "gaia-mine-3": "在盖亚星球建造矿场 +3 VP",
  "gaia-mine-4": "在盖亚星球建造矿场 +4 VP",
  "big-5a": "建造行星研究院或学院 +5 VP",
  "big-5b": "建造行星研究院或学院 +5 VP",
  "federation-structures": "联邦内建筑", structures: "建筑总数",
  "planet-types": "殖民星球类型", "gaia-planets": "殖民盖亚星球",
  sectors: "殖民星区", satellites: "放置卫星与空间站",
  "ore-qic": "立即获得 1 矿石和 1 Q.I.C.",
  "planet-type-knowledge": "每种已殖民星球类型获得 1 知识",
  "vp-7": "立即获得 7 VP",
  "gaia-mine-vp": "在盖亚星球建造矿场时获得 3 VP",
  "structure-power": "行星研究院和学院的建筑强度变为 4",
  "ore-power-income": "收入：1 矿石并充能 1",
  "knowledge-credit-income": "收入：1 知识和 1 信用点",
  "credits-income": "收入：4 信用点",
  "power-action": "特殊行动：充能 4"
};
const FACTION_ABILITIES = {
  Terrans: "盖亚区能量回到 II 区；行星研究院可在盖亚阶段兑换资源",
  Lantids: "可在对手已殖民星球共存；建成行星研究院后，每次放置共存矿场获得 2 知识",
  Xenos: "额外起始矿场；行星研究院使联邦门槛降为 6，并以 1 Q.I.C. 替代 1 能量片收入",
  Gleens: "建造右侧学院前，获得 Q.I.C. 时改为获得等量矿石；殖民盖亚星球以 1 矿石替代 1 Q.I.C. 并额外获得 2 VP；行星研究院立即获得专属联邦板块",
  Taklons: "脑石计作 1 个能量片，支付能量行动时可按 3 点使用；行星研究院在被动充能时额外获得 1 个能量片，可选择充能前或后获得",
  Ambas: "每轮一次，将行星研究院与自己的矿场交换；不算建造或升级，不获得 VP、能量或被动收益",
  "Hadsch Hallas": "起始经济科研并额外获得 3 信用点收入；建成行星研究院后，可用信用点按能量自由行动费率兑换矿石、知识和 Q.I.C.",
  Ivits: "最后放置行星研究院；仅扩建一个联邦，卫星消耗 Q.I.C.；每轮可放置 1 个空间站",
  Geodens: "起始位于地形改造 1；建成行星研究院后，每种此前未殖民的星球类型首次建矿获得 3 知识，研究院建成前的类型不追溯",
  "Bal T'aks": "建成行星研究院前不能推进航行轨；可将可用盖亚塑形者移入盖亚区换取 1 Q.I.C.，下一盖亚阶段归还；研究院建成后解锁航行轨；右侧学院提供 4 信用点行动",
  Firaks: "起始少 1 矿石和 1 知识、额外获得 1 知识收入；建成行星研究院后，每轮一次可将研究所降级为贸易站并免费推进一条科研轨，该动作计作升级贸易站",
  Bescods: "行星研究院与学院位置、贸易站与研究所收入互换；每轮一次可免费推进一条并列最低科研轨；建成行星研究院后，灰色母星建筑强度 +1，研究院收入为 2 能量片",
  Nevlas: "可将 III 区 1 个能量片移入盖亚区获得 1 知识；建成行星研究院后，III 区每个能量片按 2 点用于资源自由转换，兑换项可连续执行并任意搭配；公共能量工位同样按双倍计算，奇数费用向上取整；研究所收入为充能 2",
  Itars: "燃烧能量时，弃置的能量片进入盖亚区；建成行星研究院后，盖亚阶段每弃置 4 个盖亚区能量片可获得 1 块科技板块；左侧学院收入为 3 知识"
};
const BASE_FACTIONS = [
  { id: 0, board: 1, side: "A", name: "Terrans", home_terrain: 0, start_track: "gaia_project", starting_power: [4, 4, 0], starting_credits: 15, starting_ore: 4, starting_knowledge: 3, starting_qic: 1, starting_structures: 2, starts_with_pi: false, federation_threshold: 7 },
  { id: 1, board: 1, side: "B", name: "Lantids", home_terrain: 0, start_track: null, starting_power: [4, 0, 0], starting_credits: 13, starting_ore: 4, starting_knowledge: 3, starting_qic: 1, starting_structures: 2, starts_with_pi: false, federation_threshold: 7 },
  { id: 2, board: 2, side: "A", name: "Xenos", home_terrain: 1, start_track: "artificial_intelligence", starting_power: [2, 4, 0], starting_credits: 15, starting_ore: 4, starting_knowledge: 3, starting_qic: 2, starting_structures: 3, starts_with_pi: false, federation_threshold: 7 },
  { id: 3, board: 2, side: "B", name: "Gleens", home_terrain: 1, start_track: "navigation", starting_power: [2, 4, 0], starting_credits: 15, starting_ore: 5, starting_knowledge: 3, starting_qic: 0, starting_structures: 2, starts_with_pi: false, federation_threshold: 7 },
  { id: 4, board: 3, side: "A", name: "Taklons", home_terrain: 2, start_track: null, starting_power: [2, 4, 0], starting_brainstone_bowl: 1, starting_qic: 1, starting_structures: 2, starts_with_pi: false, federation_threshold: 7 },
  { id: 5, board: 3, side: "B", name: "Ambas", home_terrain: 2, start_track: "navigation", starting_power: [2, 4, 0], starting_qic: 2, starting_structures: 2, starts_with_pi: false, federation_threshold: 7 },
  { id: 6, board: 4, side: "A", name: "Hadsch Hallas", home_terrain: 4, start_track: "economy", starting_power: [2, 4, 0], starting_qic: 1, starting_structures: 2, starts_with_pi: false, federation_threshold: 7 },
  { id: 7, board: 4, side: "B", name: "Ivits", home_terrain: 4, start_track: null, starting_power: [2, 4, 0], starting_qic: 1, starting_structures: 1, starts_with_pi: true, places_last: true, federation_threshold: 7 },
  { id: 8, board: 5, side: "A", name: "Geodens", home_terrain: 3, start_track: "terraforming", starting_power: [2, 4, 0], starting_ore: 6, starting_qic: 1, starting_structures: 2, starts_with_pi: false, federation_threshold: 7 },
  { id: 9, board: 5, side: "B", name: "Bal T'aks", home_terrain: 3, start_track: "gaia_project", starting_power: [2, 2, 0], starting_credits: 15, starting_ore: 4, starting_knowledge: 3, starting_qic: 0, starting_structures: 2, starts_with_pi: false, federation_threshold: 7 },
  { id: 10, board: 6, side: "A", name: "Firaks", home_terrain: 5, start_track: null, starting_power: [2, 4, 0], starting_credits: 15, starting_ore: 3, starting_knowledge: 2, starting_qic: 1, starting_structures: 2, starts_with_pi: false, federation_threshold: 7 },
  { id: 11, board: 6, side: "B", name: "Bescods", home_terrain: 5, start_track: null, starting_power: [2, 4, 0], starting_credits: 15, starting_ore: 4, starting_knowledge: 1, starting_qic: 1, starting_structures: 2, starts_with_pi: false, federation_threshold: 7 },
  { id: 12, board: 7, side: "B", name: "Nevlas", home_terrain: 6, start_track: "science", starting_power: [2, 4, 0], starting_credits: 15, starting_ore: 4, starting_knowledge: 2, starting_qic: 1, starting_structures: 2, starts_with_pi: false, federation_threshold: 7 },
  { id: 13, board: 7, side: "A", name: "Itars", home_terrain: 6, start_track: null, starting_power: [4, 4, 0], starting_credits: 15, starting_ore: 5, starting_knowledge: 3, starting_qic: 1, starting_structures: 2, starts_with_pi: false, federation_threshold: 7 },
];
const STANDARD_TECH_KEYS = [
  "ore-qic", "planet-type-knowledge", "vp-7", "gaia-mine-vp", "structure-power",
  "ore-power-income", "knowledge-credit-income", "credits-income", "power-action",
];
const ROUND_SETUP_KEYS = [
  "terraform-2", "research-2", "mine-2", "federation-5", "trading-3",
  "trading-4", "gaia-mine-3", "gaia-mine-4", "big-5a", "big-5b",
];
const FINAL_SETUP_KEYS = [
  "federation-structures", "structures", "planet-types",
  "gaia-planets", "sectors", "satellites",
];
const DEFAULT_RANDOM_SETUPS = {
  2: {
    sector_tiles: [2, 1, 3, 6, 0, 4, 5],
    sector_rotations: [3, 0, 4, 4, 5, 1, 0],
    booster_tiles: [6, 4, 0, 2, 7],
    round_scoring_tiles: [0, 2, 7, 4, 9, 3],
    final_scoring_tiles: [0, 4],
    standard_tech_tiles: [7, 6, 4, 8, 2, 3, 5, 0, 1],
    advanced_tech_tiles: [12, 14, 10, 6, 9, 7],
    terraforming_federation_tile: 5,
  },
  3: {
    sector_tiles: [4, 6, 2, 7, 3, 5, 9, 0, 8, 1],
    sector_rotations: [3, 3, 5, 4, 3, 3, 3, 5, 1, 4],
    booster_tiles: [6, 0, 9, 4, 5, 7],
    round_scoring_tiles: [5, 4, 9, 6, 8, 3],
    final_scoring_tiles: [4, 0],
    standard_tech_tiles: [4, 6, 2, 3, 8, 7, 5, 0, 1],
    advanced_tech_tiles: [13, 9, 3, 4, 8, 5],
    terraforming_federation_tile: 4,
  },
  4: {
    sector_tiles: [4, 6, 2, 7, 3, 5, 9, 0, 8, 1],
    sector_rotations: [3, 3, 5, 4, 3, 3, 3, 5, 1, 4],
    booster_tiles: [2, 7, 5, 3, 1, 4, 9],
    round_scoring_tiles: [6, 4, 5, 7, 1, 3],
    final_scoring_tiles: [2, 1],
    standard_tech_tiles: [8, 5, 1, 0, 4, 7, 2, 6, 3],
    advanced_tech_tiles: [7, 2, 8, 5, 3, 13],
    terraforming_federation_tile: 1,
  },
};
const DEFAULT_REDUCED_3P_MAP = {
  sector_tiles: [2, 4, 3, 6, 5, 0, 1, 7],
  sector_rotations: [1, 4, 3, 5, 3, 3, 5, 4],
};
const BUILDING_SPECS = [
  { key: "mine", label: "矿场", short: "M", total: 8 },
  { key: "trading_station", label: "贸易站", short: "TS", total: 4 },
  { key: "research_lab", label: "研究所", short: "RL", total: 3 },
  { key: "planetary_institute", label: "行星研究院", short: "PI", total: 1 },
  { key: "academy", label: "学院", short: "AC", total: 2 },
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
const mapPieceArtworkCache = new Map();
let sectorArtworkRenderQueued = false;
const MAP_PIECE_PATHS = {
  sectorBackground: "/assets/map-pieces/blankHex.png",
  structures: "/assets/map-pieces/structures.png",
  planets: "/assets/map-pieces/planets.png",
  icons: "/assets/map-pieces/icons.png",
};
const BGA_PLANET_COLUMNS = [1, 4, 5, 3, 2, 6, 7, 9, 8, 10];
const BGA_STRUCTURE_CROPS = {
  mine: [750, 0, 69, 77],
  trading_station: [450, 0, 102, 120],
  research_lab: [600, 0, 108, 117],
  planetary_institute: [0, 0, 218, 198],
  academy: [250, 0, 186, 197],
};
const BGA_STRUCTURE_ROWS = [1, 3, 0, 2];
const BGA_STRUCTURE_HEIGHTS = {
  mine: 1.08,
  trading_station: 1.28,
  research_lab: 1.22,
  planetary_institute: 1.35,
  academy: 1.30,
};

const state = {
  events: [],
  lastSequence: 0,
  runId: null,
  source: "--",
  connected: false,
  live: true,
  polling: false,
  manualSetup: {
    initialized: false,
    hydrated: false,
    edited: false,
    configView: "map",
    mapMode: window.location.pathname === "/setup/manual" ? "manual" : "bga-random",
    randomElements: null,
    preview: null,
    selectedPlanetId: null,
    planetEditorError: null,
    planetEditorMode: "move",
    planetEditorTerrain: 0,
    busy: false,
    message: "就绪",
  },
  history: {
    index: null,
    runId: null,
    iteration: null,
    game: null,
    trace: null,
    step: 0,
    loading: false,
    deleting: false,
    indexRequestId: 0,
    message: "",
    playing: false,
    timer: null,
    mapView: {
      zoom: 1,
      gap: 0,
      background: "#171d23",
    },
  },
  bgaImport: {
    busy: false,
    status: "ready",
    message: "等待输入复盘地址",
    result: null,
    session: {
      saved: false,
      username: "",
      cookie_count: 0,
      updated_at: null,
    },
  },
  play: {
    workspace: "setup",
    session: null,
    players: 2,
    factions: [0, 2],
    roles: ["human", "ai"],
    selectedPlanetId: null,
    requestBusy: false,
    requestEpoch: 0,
    polling: false,
    autoAi: true,
    aiTimer: null,
    message: "就绪",
    messageStatus: "ready"
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
      if (event.type === "run_started") {
        state.runId = event.run_id;
      }
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
  renderSetup(state.manualSetup.preview);
  renderPlay();
  renderSelfPlay();
  renderHistory();
  renderBgaImport();
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

function sectorArtworkPath(tile, side) {
  const number = String(tile).padStart(2, "0");
  return `/assets/sectors/sector-${number}-${side}.gif`;
}

function queueSectorArtworkRender() {
  if (sectorArtworkRenderQueued) return;
  sectorArtworkRenderQueued = true;
  requestAnimationFrame(() => {
    sectorArtworkRenderQueued = false;
    renderSetup(state.manualSetup.preview);
    renderPlanetPositionEditor();
    renderBoard(latestState());
    renderPlay();
    renderHistory();
  });
}

function getMapPieceArtwork(kind) {
  const path = MAP_PIECE_PATHS[kind];
  if (!path) return null;
  if (!mapPieceArtworkCache.has(kind)) {
    const image = new Image();
    const entry = { image, loaded: false, failed: false };
    mapPieceArtworkCache.set(kind, entry);
    image.addEventListener("load", () => {
      entry.loaded = true;
      queueSectorArtworkRender();
    }, { once: true });
    image.addEventListener("error", () => {
      entry.failed = true;
      queueSectorArtworkRender();
    }, { once: true });
    image.src = path;
  }
  const entry = mapPieceArtworkCache.get(kind);
  return entry.loaded && !entry.failed ? entry.image : null;
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

function setupLabel(tile) {
  return SETUP_LABELS[tile?.key] || tile?.label || "--";
}

function tileAsset(kind, id) {
  const federationAssetOrder = [6, 4, 2, 3, 5, 1];
  const number = Number(id) + 1;
  const assetNumber = kind === "federation"
    ? federationAssetOrder[Number(id)]
    : number;
  if (!Number.isInteger(assetNumber) || assetNumber < 1) return "";
  const assets = {
    standard: ["tech-standard", "jpg"],
    advanced: ["tech-advanced", "jpg"],
    round: ["round-scoring", "gif"],
    final: ["final-scoring", "jpg"],
    booster: ["booster", "jpg"],
    federation: ["federation", "png"],
  };
  const [prefix, extension] = assets[kind] || [];
  return prefix ? `/assets/tiles/${prefix}-${String(assetNumber).padStart(2, "0")}.${extension}` : "";
}

function factionPlayerBoardAsset(id) {
  return `/assets/factions/player-board-${String(Number(id) + 1).padStart(2, "0")}.jpg?source=bga-260630-1810`;
}

function finalMetric(snapshot, key, playerId) {
  const player = snapshot.players?.[playerId] || {};
  const planets = (snapshot.planets || []).filter((planet) => planet.owner === playerId);
  const coexisting = (snapshot.planets || []).filter(
    (planet) => planet.coexisting_mine_owner === playerId,
  );
  if (key === "federation-structures") {
    return planets.filter((planet) => planet.federated).length
      + coexisting.filter((planet) => planet.coexisting_mine_federated).length;
  }
  if (key === "structures") return planets.length + coexisting.length;
  if (key === "planet-types") {
    const encodedTypes = player.colonized_types;
    if (encodedTypes !== null && encodedTypes !== undefined && Number.isFinite(Number(encodedTypes))) {
      return Number(encodedTypes).toString(2).replaceAll("0", "").length;
    }
    return new Set(planets.map((planet) => planet.terrain).filter((terrain) => terrain < 7)).size;
  }
  if (key === "gaia-planets") return planets.filter((planet) => planet.terrain === 8).length;
  if (key === "sectors") {
    return new Set([...planets, ...coexisting].map((planet) => planet.sector)).size;
  }
  if (key === "satellites") {
    return Number(player.satellites_and_space_stations ?? player.satellites ?? 0);
  }
  return 0;
}

function finalRanking(snapshot, tile) {
  const values = (snapshot.players || []).map((player) => ({
    player,
    value: finalMetric(snapshot, tile.key, player.id),
  }));
  return values
    .sort((left, right) => right.value - left.value || left.player.id - right.player.id)
    .map((entry, _index, sorted) => ({
      ...entry,
      rank: 1 + sorted.filter((candidate) => candidate.value > entry.value).length,
    }))
    .map((entry) => ({
      ...entry,
      points: [18, 12, 6, 0][Math.min(entry.rank - 1, 3)],
    }));
}

function renderHistoryConfiguredComponents(snapshot) {
  const setup = snapshot?.setup;
  const status = byId("history-components-status");
  const roundStatus = byId("history-round-status");
  const roundCurrent = byId("history-round-current");
  const roundScoreNote = byId("history-round-score-note");
  const roundTarget = byId("history-round-scoring");
  const finalTarget = byId("history-final-scoring");
  const boosterTarget = byId("history-boosters");
  const federationTarget = byId("history-federation-supply");
  if (!status || !roundTarget || !finalTarget || !boosterTarget || !federationTarget) return;

  if (!setup) {
    status.textContent = "当前快照没有初始配置";
    if (roundStatus) roundStatus.textContent = "等待历史快照";
    if (roundCurrent) roundCurrent.textContent = "第 -- 轮";
    if (roundScoreNote) roundScoreNote.textContent = "等待回合计分板块";
    roundTarget.innerHTML = "";
    finalTarget.innerHTML = "";
    boosterTarget.innerHTML = "";
    federationTarget.innerHTML = "";
    return;
  }

  const currentRound = Number(snapshot.round || 0);
  const terminal = Boolean(snapshot.terminal);
  const players = Array.isArray(snapshot.players) ? snapshot.players : [];
  const roundScoring = Array.isArray(setup.round_scoring) ? setup.round_scoring : [];
  const roundCount = roundScoring.length || 6;
  const roundStateLabel = terminal
    ? "对局结束"
    : currentRound > 0
      ? `第 ${Math.min(currentRound, roundCount)} 轮进行中`
      : "开局阶段";
  if (roundStatus) roundStatus.textContent = roundStateLabel;
  if (roundCurrent) roundCurrent.textContent = terminal
    ? "终局"
    : currentRound > 0
      ? `第 ${Math.min(currentRound, roundCount)} / ${roundCount} 轮`
      : "开局阶段";
  const activeTile = roundScoring.find((tile) => Number(tile.round) === currentRound);
  if (roundScoreNote) roundScoreNote.textContent = activeTile
    ? `${setupLabel(activeTile)} · ${formatNumber(activeTile.points)} VP`
    : roundScoring.length
      ? `${roundScoring.length} 个回合计分板块 · 当前板块已高亮`
      : "未记录回合计分板块";
  roundTarget.innerHTML = roundScoring.map((tile) => {
    const round = Number(tile.round);
    const current = !terminal && currentRound === round;
    const past = terminal || currentRound > round;
    const stateLabel = current ? "进行中" : (past ? "已完成" : "待开始");
    const name = setupLabel(tile);
    return `<article class="history-round-tile ${current ? "current" : ""} ${past ? "past" : ""}" title="第 ${round} 轮 · ${escapeHtml(name)} · ${stateLabel}">
      <span class="history-component-art"><img src="${tileAsset("round", tile.id)}" alt="${escapeHtml(name)}"></span>
      <span class="history-component-meta"><strong>R${round}</strong><small>${stateLabel}</small></span>
    </article>`;
  }).join("");

  const finalScoring = Array.isArray(setup.final_scoring) ? setup.final_scoring : [];
  const finalTrackColumns = Math.max(3, Math.min(4, players.length || 3));
  finalTarget.innerHTML = finalScoring.map((tile, index) => {
    const name = setupLabel(tile);
    const ranking = snapshot?.players?.length ? finalRanking(snapshot, tile) : [];
    const rankingLabel = ranking.map((entry) => `P${entry.player.id} ${entry.value}`).join(" · ");
    const trackRanks = Array.from({ length: Math.max(3, players.length) }, (_, rankIndex) => {
      const rank = rankIndex + 1;
      const tokens = ranking
        .filter((entry) => entry.rank === rank)
        .map((entry) => `<span class="history-score-token p${entry.player.id}" title="P${entry.player.id} · ${formatNumber(entry.value)}">P${entry.player.id}</span>`)
        .join("");
      return `<div class="history-final-rank"><small>#${rank}</small><span>${tokens || "·"}</span></div>`;
    }).join("");
    return `<article class="history-final-tile" title="${escapeHtml(name)}${rankingLabel ? ` · ${escapeHtml(rankingLabel)}` : ""}">
      <span class="history-component-art"><img src="${tileAsset("final", tile.id)}" alt="${escapeHtml(name)}"></span>
      <span class="history-component-meta"><strong>终局 ${index + 1}</strong><small>${escapeHtml(name)}</small></span>
      <span class="history-final-track" style="--history-final-columns:${finalTrackColumns}" aria-label="${escapeHtml(name)}终局计分轨">${trackRanks}</span>
    </article>`;
  }).join("");

  const boosters = Array.isArray(setup.boosters) ? setup.boosters : [];
  boosterTarget.innerHTML = boosters.map((booster) => {
    const currentOwner = players.find((player) => Number(player.booster) === Number(booster.id));
    const ownerId = currentOwner?.id ?? Number(booster.owner ?? -1);
    const owner = ownerId >= 0 ? players.find((player) => Number(player.id) === ownerId) : null;
    const ownerLabel = ownerId >= 0
      ? `P${ownerId}${owner?.faction ? ` · ${owner.faction}` : ""}`
      : "公共池";
    const name = BOOSTER_NAMES[booster.id] || booster.label || `助推 ${Number(booster.id) + 1}`;
    return `<article class="history-booster-tile ${ownerId >= 0 ? "assigned" : "available"}" title="${escapeHtml(name)} · ${escapeHtml(ownerLabel)}">
      <span class="history-component-art"><img src="${tileAsset("booster", booster.id)}" alt="${escapeHtml(name)}"></span>
      <span class="history-component-meta"><strong>B${Number(booster.id) + 1}</strong><small>${escapeHtml(ownerLabel)}</small></span>
    </article>`;
  }).join("");

  const supply = Array.isArray(setup.federation_supply) ? setup.federation_supply : [];
  federationTarget.innerHTML = Array.from({ length: 6 }, (_, id) => {
    const count = Math.max(0, Number(supply[id] || 0));
    const name = FEDERATION_NAMES[id];
    return `<article class="history-federation-tile ${count === 0 ? "depleted" : ""}" title="${escapeHtml(name)} · 剩余 ${count}">
      <span class="history-component-art"><img src="${tileAsset("federation", id)}" alt="${escapeHtml(name)}"><b>${count}</b></span>
      <span class="history-component-meta"><strong>F${id + 1}</strong><small>剩余 ${count}</small></span>
    </article>`;
  }).join("");

  const configured = roundScoring.length + finalScoring.length + boosters.length;
  status.textContent = `${terminal ? "对局结束" : (currentRound > 0 ? `第 ${currentRound} 轮` : "开局阶段")} · ${configured} 块已配置`;
}

function renderSetup(snapshot) {
  const content = byId("setup-content");
  const empty = byId("setup-empty");
  if (!content || !empty) return;
  initializeSetupEditor(snapshot);
  const setup = snapshot?.setup;
  const available = Boolean(setup?.map?.sectors?.length);
  empty.hidden = true;
  content.hidden = !available;
  if (!available) return;

  const map = setup.map;
  byId("setup-seed").textContent = setup.seed;
  const reducedMap = map.size === "reduced" || (!map.size && map.sector_count < 10);
  const mapSizeLabel = reducedMap ? "小地图" : "标准地图";
  byId("setup-map-summary").textContent = `${mapSizeLabel} · ${snapshot.planets.length} 个星球 · ${map.sector_count} 个来源星区`;
  const initialFirstPlayer = snapshot.first_player;
  byId("setup-first-player").textContent = `P${initialFirstPlayer}`;
  byId("setup-ruleset").textContent = snapshot.ruleset || "--";
  byId("setup-map-method").textContent = map.method === "manual"
    ? "手动星区 · 原图裁切 · 合法性校验"
    : "BGA 随机 · 原图裁切 · 同色母星不相邻";
  byId("setup-sector-count").textContent = `${map.sector_count} 块`;
  drawStarMapBoard(byId("setup-map-canvas"), snapshot, false);
  byId("setup-sector-table").innerHTML = map.sectors.map((sector) => `<tr>
    <td>${sector.position + 1}</td>
    <td class="mono">S${String(sector.tile).padStart(2, "0")}</td>
    <td>${sector.rotation}°</td>
    <td class="mono">${sector.q}, ${sector.r}</td>
  </tr>`).join("");

  const factionCatalog = setup.faction_catalog?.length === 14 ? setup.faction_catalog : BASE_FACTIONS;
  const catalogById = new Map(factionCatalog.map((faction) => [faction.id, faction]));
  const factionCard = (faction, assignment = null) => {
    const startingPower = faction.starting_power || [];
    const startingPowerLabel = startingPower.map((value, index) => (
      faction.starting_brainstone_bowl === index + 1
        ? `${value} + 脑石`
        : value
    )).join(" / ");
    const startBuilding = faction.starts_with_pi
      ? (faction.places_last ? "最后放置行星研究院" : "起始行星研究院")
      : `起始 ${faction.starting_structures ?? 2} 座矿场`;
    const seat = assignment
      ? `P${assignment.player}${assignment.player === snapshot.first_player ? " · 首位" : ""}`
      : `个人版图 ${faction.board}${faction.side}`;
    return `<article class="faction-setup-item ${assignment ? "assigned" : ""}">
      <img class="faction-board-art" src="${factionPlayerBoardAsset(faction.id)}" alt="${escapeHtml(faction.name)}个人主板">
      <div class="faction-card-body">
        <div class="faction-setup-head">
          <div><span class="faction-seat">${seat}</span><strong>${escapeHtml(faction.name)}</strong></div>
          <i class="home-swatch terrain-${faction.home_terrain}" aria-label="${TERRAIN_LABELS[faction.home_terrain]}"></i>
        </div>
        <div class="faction-setup-meta">
          <span class="setup-tag">${TERRAIN_LABELS[faction.home_terrain]}</span>
          ${faction.start_track
            ? `<span class="setup-tag">${TRACK_LABELS[faction.start_track] || faction.start_track} +1</span>`
            : '<span class="setup-tag">无初始科研</span>'}
          <span class="setup-tag">${startBuilding}</span>
          <span class="setup-tag">信用点 ${faction.starting_credits ?? 15}</span>
          <span class="setup-tag">矿石 ${faction.starting_ore ?? 4}</span>
          <span class="setup-tag">知识 ${faction.starting_knowledge ?? 3}</span>
          ${startingPower.length ? `<span class="setup-tag">能量 ${startingPowerLabel}</span>` : ""}
          <span class="setup-tag">Q.I.C. ${faction.starting_qic ?? 1}</span>
          ${faction.federation_threshold === 6 ? `<span class="setup-tag">联邦强度 6</span>` : ""}
        </div>
        <small>${escapeHtml(FACTION_ABILITIES[faction.name] || faction.ability || "")}</small>
      </div>
    </article>`;
  };
  const assignedFactions = setup.factions.map((assignment) => ({
    ...(catalogById.get(assignment.id) || {}),
    ...assignment,
  }));
  byId("setup-faction-count").textContent = `${assignedFactions.length} 个座位`;
  byId("setup-faction-grid").innerHTML = assignedFactions
    .map((faction) => factionCard(faction, faction))
    .join("");
  byId("setup-faction-catalog-count").textContent = `${factionCatalog.length} / 14`;
  byId("setup-faction-catalog").innerHTML = [...factionCatalog]
    .sort((left, right) => left.board - right.board || left.side.localeCompare(right.side))
    .map((faction) => factionCard(faction))
    .join("");

  byId("setup-round-track").innerHTML = setup.round_scoring.map((tile) => {
    const current = snapshot.round === tile.round && !snapshot.terminal;
    const past = snapshot.terminal || snapshot.round > tile.round;
    const image = tileAsset("round", tile.id, tile.key);
    return `<article class="round-score-tile ${current ? "current" : ""} ${past ? "past" : ""}">
      <img class="setup-tile-art round-tile-art" src="${image}" alt="${escapeHtml(setupLabel(tile))}">
      <div class="round-score-head"><span class="round-index">${String(tile.round).padStart(2, "0")}</span><span>${current ? "进行中" : (past ? "已完成" : "待开始")}</span></div>
      <strong>${escapeHtml(setupLabel(tile))}</strong>
      <small>+${tile.points} VP</small>
    </article>`;
  }).join("");
  byId("setup-final-grid").innerHTML = setup.final_scoring.map((tile, index) => {
    const ranking = finalRanking(snapshot, tile);
    const image = tileAsset("final", tile.id, tile.key);
    return `<article class="final-score-tile">
      <img class="setup-tile-art final-tile-art" src="${image}" alt="${escapeHtml(setupLabel(tile))}">
      <div class="final-score-heading"><span>终局目标 ${index + 1}</span><strong>${escapeHtml(setupLabel(tile))}</strong></div>
      <div class="final-score-scale"><span>第 1 名 <b>18</b></span><span>第 2 名 <b>12</b></span><span>第 3 名 <b>6</b></span></div>
      <div class="final-ranking-list">${ranking.map((entry) => `<div class="final-ranking-row">
        <span class="player-mini p${entry.player.id}">P${entry.player.id}</span><strong>${formatNumber(entry.value)}</strong><span class="final-rank">#${entry.rank} · ${entry.points} VP</span>
      </div>`).join("")}</div>
    </article>`;
  }).join("");

  const standardByTrack = new Map(setup.standard_tech.filter((tile) => tile.track).map((tile) => [tile.track, tile]));
  const advancedByTrack = new Map(setup.advanced_tech.map((tile) => [tile.track, tile]));
  const freeStandardTech = setup.standard_tech.filter((tile) => !tile.track);
  const researchPlayers = snapshot.players || [];
  const trackTechSlots = Object.entries(TRACK_LABELS).flatMap(([track, label], trackIndex) => {
    const standard = standardByTrack.get(track);
    const advanced = advancedByTrack.get(track);
    const standardName = setupLabel(standard);
    const advancedName = ADVANCED_TECH_NAMES[advanced?.id] || advanced?.label || "--";
    return [
      `<span class="research-tech-slot advanced track-${trackIndex}" role="img" tabindex="0" aria-label="${escapeHtml(label)}高级科技：${escapeHtml(advancedName)}" title="${escapeHtml(label)}高级科技 · ${escapeHtml(advancedName)}">
        <img src="${tileAsset("advanced", advanced?.id, advanced?.key)}" alt="" aria-hidden="true">
      </span>`,
      `<span class="research-tech-slot standard track-${trackIndex}" role="img" tabindex="0" aria-label="${escapeHtml(label)}基础科技：${escapeHtml(standardName)}" title="${escapeHtml(label)}基础科技 · ${escapeHtml(standardName)}">
        <img src="${tileAsset("standard", standard?.id, standard?.key)}" alt="" aria-hidden="true">
      </span>`,
    ];
  });
  const freeTechSlots = freeStandardTech.map((tile, index) => {
    const name = setupLabel(tile);
    return `<span class="research-tech-slot standard free-${index}" role="img" tabindex="0" aria-label="通用基础科技 ${index + 1}：${escapeHtml(name)}" title="通用基础科技 ${index + 1} · ${escapeHtml(name)}">
      <img src="${tileAsset("standard", tile.id, tile.key)}" alt="" aria-hidden="true">
    </span>`;
  });
  const researchTech = byId("setup-research-tech");
  const techSignature = [
    ...setup.standard_tech.map((tile) => `s${tile.space}:${tile.id}`),
    ...setup.advanced_tech.map((tile, index) => `a${index}:${tile.id}`),
  ].join("|");
  if (researchTech.dataset.signature !== techSignature) {
    researchTech.innerHTML = [...trackTechSlots, ...freeTechSlots].join("");
    researchTech.dataset.signature = techSignature;
  }
  byId("setup-research-markers").innerHTML = Object.entries(TRACK_LABELS).flatMap(([track, label], trackIndex) =>
    Array.from({ length: 6 }, (_, level) => {
      const players = researchPlayers.filter((player) => Number(player.tracks?.[trackIndex] || 0) === level);
      if (!players.length) return "";
      const names = players.map((player) => `P${player.id} ${player.faction || ""}`).join("、");
      return `<div class="research-board-position track-${trackIndex} level-${level}" aria-label="${escapeHtml(label)}等级 ${level}：${escapeHtml(names)}">
        ${players.map((player) => `<span class="research-marker p${player.id}" title="P${player.id} ${escapeHtml(player.faction || "")}">P${player.id}</span>`).join("")}
      </div>`;
    })
  ).join("");
  byId("setup-research-grid").innerHTML = Object.entries(TRACK_LABELS).map(([track, label], trackIndex) => {
    const standard = standardByTrack.get(track);
    const advanced = advancedByTrack.get(track);
    const standardImage = tileAsset("standard", standard?.id, standard?.key);
    const advancedImage = tileAsset("advanced", advanced?.id, advanced?.key);
    const highestLevel = Math.max(0, ...researchPlayers.map((player) => Number(player.tracks?.[trackIndex] || 0)));
    return `<article class="research-setup-column">
      <div class="research-track-heading"><span class="track-index">${String(trackIndex + 1).padStart(2, "0")}</span><strong>${label}</strong><span class="research-track-level">最高 L${highestLevel}</span></div>
      <div class="tech-track-tile advanced"><img class="setup-tile-art tech-tile-art" src="${advancedImage}" alt="${escapeHtml(advanced?.label || "Advanced technology")}"><div><span>高级科技 #${Number(advanced?.id) + 1}</span><strong>${escapeHtml(ADVANCED_TECH_NAMES[advanced?.id] || advanced?.label || "--")}</strong></div></div>
      <div class="tech-track-tile standard"><img class="setup-tile-art tech-tile-art" src="${standardImage}" alt="${escapeHtml(setupLabel(standard))}"><div><span>基础科技 #${Number(standard?.id) + 1}</span><strong>${escapeHtml(setupLabel(standard))}</strong></div></div>
    </article>`;
  }).join("");
  byId("setup-research-player-legend").innerHTML = (snapshot.players || []).map((player) => `<span><i class="player-mini p${player.id}"></i>P${player.id} ${escapeHtml(player.faction || "")}</span>`).join("");
  byId("setup-free-tech").innerHTML = freeStandardTech.map((tile) => `<article class="tech-setup-tile standard">
    <img class="setup-tile-art tech-tile-art" src="${tileAsset("standard", tile.id, tile.key)}" alt="${escapeHtml(setupLabel(tile))}">
    <span>通用槽位 ${tile.space - 5}</span><strong>${escapeHtml(setupLabel(tile))}</strong>
  </article>`).join("");
  const federation = setup.terraforming_federation;
  byId("setup-federation-tile").textContent = `改造轨顶 · ${FEDERATION_NAMES[federation.id] || federation.label}`;

  byId("setup-booster-count").textContent = `${setup.boosters.length} 块`;
  byId("setup-booster-grid").innerHTML = setup.boosters.map((booster) => `<article class="booster-setup-tile">
    <img class="setup-tile-art booster-tile-art" src="${tileAsset("booster", booster.id)}" alt="${escapeHtml(BOOSTER_NAMES[booster.id] || booster.label)}">
    <span>助推 ${booster.id + 1} · 可选</span>
    <strong>${escapeHtml(BOOSTER_NAMES[booster.id] || booster.label)}</strong>
  </article>`).join("");
}

function initializeSetupEditor(snapshot) {
  if (state.manualSetup.initialized && (
    !snapshot || state.manualSetup.hydrated || state.manualSetup.edited
  )) return;
  const setupSnapshot = snapshot;
  const runConfig = {};
  const players = Math.min(4, Math.max(2, Number(
    runConfig.players || setupSnapshot?.players?.length || 2,
  )));
  byId("setup-editor-players").value = String(players);
  byId("setup-editor-seed").value = String(Number(
    runConfig.seed ?? setupSnapshot?.setup?.seed ?? 0,
  ));
  if (Number.isInteger(Number(runConfig.simulations))) {
    byId("setup-editor-simulations").value = String(Number(runConfig.simulations));
  }
  const assigned = Array.isArray(runConfig.factions)
    ? runConfig.factions.map(Number)
    : (setupSnapshot?.setup?.factions || []).map((faction) => Number(faction.id));
  state.manualSetup.factions = Array.from(
    { length: players },
    (_, player) => assigned[player] ?? [0, 2, 4, 6][player],
  );
  renderSetupEditorSeats(Number(
    runConfig.first_player ?? setupSnapshot?.first_player ?? 0,
  ));
  state.manualSetup.randomElements = randomElementsFromSnapshot(
    setupSnapshot,
    runConfig,
    players,
    Number(runConfig.first_player ?? setupSnapshot?.first_player ?? 0),
  );
  if (window.location.pathname === "/setup/manual") {
    state.manualSetup.randomElements.map_mode = "manual";
    state.manualSetup.mapMode = "manual";
  }
  renderRandomElementEditor();
  state.manualSetup.initialized = true;
  state.manualSetup.hydrated = Boolean(snapshot);
  renderManualSetupStatus();
}

function renderSetupEditorSeats(preferredFirstPlayer = null) {
  const players = Number(byId("setup-editor-players").value || 2);
  const previous = state.manualSetup.factions || [];
  const defaults = [0, 2, 4, 6];
  state.manualSetup.factions = Array.from(
    { length: players },
    (_, player) => previous[player] ?? defaults[player],
  );
  byId("setup-editor-factions").innerHTML = state.manualSetup.factions.map((selected, player) => `<label class="setup-editor-faction">
    <span>P${player}</span>
    <select data-player="${player}" aria-label="P${player} 种族">
      ${BASE_FACTIONS
        .slice()
        .sort((left, right) => left.board - right.board || left.side.localeCompare(right.side))
        .map((faction) => `<option value="${faction.id}" ${faction.id === selected ? "selected" : ""}>版图 ${faction.board}${faction.side} · ${escapeHtml(faction.name)}</option>`)
        .join("")}
    </select>
  </label>`).join("");
  const firstSelect = byId("setup-editor-first-player");
  const currentFirst = preferredFirstPlayer === null
    ? Math.min(players - 1, Number(firstSelect.value || 0))
    : Math.min(players - 1, Number(preferredFirstPlayer));
  firstSelect.innerHTML = Array.from(
    { length: players },
    (_, player) => `<option value="${player}" ${player === currentFirst ? "selected" : ""}>P${player}</option>`,
  ).join("");
}

function normalizedMapSize(players, requested) {
  if (players === 2) return "reduced";
  if (players === 4) return "normal";
  return requested === "reduced" ? "reduced" : "normal";
}

function setupSectorCount(players, mapSize) {
  if (players === 2) return 7;
  return players === 3 && mapSize === "reduced" ? 8 : 10;
}

function defaultRandomElements(players, mapSize = null) {
  const source = DEFAULT_RANDOM_SETUPS[players] || DEFAULT_RANDOM_SETUPS[2];
  const selectedMapSize = normalizedMapSize(players, mapSize);
  const defaults = {
    map_mode: state.manualSetup.mapMode,
    map_size: selectedMapSize,
    ...Object.fromEntries(Object.entries(source).map(([key, value]) => [
    key,
    Array.isArray(value) ? [...value] : value,
    ])),
  };
  if (players === 3 && selectedMapSize === "reduced") {
    defaults.sector_tiles = [...DEFAULT_REDUCED_3P_MAP.sector_tiles];
    defaults.sector_rotations = [...DEFAULT_REDUCED_3P_MAP.sector_rotations];
  }
  return defaults;
}

function normalizedRandomElements(players, values = {}) {
  const inferredMapSize = values.map_size
    ?? (players === 3 && values.sector_tiles?.length === 8 ? "reduced" : null);
  const mapSize = normalizedMapSize(players, inferredMapSize);
  const defaults = defaultRandomElements(players, mapSize);
  const sectorCount = setupSectorCount(players, mapSize);
  const expectedLengths = {
    sector_tiles: sectorCount,
    sector_rotations: sectorCount,
    booster_tiles: players + 3,
    round_scoring_tiles: 6,
    final_scoring_tiles: 2,
    standard_tech_tiles: 9,
    advanced_tech_tiles: 6,
  };
  for (const [field, expected] of Object.entries(expectedLengths)) {
    if (Array.isArray(values[field]) && values[field].length === expected) {
      defaults[field] = values[field].map(Number);
    }
  }
  const layout = Array.isArray(values.planet_layout)
    ? values.planet_layout
    : Array.isArray(values.planet_positions)
      ? values.planet_positions.map((position) => ({ ...position, source_id: position.id }))
      : null;
  if (layout) {
    defaults.planet_layout = layout.map((item) => ({
      id: Number(item.id),
      q: Number(item.q),
      r: Number(item.r),
      source_id: Number(item.source_id),
    }));
  }
  const federation = Number(values.terraforming_federation_tile);
  if (Number.isInteger(federation) && federation >= 0 && federation < 6) {
    defaults.terraforming_federation_tile = federation;
  }
  defaults.map_mode = values.map_mode === "manual" ? "manual" : "bga-random";
  defaults.map_size = mapSize;
  state.manualSetup.mapMode = defaults.map_mode;
  return defaults;
}

function randomElementsFromSnapshot(snapshot, config, players, _firstPlayer) {
  if (config?.random_setup) {
    const configured = { ...config.random_setup };
    if (
      configured.map_mode === "manual"
      && !Array.isArray(configured.planet_layout)
      && !Array.isArray(configured.planet_positions)
      && snapshot?.planets?.length
    ) {
      configured.planet_layout = snapshot.planets.map((planet) => ({
        id: Number(planet.id),
        q: Number(planet.q),
        r: Number(planet.r),
        source_id: Number(planet.source_id ?? planet.id),
      }));
    }
    return normalizedRandomElements(players, configured);
  }
  const setup = snapshot?.setup;
  if (!setup) return defaultRandomElements(players);
  return normalizedRandomElements(players, {
    map_mode: setup.map?.method,
    map_size: setup.map?.size,
    sector_tiles: setup.map?.sectors?.map((sector) => Number(sector.tile) - 1),
    sector_rotations: setup.map?.sectors?.map((sector) => Number(sector.rotation) / 60),
    planet_layout: snapshot.planets?.map((planet) => ({
      id: Number(planet.id),
      q: Number(planet.q),
      r: Number(planet.r),
      source_id: Number(planet.source_id ?? planet.id),
    })),
    booster_tiles: setup.boosters.map((booster) => booster.id),
    round_scoring_tiles: setup.round_scoring?.map((tile) => tile.id),
    final_scoring_tiles: setup.final_scoring?.map((tile) => tile.id),
    standard_tech_tiles: setup.standard_tech?.map((tile) => tile.id),
    advanced_tech_tiles: setup.advanced_tech?.map((tile) => tile.id),
    terraforming_federation_tile: setup.terraforming_federation?.id,
  });
}

function selectOptions(count, selected, labeler) {
  return Array.from({ length: count }, (_, value) =>
    `<option value="${value}" ${value === selected ? "selected" : ""}>${escapeHtml(labeler(value))}</option>`,
  ).join("");
}

function tileControl(field, index, selected, kind, slotLabel, count, labeler) {
  return `<label class="setup-tile-control" data-kind="${kind}">
    <span>${escapeHtml(slotLabel)}</span>
    <select data-random-field="${field}" data-random-index="${index}" aria-label="${escapeHtml(slotLabel)}">
      ${selectOptions(count, selected, labeler)}
    </select>
  </label>`;
}

function renderRandomElementEditor() {
  const players = Number(byId("setup-editor-players").value || 2);
  const elements = normalizedRandomElements(players, state.manualSetup.randomElements || {});
  state.manualSetup.randomElements = elements;
  state.manualSetup.mapMode = elements.map_mode;
  const mapSizeSelect = byId("setup-editor-map-size");
  mapSizeSelect.querySelector('option[value="normal"]').textContent = players === 4
    ? "标准地图 · 10 星区（固定）"
    : "标准地图 · 10 星区";
  mapSizeSelect.querySelector('option[value="reduced"]').textContent = players === 2
    ? "小地图 · 7 星区（固定）"
    : "小地图 · 8 星区（BGA 推荐）";
  mapSizeSelect.value = elements.map_size;
  mapSizeSelect.disabled = players !== 3;
  mapSizeSelect.title = players === 2
    ? "两人局固定使用 7 星区小地图"
    : players === 4
      ? "四人局固定使用 10 星区标准地图"
      : "BGA 建议三人局使用 8 星区小地图";
  document.querySelectorAll("[data-map-mode]").forEach((button) => {
    const active = button.dataset.mapMode === elements.map_mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  byId("setup-manual-sector-editor").hidden = elements.map_mode !== "manual";
  byId("setup-config-map").classList.toggle("compact", elements.map_mode !== "manual");
  const sectorCount = setupSectorCount(players, elements.map_size);
  byId("setup-editor-sectors").innerHTML = Array.from({ length: sectorCount }, (_, index) => {
    const selected = elements.sector_tiles[index];
    const side = players === 2 && selected >= 4 && selected <= 6 ? "outlined" : "solid";
    return `<div class="setup-sector-control">
      <img src="${sectorArtworkPath(selected + 1, side)}" alt="星区 S${String(selected + 1).padStart(2, "0")}">
      <label><span>位置 ${index + 1} · 板块</span><select data-random-field="sector_tiles" data-random-index="${index}">
        ${selectOptions(sectorCount, selected, (value) => `S${String(value + 1).padStart(2, "0")}`)}
      </select></label>
      <label><span>旋转角度</span><select data-random-field="sector_rotations" data-random-index="${index}">
        ${selectOptions(6, elements.sector_rotations[index], (value) => `${value * 60}°`)}
      </select></label>
    </div>`;
  }).join("");
  renderPlanetPositionEditor();
  byId("setup-editor-round-scoring").innerHTML = elements.round_scoring_tiles.map((tile, index) =>
    tileControl(
      "round_scoring_tiles", index, tile, "round", `第 ${index + 1} 轮`, 10,
      (value) => `R${value + 1} · ${SETUP_LABELS[ROUND_SETUP_KEYS[value]]}`,
    ),
  ).join("");
  byId("setup-editor-final-scoring").innerHTML = elements.final_scoring_tiles.map((tile, index) =>
    tileControl(
      "final_scoring_tiles", index, tile, "final", `终局槽位 ${index + 1}`, 6,
      (value) => `F${value + 1} · ${SETUP_LABELS[FINAL_SETUP_KEYS[value]]}`,
    ),
  ).join("");
  byId("setup-editor-standard-tech").innerHTML = elements.standard_tech_tiles.map((tile, index) =>
    tileControl(
      "standard_tech_tiles", index, tile, "standard",
      index < 6 ? TRACK_LABELS[TRACK_KEYS[index]] : `通用槽位 ${index - 5}`,
      9,
      (value) => `T${value + 1} · ${SETUP_LABELS[STANDARD_TECH_KEYS[value]]}`,
    ),
  ).join("");
  byId("setup-editor-advanced-tech").innerHTML = elements.advanced_tech_tiles.map((tile, index) =>
    tileControl(
      "advanced_tech_tiles", index, tile, "advanced", TRACK_LABELS[TRACK_KEYS[index]], 15,
      (value) => `A${value + 1} · ${ADVANCED_TECH_NAMES[value]}`,
    ),
  ).join("");
  byId("setup-editor-boosters").innerHTML = elements.booster_tiles.map((tile, index) =>
    tileControl(
      "booster_tiles", index, tile, "booster",
      `可选槽位 ${index + 1}`,
      10,
      (value) => `B${value + 1} · ${BOOSTER_NAMES[value]}`,
    ),
  ).join("");
  byId("setup-editor-federation").innerHTML = `<label><span>轨顶板块</span>
    <select id="setup-editor-federation-tile" data-random-field="terraforming_federation_tile">
      ${selectOptions(6, elements.terraforming_federation_tile, (value) => `联邦 ${value + 1} · ${FEDERATION_NAMES[value]}`)}
    </select>
  </label>`;
}

function planetEditorSnapshot() {
  const snapshot = state.manualSetup.preview;
  const layout = state.manualSetup.randomElements?.planet_layout;
  if (!snapshot?.planets?.length || !Array.isArray(layout)) return null;
  const existing = new Map(snapshot.planets.map((planet) => [Number(planet.id), planet]));
  const sources = new Map(
    (snapshot.setup?.map?.planet_sources || []).map((source) => [Number(source.id), {
      id: Number(source.id),
      source_id: Number(source.id),
      q: Number(source.q),
      r: Number(source.r),
      source_q: Number(source.q),
      source_r: Number(source.r),
      terrain: Number(source.terrain),
      sector: Number(source.sector),
      owner: -1,
      building: "empty",
      gaiaformer: -1,
      federated: false,
    }]));
  for (const planet of snapshot.planets) {
    const sourceId = Number(planet.source_id ?? planet.id);
    if (!sources.has(sourceId)) sources.set(sourceId, planet);
  }
  const planets = layout.map((item) => {
    const id = Number(item.id);
    const sourceId = Number(item.source_id);
    const current = existing.get(id);
    const source = sources.get(sourceId);
    const template = current && Number(current.source_id ?? current.id) === sourceId
      ? current
      : source;
    if (!template) return null;
    return {
      ...template,
      id,
      q: Number(item.q),
      r: Number(item.r),
      source_id: sourceId,
      source_q: Number(source?.source_q ?? source?.q ?? template.source_q ?? template.q),
      source_r: Number(source?.source_r ?? source?.r ?? template.source_r ?? template.r),
      owner: -1,
      building: "empty",
      gaiaformer: -1,
      federated: false,
    };
  });
  if (planets.some((planet) => !planet)) return null;
  return {
    ...snapshot,
    planets,
  };
}

function renderPlanetPositionEditor() {
  const canvas = byId("setup-planet-editor-canvas");
  if (!canvas) return;
  const snapshot = state.manualSetup.mapMode === "manual" ? planetEditorSnapshot() : null;
  const empty = byId("setup-planet-editor-empty");
  const reset = byId("setup-planet-editor-reset");
  const add = byId("setup-planet-editor-add");
  const remove = byId("setup-planet-editor-delete");
  const selectedLabel = byId("setup-planet-editor-selected");
  const coordinate = byId("setup-planet-editor-coordinate");
  const terrainPicker = byId("setup-planet-editor-terrain");
  terrainPicker.value = String(state.manualSetup.planetEditorTerrain);
  empty.hidden = Boolean(snapshot);
  reset.disabled = !snapshot;
  add.disabled = !snapshot;
  byId("setup-planet-editor-count").textContent = snapshot
    ? `${snapshot.planets.length} / 70 颗 · ${assembledBoardSpaces(snapshot.setup?.map?.sectors || []).length} 个合法格`
    : "等待地图预览";
  const selected = snapshot?.planets?.find(
    (planet) => Number(planet.id) === Number(state.manualSetup.selectedPlanetId),
  );
  if (!selected) state.manualSetup.selectedPlanetId = null;
  remove.disabled = !selected;
  add.classList.toggle("active", state.manualSetup.planetEditorMode === "add");
  add.setAttribute("aria-pressed", String(state.manualSetup.planetEditorMode === "add"));
  canvas.classList.toggle("adding", state.manualSetup.planetEditorMode === "add");
  selectedLabel.textContent = selected
    ? `#${selected.id} · ${TERRAIN_LABELS[selected.terrain] || "星球"}`
    : state.manualSetup.planetEditorMode === "add"
      ? `新增 · ${TERRAIN_LABELS[state.manualSetup.planetEditorTerrain]}`
      : "未选择";
  coordinate.textContent = state.manualSetup.planetEditorError
    || (selected ? `坐标 ${selected.q}, ${selected.r}` : "--");
  coordinate.classList.toggle("error", Boolean(state.manualSetup.planetEditorError));
  if (!snapshot) {
    const { context, width, height } = setupCanvas(canvas);
    context.clearRect(0, 0, width, height);
    drawStarfield(context, width, height, 0);
    return;
  }
  drawBoard(canvas, snapshot, {
    showSectors: true,
    planetArtwork: true,
    starfield: true,
    showPlayerPieces: false,
    showPlanetIds: true,
    selectedPlanetId: state.manualSetup.selectedPlanetId,
  });
}

function axialCoordinateDistance(leftQ, leftR, rightQ, rightR) {
  const dq = Number(leftQ) - Number(rightQ);
  const dr = Number(leftR) - Number(rightR);
  return (Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2;
}

function moveSelectedPlanet(q, r) {
  const snapshot = planetEditorSnapshot();
  const selected = snapshot?.planets?.find(
    (planet) => Number(planet.id) === Number(state.manualSetup.selectedPlanetId),
  );
  if (!snapshot || !selected) return;
  const occupied = snapshot.planets.find(
    (planet) => Number(planet.id) !== Number(selected.id)
      && Number(planet.q) === Number(q)
      && Number(planet.r) === Number(r),
  );
  if (occupied) {
    state.manualSetup.selectedPlanetId = Number(occupied.id);
    state.manualSetup.planetEditorError = null;
    renderPlanetPositionEditor();
    return;
  }
  if (Number(selected.terrain) < 7 && snapshot.planets.some((planet) =>
    Number(planet.id) !== Number(selected.id)
    && Number(planet.terrain) === Number(selected.terrain)
    && axialCoordinateDistance(q, r, planet.q, planet.r) === 1
  )) {
    state.manualSetup.planetEditorError = "同色母星不能相邻";
    renderPlanetPositionEditor();
    return;
  }
  const layout = state.manualSetup.randomElements.planet_layout;
  const index = layout.findIndex((item) => Number(item.id) === Number(selected.id));
  layout[index] = { ...layout[index], q: Number(q), r: Number(r) };
  state.manualSetup.planetEditorError = null;
  state.manualSetup.edited = true;
  setSetupEditorMessage(`星球 #${selected.id} 已移动，应用预览后生效`, "ready");
  renderPlanetPositionEditor();
}

function addPlanetAt(q, r) {
  const snapshot = planetEditorSnapshot();
  const layout = state.manualSetup.randomElements?.planet_layout;
  if (!snapshot || !Array.isArray(layout)) return;
  if (layout.length >= 70) {
    state.manualSetup.planetEditorError = "已达到 70 颗星球上限";
    renderPlanetPositionEditor();
    return;
  }
  if (snapshot.planets.some((planet) => Number(planet.q) === q && Number(planet.r) === r)) {
    state.manualSetup.planetEditorError = "该小六角格已有星球";
    renderPlanetPositionEditor();
    return;
  }
  const terrain = Number(state.manualSetup.planetEditorTerrain);
  if (terrain < 7 && snapshot.planets.some((planet) =>
    Number(planet.terrain) === terrain
    && axialCoordinateDistance(q, r, planet.q, planet.r) === 1
  )) {
    state.manualSetup.planetEditorError = "同色母星不能相邻";
    renderPlanetPositionEditor();
    return;
  }
  const source = (state.manualSetup.preview.setup?.map?.planet_sources || []).find(
    (candidate) => Number(candidate.terrain) === terrain,
  );
  if (!source) {
    state.manualSetup.planetEditorError = "当前星区没有该地形的原图素材";
    renderPlanetPositionEditor();
    return;
  }
  const used = new Set(layout.map((item) => Number(item.id)));
  const id = Array.from({ length: 70 }, (_, value) => value).find((value) => !used.has(value));
  layout.push({ id, q: Number(q), r: Number(r), source_id: Number(source.id) });
  layout.sort((left, right) => left.id - right.id);
  state.manualSetup.selectedPlanetId = id;
  state.manualSetup.planetEditorError = null;
  state.manualSetup.edited = true;
  setSetupEditorMessage(`已新增${TERRAIN_LABELS[terrain]}星球 #${id}，应用预览后生效`, "ready");
  renderPlanetPositionEditor();
}

function deleteSelectedPlanet() {
  const id = state.manualSetup.selectedPlanetId;
  const layout = state.manualSetup.randomElements?.planet_layout;
  if (id === null || !Array.isArray(layout)) return;
  const index = layout.findIndex((item) => Number(item.id) === Number(id));
  if (index < 0) return;
  layout.splice(index, 1);
  state.manualSetup.selectedPlanetId = null;
  state.manualSetup.planetEditorMode = "move";
  state.manualSetup.planetEditorError = null;
  state.manualSetup.edited = true;
  setSetupEditorMessage(`星球 #${id} 已删除，应用预览后生效`, "ready");
  renderPlanetPositionEditor();
}

function toggleAddPlanetMode() {
  state.manualSetup.planetEditorMode = state.manualSetup.planetEditorMode === "add" ? "move" : "add";
  state.manualSetup.selectedPlanetId = null;
  state.manualSetup.planetEditorError = null;
  renderPlanetPositionEditor();
}

function handlePlanetEditorClick(event) {
  const snapshot = planetEditorSnapshot();
  const canvas = byId("setup-planet-editor-canvas");
  if (!snapshot || !canvas) return;
  const rect = canvas.getBoundingClientRect();
  const geometry = boardGeometry(rect.width, rect.height, snapshot, true);
  const clickX = event.clientX - rect.left;
  const clickY = event.clientY - rect.top;
  const withPixels = (item) => ({
    ...item,
    x: geometry.offsetX + Math.sqrt(3) * (Number(item.q) + Number(item.r) / 2) * geometry.scale,
    y: geometry.offsetY + 1.5 * Number(item.r) * geometry.scale,
  });
  const planets = snapshot.planets.map(withPixels);
  const nearestSpace = geometry.spaces
    .map(withPixels)
    .map((space) => ({ ...space, distance: Math.hypot(clickX - space.x, clickY - space.y) }))
    .sort((left, right) => left.distance - right.distance)[0];
  if (state.manualSetup.planetEditorMode === "add") {
    if (nearestSpace && nearestSpace.distance <= geometry.scale) {
      addPlanetAt(nearestSpace.q, nearestSpace.r);
    }
    return;
  }
  const nearestPlanet = planets
    .map((planet) => ({ ...planet, distance: Math.hypot(clickX - planet.x, clickY - planet.y) }))
    .sort((left, right) => left.distance - right.distance)[0];
  if (nearestPlanet && nearestPlanet.distance <= Math.max(12, geometry.size * 1.2)) {
    state.manualSetup.selectedPlanetId = Number(nearestPlanet.id);
    state.manualSetup.planetEditorError = null;
    renderPlanetPositionEditor();
    return;
  }
  if (state.manualSetup.selectedPlanetId === null) return;
  if (nearestSpace && nearestSpace.distance <= geometry.scale) {
    moveSelectedPlanet(nearestSpace.q, nearestSpace.r);
  }
}

function resetPlanetLayout() {
  const snapshot = state.manualSetup.preview;
  if (!snapshot?.planets?.length) return;
  state.manualSetup.randomElements.planet_layout = snapshot.planets.map((planet) => ({
    id: Number(planet.id),
    q: Number(planet.q),
    r: Number(planet.r),
    source_id: Number(planet.source_id ?? planet.id),
  }));
  state.manualSetup.selectedPlanetId = null;
  state.manualSetup.planetEditorMode = "move";
  state.manualSetup.planetEditorError = null;
  state.manualSetup.edited = true;
  setSetupEditorMessage("已撤销尚未应用的星球修改", "ready");
  renderPlanetPositionEditor();
}

function captureRandomElements() {
  const players = Number(byId("setup-editor-players").value || 2);
  const result = {
    map_mode: state.manualSetup.mapMode,
    map_size: normalizedMapSize(players, byId("setup-editor-map-size").value),
  };
  const planetLayout = state.manualSetup.randomElements?.planet_layout;
  document.querySelectorAll("[data-random-field]").forEach((select) => {
    const field = select.dataset.randomField;
    const index = select.dataset.randomIndex;
    if (index === undefined) {
      result[field] = Number(select.value);
    } else {
      if (!result[field]) result[field] = [];
      result[field][Number(index)] = Number(select.value);
    }
  });
  if (Array.isArray(planetLayout)) {
    result.planet_layout = planetLayout.map((item) => ({ ...item }));
  }
  state.manualSetup.randomElements = result;
  return result;
}

function validateTileSelection(values, expected, available, label, requireAll = false) {
  if (!Array.isArray(values) || values.length !== expected) {
    throw new Error(`${label}需要 ${expected} 个槽位`);
  }
  if (values.some((value) => !Number.isInteger(value) || value < 0 || value >= available)) {
    throw new Error(`${label}包含无效板块`);
  }
  if (new Set(values).size !== values.length) {
    throw new Error(`${label}不能重复`);
  }
  if (requireAll && new Set(values).size !== available) {
    throw new Error(`${label}必须使用全部板块`);
  }
}

function validatePlanetLayout(layout) {
  if (layout === undefined) return;
  if (!Array.isArray(layout) || layout.length < 1 || layout.length > 70) {
    throw new Error("手动星图必须包含 1–70 颗星球");
  }
  if (layout.some((item) =>
    !Number.isInteger(item.id)
    || !Number.isInteger(item.q)
    || !Number.isInteger(item.r)
    || !Number.isInteger(item.source_id)
  )) {
    throw new Error("手动星图包含无效星球数据");
  }
  if (new Set(layout.map((item) => item.id)).size !== layout.length) {
    throw new Error("单颗星球编号不能重复");
  }
}

function manualSetupPayload({ includeRandom = true } = {}) {
  const players = Number(byId("setup-editor-players").value);
  const factions = [...byId("setup-editor-factions").querySelectorAll("select")]
    .slice(0, players)
    .map((select) => Number(select.value));
  const boards = factions.map((id) => BASE_FACTIONS.find((faction) => faction.id === id)?.board);
  if (factions.length !== players || boards.some((board) => board === undefined)) {
    throw new Error("请为每个座位选择种族");
  }
  if (new Set(boards).size !== boards.length) {
    throw new Error("同一张双面版图不能分配给多个玩家");
  }
  const seed = Number(byId("setup-editor-seed").value);
  const firstPlayer = Number(byId("setup-editor-first-player").value);
  const simulations = Number(byId("setup-editor-simulations").value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) {
    throw new Error("随机种子必须是 0–2147483647 的整数");
  }
  if (!Number.isInteger(simulations) || simulations < 1 || simulations > 128) {
    throw new Error("每步搜索次数必须是 1–128 的整数");
  }
  const config = {
    players,
    seed,
    first_player: firstPlayer,
    factions,
    simulations,
  };
  if (!includeRandom) return config;
  const randomSetup = captureRandomElements();
  const sectorCount = setupSectorCount(players, randomSetup.map_size);
  if (randomSetup.map_mode === "manual") {
    validateTileSelection(randomSetup.sector_tiles, sectorCount, sectorCount, "星区板块", true);
    if (randomSetup.sector_rotations?.length !== sectorCount) {
      throw new Error(`星区旋转需要 ${sectorCount} 个槽位`);
    }
    validatePlanetLayout(randomSetup.planet_layout);
  } else {
    delete randomSetup.sector_tiles;
    delete randomSetup.sector_rotations;
    delete randomSetup.planet_positions;
    delete randomSetup.planet_layout;
  }
  validateTileSelection(randomSetup.booster_tiles, players + 3, 10, "助推板块");
  validateTileSelection(randomSetup.round_scoring_tiles, 6, 10, "轮次计分板块");
  validateTileSelection(randomSetup.final_scoring_tiles, 2, 6, "终局计分板块");
  validateTileSelection(randomSetup.standard_tech_tiles, 9, 9, "基础科技板块", true);
  validateTileSelection(randomSetup.advanced_tech_tiles, 6, 15, "高级科技板块");
  config.random_setup = randomSetup;
  return config;
}

function setSetupEditorMessage(message, status = "ready") {
  state.manualSetup.message = message;
  state.manualSetup.messageStatus = status;
  renderManualSetupStatus();
}

function renderManualSetupStatus() {
  const element = byId("setup-editor-status");
  element.textContent = state.manualSetup.message || "就绪";
  element.className = `setup-editor-status ${state.manualSetup.messageStatus || "ready"}`;
  const disabled = state.manualSetup.busy;
  byId("setup-editor-run").disabled = disabled;
  byId("setup-editor-preview").disabled = disabled;
  byId("setup-editor-randomize").disabled = disabled;
  byId("setup-editor-form").setAttribute("aria-busy", String(disabled));
}

async function previewManualSetup({ quiet = false } = {}) {
  const config = manualSetupPayload();
  state.manualSetup.busy = true;
  if (!quiet) setSetupEditorMessage("正在生成合法初始设置", "running");
  try {
    const response = await fetch("/api/setup/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    state.manualSetup.randomElements = normalizedRandomElements(
      data.config.players,
      data.config.random_setup,
    );
    state.manualSetup.preview = data.state;
    state.manualSetup.hydrated = true;
    state.manualSetup.planetEditorError = null;
    renderRandomElementEditor();
    renderSetup(data.state);
    if (!quiet) setSetupEditorMessage("预览已应用", "complete");
    return data.config;
  } finally {
    state.manualSetup.busy = false;
    renderManualSetupStatus();
  }
}

async function randomizeManualSetup() {
  state.manualSetup.edited = true;
  const requestedMapMode = state.manualSetup.mapMode;
  const random = new Uint32Array(1);
  crypto.getRandomValues(random);
  byId("setup-editor-seed").value = String(random[0] & 0x7fffffff);
  const players = Number(byId("setup-editor-players").value);
  const boards = Array.from({ length: 7 }, (_, board) => board + 1);
  for (let index = boards.length - 1; index > 0; index -= 1) {
    const swap = random[0] % (index + 1);
    [boards[index], boards[swap]] = [boards[swap], boards[index]];
    random[0] = (Math.imul(random[0], 1664525) + 1013904223) >>> 0;
  }
  state.manualSetup.factions = boards.slice(0, players).map((board) => {
    const faces = BASE_FACTIONS.filter((faction) => faction.board === board);
    const face = faces[random[0] % faces.length];
    random[0] = (Math.imul(random[0], 1664525) + 1013904223) >>> 0;
    return face.id;
  });
  renderSetupEditorSeats(random[0] % players);
  state.manualSetup.busy = true;
  setSetupEditorMessage("正在随机生成全部初始板块", "running");
  try {
    const config = manualSetupPayload({ includeRandom: false });
    config.random_setup = {
      map_size: normalizedMapSize(players, byId("setup-editor-map-size").value),
    };
    const response = await fetch("/api/setup/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    state.manualSetup.preview = data.state;
    state.manualSetup.hydrated = true;
    const resolvedSetup = { ...data.config.random_setup, map_mode: requestedMapMode };
    if (requestedMapMode === "manual") {
      resolvedSetup.planet_layout = data.state.planets.map((planet) => ({
        id: Number(planet.id),
        q: Number(planet.q),
        r: Number(planet.r),
        source_id: Number(planet.source_id ?? planet.id),
      }));
    }
    state.manualSetup.randomElements = normalizedRandomElements(players, resolvedSetup);
    state.manualSetup.selectedPlanetId = null;
    state.manualSetup.planetEditorError = null;
    renderRandomElementEditor();
    renderSetup(data.state);
    setSetupEditorMessage("全部随机元素已更新", "complete");
  } catch (error) {
    setSetupEditorMessage(error.message || String(error), "failed");
  } finally {
    state.manualSetup.busy = false;
    renderManualSetupStatus();
  }
}

function setBgaImportMessage(message, status = "ready") {
  state.bgaImport.message = message;
  state.bgaImport.status = status;
  renderBgaImport();
}

function renderBgaImport() {
  const badge = byId("bga-import-badge");
  const message = byId("bga-import-message");
  const submit = byId("bga-import-submit");
  if (!badge || !message || !submit) return;
  const statusLabels = {
    ready: "等待",
    running: "下载中",
    complete: "已保存",
    failed: "失败",
  };
  const statusClasses = {
    ready: "waiting",
    running: "warning",
    complete: "connected",
    failed: "failed",
  };
  badge.className = `health-badge ${statusClasses[state.bgaImport.status] || "waiting"}`;
  badge.textContent = statusLabels[state.bgaImport.status] || "等待";
  message.className = `bga-import-message ${state.bgaImport.status}`;
  message.textContent = state.bgaImport.message;
  submit.disabled = state.bgaImport.busy;
  submit.textContent = state.bgaImport.busy ? "正在下载" : "下载并转换";

  const session = state.bgaImport.session || {};
  const usernameInput = byId("bga-import-username");
  const passwordInput = byId("bga-import-password");
  const clearSession = byId("bga-import-clear-session");
  const sessionStatus = byId("bga-import-session-status");
  if (session.saved && usernameInput && !usernameInput.value) {
    usernameInput.value = session.username || "";
  }
  if (passwordInput) {
    passwordInput.required = !session.saved;
    passwordInput.placeholder = session.saved ? "已保存，可留空" : "";
  }
  if (clearSession) {
    clearSession.hidden = !session.saved;
    clearSession.disabled = state.bgaImport.busy;
  }
  if (sessionStatus) {
    if (session.saved) {
      const updated = session.updated_at ? formatTime(session.updated_at) : "--";
      sessionStatus.textContent = `已保存 ${session.username || "BGA 账号"} · ${formatNumber(session.cookie_count || 0)} 个 Cookie · ${updated}`;
    } else {
      sessionStatus.textContent = "登录信息使用 Windows 当前用户加密，不写入复盘历史";
    }
  }

  const result = state.bgaImport.result;
  byId("bga-import-empty").hidden = Boolean(result);
  byId("bga-import-result").hidden = !result;
  if (!result) return;
  byId("bga-import-table").textContent = String(result.table_id ?? "--");
  byId("bga-import-run").textContent = result.run_id || "--";
  byId("bga-import-moves").textContent = `${formatNumber(result.moves)} 步`;
  byId("bga-import-scores").textContent = (result.scores || []).map((value) => formatNumber(value, 1)).join(" / ") || "--";
  byId("bga-import-file").textContent = result.archive_path || "--";
  byId("bga-import-players").innerHTML = (result.players || []).map((player) => `<div>
    <span>P${formatNumber(player.seat)} · ${escapeHtml(player.name || "--")}</span>
    <strong>${escapeHtml(player.faction || "--")}</strong>
    <b>${formatNumber(player.score, 1)} VP</b>
  </div>`).join("");
}

async function submitBgaImport(event) {
  event.preventDefault();
  if (state.bgaImport.busy) return;
  const passwordInput = byId("bga-import-password");
  const request = {
    username: byId("bga-import-username").value.trim(),
    password: passwordInput.value,
    replay_address: byId("bga-import-address").value.trim(),
    remember: byId("bga-import-remember").checked,
  };
  state.bgaImport.busy = true;
  state.bgaImport.result = null;
  setBgaImportMessage("正在登录 BGA 并下载复盘", "running");
  try {
    const response = await fetch("/api/bga/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    state.bgaImport.result = data;
    const sessionNote = data.used_cached_session ? " · 已复用 Cookie" : "";
    const replayNote = data.used_cached_replay_url ? " · 已复用本地验证过的 archive 地址" : "";
    setBgaImportMessage(`桌号 ${data.table_id} 已写入本地历史${sessionNote}${replayNote}`, "complete");
    await loadBgaSession();
    await refreshHistoryIndex();
  } catch (error) {
    setBgaImportMessage(error.message || String(error), "failed");
  } finally {
    passwordInput.value = "";
    request.password = "";
    state.bgaImport.busy = false;
    renderBgaImport();
  }
}

async function loadBgaSession() {
  try {
    const response = await fetch("/api/bga/session", { cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    state.bgaImport.session = {
      saved: Boolean(data.saved),
      username: data.username || "",
      cookie_count: Number(data.cookie_count || 0),
      updated_at: data.updated_at || null,
    };
  } catch (error) {
    state.bgaImport.session = {
      saved: false,
      username: "",
      cookie_count: 0,
      updated_at: null,
    };
    if (state.bgaImport.status === "ready") {
      state.bgaImport.message = error.message || String(error);
      state.bgaImport.status = "failed";
    }
  }
  renderBgaImport();
}

async function clearBgaSession() {
  if (state.bgaImport.busy || !state.bgaImport.session?.saved) return;
  state.bgaImport.busy = true;
  setBgaImportMessage("正在清除本地 BGA 登录信息", "running");
  try {
    const response = await fetch("/api/bga/session/clear", { method: "POST" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    state.bgaImport.session = {
      saved: false,
      username: "",
      cookie_count: 0,
      updated_at: null,
    };
    byId("bga-import-password").value = "";
    setBgaImportMessage("本地 BGA 账号、密码和 Cookie 已清除", "complete");
  } catch (error) {
    setBgaImportMessage(error.message || String(error), "failed");
  } finally {
    state.bgaImport.busy = false;
    renderBgaImport();
  }
}

async function openImportedBgaHistory() {
  const runId = state.bgaImport.result?.run_id;
  if (!runId) return;
  state.history.runId = runId;
  state.history.iteration = 1;
  state.history.game = 1;
  state.history.trace = null;
  selectView("history");
  await refreshHistoryIndex();
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

async function refreshHistoryIndex({ loadTrace = true, force = false } = {}) {
  if (state.history.deleting && !force) return;
  const requestId = ++state.history.indexRequestId;
  try {
    const response = await fetch("/api/history", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const index = await response.json();
    if (requestId !== state.history.indexRequestId) return;
    state.history.index = index;
    const runs = historyRuns();
    if (!runs.length) {
      state.history.runId = null;
      state.history.iteration = null;
      state.history.game = null;
      state.history.trace = null;
      state.history.loading = false;
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
    if (loadTrace && state.history.runId && state.history.iteration !== null && state.history.game !== null) {
      const current = state.history.trace;
      const sameGame = current
        && current.run_id === state.history.runId
        && current.iteration === state.history.iteration
        && current.game === state.history.game;
      await loadHistoryGame(!sameGame);
    } else {
      state.history.trace = null;
      state.history.loading = false;
      renderHistory();
    }
  } catch (error) {
    if (requestId !== state.history.indexRequestId) return;
    state.history.index = state.history.index || { runs: [] };
    state.history.loading = false;
    state.history.message = error.message || String(error);
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
  const requestId = state.history.indexRequestId;
  state.history.loading = true;
  renderHistory();
  try {
    const params = new URLSearchParams({ run_id: runId, iteration: String(iteration), game: String(game) });
    const response = await fetch(`/api/game?${params.toString()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const trace = await response.json();
    if (requestId === state.history.indexRequestId
      && !state.history.deleting
      && trace.run_id === state.history.runId
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
  const deleteButton = byId("history-delete");
  if (!runSelect || !iterationSelect || !gameSelect) return;
  const runs = historyRuns();
  runSelect.innerHTML = runs.length
    ? runs.map((run) => {
      const source = run.source === "bga" ? "BGA 复盘" : run.source === "local" ? "本地对战" : "训练";
      const status = run.status === "complete" ? "已完成" : run.status === "active" ? "进行中" : run.status;
      return `<option value="${escapeHtml(run.run_id)}">${source} · ${escapeHtml(run.ruleset || "unknown")} · ${escapeHtml(run.run_id)} · ${escapeHtml(status)}</option>`;
    }).join("")
    : '<option value="">暂无运行</option>';
  runSelect.value = state.history.runId || "";
  const iterations = historyRun()?.iterations || [];
  iterationSelect.innerHTML = iterations.length
    ? iterations.map((item) => ["local", "bga"].includes(historyRun()?.source)
      ? `<option value="${item.iteration}">本地记录 · ${item.games.length} 局</option>`
      : `<option value="${item.iteration}">第 ${item.iteration} 轮 · ${item.games.length} 局</option>`).join("")
    : '<option value="">暂无迭代</option>';
  iterationSelect.value = state.history.iteration === null ? "" : String(state.history.iteration);
  const games = historyIteration()?.games || [];
  gameSelect.innerHTML = games.length
    ? games.map((item) => {
      const score = item.scores ? ` · ${(item.scores || []).map((value) => formatNumber(value, 1)).join("/")}` : "";
      const coverage = item.moves === null ? "" : ` · ${item.captured_moves}/${item.moves} 步`;
      const label = historyRun()?.source === "bga"
        ? "BGA 对局"
        : historyRun()?.source === "local" ? "人工对局" : `第 ${item.game} 局`;
      return `<option value="${item.game}">${label}${score}${coverage}</option>`;
    }).join("")
    : '<option value="">暂无对局</option>';
  gameSelect.value = state.history.game === null ? "" : String(state.history.game);
  runSelect.disabled = !runs.length;
  iterationSelect.disabled = !iterations.length;
  gameSelect.disabled = !games.length;
  if (deleteButton) {
    const source = historyRun()?.source;
    const deletable = ["local", "bga"].includes(source);
    deleteButton.disabled = state.history.deleting || state.history.loading || !deletable;
    deleteButton.textContent = state.history.deleting ? "删除中" : "删除";
    deleteButton.title = deletable
      ? "永久删除当前本地历史"
      : source === "training" ? "训练记录来自指标日志，不能在此删除" : "没有可删除的历史";
  }
}

async function deleteSelectedHistory() {
  const selected = historyRun();
  if (state.history.deleting || !["local", "bga"].includes(selected?.source)) return;
  const sourceLabel = selected.source === "bga" ? "BGA 复盘" : "人工对局";
  const confirmed = window.confirm(`永久删除当前${sourceLabel}？\n${selected.run_id}`);
  if (!confirmed) return;
  const runs = historyRuns();
  const selectedIndex = runs.findIndex((run) => run.run_id === selected.run_id);
  const fallback = runs[selectedIndex - 1] || runs[selectedIndex + 1] || null;
  state.history.deleting = true;
  state.history.indexRequestId += 1;
  state.history.loading = false;
  state.history.message = `正在删除${sourceLabel}...`;
  stopHistoryPlayback();
  renderHistorySelectors();
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 15_000);
  try {
    const params = new URLSearchParams({ run_id: selected.run_id });
    const response = await fetch(`/api/history?${params.toString()}`, {
      method: "DELETE",
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    state.history.runId = fallback?.run_id || null;
    state.history.iteration = null;
    state.history.game = null;
    state.history.trace = null;
    state.history.step = 0;
    state.history.message = `${sourceLabel}已删除`;
    state.history.deleting = false;
    await refreshHistoryIndex({ loadTrace: true, force: true });
  } catch (error) {
    state.history.message = error.name === "AbortError"
      ? "删除请求超时，请稍后重试"
      : (error.message || String(error));
  } finally {
    window.clearTimeout(timeout);
    state.history.deleting = false;
    renderHistorySelectors();
    renderHistory();
  }
}

function snapshotRoundLabel(snapshot, compact = false) {
  if (!snapshot) return compact ? "--" : "第 -- 轮";
  if (snapshot.phase === "starting_placement" || snapshot.placement?.active) {
    const step = Number(snapshot.placement?.step || 0);
    const total = Number(snapshot.placement?.total || 0);
    return compact ? `开局 ${step}/${total}` : `开局基地 ${step} / ${total}`;
  }
  if (snapshot.phase === "booster_selection" || snapshot.booster_selection?.active) {
    const step = Number(snapshot.booster_selection?.step || 0);
    const total = Number(snapshot.booster_selection?.total || 0);
    return compact ? `助推 ${step}/${total}` : `选择助推 ${step} / ${total}`;
  }
  return compact ? `第 ${snapshot.round} 轮` : `第 ${snapshot.round} / ${snapshot.max_rounds} 轮`;
}

function auditHistoryTrace(trace) {
  if (!trace) return [];
  if (trace.source === "bga") return auditBgaHistoryTrace(trace);
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
  const roundsValid = rounds.every((round, index) => round >= 0 && round <= 6 && (index === 0 || round >= rounds[index - 1]));
  checks.push({
    status: roundsValid && rounds.length ? "pass" : "fail",
    title: roundsValid && rounds.length ? "轮次顺序连续" : "轮次顺序异常",
    detail: rounds.length ? `含开局阶段，观测到 ${new Set(rounds).size} 个阶段/轮次` : "没有可验证的状态"
  });
  const placementStates = states.filter((snapshot) => snapshot.phase === "starting_placement");
  const placementValid = placementStates.length > 0 && placementStates.every((snapshot) => {
    const placement = snapshot.placement || {};
    const step = Number(placement.step);
    const total = Number(placement.total);
    const order = placement.order || [];
    const structures = (snapshot.planets || []).filter((planet) => Number(planet.owner) >= 0).length;
    return Number(snapshot.round) === 0
      && Number.isInteger(step) && step >= 0 && step < total
      && order.length === total
      && Number(snapshot.current_player) === Number(order[step])
      && structures === step;
  });
  checks.push({
    status: placementValid ? "pass" : placementStates.length ? "fail" : "warn",
    title: placementValid ? "开局基地按蛇形顺位放置" : "开局基地流程待验证",
    detail: placementStates.length
      ? `记录了 ${placementStates.length} 个开局状态，顺序与建筑数量一致`
      : "旧日志没有开局放置阶段"
  });
  const boosterStates = states.filter((snapshot) => snapshot.phase === "booster_selection");
  const boosterValid = boosterStates.length > 0 && boosterStates.every((snapshot) => {
    const selection = snapshot.booster_selection || {};
    const step = Number(selection.step);
    const total = Number(selection.total);
    const order = selection.order || [];
    const assigned = (snapshot.setup?.boosters || []).filter((booster) => Number(booster.owner) >= 0).length;
    return Number(snapshot.round) === 0
      && Number.isInteger(step) && step >= 0 && step < total
      && order.length === total
      && Number(snapshot.current_player) === Number(order[step])
      && assigned === step;
  });
  checks.push({
    status: boosterValid ? "pass" : boosterStates.length ? "fail" : "warn",
    title: boosterValid ? "助推板块按逆顺位选取" : "助推板块选取流程待验证",
    detail: boosterStates.length
      ? `记录了 ${boosterStates.length} 个助推选择状态，顺序与归属数量一致`
      : "旧日志没有开局助推选择阶段"
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

function auditBgaHistoryTrace(trace) {
  const steps = trace.steps || [];
  const moves = steps.filter((step) => Number(step.move) > 0);
  const expected = Number(trace.summary?.moves);
  const states = steps.map((step) => step.state).filter(Boolean);
  const complete = Number.isFinite(expected)
    && trace.trace_complete
    && moves.length === expected
    && moves.every((step, index) => Number(step.move) === index + 1);
  const rounds = states.map((snapshot) => Number(snapshot.round)).filter(Number.isFinite);
  const roundsValid = rounds.length > 0 && rounds.every((round, index) => (
    round >= 0 && round <= 6 && (index === 0 || round >= rounds[index - 1])
  ));
  const noticesPreserved = moves.length > 0 && moves.every((step) => (
    step.record?.role === "bga"
    && Array.isArray(step.record?.bga?.notifications)
  ));
  const snapshotsCompatible = states.length === steps.length && states.every((snapshot) => (
    Array.isArray(snapshot.players)
    && Array.isArray(snapshot.planets)
    && snapshot.players.every((player, index) => Number(player.id) === index)
  ));
  const vpLedgersComplete = moves.length > 0 && moves.every((step) => {
    const ledger = step.record?.vp;
    const beforeState = steps.find((candidate) => Number(candidate.move) === Number(step.move) - 1)?.state;
    const beforeScores = beforeState?.scores || [];
    const afterScores = step.state?.scores || [];
    return Array.isArray(ledger?.before)
      && Array.isArray(ledger?.after)
      && Array.isArray(ledger?.events)
      && ledger.before.length === beforeScores.length
      && ledger.after.length === afterScores.length
      && ledger.before.every((value, index) => Number(value) === Number(beforeScores[index]))
      && ledger.after.every((value, index) => Number(value) === Number(afterScores[index]));
  });
  const finalLedger = moves.at(-1)?.record?.vp;
  const finalVpMatches = vpLedgersComplete && finalLedger?.matches_result_page !== false;
  const terminal = Boolean(states.at(-1)?.terminal);
  return [
    {
      status: complete ? "pass" : "fail",
      title: complete ? "BGA 行动已完整转换" : "BGA 行动编号不连续",
      detail: `${moves.length} / ${Number.isFinite(expected) ? expected : "--"} 步`,
    },
    {
      status: noticesPreserved ? "pass" : "warn",
      title: noticesPreserved ? "BGA 原始通知已关联到步骤" : "部分步骤缺少 BGA 通知",
      detail: noticesPreserved ? "可用板块 ID、开销和收入字段核对转换结果" : "请重新下载该复盘",
    },
    {
      status: finalVpMatches ? "pass" : vpLedgersComplete ? "warn" : "warn",
      title: finalVpMatches ? "逐步 VP 与 BGA 终局分数一致" : vpLedgersComplete ? "逐步 VP 存在终局校准" : "旧记录缺少逐步 VP 账本",
      detail: finalVpMatches
        ? "每步均保存 VP 前值、计分事件和后值"
        : vpLedgersComplete
          ? `${(finalLedger.reconciliation || []).length} 名玩家需要对照 BGA 通知排查`
          : "重新下载该复盘可补齐过轮和行动计分",
    },
    {
      status: snapshotsCompatible ? "pass" : "fail",
      title: snapshotsCompatible ? "本地回放状态格式兼容" : "本地状态快照不完整",
      detail: snapshotsCompatible ? `${states.at(-1)?.players?.length || 0} 名玩家 · ${states.at(-1)?.planets?.length || 0} 颗星球` : "玩家或星图字段缺失",
    },
    {
      status: roundsValid ? "pass" : "fail",
      title: roundsValid ? "轮次顺序有效" : "轮次顺序异常",
      detail: rounds.length ? `范围 ${Math.min(...rounds)} - ${Math.max(...rounds)}` : "没有轮次状态",
    },
    {
      status: terminal ? "pass" : "warn",
      title: terminal ? "BGA 复盘已到终局" : "BGA 复盘未到终局",
      detail: terminal ? `最终分数 ${(trace.summary?.scores || []).map((value) => formatNumber(value, 1)).join(" / ")}` : "可能下载了未完成或截断的记录",
    },
  ];
}

function historyDelta(previous, current, step) {
  if (!current) return "等待状态变化";
  if (!previous) return "初始状态 · 没有前置动作";
  const changes = [];
  const vpLedger = step.record?.vp;
  const labels = [["credits", "信用点"], ["ore", "矿石"], ["knowledge", "知识"], ["qic", "QIC"]];
  if (!vpLedger) labels.push(["vp", "VP"]);
  for (const after of current.players || []) {
    const player = Number(after.id);
    const before = previous.players?.find((candidate) => Number(candidate.id) === player);
    if (!before) continue;
    for (const [key, label] of labels) {
      const delta = Number(after[key]) - Number(before[key]);
      if (delta) changes.push(`P${player} ${label} ${delta > 0 ? "+" : ""}${delta}`);
    }
    if (before.power && after.power && before.power.join() !== after.power.join()) {
      changes.push(`P${player} 能量 ${before.power.join("/")} → ${after.power.join("/")}`);
    }
    if (before.tracks && after.tracks && before.tracks.join() !== after.tracks.join()) {
      const trackChanges = after.tracks.flatMap((level, track) => (
        Number(level) === Number(before.tracks[track])
          ? []
          : [`${TRACK_LABELS[TRACK_KEYS[track]] || `科研轨 ${track + 1}`} L${before.tracks[track]} → L${level}`]
      ));
      changes.push(`P${player} ${trackChanges.join("，")}`);
    }
  }
  const previousPlanets = new Map((previous.planets || []).map((planet) => [Number(planet.id), planet]));
  for (const planet of current.planets || []) {
    const before = previousPlanets.get(Number(planet.id));
    if (!before) continue;
    if (Number(before.owner) !== Number(planet.owner) || before.building !== planet.building) {
      const building = BUILDING_SPECS.find((item) => item.key === planet.building)?.label || planet.building;
      changes.push(`星球 P-${planet.id}：${Number(planet.owner) >= 0 ? `P${planet.owner} ${building}` : "建筑移除"}`);
    }
    const beforeCoexisting = Number(before.coexisting_mine_owner ?? -1);
    const afterCoexisting = Number(planet.coexisting_mine_owner ?? -1);
    if (beforeCoexisting !== afterCoexisting) {
      changes.push(`星球 P-${planet.id}：${afterCoexisting >= 0 ? `P${afterCoexisting} 共存矿场` : "共存矿场移除"}`);
    }
    const beforeGaiaformer = Number(before.gaiaformer ?? -1);
    const afterGaiaformer = Number(planet.gaiaformer ?? -1);
    if (beforeGaiaformer !== afterGaiaformer) {
      changes.push(`星球 P-${planet.id}：盖亚塑形者 ${beforeGaiaformer} → ${afterGaiaformer}`);
    }
    if (Number(before.terrain) !== Number(planet.terrain)) {
      changes.push(`星球 P-${planet.id}：${TERRAIN_LABELS[before.terrain] || before.terrain} → ${TERRAIN_LABELS[planet.terrain] || planet.terrain}`);
    }
  }
  if (Array.isArray(vpLedger?.events) && vpLedger.events.length) {
    for (const event of vpLedger.events) {
      const delta = Number(event.delta || 0);
      changes.push(`P${event.player} ${event.reason || "计分"} ${delta >= 0 ? "+" : ""}${delta} VP`);
    }
  } else if (Array.isArray(vpLedger?.changes)) {
    for (const change of vpLedger.changes) {
      const delta = Number(change.delta || 0);
      changes.push(`P${change.player} VP ${formatNumber(change.before, 1)} → ${formatNumber(change.after, 1)} (${delta >= 0 ? "+" : ""}${formatNumber(delta, 1)})`);
    }
  }
  if (Array.isArray(vpLedger?.reconciliation) && vpLedger.reconciliation.length) {
    for (const change of vpLedger.reconciliation) {
      const delta = Number(change.delta || 0);
      changes.push(`P${change.player} BGA 终局校准 ${delta >= 0 ? "+" : ""}${formatNumber(delta, 1)} VP`);
    }
  }
  if (current.phase === "starting_placement" || current.placement?.active) {
    changes.push(`开局基地 ${current.placement?.step || 0}/${current.placement?.total || 0}`);
  } else if (current.phase === "booster_selection") {
    if (previous.phase === "starting_placement") changes.push("开局基地摆放完成");
    changes.push(`选择助推 ${current.booster_selection?.step || 0}/${current.booster_selection?.total || 0}`);
  } else if (previous.phase === "booster_selection" && Number(current.round) === 1) {
    changes.push("助推板块选择完成");
    changes.push("进入第 1 轮");
  } else if (previous.phase === "starting_placement" && Number(current.round) === 1) {
    changes.push("开局基地摆放完成");
    changes.push("进入第 1 轮");
  } else if (Number(previous.round) !== Number(current.round)) {
    changes.push(`进入第 ${current.round} 轮`);
  }
  return changes.length ? changes.join(" · ") : "状态已转移，资源数值无变化";
}

function renderHistory() {
  const content = byId("history-content");
  const empty = byId("history-empty");
  if (!content || !empty) return;
  renderHistorySelectors();
  const status = byId("history-action-status");
  if (status) {
    status.textContent = state.history.message || "";
    status.hidden = !state.history.message;
  }
  const trace = state.history.trace;
  const hasTrace = Boolean(trace?.steps?.length);
  empty.hidden = hasTrace || state.history.loading;
  content.hidden = !hasTrace;
  if (!hasTrace) {
    empty.textContent = state.history.loading
      ? "正在读取历史快照"
      : "暂无可加载的本地对战或自博弈记录";
    return;
  }
  const steps = trace.steps;
  state.history.step = Math.max(0, Math.min(state.history.step, steps.length - 1));
  const step = steps[state.history.step];
  const snapshot = step.state;
  const mapView = state.history.mapView;
  const mapFrame = byId("history-board-canvas")?.parentElement;
  if (mapFrame) mapFrame.style.backgroundColor = mapView.background;
  const zoomInput = byId("history-map-zoom");
  const backgroundInput = byId("history-map-background");
  if (zoomInput) zoomInput.value = String(mapView.zoom);
  if (backgroundInput) backgroundInput.value = mapView.background;
  const zoomOutput = byId("history-map-zoom-value");
  if (zoomOutput) zoomOutput.value = `${Math.round(mapView.zoom * 100)}%`;
  drawStarMapBoard(byId("history-board-canvas"), snapshot, true, {
    showPlanetIds: true,
    selectedPlanetId: step.record?.target ?? null,
    zoom: mapView.zoom,
    gap: 0,
    backgroundColor: mapView.background,
  });
  byId("history-board-empty").hidden = Boolean(snapshot);
  byId("history-board-round").textContent = snapshotRoundLabel(snapshot);
  const summary = trace.summary || {};
  byId("history-final-scores").textContent = (summary.scores || snapshot?.scores || []).map((value) => formatNumber(value, 1)).join(" / ") || "--";
  byId("history-trace-coverage").textContent = summary.moves === undefined ? `${trace.captured_moves} 步` : `${trace.captured_moves} / ${summary.moves} 步`;
  byId("history-duration").textContent = summary.duration_seconds === undefined ? "--" : formatDuration(Number(summary.duration_seconds));
  byId("history-ruleset").textContent = snapshot?.ruleset || "--";
  byId("history-action-code").textContent = step.action === null || step.action === undefined ? "--" : String(step.action);
  byId("history-action-label").textContent = step.action_label || "状态快照";
  const actionPlayer = snapshot?.players?.find((player) => Number(player.id) === Number(step.player));
  byId("history-action-player").textContent = step.player === null || step.player === undefined
    ? "P--"
    : `P${step.player}${actionPlayer?.name ? ` · ${actionPlayer.name}` : ""}`;
  byId("history-step-slider").max = String(Math.max(0, steps.length - 1));
  byId("history-step-slider").value = String(state.history.step);
  byId("history-step-label").textContent = `${step.move} / ${Math.max(0, steps.length - 1)}`;
  const previous = steps[state.history.step - 1]?.state;
  byId("history-delta").textContent = historyDelta(previous, snapshot, step);
  renderHistoryConfiguredComponents(snapshot);
  renderHistoryResearchBoard(snapshot);
  renderPlayerRows("history-players-table", snapshot, "history-active-player");
  renderPersonalBoards("history-player-board-grid", snapshot);
  const checks = auditHistoryTrace(trace);
  const failed = checks.some((check) => check.status === "fail");
  const warned = checks.some((check) => check.status === "warn");
  const badge = byId("history-audit-badge");
  badge.className = `health-badge ${failed ? "failed" : warned ? "warning" : "connected"}`;
  badge.textContent = failed ? "发现异常" : warned ? "需补录" : "通过";
  byId("history-check-list").innerHTML = checks.map((check) => `<li class="${check.status}"><span>${escapeHtml(check.title)}</span><small>${escapeHtml(check.detail)}</small></li>`).join("");
  const automaticSteps = steps.reduce(
    (total, item) => total + (item.record?.automatic_steps?.length || 0),
    0,
  );
  byId("history-move-count").textContent = automaticSteps
    ? `${Math.max(0, steps.length - 1)} 个动作 · ${automaticSteps} 个系统步骤`
    : `${Math.max(0, steps.length - 1)} 个动作`;
  const actionLog = byId("history-action-log");
  actionLog.innerHTML = steps.length
    ? steps.map((item, index) => renderHistoryActionEntry(
      item,
      steps[index - 1],
      index,
      state.history.step,
    )).join("")
    : '<div class="play-action-log-empty">暂无动作记录</div>';
  const currentEntry = actionLog.querySelector(".current-step");
  if (currentEntry) {
    const logBounds = actionLog.getBoundingClientRect();
    const entryBounds = currentEntry.getBoundingClientRect();
    if (entryBounds.top < logBounds.top || entryBounds.bottom > logBounds.bottom) {
      actionLog.scrollTop += entryBounds.top - logBounds.top - actionLog.clientHeight / 3;
    }
  }
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
  byId("board-round").textContent = snapshotRoundLabel(snapshot);
  drawBoard(byId("board-canvas"), snapshot);
}

function seededCanvasRandom(seed) {
  let stateValue = (Number(seed) || 0) ^ 0x9e3779b9;
  return () => {
    stateValue ^= stateValue << 13;
    stateValue ^= stateValue >>> 17;
    stateValue ^= stateValue << 5;
    return (stateValue >>> 0) / 4294967296;
  };
}

function drawStarfield(context, width, height, seed, backgroundColor = "#050b14") {
  context.save();
  context.fillStyle = backgroundColor;
  context.fillRect(0, 0, width, height);
  const random = seededCanvasRandom(seed);
  const stars = Math.max(120, Math.round(width * height / 1050));
  for (let index = 0; index < stars; index += 1) {
    const x = random() * width;
    const y = random() * height;
    const radius = random() > 0.93 ? 1.25 : random() > 0.72 ? 0.75 : 0.4;
    const palette = random();
    context.fillStyle = palette > 0.86
      ? `rgba(157,204,255,${0.38 + random() * 0.42})`
      : palette < 0.08
        ? `rgba(255,224,165,${0.34 + random() * 0.34})`
        : `rgba(238,245,255,${0.24 + random() * 0.48})`;
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fill();
  }
  context.restore();
}

function sectorLocalSpaces() {
  const spaces = [];
  for (let q = -2; q <= 2; q += 1) {
    for (let r = -2; r <= 2; r += 1) {
      if (Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r)) <= 2) {
        spaces.push({ q, r });
      }
    }
  }
  return spaces;
}

function assembledBoardSpaces(sectors) {
  const spaces = new Map();
  for (const sector of sectors || []) {
    for (const local of sectorLocalSpaces()) {
      const q = Number(sector.q) + local.q;
      const r = Number(sector.r) + local.r;
      spaces.set(`${q},${r}`, { q, r });
    }
  }
  return [...spaces.values()];
}

function boardGeometry(width, height, snapshot, showSectors, zoom = 1) {
  const sectors = snapshot.setup?.map?.sectors || [];
  const spaces = assembledBoardSpaces(sectors);
  const tileWidthUnits = 5 * Math.sqrt(3);
  const tileHeightUnits = 8;
  const raw = showSectors && sectors.length
    ? sectors.flatMap((sector) => {
      const rawX = Math.sqrt(3) * (Number(sector.q) + Number(sector.r) / 2);
      const rawY = 1.5 * Number(sector.r);
      return [
        { rawX: rawX - tileWidthUnits / 2, rawY: rawY - tileHeightUnits / 2 },
        { rawX: rawX + tileWidthUnits / 2, rawY: rawY + tileHeightUnits / 2 },
      ];
    })
    : (showSectors && spaces.length ? spaces : snapshot.planets).map((item) => ({
      rawX: Math.sqrt(3) * (Number(item.q) + Number(item.r) / 2),
      rawY: 1.5 * Number(item.r),
    }));
  const minX = Math.min(...raw.map((point) => point.rawX));
  const maxX = Math.max(...raw.map((point) => point.rawX));
  const minY = Math.min(...raw.map((point) => point.rawY));
  const maxY = Math.max(...raw.map((point) => point.rawY));
  const compactMap = showSectors && width < 500;
  const padding = compactMap ? 54 : (showSectors ? 90 : 70);
  const baseScale = Math.max(1, Math.min(
    Math.max(1, width - padding) / Math.max(1, maxX - minX),
    Math.max(1, height - padding) / Math.max(1, maxY - minY),
  ));
  const scale = baseScale * Math.max(0.7, Math.min(1.8, Number(zoom) || 1));
  return {
    spaces,
    compactMap,
    scale,
    size: Math.max(compactMap ? 8 : 15, Math.min(29, scale * 0.47)),
    offsetX: (width - (maxX - minX) * scale) / 2 - minX * scale,
    offsetY: (height - (maxY - minY) * scale) / 2 - minY * scale,
    tileWidthUnits,
    tileHeightUnits,
  };
}

function drawSectorArtworkBackground(
  context,
  sectors,
  scale,
  offsetX,
  offsetY,
  compactMap,
  tileWidthUnits = 5 * Math.sqrt(3),
  tileHeightUnits = 8,
  gap = 0,
) {
  // Keep a transparent border around every BGA sector so adjacent tiles remain
  // visually distinct instead of merging into one continuous hex field.
  const artworkScale = Math.max(0.72, (compactMap ? 0.89 : 0.92) - Math.max(0, Number(gap) || 0) * 0.012);
  const imageWidth = tileWidthUnits * scale * artworkScale;
  const imageHeight = tileHeightUnits * scale * artworkScale;
  const artwork = getMapPieceArtwork("sectorBackground");
  for (const sector of sectors) {
    const rawX = Math.sqrt(3) * (Number(sector.q) + Number(sector.r) / 2);
    const rawY = 1.5 * Number(sector.r);
    const x = offsetX + rawX * scale;
    const y = offsetY + rawY * scale;
    if (!artwork) {
      drawSectorHex(context, x, y, Math.min(imageWidth, imageHeight) * 0.48, "#111a2d");
      continue;
    }
    context.save();
    context.translate(x, y);
    context.rotate(Math.PI / 2 + Number(sector.rotation || 0) * Math.PI / 180);
    context.globalAlpha = 0.98;
    context.drawImage(artwork, -imageHeight / 2, -imageWidth / 2, imageHeight, imageWidth);
    context.restore();
  }
}

function drawAssembledHexGrid(context, sectors, scale, offsetX, offsetY, compactMap) {
  const localSpaces = sectorLocalSpaces();
  context.save();
  context.strokeStyle = "rgba(145, 177, 204, 0.16)";
  context.lineWidth = compactMap ? 0.65 : 0.85;
  for (const sector of sectors) {
    for (const local of localSpaces) {
      const q = Number(sector.q) + local.q;
      const r = Number(sector.r) + local.r;
      const x = offsetX + Math.sqrt(3) * (q + r / 2) * scale;
      const y = offsetY + 1.5 * r * scale;
      drawHexOutline(context, x, y, scale * 0.97);
    }
    const rawX = Math.sqrt(3) * (Number(sector.q) + Number(sector.r) / 2);
    const rawY = 1.5 * Number(sector.r);
    context.fillStyle = "rgba(194, 213, 230, 0.55)";
    context.font = `650 ${compactMap ? 7 : 9}px Segoe UI`;
    context.textAlign = "center";
    context.fillText(
      `S${String(sector.tile).padStart(2, "0")}`,
      offsetX + rawX * scale,
      offsetY + rawY * scale + (compactMap ? 2 : 3),
    );
  }
  context.restore();
}

function drawSectorSeparators(context, sectors, scale, offsetX, offsetY, compactMap, gap = 0) {
  const localSpaces = sectorLocalSpaces();
  for (const sector of sectors) {
    const edges = new Map();
    for (const local of localSpaces) {
      const q = Number(sector.q) + local.q;
      const r = Number(sector.r) + local.r;
      const centerX = offsetX + Math.sqrt(3) * (q + r / 2) * scale;
      const centerY = offsetY + 1.5 * r * scale;
      const vertices = Array.from({ length: 6 }, (_, side) => {
        const angle = Math.PI / 180 * (60 * side - 30);
        return {
          x: centerX + scale * Math.cos(angle),
          y: centerY + scale * Math.sin(angle),
        };
      });
      for (let side = 0; side < 6; side += 1) {
        const start = vertices[side];
        const end = vertices[(side + 1) % 6];
        const first = `${start.x.toFixed(3)},${start.y.toFixed(3)}`;
        const second = `${end.x.toFixed(3)},${end.y.toFixed(3)}`;
        const key = first < second ? `${first}|${second}` : `${second}|${first}`;
        const edge = edges.get(key);
        if (edge) edge.count += 1;
        else edges.set(key, { start, end, count: 1 });
      }
    }

    context.save();
    context.beginPath();
    for (const edge of edges.values()) {
      if (edge.count !== 1) continue;
      context.moveTo(edge.start.x, edge.start.y);
      context.lineTo(edge.end.x, edge.end.y);
    }
    context.lineCap = "round";
    context.lineJoin = "round";
    context.strokeStyle = "rgba(1, 5, 12, 0.96)";
    context.lineWidth = Math.max(compactMap ? 3.5 : 5.5, scale * (0.24 + Math.max(0, Number(gap) || 0) * 0.012));
    context.stroke();
    context.strokeStyle = "rgba(126, 161, 190, 0.42)";
    context.lineWidth = compactMap ? 0.7 : 1;
    context.stroke();
    context.restore();
  }
}

function drawPlanetArtwork(context, x, y, size, cellScale, planet, snapshot) {
  void snapshot;
  const terrain = Number(planet.terrain);
  const column = BGA_PLANET_COLUMNS[terrain];
  const image = getMapPieceArtwork("planets");
  if (!image || !Number.isInteger(column)) return false;
  const terrainScale = terrain === 7 || terrain === 9 ? 0.76 : terrain === 5 ? 0.88 : 0.84;
  const radius = Math.min(size * terrainScale, cellScale * 0.72);

  context.save();
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(
    image,
    column * 132,
    0,
    132,
    132,
    x - radius,
    y - radius,
    radius * 2,
    radius * 2,
  );
  context.restore();
  context.save();
  context.strokeStyle = "rgba(226, 239, 249, 0.24)";
  context.lineWidth = Math.max(0.75, size * 0.05);
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.stroke();
  context.restore();
  return true;
}

function drawPlayerToken(context, x, y, playerId, cellSize, compactMap) {
  const radius = Math.max(compactMap ? 4 : 5.5, cellSize * (compactMap ? 0.22 : 0.27));
  context.save();
  context.fillStyle = PLAYER_COLORS[playerId] || "#d9e2e8";
  context.strokeStyle = "rgba(255,255,255,0.92)";
  context.lineWidth = compactMap ? 0.8 : 1.1;
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.fillStyle = "#ffffff";
  context.font = `800 ${Math.max(5, radius * 0.95)}px Segoe UI`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(`P${playerId}`, x, y + 0.2);
  context.restore();
}

function drawStarMapBoard(canvas, snapshot, showPlayerPieces, extraOptions = {}) {
  drawBoard(canvas, snapshot, {
    showSectors: true,
    planetArtwork: true,
    starfield: true,
    showPlayerPieces,
    ...extraOptions,
  });
}

function drawBoard(canvas, snapshot, options = {}) {
  if (!canvas) return;
  const { context, width, height } = setupCanvas(canvas);
  context.clearRect(0, 0, width, height);
  if (!snapshot || !snapshot.planets?.length) return;
  if (options.starfield) drawStarfield(
    context,
    width,
    height,
    snapshot.setup?.seed || 0,
    options.backgroundColor || "#050b14",
  );
  const showSectors = options.showSectors ?? Boolean(snapshot.setup?.map?.sectors?.length);
  const sectors = snapshot.setup?.map?.sectors || [];

  const points = snapshot.planets.map((planet) => ({
    ...planet,
    rawX: Math.sqrt(3) * (planet.q + planet.r / 2),
    rawY: 1.5 * planet.r
  }));
  const geometry = boardGeometry(width, height, snapshot, showSectors, options.zoom || 1);
  const {
    compactMap,
    scale,
    size,
    offsetX,
    offsetY,
    tileWidthUnits,
    tileHeightUnits,
  } = geometry;
  const legalPlanetIds = new Set((options.legalPlanetIds || []).map(Number));
  const legalSpaceStations = options.legalSpaceStations || [];

  if (showSectors && sectors.length) {
    const useSectorArtwork = options.sectorArtwork ?? true;
    if (useSectorArtwork) {
      drawSectorArtworkBackground(
        context,
        sectors,
        scale,
        offsetX,
        offsetY,
        compactMap,
        tileWidthUnits,
        tileHeightUnits,
        options.gap || 0,
      );
      drawAssembledHexGrid(context, sectors, scale, offsetX, offsetY, compactMap);
      drawSectorSeparators(context, sectors, scale, offsetX, offsetY, compactMap, options.gap || 0);
    } else if (options.starfield) {
      drawAssembledHexGrid(context, sectors, scale, offsetX, offsetY, compactMap);
    } else {
      const sectorSize = Math.max(size * 3.65, scale * 1.9);
      for (const sector of sectors) {
        const rawX = Math.sqrt(3) * (sector.q + sector.r / 2);
        const rawY = 1.5 * sector.r;
        const x = offsetX + rawX * scale;
        const y = offsetY + rawY * scale;
        drawSectorHex(context, x, y, sectorSize, SECTOR_FILLS[sector.position % SECTOR_FILLS.length]);
        context.fillStyle = "#7a847e";
        context.font = `700 ${compactMap ? 7 : 9}px Segoe UI`;
        context.textAlign = "center";
        context.fillText(`S${String(sector.tile).padStart(2, "0")} · ${sector.rotation}°`, x, y - sectorSize * 0.68);
      }
    }
  }

  for (const space of legalSpaceStations) {
    const x = offsetX + Math.sqrt(3) * (Number(space.q) + Number(space.r) / 2) * scale;
    const y = offsetY + 1.5 * Number(space.r) * scale;
    context.save();
    context.strokeStyle = "rgba(117, 202, 255, 0.92)";
    context.lineWidth = compactMap ? 1.6 : 2.3;
    context.setLineDash(compactMap ? [3, 2] : [5, 3]);
    drawHexOutline(context, x, y, size * 0.9);
    context.restore();
  }

  for (const planet of points) {
    const x = offsetX + planet.rawX * scale;
    const y = offsetY + planet.rawY * scale;
    const artworkDrawn = options.planetArtwork
      && drawPlanetArtwork(context, x, y, size, scale, planet, snapshot);
    if (!artworkDrawn) {
      drawHex(context, x, y, size, TERRAIN_COLORS[planet.terrain] || "#c7cec8");
      context.fillStyle = "rgba(255,255,255,0.86)";
      context.font = `700 ${Math.max(7, Math.min(9, size * 0.55))}px Segoe UI`;
      context.textAlign = "center";
      context.fillText(String(planet.id), x, y + 3);
    }
    if (options.showPlanetIds) {
      const badgeRadius = compactMap ? 5 : 6.5;
      const badgeX = x + size * 0.56;
      const badgeY = y + size * 0.56;
      context.fillStyle = "rgba(5, 11, 20, 0.88)";
      context.beginPath();
      context.arc(badgeX, badgeY, badgeRadius, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#e7f0f8";
      context.font = `700 ${compactMap ? 6 : 7}px Segoe UI`;
      context.textAlign = "center";
      context.fillText(String(planet.id), badgeX, badgeY + 2.4);
    }
    if (legalPlanetIds.has(Number(planet.id))) {
      context.save();
      context.strokeStyle = "rgba(117, 202, 255, 0.9)";
      context.lineWidth = compactMap ? 1.7 : 2.4;
      context.setLineDash(compactMap ? [3, 2] : [5, 3]);
      context.beginPath();
      context.arc(x, y, size * 1.18, 0, Math.PI * 2);
      context.stroke();
      context.restore();
    }
    if (
      options.selectedPlanetId !== null
      && options.selectedPlanetId !== undefined
      && Number(options.selectedPlanetId) === Number(planet.id)
    ) {
      context.strokeStyle = "#f2c85c";
      context.lineWidth = compactMap ? 2 : 3;
      context.beginPath();
      context.arc(x, y, size * 1.02, 0, Math.PI * 2);
      context.stroke();
    }
    if (options.showPlayerPieces !== false && planet.owner >= 0) {
      drawBuilding(
        context,
        planet.coexisting_mine_owner >= 0 ? x - size * 0.2 : x,
        planet.coexisting_mine_owner >= 0 ? y - size * 0.12 : y,
        planet.building,
        PLAYER_COLORS[planet.owner],
        planet.coexisting_mine_owner >= 0
          ? (compactMap ? 0.54 : 0.78)
          : (compactMap ? 0.68 : 1),
        planet.owner,
        size,
      );
    }
    if (options.showPlayerPieces !== false && planet.coexisting_mine_owner >= 0) {
      const coexistingX = x + size * 0.38;
      const coexistingY = y + size * 0.24;
      drawBuilding(
        context,
        coexistingX,
        coexistingY,
        "mine",
        PLAYER_COLORS[planet.coexisting_mine_owner],
        compactMap ? 0.42 : 0.58,
        planet.coexisting_mine_owner,
        size,
      );
      if (planet.coexisting_mine_federated) {
        context.save();
        context.strokeStyle = "rgba(255,255,255,0.82)";
        context.lineWidth = 1;
        context.setLineDash([2, 2]);
        context.beginPath();
        context.arc(coexistingX, coexistingY, size * 0.55, 0, Math.PI * 2);
        context.stroke();
        context.restore();
      }
    }
    if (options.showPlayerPieces !== false && planet.owner >= 0) {
      drawPlayerToken(context, x - size * 0.62, y - size * 0.62, planet.owner, size, compactMap);
    }
    if (options.showPlayerPieces !== false && planet.coexisting_mine_owner >= 0) {
      drawPlayerToken(
        context,
        x + size * 0.62,
        y - size * 0.62,
        planet.coexisting_mine_owner,
        size,
        compactMap,
      );
    }
    if (options.showPlayerPieces !== false && planet.gaiaformer >= 0 && planet.owner < 0) {
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
    if (options.showPlayerPieces !== false && planet.federated && planet.owner >= 0) {
      context.strokeStyle = "rgba(23,33,29,0.7)";
      context.lineWidth = 1;
      context.setLineDash([2, 3]);
      context.beginPath();
      context.arc(x, y, size + 9, 0, Math.PI * 2);
      context.stroke();
      context.setLineDash([]);
    }
  }

  if (options.showPlayerPieces !== false) {
    for (const satellite of snapshot.satellites || []) {
      const x = offsetX + Math.sqrt(3) * (Number(satellite.q) + Number(satellite.r) / 2) * scale;
      const y = offsetY + 1.5 * Number(satellite.r) * scale;
      const owners = Array.isArray(satellite.owners) ? satellite.owners : [];
      const markerSize = compactMap ? Math.max(4, size * 0.34) : Math.max(5.5, size * 0.38);
      owners.forEach((owner, index) => {
        const offset = (index - (owners.length - 1) / 2) * markerSize * 1.35;
        context.save();
        context.translate(x + offset, y);
        context.rotate(Math.PI / 4);
        context.fillStyle = PLAYER_COLORS[owner] || "#d9e2e8";
        context.strokeStyle = "rgba(5, 11, 20, 0.9)";
        context.lineWidth = compactMap ? 0.7 : 1;
        context.fillRect(-markerSize / 2, -markerSize / 2, markerSize, markerSize);
        context.strokeRect(-markerSize / 2, -markerSize / 2, markerSize, markerSize);
        context.restore();
      });
    }
    for (const station of snapshot.space_stations || []) {
      const x = offsetX + Math.sqrt(3) * (Number(station.q) + Number(station.r) / 2) * scale;
      const y = offsetY + 1.5 * Number(station.r) * scale;
      const stationDimension = compactMap ? size * 0.96 : size * 1.15;
      const stationImage = getMapPieceArtwork("icons");
      if (stationImage) {
        context.drawImage(
          stationImage,
          322,
          0,
          79,
          79,
          x - stationDimension / 2,
          y - stationDimension / 2,
          stationDimension,
          stationDimension,
        );
      } else {
        drawHex(context, x, y, stationDimension * 0.46, PLAYER_COLORS[station.owner] || "#b7202b");
        context.fillStyle = "#ffffff";
        context.font = `800 ${Math.max(6, stationDimension * 0.42)}px Segoe UI`;
        context.textAlign = "center";
        context.fillText("S", x, y + Math.max(2, stationDimension * 0.14));
      }
      if (station.federated) {
        context.save();
        context.strokeStyle = "rgba(255,255,255,0.85)";
        context.lineWidth = 1;
        context.setLineDash([2, 2]);
        context.beginPath();
        context.arc(x, y, stationDimension * 0.72, 0, Math.PI * 2);
        context.stroke();
        context.restore();
      }
    }
  }
}

function drawSectorHex(context, x, y, size, fill) {
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
  context.strokeStyle = "#c5cec7";
  context.lineWidth = 1.2;
  context.setLineDash([5, 4]);
  context.stroke();
  context.setLineDash([]);
}

function drawHexOutline(context, x, y, size) {
  context.beginPath();
  for (let side = 0; side < 6; side += 1) {
    const angle = Math.PI / 180 * (60 * side - 30);
    const px = x + size * Math.cos(angle);
    const py = y + size * Math.sin(angle);
    if (side === 0) context.moveTo(px, py); else context.lineTo(px, py);
  }
  context.closePath();
  context.stroke();
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

function drawBuilding(context, x, y, building, color, scale = 1, owner = 0, cellSize = 20) {
  const crop = BGA_STRUCTURE_CROPS[building];
  const image = crop ? getMapPieceArtwork("structures") : null;
  if (image) {
    const [sourceX, sourceY, sourceWidth, sourceHeight] = crop;
    const row = BGA_STRUCTURE_ROWS[Math.max(0, Number(owner) % BGA_STRUCTURE_ROWS.length)];
    const targetHeight = Math.max(8, cellSize * (BGA_STRUCTURE_HEIGHTS[building] || 1) * scale);
    const targetWidth = sourceWidth / sourceHeight * targetHeight;
    context.drawImage(
      image,
      sourceX,
      sourceY + row * 200,
      sourceWidth,
      sourceHeight,
      x - targetWidth / 2,
      y - targetHeight / 2,
      targetWidth,
      targetHeight,
    );
    return;
  }
  context.fillStyle = color;
  context.strokeStyle = "#ffffff";
  context.lineWidth = Math.max(1, 1.5 * scale);
  if (building === "mine") {
    context.beginPath();
    context.arc(x, y, 6 * scale, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  } else if (building === "trading_station") {
    context.fillRect(x - 7 * scale, y - 7 * scale, 14 * scale, 14 * scale);
    context.strokeRect(x - 7 * scale, y - 7 * scale, 14 * scale, 14 * scale);
  } else if (building === "research_lab") {
    context.beginPath();
    context.moveTo(x, y - 9 * scale);
    context.lineTo(x + 8 * scale, y + 7 * scale);
    context.lineTo(x - 8 * scale, y + 7 * scale);
    context.closePath();
    context.fill();
    context.stroke();
  } else if (building === "planetary_institute") {
    context.beginPath();
    context.moveTo(x, y - 10 * scale);
    context.lineTo(x + 10 * scale, y);
    context.lineTo(x, y + 10 * scale);
    context.lineTo(x - 10 * scale, y);
    context.closePath();
    context.fill();
    context.stroke();
  } else if (building === "academy") {
    context.fillRect(x - 9 * scale, y - 6 * scale, 18 * scale, 12 * scale);
    context.fillRect(x - 5 * scale, y - 10 * scale, 10 * scale, 20 * scale);
    context.strokeRect(x - 9 * scale, y - 6 * scale, 18 * scale, 12 * scale);
    context.strokeRect(x - 5 * scale, y - 10 * scale, 10 * scale, 20 * scale);
  }
}

function renderPlayers(snapshot) {
  byId("active-player-note").textContent = snapshot?.current_player === null || snapshot?.current_player === undefined
    ? "对局已结束"
    : `当前行动 P${snapshot.current_player}`;
  renderPlayerRows("players-table", snapshot);
  renderPersonalBoards("player-board-grid", snapshot);
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

function renderPersonalBoards(containerId, snapshot) {
  const container = byId(containerId);
  if (!container) return;
  const players = snapshot?.players || [];
  if (!players.length) {
    container.innerHTML = '<div class="personal-board-empty">暂无个人版图状态</div>';
    container.dataset.signature = "empty";
    return;
  }

  const signature = JSON.stringify({
    round: snapshot.round,
    current: snapshot.current_player,
    terminal: snapshot.terminal,
    players,
    planets: (snapshot.planets || []).map((planet) => [
      planet.id, planet.owner, planet.building, planet.q, planet.r, planet.gaiaformer,
      planet.coexisting_mine_owner, planet.coexisting_mine_federated,
    ]),
  });
  if (container.dataset.signature === signature) return;

  const techCatalog = new Map(
    (snapshot.setup?.standard_tech || []).map((tile) => [Number(tile.id), tile]),
  );
  const advancedTechCatalog = new Map(
    (snapshot.setup?.advanced_tech || []).map((tile) => [Number(tile.id), tile]),
  );
  const planets = snapshot.planets || [];
  const factionIdByName = new Map(BASE_FACTIONS.map((faction) => [faction.name, faction.id]));
  const resourceSpecs = [
    { key: "credits", label: "信用点", cap: 30 },
    { key: "ore", label: "矿石", cap: 15 },
    { key: "knowledge", label: "知识", cap: 15 },
    { key: "qic", label: "Q.I.C.", cap: null },
  ];
  const incomeSpecs = [
    { key: "credits", label: "信用点", tone: "credits" },
    { key: "ore", label: "矿石", tone: "ore" },
    { key: "knowledge", label: "知识", tone: "knowledge" },
    { key: "qic", label: "Q.I.C.", tone: "qic" },
    { key: "power_tokens", label: "能量片", tone: "power-token" },
    { key: "power_charge", label: "充能", tone: "power-charge" },
  ];

  container.innerHTML = players.map((player) => {
    const factionId = Number.isInteger(Number(player.faction_id))
      ? Number(player.faction_id)
      : (factionIdByName.get(player.faction) ?? 0);
    const faction = BASE_FACTIONS.find((item) => item.id === factionId) || BASE_FACTIONS[0];
    const colonies = planets.flatMap((planet) => {
      if (Number(planet.owner) === Number(player.id) && planet.building !== "empty") {
        return [planet];
      }
      if (Number(planet.coexisting_mine_owner) === Number(player.id)) {
        return [{
          ...planet,
          owner: player.id,
          building: "mine",
          federated: Boolean(planet.coexisting_mine_federated),
          coexisting: true,
        }];
      }
      return [];
    });
    const structureRows = BUILDING_SPECS.map((spec) => {
      const fallbackBuilt = colonies.filter((planet) => planet.building === spec.key).length;
      const recorded = player.structures?.[spec.key];
      const built = Number(recorded?.built ?? fallbackBuilt);
      const supply = Number(recorded?.supply ?? Math.max(0, spec.total - built));
      const slots = Array.from({ length: spec.total }, (_, index) => {
        const inSupply = index < supply;
        return `<i class="structure-slot ${inSupply ? "in-supply" : "deployed"}" title="${inSupply ? "版图库存" : "已部署到星图"}">
          <span class="structure-piece ${spec.key}"></span>
        </i>`;
      }).join("");
      return `<div class="structure-inventory-row">
        <div class="structure-name"><span class="structure-code">${spec.short}</span><strong>${spec.label}</strong></div>
        <div class="structure-slots" aria-label="${spec.label}库存 ${supply}，已部署 ${built}">${slots}</div>
        <span class="structure-count">库存 ${supply} · 星图 ${built}</span>
      </div>`;
    }).join("");
    const baseLocations = colonies.length
      ? colonies.map((planet) => {
        const spec = BUILDING_SPECS.find((item) => item.key === planet.building);
        const coexistence = planet.coexisting ? " · 共存" : "";
        return `<span class="base-location">#${planet.id} ${spec?.short || "--"}${coexistence} · ${planet.q},${planet.r}</span>`;
      }).join("")
      : '<span class="base-location empty">尚未部署基地</span>';
    const acquiredTech = Array.isArray(player.tech_tiles) ? player.tech_tiles : [];
    const acquiredAdvancedTech = Array.isArray(player.advanced_tech_tiles)
      ? player.advanced_tech_tiles
      : [];
    const coveredTech = new Set(
      Array.isArray(player.covered_tech_tiles)
        ? player.covered_tech_tiles.map(Number)
        : [],
    );
    const acquiredTechTotal = acquiredTech.length + acquiredAdvancedTech.length;
    const federationCounts = Array.from({ length: FEDERATION_NAMES.length }, (_, id) => {
      const localCount = Number(player.federation_tile_counts?.[id]);
      if (Number.isFinite(localCount) && localCount > 0) return localCount;
      return Array.isArray(player.federation_tiles)
        ? player.federation_tiles.filter((tileId) => Number(tileId) === id).length
        : 0;
    });
    const standardFederationTotal = federationCounts.reduce((total, count) => total + count, 0);
    const gleensFederationTotal = Number(player.gleens_federation_tokens || 0);
    const trackedFederationTotal = Number(player.federations);
    const federationTotal = Math.max(
      standardFederationTotal + gleensFederationTotal,
      Number.isFinite(trackedFederationTotal) ? trackedFederationTotal : 0,
    );
    const federationUnusedValue = Number(
      player.federation_unused ?? player.federation_keys,
    );
    const federationUnused = Number.isFinite(federationUnusedValue)
      ? Math.max(0, Math.min(federationTotal, federationUnusedValue))
      : federationTotal;
    const federationUsedValue = Number(player.federation_used);
    const federationUsed = Number.isFinite(federationUsedValue)
      ? Math.max(0, Math.min(federationTotal, federationUsedValue))
      : Math.max(0, federationTotal - federationUnused);
    const federationUsage = federationTotal
      ? `<div class="owned-federation-usage" aria-label="联邦板块使用状态">
          <div class="owned-federation-state unused" title="绿色面：尚未使用">
            <i class="owned-federation-face" aria-hidden="true"></i>
            <span>未使用</span><strong>${formatNumber(federationUnused)}</strong>
          </div>
          <div class="owned-federation-state used" title="灰色面：已经使用">
            <i class="owned-federation-face" aria-hidden="true"></i>
            <span>已使用</span><strong>${formatNumber(federationUsed)}</strong>
          </div>
        </div>`
      : "";
    const boosterId = Number(player.booster);
    const ownedBooster = Number.isInteger(boosterId) && boosterId >= 0
      ? `<div class="personal-booster-tile">
          <img src="${tileAsset("booster", boosterId)}" alt="${escapeHtml(BOOSTER_NAMES[boosterId] || `助推 ${boosterId + 1}`)}">
          <div><span>当前助推</span><strong>${escapeHtml(BOOSTER_NAMES[boosterId] || `助推 ${boosterId + 1}`)}</strong></div>
        </div>`
      : '<div class="personal-booster-empty">等待选择助推板块</div>';
    const techTiles = acquiredTechTotal
      ? [
        ...acquiredTech.map((tileId) => {
        const id = Number(tileId);
        const tile = techCatalog.get(id) || { id, key: STANDARD_TECH_KEYS[id], label: "" };
        const label = setupLabel(tile);
        const isCovered = coveredTech.has(id);
        return `<div class="owned-tech-tile standard ${isCovered ? "covered" : ""}" title="${escapeHtml(label)}${isCovered ? " · 已被高级科技覆盖" : ""}">
          <div class="owned-tech-art"><img src="${tileAsset("standard", id, tile.key)}" alt="${escapeHtml(label)}"></div>
          <strong>${escapeHtml(label)}</strong>${isCovered ? "<small>已覆盖</small>" : ""}
        </div>`;
        }),
        ...acquiredAdvancedTech.map((tileId) => {
          const id = Number(tileId);
          const tile = advancedTechCatalog.get(id) || { id, label: ADVANCED_TECH_NAMES[id] || "" };
          const label = setupLabel(tile);
          return `<div class="owned-tech-tile advanced" title="高级科技 · ${escapeHtml(label)}">
            <div class="owned-tech-art"><img src="${tileAsset("advanced", id, tile.key)}" alt="${escapeHtml(label)}"></div>
            <strong>${escapeHtml(label)}</strong><small>高级科技</small>
          </div>`;
        }),
      ].join("")
      : '<div class="owned-tech-empty">尚未获得科技</div>';
    const federationTileDetailsTotal = standardFederationTotal + gleensFederationTotal;
    const federationTiles = federationTileDetailsTotal
      ? [
        ...federationCounts.flatMap((count, id) => count > 0 ? [`<div class="owned-federation-tile" title="${escapeHtml(FEDERATION_NAMES[id])} × ${count}">
          <div class="owned-federation-art">
            <img src="${tileAsset("federation", id)}" alt="${escapeHtml(FEDERATION_NAMES[id])}">
            ${count > 1 ? `<b class="owned-federation-count">×${count}</b>` : ""}
          </div>
          <strong>${escapeHtml(FEDERATION_NAMES[id])}</strong>
        </div>`] : []),
        ...(gleensFederationTotal > 0 ? [`<div class="owned-federation-tile gleens" title="Gleens 专属联邦板块 × ${gleensFederationTotal}">
          <div class="owned-federation-art">
            <img src="/assets/tiles/federation-07.png" alt="Gleens 专属联邦板块">
            ${gleensFederationTotal > 1 ? `<b class="owned-federation-count">×${gleensFederationTotal}</b>` : ""}
          </div>
          <strong>Gleens 专属联邦</strong>
        </div>`] : []),
      ].join("")
      : federationTotal
        ? '<div class="owned-federation-empty">历史数据未记录板块种类</div>'
        : '<div class="owned-federation-empty">尚未获得联邦板块</div>';
    const power = Array.isArray(player.power) ? player.power : [0, 0, 0];
    const brainstoneBowl = Number(player.brainstone_bowl || 0);
    const powerAreas = [
      ["I", power[0], 1], ["II", power[1], 2], ["III", power[2], 3], ["盖亚区", player.gaia_power, 4],
    ].map(([label, value, bowl]) => `<div class="${brainstoneBowl === bowl ? "has-brainstone" : ""}"><span>${label}</span><strong>${formatNumber(value)}</strong>${brainstoneBowl === bowl ? '<small class="brainstone-marker">脑石</small>' : ""}</div>`).join("");
    const resources = resourceSpecs.map((resource) => `<div class="personal-resource">
      <span>${resource.label}</span><strong>${formatNumber(player[resource.key])}</strong>${resource.cap ? `<small>/ ${resource.cap}</small>` : ""}
    </div>`).join("");
    const income = player.round_income || {};
    const incomeItems = incomeSpecs.map((item) => {
      const value = Number(income[item.key]);
      const display = Number.isFinite(value) ? `+${formatNumber(value)}` : "--";
      return `<div class="personal-income-item ${item.tone}">
        <span>${item.label}</span><strong>${display}</strong>
      </div>`;
    }).join("");
    const status = snapshot.terminal
      ? "对局结束"
      : player.passed ? "已过轮" : player.id === snapshot.current_player ? "当前行动" : "等待行动";

    return `<article class="personal-board-card p${player.id} ${player.id === snapshot.current_player ? "active" : ""}">
      <div class="personal-board-layout">
        <div class="personal-board-surface">
          <header class="personal-board-heading">
            <div><span class="personal-board-seat"><i class="player-color p${player.id}"></i>P${player.id} · ${status}</span><h4>${escapeHtml(player.faction || `Faction ${factionId + 1}`)}</h4></div>
            <div class="personal-board-score"><span>VP</span><strong>${formatNumber(player.vp ?? snapshot.scores?.[player.id])}</strong></div>
          </header>
          <div class="personal-board-overview">
            <figure class="personal-board-artwork">
              <img class="personal-board-faction-art" src="${factionPlayerBoardAsset(factionId)}" alt="${escapeHtml(player.faction || "种族")}个人主板">
            </figure>
            <div class="personal-board-resources">
              <div class="personal-resource-grid">${resources}</div>
              <div class="personal-income-panel" aria-label="下一次收入阶段的回合收入">
                <div class="personal-income-heading"><strong>回合收入</strong><span>实时预览</span></div>
                <div class="personal-income-grid">${incomeItems}</div>
              </div>
              <div class="power-cycle" aria-label="能量碗 I、II、III 和盖亚区">${powerAreas}</div>
              <div class="personal-board-counters">
                <span>盖亚塑形者 <strong>${formatNumber(player.gaiaformers)}</strong></span>
                <span>场上塑形者 <strong>${formatNumber(player.gaiaformers_on_board ?? planets.filter((planet) => planet.gaiaformer === player.id).length)}</strong></span>
                ${Number(player.gaiaformers_in_gaia || 0) > 0 ? `<span>盖亚区塑形者 <strong>${formatNumber(player.gaiaformers_in_gaia)}</strong></span>` : ""}
                <span>联邦 <strong>${formatNumber(player.federations)}</strong></span>
                ${Number(player.gleens_federation_tokens || 0) > 0 ? `<span>GLE-FED <strong>${formatNumber(player.gleens_federation_tokens)}</strong></span>` : ""}
                <span>联邦门槛 <strong>${formatNumber(player.federation_threshold ?? 7)}</strong></span>
                <span>卫星 <strong>${formatNumber(player.satellites)}</strong></span>
                ${Number(player.space_stations || 0) > 0 ? `<span>空间站 <strong>${formatNumber(player.space_stations)}</strong></span>` : ""}
              </div>
            </div>
          </div>
          <section class="personal-board-section">
            <div class="personal-board-section-heading"><strong>建筑与基地</strong><span>${colonies.length} 处星图基地</span></div>
            <div class="structure-inventory">${structureRows}</div>
            <div class="base-location-list" aria-label="星图基地位置">${baseLocations}</div>
          </section>
        </div>
        <aside class="personal-tech-rack">
          ${ownedBooster}
          <div class="personal-tech-heading"><strong>已获科技</strong><span>${acquiredTechTotal} 块</span></div>
          <div class="owned-tech-grid">${techTiles}</div>
          <div class="personal-federation-heading"><strong>已获联邦板块</strong><span>${federationTotal} 块</span></div>
          ${federationUsage}
          <div class="owned-federation-grid">${federationTiles}</div>
        </aside>
      </div>
    </article>`;
  }).join("");
  container.dataset.signature = signature;
}

function drawLostPlanetMarker(context, x, y, color, scale = 1) {
  context.save();
  context.fillStyle = "#d3b65b";
  context.strokeStyle = "#fff4c2";
  context.lineWidth = Math.max(1, 1.5 * scale);
  context.beginPath();
  context.arc(x, y, 8 * scale, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.fillStyle = "#182333";
  context.font = `800 ${Math.max(7, 8 * scale)}px Segoe UI`;
  context.textAlign = "center";
  context.fillText("L", x, y + 2.7 * scale);
  context.fillStyle = color;
  context.strokeStyle = "#ffffff";
  context.fillRect(x + 5 * scale, y - 9 * scale, 7 * scale, 4 * scale);
  context.strokeRect(x + 5 * scale, y - 9 * scale, 7 * scale, 4 * scale);
  context.restore();
}

function renderResearchBoard(snapshot, scope = "play") {
  const stage = byId(`${scope}-research-stage`);
  const techLayer = byId(`${scope}-research-tech`);
  const federationLayer = byId(`${scope}-research-federation`);
  const markerLayer = byId(`${scope}-research-markers`);
  const legend = byId(`${scope}-research-player-legend`);
  const status = byId(`${scope}-research-status`);
  if (!stage || !techLayer || !federationLayer || !markerLayer || !legend || !status) return;

  const setup = snapshot?.setup;
  const players = snapshot?.players || [];
  if (!players.length) {
    status.textContent = "等待科研轨数据";
    techLayer.innerHTML = "";
    federationLayer.innerHTML = "";
    markerLayer.innerHTML = "";
    legend.innerHTML = "";
    return;
  }

  const standardTech = Array.isArray(setup?.standard_tech) ? setup.standard_tech : [];
  const advancedTech = Array.isArray(setup?.advanced_tech) ? setup.advanced_tech : [];
  const standardByTrack = new Map(standardTech.filter((tile) => tile.track).map((tile) => [tile.track, tile]));
  const advancedByTrack = new Map(advancedTech.map((tile) => [tile.track, tile]));
  const freeStandardTech = standardTech.filter((tile) => !tile.track);
  const techSignature = [
    ...standardTech.map((tile) => `s${tile.space}:${tile.id}`),
    ...advancedTech.map((tile, index) => `a${index}:${tile.id}`),
  ].join("|");
  if (techLayer.dataset.signature !== techSignature) {
    const trackTechSlots = Object.entries(TRACK_LABELS).flatMap(([track, label], trackIndex) => {
      const standard = standardByTrack.get(track);
      const advanced = advancedByTrack.get(track);
      const standardName = setupLabel(standard);
      const advancedName = ADVANCED_TECH_NAMES[advanced?.id] || advanced?.label || "--";
      return [
        advanced ? `<span class="research-tech-slot advanced track-${trackIndex}" role="img" tabindex="0" aria-label="${escapeHtml(label)}高级科技：${escapeHtml(advancedName)}" title="${escapeHtml(label)}高级科技 · ${escapeHtml(advancedName)}"><img src="${tileAsset("advanced", advanced.id, advanced.key)}" alt="" aria-hidden="true"></span>` : "",
        standard ? `<span class="research-tech-slot standard track-${trackIndex}" role="img" tabindex="0" aria-label="${escapeHtml(label)}基础科技：${escapeHtml(standardName)}" title="${escapeHtml(label)}基础科技 · ${escapeHtml(standardName)}"><img src="${tileAsset("standard", standard.id, standard.key)}" alt="" aria-hidden="true"></span>` : "",
      ].filter(Boolean);
    });
    const freeTechSlots = freeStandardTech.map((tile, index) => {
      const name = setupLabel(tile);
      return `<span class="research-tech-slot standard free-${index}" role="img" tabindex="0" aria-label="通用基础科技 ${index + 1}：${escapeHtml(name)}" title="通用基础科技 ${index + 1} · ${escapeHtml(name)}"><img src="${tileAsset("standard", tile.id, tile.key)}" alt="" aria-hidden="true"></span>`;
    });
    techLayer.innerHTML = [...trackTechSlots, ...freeTechSlots].join("");
    techLayer.dataset.signature = techSignature;
  }

  const federation = setup?.terraforming_federation;
  const federationId = Number(federation?.id);
  if (Number.isInteger(federationId) && federationId >= 0 && federationId < 6) {
    const federationName = FEDERATION_NAMES[federationId] || federation.label || `联邦 ${federationId + 1}`;
    if (federationLayer.dataset.signature !== String(federationId)) {
      federationLayer.innerHTML = `<span class="research-federation-tile" role="img" tabindex="0" aria-label="改造轨顶联邦板块：${escapeHtml(federationName)}" title="改造轨顶 · ${escapeHtml(federationName)}"><img src="${tileAsset("federation", federationId)}" alt="" aria-hidden="true"></span>`;
      federationLayer.dataset.signature = String(federationId);
    }
  } else {
    federationLayer.innerHTML = "";
    federationLayer.dataset.signature = "";
  }

  markerLayer.innerHTML = Object.entries(TRACK_LABELS).flatMap(([track, label], trackIndex) =>
    Array.from({ length: 6 }, (_, level) => {
      const atLevel = players.filter((player) => Number(player.tracks?.[trackIndex] || 0) === level);
      if (!atLevel.length) return "";
      const names = atLevel.map((player) => `P${player.id} ${player.faction || ""}`).join("、");
      return `<div class="research-board-position track-${trackIndex} level-${level}" aria-label="${escapeHtml(label)}等级 ${level}：${escapeHtml(names)}">${atLevel.map((player) => `<span class="research-marker p${player.id}" title="P${player.id} ${escapeHtml(player.faction || "")}">P${player.id}</span>`).join("")}</div>`;
    })
  ).join("");
  legend.innerHTML = players.map((player) => `<span><i class="player-mini p${player.id}"></i>P${player.id} ${escapeHtml(player.faction || "")}</span>`).join("");
  const active = snapshot.current_player === null || snapshot.current_player === undefined
    ? "对局已结束"
    : `当前行动 P${snapshot.current_player}`;
  const techStatus = standardTech.length || advancedTech.length
    ? ""
    : " · 科技板块编号未记录";
  status.textContent = `${active} · ${players.length} 位玩家${techStatus}`;
}

function renderLiveResearchBoard(snapshot) {
  renderResearchBoard(snapshot, "play");
}

function renderHistoryResearchBoard(snapshot) {
  renderResearchBoard(snapshot, "history");
}

const PLAY_ACTION_LABELS = {
  history_state: "历史状态快照",
  bga_state: "BGA 状态同步",
  starting_placement: "放置起始基地",
  select_booster: "选择起始助推板块",
  build: "建造矿场",
  gaia: "启动盖亚计划",
  upgrade_trading: "升级贸易站",
  firaks_downgrade: "Firaks：研究所降级为贸易站",
  bescods_research: "Bescods：推进最低科研轨",
  upgrade_lab: "升级研究所",
  upgrade_pi: "升级行星研究院",
  ambas_swap: "Ambas：交换行星研究院与矿场",
  upgrade_academy: "升级学院",
  upgrade_qic_academy: "升级 Q.I.C. 学院",
  upgrade_credits_academy: "升级信用点学院",
  research: "推进科研轨",
  skip_tech_research: "放弃科技板块的科研推进",
  power: "执行能量行动",
  brainstone: "选择脑石（按 3 能量）",
  passive_charge_accept: "接受被动充能",
  passive_charge_decline: "拒绝被动充能",
  taklons_passive_before: "Taklons 研究院：先获得能量片，再被动充能",
  taklons_passive_after: "Taklons 研究院：先被动充能，再获得能量片",
  ivits_space_station: "Ivits：放置空间站",
  lost_planet: "航行 5 级：放置失落星球",
  bal_taks_gaiaformer_qic: "Bal T'aks：盖亚塑形者兑换 Q.I.C.",
  itars_burn_power: "Itars：燃烧能量",
  itars_gaia_tech: "Itars 研究院：兑换科技板块",
  itars_gaia_finish: "结束 Itars 研究院结算",
  nevlas_power_to_gaia: "Nevlas：能量移入盖亚区换知识",
  nevlas_convert_credits: "Nevlas 自由兑换：1 能量兑换 2 信用点",
  nevlas_convert_credit_ore: "Nevlas 自由兑换：2 能量兑换 1 信用点和 1 矿石",
  nevlas_convert_ore: "Nevlas 自由兑换：3 能量兑换 2 矿石",
  nevlas_convert_qic: "Nevlas 自由兑换：2 能量兑换 1 Q.I.C.",
  nevlas_convert_knowledge: "Nevlas 自由兑换：2 能量兑换 1 知识",
  terrans_gaia_credit: "Terrans 盖亚兑换：信用点",
  terrans_gaia_ore: "Terrans 盖亚兑换：矿石",
  terrans_gaia_knowledge: "Terrans 盖亚兑换：知识",
  terrans_gaia_qic: "Terrans 盖亚兑换：Q.I.C.",
  terrans_gaia_finish: "结束 Terrans 盖亚兑换",
  hadsch_credit_ore: "Hadsch Hallas：3 信用点兑换 1 矿石",
  hadsch_credit_knowledge: "Hadsch Hallas：4 信用点兑换 1 知识",
  hadsch_credit_qic: "Hadsch Hallas：4 信用点兑换 1 Q.I.C.",
  technology: "获取科技板块",
  federation: "组建联邦",
  qic_academy_action: "使用 Q.I.C. 学院行动",
  credits_academy_action: "使用信用点学院行动",
  standard_tech_action: "使用基础科技行动",
  advanced_tech_action: "使用高级科技行动",
  qic_tech_action: "使用 Q.I.C. 科技行动",
  qic_federation_action: "使用 Q.I.C. 联邦行动",
  qic_planet_types_action: "使用 Q.I.C. 星球类型行动",
  booster_terraform_action: "使用助推地形改造行动",
  booster_range_action: "使用助推航程行动",
  pass_booster: "过轮并选择助推板块",
  pass_final: "最终过轮",
  other: "执行动作",
};

const PLAY_LOG_RESOURCE_LABELS = {
  credits: "信用点",
  ore: "矿石",
  knowledge: "知识",
  qic: "Q.I.C.",
  vp: "VP",
  power: "能量",
  power_tokens: "能量标记",
  power_to_gaia: "移入盖亚区",
  gaia_conversion_power: "盖亚兑换额度",
  gaia_power: "盖亚区能量",
  gaiaformers: "盖亚塑形者",
  federation_key: "未使用联邦片",
};

const PLAY_LOG_COUNTER_LABELS = {
  gaia_power: "盖亚区能量",
  gaiaformers: "可用盖亚塑形者",
  gaiaformers_in_gaia: "盖亚区塑形者",
  federation_tokens: "联邦板块",
  federation_keys: "未使用联邦片",
  federation_used: "已使用联邦片",
  gleens_federation_tokens: "Gleens 专属联邦板块",
  satellites: "卫星",
};

const PLAY_LOG_RELATION_LABELS = {
  selected: "选择",
  returned: "归还",
  gained: "获得",
  scored: "计分",
  income: "收入来源",
  round: "本轮",
  uses: "关联",
  spent: "支付",
};

function renderPlayLogComponents(components = []) {
  if (!components.length) return "";
  return `<div class="play-log-components">${components.map((component) => {
    const relation = PLAY_LOG_RELATION_LABELS[component.relation] || PLAY_LOG_RELATION_LABELS.uses;
    let label = component.label || component.code;
    if (component.kind === "booster") label = BOOSTER_NAMES[component.id] || label;
    if (component.kind === "standard_tech") label = SETUP_LABELS[STANDARD_TECH_KEYS[component.id]] || label;
    if (component.kind === "round_scoring") label = SETUP_LABELS[ROUND_SETUP_KEYS[component.id]] || label;
    if (component.kind === "federation") label = FEDERATION_NAMES[component.id] || label;
    if (component.kind === "research_track") label = TRACK_LABELS[TRACK_KEYS[component.id]] || label;
    if (component.kind === "planet") label = `星球 ${component.id}`;
    return `<span class="play-log-component kind-${escapeHtml(component.kind || "other")}" title="${escapeHtml(label)}"><i>${escapeHtml(relation)}</i><strong>${escapeHtml(component.code)}</strong></span>`;
  }).join("")}</div>`;
}

function renderPlayLogResources(label, items = [], tone = "gain") {
  if (!items.length) return "";
  const sign = tone === "cost" ? "−" : "+";
  return `<div class="play-log-resource-group ${tone}"><span>${label}</span><div>${items.map((item) => {
    const resource = PLAY_LOG_RESOURCE_LABELS[item.resource] || item.resource;
    return `<b>${sign}${formatNumber(item.amount)} ${escapeHtml(resource)}</b>`;
  }).join("")}</div></div>`;
}

function renderPlayLogChange(change) {
  if (change.kind === "power") {
    return `能量碗 ${change.before.join("/")} → ${change.after.join("/")}`;
  }
  if (change.kind === "brainstone") {
    const bowlLabel = (value) => ["不在循环", "I 区", "II 区", "III 区", "盖亚区"][value] || `区域 ${value}`;
    return `脑石 ${bowlLabel(change.before)} → ${bowlLabel(change.after)}`;
  }
  if (change.kind === "brainstone_selection") {
    return change.after ? "已选择脑石按 3 点能量支付" : "已取消或完成脑石支付";
  }
  if (change.kind === "gaia_conversion_budget") {
    return `盖亚兑换额度 ${change.before} → ${change.after}`;
  }
  if (change.kind === "counter") {
    return `${PLAY_LOG_COUNTER_LABELS[change.counter] || change.counter} ${change.before} → ${change.after}`;
  }
  if (change.kind === "track") {
    return `${TRACK_LABELS[TRACK_KEYS[change.track]] || `科研轨 ${change.track + 1}`} L${change.before} → L${change.after}`;
  }
  if (change.kind === "tech") return `获得基础科技 TEC-S${String(change.id + 1).padStart(2, "0")}`;
  if (change.kind === "booster") {
    const before = change.before >= 0 ? `BST-${String(change.before + 1).padStart(2, "0")}` : "无";
    const after = change.after >= 0 ? `BST-${String(change.after + 1).padStart(2, "0")}` : "无";
    return `助推板块 ${before} → ${after}`;
  }
  if (change.kind === "passed") return change.after ? "完成本轮过轮" : "重置为未过轮";
  if (change.kind === "building") {
    const building = BUILDING_SPECS.find((item) => item.key === change.building_after)?.label || change.building_after;
    return `星球 P-${change.planet}：P${change.owner_after} 放置${building}`;
  }
  if (change.kind === "coexisting_mine") {
    return `星球 P-${change.planet}：P${change.owner_after} 放置共存矿场`;
  }
  if (change.kind === "coexisting_federated") {
    return `星球 P-${change.planet}：P${change.owner} 的共存矿场${change.after ? "加入" : "离开"}联邦`;
  }
  if (change.kind === "gaiaformer") return `星球 P-${change.planet}：盖亚塑形者 ${change.before} → ${change.after}`;
  if (change.kind === "terrain") {
    return `星球 P-${change.planet}：${TERRAIN_LABELS[change.before] || change.before} → ${TERRAIN_LABELS[change.after] || change.after}`;
  }
  if (change.kind === "federated") return `${change.amount} 个建筑位置加入联邦`;
  return "状态已更新";
}

function renderPlayLogChanges(changes = []) {
  if (!changes.length) return "";
  return `<div class="play-log-changes">${changes.map((change) => `<span>${escapeHtml(renderPlayLogChange(change))}</span>`).join("")}</div>`;
}

function renderPlayLogEffect(effect, system = false) {
  const resources = [
    renderPlayLogResources("开销", effect.costs, "cost"),
    renderPlayLogResources(system ? "回合收入" : "获得", effect.gains, "gain"),
  ].join("");
  const changes = renderPlayLogChanges(effect.changes);
  const empty = !resources && !changes ? '<span class="play-log-no-change">无资源变化</span>' : "";
  return `<div class="play-log-effect">
    <div class="play-log-effect-player"><i class="player-color p${effect.player}"></i><strong>P${effect.player}</strong></div>
    <div class="play-log-effect-detail">${resources}${changes}${renderPlayLogComponents(effect.sources)}${empty}</div>
  </div>`;
}

function renderPlayAutomaticStep(step) {
  const phaseLabel = step.gaia_phase ? "自动收入与盖亚阶段" : "自动收入";
  return `<section class="play-log-system-step">
    <header><span>系统步骤</span><strong>第 ${step.round} 大轮${phaseLabel}</strong></header>
    ${renderPlayLogComponents(step.components)}
    <div class="play-log-effects">${(step.effects || []).map((effect) => renderPlayLogEffect(effect, true)).join("")}</div>
    ${renderPlayLogChanges(step.changes)}
  </section>`;
}

function renderPlayActionEntry(item) {
  const actionName = ["bga_state", "history_state"].includes(item.kind) && (item.label || item.action_label)
    ? (item.label || item.action_label)
    : (PLAY_ACTION_LABELS[item.kind] || item.label || item.action_label || PLAY_ACTION_LABELS.other);
  const round = item.round > 0 ? `第 ${item.round} 轮` : "初始设置";
  const player = item.player === null || item.player === undefined ? null : Number(item.player);
  const actor = player === null
    ? "系统"
    : `<i class="player-color p${player}"></i>P${player}`;
  const role = item.role === "human" ? "人工" : item.role === "ai" ? "AI" : item.role === "bga" ? "BGA" : "系统";
  return `<article class="play-log-entry">
    <header class="play-log-entry-heading">
      <span class="play-log-move">#${item.move}</span>
      <div><strong>${actor} · ${escapeHtml(actionName)}</strong><small>${round} · ${role}</small></div>
    </header>
    ${renderPlayLogComponents(item.components)}
    <div class="play-log-effects">${(item.effects || []).map((effect) => renderPlayLogEffect(effect)).join("")}</div>
    ${renderPlayLogChanges(item.changes)}
    ${(item.automatic_steps || []).map(renderPlayAutomaticStep).join("")}
  </article>`;
}

function renderHistoryVpChanges(vp = {}) {
  const before = Array.isArray(vp.before) ? vp.before : [];
  const after = Array.isArray(vp.after) ? vp.after : [];
  const events = Array.isArray(vp.events) ? vp.events : [];
  const changes = after.map((value, player) => ({
    player,
    before: Number(before[player]),
    after: Number(value),
  })).filter((item) => Number.isFinite(item.before) && item.after !== item.before);
  if (!changes.length && !events.length) return "";
  const totals = changes.map((item) => {
    const delta = item.after - item.before;
    return `<span><i class="player-color p${item.player}"></i>P${item.player} VP ${formatNumber(item.before, 1)} → ${formatNumber(item.after, 1)} <b>${delta >= 0 ? "+" : ""}${formatNumber(delta, 1)}</b></span>`;
  }).join("");
  const details = events.map((event) => {
    const delta = Number(event.delta || 0);
    const source = event.source ? ` · ${event.source}` : "";
    return `<span class="history-vp-event"><i class="player-color p${Number(event.player)}"></i>P${Number(event.player)} ${escapeHtml(event.reason || "BGA 计分")}<small>${escapeHtml(source)}</small><b>${delta >= 0 ? "+" : ""}${formatNumber(delta, 1)} VP</b></span>`;
  }).join("");
  const auditFailed = vp.audit && vp.audit.matches_state === false;
  return `<div class="history-log-vp">
    ${totals}
    ${details ? `<div class="history-vp-events">${details}</div>` : ""}
    ${auditFailed ? '<em class="history-vp-audit-error">VP 明细与局面变化不一致</em>' : ""}
  </div>`;
}

function renderHistoryActionEntry(step, previousStep, index, currentStep) {
  const record = step.record || {};
  const item = {
    ...record,
    move: Number(step.move ?? index),
    player: step.player ?? record.player ?? null,
    role: record.role || step.role || (step.record ? "system" : null),
    round: Number(record.round ?? step.state?.round ?? 0),
    kind: record.kind || "history_state",
    label: record.label || step.action_label || "状态快照",
    components: record.components || [],
    effects: record.effects || [],
    changes: record.changes || [],
    automatic_steps: record.automatic_steps || [],
  };
  const scores = (step.state?.scores || []).map((value) => formatNumber(value, 1)).join(" / ");
  const notifications = record.bga?.notifications || [];
  const notificationTypes = [...new Set(notifications.map((notice) => notice.type).filter(Boolean))];
  const metadata = [
    step.action === null || step.action === undefined ? null : `动作 ${step.action}`,
    scores ? `局面分数 ${scores}` : null,
    notifications.length ? `${notifications.length} 条 BGA 通知` : null,
  ].filter(Boolean);
  const hasStructuredChanges = item.effects.length
    || item.changes.length
    || item.automatic_steps.length;
  const fallbackDelta = hasStructuredChanges
    ? ""
    : historyDelta(previousStep?.state, step.state, step);
  return `<div class="history-action-entry ${index === currentStep ? "current-step" : ""}" data-step="${index}" tabindex="0" role="button" aria-label="跳转到第 ${item.move} 步">
    ${renderPlayActionEntry(item)}
    ${fallbackDelta ? `<div class="history-log-delta">${escapeHtml(fallbackDelta)}</div>` : ""}
    ${renderHistoryVpChanges(record.vp)}
    ${notificationTypes.length ? `<div class="history-log-notifications">${notificationTypes.slice(0, 6).map((type) => `<span>${escapeHtml(type)}</span>`).join("")}</div>` : ""}
    ${metadata.length ? `<div class="history-log-meta">${metadata.map((value) => `<span>${escapeHtml(value)}</span>`).join("")}</div>` : ""}
  </div>`;
}

function ensurePlaySeats(players = state.play.players) {
  state.play.players = players;
  const defaults = [0, 2, 4, 6];
  state.play.factions = Array.from({ length: players }, (_, index) =>
    Number.isInteger(state.play.factions[index]) ? state.play.factions[index] : defaults[index]
  );
  state.play.roles = Array.from({ length: players }, (_, index) =>
    ["human", "ai"].includes(state.play.roles[index])
      ? state.play.roles[index]
      : (index === 0 ? "human" : "ai")
  );
}

function playRoleSegment(role, player, live = false, disabled = false) {
  const scope = live ? "live" : "config";
  return `<div class="play-role-segment" role="group" aria-label="P${player} 控制方式">
    <button type="button" data-${scope}-role="human" data-player="${player}" class="${role === "human" ? "active" : ""}" ${disabled ? "disabled" : ""}>人工</button>
    <button type="button" data-${scope}-role="ai" data-player="${player}" class="${role === "ai" ? "active" : ""}" ${disabled ? "disabled" : ""}>AI</button>
  </div>`;
}

function renderPlayConfig() {
  const players = Number(byId("setup-editor-players").value || 2);
  ensurePlaySeats(players);
  state.play.factions = Array.from(
    { length: players },
    (_, player) => Number(state.manualSetup.factions?.[player] ?? [0, 2, 4, 6][player]),
  );
  const seed = Number(byId("setup-editor-seed").value || 0);
  const firstPlayer = Number(byId("setup-editor-first-player").value || 0);
  byId("play-config-players-summary").textContent = `${players} 人`;
  byId("play-config-seed-summary").textContent = String(seed);
  byId("play-config-first-player-summary").textContent = `P${firstPlayer}`;
  byId("play-config-seats").innerHTML = state.play.factions.map((selected, player) => {
    const faction = BASE_FACTIONS.find((item) => item.id === selected) || BASE_FACTIONS[0];
    return `<article class="play-seat-config">
      <div class="play-seat-heading"><strong><i class="player-color p${player}"></i>P${player}</strong><span>版图 ${faction.board}${faction.side}</span></div>
      <div class="play-seat-faction-summary"><span>种族</span><strong>${escapeHtml(faction.name)}</strong></div>
      ${playRoleSegment(state.play.roles[player], player)}
    </article>`;
  }).join("");
  const message = byId("play-config-message");
  message.textContent = state.play.message;
  message.className = `play-message ${state.play.messageStatus}`;
  const disabled = state.play.requestBusy || Boolean(state.play.session?.busy);
  byId("play-start").disabled = disabled;
  byId("play-back-to-setup").disabled = disabled;
  byId("play-config-form").setAttribute("aria-busy", String(disabled));
}

function playConfigPayload() {
  const config = manualSetupPayload();
  const simulations = Number(byId("play-config-simulations").value);
  if (!Number.isInteger(simulations) || simulations < 1 || simulations > 128) {
    throw new Error("AI 每步搜索次数必须是 1–128 的整数");
  }
  return {
    ...config,
    simulations,
    roles: [...state.play.roles],
  };
}

function setPlayMessage(message, status = "ready") {
  state.play.message = message;
  state.play.messageStatus = status;
  const element = byId("play-config-message");
  if (element) {
    element.textContent = message;
    element.className = `play-message ${status}`;
  }
}

function switchPlayWorkspace(workspace) {
  state.play.workspace = workspace === "match" ? "match" : "setup";
  byId("play-setup-workspace").hidden = state.play.workspace !== "setup";
  byId("play-match-workspace").hidden = state.play.workspace !== "match";
  document.querySelectorAll("[data-play-workspace]").forEach((button) => {
    const active = button.dataset.playWorkspace === state.play.workspace;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  requestAnimationFrame(() => {
    if (state.play.workspace === "setup") {
      renderSetup(state.manualSetup.preview);
      renderPlanetPositionEditor();
    } else {
      renderPlay();
    }
  });
}

async function prepareInteractiveMatch() {
  state.manualSetup.busy = true;
  setSetupEditorMessage("正在验证人工对局初始设置", "running");
  try {
    await previewManualSetup({ quiet: true });
    ensurePlaySeats(Number(byId("setup-editor-players").value));
    setSetupEditorMessage("初始设置已确认，可配置人工与 AI 角色", "complete");
    setPlayMessage("初始盘面已载入", "ready");
    switchPlayWorkspace("match");
  } catch (error) {
    setSetupEditorMessage(error.message || String(error), "failed");
  } finally {
    state.manualSetup.busy = false;
    renderManualSetupStatus();
  }
}

async function postPlay(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function acceptPlaySession(session) {
  state.play.session = session?.status === "idle" ? null : session;
  if (state.play.session) state.play.workspace = "match";
  if (state.play.session?.archive_error) {
    setPlayMessage(`本地历史保存失败：${state.play.session.archive_error}`, "failed");
  }
  const legalTargets = new Set(
    (state.play.session?.legal_actions || [])
      .filter((action) => action.target !== null && action.target !== undefined)
      .map((action) => Number(action.target)),
  );
  if (state.play.selectedPlanetId !== null && !legalTargets.has(state.play.selectedPlanetId)) {
    state.play.selectedPlanetId = null;
  }
  renderPlay();
}

async function startInteractiveGame() {
  state.play.requestEpoch += 1;
  state.play.requestBusy = true;
  setPlayMessage("正在创建初始盘面", "running");
  renderPlay();
  try {
    const session = await postPlay("/api/play/start", playConfigPayload());
    state.play.selectedPlanetId = null;
    setPlayMessage("对局已开始", "complete");
    acceptPlaySession(session);
  } catch (error) {
    setPlayMessage(error.message || String(error), "failed");
  } finally {
    state.play.requestBusy = false;
    renderPlay();
  }
}

async function submitHumanAction(action) {
  if (state.play.requestBusy) return;
  state.play.requestEpoch += 1;
  state.play.requestBusy = true;
  setPlayMessage("正在提交人工动作", "running");
  renderPlay();
  try {
    const session = await postPlay("/api/play/action", { action: Number(action) });
    setPlayMessage("人工动作已执行", "complete");
    acceptPlaySession(session);
  } catch (error) {
    setPlayMessage(error.message || String(error), "failed");
  } finally {
    state.play.requestBusy = false;
    renderPlay();
  }
}

async function runInteractiveAiTurn() {
  if (state.play.requestBusy || state.play.session?.busy) return;
  state.play.requestEpoch += 1;
  state.play.requestBusy = true;
  setPlayMessage("AI 正在运行 PIMCTS 搜索", "running");
  renderPlay();
  try {
    const session = await postPlay("/api/play/ai");
    setPlayMessage("AI 动作已执行", "complete");
    acceptPlaySession(session);
  } catch (error) {
    state.play.autoAi = false;
    byId("play-auto-ai").checked = false;
    setPlayMessage(error.message || String(error), "failed");
  } finally {
    state.play.requestBusy = false;
    renderPlay();
  }
}

async function undoInteractiveTurn() {
  const session = state.play.session;
  if (!session?.can_undo || state.play.requestBusy || session.busy) return;
  state.play.requestEpoch += 1;
  state.play.requestBusy = true;
  state.play.autoAi = false;
  byId("play-auto-ai").checked = false;
  setPlayMessage("正在撤销最近一次人工操作", "running");
  renderPlay();
  try {
    const updated = await postPlay("/api/play/undo");
    state.play.selectedPlanetId = null;
    const count = Number(updated.undone_actions || 1);
    setPlayMessage(`已撤销 ${count} 步${count > 1 ? "（含后续 AI 动作）" : ""}`, "complete");
    acceptPlaySession(updated);
  } catch (error) {
    setPlayMessage(error.message || String(error), "failed");
  } finally {
    state.play.requestBusy = false;
    renderPlay();
  }
}

async function updateLivePlayRole(player, role) {
  const session = state.play.session;
  if (!session || state.play.requestBusy || session.busy) return;
  const roles = [...session.roles];
  roles[player] = role;
  state.play.requestEpoch += 1;
  state.play.requestBusy = true;
  setPlayMessage(`正在将 P${player} 切换为${role === "human" ? "人工" : "AI"}`, "running");
  renderPlay();
  try {
    const updated = await postPlay("/api/play/roles", { roles });
    setPlayMessage(`P${player} 已切换为${role === "human" ? "人工" : "AI"}`, "complete");
    acceptPlaySession(updated);
  } catch (error) {
    setPlayMessage(error.message || String(error), "failed");
  } finally {
    state.play.requestBusy = false;
    renderPlay();
  }
}

async function pollInteractiveGame() {
  if (state.play.polling || state.play.requestBusy) return;
  state.play.polling = true;
  const requestEpoch = state.play.requestEpoch;
  try {
    const response = await fetch("/api/play", { cache: "no-store" });
    if (!response.ok) return;
    const session = await response.json();
    if (requestEpoch !== state.play.requestEpoch) return;
    const current = state.play.session;
    if (
      current
      && session.session_id === current.session_id
      && Number(session.revision) < Number(current.revision)
    ) return;
    const changed = session.status !== "idle" && (
      !current
      || session.session_id !== current.session_id
      || session.revision !== current.revision
      || session.busy !== current.busy
    );
    if (changed || (session.status === "idle" && state.play.session)) acceptPlaySession(session);
  } catch (_error) {
    // Training telemetry owns the global connectivity indicator.
  } finally {
    state.play.polling = false;
  }
}

function scheduleInteractiveAi() {
  const session = state.play.session;
  const shouldRun = state.play.autoAi
    && session?.status === "active"
    && session.current_role === "ai"
    && !session.busy
    && !state.play.requestBusy;
  if (!shouldRun) {
    if (state.play.aiTimer) window.clearTimeout(state.play.aiTimer);
    state.play.aiTimer = null;
    return;
  }
  if (state.play.aiTimer) return;
  state.play.aiTimer = window.setTimeout(() => {
    state.play.aiTimer = null;
    runInteractiveAiTurn();
  }, 350);
}

function playPhaseLabel(snapshot) {
  if (!snapshot) return "--";
  if (snapshot.terminal) return "对局结束";
  if (snapshot.phase === "starting_placement") {
    return `起始基地 ${snapshot.placement.step + 1}/${snapshot.placement.total}`;
  }
  if (snapshot.phase === "booster_selection") {
    return `选择助推 ${snapshot.booster_selection.step + 1}/${snapshot.booster_selection.total}`;
  }
  if (snapshot.phase === "passive_charge") {
    const charge = snapshot.passive_charge || {};
    return `被动充能：范围内最高建筑强度 ${charge.structure_power ?? 0}，可充 ${charge.chargeable ?? 0} 点，支付 ${charge.vp_cost ?? 0} VP`;
  }
  if (snapshot.phase === "taklons_passive_charge") {
    return `Taklons 研究院：选择被动充能顺序（${snapshot.taklons_passive_charge?.amount ?? 0} 点）`;
  }
  if (snapshot.phase === "gaia_conversion") {
    return `Terrans 盖亚兑换 · 剩余 ${snapshot.gaia_conversion?.remaining_power ?? 0}`;
  }
  if (snapshot.phase === "lost_planet_placement") {
    return "航行 5 级：放置失落星球";
  }
  if (snapshot.phase === "itars_gaia_technology") {
    return `Itars 研究院科技兑换 · 剩余 ${snapshot.itars_gaia_technology?.remaining_power ?? 0}`;
  }
  return `第 ${snapshot.round}/${snapshot.max_rounds} 轮`;
}

function renderPlayActions(session) {
  const actions = session.legal_actions || [];
  const humanTurn = session.current_role === "human" && session.status === "active";
  const disabled = !humanTurn || state.play.requestBusy || session.busy;
  const targeted = actions.filter((action) => action.kind !== "select_booster" && action.target !== null && action.target !== undefined);
  const visibleTargeted = state.play.selectedPlanetId === null
    ? targeted
    : targeted.filter((action) => Number(action.target) === state.play.selectedPlanetId);
  const general = actions.filter((action) => action.kind !== "select_booster" && (action.target === null || action.target === undefined));
  const boosterActions = actions.filter((action) => action.kind === "select_booster");
  const otherGeneral = general.filter(
    (action) => action.kind !== "select_booster" && action.kind !== "lost_planet",
  );
  const actionButton = (action) => `<button type="button" class="play-action-command" data-play-action="${action.id}" ${disabled ? "disabled" : ""}>
    <strong>${escapeHtml(PLAY_ACTION_LABELS[action.kind] || PLAY_ACTION_LABELS.other)}${action.target === null || action.target === undefined ? "" : ` · #${action.target}`}</strong>
    <small>${escapeHtml(action.label)}</small>
  </button>`;
  byId("play-planet-actions").innerHTML = visibleTargeted.length
    ? visibleTargeted.map(actionButton).join("")
    : `<div class="play-action-empty">${state.play.selectedPlanetId === null ? "当前没有星球动作" : "所选星球当前没有合法动作"}</div>`;
  byId("play-general-actions").classList.toggle("booster-choice-grid", boosterActions.length > 0);
  byId("play-general-actions").innerHTML = boosterActions.length
    ? boosterActions.map((action) => {
        const resolvedBooster = Number(action.booster);
        const label = BOOSTER_NAMES[resolvedBooster] || action.label;
        return `<button type="button" class="play-booster-choice" data-play-action="${action.id}" ${disabled ? "disabled" : ""}>
          <img src="${tileAsset("booster", resolvedBooster)}" alt="${escapeHtml(label)}">
          <span>助推 ${resolvedBooster + 1}</span><strong>${escapeHtml(label)}</strong>
        </button>`;
      }).join("")
    : otherGeneral.length
    ? otherGeneral.map(actionButton).join("")
    : '<div class="play-action-empty">当前没有其他动作</div>';
  byId("play-legal-count").textContent = `${actions.length} 项`;
  byId("play-selected-planet").textContent = state.play.selectedPlanetId === null
    ? "未选择（显示全部合法目标）"
    : `#${state.play.selectedPlanetId}`;
  const notice = byId("play-turn-notice");
  if (session.status === "complete") {
    notice.textContent = `对局结束 · ${session.state.scores.map((score) => formatNumber(score, 1)).join(" / ")} VP`;
    notice.className = "play-turn-notice";
  } else if (session.busy || state.play.requestBusy) {
    notice.textContent = "正在处理当前动作…";
    notice.className = "play-turn-notice ai";
  } else if (humanTurn) {
    notice.textContent = session.state.phase === "booster_selection"
      ? `P${session.state.current_player} 由人工选择一块起始助推板块`
      : session.state.phase === "passive_charge"
      ? `P${session.state.current_player} 决定是否接受 ${session.state.passive_charge?.chargeable ?? 0} 点被动充能（范围内最高建筑强度 ${session.state.passive_charge?.structure_power ?? 0}，支付 ${session.state.passive_charge?.vp_cost ?? 0} VP）`
      : session.state.phase === "taklons_passive_charge"
      ? `P${session.state.current_player} 选择 Taklons 研究院的被动充能顺序`
      : session.state.phase === "gaia_conversion"
      ? `P${session.state.current_player} 结算 Terrans 盖亚兑换：剩余 ${session.state.gaia_conversion?.remaining_power ?? 0} 点`
      : session.state.phase === "lost_planet_placement"
      ? `P${session.state.current_player} 放置失落星球`
      : session.state.phase === "itars_gaia_technology"
      ? `P${session.state.current_player} 结算 Itars 研究院：每 4 个盖亚区能量片可获得 1 块科技板块`
      : `P${session.state.current_player} 由人工操作：点击星图筛选目标，再选择合法动作`;
    notice.className = "play-turn-notice human";
  } else {
    notice.textContent = `P${session.state.current_player} 由 AI 操作${state.play.autoAi ? "，将自动执行" : "，可手动单步"}`;
    notice.className = "play-turn-notice ai";
  }
}

function renderPlaySearch(session) {
  const search = session.last_search;
  const candidates = search?.candidates || [];
  byId("play-search-candidates").innerHTML = candidates.length
    ? candidates.map((candidate) => `<div class="candidate-row">
        <div><strong>${escapeHtml(candidate.label)}</strong><small>${formatNumber(candidate.visits)} visits</small><progress class="mini-track" max="1" value="${Number(candidate.probability)}"></progress></div>
        <div class="candidate-probability">${formatNumber(Number(candidate.probability) * 100, 1)}%</div>
      </div>`).join("")
    : '<div class="candidate-empty">暂无 AI 搜索</div>';
  byId("play-root-values").innerHTML = search?.root_value
    ? search.root_value.map((value, player) => `<span class="value-pill">P${player} ${formatNumber(value, 2)}</span>`).join("")
    : "--";
}

function renderPlay() {
  byId("play-setup-workspace").hidden = state.play.workspace !== "setup";
  byId("play-match-workspace").hidden = state.play.workspace !== "match";
  document.querySelectorAll("[data-play-workspace]").forEach((button) => {
    const active = button.dataset.playWorkspace === state.play.workspace;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  renderPlayConfig();
  const session = state.play.session;
  byId("play-empty").hidden = Boolean(session);
  byId("play-content").hidden = !session;
  if (!session) {
    scheduleInteractiveAi();
    return;
  }
  const snapshot = session.state;
  const legalPlanetIds = (session.legal_actions || [])
    .filter((action) => action.target !== null && action.target !== undefined)
    .map((action) => Number(action.target));
  const legalSpaceStations = (session.legal_actions || [])
    .filter((action) => ["ivits_space_station", "lost_planet"].includes(action.kind))
    .map((action) => ({ q: Number(action.space_q), r: Number(action.space_r) }))
    .filter((space) => Number.isFinite(space.q) && Number.isFinite(space.r));
  const mapView = state.history.mapView;
  const playMapFrame = byId("play-board-canvas")?.parentElement;
  if (playMapFrame) playMapFrame.style.backgroundColor = mapView.background;
  drawStarMapBoard(byId("play-board-canvas"), snapshot, true, {
    legalPlanetIds,
    legalSpaceStations,
    selectedPlanetId: state.play.selectedPlanetId,
    showPlanetIds: true,
    zoom: mapView.zoom,
    gap: 0,
    backgroundColor: mapView.background,
  });
  byId("play-board-empty").hidden = Boolean(snapshot?.planets?.length);
  byId("play-board-round").textContent = playPhaseLabel(snapshot);
  byId("play-phase").textContent = playPhaseLabel(snapshot);
  byId("play-current-player").textContent = snapshot.current_player === null ? "--" : `P${snapshot.current_player}`;
  byId("play-current-role").textContent = session.current_role === "human" ? "人工" : session.current_role === "ai" ? "AI" : "--";
  byId("play-move-count").textContent = formatNumber(session.move);
  byId("play-ai-engine").textContent = session.ai_engine || "--";
  byId("play-live-roles").innerHTML = session.roles.map((role, player) => {
    const faction = snapshot.players?.[player]?.faction || `P${player}`;
    return `<article class="play-live-role ${player === snapshot.current_player ? "current" : ""}">
      <div class="play-live-role-heading"><strong><i class="player-color p${player}"></i>P${player}</strong><span>${escapeHtml(faction)}</span></div>
      ${playRoleSegment(role, player, true, state.play.requestBusy || session.busy)}
    </article>`;
  }).join("");
  byId("play-auto-ai").checked = state.play.autoAi;
  byId("play-ai-step").disabled = session.status !== "active"
    || session.current_role !== "ai"
    || state.play.requestBusy
    || session.busy;
  byId("play-undo").disabled = !session.can_undo
    || state.play.requestBusy
    || session.busy;
  byId("play-undo").title = session.undo_count > 1
    ? `撤销最近一次人工操作及其后的 ${session.undo_count - 1} 步 AI 动作`
    : "撤销最近一次人工操作";
  renderPlayActions(session);
  renderPlaySearch(session);
  renderLiveResearchBoard(snapshot);
  renderPlayerRows("play-players-table", snapshot, "play-active-player");
  renderPersonalBoards("play-player-board-grid", snapshot);
  const history = session.history || [];
  const automaticSteps = history.reduce((total, item) => total + (item.automatic_steps?.length || 0), 0);
  byId("play-action-log-count").textContent = automaticSteps
    ? `${history.length} 个动作 · ${automaticSteps} 个系统步骤`
    : `${history.length} 个动作`;
  byId("play-action-log").innerHTML = history.length
    ? [...history].reverse().map(renderPlayActionEntry).join("")
    : '<div class="play-action-log-empty">暂无动作</div>';
  scheduleInteractiveAi();
}

function planetAtPlayEvent(event) {
  const snapshot = state.play.session?.state;
  const canvas = byId("play-board-canvas");
  if (!snapshot?.planets?.length || !canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const clickX = event.clientX - rect.left;
  const clickY = event.clientY - rect.top;
  const geometry = boardGeometry(rect.width, rect.height, snapshot, true);
  const nearest = snapshot.planets.map((planet) => {
    const x = geometry.offsetX + Math.sqrt(3) * (Number(planet.q) + Number(planet.r) / 2) * geometry.scale;
    const y = geometry.offsetY + 1.5 * Number(planet.r) * geometry.scale;
    return { id: Number(planet.id), distance: Math.hypot(clickX - x, clickY - y) };
  }).sort((left, right) => left.distance - right.distance)[0];
  return nearest && nearest.distance <= Math.max(14, geometry.size * 1.35) ? nearest.id : null;
}

function spaceActionAtPlayEvent(event, kind) {
  const session = state.play.session;
  const snapshot = session?.state;
  const canvas = byId("play-board-canvas");
  if (
    !snapshot?.planets?.length
    || !canvas
    || session.current_role !== "human"
    || session.status !== "active"
    || state.play.requestBusy
    || session.busy
  ) return null;
  const candidates = (session.legal_actions || []).filter(
    (action) => action.kind === kind
      && Number.isFinite(Number(action.space_q))
      && Number.isFinite(Number(action.space_r)),
  );
  if (!candidates.length) return null;
  const rect = canvas.getBoundingClientRect();
  const clickX = event.clientX - rect.left;
  const clickY = event.clientY - rect.top;
  const geometry = boardGeometry(rect.width, rect.height, snapshot, true);
  const nearest = candidates.map((action) => {
    const q = Number(action.space_q);
    const r = Number(action.space_r);
    const x = geometry.offsetX + Math.sqrt(3) * (q + r / 2) * geometry.scale;
    const y = geometry.offsetY + 1.5 * r * geometry.scale;
    return { id: Number(action.id), distance: Math.hypot(clickX - x, clickY - y) };
  }).sort((left, right) => left.distance - right.distance)[0];
  return nearest && nearest.distance <= Math.max(14, geometry.size * 1.35)
    ? nearest.id
    : null;
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
    ["architecture", "网络架构"], ["hidden_size", "隐藏层"], ["residual_blocks", "残差块"], ["learning_rate", "学习率"],
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
  const selected = ["overview", "play", "selfplay", "history", "bga-import", "diagnostics"].includes(name) ? name : "overview";
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
  if (selected === "history") void refreshHistoryIndex();
  requestAnimationFrame(() => {
    renderLossChart();
    renderSetup(state.manualSetup.preview);
    renderPlay();
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
document.querySelectorAll("[data-play-workspace]").forEach((button) => {
  button.addEventListener("click", () => switchPlayWorkspace(button.dataset.playWorkspace));
});
byId("play-config-seats").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-config-role]");
  if (!button) return;
  state.play.roles[Number(button.dataset.player)] = button.dataset.configRole;
  setPlayMessage("座位控制方式已更新", "ready");
  renderPlayConfig();
});
byId("play-config-form").addEventListener("submit", (event) => {
  event.preventDefault();
  startInteractiveGame();
});
byId("play-back-to-setup").addEventListener("click", () => switchPlayWorkspace("setup"));
byId("play-board-canvas").addEventListener("click", (event) => {
  const lostPlanetAction = spaceActionAtPlayEvent(event, "lost_planet");
  if (lostPlanetAction !== null) {
    submitHumanAction(lostPlanetAction);
    return;
  }
  const planetId = planetAtPlayEvent(event);
  if (planetId === null) return;
  state.play.selectedPlanetId = state.play.selectedPlanetId === planetId ? null : planetId;
  renderPlay();
});
byId("play-planet-actions").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-play-action]");
  if (button) submitHumanAction(button.dataset.playAction);
});
byId("play-general-actions").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-play-action]");
  if (button) submitHumanAction(button.dataset.playAction);
});
byId("play-live-roles").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-live-role]");
  if (button) updateLivePlayRole(Number(button.dataset.player), button.dataset.liveRole);
});
byId("play-auto-ai").addEventListener("change", (event) => {
  state.play.autoAi = event.target.checked;
  setPlayMessage(state.play.autoAi ? "AI 自动行动已开启" : "AI 自动行动已暂停", "ready");
  renderPlay();
});
byId("play-ai-step").addEventListener("click", runInteractiveAiTurn);
byId("play-undo").addEventListener("click", undoInteractiveTurn);
byId("setup-editor-players").addEventListener("change", () => {
  state.manualSetup.edited = true;
  state.manualSetup.preview = null;
  state.manualSetup.selectedPlanetId = null;
  state.manualSetup.planetEditorError = null;
  state.manualSetup.planetEditorMode = "move";
  renderSetupEditorSeats();
  state.manualSetup.randomElements = defaultRandomElements(
    Number(byId("setup-editor-players").value),
  );
  renderRandomElementEditor();
  setSetupEditorMessage("玩家人数已修改，随机元素已按合法模板重置", "ready");
  renderPlayConfig();
});
byId("setup-editor-map-size").addEventListener("change", () => {
  const players = Number(byId("setup-editor-players").value);
  const mapSize = normalizedMapSize(players, byId("setup-editor-map-size").value);
  const previous = captureRandomElements();
  const mapDefaults = defaultRandomElements(players, mapSize);
  state.manualSetup.randomElements = normalizedRandomElements(players, {
    ...previous,
    map_size: mapSize,
    sector_tiles: mapDefaults.sector_tiles,
    sector_rotations: mapDefaults.sector_rotations,
    planet_layout: undefined,
  });
  state.manualSetup.preview = null;
  state.manualSetup.selectedPlanetId = null;
  state.manualSetup.planetEditorError = null;
  state.manualSetup.planetEditorMode = "move";
  state.manualSetup.edited = true;
  renderRandomElementEditor();
  setSetupEditorMessage(
    mapSize === "reduced"
      ? "已切换为 BGA 三人小地图（8 星区）"
      : "已切换为标准地图（10 星区）",
    "ready",
  );
  renderPlayConfig();
});
byId("setup-editor-first-player").addEventListener("change", () => {
  captureRandomElements();
  renderRandomElementEditor();
  renderPlayConfig();
});
byId("setup-editor-map-mode").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-map-mode]");
  if (!button) return;
  captureRandomElements();
  state.manualSetup.mapMode = button.dataset.mapMode;
  state.manualSetup.randomElements.map_mode = state.manualSetup.mapMode;
  state.manualSetup.edited = true;
  renderRandomElementEditor();
  setSetupEditorMessage(
    state.manualSetup.mapMode === "manual"
      ? "已进入手动地图路径，可调整星区和单颗星球"
      : "已切换为 BGA 随机星球路径",
    "ready",
  );
});
byId("setup-editor-factions").addEventListener("change", (event) => {
  const select = event.target.closest("select[data-player]");
  if (!select) return;
  state.manualSetup.edited = true;
  state.manualSetup.factions[Number(select.dataset.player)] = Number(select.value);
  setSetupEditorMessage("设置已修改", "ready");
  renderPlayConfig();
});
byId("setup-editor-form").addEventListener("input", (event) => {
  const randomSelect = event.target.closest("select[data-random-field]");
  if (randomSelect) {
    captureRandomElements();
    if (["sector_tiles", "sector_rotations"].includes(randomSelect.dataset.randomField)) {
      delete state.manualSetup.randomElements.planet_layout;
      state.manualSetup.selectedPlanetId = null;
      state.manualSetup.planetEditorError = null;
      state.manualSetup.planetEditorMode = "move";
      renderPlanetPositionEditor();
    }
    if (randomSelect.dataset.randomField === "sector_tiles") {
      const players = Number(byId("setup-editor-players").value);
      const tile = Number(randomSelect.value);
      const side = players === 2 && tile >= 4 && tile <= 6 ? "outlined" : "solid";
      const image = randomSelect.closest(".setup-sector-control")?.querySelector("img");
      if (image) {
        image.src = sectorArtworkPath(tile + 1, side);
        image.alt = `星区 S${String(tile + 1).padStart(2, "0")}`;
      }
    }
  }
  if (!event.target.closest("select[data-player]")) {
    state.manualSetup.edited = true;
    setSetupEditorMessage("设置已修改", "ready");
    renderPlayConfig();
  }
});
byId("setup-editor-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await previewManualSetup();
  } catch (error) {
    setSetupEditorMessage(error.message || String(error), "failed");
  }
});
byId("setup-editor-randomize").addEventListener("click", randomizeManualSetup);
byId("setup-editor-run").addEventListener("click", prepareInteractiveMatch);
byId("setup-planet-editor-canvas").addEventListener("click", handlePlanetEditorClick);
byId("setup-planet-editor-add").addEventListener("click", toggleAddPlanetMode);
byId("setup-planet-editor-delete").addEventListener("click", deleteSelectedPlanet);
byId("setup-planet-editor-reset").addEventListener("click", resetPlanetLayout);
byId("setup-planet-editor-terrain").addEventListener("change", (event) => {
  state.manualSetup.planetEditorTerrain = Number(event.target.value);
  state.manualSetup.planetEditorError = null;
  renderPlanetPositionEditor();
});
document.querySelectorAll("[data-setup-config-view]").forEach((tab) => {
  tab.addEventListener("click", () => {
    state.manualSetup.configView = tab.dataset.setupConfigView;
    document.querySelectorAll("[data-setup-config-view]").forEach((item) => {
      const active = item.dataset.setupConfigView === state.manualSetup.configView;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll("[data-setup-config-panel]").forEach((panel) => {
      const active = panel.dataset.setupConfigPanel === state.manualSetup.configView;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
    });
    if (state.manualSetup.configView === "map") renderPlanetPositionEditor();
  });
});
byId("history-run-select").addEventListener("change", async (event) => {
  stopHistoryPlayback();
  state.history.runId = event.target.value || null;
  const iterations = historyRun()?.iterations || [];
  state.history.iteration = iterations.at(-1)?.iteration ?? null;
  state.history.game = iterations.at(-1)?.games?.at(-1)?.game ?? null;
  state.history.trace = null;
  await loadHistoryGame(true);
});
byId("history-refresh").addEventListener("click", () => refreshHistoryIndex());
byId("history-delete").addEventListener("click", deleteSelectedHistory);
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
byId("history-map-zoom").addEventListener("input", (event) => {
  state.history.mapView.zoom = Math.max(0.75, Math.min(1.6, Number(event.target.value) || 1));
  renderHistory();
});
byId("history-map-background").addEventListener("change", (event) => {
  state.history.mapView.background = event.target.value || "#171d23";
  renderHistory();
});
byId("history-map-reset").addEventListener("click", () => {
  state.history.mapView = { zoom: 1, gap: 0, background: "#171d23" };
  renderHistory();
});
byId("history-step-slider").addEventListener("input", (event) => setHistoryStep(event.target.value));
byId("history-previous").addEventListener("click", () => setHistoryStep(state.history.step - 1));
byId("history-next").addEventListener("click", () => setHistoryStep(state.history.step + 1));
byId("history-play").addEventListener("click", toggleHistoryPlayback);
byId("history-action-log").addEventListener("click", (event) => {
  const entry = event.target.closest("[data-step]");
  if (entry) {
    stopHistoryPlayback();
    setHistoryStep(entry.dataset.step);
  }
});
byId("history-action-log").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const entry = event.target.closest("[data-step]");
  if (!entry) return;
  event.preventDefault();
  stopHistoryPlayback();
  setHistoryStep(entry.dataset.step);
});
byId("bga-import-form").addEventListener("submit", submitBgaImport);
byId("bga-import-open-history").addEventListener("click", openImportedBgaHistory);
byId("bga-import-clear-session").addEventListener("click", clearBgaSession);
window.addEventListener("resize", () => {
  renderLossChart();
  renderSetup(state.manualSetup.preview);
  renderBoard(latestState());
  renderPlay();
  renderHistory();
  renderPlanetPositionEditor();
});

const initialHash = window.location.hash.replace("#", "");
if (window.location.pathname.startsWith("/setup/") || initialHash === "setup") state.play.workspace = "setup";
const pathView = window.location.pathname.startsWith("/setup/")
  ? "play"
  : window.location.pathname === "/play"
    ? "play"
    : window.location.pathname === "/import/bga" ? "bga-import" : "";
selectView((initialHash === "setup" ? "play" : initialHash) || pathView || "overview");
loadBgaSession();
pollEvents(true);
pollInteractiveGame();
setInterval(() => pollEvents(false), POLL_INTERVAL_MS);
setInterval(pollInteractiveGame, 1200);
setInterval(() => {
  renderStatus();
  renderMetrics();
  byId("footer-clock").textContent = formatTime(new Date().toISOString());
}, 1000);
