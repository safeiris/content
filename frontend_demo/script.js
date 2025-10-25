const API_BASE = "";

const STRUCTURE_PRESETS = {
  seo: [
    "Заголовок и вводная",
    "Ключевые выгоды",
    "Пошаговый план",
    "FAQ",
    "CTA",
  ],
  faq: [
    "Вводный абзац",
    "Основные вопросы",
    "Сложные кейсы",
    "Дополнительные ресурсы",
    "Вывод",
  ],
  overview: [
    "Краткое резюме",
    "Описание продукта",
    "Преимущества и ограничения",
    "Сравнение",
    "Рекомендации",
  ],
};

const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".tab-panel");
const pipeSelect = document.getElementById("pipe-select");
const pipesList = document.getElementById("pipes-list");
const artifactsList = document.getElementById("artifacts-list");
const cleanupArtifactsBtn = document.getElementById("cleanup-artifacts");
const briefForm = document.getElementById("brief-form");
const previewBtn = document.getElementById("preview-btn");
let reindexBtn = document.getElementById("reindex-btn");
let healthBtn = document.getElementById("health-btn");
const progressOverlay = document.getElementById("progress-overlay");
const progressMessage = progressOverlay?.querySelector('[data-role="progress-message"]') || null;
const progressStage = progressOverlay?.querySelector('[data-role="progress-stage"]') || null;
const progressPercent = progressOverlay?.querySelector('[data-role="progress-percent"]') || null;
const progressBar = progressOverlay?.querySelector('[data-role="progress-bar"]') || null;
const progressBarFill = progressOverlay?.querySelector('[data-role="progress-bar-fill"]') || null;
const progressDetails = progressOverlay?.querySelector('[data-role="progress-details"]') || null;
if (progressStage && progressStage.textContent) {
  progressStage.dataset.defaultLabel = progressStage.textContent.trim();
}
const toastRoot = document.getElementById("toast-root");
const draftView = document.getElementById("draft-view");
const reportView = document.getElementById("report-view");
const resultTitle = document.getElementById("result-title");
const resultMeta = document.getElementById("result-meta");
const resultBadges = document.getElementById("result-badges");
const retryBtn = document.getElementById("retry-btn");
const downloadMdBtn = document.getElementById("download-md");
const downloadReportBtn = document.getElementById("download-report");
const clearLogBtn = document.getElementById("clear-log");
const structurePreset = document.getElementById("structure-preset");
const structureInput = document.getElementById("structure-input");
const keywordsInput = document.getElementById("keywords-input");
const titleInput = document.getElementById("title-input");
const audienceInput = document.getElementById("audience-input");
const goalInput = document.getElementById("goal-input");
const toneSelect = document.getElementById("tone-select");
const minCharsInput = document.getElementById("min-chars-input");
const maxCharsInput = document.getElementById("max-chars-input");
const keywordModeInputs = document.querySelectorAll("input[name='keywords-mode']");
const styleProfileSelect = document.getElementById("style-profile-select");
const styleProfileHint = document.getElementById("style-profile-hint");
const sourcesList = document.getElementById("sources-list");
const addSourceBtn = document.getElementById("add-source-btn");
const contextSourceSelect = document.getElementById("context-source-select");
const healthStatus = document.getElementById("health-status");
const reindexLog = document.getElementById("reindex-log");
const previewSystem = document.getElementById("preview-system");
const previewUser = document.getElementById("preview-user");
const contextList = document.getElementById("context-list");
const contextSummary = document.getElementById("context-summary");
const contextBadge = document.getElementById("context-badge");
const customContextBlock = document.getElementById("custom-context-block");
const customContextTextarea = document.getElementById("customContext");
const customContextCounter = document.getElementById("customContextCounter");
const customContextFileInput = document.getElementById("customContextFile");
const customContextClearBtn = document.getElementById("customContextClear");
const generateBtn = briefForm.querySelector("button[type='submit']");
const advancedSettings = document.getElementById("advanced-settings");
const advancedSupportSection = document.querySelector("[data-section='support']");
const usedKeywordsSection = document.getElementById("used-keywords");
const usedKeywordsList = document.getElementById("used-keywords-list");
const usedKeywordsEmpty = document.getElementById("used-keywords-empty");

const ADVANCED_SETTINGS_STORAGE_KEY = "content-demo:advanced-settings-open";

const LOG_STATUS_LABELS = {
  info: "INFO",
  success: "SUCCESS",
  warn: "WARN",
  error: "ERROR",
};

const STEP_LABELS = {
  draft: "Черновик",
  refine: "Уточнение",
  jsonld: "JSON-LD",
  post_analysis: "Пост-анализ",
};

const PROGRESS_STAGE_LABELS = {
  draft: "Черновик",
  refine: "Доработка",
  trim: "Нормализация",
  validate: "Проверка",
  done: "Готово",
  error: "Ошибка",
};

const PROGRESS_STAGE_MESSAGES = {
  draft: "Генерируем черновик",
  refine: "Дорабатываем черновик",
  trim: "Нормализуем объём",
  validate: "Проверяем результат",
  done: "Готово",
  error: "Завершено с ошибкой",
};

const DEGRADATION_LABELS = {
  draft_failed: "Черновик по запасному сценарию",
  draft_max_tokens: "Лимит токенов — результат неполный",
  refine_skipped: "Доработка пропущена",
  jsonld_missing: "JSON-LD не сформирован",
  jsonld_repaired: "JSON-LD восстановлен вручную",
  post_analysis_skipped: "Отчёт о качестве недоступен",
  soft_timeout: "Мягкий таймаут — результат сохранён",
};

const DEFAULT_PROGRESS_MESSAGE =
  progressMessage?.textContent?.trim() || PROGRESS_STAGE_MESSAGES.draft;
const MAX_TOASTS = 3;
const MAX_CUSTOM_CONTEXT_CHARS = 20000;
const MAX_CUSTOM_CONTEXT_LABEL = MAX_CUSTOM_CONTEXT_CHARS.toLocaleString("ru-RU");

const DEFAULT_LENGTH_RANGE = Object.freeze({ min: 3500, max: 6000, hard: 6500 });

const HEALTH_STATUS_MESSAGES = {
  openai_key: {
    label: "OpenAI",
    ok: "активен",
    fail: "не найден",
  },
  llm_ping: {
    label: "LLM",
    ok: "отвечает",
    fail: "нет ответа",
  },
  retrieval_index: {
    label: "Retrieval index",
    ok: "найден",
    fail: "не найден",
  },
  artifacts_writable: {
    label: "Каталог артефактов",
    ok: "доступен",
    fail: "недоступен",
  },
  theme_index: {
    label: "Индекс темы",
    ok: "найден",
    fail: "не найден",
  },
};

const STYLE_PROFILE_HINTS = {
  "sravni.ru": "Экспертный, структурный стиль: введение → основная часть → FAQ → вывод.",
  "tinkoff.ru": "Дружелюбный и прагматичный тон: объясняем шаги на примерах и даём советы.",
  "banki.ru": "Аналитичный стиль: выделяем выгоды и риски, формулируем выводы по фактам.",
  off: "Нейтральный деловой стиль без привязки к порталу.",
};

const state = {
  pipes: new Map(),
  artifacts: [],
  artifactFiles: [],
  pendingArtifactFiles: null,
  hasMissingArtifacts: false,
  currentResult: null,
  currentDownloads: { markdown: null, report: null },
};

const featureState = {
  hideModelSelector: true,
  hideTokenSliders: true,
};

const customContextState = {
  textareaText: "",
  fileText: "",
  fileName: "",
  noticeShown: false,
};

const progressState = {
  currentPercent: 0,
  lastStage: "draft",
  hideTimer: null,
};

function resolveApiPath(path) {
  if (typeof path !== "string" || !path) {
    return API_BASE || "";
  }
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  const base = API_BASE || "";
  const normalizedBase = base.endsWith("/") ? base.slice(0, -1) : base;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (/^https?:\/\//i.test(normalizedBase)) {
    return `${normalizedBase}${normalizedPath}`;
  }
  return `${normalizedBase}${normalizedPath}`;
}

function escapeHtml(value) {
  if (typeof value !== "string") {
    return "";
  }
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };
  return value.replace(/[&<>"']/g, (char) => map[char] || char);
}

const devActionsConfig = resolveDevActions();
if (!devActionsConfig.show && advancedSupportSection) {
  advancedSupportSection.remove();
  reindexBtn = null;
  healthBtn = null;
}

initFeatureFlags();

const interactiveElements = [generateBtn];

if (reindexBtn) {
  interactiveElements.push(reindexBtn);
}
if (healthBtn) {
  interactiveElements.push(healthBtn);
}
if (cleanupArtifactsBtn) {
  interactiveElements.push(cleanupArtifactsBtn);
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

if (structurePreset) {
  structurePreset.addEventListener("change", () => applyStructurePreset(structurePreset.value));
}
pipeSelect.addEventListener("change", () => applyPipeDefaults(pipeSelect.value));
briefForm.addEventListener("submit", handleGenerate);
if (retryBtn) {
  retryBtn.addEventListener("click", handleRetryClick);
}
if (styleProfileSelect) {
  styleProfileSelect.addEventListener("change", handleStyleProfileChange);
}
if (contextSourceSelect) {
  contextSourceSelect.addEventListener("change", handleContextSourceChange);
}
if (customContextTextarea) {
  customContextTextarea.addEventListener("input", handleCustomContextInput);
}
if (customContextFileInput) {
  customContextFileInput.addEventListener("change", handleCustomContextFileChange);
}
if (customContextClearBtn) {
  customContextClearBtn.addEventListener("click", handleCustomContextClear);
}
if (addSourceBtn) {
  addSourceBtn.addEventListener("click", handleAddSource);
}
if (sourcesList) {
  sourcesList.addEventListener("click", handleSourceListClick);
}
if (reindexBtn) {
  reindexBtn.addEventListener("click", handleReindex);
}
if (healthBtn) {
  healthBtn.addEventListener("click", handleHealthCheck);
}
if (cleanupArtifactsBtn) {
  cleanupArtifactsBtn.addEventListener("click", handleArtifactsCleanup);
}
if (downloadMdBtn) {
  setDownloadLinkAvailability(downloadMdBtn, null);
  downloadMdBtn.addEventListener("click", (event) => handleDownloadClick(event, "markdown"));
}
if (downloadReportBtn) {
  setDownloadLinkAvailability(downloadReportBtn, null);
  downloadReportBtn.addEventListener("click", (event) => handleDownloadClick(event, "report"));
}
if (clearLogBtn) {
  clearLogBtn.addEventListener("click", () => {
    clearReindexLog();
    showToast({ message: "Журнал очищен", type: "info" });
  });
}

setupAdvancedSettings();
handleStyleProfileChange();
handleContextSourceChange();
updateCustomContextCounter();
init();

function switchTab(tabId) {
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabId));
  panels.forEach((panel) => panel.classList.toggle("active", panel.id === tabId));
}

function handleStyleProfileChange() {
  if (!styleProfileSelect || !styleProfileHint) {
    return;
  }
  const value = styleProfileSelect.value || "off";
  styleProfileHint.textContent = STYLE_PROFILE_HINTS[value] || STYLE_PROFILE_HINTS.off;
}

function handleAddSource(event) {
  event.preventDefault();
  addSourceRow();
}

function handleSourceListClick(event) {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  if (target.classList.contains("remove-source")) {
    const row = target.closest(".source-row");
    if (row) {
      row.remove();
    }
  }
}

function handleContextSourceChange() {
  if (!contextSourceSelect) {
    return;
  }
  const value = String(contextSourceSelect.value || "index.json").toLowerCase();
  const isCustom = value === "custom";
  const isOff = value === "off";
  if (customContextBlock) {
    customContextBlock.hidden = !isCustom;
    if (!isCustom) {
      resetCustomContextState();
    } else {
      updateCustomContextCounter();
    }
  }
}

function resetCustomContextState() {
  customContextState.textareaText = "";
  customContextState.fileText = "";
  customContextState.fileName = "";
  customContextState.noticeShown = false;
  if (customContextTextarea) {
    customContextTextarea.value = "";
  }
  if (customContextFileInput) {
    customContextFileInput.value = "";
  }
  updateCustomContextCounter();
}

function updateCustomContextCounter() {
  if (!customContextCounter) {
    return;
  }
  const activeText = customContextState.fileText || customContextState.textareaText;
  const current = activeText.length;
  customContextCounter.textContent = `${current.toLocaleString("ru-RU")} / ${MAX_CUSTOM_CONTEXT_LABEL} символов`;
}

function normalizeCustomContext(value) {
  if (typeof value !== "string") {
    return { text: "", truncated: false };
  }
  let normalized = value.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  normalized = normalized.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, "");
  normalized = normalized.replace(/\t/g, " ");
  const lines = normalized.split("\n").map((line) => line.replace(/\s+$/, ""));
  const compact = [];
  let blankPending = false;
  for (const line of lines) {
    if (line.trim()) {
      compact.push(line);
      blankPending = false;
    } else if (!blankPending && compact.length) {
      compact.push("");
      blankPending = true;
    }
  }
  normalized = compact.join("\n").trim();
  let truncated = false;
  if (normalized.length > MAX_CUSTOM_CONTEXT_CHARS) {
    normalized = normalized.slice(0, MAX_CUSTOM_CONTEXT_CHARS);
    truncated = true;
  }
  return { text: normalized, truncated };
}

function handleCustomContextInput() {
  if (!customContextTextarea) {
    return;
  }
  const { text, truncated } = normalizeCustomContext(customContextTextarea.value);
  if (text !== customContextTextarea.value) {
    customContextTextarea.value = text;
  }
  customContextState.textareaText = text;
  const shouldNotify = truncated && !customContextState.noticeShown;
  customContextState.noticeShown = truncated;
  if (shouldNotify) {
    showToast({ message: "Слишком длинный контекст: сокращён до 20 000 символов", type: "warn" });
  }
  if (!customContextState.fileText) {
    updateCustomContextCounter();
  } else if (!truncated) {
    customContextState.noticeShown = false;
  }
}

async function handleCustomContextFileChange(event) {
  if (!customContextFileInput) {
    return;
  }
  const file = customContextFileInput.files && customContextFileInput.files[0];
  customContextState.fileText = "";
  customContextState.fileName = "";
  customContextState.noticeShown = false;
  if (!file) {
    updateCustomContextCounter();
    return;
  }
  const dotIndex = file.name.lastIndexOf(".");
  const extension = dotIndex >= 0 ? file.name.slice(dotIndex).toLowerCase() : "";
  if (![".txt", ".json"].includes(extension)) {
    showToast({ message: "Поддерживаются только .txt и .json", type: "error" });
    customContextFileInput.value = "";
    updateCustomContextCounter();
    return;
  }
  try {
    let rawText = await file.text();
    if (extension === ".json") {
      try {
        const parsed = JSON.parse(rawText);
        if (Array.isArray(parsed)) {
          rawText = parsed.filter((item) => typeof item === "string").join("\n\n");
        } else if (parsed && typeof parsed === "object") {
          rawText = Object.values(parsed)
            .filter((value) => typeof value === "string")
            .join("\n\n");
        } else if (typeof parsed === "string") {
          rawText = parsed;
        } else {
          rawText = "";
        }
      } catch (error) {
        showToast({ message: "Некорректный JSON", type: "error" });
        customContextFileInput.value = "";
        updateCustomContextCounter();
        return;
      }
    }
    const { text, truncated } = normalizeCustomContext(rawText);
    customContextState.fileText = text;
    customContextState.fileName = file.name;
    customContextState.noticeShown = truncated;
    if (customContextTextarea) {
      customContextTextarea.value = text;
    }
    customContextState.textareaText = text;
    if (truncated) {
      showToast({ message: "Слишком длинный контекст: сокращён до 20 000 символов", type: "warn" });
    }
    updateCustomContextCounter();
  } catch (error) {
    showToast({ message: `Не удалось прочитать файл: ${getErrorMessage(error)}`, type: "error" });
    customContextFileInput.value = "";
    customContextState.fileText = "";
    customContextState.fileName = "";
    customContextState.noticeShown = false;
    updateCustomContextCounter();
  }
}

function handleCustomContextClear(event) {
  if (event) {
    event.preventDefault();
  }
  resetCustomContextState();
}

function resolveCustomContextPayload(contextSource) {
  if (contextSource !== "custom") {
    return { text: null, filename: null };
  }
  const sourceText = customContextState.fileText || customContextState.textareaText;
  const { text, truncated } = normalizeCustomContext(sourceText);
  if (!text) {
    throw new Error("Добавьте текст для пользовательского контекста");
  }
  const shouldNotify = truncated && !customContextState.noticeShown;
  customContextState.noticeShown = truncated;
  if (shouldNotify) {
    showToast({ message: "Слишком длинный контекст: сокращён до 20 000 символов", type: "warn" });
  }
  if (text !== sourceText) {
    if (customContextState.fileText) {
      customContextState.fileText = text;
    } else {
      customContextState.textareaText = text;
    }
    if (customContextTextarea) {
      customContextTextarea.value = text;
    }
    updateCustomContextCounter();
  }
  const filename = customContextState.fileText ? customContextState.fileName : "";
  return { text, filename: filename || null };
}

function addSourceRow(source = { value: "", usage: "summary" }) {
  if (!sourcesList) {
    return;
  }
  const template = document.getElementById("source-row-template");
  if (!(template instanceof HTMLTemplateElement)) {
    return;
  }
  const fragment = template.content.cloneNode(true);
  const row = fragment.querySelector(".source-row");
  if (!row) {
    return;
  }
  const input = row.querySelector(".source-input");
  if (input instanceof HTMLInputElement || input instanceof HTMLTextAreaElement) {
    input.value = source?.value || "";
  }
  const usageSelect = row.querySelector(".source-usage");
  if (usageSelect instanceof HTMLSelectElement && source?.usage) {
    const normalized = String(source.usage).trim().toLowerCase();
    const option = Array.from(usageSelect.options).find((item) => item.value === normalized);
    usageSelect.value = option ? option.value : usageSelect.value;
  }
  sourcesList.append(row);
}

function collectSources() {
  if (!sourcesList) {
    return [];
  }
  const rows = Array.from(sourcesList.querySelectorAll(".source-row"));
  const items = [];
  for (const row of rows) {
    const input = row.querySelector(".source-input");
    const value =
      input instanceof HTMLInputElement || input instanceof HTMLTextAreaElement ? input.value.trim() : "";
    if (!value) {
      continue;
    }
    const usageSelect = row.querySelector(".source-usage");
    const usage =
      usageSelect instanceof HTMLSelectElement ? String(usageSelect.value || "").trim().toLowerCase() : "";
    items.push({ value, usage });
  }
  return items;
}

async function init() {
  let pipesLoaded = false;
  try {
    await loadPipes();
    pipesLoaded = true;
  } catch (error) {
    console.error(error);
    showToast({ message: `Не удалось загрузить тематики: ${getErrorMessage(error)}`, type: "error" });
  }

  try {
    await loadArtifacts();
  } catch (error) {
    console.error(error);
    showToast({ message: `Не удалось загрузить материалы: ${getErrorMessage(error)}`, type: "error" });
  }

  if (pipesLoaded) {
    applyStructurePreset(structurePreset.value);
  }

  if (sourcesList && sourcesList.childElementCount === 0) {
    addSourceRow();
  }

  await handleHealthCheck();
}

async function loadPipes() {
  let pipes;
  try {
    pipes = await fetchJson("/api/pipes");
  } catch (error) {
    pipesList.innerHTML = '<div class="empty-state">Не удалось загрузить тематики.</div>';
    pipeSelect.innerHTML = "";
    state.pipes.clear();
    throw error;
  }
  if (!Array.isArray(pipes)) {
    pipesList.innerHTML = '<div class="empty-state">Некорректный ответ сервера.</div>';
    pipeSelect.innerHTML = "";
    state.pipes.clear();
    throw new Error("Некорректный ответ сервера");
  }
  pipesList.innerHTML = "";
  pipeSelect.innerHTML = "";
  state.pipes.clear();

  if (!pipes.length) {
    pipesList.innerHTML = '<div class="empty-state">Тематики пока не добавлены.</div>';
    return;
  }

  pipes.forEach((pipe, idx) => {
    state.pipes.set(pipe.id, pipe);
    const option = document.createElement("option");
    option.value = pipe.id;
    option.textContent = pipe.name;
    if (idx === 0) {
      option.selected = true;
    }
    pipeSelect.append(option);
  });

  renderPipeCards(pipes);
  if (pipes.length) {
    applyPipeDefaults(pipes[0].id);
  }
}

function renderPipeCards(pipes) {
  if (!pipes.length) {
    pipesList.innerHTML = '<div class="empty-state">Тематики пока не добавлены.</div>';
    return;
  }

  pipesList.innerHTML = "";
  pipes.forEach((pipe) => {
    const card = document.createElement("article");
    card.className = "pipe-card";
    const header = document.createElement("div");
    const title = document.createElement("h3");
    title.textContent = pipe.name;
    const description = document.createElement("p");
    description.textContent = pipe.description || `Тематика ${pipe.name}`;
    header.append(title, description);

    const meta = document.createElement("div");
    meta.className = "pipe-meta";
    meta.innerHTML = `Тон: <strong>${pipe.tone || "—"}</strong>`;

    const keywordsWrapper = document.createElement("div");
    keywordsWrapper.className = "pipe-keywords";
    (pipe.keywords || []).slice(0, 8).forEach((keyword) => {
      const badge = document.createElement("span");
      badge.textContent = keyword;
      keywordsWrapper.append(badge);
    });

    card.append(header, meta, keywordsWrapper);
    pipesList.append(card);
  });
}

async function fetchArtifactFiles() {
  const artifacts = await fetchJson("/api/artifacts");
  if (Array.isArray(artifacts)) {
    return artifacts;
  }
  if (artifacts && Array.isArray(artifacts.items)) {
    return artifacts.items;
  }
  throw new Error("Некорректный ответ сервера");
}

async function loadArtifacts(prefetched = null) {
  let files = prefetched;
  if (!Array.isArray(files)) {
    try {
      files = await fetchArtifactFiles();
    } catch (error) {
      artifactsList.innerHTML = '<div class="empty-state">Не удалось загрузить материалы.</div>';
      state.artifacts = [];
      state.artifactFiles = [];
      state.hasMissingArtifacts = false;
      updateArtifactsToolbar();
      throw error;
    }
  }
  if (!Array.isArray(files)) {
    artifactsList.innerHTML = '<div class="empty-state">Некорректный ответ сервера.</div>';
    state.artifacts = [];
    state.artifactFiles = [];
    state.hasMissingArtifacts = false;
    updateArtifactsToolbar();
    throw new Error("Некорректный ответ сервера");
  }
  state.artifactFiles = files;
  state.artifacts = normalizeArtifactList(files);
  state.hasMissingArtifacts = state.artifacts.some((artifact) => artifact.missing);
  renderArtifacts();
  updateArtifactsToolbar();
}

function renderArtifacts() {
  state.hasMissingArtifacts = state.artifacts.some((item) => item.missing);
  if (!state.artifacts.length) {
    artifactsList.innerHTML = '<div class="empty-state">Пока нет сгенерированных материалов.</div>';
    updateArtifactsToolbar();
    return;
  }

  artifactsList.innerHTML = "";
  const template = document.getElementById("artifact-card-template");

  state.artifacts.forEach((artifact) => {
    const card = template.content.firstElementChild.cloneNode(true);
    const metadata = artifact.metadata || {};
    const title = artifact.name || metadata.name || artifact.id || "Без названия";
    const themeName = metadata.theme || extractThemeFromName(title);
    const topic =
      metadata.input_data?.title
      || metadata.data?.title
      || metadata.title
      || metadata.input_data?.goal
      || metadata.input_data?.theme
      || metadata.data?.theme
      || metadata.topic
      || "";
    card.dataset.artifactId = artifact.id || artifact.path || "";
    card.querySelector(".card-title").textContent = title;
    const statusInfo = resolveArtifactStatus(artifact);
    const statusEl = card.querySelector(".status");
    statusEl.textContent = statusInfo.label;
    statusEl.dataset.status = statusInfo.value;
    const topicText = [themeName, topic].filter(Boolean).join(" · ");
    card.querySelector(".card-topic").textContent = topicText || themeName || "Без темы";
    const updatedAt = artifact.modified_at ? new Date(artifact.modified_at) : null;
    card.querySelector(".card-meta").textContent = updatedAt
      ? `Обновлено ${updatedAt.toLocaleString("ru-RU")}`
      : "Дата обновления неизвестна";

    card.addEventListener("click", () => showArtifact(artifact));

    const openBtn = card.querySelector(".open-btn");
    const downloadBtn = card.querySelector(".download-btn");
    const deleteBtn = card.querySelector(".delete-btn");
    const hasMarkdown = Boolean(artifact.downloads?.markdown);
    if (openBtn) {
      openBtn.disabled = !hasMarkdown;
      openBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        showArtifact(artifact);
      });
    }
    if (downloadBtn) {
      downloadBtn.disabled = !hasMarkdown;
      downloadBtn.addEventListener("click", async (event) => {
        event.stopPropagation();
        await handleArtifactDownload(artifact);
      });
    }
    if (deleteBtn) {
      deleteBtn.addEventListener("click", async (event) => {
        event.stopPropagation();
        await handleDeleteArtifact(artifact, deleteBtn);
      });
    }

    artifactsList.append(card);
  });
  updateArtifactsToolbar();
}

function resolveArtifactStatus(artifact) {
  const rawStatus = artifact?.status || artifact?.metadata?.status;
  if (!rawStatus) {
    return { value: "Ready", label: "Готов" };
  }
  const normalized = String(rawStatus).trim().toLowerCase();
  if (!normalized) {
    return { value: "Ready", label: "Готов" };
  }
  if (["ready", "ok", "success", "final"].includes(normalized)) {
    return { value: "Ready", label: "Готов" };
  }
  if (["draft", "dry-run", "dry_run"].includes(normalized)) {
    return { value: "Draft", label: "Черновик" };
  }
  if (["error", "failed"].includes(normalized)) {
    return { value: "Error", label: "Ошибка" };
  }
  return { value: rawStatus, label: rawStatus };
}

function normalizeArtifactList(items) {
  const groups = new Map();

  items.forEach((item) => {
    if (!item || typeof item !== "object") {
      return;
    }
    const metadata = item.metadata && typeof item.metadata === "object" && !Array.isArray(item.metadata)
      ? item.metadata
      : {};
    const fallbackId =
      typeof item.artifact_id === "string" && item.artifact_id.trim()
        ? item.artifact_id.trim()
        : typeof item.id === "string" && item.id.trim()
          ? item.id.trim()
          : typeof item.path === "string" && item.path.trim()
            ? item.path.trim()
            : metadata.id || metadata.artifact_id || null;
    const groupId = fallbackId || (typeof item.name === "string" ? item.name.replace(/\.[^.]+$/, "").trim() : null);
    if (!groupId) {
      return;
    }
    if (!groups.has(groupId)) {
      groups.set(groupId, {
        id: groupId,
        name: null,
        metadata: {},
        status: null,
        path: null,
        metadata_path: null,
        downloads: { markdown: null, report: null },
        modified_at: null,
        modified_ts: null,
        size: null,
        job_id: null,
      });
    }
    const group = groups.get(groupId);
    if (!group) {
      return;
    }

    const downloadInfo = createDownloadInfoFromFile(item);
    const type = (item.type || "").toString().toLowerCase();
    const isMarkdown = type === "md" || (typeof item.name === "string" && item.name.toLowerCase().endsWith?.(".md"));
    const isJson = type === "json" || (typeof item.name === "string" && item.name.toLowerCase().endsWith?.(".json"));

    if (isMarkdown && downloadInfo) {
      group.downloads.markdown = downloadInfo;
      group.path = downloadInfo.path || item.artifact_path || downloadInfo.url || group.path;
      if (!group.name) {
        group.name = typeof item.name === "string" && item.name.trim()
          ? item.name.trim()
          : metadata.title || metadata.name || null;
      }
    }
    if (isJson && downloadInfo) {
      group.downloads.report = downloadInfo;
      group.metadata_path = downloadInfo.path || item.metadata_path || downloadInfo.url || group.metadata_path;
    }

    if (Object.keys(metadata).length) {
      group.metadata = metadata;
      if (!group.name) {
        const metaName = metadata.input_data?.theme || metadata.theme || metadata.name;
        if (metaName) {
          group.name = metaName;
        }
      }
      if (!group.status && typeof metadata.status === "string") {
        group.status = metadata.status;
      }
      if (!group.job_id && typeof metadata.job_id === "string") {
        group.job_id = metadata.job_id;
      }
    }

    if (!group.status && typeof item.status === "string") {
      group.status = item.status;
    }
    if (!group.job_id && typeof item.job_id === "string") {
      group.job_id = item.job_id;
    }
    if (typeof item.size === "number") {
      group.size = item.size;
    }

    const createdAt = typeof item.created_at === "string" && item.created_at.trim()
      ? item.created_at.trim()
      : downloadInfo?.created_at || null;
    const createdTs = parseTimestamp(createdAt);
    if (createdTs !== null && (group.modified_ts === null || createdTs > group.modified_ts)) {
      group.modified_ts = createdTs;
      group.modified_at = createdAt;
    }
  });

  return Array.from(groups.values()).map((group) => {
    const name = group.name || group.metadata?.name || group.metadata?.input_data?.theme || group.id;
    return {
      id: group.id,
      name,
      path: group.path,
      metadata_path: group.metadata_path,
      metadata: group.metadata || {},
      size: typeof group.size === "number" ? group.size : null,
      modified_at: group.modified_at,
      status: group.status || null,
      job_id: group.job_id || null,
      downloads: group.downloads,
      missing: !group.downloads.markdown,
    };
  });
}

function parseTimestamp(value) {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }
  const direct = Date.parse(value);
  if (!Number.isNaN(direct)) {
    return direct;
  }
  const normalized = Date.parse(value.replace(/\s+/, "T"));
  return Number.isNaN(normalized) ? null : normalized;
}

function createDownloadInfoFromFile(item) {
  if (!item || typeof item !== "object") {
    return null;
  }
  const url = typeof item.url === "string" && item.url.trim() ? item.url.trim() : null;
  const path = typeof item.path === "string" && item.path.trim() ? item.path.trim() : null;
  const name = typeof item.name === "string" && item.name.trim() ? item.name.trim() : null;
  const size = typeof item.size === "number" ? item.size : null;
  const createdAt = typeof item.created_at === "string" && item.created_at.trim() ? item.created_at.trim() : null;
  if (!url && !path) {
    return null;
  }
  const info = {
    url,
    path,
    name,
    size,
    created_at: createdAt,
    job_id: item.job_id || null,
  };
  if (!info.url && info.path) {
    info.url = `/api/artifacts/download?path=${encodeURIComponent(info.path)}`;
  }
  return info;
}

function setDownloadLinkAvailability(link, downloadInfo) {
  if (!link) {
    return;
  }
  const fallbackName = link.dataset.fallbackName || (link.id === "download-md" ? "draft.md" : "report.json");
  if (downloadInfo && (downloadInfo.url || downloadInfo.path)) {
    const targetUrl = downloadInfo.url || `/api/artifacts/download?path=${encodeURIComponent(downloadInfo.path)}`;
    const resolvedUrl = resolveApiPath(targetUrl);
    link.href = resolvedUrl;
    link.dataset.downloadUrl = resolvedUrl;
    link.setAttribute("download", downloadInfo.name || fallbackName);
    link.classList.remove("is-disabled");
    link.removeAttribute("aria-disabled");
    return;
  }
  link.removeAttribute("href");
  link.removeAttribute("download");
  link.classList.add("is-disabled");
  link.setAttribute("aria-disabled", "true");
  delete link.dataset.downloadUrl;
}

function setActiveArtifactDownloads(downloads) {
  const normalized = downloads && typeof downloads === "object" ? downloads : {};
  state.currentDownloads = {
    markdown: normalized.markdown || normalized.article || null,
    report: normalized.report || normalized.json || normalized.metadata || null,
  };
  setDownloadLinkAvailability(downloadMdBtn, state.currentDownloads.markdown);
  setDownloadLinkAvailability(downloadReportBtn, state.currentDownloads.report);
}

function hasDownloadFiles(downloads) {
  if (!downloads || typeof downloads !== "object") {
    return false;
  }
  const markdown = downloads.markdown || downloads.article || null;
  const report = downloads.report || downloads.json || downloads.metadata || null;
  return Boolean(markdown || report);
}

function resetDownloadButtonsForNewJob() {
  state.currentDownloads = { markdown: null, report: null };
  setDownloadLinkAvailability(downloadMdBtn, null);
  setDownloadLinkAvailability(downloadReportBtn, null);
  setButtonLoading(downloadMdBtn, true);
  setButtonLoading(downloadReportBtn, true);
}

function handleDownloadClick(event, type) {
  const link = type === "markdown" ? downloadMdBtn : downloadReportBtn;
  if (!link) {
    return;
  }
  if (link.classList.contains("loading") || link.getAttribute("aria-disabled") === "true") {
    event.preventDefault();
    if (!link.classList.contains("loading")) {
      showToast({ message: "Файл недоступен для скачивания", type: "warn" });
    }
    return;
  }
  const downloadInfo = type === "markdown" ? state.currentDownloads.markdown : state.currentDownloads.report;
  if (!downloadInfo) {
    event.preventDefault();
    showToast({ message: "Файл недоступен для скачивания", type: "warn" });
    return;
  }
  event.preventDefault();
  const fallbackName = link.dataset.fallbackName || (type === "markdown" ? "draft.md" : "report.json");
  downloadArtifactFile(downloadInfo, fallbackName);
}

function extractArtifactPathHints(artifactPaths) {
  const hints = new Set();
  if (!artifactPaths || typeof artifactPaths !== "object") {
    return hints;
  }
  ["markdown", "metadata", "json"].forEach((key) => {
    const raw = artifactPaths[key];
    if (typeof raw === "string" && raw.trim()) {
      const name = raw.split("/").pop() || raw;
      hints.add(name.replace(/\.[^.]+$/, ""));
    }
  });
  return hints;
}

function findLatestArtifactPair(files, { jobId = null, artifactPaths = null } = {}) {
  if (!Array.isArray(files) || !files.length) {
    return null;
  }

  const groups = new Map();
  const hints = extractArtifactPathHints(artifactPaths);

  const isMarkdownFile = (file) => {
    const type = (file?.type || "").toString().toLowerCase();
    if (type === "md") {
      return true;
    }
    const name = typeof file?.name === "string" ? file.name.toLowerCase() : "";
    return name.endsWith(".md");
  };

  const isJsonFile = (file) => {
    const type = (file?.type || "").toString().toLowerCase();
    if (type === "json") {
      return true;
    }
    const name = typeof file?.name === "string" ? file.name.toLowerCase() : "";
    return name.endsWith(".json");
  };

  files.forEach((file) => {
    if (!file || typeof file !== "object") {
      return;
    }
    const artifactId = typeof file.artifact_id === "string" && file.artifact_id.trim()
      ? file.artifact_id.trim()
      : null;
    const name = typeof file.name === "string" ? file.name : "";
    const baseName = name ? name.replace(/\.[^.]+$/, "") : null;
    const key = artifactId || baseName;
    if (!key) {
      return;
    }
    if (!groups.has(key)) {
      groups.set(key, {
        files: [],
        jobIds: new Set(),
        createdTs: null,
        baseName: baseName || key,
      });
    }
    const group = groups.get(key);
    if (!group) {
      return;
    }
    group.files.push(file);
    if (file.job_id) {
      group.jobIds.add(file.job_id);
    }
    const createdTs = parseTimestamp(typeof file.created_at === "string" ? file.created_at : null);
    if (createdTs !== null && (group.createdTs === null || createdTs > group.createdTs)) {
      group.createdTs = createdTs;
    }
  });

  const availableGroups = Array.from(groups.entries()).filter(([_, group]) => group.files.some(isMarkdownFile));
  if (!availableGroups.length) {
    return null;
  }

  const selectByJob = jobId
    ? availableGroups.find(([_, group]) => group.jobIds.has(jobId))
    : null;
  if (selectByJob) {
    const group = selectByJob[1];
    const markdown = group.files.find(isMarkdownFile) || null;
    const report = group.files.find(isJsonFile) || null;
    return { markdown, report };
  }

  const selectByHint = hints.size
    ? availableGroups.find(([key, group]) => hints.has(group.baseName) || hints.has(key))
    : null;
  if (selectByHint) {
    const group = selectByHint[1];
    const markdown = group.files.find(isMarkdownFile) || null;
    const report = group.files.find(isJsonFile) || null;
    return { markdown, report };
  }

  availableGroups.sort((a, b) => (b[1].createdTs || 0) - (a[1].createdTs || 0));
  const latest = availableGroups[0]?.[1];
  if (!latest) {
    return null;
  }
  const markdown = latest.files.find(isMarkdownFile) || null;
  const report = latest.files.find(isJsonFile) || null;
  return { markdown, report };
}

function hasDraftStepCompleted(snapshot) {
  if (!snapshot || !Array.isArray(snapshot.steps)) {
    return false;
  }
  return snapshot.steps.some((step) => {
    if (!step || step.name !== "draft") {
      return false;
    }
    const status = typeof step.status === "string" ? step.status.trim().toLowerCase() : "";
    return status === "succeeded" || status === "degraded";
  });
}

async function refreshDownloadLinksForJob({ jobId = null, artifactPaths = null } = {}) {
  if (!downloadMdBtn || !downloadReportBtn) {
    return null;
  }
  try {
    const files = await fetchArtifactFiles();
    state.pendingArtifactFiles = files;
    const pair = findLatestArtifactPair(files, { jobId, artifactPaths });
    const downloads = pair
      ? {
          markdown: pair.markdown ? createDownloadInfoFromFile(pair.markdown) : null,
          report: pair.report ? createDownloadInfoFromFile(pair.report) : null,
        }
      : { markdown: null, report: null };
    setButtonLoading(downloadMdBtn, false);
    setButtonLoading(downloadReportBtn, false);
    setActiveArtifactDownloads(downloads);
    if (state.currentResult && typeof state.currentResult === "object") {
      state.currentResult.downloads = downloads;
    }
    return downloads;
  } catch (error) {
    console.warn("Не удалось обновить ссылки на артефакты", error);
    setButtonLoading(downloadMdBtn, false);
    setButtonLoading(downloadReportBtn, false);
    setActiveArtifactDownloads({ markdown: null, report: null });
    return null;
  }
}

function resolveArtifactIdentifier(artifact) {
  if (!artifact) {
    return null;
  }
  if (typeof artifact === "string") {
    return artifact;
  }
  if (typeof artifact.id === "string" && artifact.id.trim()) {
    return artifact.id.trim();
  }
  if (typeof artifact.path === "string" && artifact.path.trim()) {
    return artifact.path.trim();
  }
  return null;
}

function removeArtifactFromState(artifact) {
  const identifier = resolveArtifactIdentifier(artifact);
  if (!identifier) {
    return false;
  }
  const before = state.artifacts.length;
  state.artifacts = state.artifacts.filter((item) => resolveArtifactIdentifier(item) !== identifier);
  const changed = state.artifacts.length !== before;
  state.hasMissingArtifacts = state.artifacts.some((item) => item.missing);
  if (changed) {
    renderArtifacts();
  }
  return changed;
}

async function handleArtifactDownload(artifact) {
  const downloadInfo = artifact?.downloads?.markdown || null;
  if (!downloadInfo) {
    showToast({ message: "Файл недоступен для скачивания", type: "warn" });
    return false;
  }
  const fallbackName = artifact?.name || downloadInfo.name || "artifact.txt";
  const result = await downloadArtifactFile(downloadInfo, fallbackName, artifact);
  return result === "ok";
}

async function handleDeleteArtifact(artifact, button) {
  const title = artifact?.name || artifact?.id || "этот результат";
  const confirmed = window.confirm(`Удалить результат «${title}»?`);
  if (!confirmed) {
    return;
  }
  try {
    setButtonLoading(button, true);
    const payload = {};
    if (artifact.id) {
      payload.id = artifact.id;
    }
    if (artifact.path) {
      payload.path = artifact.path;
    }
    const response = await fetchJson("/api/artifacts", {
      method: "DELETE",
      body: JSON.stringify(payload),
    });
    const removed = removeArtifactFromState(artifact);
    if (!removed) {
      try {
        await loadArtifacts();
      } catch (refreshError) {
        console.error(refreshError);
      }
    }
    showToast({ message: "Удалено", type: "success" });
    if (Array.isArray(response?.errors) && response.errors.length) {
      showToast({ message: "Удалено, но индекс обновится при следующем открытии", type: "warn" });
    }
  } catch (error) {
    console.error(error);
    showToast({ message: `Не удалось удалить: ${getErrorMessage(error)}`, type: "error" });
  } finally {
    setButtonLoading(button, false);
  }
}

async function handleArtifactsCleanup(event) {
  if (event) {
    event.preventDefault();
  }
  if (!cleanupArtifactsBtn) {
    return;
  }
  try {
    setButtonLoading(cleanupArtifactsBtn, true);
    const result = await fetchJson("/api/artifacts/cleanup", { method: "POST" });
    const removed = typeof result?.removed === "number" ? result.removed : 0;
    if (removed > 0) {
      showToast({ message: `Удалено ${removed} записей`, type: "success" });
    } else if (Array.isArray(result?.errors) && result.errors.length) {
      showToast({ message: `Очистка: ${result.errors.join(", ")}`, type: "error" });
    } else {
      showToast({ message: "Список актуален", type: "info" });
    }
  } catch (error) {
    console.error(error);
    showToast({ message: `Очистка не удалась: ${getErrorMessage(error)}`, type: "error" });
  } finally {
    setButtonLoading(cleanupArtifactsBtn, false);
  }
  try {
    await loadArtifacts();
  } catch (error) {
    console.error(error);
  }
}

function handleMissingArtifact(artifact) {
  const removed = removeArtifactFromState(artifact);
  if (removed) {
    showToast({ message: "Файл недоступен: запись удалена из списка", type: "warn" });
  }
  return removed;
}

function updateArtifactsToolbar() {
  if (!cleanupArtifactsBtn) {
    return;
  }
  const hasArtifacts = state.artifacts.length > 0;
  cleanupArtifactsBtn.disabled = !hasArtifacts;
  cleanupArtifactsBtn.classList.toggle("attention", Boolean(state.hasMissingArtifacts));
}

function extractThemeFromName(filename) {
  const parts = filename.split("__");
  if (parts.length >= 2) {
    return parts[1];
  }
  return "";
}

function markdownToHtml(markdown) {
  const lines = markdown.split(/\r?\n/);
  let html = "";
  let inUl = false;
  let inOl = false;

  const closeLists = () => {
    if (inUl) {
      html += "</ul>";
      inUl = false;
    }
    if (inOl) {
      html += "</ol>";
      inOl = false;
    }
  };

  lines.forEach((rawLine) => {
    const line = rawLine.trimEnd();
    if (!line.trim()) {
      closeLists();
      html += "";
      return;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      closeLists();
      const level = headingMatch[1].length;
      const text = headingMatch[2];
      html += `<h${level}>${inlineFormat(text)}</h${level}>`;
      return;
    }

    const olMatch = line.match(/^\d+\.\s+(.*)$/);
    if (olMatch) {
      if (!inOl) {
        closeLists();
        html += "<ol>";
        inOl = true;
      }
      html += `<li>${inlineFormat(olMatch[1])}</li>`;
      return;
    }

    const ulMatch = line.match(/^[-*]\s+(.*)$/);
    if (ulMatch) {
      if (!inUl) {
        closeLists();
        html += "<ul>";
        inUl = true;
      }
      html += `<li>${inlineFormat(ulMatch[1])}</li>`;
      return;
    }

    closeLists();
    html += `<p>${inlineFormat(line)}</p>`;
  });

  closeLists();
  return html;
}

function inlineFormat(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function applyStructurePreset(presetKey) {
  if (!structureInput) {
    return;
  }
  if (presetKey === "custom") {
    return;
  }
  const preset = STRUCTURE_PRESETS[presetKey];
  if (preset) {
    structureInput.value = preset.join("\n");
  }
}

function applyPipeDefaults(pipeId) {
  const pipe = state.pipes.get(pipeId);
  if (!pipe) {
    return;
  }
  if (structureInput && !structureInput.value && Array.isArray(pipe.default_structure)) {
    structureInput.value = pipe.default_structure.join("\n");
  }
  if (toneSelect && pipe.tone) {
    const normalized = String(pipe.tone).trim().toLowerCase();
    const option = Array.from(toneSelect.options || []).find((item) => item.value === normalized);
    if (option) {
      toneSelect.value = option.value;
    }
  }
}

async function handlePromptPreview() {
  try {
    const payload = buildRequestPayload();
    setInteractiveBusy(true);
    setButtonLoading(previewBtn, true);
    showProgress(true, "Собираем промпт…");
    const previewRequest = {
      theme: payload.theme,
      data: payload.data,
      k: payload.k,
      context_source: payload.context_source,
    };
    if (payload.context_source === "custom") {
      previewRequest.context_text = payload.context_text;
      if (payload.context_filename) {
        previewRequest.context_filename = payload.context_filename;
      }
    }
    const preview = await fetchJson("/api/prompt/preview", {
      method: "POST",
      body: JSON.stringify(previewRequest),
    });
    updatePromptPreview(preview);
    switchTab("result");
  } catch (error) {
    console.error(error);
    showToast({ message: `Не удалось собрать промпт: ${getErrorMessage(error)}`, type: "error" });
  } finally {
    setButtonLoading(previewBtn, false);
    setInteractiveBusy(false);
    showProgress(false);
  }
}

async function handleGenerate(event) {
  event.preventDefault();
  try {
    const payload = buildRequestPayload();
    resetDownloadButtonsForNewJob();
    state.pendingArtifactFiles = null;
    let downloadsRequested = false;
    let downloadsResolved = false;
    let pendingDownloadRefresh = null;
    let activeJobId = null;
    let artifactPathsHint = null;
    toggleRetryButton(false);
    setInteractiveBusy(true);
    setButtonLoading(generateBtn, true);
    showProgress(true, DEFAULT_PROGRESS_MESSAGE);
    renderUsedKeywords(null);
    const requestBody = {
      theme: payload.theme,
      data: payload.data,
      k: payload.k,
      context_source: payload.context_source,
    };
    if (Array.isArray(payload.data?.keywords)) {
      requestBody.keywords = payload.data.keywords;
    }
    requestBody.faq_required = true;
    requestBody.faq_count = payload.data?.faq_questions || 5;
    if (payload.context_source === "custom") {
      requestBody.context_text = payload.context_text;
      if (payload.context_filename) {
        requestBody.context_filename = payload.context_filename;
      }
    }
    const initialResponse = await fetchJson("/api/generate", {
      method: "POST",
      body: JSON.stringify(requestBody),
    });
    let snapshot = normalizeJobResponse(initialResponse);
    activeJobId = snapshot.job_id || snapshot.id || activeJobId;
    if (snapshot.result && typeof snapshot.result === "object" && snapshot.result.artifact_paths) {
      artifactPathsHint = snapshot.result.artifact_paths;
    }
    applyProgressiveResult(snapshot);
    updateProgressFromSnapshot(snapshot);
    if ((!downloadsRequested || !downloadsResolved) && hasDraftStepCompleted(snapshot)) {
      downloadsRequested = true;
      if (!pendingDownloadRefresh) {
        const refreshPromise = refreshDownloadLinksForJob({ jobId: activeJobId, artifactPaths: artifactPathsHint });
        pendingDownloadRefresh = refreshPromise;
        refreshPromise
          .then((downloads) => {
            if (hasDownloadFiles(downloads)) {
              downloadsResolved = true;
            }
          })
          .catch((error) => {
            console.warn("Не удалось заранее получить ссылки на артефакты", error);
          })
          .finally(() => {
            if (pendingDownloadRefresh === refreshPromise) {
              pendingDownloadRefresh = null;
            }
          });
      }
    }
    if (snapshot.status !== "succeeded" || !snapshot.result) {
      if (!snapshot.job_id) {
        throw new Error("Сервер вернул пустой ответ без идентификатора задания.");
      }
      snapshot = await pollJobUntilDone(snapshot.job_id, {
        onUpdate: (update) => {
          applyProgressiveResult(update);
          updateProgressFromSnapshot(update);
          if ((!downloadsRequested || !downloadsResolved) && hasDraftStepCompleted(update)) {
            downloadsRequested = true;
            activeJobId = update?.id || update?.job_id || activeJobId;
            if (update?.result && typeof update.result === "object" && update.result.artifact_paths) {
              artifactPathsHint = update.result.artifact_paths;
            }
            if (!pendingDownloadRefresh) {
              const refreshPromise = refreshDownloadLinksForJob({ jobId: activeJobId, artifactPaths: artifactPathsHint });
              pendingDownloadRefresh = refreshPromise;
              refreshPromise
                .then((downloads) => {
                  if (hasDownloadFiles(downloads)) {
                    downloadsResolved = true;
                  }
                })
                .catch((error) => {
                  console.warn("Не удалось заранее получить ссылки на артефакты", error);
                })
                .finally(() => {
                  if (pendingDownloadRefresh === refreshPromise) {
                    pendingDownloadRefresh = null;
                  }
                });
            }
          }
        },
      });
    }
    activeJobId = snapshot?.id || snapshot?.job_id || activeJobId;
    if (snapshot?.result && typeof snapshot.result === "object" && snapshot.result.artifact_paths) {
      artifactPathsHint = snapshot.result.artifact_paths;
    }
    updateProgressFromSnapshot(snapshot);
    if (hasDraftStepCompleted(snapshot)) {
      downloadsRequested = true;
      if (pendingDownloadRefresh) {
        try {
          const pendingResult = await pendingDownloadRefresh;
          if (hasDownloadFiles(pendingResult)) {
            downloadsResolved = true;
          }
        } catch (error) {
          console.warn("Не удалось заранее получить ссылки на артефакты", error);
        }
        pendingDownloadRefresh = null;
      }
      const finalPromise = refreshDownloadLinksForJob({ jobId: activeJobId, artifactPaths: artifactPathsHint });
      pendingDownloadRefresh = finalPromise;
      try {
        const finalDownloads = await finalPromise;
        if (hasDownloadFiles(finalDownloads)) {
          downloadsResolved = true;
        }
      } catch (error) {
        console.warn("Не удалось заранее получить ссылки на артефакты", error);
      } finally {
        if (pendingDownloadRefresh === finalPromise) {
          pendingDownloadRefresh = null;
        }
      }
    }
    renderGenerationResult(snapshot, { payload });
    try {
      const pendingFiles = state.pendingArtifactFiles;
      await loadArtifacts(pendingFiles);
    } catch (refreshError) {
      console.error(refreshError);
      showToast({ message: `Не удалось обновить список материалов: ${getErrorMessage(refreshError)}`, type: "warn" });
    }
    state.pendingArtifactFiles = null;
    switchTab("result");
    showToast({ message: "Готово", type: "success" });
  } catch (error) {
    console.error(error);
    showToast({ message: `Не удалось выполнить генерацию: ${getErrorMessage(error)}`, type: "error" });
    setButtonLoading(downloadMdBtn, false);
    setButtonLoading(downloadReportBtn, false);
    setActiveArtifactDownloads(null);
    hideProgressOverlay({ immediate: true });
  } finally {
    setButtonLoading(generateBtn, false);
    setInteractiveBusy(false);
    hideProgressOverlay();
    state.pendingArtifactFiles = null;
  }
}

function normalizeJobResponse(response) {
  if (!response || typeof response !== "object") {
    return { status: "pending", result: null, steps: [], degradation_flags: [], job_id: null };
  }
  if (typeof response.markdown === "string" || typeof response.meta_json === "object") {
    return {
      status: "succeeded",
      job_id: response.job_id || null,
      steps: Array.isArray(response.steps) ? response.steps : [],
      degradation_flags: Array.isArray(response.degradation_flags) ? response.degradation_flags : [],
      trace_id: response.trace_id || null,
      message: typeof response.message === "string" ? response.message : null,
      progress: typeof response.progress === "number" ? response.progress : null,
      step: typeof response.step === "string" ? response.step : null,
      last_event_at: response.last_event_at || null,
      progress_stage:
        typeof response.progress_stage === "string" ? response.progress_stage : null,
      progress_message:
        typeof response.progress_message === "string" ? response.progress_message : null,
      progress_payload:
        response.progress_payload && typeof response.progress_payload === "object"
          ? response.progress_payload
          : null,
      result: {
        markdown: typeof response.markdown === "string" ? response.markdown : "",
        meta_json: (response.meta_json && typeof response.meta_json === "object") ? response.meta_json : {},
        faq_entries: Array.isArray(response.faq_entries) ? response.faq_entries : [],
      },
    };
  }
  return {
    status: typeof response.status === "string" ? response.status : "pending",
    job_id: response.job_id || null,
    steps: Array.isArray(response.steps) ? response.steps : [],
    degradation_flags: Array.isArray(response.degradation_flags) ? response.degradation_flags : [],
    trace_id: response.trace_id || null,
    message: typeof response.message === "string" ? response.message : null,
    progress: typeof response.progress === "number" ? response.progress : null,
    step: typeof response.step === "string" ? response.step : null,
    last_event_at: response.last_event_at || null,
    progress_stage: typeof response.progress_stage === "string" ? response.progress_stage : null,
    progress_message:
      typeof response.progress_message === "string" ? response.progress_message : null,
    progress_payload:
      response.progress_payload && typeof response.progress_payload === "object"
        ? response.progress_payload
        : null,
    result: response.result && typeof response.result === "object" ? response.result : null,
  };
}

function applyProgressiveResult(snapshot) {
  const result = snapshot?.result;
  if (!result || typeof result !== "object") {
    return 0;
  }
  const markdown = typeof result.markdown === "string" ? result.markdown : "";
  if (markdown) {
    draftView.innerHTML = markdownToHtml(markdown);
  }
  const meta = (result.meta_json && typeof result.meta_json === "object") ? result.meta_json : {};
  updateResultBadges(meta, Array.isArray(snapshot?.degradation_flags) ? snapshot.degradation_flags : []);
  const characters = typeof meta.characters === "number"
    ? meta.characters
    : markdown.replace(/\s+/g, "").length;
  return characters;
}

async function pollJobUntilDone(jobId, { onUpdate } = {}) {
  if (typeof EventSource === "function") {
    try {
      return await watchJobViaSse(jobId, { onUpdate });
    } catch (error) {
      console.warn("SSE unavailable, falling back to polling", error);
    }
  }
  return watchJobViaPolling(jobId, { onUpdate });
}

function watchJobViaSse(jobId, { onUpdate } = {}) {
  return new Promise((resolve, reject) => {
    let settled = false;
    let retries = 0;
    const maxRetries = 4;
    let source = null;
    let retryTimer = null;

    const clearTimers = () => {
      if (retryTimer) {
        window.clearTimeout(retryTimer);
        retryTimer = null;
      }
    };

    const closeSource = () => {
      if (source) {
        try {
          source.close();
        } catch (error) {
          console.debug("Failed to close SSE source", error);
        }
        source = null;
      }
    };

    const cleanup = () => {
      clearTimers();
      closeSource();
    };

    const handleSnapshot = (snapshot) => {
      if (typeof onUpdate === "function") {
        try {
          onUpdate(snapshot);
        } catch (error) {
          console.error("Progress handler failed", error);
        }
      }
      if (snapshot?.status === "failed") {
        settled = true;
        cleanup();
        const message = extractErrorMessage(snapshot) || "Генерация завершилась с ошибкой.";
        const error = new Error(message);
        if (snapshot?.trace_id) {
          error.traceId = snapshot.trace_id;
        }
        reject(error);
      } else if (snapshot?.status === "succeeded" && snapshot.result) {
        settled = true;
        cleanup();
        resolve(snapshot);
      }
    };

    const connect = () => {
      if (settled) {
        return;
      }
      clearTimers();
      closeSource();
      source = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/stream`);
      source.onmessage = (event) => {
        retries = 0;
        let snapshot;
        try {
          snapshot = JSON.parse(event.data);
        } catch (error) {
          console.warn("Failed to parse SSE payload", error);
          return;
        }
        handleSnapshot(snapshot);
      };
      source.onerror = () => {
        if (settled) {
          return;
        }
        retries += 1;
        if (retries > maxRetries) {
          settled = true;
          cleanup();
          watchJobViaPolling(jobId, { onUpdate }).then(resolve).catch(reject);
          return;
        }
        closeSource();
        clearTimers();
        retryTimer = window.setTimeout(connect, Math.min(2000 * retries, 6000));
      };
    };

    connect();
  });
}

async function watchJobViaPolling(jobId, { onUpdate } = {}) {
  let delayMs = 1000;
  while (true) {
    await delay(delayMs);
    const snapshot = await fetchJson(`/api/jobs/${encodeURIComponent(jobId)}`);
    if (typeof onUpdate === "function") {
      onUpdate(snapshot);
    }
    if (snapshot?.status === "failed") {
      const message = snapshot?.error?.message || snapshot?.message || "Генерация завершилась с ошибкой.";
      const error = new Error(message);
      if (snapshot?.trace_id) {
        error.traceId = snapshot.trace_id;
      }
      throw error;
    }
    if (snapshot?.status === "succeeded" && snapshot.result) {
      return snapshot;
    }
    delayMs = Math.min(delayMs + 200, 1500);
  }
}

function renderGenerationResult(snapshot, { payload }) {
  const normalized = normalizeJobResponse(snapshot);
  const result = normalized.result || {};
  const markdown = typeof result.markdown === "string" ? result.markdown : "";
  const meta = (result.meta_json && typeof result.meta_json === "object") ? result.meta_json : {};
  const degradationFlags = Array.isArray(normalized.degradation_flags) ? normalized.degradation_flags : [];
  const characters = typeof meta.characters === "number" ? meta.characters : markdown.replace(/\s+/g, "").length;
  const hasContent = markdown.trim().length > 0;
  state.currentResult = {
    markdown,
    meta,
    artifactPaths: result.artifact_paths ?? null,
    characters,
    hasContent,
    degradationFlags,
    downloads:
      state.currentDownloads && (state.currentDownloads.markdown || state.currentDownloads.report)
        ? { ...state.currentDownloads }
        : null,
  };
  draftView.innerHTML = markdownToHtml(markdown);
  const requestedLabel =
    payload?.data?.title
      || payload?.data?.theme
      || payload?.data?.goal
      || payload?.theme
      || "";
  resultTitle.textContent = requestedLabel || "Результат генерации";
  const metaParts = [];
  if (hasContent) {
    metaParts.push(`Символов: ${characters.toLocaleString("ru-RU")}`);
    metaParts.push(
      `Ориентир: ${DEFAULT_LENGTH_RANGE.min.toLocaleString("ru-RU")}` +
        `–${DEFAULT_LENGTH_RANGE.max.toLocaleString("ru-RU")}` +
        ` (≤ ${DEFAULT_LENGTH_RANGE.hard.toLocaleString("ru-RU")})`,
    );
  }
  if (degradationFlags.length) {
    metaParts.push(`Деградации: ${degradationFlags.length}`);
  }
  if (meta.model_used) {
    metaParts.push(`Модель: ${meta.model_used}`);
  }
  resultMeta.textContent = metaParts.join(" · ") || "Деградаций нет";
  renderMetadata(meta);
  renderUsedKeywords(meta);
  updateResultBadges(meta, degradationFlags);
  toggleRetryButton(!hasContent);
  updatePromptPreview({
    system: meta.system_prompt_preview,
    context: Array.isArray(meta.clips) ? meta.clips : [],
    user: meta.user_prompt_preview,
    context_used: meta.context_used,
    context_index_missing: meta.context_index_missing,
    context_budget_tokens_est: meta.context_budget_tokens_est,
    context_budget_tokens_limit: meta.context_budget_tokens_limit,
    k: payload?.k,
  });
  if (state.currentResult.downloads) {
    setButtonLoading(downloadMdBtn, false);
    setButtonLoading(downloadReportBtn, false);
    setActiveArtifactDownloads(state.currentResult.downloads);
  }
  if (degradationFlags.length) {
    const label = degradationFlags.map(describeDegradationFlag).join(", ");
    showToast({ message: `Частичная деградация: ${label}`, type: "warn", duration: 6000 });
  }
}

function describeDegradationFlag(flag) {
  if (!flag) {
    return "unknown";
  }
  return DEGRADATION_LABELS[flag] || flag;
}

const delay = (ms) => new Promise((resolve) => {
  setTimeout(resolve, ms);
});

function handleRetryClick(event) {
  event.preventDefault();
  if (briefForm && typeof briefForm.requestSubmit === "function") {
    briefForm.requestSubmit(generateBtn);
    return;
  }
  if (generateBtn) {
    generateBtn.click();
  }
}

function parsePositiveInt(value) {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return Math.floor(value);
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const parsed = Number.parseInt(trimmed, 10);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  }
  return null;
}

function buildRequestPayload() {
  const theme = pipeSelect.value;
  if (!theme) {
    throw new Error("Выберите тематику");
  }

  const keywords = keywordsInput.value
    .split(/\r?\n|,/)
    .map((item) => item.trim())
    .filter(Boolean);
  const structure = structureInput
    ? structureInput.value
        .split(/\r?\n/)
        .map((item) => item.trim())
        .filter(Boolean)
    : [];

  const keywordMode = Array.from(keywordModeInputs).find((input) => input.checked)?.value || "strict";
  const styleProfile = styleProfileSelect?.value || "sravni.ru";
  const contextSource = String(contextSourceSelect?.value || "index.json").toLowerCase();
  const contextPayload = resolveCustomContextPayload(contextSource);

  const data = {
    keywords,
    keywords_mode: keywordMode,
    include_faq: true,
    faq_questions: 5,
    include_jsonld: true,
    structure,
    pipe_id: theme,
    style_profile: styleProfile,
    context_source: contextSource,
  };

  const titleValue = titleInput?.value?.trim();
  if (titleValue) {
    data.title = titleValue;
  }

  const goalValue = goalInput?.value?.trim();
  if (goalValue) {
    data.goal = goalValue;
  }

  const audienceValue = audienceInput?.value?.trim();
  if (audienceValue) {
    data.target_audience = audienceValue;
  }

  const toneValue = toneSelect?.value?.trim();
  if (toneValue) {
    data.tone = toneValue;
  }

  const minValue = parsePositiveInt(minCharsInput?.value);
  const maxValue = parsePositiveInt(maxCharsInput?.value);
  const effectiveMin = minValue ?? DEFAULT_LENGTH_RANGE.min;
  const effectiveMax = maxValue ?? DEFAULT_LENGTH_RANGE.max;
  if (effectiveMax < effectiveMin) {
    throw new Error("Максимальный объём должен быть больше или равен минимальному");
  }
  data.length_limits = {
    min_chars: effectiveMin,
    max_chars: effectiveMax,
  };

  const sources = collectSources();
  if (sources.length) {
    data.sources = sources;
  }

  if (contextSource === "custom") {
    data.context_source = "custom";
    if (contextPayload.filename) {
      data.context_filename = contextPayload.filename;
    }
  } else {
    delete data.context_filename;
  }

  const inferredTopic = titleValue || goalValue || "";
  if (inferredTopic) {
    data.theme = inferredTopic;
  }

  const payload = {
    theme,
    data,
    k: 0,
    context_source: contextSource,
  };

  if (contextSource === "custom") {
    payload.context_text = contextPayload.text;
    if (contextPayload.filename) {
      payload.context_filename = contextPayload.filename;
    }
  }

  return payload;
}

function renderMetadata(meta) {
  reportView.innerHTML = "";
  const lengthWarnings = [];
  if (Array.isArray(meta?.length_limits_warnings)) {
    meta.length_limits_warnings.forEach((note) => {
      if (typeof note === "string" && note.trim()) {
        lengthWarnings.push(note.trim());
      }
    });
  }
  if (typeof meta?.length_limits_warning === "string" && meta.length_limits_warning.trim()) {
    lengthWarnings.push(meta.length_limits_warning.trim());
  }
  if (lengthWarnings.length) {
    showToast({ message: lengthWarnings[0], type: "warn" });
  }
  const summary = buildQualityReport(meta);
  if (summary) {
    reportView.append(summary);
  }
  const report = document.createElement("pre");
  report.className = "metadata-view";
  if (!meta || typeof meta !== "object" || !Object.keys(meta).length) {
    report.textContent = "Метаданные недоступны.";
  } else {
    report.textContent = JSON.stringify(meta, null, 2);
  }
  reportView.append(report);
}

function buildQualityReport(meta) {
  if (!meta || typeof meta !== "object") {
    return null;
  }
  const post = meta.post_analysis;
  if (!post || typeof post !== "object") {
    return null;
  }
  const list = document.createElement("ul");
  list.className = "quality-report";
  const appliedLimits =
    post.length_limits_applied && typeof post.length_limits_applied === "object"
      ? post.length_limits_applied
      : null;

  const lengthBlock = post.length;
  if (lengthBlock && typeof lengthBlock === "object") {
    const chars = Number(lengthBlock.chars_no_spaces) || 0;
    const min = Number(
      lengthBlock.min ?? appliedLimits?.min ?? meta.length_limits?.min_chars ?? 0,
    );
    const max = Number(
      lengthBlock.max ?? appliedLimits?.max ?? meta.length_limits?.max_chars ?? 0,
    );
    const within = Boolean(lengthBlock.within_limits);
    const relaxedAccepted =
      (typeof meta?.length_relaxed_status === "string" && meta.length_relaxed_status === "accepted") ||
      (typeof lengthBlock.relaxed_status === "string" && lengthBlock.relaxed_status === "accepted");
    const statusOk = within || relaxedAccepted;
    const rangeLabel = `${min.toLocaleString("ru-RU")}–${max.toLocaleString("ru-RU")}`;
    let label = `Объём: ${chars.toLocaleString("ru-RU")} зн.`;
    if (statusOk && !within && relaxedAccepted) {
      label += ` (length_relaxed, допустимо до ${DEFAULT_LENGTH_RANGE.hard.toLocaleString("ru-RU")})`;
    } else if (statusOk) {
      label += " (в норме)";
    } else {
      label += ` (нужно ${rangeLabel})`;
    }
    list.append(createQualityItem(statusOk ? "success" : "warning", label));
  }

  const limitWarnings = [];
  if (Array.isArray(meta.length_limits_warnings)) {
    meta.length_limits_warnings.forEach((note) => {
      if (typeof note === "string" && note.trim()) {
        limitWarnings.push(note.trim());
      }
    });
  }
  if (typeof meta.length_limits_warning === "string" && meta.length_limits_warning.trim()) {
    limitWarnings.push(meta.length_limits_warning.trim());
  }
  [...new Set(limitWarnings)].forEach((note) => {
    list.append(createQualityItem("warning", note));
  });

  const coverage = Array.isArray(post.keywords_coverage) ? post.keywords_coverage : [];
  if (coverage.length > 0) {
    const foundCount = coverage.filter((item) => item && item.found).length;
    const missingItems = coverage.filter((item) => item && !item.found);
    const total = coverage.length;
    const label = missingItems.length
      ? `Ключевые слова: ${foundCount}/${total} (нет: ${missingItems
          .slice(0, 3)
          .map((item) => item.term)
          .join(", ")}${missingItems.length > 3 ? "…" : ""})`
      : `Ключевые слова: ${foundCount}/${total} найдены`;
    list.append(createQualityItem(missingItems.length ? "warning" : "success", label));
  } else {
    list.append(createQualityItem("info", "Ключевые слова: не заданы"));
  }

  if (meta.include_faq) {
    const targetFaq = Number(meta.faq_questions) || null;
    const actualFaq = Number(post.faq_count) || 0;
    const matches = targetFaq ? actualFaq === targetFaq : actualFaq > 0;
    const label = targetFaq
      ? `FAQ: ${actualFaq} вопросов (ожидание ${targetFaq})`
      : `FAQ: ${actualFaq} вопросов`;
    list.append(createQualityItem(matches ? "success" : "warning", label));
  } else {
    list.append(createQualityItem("info", "FAQ: отключён"));
  }

  if (typeof meta.include_jsonld === "boolean") {
    list.append(
      createQualityItem(meta.include_jsonld ? "success" : "info", meta.include_jsonld ? "JSON-LD: включён" : "JSON-LD: отключён"),
    );
  }

  const extendIterations = Array.isArray(meta.quality_extend_iterations)
    ? meta.quality_extend_iterations
    : [];
  const maxExtend = Number(meta.quality_extend_max_iterations) || 3;
  if (extendIterations.length) {
    extendIterations.forEach((step, index) => {
      if (!step || typeof step !== "object") {
        return;
      }
      const mode = step.mode === "keywords" ? "keywords" : "quality";
      const iteration = Number(step.iteration) || index + 1;
      const before = Number(step.before_chars_no_spaces ?? step.before_chars ?? 0);
      const after = Number(step.after_chars_no_spaces ?? step.after_chars ?? 0);
      const labelBase = mode === "keywords" ? "Extend ключи" : `Extend ${iteration}/${maxExtend}`;
      const label = `${labelBase}: ${before.toLocaleString("ru-RU")} → ${after.toLocaleString("ru-RU")} зн.`;
      let status = "info";
      if (after > before && mode !== "keywords") {
        status = "success";
      } else if (mode === "keywords" && after > before) {
        status = "success";
      }
      list.append(createQualityItem(status, label));
    });
  }
  if (meta.extend_incomplete) {
    list.append(
      createQualityItem(
        "warning",
        "Extend: после трёх попыток объём остаётся ниже минимального требования.",
      ),
    );
  }

  const requestedSources = Array.isArray(meta.sources_requested)
    ? meta.sources_requested.map((item) => item?.value || item)
    : [];
  const usedSourcesRaw = Array.isArray(post.sources_used) ? post.sources_used : [];
  const usedSourcesClean = usedSourcesRaw
    .map((value) => (typeof value === "string" ? value.trim() : ""))
    .filter(Boolean);
  const usedSourcesUnique = Array.from(new Set(usedSourcesClean));
  const usedSourcesLookup = new Set(usedSourcesUnique.map((value) => value.toLowerCase()));
  if (requestedSources.length > 0) {
    const missingSources = requestedSources
      .map((source) => (typeof source === "string" ? source.trim() : ""))
      .filter(Boolean)
      .filter((source) => !usedSourcesLookup.has(source.toLowerCase()));
    const label = missingSources.length
      ? `Источники: ${usedSourcesUnique.length}/${requestedSources.length} (нет: ${missingSources.join(", ")})`
      : `Источники: ${usedSourcesUnique.length}/${requestedSources.length} использованы`;
    list.append(createQualityItem(missingSources.length ? "warning" : "success", label));
    if (usedSourcesUnique.length) {
      list.append(createQualityItem("info", `Использованы: ${usedSourcesUnique.join(", ")}`));
    } else {
      list.append(createQualityItem("warning", "Источники из брифа не использованы"));
    }
  } else if (usedSourcesUnique.length > 0) {
    list.append(createQualityItem("info", `Источники: ${usedSourcesUnique.join(", ")}`));
  }

  if (meta.style_profile) {
    list.append(createQualityItem("info", `Стиль: ${meta.style_profile}`));
  }

  if (meta.model_used) {
    const route = meta.api_route ? `, ${meta.api_route}` : "";
    list.append(createQualityItem("success", `Модель: ${meta.model_used}${route}`));
  }

  const fallbackUsed = Boolean(meta.fallback_used || post.fallback);
  if (fallbackUsed) {
    list.append(createQualityItem("warning", `Fallback: ${describeFallbackNotice(meta.fallback_reason)}`));
  }

  const retries = Number(meta.post_analysis_retry_count ?? post.retry_count ?? 0);
  if (retries > 0) {
    list.append(createQualityItem("info", `Повторов: ${retries}`));
  }

  return list;
}

function createQualityItem(status, text) {
  const item = document.createElement("li");
  item.className = "quality-report__item";
  if (status === "warning") {
    item.dataset.status = "warning";
  } else if (status === "success") {
    item.dataset.status = "success";
  } else {
    item.dataset.status = "info";
  }
  const icon = document.createElement("span");
  icon.className = "quality-report__icon";
  icon.textContent = status === "warning" ? "⚠️" : status === "success" ? "✅" : "ℹ️";
  const label = document.createElement("span");
  label.className = "quality-report__label";
  label.textContent = text;
  item.append(icon, label);
  return item;
}

function renderUsedKeywords(meta) {
  if (!usedKeywordsSection || !usedKeywordsList || !usedKeywordsEmpty) {
    return;
  }

  if (!meta || typeof meta !== "object") {
    usedKeywordsSection.hidden = true;
    usedKeywordsList.innerHTML = "";
    usedKeywordsEmpty.hidden = true;
    return;
  }

  const coverage = Array.isArray(meta.post_analysis?.keywords_coverage)
    ? meta.post_analysis.keywords_coverage
    : [];

  usedKeywordsList.innerHTML = "";
  if (!coverage.length) {
    usedKeywordsSection.hidden = false;
    usedKeywordsList.style.display = "none";
    usedKeywordsEmpty.hidden = false;
    usedKeywordsEmpty.textContent = "Ключевые слова не заданы.";
    return;
  }

  usedKeywordsSection.hidden = false;
  usedKeywordsList.style.display = "flex";
  usedKeywordsEmpty.hidden = true;

  coverage.forEach((entry) => {
    if (!entry || typeof entry.term !== "string") {
      return;
    }
    const li = document.createElement("li");
    const count = Number(entry.count) || 0;
    li.textContent = count > 0 ? `${entry.term} (${count}\u00d7)` : `${entry.term} (нет)`;
    li.dataset.status = entry.found ? "found" : "missing";
    usedKeywordsList.append(li);
  });
}

function updateResultBadges(meta, degradationFlags = []) {
  resultBadges.innerHTML = "";
  if (!meta || typeof meta !== "object") {
    return;
  }

  const appendBadge = (text, type = "neutral") => {
    if (!text) {
      return;
    }
    const badge = document.createElement("span");
    badge.className = `badge ${type}`;
    badge.textContent = text;
    resultBadges.append(badge);
  };

  const post = meta.post_analysis && typeof meta.post_analysis === "object" ? meta.post_analysis : null;
  const currentResult = state.currentResult;
  const characters = typeof meta.characters === "number"
    ? meta.characters
    : currentResult?.characters ?? (currentResult?.markdown?.trim().length ?? 0);
  const hasContent = Boolean(currentResult?.hasContent ?? (characters > 0));
  if (!hasContent) {
    appendBadge("Пустой ответ", "warning");
  }
  const appliedLimits =
    post && typeof post.length_limits_applied === "object" ? post.length_limits_applied : null;
  const lengthInfo = post?.length && typeof post.length === "object" ? post.length : null;
  if (lengthInfo) {
    const chars = Number(lengthInfo.chars_no_spaces ?? meta.characters_no_spaces ?? meta.characters) || 0;
    const within = Boolean(lengthInfo.within_limits);
    const min = Number(lengthInfo.min ?? appliedLimits?.min ?? meta.length_limits?.min_chars ?? 0);
    const max = Number(lengthInfo.max ?? appliedLimits?.max ?? meta.length_limits?.max_chars ?? 0);
    const label = within
      ? `Объём ${chars.toLocaleString("ru-RU")} зн.`
      : `Объём ${chars.toLocaleString("ru-RU")} (нужно ${min}–${max})`;
    appendBadge(label, within ? "success" : "warning");
  }

  const coverage = Array.isArray(post?.keywords_coverage) ? post.keywords_coverage : [];
  const hasKeywordsInBrief = Array.isArray(meta.input_data?.keywords) && meta.input_data.keywords.length > 0;
  if (coverage.length > 0) {
    const found = coverage.filter((item) => item && item.found).length;
    const total = coverage.length;
    const missing = total - found;
    appendBadge(`Ключи ${found}/${total}`, missing > 0 ? "warning" : "success");
  } else if (hasKeywordsInBrief) {
    appendBadge("Ключи: нет данных", "warning");
  } else {
    appendBadge("Ключи: не заданы", "neutral");
  }

  const extendIterations = Array.isArray(meta.quality_extend_iterations)
    ? meta.quality_extend_iterations
    : [];
  const maxExtend = Number(meta.quality_extend_max_iterations) || 3;
  extendIterations.forEach((step, index) => {
    if (!step || typeof step !== "object") {
      return;
    }
    const mode = step.mode === "keywords" ? "keywords" : "quality";
    const iteration = Number(step.iteration) || index + 1;
    const before = Number(step.before_chars_no_spaces ?? step.before_chars ?? 0);
    const after = Number(step.after_chars_no_spaces ?? step.after_chars ?? 0);
    if (mode === "keywords") {
      const grew = after >= before && after > 0;
      appendBadge(
        "Extend ключи",
        grew ? "success" : "warning",
      );
    } else {
      const grew = after > before;
      appendBadge(`Extend ${iteration}/${maxExtend}`, grew ? "success" : "neutral");
    }
  });
  if (meta.extend_incomplete) {
    appendBadge("Extend: объём ниже минимума", "warning");
  }

  if (meta.include_faq) {
    const actualFaq = Number(post?.faq_count) || 0;
    const targetFaq = Number(meta.faq_questions) || null;
    const ok = targetFaq ? actualFaq === targetFaq : actualFaq > 0;
    const faqLabel = targetFaq ? `FAQ ${actualFaq}/${targetFaq}` : `FAQ ${actualFaq}`;
    appendBadge(faqLabel, ok ? "success" : "warning");
  } else if (Number(post?.faq_count) > 0) {
    appendBadge(`FAQ ${post.faq_count}`, "neutral");
  }

  if (typeof meta.include_jsonld === "boolean") {
    appendBadge(meta.include_jsonld ? "JSON-LD" : "Без JSON-LD", meta.include_jsonld ? "success" : "neutral");
  }

  const requestedRaw = Array.isArray(meta.sources_requested) ? meta.sources_requested : [];
  const requestedValues = requestedRaw
    .map((item) => (typeof item === "string" ? item : item && item.value))
    .map((value) => (typeof value === "string" ? value.trim() : ""))
    .filter(Boolean);
  const requestedCount = requestedValues.length;
  const usedSources = Array.isArray(post?.sources_used) ? post.sources_used : [];
  if (requestedCount > 0) {
    const normalizedUsed = Array.from(new Set(usedSources.map((source) => String(source).trim().toLowerCase()).filter(Boolean)));
    const usedCount = normalizedUsed.length;
    const missingCount = requestedValues.filter((value) => !normalizedUsed.includes(value.toLowerCase())).length;
    appendBadge(`Источники ${usedCount}/${requestedCount}`, missingCount > 0 ? "warning" : "success");
  } else if (usedSources.length > 0) {
    appendBadge(`Источники: ${usedSources.length}`, "neutral");
  }

  const fallbackUsed = Boolean(meta.fallback_used || post?.fallback);
  if (fallbackUsed) {
    appendBadge(`Fallback: ${describeFallbackNotice(meta.fallback_reason)}`, "warning");
  }

  const modelLabel = meta.model_used || meta.model;
  if (modelLabel) {
    appendBadge(`Модель: ${modelLabel}`, fallbackUsed ? "warning" : "neutral");
  }

  const retries = Number(meta.post_analysis_retry_count ?? post?.retry_count ?? 0);
  if (retries > 0) {
    appendBadge(`Повторов: ${retries}`, "neutral");
  }

  if (meta.length_adjustment) {
    appendBadge(`Коррекция длины: ${meta.length_adjustment}`, "neutral");
  }

  if (Array.isArray(degradationFlags) && degradationFlags.length) {
    [...new Set(degradationFlags)].forEach((flag) => {
      appendBadge(describeDegradationFlag(flag), "warning");
    });
  }
}

const FALLBACK_REASON_MESSAGES = {
  model_unavailable: "Основная модель недоступна для текущего ключа или тарифа.",
  empty_completion: "Основная модель вернула пустой ответ.",
};

function describeFallbackNotice(reason) {
  if (!reason) {
    return "Причина не указана.";
  }
  return FALLBACK_REASON_MESSAGES[reason] ?? `Причина: ${reason}`;
}

function updatePromptPreview(preview) {
  if (!preview) {
    previewSystem.textContent = "";
    previewUser.textContent = "";
    contextList.innerHTML = "";
    contextSummary.textContent = "";
    contextBadge.textContent = "не запрошен";
    contextBadge.className = "badge neutral";
    return;
  }
  previewSystem.textContent = preview.system ?? "";
  previewUser.textContent = preview.user ?? "";
  const contextSource = String(preview.context_source || "index.json").toLowerCase();
  const isCustom = contextSource === "custom";
  const contextUsed = Boolean(preview.context_used);
  const indexMissing = Boolean(preview.context_index_missing);
  const k = preview.k ?? 0;
  contextList.innerHTML = "";

  if (isCustom) {
    const customText = typeof preview.context_text === "string" ? preview.context_text : "";
    const filename = typeof preview.context_filename === "string" ? preview.context_filename : "";
    if (customText) {
      const li = document.createElement("li");
      li.className = "context-custom";
      const titleEl = document.createElement("strong");
      titleEl.textContent = filename || "Пользовательский контекст";
      const pre = document.createElement("pre");
      pre.className = "context-custom__text";
      pre.textContent = customText;
      li.append(titleEl);
      li.append(pre);
      contextList.append(li);
    } else {
      contextList.innerHTML = '<li>Пользовательский контекст пуст.</li>';
    }
  } else {
    (preview.context || []).forEach((item, idx) => {
      const li = document.createElement("li");
      const title = item.path || `Фрагмент #${idx + 1}`;
      const score = Number(item.score ?? 0).toFixed(2);
      const textFragment = typeof item.text === "string" ? item.text : "";
      li.innerHTML = `<strong>${escapeHtml(title)}</strong><span>score: ${score}</span><br />${escapeHtml(
        textFragment.slice(0, 320)
      )}${textFragment.length > 320 ? "…" : ""}`;
      contextList.append(li);
    });
    if (!preview.context || !preview.context.length) {
      contextList.innerHTML = '<li>Контекст не используется.</li>';
    }
  }

  if (isCustom) {
    contextBadge.textContent = contextUsed ? "custom" : "custom (пустой)";
    contextBadge.className = contextUsed ? "badge success" : "badge warning";
  } else if (contextSource === "off") {
    contextBadge.textContent = "off";
    contextBadge.className = "badge neutral";
  } else if (k === 0) {
    contextBadge.textContent = "disabled";
    contextBadge.className = "badge neutral";
  } else if (indexMissing) {
    contextBadge.textContent = "index missing";
    contextBadge.className = "badge error";
  } else if (!contextUsed) {
    contextBadge.textContent = "none";
    contextBadge.className = "badge neutral";
  } else {
    contextBadge.textContent = `used (${preview.context.length})`;
    contextBadge.className = "badge success";
  }

  if (isCustom) {
    const lengthValue = Number(preview.context_len || 0);
    contextSummary.textContent = lengthValue
      ? `Пользовательский контекст: ${lengthValue.toLocaleString("ru-RU")} символов`
      : "";
  } else if (typeof preview.context_budget_tokens_est === "number") {
    const limit = preview.context_budget_tokens_limit;
    contextSummary.textContent = limit
      ? `Контекст: ~${preview.context_budget_tokens_est} токенов из лимита ${limit}`
      : `Контекст: ~${preview.context_budget_tokens_est} токенов`;
  } else {
    contextSummary.textContent = "";
  }
}

function toggleRetryButton(show) {
  if (!retryBtn) {
    return;
  }
  retryBtn.hidden = !show;
  retryBtn.disabled = !show;
}

async function showArtifact(artifact) {
  try {
    showProgress(true, "Загружаем артефакт…");
    const markdownDownload = artifact?.downloads?.markdown || null;
    const markdownUrl = markdownDownload?.url
      || (artifact.path ? `/api/artifacts/download?path=${encodeURIComponent(artifact.path)}` : null);
    if (!markdownUrl) {
      handleMissingArtifact(artifact);
      return;
    }
    const markdown = await fetchText(markdownUrl);
    const metadataPath = artifact.metadata_path;
    const metadataDownload = artifact?.downloads?.report || null;
    let metadata = artifact.metadata || {};
    if (!Object.keys(metadata).length && (metadataDownload?.url || metadataPath)) {
      try {
        const metadataUrl = metadataDownload?.url
          || (metadataPath ? `/api/artifacts/download?path=${encodeURIComponent(metadataPath)}` : null);
        const jsonText = metadataUrl ? await fetchText(metadataUrl) : "";
        metadata = JSON.parse(jsonText);
      } catch (parseError) {
        console.warn("Не удалось разобрать метаданные", parseError);
        metadata = {};
      }
    }
    draftView.innerHTML = markdownToHtml(markdown);
    resultTitle.textContent = metadata.input_data?.theme || metadata.theme || artifact.name;
    const characters = metadata.characters ?? markdown.length;
    resultMeta.textContent = `Символов: ${characters.toLocaleString("ru-RU")} · Модель: ${metadata.model_used ?? "—"}`;
    renderMetadata(metadata);
    updateResultBadges(metadata, Array.isArray(metadata?.degradation_flags) ? metadata.degradation_flags : []);
    setButtonLoading(downloadMdBtn, false);
    setButtonLoading(downloadReportBtn, false);
    setActiveArtifactDownloads(artifact.downloads || {
      markdown: markdownDownload,
      report: metadataDownload,
    });
    updatePromptPreview({
      system: metadata.system_prompt_preview,
      context: metadata.clips || [],
      user: metadata.user_prompt_preview,
      context_used: metadata.context_used,
      context_index_missing: metadata.context_index_missing,
      context_budget_tokens_est: metadata.context_budget_tokens_est,
      context_budget_tokens_limit: metadata.context_budget_tokens_limit,
      k: metadata.retrieval_k,
      context_source: metadata.context_source,
      context_text: metadata.custom_context_text,
      context_len: metadata.context_len,
      context_filename: metadata.context_filename,
    });
    state.currentResult = {
      markdown,
      meta: metadata,
      artifactPaths: { markdown: artifact.path || null, metadata: metadataPath || null },
      characters,
      hasContent: true,
      degradationFlags: Array.isArray(metadata?.degradation_flags) ? metadata.degradation_flags : [],
      downloads: artifact.downloads || {
        markdown: markdownDownload,
        report: metadataDownload,
      },
    };
    switchTab("result");
  } catch (error) {
    console.error(error);
    if (isNotFoundError(error)) {
      handleMissingArtifact(artifact);
      return;
    }
    showToast({ message: `Не удалось открыть артефакт: ${getErrorMessage(error)}`, type: "error" });
  } finally {
    showProgress(false);
  }
}

async function handleReindex(event) {
  if (event) {
    event.preventDefault();
  }
  const theme = pipeSelect.value;
  if (!theme) {
    showToast({ message: "Выберите тематику для переиндексации", type: "warn" });
    return;
  }

  appendLogEntry("info", `Запрос на переиндексацию темы «${theme}»`);
  try {
    setInteractiveBusy(true);
    setButtonLoading(reindexBtn, true);
    const stats = await fetchJson("/api/reindex", {
      method: "POST",
      body: JSON.stringify({ theme }),
    });
    appendLogEntry("success", summariseReindexStats(stats));
    if (typeof stats?.clips === "number" && typeof stats?.avg_truncated_words === "number") {
      const details = [
        `Клипы: ${stats.clips}`,
        `средняя длина ~${Math.round(stats.avg_truncated_words)} слов`,
      ];
      if (typeof stats.avg_truncated_tokens_est === "number") {
        details.push(`≈${Math.round(stats.avg_truncated_tokens_est)} токенов`);
      }
      appendLogEntry("info", details.join(", "));
    }
    showToast({ message: "Индекс обновлён", type: "success" });
    await handleHealthCheck();
  } catch (error) {
    console.error(error);
    appendLogEntry("error", getErrorMessage(error));
    showToast({ message: `Переиндексация: ${getErrorMessage(error)}`, type: "error" });
  } finally {
    setButtonLoading(reindexBtn, false);
    setInteractiveBusy(false);
  }
}

async function handleHealthCheck(event) {
  const triggeredByUser = event instanceof Event;
  if (triggeredByUser) {
    event.preventDefault();
    setInteractiveBusy(true);
    setButtonLoading(healthBtn, true);
  }
  try {
    const theme = pipeSelect.value;
    const query = theme ? `?theme=${encodeURIComponent(theme)}` : "";
    const response = await fetch(`${API_BASE}/api/health${query}`);
    let text = "";
    try {
      text = await response.text();
    } catch (readError) {
      renderHealthError("Ошибка при запросе Health", "error");
      return;
    }

    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (parseError) {
        renderHealthError("Ошибка при разборе Health", "error");
        return;
      }
    }

    if (data && typeof data === "object" && data.checks) {
      renderHealthStatus(data);
      return;
    }

    const errorMessage =
      (data && typeof data === "object" && (data.error || data.message)) ||
      text ||
      `HTTP ${response.status}`;
    throw new Error(errorMessage);
  } catch (error) {
    console.error(error);
    if (error instanceof TypeError) {
      renderHealthError("Сервер недоступен", "offline");
      return;
    }
    renderHealthError(error.message || "Ошибка при запросе Health", "error");
  } finally {
    if (triggeredByUser) {
      setButtonLoading(healthBtn, false);
      setInteractiveBusy(false);
    }
  }
}

function renderHealthStatus(status) {
  healthStatus.innerHTML = "";
  const checks = status?.checks;
  if (!checks || typeof checks !== "object" || !Object.keys(checks).length) {
    renderHealthError("Нет данных", "error");
    return;
  }

  const entries = Object.entries(checks);

  entries.forEach(([key, value], index) => {
    const normalized = normalizeHealthCheck(value);
    const tone = normalized.ok ? "success" : "error";
    const icon = normalized.ok ? "🟢" : "🔴";
    const dictionary = HEALTH_STATUS_MESSAGES[key] || {};
    const label = dictionary.label || key.replace(/_/g, " ");
    const description =
      normalized.message || (normalized.ok ? dictionary.ok : dictionary.fail) || (normalized.ok ? "активен" : "недоступен");

    const card = createHealthCard({
      tone,
      icon,
      label,
      description,
      delay: index * 80,
    });
    card.dataset.healthKey = key;
    healthStatus.append(card);
  });
  const bad = entries.some(([, value]) => value && value.ok === false);
  const overallOk = status?.ok === true && !bad;
  const failingEntry = entries.find(([, value]) => value && value.ok === false);
  let failingMessage = "";
  if (failingEntry) {
    const [, rawValue] = failingEntry;
    if (rawValue && typeof rawValue === "object" && typeof rawValue.message === "string") {
      failingMessage = rawValue.message;
    } else {
      const dictionary = HEALTH_STATUS_MESSAGES[failingEntry[0]] || {};
      failingMessage = dictionary.fail || "Модель недоступна";
    }
  }
  setGenerateAvailability(overallOk, failingMessage);
}

function normalizeHealthCheck(value) {
  if (value && typeof value === "object") {
    return {
      ok: value.ok === true,
      message: (typeof value.message === "string" && value.message) || value.status || "",
    };
  }
  return {
    ok: Boolean(value),
    message: "",
  };
}

function renderHealthError(message, tone = "error") {
  healthStatus.innerHTML = "";
  const icon = tone === "success" ? "🟢" : tone === "offline" ? "⚪" : "🔴";
  const card = createHealthCard({
    tone,
    icon,
    label: message,
    description: tone === "offline" ? "Попробуйте обновить позже" : "",
  });
  healthStatus.append(card);
  setGenerateAvailability(false, message);
}

function createHealthCard({ tone, icon, label, description = "", delay = 0 }) {
  const card = document.createElement("div");
  card.className = `health-card ${tone}`;
  card.style.animationDelay = `${delay}ms`;

  const iconEl = document.createElement("span");
  iconEl.className = "health-icon";
  iconEl.textContent = icon;

  const contentEl = document.createElement("div");
  contentEl.className = "health-content";

  const labelEl = document.createElement("div");
  labelEl.className = "health-label";
  labelEl.textContent = label;
  contentEl.append(labelEl);

  if (description) {
    const descriptionEl = document.createElement("div");
    descriptionEl.className = "health-description";
    descriptionEl.textContent = description;
    contentEl.append(descriptionEl);
  }

  card.append(iconEl, contentEl);
  return card;
}

function setGenerateAvailability(ok, reason = "") {
  if (!generateBtn) {
    return;
  }
  if (ok) {
    generateBtn.disabled = false;
    generateBtn.removeAttribute("title");
  } else {
    generateBtn.disabled = true;
    if (reason) {
      generateBtn.title = reason;
    } else {
      generateBtn.removeAttribute("title");
    }
  }
}

async function fetchJson(path, options = {}) {
  const headers = options.headers ? { ...options.headers } : {};
  if (options.method && options.method !== "GET") {
    headers["Content-Type"] = "application/json";
  }

  const url = resolveApiPath(path);
  let response;
  try {
    response = await fetch(url, { ...options, headers });
  } catch (error) {
    throw new Error("Сервер недоступен");
  }

  let text;
  try {
    text = await response.text();
  } catch (error) {
    throw new Error("Не удалось прочитать ответ сервера");
  }

  if (!response.ok) {
    let message = text || `HTTP ${response.status}`;
    if (text) {
      try {
        const data = JSON.parse(text);
        if (data && typeof data.error === "string" && data.error.trim()) {
          message = data.error.trim();
        } else if (data && data.error && typeof data.error.message === "string" && data.error.message.trim()) {
          message = data.error.message.trim();
        }
      } catch (parseError) {
        if (!(parseError instanceof Error) || parseError.name !== "SyntaxError") {
          parseError.status = response.status;
          throw parseError;
        }
      }
    }
    const error = new Error(message || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }

  if (!text.trim()) {
    return {};
  }

  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error("Некорректный JSON в ответе сервера");
  }
}

async function fetchText(path) {
  const url = resolveApiPath(path);
  let response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new Error("Сервер недоступен");
  }
  const text = await response.text();
  if (!response.ok) {
    const error = new Error(text || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return text;
}

function setupAdvancedSettings() {
  if (!advancedSettings) {
    return;
  }

  if (advancedSupportSection && devActionsConfig.show && devActionsConfig.hasExplicit) {
    const title = advancedSupportSection.querySelector(".advanced-section-title");
    if (title) {
      title.textContent = "Техподдержка";
    }
    const caption = advancedSupportSection.querySelector(".advanced-section-caption");
    if (caption) {
      caption.remove();
    }
  }

  try {
    const saved = window.localStorage.getItem(ADVANCED_SETTINGS_STORAGE_KEY);
    if (saved === "open") {
      advancedSettings.setAttribute("open", "");
    } else if (saved === "closed") {
      advancedSettings.removeAttribute("open");
    }
  } catch (error) {
    console.debug("Не удалось восстановить состояние расширенных настроек", error);
  }

  advancedSettings.addEventListener("toggle", () => {
    try {
      window.localStorage.setItem(
        ADVANCED_SETTINGS_STORAGE_KEY,
        advancedSettings.open ? "open" : "closed"
      );
    } catch (error) {
      console.debug("Не удалось сохранить состояние расширенных настроек", error);
    }
  });
}

function resetProgressIndicator(message = DEFAULT_PROGRESS_MESSAGE) {
  if (progressStage) {
    const fallbackStage = progressStage.dataset.defaultLabel
      || progressStage.textContent
      || "Подготовка…";
    progressStage.textContent = fallbackStage;
  }
  if (progressPercent) {
    progressPercent.textContent = "0%";
  }
  if (progressBarFill) {
    progressBarFill.style.width = "0%";
  }
  if (progressBar) {
    progressBar.setAttribute("aria-valuenow", "0");
  }
  if (progressMessage) {
    progressMessage.textContent = message;
  }
  if (progressDetails) {
    progressDetails.textContent = "";
  }
  progressState.currentPercent = 0;
  progressState.lastStage = "draft";
}

function showProgress(visible, message = DEFAULT_PROGRESS_MESSAGE) {
  if (!progressOverlay) {
    return;
  }
  if (visible) {
    if (progressState.hideTimer) {
      window.clearTimeout(progressState.hideTimer);
      progressState.hideTimer = null;
    }
    progressOverlay.classList.remove("hidden");
    resetProgressIndicator(message);
  } else {
    hideProgressOverlay({ immediate: true });
  }
}

function hideProgressOverlay({ immediate = false } = {}) {
  if (!progressOverlay) {
    return;
  }
  if (!immediate && progressState.hideTimer) {
    return;
  }
  if (progressState.hideTimer) {
    window.clearTimeout(progressState.hideTimer);
    progressState.hideTimer = null;
  }
  progressOverlay.classList.add("hidden");
  resetProgressIndicator(DEFAULT_PROGRESS_MESSAGE);
}

function scheduleProgressHide(delay = 1200) {
  if (!progressOverlay) {
    return;
  }
  if (progressState.hideTimer) {
    window.clearTimeout(progressState.hideTimer);
  }
  progressState.hideTimer = window.setTimeout(() => {
    progressState.hideTimer = null;
    hideProgressOverlay({ immediate: true });
  }, Math.max(0, delay));
}

function clamp01(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return 0;
  }
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function toNumber(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function formatProgressDetails(stage, payload) {
  if (!payload || typeof payload !== "object") {
    return "";
  }
  const parts = [];
  if (stage === "draft") {
    const completed = toNumber(payload.completed);
    const total = toNumber(payload.total);
    if (Number.isFinite(completed) && Number.isFinite(total) && total > 0) {
      const safeCompleted = Math.max(0, Math.min(total, completed));
      parts.push(`Батчи: ${safeCompleted}/${total}`);
    }
    if (typeof payload.batch === "string" && payload.batch.trim()) {
      parts.push(`Блок: ${payload.batch.trim()}`);
    }
    if (payload.partial) {
      parts.push("частично");
    }
  } else if (stage === "trim") {
    const chars = toNumber(payload.chars);
    if (Number.isFinite(chars) && chars > 0) {
      parts.push(`Символов: ${Math.round(chars).toLocaleString("ru-RU")}`);
    }
  } else if (stage === "validate") {
    const length = toNumber(payload.length ?? payload.chars);
    if (Number.isFinite(length) && length > 0) {
      parts.push(`Символов: ${Math.round(length).toLocaleString("ru-RU")}`);
    }
    const faqCount = toNumber(payload.faq ?? payload.faq_count);
    if (Number.isFinite(faqCount)) {
      parts.push(`FAQ: ${Math.round(faqCount)}`);
    }
  }
  return parts.join(" · ");
}

function extractErrorMessage(snapshot) {
  if (!snapshot || typeof snapshot !== "object") {
    return "";
  }
  const errorPayload = snapshot.error;
  if (errorPayload && typeof errorPayload === "object") {
    if (typeof errorPayload.message === "string" && errorPayload.message.trim()) {
      return errorPayload.message.trim();
    }
    if (typeof errorPayload.error === "string" && errorPayload.error.trim()) {
      return errorPayload.error.trim();
    }
  } else if (typeof errorPayload === "string" && errorPayload.trim()) {
    return errorPayload.trim();
  }
  if (typeof snapshot.message === "string" && snapshot.message.trim()) {
    return snapshot.message.trim();
  }
  return "";
}

function updateProgressFromSnapshot(snapshot) {
  if (!progressOverlay || !snapshot || typeof snapshot !== "object") {
    return;
  }
  if (progressState.hideTimer) {
    window.clearTimeout(progressState.hideTimer);
    progressState.hideTimer = null;
  }

  progressOverlay.classList.remove("hidden");

  const status = typeof snapshot.status === "string" ? snapshot.status : "running";
  let stage = typeof snapshot.progress_stage === "string" && snapshot.progress_stage.trim()
    ? snapshot.progress_stage.trim().toLowerCase()
    : "";
  if (!stage && typeof snapshot.step === "string" && snapshot.step.trim()) {
    stage = snapshot.step.trim().toLowerCase();
  }
  if (status === "succeeded") {
    stage = "done";
  } else if (status === "failed") {
    stage = "error";
  }
  if (!PROGRESS_STAGE_LABELS[stage]) {
    stage = progressState.lastStage || "draft";
  }
  progressState.lastStage = stage;

  let percentValue = null;
  if (typeof snapshot.progress === "number") {
    percentValue = Math.round(clamp01(snapshot.progress) * 1000) / 10;
  }
  if (percentValue === null || Number.isNaN(percentValue)) {
    percentValue = progressState.currentPercent || 0;
  }
  percentValue = Math.max(progressState.currentPercent || 0, percentValue);
  percentValue = Math.min(100, percentValue);
  progressState.currentPercent = percentValue;

  if (progressBarFill) {
    progressBarFill.style.width = `${percentValue}%`;
  }
  if (progressPercent) {
    progressPercent.textContent = `${Math.round(percentValue)}%`;
  }
  if (progressBar) {
    progressBar.setAttribute("aria-valuenow", String(Math.round(percentValue)));
  }

  let message = "";
  if (typeof snapshot.progress_message === "string" && snapshot.progress_message.trim()) {
    message = snapshot.progress_message.trim();
  } else if (status === "succeeded") {
    message = PROGRESS_STAGE_MESSAGES.done;
  } else if (status === "failed") {
    message = extractErrorMessage(snapshot) || PROGRESS_STAGE_MESSAGES.error;
  } else if (typeof snapshot.message === "string" && snapshot.message.trim()) {
    message = snapshot.message.trim();
  } else {
    message = PROGRESS_STAGE_MESSAGES[stage] || DEFAULT_PROGRESS_MESSAGE;
  }
  if (progressMessage) {
    progressMessage.textContent = message;
  }

  if (progressStage) {
    const label = PROGRESS_STAGE_LABELS[stage] || PROGRESS_STAGE_LABELS.draft;
    progressStage.textContent = `Шаг: ${label} 0→100%`;
  }

  const payload = snapshot.progress_payload && typeof snapshot.progress_payload === "object"
    ? snapshot.progress_payload
    : null;
  if (progressDetails) {
    progressDetails.textContent = formatProgressDetails(stage, payload);
  }

  if (status === "succeeded") {
    progressState.currentPercent = 100;
    if (progressPercent) {
      progressPercent.textContent = "100%";
    }
    if (progressBarFill) {
      progressBarFill.style.width = "100%";
    }
    if (progressBar) {
      progressBar.setAttribute("aria-valuenow", "100");
    }
    scheduleProgressHide(1200);
  } else if (status === "failed") {
    progressState.currentPercent = 100;
    if (progressPercent) {
      progressPercent.textContent = "100%";
    }
    if (progressBarFill) {
      progressBarFill.style.width = "100%";
    }
    if (progressBar) {
      progressBar.setAttribute("aria-valuenow", "100");
    }
    scheduleProgressHide(2500);
  }
}

function setInteractiveBusy(isBusy) {
  interactiveElements.forEach((element) => {
    if (!element) {
      return;
    }
    if (isBusy) {
      if (typeof element.dataset.interactiveLocked === "undefined") {
        element.dataset.interactiveLocked = element.disabled ? "true" : "false";
      }
      element.disabled = true;
    } else if (typeof element.dataset.interactiveLocked !== "undefined") {
      const shouldRemainDisabled = element.dataset.interactiveLocked === "true";
      if (!shouldRemainDisabled) {
        element.disabled = false;
      }
      delete element.dataset.interactiveLocked;
    }
  });

  if (advancedSettings) {
    advancedSettings.classList.toggle("is-disabled", isBusy);
  }
}

function setButtonLoading(button, isLoading) {
  if (!button) {
    return;
  }
  if (isLoading) {
    if (!button.dataset.originalDisabled) {
      button.dataset.originalDisabled = button.disabled ? "true" : "false";
    }
    button.disabled = true;
    button.classList.add("loading");
    if (!button.querySelector(".btn-spinner")) {
      const spinner = document.createElement("span");
      spinner.className = "btn-spinner";
      button.prepend(spinner);
    }
  } else {
    const spinner = button.querySelector(".btn-spinner");
    if (spinner) {
      spinner.remove();
    }
    button.classList.remove("loading");
    const shouldRemainDisabled = button.dataset.originalDisabled === "true";
    if (!shouldRemainDisabled) {
      button.disabled = false;
    }
    delete button.dataset.originalDisabled;
  }
}

function toggleFeatureElements(elements, hidden) {
  elements
    .filter(Boolean)
    .forEach((element) => {
      const target = element.closest?.("[data-feature-root]")
        || element.closest?.(".form-row")
        || element;
      if (!target) {
        return;
      }
      if (hidden) {
        target.classList.add("feature-hidden");
      } else {
        target.classList.remove("feature-hidden");
      }
    });
}

function applyFeatureFlags() {
  const modelElements = [];
  document.querySelectorAll('[data-feature="model-selector"]').forEach((element) => {
    modelElements.push(element);
  });
  toggleFeatureElements(modelElements, featureState.hideModelSelector);

  const tokenElements = [];
  if (minCharsInput) {
    tokenElements.push(minCharsInput);
  }
  if (maxCharsInput) {
    tokenElements.push(maxCharsInput);
  }
  document.querySelectorAll('[data-feature="token-sliders"]').forEach((element) => {
    tokenElements.push(element);
  });
  toggleFeatureElements(tokenElements, featureState.hideTokenSliders);
}

async function initFeatureFlags() {
  applyFeatureFlags();
  try {
    const config = await fetchJson("/api/features");
    if (config && typeof config === "object") {
      if (typeof config.hide_model_selector === "boolean") {
        featureState.hideModelSelector = config.hide_model_selector;
      }
      if (typeof config.hide_token_sliders === "boolean") {
        featureState.hideTokenSliders = config.hide_token_sliders;
      }
    }
  } catch (error) {
    console.debug("Не удалось загрузить настройки фич", error);
  } finally {
    applyFeatureFlags();
  }
}

function resolveDevActions() {
  let rawValue;
  if (typeof globalThis !== "undefined" && typeof globalThis.SHOW_DEV_ACTIONS !== "undefined") {
    rawValue = globalThis.SHOW_DEV_ACTIONS;
  } else if (typeof document !== "undefined" && document.body?.dataset?.showDevActions) {
    rawValue = document.body.dataset.showDevActions;
  }

  const hasExplicit = typeof rawValue !== "undefined" && rawValue !== null;
  let show = true;

  if (hasExplicit) {
    if (typeof rawValue === "string") {
      const normalized = rawValue.trim().toLowerCase();
      show = normalized === "true" || normalized === "1";
    } else if (typeof rawValue === "boolean") {
      show = rawValue;
    } else {
      show = Boolean(rawValue);
    }
  }

  return { show, hasExplicit };
}

function getErrorMessage(error, fallback = "Неизвестная ошибка") {
  if (!error) {
    return fallback;
  }
  if (typeof error === "string") {
    return error;
  }
  if (typeof error.message === "string" && error.message.trim()) {
    return error.message;
  }
  return fallback;
}

function isNotFoundError(error) {
  return Boolean(error && typeof error.status === "number" && error.status === 404);
}

async function downloadArtifactFile(resource, fallbackName = "artifact.txt", artifact = null) {
  const normalized = (() => {
    if (!resource) {
      return null;
    }
    if (typeof resource === "string") {
      const trimmed = resource.trim();
      if (!trimmed) {
        return null;
      }
      return {
        url: resolveApiPath(`/api/artifacts/download?path=${encodeURIComponent(trimmed)}`),
        name: fallbackName,
      };
    }
    if (typeof resource === "object") {
      const candidateUrl = typeof resource.url === "string" && resource.url.trim() ? resource.url.trim() : null;
      const candidatePath = typeof resource.path === "string" && resource.path.trim() ? resource.path.trim() : null;
      const name = typeof resource.name === "string" && resource.name.trim() ? resource.name.trim() : fallbackName;
      if (candidateUrl) {
        return { url: resolveApiPath(candidateUrl), name };
      }
      if (candidatePath) {
        return {
          url: resolveApiPath(`/api/artifacts/download?path=${encodeURIComponent(candidatePath)}`),
          name,
        };
      }
    }
    return null;
  })();

  if (!normalized) {
    showToast({ message: "Файл недоступен", type: "warn" });
    return "error";
  }

  try {
    const response = await fetch(normalized.url);
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      let notFound = false;
      let raw = "";
      try {
        raw = await response.text();
      } catch (readError) {
        raw = "";
      }
      if (raw) {
        message = raw;
        try {
          const data = JSON.parse(raw);
          if (data && typeof data.error === "string") {
            message = data.error;
            if (response.status === 404 && data.error === "file_not_found") {
              notFound = true;
              message = "Файл не найден";
            }
          }
        } catch (parseError) {
          // ignore JSON parse failures
        }
      }
      const error = new Error(message || `HTTP ${response.status}`);
      error.status = response.status;
      if (notFound) {
        error.code = "file_not_found";
      }
      throw error;
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = normalized.name || fallbackName;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    return "ok";
  } catch (error) {
    console.error(error);
    if (isNotFoundError(error)) {
      const message = error.code === "file_not_found" ? "Файл не найден" : getErrorMessage(error);
      showToast({ message, type: "warn" });
      if (artifact) {
        handleMissingArtifact(artifact);
      }
      return error.code === "file_not_found" ? "not_found" : "error";
    }
    showToast({ message: `Не удалось скачать файл: ${getErrorMessage(error)}`, type: "error" });
    return "error";
  }
}

function appendLogEntry(status, message) {
  if (!reindexLog) {
    return;
  }
  const tone = LOG_STATUS_LABELS[status] ? status : "info";
  if (reindexLog.querySelector(".log-empty")) {
    reindexLog.innerHTML = "";
  }
  const entry = document.createElement("div");
  entry.className = `log-entry ${tone}`;

  const timeEl = document.createElement("span");
  timeEl.className = "log-time";
  timeEl.textContent = formatLogTime(new Date());

  const statusEl = document.createElement("span");
  statusEl.className = "log-status";
  statusEl.textContent = LOG_STATUS_LABELS[tone];

  const messageEl = document.createElement("p");
  messageEl.className = "log-message";
  messageEl.textContent = message;

  entry.append(timeEl, statusEl, messageEl);
  reindexLog.append(entry);
  reindexLog.scrollTop = reindexLog.scrollHeight;
}

function clearReindexLog() {
  if (!reindexLog) {
    return;
  }
  reindexLog.innerHTML = '<div class="log-empty">Журнал пуст.</div>';
}

function summariseReindexStats(stats) {
  if (!stats || typeof stats !== "object") {
    return "Индекс обновлён.";
  }
  const parts = [];
  if (typeof stats.clips === "number") {
    parts.push(`${stats.clips} клипов`);
  }
  if (typeof stats.avg_truncated_words === "number") {
    parts.push(`~${Math.round(stats.avg_truncated_words)} слов`);
  }
  if (typeof stats.avg_truncated_tokens_est === "number") {
    parts.push(`≈${Math.round(stats.avg_truncated_tokens_est)} токенов`);
  }
  return parts.length ? `Индекс обновлён: ${parts.join(", ")}` : "Индекс обновлён.";
}

function formatLogTime(date) {
  return date.toLocaleTimeString("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function showToast({ message, type = "info", duration = 4000 } = {}) {
  if (!toastRoot || !message) {
    return;
  }

  const toast = document.createElement("div");
  toast.className = type && type !== "info" ? `toast ${type}` : "toast";
  const textNode = document.createElement("span");
  textNode.textContent = message;
  toast.append(textNode);

  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "toast-close";
  closeBtn.setAttribute("aria-label", "Закрыть уведомление");
  closeBtn.textContent = "×";
  closeBtn.addEventListener("click", () => removeToast(toast));
  toast.append(closeBtn);

  toastRoot.append(toast);
  while (toastRoot.children.length > MAX_TOASTS) {
    toastRoot.firstElementChild?.remove();
  }

  const timeoutId = window.setTimeout(() => removeToast(toast), duration);
  toast.dataset.timeoutId = String(timeoutId);
}

function removeToast(toast) {
  if (!toast) {
    return;
  }
  const timeoutId = toast.dataset.timeoutId;
  if (timeoutId) {
    window.clearTimeout(Number(timeoutId));
  }
  if (toast.isConnected) {
    toast.remove();
  }
}
