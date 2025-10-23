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
const kInput = document.getElementById("k-input");
const temperatureInput = document.getElementById("temperature-input");
const temperatureHint = document.querySelector('[data-role="temperature-hint"]');
const maxTokensInput = document.getElementById("max-tokens-input");
const modelInput = document.getElementById("model-input");
const includeFaq = document.getElementById("include-faq");
const includeJsonld = document.getElementById("include-jsonld");
const minCharsInput = document.getElementById("min-chars-input");
const maxCharsInput = document.getElementById("max-chars-input");
const keywordModeInputs = document.querySelectorAll("input[name='keywords-mode']");
const styleProfileSelect = document.getElementById("style-profile-select");
const styleProfileHint = document.getElementById("style-profile-hint");
const useStyleCheckbox = document.getElementById("useStyleCheckbox");
const styleSettings = document.getElementById("styleSettings");
const styleSelect = document.getElementById("styleSelect");
const styleStrengthInput = document.getElementById("styleStrength");
const sourcesList = document.getElementById("sources-list");
const addSourceBtn = document.getElementById("add-source-btn");
const faqCountInput = document.getElementById("faq-count-input");
const faqCountWrapper = document.getElementById("faq-count-wrapper");
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

const DEFAULT_PROGRESS_MESSAGE = progressMessage?.textContent?.trim() || "Готовим данные…";
const MAX_TOASTS = 3;
const MAX_CUSTOM_CONTEXT_CHARS = 20000;
const MAX_CUSTOM_CONTEXT_LABEL = MAX_CUSTOM_CONTEXT_CHARS.toLocaleString("ru-RU");

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
  hasMissingArtifacts: false,
  currentResult: null,
};

const customContextState = {
  textareaText: "",
  fileText: "",
  fileName: "",
  noticeShown: false,
};

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

const interactiveElements = [
  generateBtn,
  kInput,
  temperatureInput,
  maxTokensInput,
  modelInput,
];

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

structurePreset.addEventListener("change", () => applyStructurePreset(structurePreset.value));
pipeSelect.addEventListener("change", () => applyPipeDefaults(pipeSelect.value));
briefForm.addEventListener("submit", handleGenerate);
if (retryBtn) {
  retryBtn.addEventListener("click", handleRetryClick);
}
if (modelInput) {
  modelInput.addEventListener("change", () => updateTemperatureControlState(modelInput.value));
}
if (styleProfileSelect) {
  styleProfileSelect.addEventListener("change", handleStyleProfileChange);
}
if (useStyleCheckbox && styleSettings) {
  const toggleStyleSettingsVisibility = () => {
    styleSettings.style.display = useStyleCheckbox.checked ? "flex" : "none";
  };
  useStyleCheckbox.addEventListener("change", toggleStyleSettingsVisibility);
  toggleStyleSettingsVisibility();
}
if (includeFaq) {
  includeFaq.addEventListener("change", handleFaqToggle);
}
if (addSourceBtn) {
  addSourceBtn.addEventListener("click", handleAddSource);
}
if (sourcesList) {
  sourcesList.addEventListener("click", handleSourceListClick);
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
if (reindexBtn) {
  reindexBtn.addEventListener("click", handleReindex);
}
if (healthBtn) {
  healthBtn.addEventListener("click", handleHealthCheck);
}
if (cleanupArtifactsBtn) {
  cleanupArtifactsBtn.addEventListener("click", handleArtifactsCleanup);
}
downloadMdBtn.addEventListener("click", () => handleDownload("markdown"));
downloadReportBtn.addEventListener("click", () => handleDownload("metadata"));
if (clearLogBtn) {
  clearLogBtn.addEventListener("click", () => {
    clearReindexLog();
    showToast({ message: "Журнал очищен", type: "info" });
  });
}

setupAdvancedSettings();
updateTemperatureControlState(modelInput?.value);
handleStyleProfileChange();
handleFaqToggle();
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

function handleFaqToggle() {
  if (!faqCountWrapper || !faqCountInput || !includeFaq) {
    return;
  }
  const enabled = includeFaq.checked;
  faqCountWrapper.hidden = !enabled;
  faqCountInput.disabled = !enabled;
  if (enabled && !faqCountInput.value) {
    faqCountInput.value = "5";
  }
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
  if (kInput) {
    if (isCustom || isOff) {
      kInput.value = "0";
      kInput.disabled = true;
    } else {
      kInput.disabled = false;
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

function addSourceRow(value = "") {
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
    input.value = value;
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
    items.push(value);
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

async function loadArtifacts() {
  let artifacts;
  try {
    artifacts = await fetchJson("/api/artifacts");
  } catch (error) {
    artifactsList.innerHTML = '<div class="empty-state">Не удалось загрузить материалы.</div>';
    state.artifacts = [];
    state.hasMissingArtifacts = false;
    updateArtifactsToolbar();
    throw error;
  }
  const items = Array.isArray(artifacts)
    ? artifacts
    : artifacts && Array.isArray(artifacts.items)
      ? artifacts.items
      : null;
  if (!items) {
    artifactsList.innerHTML = '<div class="empty-state">Некорректный ответ сервера.</div>';
    state.artifacts = [];
    state.hasMissingArtifacts = false;
    updateArtifactsToolbar();
    throw new Error("Некорректный ответ сервера");
  }
  state.artifacts = normalizeArtifactList(items);
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
    const topic = metadata.input_data?.theme || metadata.data?.theme || metadata.topic || "Без темы";
    card.dataset.artifactId = artifact.id || artifact.path || "";
    card.querySelector(".card-title").textContent = title;
    const statusInfo = resolveArtifactStatus(artifact);
    const statusEl = card.querySelector(".status");
    statusEl.textContent = statusInfo.label;
    statusEl.dataset.status = statusInfo.value;
    const topicText = [themeName, topic].filter(Boolean).join(" · ") || topic;
    card.querySelector(".card-topic").textContent = topicText;
    const updatedAt = artifact.modified_at ? new Date(artifact.modified_at) : null;
    card.querySelector(".card-meta").textContent = updatedAt
      ? `Обновлено ${updatedAt.toLocaleString("ru-RU")}`
      : "Дата обновления неизвестна";

    card.addEventListener("click", () => showArtifact(artifact));

    const openBtn = card.querySelector(".open-btn");
    const downloadBtn = card.querySelector(".download-btn");
    const deleteBtn = card.querySelector(".delete-btn");
    if (openBtn) {
      openBtn.disabled = !artifact.path;
      openBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        showArtifact(artifact);
      });
    }
    if (downloadBtn) {
      downloadBtn.disabled = !artifact.path;
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
  return items.map((item) => {
    const metadata = item && typeof item.metadata === "object" && !Array.isArray(item.metadata) ? item.metadata : {};
    const fallbackId =
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `artifact-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const id = typeof item.id === "string" && item.id.trim()
      ? item.id.trim()
      : typeof item.path === "string" && item.path.trim()
        ? item.path.trim()
        : metadata.id || metadata.artifact_id || fallbackId;
    const path = typeof item.path === "string" && item.path.trim() ? item.path.trim() : null;
    const metadataPath = typeof item.metadata_path === "string" && item.metadata_path.trim()
      ? item.metadata_path.trim()
      : null;
    const name = typeof item.name === "string" && item.name.trim()
      ? item.name.trim()
      : typeof metadata.name === "string" && metadata.name.trim()
        ? metadata.name.trim()
        : id;
    return {
      id,
      name,
      path,
      metadata_path: metadataPath,
      metadata,
      size: typeof item.size === "number" ? item.size : null,
      modified_at: item.modified_at || null,
      status: item.status || metadata.status || null,
      missing: Boolean(item.missing),
    };
  });
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
  if (!artifact?.path) {
    showToast({ message: "Файл недоступен для скачивания", type: "warn" });
    return false;
  }
  const result = await downloadArtifactFile(artifact.path, artifact.name || "artifact.txt", artifact);
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

function isTemperatureLocked(modelValue) {
  return typeof modelValue === "string" && modelValue.toLowerCase().includes("gpt-5");
}

function updateTemperatureControlState(modelValue) {
  const locked = isTemperatureLocked(modelValue);
  if (temperatureInput) {
    temperatureInput.disabled = locked;
    if (locked) {
      temperatureInput.value = "0.3";
    }
  }
  if (temperatureHint) {
    temperatureHint.hidden = !locked;
  }
}

function applyStructurePreset(presetKey) {
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
  if (!structureInput.value && Array.isArray(pipe.default_structure)) {
    structureInput.value = pipe.default_structure.join("\n");
  }
  if (!goalInput.value) {
    goalInput.value = "SEO-статья";
  }
  if (modelInput && !modelInput.value && pipe.default_model) {
    modelInput.value = pipe.default_model;
    updateTemperatureControlState(pipe.default_model);
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
    toggleRetryButton(false);
    setInteractiveBusy(true);
    setButtonLoading(generateBtn, true);
    showProgress(true, "Генерируем материалы…");
    renderUsedKeywords(null);
    const requestModel = payload.model || null;
    const requestBody = {
      theme: payload.theme,
      data: payload.data,
      k: payload.k,
      model: requestModel,
      temperature: payload.temperature,
      max_tokens: payload.maxTokens,
      context_source: payload.context_source,
      keywords: Array.isArray(payload.data?.keywords) ? payload.data.keywords : [],
      length_range: { min: 3500, max: 6000, mode: "no_spaces" },
      faq_required: true,
      faq_count: 5,
    };
    if (payload.context_source === "custom") {
      requestBody.context_text = payload.context_text;
      if (payload.context_filename) {
        requestBody.context_filename = payload.context_filename;
      }
    }
    const response = await fetchJson("/api/generate", {
      method: "POST",
      body: JSON.stringify(requestBody),
    });
    const markdown = response?.markdown ?? "";
    const meta = (response?.meta_json && typeof response.meta_json === "object") ? response.meta_json : {};
    const responseStatus = typeof response?.status === "string" ? response.status.trim().toLowerCase() : "";
    const metaStatus = typeof meta?.status === "string" ? meta.status.trim().toLowerCase() : "";
    if (["failed", "error"].includes(responseStatus) || ["failed", "error"].includes(metaStatus)) {
      const backendError = typeof response?.error === "string" && response.error.trim()
        ? response.error.trim()
        : typeof meta?.error === "string" && meta.error.trim()
          ? meta.error.trim()
          : "";
      const backendMessage = typeof response?.backend_message === "string" && response.backend_message.trim()
        ? response.backend_message.trim()
        : typeof meta?.backend_message === "string" && meta.backend_message.trim()
          ? meta.backend_message.trim()
          : "";
      const messageParts = [backendMessage, backendError].filter(Boolean);
      const failureMessage = messageParts.length > 0
        ? messageParts.join(" ")
        : "Генерация завершилась с ошибкой.";
      throw new Error(failureMessage);
    }
    const artifactPaths = response?.artifact_paths;
    const metadataCharacters = typeof meta.characters === "number" ? meta.characters : undefined;
    const characters = typeof metadataCharacters === "number" ? metadataCharacters : markdown.trim().length;
    const hasContent = characters > 0;
    state.currentResult = { markdown, meta, artifactPaths, characters, hasContent };
    const fallbackModel = response?.fallback_used ?? meta.fallback_used;
    const fallbackReason = response?.fallback_reason ?? meta.fallback_reason;
    draftView.innerHTML = markdownToHtml(markdown);
    resultTitle.textContent = payload.data.theme || "Результат генерации";
    const metaParts = [];
    if (hasContent) {
      metaParts.push(`Символов: ${characters.toLocaleString("ru-RU")}`);
    }
    metaParts.push(`Модель: ${meta.model_used ?? "—"}`);
    resultMeta.textContent = metaParts.join(" · ");
    renderMetadata(meta);
    renderUsedKeywords(meta);
    updateResultBadges(meta);
    toggleRetryButton(!hasContent);
    updatePromptPreview({
      system: meta.system_prompt_preview,
      context: meta.clips || [],
      user: meta.user_prompt_preview,
      context_used: meta.context_used,
      context_index_missing: meta.context_index_missing,
      context_budget_tokens_est: meta.context_budget_tokens_est,
      context_budget_tokens_limit: meta.context_budget_tokens_limit,
      k: payload.k,
    });
    if (fallbackModel) {
      const reasonText = describeFallbackNotice(fallbackReason);
      showToast({
        message: `Использована резервная модель (${fallbackModel}). ${reasonText}`,
        type: "warn",
        duration: 6000,
      });
    }
    enableDownloadButtons(artifactPaths);
    try {
      await loadArtifacts();
    } catch (refreshError) {
      console.error(refreshError);
      showToast({ message: `Не удалось обновить список материалов: ${getErrorMessage(refreshError)}`, type: "warn" });
    }
    switchTab("result");
    showToast({ message: "Готово", type: "success" });
  } catch (error) {
    console.error(error);
    showToast({ message: `Не удалось выполнить генерацию: ${getErrorMessage(error)}`, type: "error" });
  } finally {
    setButtonLoading(generateBtn, false);
    setInteractiveBusy(false);
    showProgress(false);
  }
}

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

function buildRequestPayload() {
  const theme = pipeSelect.value;
  if (!theme) {
    throw new Error("Выберите тематику");
  }
  const topic = document.getElementById("topic-input").value.trim();
  if (!topic) {
    throw new Error("Укажите тему материала");
  }

  const keywords = keywordsInput.value
    .split(/\r?\n|,/)
    .map((item) => item.trim())
    .filter(Boolean);
  const structure = structureInput.value
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);

  const minCharsRaw = String(minCharsInput?.value ?? "").trim();
  const maxCharsRaw = String(maxCharsInput?.value ?? "").trim();
  const minChars = minCharsRaw === "" ? 3500 : Number.parseInt(minCharsRaw, 10);
  const maxChars = maxCharsRaw === "" ? 6000 : Number.parseInt(maxCharsRaw, 10);
  if (!Number.isInteger(minChars) || minChars <= 0) {
    throw new Error("Минимальный объём должен быть положительным целым числом");
  }
  if (!Number.isInteger(maxChars) || maxChars <= 0) {
    throw new Error("Максимальный объём должен быть положительным целым числом");
  }
  if (maxChars < minChars) {
    throw new Error("Максимальный объём должен быть больше или равен минимальному");
  }

  const keywordMode = Array.from(keywordModeInputs).find((input) => input.checked)?.value || "strict";
  const sources = collectSources();
  const styleProfile = styleProfileSelect?.value || "sravni.ru";
  const title = titleInput?.value.trim();
  const audience = audienceInput?.value.trim();
  const toneSelect = document.getElementById("tone-select");
  const tone = toneSelect ? toneSelect.value : "экспертный";

  const contextSource = String(contextSourceSelect?.value || "index.json").toLowerCase();

  const data = {
    theme: topic,
    goal: goalInput.value.trim() || "SEO-статья",
    tone,
    keywords,
    keywords_mode: keywordMode,
    structure,
    include_faq: includeFaq.checked,
    include_jsonld: includeJsonld.checked,
    structure_preset: structurePreset.value,
    pipe_id: theme,
    length_limits: { min_chars: minChars, max_chars: maxChars },
    style_profile: styleProfile,
    context_source: contextSource,
  };

  if (title) {
    data.title = title;
  }
  if (audience) {
    data.target_audience = audience;
  }
  if (sources.length) {
    data.sources = sources;
  }
  if (includeFaq.checked) {
    const faqValue = String(faqCountInput?.value ?? "").trim();
    let faqCount = faqValue === "" ? 5 : Number.parseInt(faqValue, 10);
    if (!Number.isInteger(faqCount) || faqCount <= 0) {
      faqCount = 5;
    }
    data.faq_questions = faqCount;
  }

  const kValue = String(kInput?.value ?? "").trim();
  let k = kValue === "" ? 0 : Number.parseInt(kValue, 10);
  if (!Number.isInteger(k) || k < 0 || k > 6) {
    throw new Error("Контекст (k) должен быть целым числом от 0 до 6");
  }
  if (contextSource === "off" || contextSource === "custom") {
    k = 0;
  }
  if (kInput) {
    kInput.value = String(k);
  }

  const contextPayload = resolveCustomContextPayload(contextSource);
  if (contextSource === "custom") {
    data.context_source = "custom";
    if (contextPayload.filename) {
      data.context_filename = contextPayload.filename;
    } else {
      delete data.context_filename;
    }
  } else {
    delete data.context_filename;
  }

  let model = (modelInput?.value || "").trim();
  if (!model) {
    const pipeDefaults = state.pipes.get(theme);
    if (pipeDefaults?.default_model) {
      model = pipeDefaults.default_model;
      if (modelInput) {
        modelInput.value = model;
      }
    }
  }
  const temperatureLocked = isTemperatureLocked(model);
  let temperature;
  if (temperatureLocked) {
    temperature = undefined;
    if (temperatureInput) {
      temperatureInput.value = "0.3";
    }
  } else {
    const temperatureValue = String(temperatureInput?.value ?? "").trim();
    temperature = temperatureValue === "" ? 0.3 : Number.parseFloat(temperatureValue);
    if (Number.isNaN(temperature) || temperature < 0 || temperature > 1) {
      throw new Error("Temperature должна быть числом от 0 до 1");
    }
    if (temperatureInput) {
      temperatureInput.value = String(temperature);
    }
  }

  const maxTokensValue = String(maxTokensInput?.value ?? "").trim();
  let maxTokens = maxTokensValue === "" ? 1400 : Number.parseInt(maxTokensValue, 10);
  if (!Number.isInteger(maxTokens) || maxTokens <= 0) {
    throw new Error("Max tokens должно быть положительным целым числом");
  }
  if (maxTokensInput) {
    maxTokensInput.value = String(maxTokens);
  }

  const payload = {
    theme,
    data,
    k,
    temperature: temperatureLocked ? undefined : temperature,
    maxTokens,
    model: model || undefined,
    context_source: contextSource,
  };

  if (contextSource === "custom") {
    payload.context_text = contextPayload.text;
    if (contextPayload.filename) {
      payload.context_filename = contextPayload.filename;
    }
  }

  let stylePayload;
  if (useStyleCheckbox) {
    if (useStyleCheckbox.checked) {
      const selectedStyle = (styleSelect?.value || "sravni").trim() || "sravni";
      const strengthRaw = String(styleStrengthInput?.value ?? "").trim();
      let strength = Number.parseFloat(strengthRaw || "0.6");
      if (Number.isNaN(strength)) {
        strength = 0.6;
      }
      stylePayload = {
        enabled: true,
        style: selectedStyle,
        strength,
      };
    } else {
      stylePayload = { enabled: false };
    }
    data.style = stylePayload;
  }

  if (stylePayload) {
    payload.style = stylePayload;
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
    const label = within
      ? `Объём: ${chars.toLocaleString("ru-RU")} зн. (в норме)`
      : `Объём: ${chars.toLocaleString("ru-RU")} зн. (нужно ${min}–${max})`;
    list.append(createQualityItem(within ? "success" : "warning", label));
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

function updateResultBadges(meta) {
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

function enableDownloadButtons(paths) {
  const normalizePath = (value) =>
    typeof value === "string" && value.trim().length > 0 ? value.trim() : null;

  const markdownPath = normalizePath(paths?.markdown);
  const metadataPath = normalizePath(paths?.metadata);

  if (markdownPath) {
    downloadMdBtn.dataset.path = markdownPath;
    downloadMdBtn.disabled = false;
  } else {
    downloadMdBtn.disabled = true;
    delete downloadMdBtn.dataset.path;
  }

  if (metadataPath) {
    downloadReportBtn.dataset.path = metadataPath;
    downloadReportBtn.disabled = false;
  } else {
    downloadReportBtn.disabled = true;
    delete downloadReportBtn.dataset.path;
  }
}

async function handleDownload(type) {
  const button = type === "markdown" ? downloadMdBtn : downloadReportBtn;
  const path = button.dataset.path;
  if (!path) {
    showToast({ message: "Файл недоступен для скачивания", type: "warn" });
    return;
  }
  const result = await downloadArtifactFile(
    path,
    type === "markdown" ? "draft.md" : "report.json"
  );
  if (result === "not_found") {
    button.disabled = true;
    delete button.dataset.path;
  }
}

async function showArtifact(artifact) {
  try {
    showProgress(true, "Загружаем артефакт…");
    const markdownPath = artifact.path;
    if (!markdownPath) {
      handleMissingArtifact(artifact);
      return;
    }
    const markdown = await fetchText(`/api/artifacts/download?path=${encodeURIComponent(markdownPath)}`);
    const metadataPath = artifact.metadata_path;
    let metadata = artifact.metadata || {};
    if (!Object.keys(metadata).length && metadataPath) {
      try {
        const jsonText = await fetchText(`/api/artifacts/download?path=${encodeURIComponent(metadataPath)}`);
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
    updateResultBadges(metadata);
    enableDownloadButtons({ markdown: artifact.path, metadata: metadataPath });
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
    const text = await response.text();

    if (!response.ok) {
      let message = text || `HTTP ${response.status}`;
      try {
        const data = JSON.parse(text);
        message = data.error || message;
      } catch (parseError) {
        // keep original message
      }
      throw new Error(message);
    }

    let data;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (parseError) {
      renderHealthError("Ошибка при запросе Health", "error");
      return;
    }

    renderHealthStatus(data);
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

  Object.entries(checks).forEach(([key, value], index) => {
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
  const failingEntry = Object.entries(checks).find(([, value]) => !normalizeHealthCheck(value).ok);
  const fallbackDictionary = failingEntry ? HEALTH_STATUS_MESSAGES[failingEntry[0]] || {} : {};
  const reason = failingEntry
    ? normalizeHealthCheck(failingEntry[1]).message || fallbackDictionary.fail || "Модель недоступна"
    : "";
  setGenerateAvailability(Boolean(status?.ok), reason);
}

function normalizeHealthCheck(value) {
  if (value && typeof value === "object") {
    return {
      ok: Boolean(value.ok),
      message: value.message || value.status || "",
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
    }
  }
}

async function fetchJson(path, options = {}) {
  const headers = options.headers ? { ...options.headers } : {};
  if (options.method && options.method !== "GET") {
    headers["Content-Type"] = "application/json";
  }

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...options, headers });
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
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`);
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

function showProgress(visible, message = DEFAULT_PROGRESS_MESSAGE) {
  if (!progressOverlay) {
    return;
  }
  if (visible) {
    if (progressMessage) {
      progressMessage.textContent = message;
    }
    progressOverlay.classList.remove("hidden");
  } else {
    progressOverlay.classList.add("hidden");
    if (progressMessage) {
      progressMessage.textContent = DEFAULT_PROGRESS_MESSAGE;
    }
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

async function downloadArtifactFile(path, fallbackName = "artifact.txt", artifact = null) {
  if (!path) {
    showToast({ message: "Файл недоступен", type: "warn" });
    return "error";
  }
  try {
    const response = await fetch(`${API_BASE}/api/artifacts/download?path=${encodeURIComponent(path)}`);
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
    anchor.download = path.split("/").pop() || fallbackName;
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
