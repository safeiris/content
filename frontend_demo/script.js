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
const downloadMdBtn = document.getElementById("download-md");
const downloadReportBtn = document.getElementById("download-report");
const clearLogBtn = document.getElementById("clear-log");
const structurePreset = document.getElementById("structure-preset");
const structureInput = document.getElementById("structure-input");
const keywordsInput = document.getElementById("keywords-input");
const goalInput = document.getElementById("goal-input");
const kInput = document.getElementById("k-input");
const temperatureInput = document.getElementById("temperature-input");
const maxTokensInput = document.getElementById("max-tokens-input");
const modelInput = document.getElementById("model-input");
const includeFaq = document.getElementById("include-faq");
const includeTable = document.getElementById("include-table");
const includeJsonld = document.getElementById("include-jsonld");
const factsMode = document.getElementById("facts-mode");
const addDisclaimer = document.getElementById("add-disclaimer");
const healthStatus = document.getElementById("health-status");
const reindexLog = document.getElementById("reindex-log");
const previewSystem = document.getElementById("preview-system");
const previewUser = document.getElementById("preview-user");
const contextList = document.getElementById("context-list");
const contextSummary = document.getElementById("context-summary");
const contextBadge = document.getElementById("context-badge");
const generateBtn = briefForm.querySelector("button[type='submit']");
const advancedSettings = document.getElementById("advanced-settings");
const advancedSupportSection = document.querySelector("[data-section='support']");

const ADVANCED_SETTINGS_STORAGE_KEY = "content-demo:advanced-settings-open";

const LOG_STATUS_LABELS = {
  info: "INFO",
  success: "SUCCESS",
  warn: "WARN",
  error: "ERROR",
};

const DEFAULT_PROGRESS_MESSAGE = progressMessage?.textContent?.trim() || "Готовим данные…";
const MAX_TOASTS = 3;

const HEALTH_STATUS_MESSAGES = {
  openai_key: {
    label: "OpenAI",
    ok: "активен",
    fail: "не найден",
  },
  retrieval_index: {
    label: "Retrieval index",
    ok: "найден",
    fail: "не найден",
  },
  artifacts_dir: {
    label: "Каталог артефактов",
    ok: "доступен",
    fail: "недоступен",
  },
};

const state = {
  pipes: new Map(),
  artifacts: [],
  hasMissingArtifacts: false,
  currentResult: null,
};

const devActionsConfig = resolveDevActions();
if (!devActionsConfig.show && advancedSupportSection) {
  advancedSupportSection.remove();
  reindexBtn = null;
  healthBtn = null;
}

const interactiveElements = [
  previewBtn,
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
previewBtn.addEventListener("click", handlePromptPreview);
briefForm.addEventListener("submit", handleGenerate);
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
init();

function switchTab(tabId) {
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabId));
  panels.forEach((panel) => panel.classList.toggle("active", panel.id === tabId));
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
  const ok = await downloadArtifactFile(artifact.path, artifact.name || "artifact.txt", artifact);
  return ok;
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
}

async function handlePromptPreview() {
  try {
    const payload = buildRequestPayload();
    setInteractiveBusy(true);
    setButtonLoading(previewBtn, true);
    showProgress(true, "Собираем промпт…");
    const preview = await fetchJson("/api/prompt/preview", {
      method: "POST",
      body: JSON.stringify({ theme: payload.theme, data: payload.data, k: payload.k }),
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
    setInteractiveBusy(true);
    setButtonLoading(generateBtn, true);
    showProgress(true, "Генерируем материалы…");
    const response = await fetchJson("/api/generate", {
      method: "POST",
      body: JSON.stringify({
        theme: payload.theme,
        data: payload.data,
        k: payload.k,
        model: payload.model,
        temperature: payload.temperature,
        max_tokens: payload.maxTokens,
      }),
    });
    const markdown = response?.markdown ?? "";
    const meta = (response?.meta_json && typeof response.meta_json === "object") ? response.meta_json : {};
    const artifactPaths = response?.artifact_paths;
    state.currentResult = { markdown, meta, artifactPaths };
    draftView.innerHTML = markdownToHtml(markdown);
    resultTitle.textContent = payload.data.theme || "Результат генерации";
    const characters = meta.characters ?? markdown.length;
    resultMeta.textContent = `Символов: ${characters.toLocaleString("ru-RU")} · Модель: ${meta.model_used ?? "—"}`;
    renderMetadata(meta);
    updateResultBadges(meta);
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
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const structure = structureInput.value
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);

  const data = {
    theme: topic,
    goal: goalInput.value.trim() || "SEO-статья",
    tone: document.getElementById("tone-select").value,
    keywords,
    structure,
    include_faq: includeFaq.checked,
    include_table: includeTable.checked,
    include_jsonld: includeJsonld.checked,
    facts_mode: factsMode.checked ? "cautious" : undefined,
    add_disclaimer: addDisclaimer.checked,
    structure_preset: structurePreset.value,
    pipe_id: theme,
  };

  const kValue = String(kInput?.value ?? "").trim();
  let k = kValue === "" ? 3 : Number.parseInt(kValue, 10);
  if (!Number.isInteger(k) || k < 0 || k > 6) {
    throw new Error("Контекст (k) должен быть целым числом от 0 до 6");
  }
  if (kInput) {
    kInput.value = String(k);
  }

  const temperatureValue = String(temperatureInput?.value ?? "").trim();
  let temperature = temperatureValue === "" ? 0.3 : Number.parseFloat(temperatureValue);
  if (Number.isNaN(temperature) || temperature < 0 || temperature > 1) {
    throw new Error("Temperature должна быть числом от 0 до 1");
  }
  if (temperatureInput) {
    temperatureInput.value = String(temperature);
  }

  const maxTokensValue = String(maxTokensInput?.value ?? "").trim();
  let maxTokens = maxTokensValue === "" ? 1400 : Number.parseInt(maxTokensValue, 10);
  if (!Number.isInteger(maxTokens) || maxTokens <= 0) {
    throw new Error("Max tokens должно быть положительным целым числом");
  }
  if (maxTokensInput) {
    maxTokensInput.value = String(maxTokens);
  }

  const model = modelInput.value || undefined;

  return {
    theme,
    data,
    k,
    temperature,
    maxTokens,
    model,
  };
}

function renderMetadata(meta) {
  reportView.innerHTML = "";
  const report = document.createElement("pre");
  report.className = "metadata-view";
  if (!meta || typeof meta !== "object" || !Object.keys(meta).length) {
    report.textContent = "Метаданные недоступны.";
  } else {
    report.textContent = JSON.stringify(meta, null, 2);
  }
  reportView.append(report);
}

function updateResultBadges(meta) {
  resultBadges.innerHTML = "";
  if (!meta) {
    return;
  }

  const entries = [
    badgeInfo("plagiarism_detected", meta.plagiarism_detected, {
      true: { text: "Plagiarism detected", type: "error" },
      false: { text: "Plagiarism clean", type: "success" },
    }),
    badgeInfo("retry_used", meta.retry_used, {
      true: { text: "Retry used", type: "neutral" },
      false: { text: "Single pass", type: "success" },
    }),
    badgeInfo("postfix_appended", meta.postfix_appended, {
      true: { text: "CTA appended", type: "neutral" },
      false: { text: "CTA not needed", type: "success" },
    }),
    badgeInfo("disclaimer_appended", meta.disclaimer_appended, {
      true: { text: "Дисклеймер добавлен", type: "neutral" },
      false: { text: "Без дисклеймера", type: "neutral" },
    }),
    badgeInfo("length_adjustment", meta.length_adjustment, null, (value) =>
      value ? `Length fix: ${value}` : "Length OK"
    ),
    meta.model_used ? { text: `Model: ${meta.model_used}`, type: "neutral" } : null,
    typeof meta.temperature_used === "number"
      ? { text: `T=${meta.temperature_used}`, type: "neutral" }
      : null,
    typeof meta.context_budget_tokens_est === "number"
      ? { text: `Context tokens ≈ ${meta.context_budget_tokens_est}` }
      : null,
  ].filter(Boolean);

  entries.forEach((entry) => {
    const badge = document.createElement("span");
    badge.className = `badge ${entry.type ?? "neutral"}`;
    badge.textContent = entry.text;
    resultBadges.append(badge);
  });
}

function badgeInfo(key, value, mapping, fallbackFormatter) {
  if (mapping && Object.prototype.hasOwnProperty.call(mapping, value)) {
    return mapping[value];
  }
  if (fallbackFormatter) {
    return { text: fallbackFormatter(value), type: "neutral" };
  }
  return value
    ? { text: `${key}: ${value}`, type: "neutral" }
    : { text: `${key}: нет`, type: "neutral" };
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
  contextList.innerHTML = "";
  (preview.context || []).forEach((item, idx) => {
    const li = document.createElement("li");
    const title = item.path || `Фрагмент #${idx + 1}`;
    li.innerHTML = `<strong>${title}</strong><span>score: ${Number(item.score ?? 0).toFixed(2)}</span><br />${(item.text || "").slice(0, 320)}${
      item.text && item.text.length > 320 ? "…" : ""
    }`;
    contextList.append(li);
  });

  if (!preview.context || !preview.context.length) {
    contextList.innerHTML = '<li>Контекст не используется.</li>';
  }

  const contextUsed = Boolean(preview.context_used);
  const indexMissing = Boolean(preview.context_index_missing);
  const k = preview.k ?? 0;
  if (k === 0) {
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

  if (typeof preview.context_budget_tokens_est === "number") {
    const limit = preview.context_budget_tokens_limit;
    contextSummary.textContent = limit
      ? `Контекст: ~${preview.context_budget_tokens_est} токенов из лимита ${limit}`
      : `Контекст: ~${preview.context_budget_tokens_est} токенов`;
  } else {
    contextSummary.textContent = "";
  }
}

function enableDownloadButtons(paths) {
  const markdownPath = paths?.markdown;
  const metadataPath = paths?.metadata;

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
  await downloadArtifactFile(path, type === "markdown" ? "draft.md" : "report.json");
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
    return false;
  }
  try {
    const response = await fetch(`${API_BASE}/api/artifacts/download?path=${encodeURIComponent(path)}`);
    if (!response.ok) {
      const text = await response.text();
      const error = new Error(text || `HTTP ${response.status}`);
      error.status = response.status;
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
    return true;
  } catch (error) {
    console.error(error);
    if (isNotFoundError(error) && artifact) {
      handleMissingArtifact(artifact);
      return false;
    }
    showToast({ message: `Не удалось скачать файл: ${getErrorMessage(error)}`, type: "error" });
    return false;
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
