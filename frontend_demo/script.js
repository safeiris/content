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
const briefForm = document.getElementById("brief-form");
const previewBtn = document.getElementById("preview-btn");
const reindexBtn = document.getElementById("reindex-btn");
const healthBtn = document.getElementById("health-btn");
const progressOverlay = document.getElementById("progress-overlay");
const draftView = document.getElementById("draft-view");
const reportView = document.getElementById("report-view");
const resultTitle = document.getElementById("result-title");
const resultMeta = document.getElementById("result-meta");
const resultBadges = document.getElementById("result-badges");
const downloadMdBtn = document.getElementById("download-md");
const downloadReportBtn = document.getElementById("download-report");
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

const state = {
  pipes: new Map(),
  artifacts: [],
  currentResult: null,
};

tabs.forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

structurePreset.addEventListener("change", () => applyStructurePreset(structurePreset.value));
pipeSelect.addEventListener("change", () => applyPipeDefaults(pipeSelect.value));
previewBtn.addEventListener("click", handlePromptPreview);
briefForm.addEventListener("submit", handleGenerate);
reindexBtn.addEventListener("click", handleReindex);
healthBtn.addEventListener("click", handleHealthCheck);
downloadMdBtn.addEventListener("click", () => handleDownload("markdown"));
downloadReportBtn.addEventListener("click", () => handleDownload("metadata"));

init();

function switchTab(tabId) {
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === tabId));
  panels.forEach((panel) => panel.classList.toggle("active", panel.id === tabId));
}

async function init() {
  try {
    await Promise.all([loadPipes(), loadArtifacts()]);
    applyStructurePreset(structurePreset.value);
  } catch (error) {
    console.error(error);
    alert(`Не удалось загрузить данные: ${error.message}`);
  }
}

async function loadPipes() {
  const pipes = await fetchJson("/api/pipes");
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
    card.innerHTML = `
      <div>
        <h3>${pipe.name}</h3>
        <p>${pipe.description}</p>
      </div>
      <div class="pipe-meta">Тон: <strong>${pipe.tone}</strong></div>
      <div class="pipe-keywords">
        ${(pipe.keywords || []).map((kw) => `<span>${kw}</span>`).join("")}
      </div>
    `;
    pipesList.append(card);
  });
}

async function loadArtifacts() {
  const artifacts = await fetchJson("/api/artifacts");
  state.artifacts = artifacts;
  renderArtifacts();
}

function renderArtifacts() {
  if (!state.artifacts.length) {
    artifactsList.innerHTML = '<div class="empty-state">Пока нет сгенерированных материалов.</div>';
    return;
  }

  artifactsList.innerHTML = "";
  const template = document.getElementById("artifact-card-template");

  state.artifacts.forEach((artifact) => {
    const card = template.content.firstElementChild.cloneNode(true);
    const metadata = artifact.metadata || {};
    const themeName = metadata.theme || extractThemeFromName(artifact.name);
    const topic = metadata.input_data?.theme || metadata.data?.theme || "Без темы";
    card.querySelector(".card-title").textContent = artifact.name;
    card.querySelector(".status").textContent = "Готов";
    card.querySelector(".status").dataset.status = "Ready";
    card.querySelector(".card-topic").textContent = `${themeName ?? ""} · ${topic}`;
    const updatedAt = new Date(artifact.modified_at);
    card.querySelector(".card-meta").textContent = `Обновлено ${updatedAt.toLocaleString("ru-RU")}`;
    card.addEventListener("click", () => showArtifact(artifact));
    artifactsList.append(card);
  });
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
    showProgress(true);
    const preview = await fetchJson("/api/prompt/preview", {
      method: "POST",
      body: JSON.stringify({ theme: payload.theme, data: payload.data, k: payload.k }),
    });
    updatePromptPreview(preview);
    switchTab("result");
  } catch (error) {
    console.error(error);
    alert(`Не удалось собрать промпт: ${error.message}`);
  } finally {
    showProgress(false);
  }
}

async function handleGenerate(event) {
  event.preventDefault();
  try {
    const payload = buildRequestPayload();
    showProgress(true);
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
    const { markdown, meta_json: meta, artifact_paths: artifactPaths } = response;
    state.currentResult = { markdown, meta, artifactPaths };
    draftView.innerHTML = markdownToHtml(markdown);
    resultTitle.textContent = payload.data.theme || "Результат генерации";
    const characters = meta.characters ?? markdown.length;
    resultMeta.textContent = `Символов: ${characters.toLocaleString("ru-RU")} · Модель: ${meta.model_used}`;
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
    await loadArtifacts();
    switchTab("result");
  } catch (error) {
    console.error(error);
    alert(`Не удалось выполнить генерацию: ${error.message}`);
  } finally {
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

  const k = Number.parseInt(kInput.value, 10);
  const temperature = Number.parseFloat(temperatureInput.value);
  const maxTokens = Number.parseInt(maxTokensInput.value, 10);
  const model = modelInput.value.trim() || undefined;

  return {
    theme,
    data,
    k: Number.isNaN(k) ? 3 : Math.max(0, k),
    temperature: Number.isNaN(temperature) ? 0.3 : temperature,
    maxTokens: Number.isNaN(maxTokens) ? 1400 : maxTokens,
    model,
  };
}

function renderMetadata(meta) {
  const report = document.createElement("pre");
  report.className = "status-log";
  report.textContent = JSON.stringify(meta, null, 2);
  reportView.innerHTML = "";
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
  if (!paths) {
    downloadMdBtn.disabled = true;
    downloadReportBtn.disabled = true;
    return;
  }
  downloadMdBtn.dataset.path = paths.markdown;
  downloadReportBtn.dataset.path = paths.metadata;
  downloadMdBtn.disabled = false;
  downloadReportBtn.disabled = false;
}

async function handleDownload(type) {
  const button = type === "markdown" ? downloadMdBtn : downloadReportBtn;
  const path = button.dataset.path;
  if (!path) {
    return;
  }
  try {
    const response = await fetch(`${API_BASE}/api/artifacts/download?path=${encodeURIComponent(path)}`);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = path.split("/").pop() || `artifact.${type === "markdown" ? "md" : "json"}`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  } catch (error) {
    console.error(error);
    alert(`Не удалось скачать файл: ${error.message}`);
  }
}

async function showArtifact(artifact) {
  try {
    showProgress(true);
    const markdown = await fetchText(`/api/artifacts/download?path=${encodeURIComponent(artifact.path)}`);
    const metadataPath = artifact.metadata_path;
    let metadata = artifact.metadata || {};
    if (!Object.keys(metadata).length && metadataPath) {
      const jsonText = await fetchText(`/api/artifacts/download?path=${encodeURIComponent(metadataPath)}`);
      metadata = JSON.parse(jsonText);
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
    alert(`Не удалось открыть артефакт: ${error.message}`);
  } finally {
    showProgress(false);
  }
}

async function handleReindex() {
  try {
    const theme = pipeSelect.value;
    if (!theme) {
      throw new Error("Выберите тематику");
    }
    showProgress(true);
    const stats = await fetchJson("/api/reindex", {
      method: "POST",
      body: JSON.stringify({ theme }),
    });
    reindexLog.textContent = `Индекс обновлён: ${stats.clips} клипов, средняя длина ${stats.avg_truncated_words} слов (~${stats.avg_truncated_tokens_est} токенов).`;
    await handleHealthCheck();
  } catch (error) {
    console.error(error);
    reindexLog.textContent = `Ошибка: ${error.message}`;
  } finally {
    showProgress(false);
  }
}

async function handleHealthCheck() {
  try {
    const theme = pipeSelect.value;
    const query = theme ? `?theme=${encodeURIComponent(theme)}` : "";
    const status = await fetchJson(`/api/health${query}`);
    renderHealthStatus(status);
  } catch (error) {
    console.error(error);
    healthStatus.innerHTML = `<span class="badge error">${error.message}</span>`;
  }
}

function renderHealthStatus(status) {
  healthStatus.innerHTML = "";
  if (!status || !status.checks) {
    healthStatus.innerHTML = '<span class="badge error">Нет данных</span>';
    return;
  }
  Object.entries(status.checks).forEach(([key, value]) => {
    const info = value;
    const badge = document.createElement("span");
    badge.className = `badge ${info.ok ? "success" : "error"}`;
    badge.textContent = `${key}: ${info.message}`;
    healthStatus.append(badge);
  });
}

async function fetchJson(path, options = {}) {
  const headers = options.headers ? { ...options.headers } : {};
  if (options.method && options.method !== "GET") {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!response.ok) {
    const text = await response.text();
    try {
      const data = JSON.parse(text);
      throw new Error(data.error || text || `HTTP ${response.status}`);
    } catch (error) {
      if (error instanceof SyntaxError) {
        throw new Error(text || `HTTP ${response.status}`);
      }
      throw error;
    }
  }
  return response.json();
}

async function fetchText(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.text();
}

function showProgress(visible) {
  progressOverlay.classList.toggle("hidden", !visible);
}
